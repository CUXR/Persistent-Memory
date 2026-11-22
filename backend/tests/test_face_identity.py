"""Persistent unknown face IDs and their handoff to automatic voice learning."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import func, select

from audio_pipeline.diarization import DiarizedTurn
from audio_pipeline.segment import Recording, SAMPLE_RATE, VOICE_MODEL, VoiceProfile
from app.crud.memory_store import MemoryStore
from app.crud.voice import VoiceStore
from app.models import Person, User
from app.schema.asr import RawTranscription
from app.schema.ingestion import EpisodeSummaryLLMResponse, FactExtractionLLMResponse
from app.services.audio_memory import AudioMemoryService
from face_recog_local import FaceRecognizer


def vector(index):
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1
    return value


def face(index=1, score=1):
    return SimpleNamespace(embedding=vector(index), bbox=np.array([0, 0, 100, 100]), det_score=score)


class FixtureIndex:
    """Exercise the recognizer's FAISS boundary with deterministic squared L2."""
    def __init__(self, dimension):
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.concatenate([self.vectors, vectors])

    def search(self, vectors, count):
        distances = ((vectors[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        indices = distances.argsort(axis=1)[:, :count]
        return np.take_along_axis(distances, indices, axis=1), indices


@pytest.fixture
def store(tmp_path):
    store = MemoryStore(f"sqlite+pysqlite:///{tmp_path / 'faces.sqlite'}", owner_user_id=uuid4())
    store.initialize()
    with store.Session.begin() as session:
        session.add(User(id=store.owner_user_id, first_name="Alex", username="alex"))
    try:
        yield store
    finally:
        store.close()


def recognizer(store, session, *faces, **kwargs):
    detector = Mock()
    detector.get.return_value = list(faces)
    return FaceRecognizer(session, store.owner_user_id, detector=detector, index_factory=FixtureIndex, **kwargs)


def test_unknown_face_is_persisted_and_reused_across_frames_and_restart(store):
    with store.Session() as session:
        tracker = recognizer(store, session, face())
        first = tracker.recognize_faces(None)[0]
        second = tracker.recognize_faces(None)[0]
        assert first['created'] and not second['created']
        assert first['person_id'] == second['person_id']
        assert first['score'] is None
        person = session.get(Person, UUID(first['person_id']))
        assert person.user_id == store.owner_user_id
        assert person.voice_embedding is None
        assert person.display_name.startswith('Unknown person ')
    with store.Session() as session:
        restarted = recognizer(store, session, face())
        assert str(restarted.get_current_person_id(None)) == first['person_id']
        assert session.scalar(select(func.count()).select_from(Person)) == 1


def test_different_unknown_faces_get_distinct_ids_and_no_arbitrary_audio_target(store):
    with store.Session() as session:
        tracker = recognizer(store, session, face(1), face(2))
        results = tracker.recognize_faces(None)
        assert len({result['person_id'] for result in results}) == 2
        assert tracker.get_current_person_id(None) is None
        assert session.scalar(select(func.count()).select_from(Person)) == 2
        tracker.app.get.return_value = []
        assert tracker.get_current_person_id(None) is None


def angled_face(squared_distance):
    """Unit vector whose squared L2 distance from vector(0) is squared_distance."""
    cosine = 1 - squared_distance / 2
    value = np.zeros(512, dtype=np.float32)
    value[0], value[1] = cosine, np.sqrt(1 - cosine ** 2)
    return SimpleNamespace(embedding=value, bbox=np.array([0, 0, 100, 100]), det_score=1)


@pytest.mark.parametrize('threshold, distance, matches', [
    (0.6, 0.55, True),
    (0.6, 0.65, False),
    (1.2, 1.1, True),
    (1.2, 1.3, False),
])
def test_match_boundary_follows_configured_threshold(store, threshold, distance, matches):
    with store.Session() as session:
        recognizer(store, session, face(0), l2_threshold=threshold).recognize_faces(None)
        probe = recognizer(store, session, angled_face(distance), l2_threshold=threshold)
        result = probe.recognize_faces(None)[0]
        assert result['created'] is not matches
        assert session.scalar(select(func.count()).select_from(Person)) == (1 if matches else 2)


def test_threshold_defaults_to_settings(store, monkeypatch):
    monkeypatch.setenv('FACE_MATCH_L2_THRESHOLD', '0.4')
    from app.core.config import get_settings
    get_settings.cache_clear()
    try:
        with store.Session() as session:
            assert recognizer(store, session).l2_threshold == 0.4
    finally:
        get_settings.cache_clear()


def test_weak_detection_is_not_registered(store):
    with store.Session() as session:
        tracker = recognizer(store, session, face(score=0.1))
        assert tracker.recognize_faces(None) == []
        assert session.scalar(select(func.count()).select_from(Person)) == 0


@pytest.mark.parametrize('bad_vector', [np.zeros(512), np.full(512, np.nan), np.ones(256)])
def test_invalid_face_embedding_cannot_create_a_person(store, bad_vector):
    detection = face()
    detection.embedding = bad_vector
    with store.Session() as session:
        with pytest.raises(ValueError):
            recognizer(store, session, detection).recognize_faces(None)
        assert session.scalar(select(func.count()).select_from(Person)) == 0


def test_same_face_under_different_owners_does_not_share_identity(store):
    with store.Session() as session:
        first = recognizer(store, session, face()).get_current_person_id(None)
        other_owner = User(first_name='Other', username='other')
        session.add(other_owner)
        session.commit()
        detector = Mock()
        detector.get.return_value = [face()]
        second = FaceRecognizer(session, other_owner.id, detector=detector,
                                index_factory=FixtureIndex).get_current_person_id(None)
        assert first != second
        assert session.get(Person, second).user_id == other_owner.id


def test_unknown_face_id_receives_voice_from_normal_conversation(store, tmp_path):
    with store.Session() as session:
        person_id = recognizer(store, session, face()).get_current_person_id(None)
    voices = VoiceStore(store)
    voices.save(VoiceProfile(model=VOICE_MODEL, embedding=vector(0).tolist()))
    assert voices.load(person_id).interlocutor is None
    source = tmp_path / 'conversation.wav'
    sf.write(source, np.full(4 * SAMPLE_RATE, 0.1, np.float32), SAMPLE_RATE)
    recording = Recording(audio_path=source, sample_count=4 * SAMPLE_RATE,
                          started_at=datetime.now(timezone.utc))
    diarizer = Mock(embedding_model=VOICE_MODEL)
    diarizer.diarize.return_value = [DiarizedTurn(0, 2, 'S0'), DiarizedTurn(2, 4, 'S1')]
    diarizer.extract_per_speaker_embeddings.return_value = {'S0': vector(0), 'S1': vector(1)}
    asr = Mock()
    asr.transcribe.side_effect = lambda segment: RawTranscription(
        segment=segment, text='Hello', avg_logprob=-0.1, no_speech_prob=0)
    llm = SimpleNamespace(
        generate_summary=AsyncMock(return_value=EpisodeSummaryLLMResponse(summary='Greeting', importance_score=0.1)),
        extract_facts=AsyncMock(return_value=FactExtractionLLMResponse()),
    )
    result = asyncio.run(AudioMemoryService(store, diarizer, asr, llm_client=llm).process(recording, person_id))
    assert result.status == 'complete'
    assert result.person_id == person_id
    assert result.dialog.turns[1].person_id == person_id
    assert result.dialog.turns[1].attribution_method == 'conversation'
    assert voices.load(person_id).interlocutor == VoiceProfile(model=VOICE_MODEL, embedding=vector(1).tolist())
    with store.Session() as session:
        assert session.scalar(select(func.count()).select_from(Person)) == 1
