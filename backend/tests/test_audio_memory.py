"""End-to-end audio memory with real storage and deterministic model substitutes."""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import event, func, select

from audio_pipeline.diarization import DiarizedTurn
from audio_pipeline.ingestion import AudioIngestionPipeline
from audio_pipeline.segment import SAMPLE_RATE, VOICE_MODEL, VoiceProfile, Recording, SpeechSegment, DiscoveredVoice
from app.crud.memory_store import MemoryStore
from app.crud.voice import VoiceStore
from app.crud.audio_jobs import AudioJobStore
from app.crud.conversations import commit_conversation
from app.models import AudioJob, Episode, Summary, User, Person, PersonFact
from app.schema.asr import RawTranscription
from app.schema.ingestion import EpisodeSummaryLLMResponse, FactExtractionLLMResponse, ExtractedFact
from app.services.asr import transcribe_segments
from app.services.asr_engine import WhisperEngine
from app.services.audio_memory import AudioMemoryService
from app.services.llm_client import LLMClient
from app.services.voice_enrollment import VoiceEnrollmentService

START = datetime(2026, 9, 26, tzinfo=timezone.utc)


def vector(index):
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1
    return result


def profile(index):
    return VoiceProfile(model=VOICE_MODEL, embedding=vector(index).tolist())


@pytest.fixture
def store(tmp_path):
    result = MemoryStore("sqlite+pysqlite:///" + str(tmp_path / "memory.db"), owner_user_id=uuid4())
    result.initialize()
    # The engine has already opened a connection during create_all.
    with result._engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    with result.Session.begin() as session:
        session.add(User(id=result.owner_user_id, first_name="Alex", username="alex"))
    try:
        yield result
    finally:
        result.close()


@pytest.fixture
def person(store):
    return store.upsert_person("Jordan Lee")


@pytest.fixture
def recording(tmp_path):
    vad = Mock()
    vad.contains_speech.return_value = True
    capture = AudioIngestionPipeline(vad, tmp_path / "audio", START)
    # Different waveforms let the ASR test assert that segment slices are correct.
    samples = np.concatenate([np.full(2 * SAMPLE_RATE, 0.1, np.float32),
                              np.full(2 * SAMPLE_RATE, 0.2, np.float32)])
    assert capture.push_audio(samples) == []
    return capture.flush()[0]


class FixtureASR:
    def __init__(self):
        self.calls = []
        self.fail = False

    def transcribe(self, segment):
        self.calls.append(segment)
        if self.fail:
            raise RuntimeError("fixture transcription failure")
        audio, rate = sf.read(segment.audio_path, dtype="float32")
        clip = WhisperEngine._slice(audio, segment.start_time, segment.end_time)
        expected = 0.1 if segment.speaker_id == "S0" else 0.2
        assert float(clip.mean()) == pytest.approx(expected, abs=1e-6)
        return RawTranscription(segment=segment, text="Hello" if segment.speaker_id == "S0" else "I enjoy climbing",
                                avg_logprob=-0.1, no_speech_prob=0)


@pytest.fixture
def setup_service(store, person):
    voices = VoiceStore(store)
    voices.save(profile(0))
    voices.save(profile(1), person_id=person.id)
    diarizer = Mock()
    diarizer.embedding_model = VOICE_MODEL
    diarizer.diarize.return_value = [DiarizedTurn(0, 2, "S0"), DiarizedTurn(2, 4, "S1")]
    diarizer.extract_per_speaker_embeddings.return_value = {"S0": vector(0), "S1": vector(1)}
    llm = Mock(spec=LLMClient)
    llm.generate_summary.return_value = EpisodeSummaryLLMResponse(summary="Jordan enjoys climbing.", importance_score=0.8)
    llm.extract_facts.return_value = FactExtractionLLMResponse(facts=[
        ExtractedFact(fact_text="Enjoys climbing", category="hobby", confidence=0.9),
    ])
    asr = FixtureASR()
    service = AudioMemoryService(store, diarizer, asr, llm_client=llm)
    return SimpleNamespace(service=service, voices=voices, diarizer=diarizer, llm=llm, asr=asr)


def run(setup_service, recording, person, **kwargs):
    return asyncio.run(setup_service.service.process(recording, person.id, **kwargs))


def test_capture_to_memory_preserves_audio_speakers_source_and_is_idempotent(store, person, recording, setup_service):
    result = run(setup_service, recording, person)
    assert result.status == "complete"
    assert result.transcript == "Alex: Hello\nJordan Lee: I enjoy climbing"
    assert [turn.speaker_id for turn in result.dialog.turns] == ["S0", "S1"]
    assert result.dialog.turns[1].person_id == person.id
    assert result.dialog.turns[1].speaker_similarity == pytest.approx(1)
    assert result.dialog.turns[1].start_time == 2
    assert result.dialog.turns[1].asr_confidence < 1
    with store.Session() as session:
        episode = session.get(Episode, result.result.episode_id)
        assert episode.transcript == result.transcript
        assert float(episode.importance_score) == pytest.approx(0.8)
        assert session.scalar(select(func.count()).select_from(PersonFact)) == 1
    again = run(setup_service, recording, person)
    assert again.result == result.result
    setup_service.llm.extract_facts.assert_awaited_once()
    assert recording.audio_path.exists()


def test_failure_retains_wav_and_retry_after_recreating_service_does_not_duplicate(store, person, recording, setup_service):
    setup_service.llm.generate_summary.side_effect = RuntimeError("provider unavailable")
    failed = run(setup_service, recording, person)
    assert failed.status == "failed" and failed.stage == "memory"
    assert failed.dialog is not None
    assert recording.audio_path.exists()
    with store.Session() as session:
        assert session.scalar(select(func.count()).select_from(Episode)) == 0
    setup_service.llm.generate_summary.side_effect = None
    fresh = AudioMemoryService(store, setup_service.diarizer, setup_service.asr, llm_client=setup_service.llm)
    done = asyncio.run(fresh.process(recording, person.id, retry=True))
    assert done.status == "complete"
    assert asyncio.run(fresh.process(recording, person.id)).result == done.result
    with store.Session() as session:
        assert session.scalar(select(func.count()).select_from(Episode)) == 1


@pytest.mark.parametrize("failure_stage", ["diarization", "attribution"])
def test_model_failure_is_visible_and_retryable(person, recording, setup_service, failure_stage):
    method = setup_service.diarizer.diarize if failure_stage == "diarization" else setup_service.diarizer.extract_per_speaker_embeddings
    method.side_effect = RuntimeError("model failed")
    result = run(setup_service, recording, person)
    assert result.status == "failed" and result.stage == failure_stage
    assert result.error == "model failed"
    assert recording.audio_path.exists()
    setup_service.llm.extract_facts.assert_not_called()


def test_asr_failure_preserves_segment_errors_and_prevents_extraction(person, recording, setup_service):
    setup_service.asr.fail = True
    result = run(setup_service, recording, person)
    assert result.status == "failed" and result.stage == "asr"
    assert len(result.dialog.errors) == 2
    setup_service.llm.extract_facts.assert_not_called()


def test_unknown_partner_keeps_dialog_for_review_without_extracting(person, recording, setup_service):
    setup_service.diarizer.extract_per_speaker_embeddings.return_value["S1"] = vector(2)
    result = run(setup_service, recording, person)
    assert result.status == "needs_review"
    assert "Unknown (S1): I enjoy climbing" in result.transcript
    setup_service.llm.extract_facts.assert_not_called()
    setup_service.voices.save(profile(2), person_id=person.id)
    assert run(setup_service, recording, person, retry=True).status == "complete"


def clear_partner_voice(store, person):
    with store.Session.begin() as session:
        row = session.get(Person, person.id)
        row.voice_embedding = None
        row.voice_embedding_model = None


def test_new_partner_voice_is_learned_from_two_seconds_of_normal_speech(store, person, recording, setup_service):
    clear_partner_voice(store, person)
    result = run(setup_service, recording, person)
    assert result.status == "complete"
    assert result.person_id == person.id
    assert setup_service.voices.load(person.id).interlocutor == profile(1)
    assert setup_service.voices.load(person.id).user == profile(0)
    assert result.voice_discovery.speaker_id == "S1"
    turn = result.dialog.turns[1]
    assert turn.person_id == person.id and turn.attribution_method == "conversation"
    assert turn.speaker_similarity is None  # A vector matching itself is not evidence.
    with store.Session() as session:
        assert session.scalar(select(func.count()).select_from(Person)) == 1
    # The next conversation uses the stored voice, without enrolling again.
    subsequent = recording.model_copy(update={"id": uuid4()})
    result = run(setup_service, subsequent, person)
    assert result.status == "complete" and result.voice_discovery is None
    assert result.dialog.turns[1].attribution_method == "voice_match"


def test_discovery_survives_llm_failure_and_retry_preserves_provenance(store, person, recording, setup_service):
    clear_partner_voice(store, person)
    setup_service.llm.generate_summary.side_effect = RuntimeError("provider unavailable")
    assert run(setup_service, recording, person).status == "failed"
    assert setup_service.voices.load(person.id).interlocutor == profile(1)
    setup_service.llm.generate_summary.side_effect = None
    result = run(setup_service, recording, person, retry=True)
    assert result.status == "complete"
    assert result.dialog.turns[1].attribution_method == "conversation"
    assert result.dialog.turns[1].speaker_similarity is None


@pytest.mark.parametrize("problem", ["missing_wearer", "missing_embedding", "close_to_wearer", "multiple_unknown"])
def test_discovery_does_not_assign_ambiguous_voices(store, person, recording, setup_service, problem):
    clear_partner_voice(store, person)
    if problem == "missing_wearer":
        with store.Session.begin() as session:
            session.get(User, store.owner_user_id).voice_embedding = None
    elif problem == "missing_embedding":
        setup_service.diarizer.extract_per_speaker_embeddings.return_value.pop("S1")
    elif problem == "close_to_wearer":
        setup_service.diarizer.extract_per_speaker_embeddings.return_value["S1"] = vector(0) * 0.7 + vector(1) * np.sqrt(0.51)
    else:
        setup_service.diarizer.diarize.return_value = [DiarizedTurn(0, 2, "S0"),
                                                      DiarizedTurn(2, 3, "S1"), DiarizedTurn(3, 4, "S2")]
        setup_service.diarizer.extract_per_speaker_embeddings.return_value["S2"] = vector(2)
    result = run(setup_service, recording, person)
    assert result.status == "needs_review"
    assert result.voice_discovery is None
    assert setup_service.voices.load(person.id).interlocutor is None
    setup_service.llm.extract_facts.assert_not_called()


def test_discovery_does_not_require_wearer_to_speak_in_the_same_chunk(store, person, recording, setup_service):
    clear_partner_voice(store, person)
    setup_service.diarizer.diarize.return_value = [DiarizedTurn(2, 4, "S1")]
    assert run(setup_service, recording, person).status == "complete"
    assert setup_service.voices.load(person.id).interlocutor == profile(1)


def test_voice_discovery_does_not_overwrite_a_concurrent_profile(store, person, recording, setup_service):
    jobs = AudioJobStore(store)
    jobs.register(recording, person.id)
    attempt = jobs.claim(recording.id)
    saved, created = setup_service.voices.learn_from_conversation(
        DiscoveredVoice(speaker_id="S1", profile=profile(2)), person_id=person.id,
        recording_id=recording.id, attempt_id=attempt,
    )
    assert not created and saved == profile(1)
    assert jobs.get(recording.id).voice_discovery is None


def test_stale_attempt_cannot_save_a_discovered_voice(store, person, recording, setup_service):
    clear_partner_voice(store, person)
    jobs = AudioJobStore(store)
    jobs.register(recording, person.id)
    old = jobs.claim(recording.id)
    jobs.claim(recording.id, recover_interrupted=True)
    with pytest.raises(ValueError, match="no longer current"):
        setup_service.voices.learn_from_conversation(
            DiscoveredVoice(speaker_id="S1", profile=profile(1)), person_id=person.id,
            recording_id=recording.id, attempt_id=old,
        )
    assert setup_service.voices.load(person.id).interlocutor is None


def test_cross_talk_requires_review(person, recording, setup_service):
    setup_service.diarizer.diarize.return_value = [DiarizedTurn(0, 2, "S0"), DiarizedTurn(1.9, 4, "S1")]
    # This fixture is about overlap policy, not waveform slice assertions.
    setup_service.asr.transcribe = lambda seg: RawTranscription(segment=seg, text="overlap", avg_logprob=-0.1, no_speech_prob=0)
    result = run(setup_service, recording, person)
    assert result.status == "needs_review"
    setup_service.llm.extract_facts.assert_not_called()


def test_no_speech_is_distinct_from_failure(person, recording, setup_service):
    setup_service.diarizer.diarize.return_value = []
    result = run(setup_service, recording, person)
    assert result.status == "no_speech" and result.error is None


def test_retry_with_no_speech_clears_previous_attempt_transcript(person, recording, setup_service):
    setup_service.llm.generate_summary.side_effect = RuntimeError("provider unavailable")
    assert run(setup_service, recording, person).transcript
    setup_service.diarizer.diarize.return_value = []
    result = run(setup_service, recording, person, retry=True)
    assert result.status == "no_speech"
    assert result.transcript == "" and result.dialog.turns == []


def test_voice_model_mismatch_fails_before_memory(person, recording, setup_service):
    setup_service.voices.save(VoiceProfile(model="other", embedding=vector(1).tolist()), person_id=person.id)
    result = run(setup_service, recording, person)
    assert result.status == "failed" and result.stage == "attribution"


def test_invalid_extraction_rolls_back_all_memories(store, person, recording, setup_service):
    setup_service.llm.extract_facts.return_value = FactExtractionLLMResponse(facts=[
        ExtractedFact(fact_text="", category="hobby", confidence=0.9),
    ])
    result = run(setup_service, recording, person)
    assert result.status == "failed"
    with store.Session() as session:
        for model in (Episode, Summary, PersonFact):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_database_failure_rolls_back_episode_summary_and_facts(store, person, recording, setup_service):
    def reject_insert(mapper, connection, target):
        raise RuntimeError("injected database failure")
    event.listen(PersonFact, "before_insert", reject_insert)
    try:
        result = run(setup_service, recording, person)
    finally:
        event.remove(PersonFact, "before_insert", reject_insert)
    assert result.status == "failed"
    with store.Session() as session:
        assert session.scalar(select(func.count()).select_from(Episode)) == 0
        assert session.scalar(select(func.count()).select_from(Summary)) == 0


def test_foreign_owner_cannot_enroll_load_or_process(store, person, recording, setup_service):
    other_id, other_person_id = uuid4(), uuid4()
    with store.Session.begin() as session:
        session.add(User(id=other_id, first_name="Other", username="other"))
        session.flush()
        session.add(Person(id=other_person_id, user_id=other_id, first_name="Other", display_name="Other"))
    with pytest.raises(ValueError):
        setup_service.voices.save(profile(2), person_id=other_person_id)
    with pytest.raises(ValueError):
        setup_service.voices.load(other_person_id)
    with pytest.raises(ValueError):
        asyncio.run(setup_service.service.process(recording, other_person_id))
    with pytest.raises(ValueError):
        store.get_disambiguation_hints(other_person_id)


def test_recording_id_cannot_be_rebound_to_another_person(store, person, recording, setup_service):
    setup_service.service.jobs.register(recording, person.id)
    other = store.upsert_person("Another Person")
    with pytest.raises(ValueError, match="different ownership"):
        setup_service.service.jobs.register(recording, other.id)


def test_claim_prevents_duplicate_workers_and_recovery_fences_old_attempt(store, person, recording):
    jobs = AudioJobStore(store)
    jobs.register(recording, person.id)
    old = jobs.claim(recording.id)
    with pytest.raises(ValueError):
        jobs.claim(recording.id)
    current = jobs.claim(recording.id, recover_interrupted=True)
    with pytest.raises(ValueError, match="no longer current"):
        jobs.update(recording.id, old, status="failed")
    jobs.update(recording.id, current, status="failed", error="retry fixture")
    assert jobs.get(recording.id).error == "retry fixture"


def test_recovered_attempt_prevents_old_worker_from_committing_memories(store, person, recording):
    jobs = AudioJobStore(store)
    jobs.register(recording, person.id)
    old = jobs.claim(recording.id)
    current = jobs.claim(recording.id, recover_interrupted=True)
    with pytest.raises(ValueError, match="no longer current"):
        commit_conversation(
            store, person_id=person.id, transcript="Jordan: Hello", time_start=START,
            time_end=START, summary=EpisodeSummaryLLMResponse(summary="Greeting", importance_score=0.1),
            facts=[], edges=[], recording_id=recording.id, attempt_id=old,
        )
    with store.Session() as session:
        assert session.scalar(select(func.count()).select_from(Episode)) == 0
    assert jobs.get(recording.id).attempt_id == current


@pytest.mark.parametrize("person_target", [False, True])
def test_enrollment_persists_both_parties_with_same_model(store, person, tmp_path, person_target):
    source = tmp_path / "enrollment.wav"
    sf.write(source, np.ones(8 * SAMPLE_RATE, np.float32) * 0.1, SAMPLE_RATE)
    diarizer = Mock(embedding_model=VOICE_MODEL)
    diarizer.diarize.return_value = [DiarizedTurn(0, 8, "S0")]
    diarizer.extract_per_speaker_embeddings.return_value = {"S0": vector(1)}
    voices = VoiceStore(store)
    service = VoiceEnrollmentService(voices, diarizer)
    result = service.enroll(source, person_id=person.id if person_target else None)
    loaded = voices.load(person.id)
    saved = loaded.interlocutor if person_target else loaded.user
    assert saved == result and saved.model == VOICE_MODEL


@pytest.mark.parametrize("turns", [[], [DiarizedTurn(0, 2, "S0")],
    [DiarizedTurn(0, 6, "S0"), DiarizedTurn(6, 8, "S1")]])
def test_bad_enrollment_does_not_overwrite_existing_profile(store, person, tmp_path, turns):
    source = tmp_path / "enrollment.wav"
    sf.write(source, np.ones(8 * SAMPLE_RATE, np.float32) * 0.1, SAMPLE_RATE)
    voices = VoiceStore(store)
    voices.save(profile(0), person_id=person.id)
    diarizer = Mock(embedding_model=VOICE_MODEL)
    diarizer.diarize.return_value = turns
    with pytest.raises(ValueError):
        VoiceEnrollmentService(voices, diarizer).enroll(source, person_id=person.id)
    assert voices.load(person.id).interlocutor == profile(0)


def test_enrollment_normalizes_stereo_and_sample_rate(store, person, tmp_path):
    source = tmp_path / "stereo.wav"
    sf.write(source, np.ones((8 * 48000, 2), np.float32) * 0.1, 48000)
    diarizer = Mock(embedding_model=VOICE_MODEL)
    diarizer.diarize.return_value = [DiarizedTurn(0, 8, "S0")]
    diarizer.extract_per_speaker_embeddings.return_value = {"S0": vector(1)}
    VoiceEnrollmentService(VoiceStore(store), diarizer).enroll(source, person_id=person.id)
    samples, rate = diarizer.diarize.call_args.args
    assert samples.ndim == 1 and len(samples) == 8 * SAMPLE_RATE and rate == SAMPLE_RATE


def test_asr_does_not_merge_different_speakers_with_same_role(tmp_path):
    segments = [SpeechSegment(start_time=i, end_time=i + 0.9, speaker_label="unknown",
                              speaker_id=f"S{i}", audio_path=tmp_path / "x.wav") for i in range(2)]
    engine = Mock()
    engine.transcribe.side_effect = lambda segment: RawTranscription(
        segment=segment, text="hi", avg_logprob=-0.1, no_speech_prob=0)
    assert len(transcribe_segments(segments, engine).turns) == 2
