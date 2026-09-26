"""Capture, attribution, and real adapter boundaries without downloading models."""
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4
import sys

import numpy as np
import pytest
import soundfile as sf

from audio_pipeline.diarization import DiarizationEngine, DiarizedTurn
from audio_pipeline.ingestion import AudioIngestionPipeline, ConversationState
from audio_pipeline.segment import Recording, SpeechSegment, VoiceProfile, SAMPLE_RATE, VOICE_MODEL
from audio_pipeline.speaker_attribution import SpeakerAttributor, AttributionConfig
from audio_pipeline.vad import SileroVAD

START = datetime(2026, 9, 26, tzinfo=timezone.utc)


def audio(seconds, value=1):
    return np.full(round(seconds * SAMPLE_RATE), value, dtype=np.float32)


def vector(index):
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1
    return result


def profile(index):
    return VoiceProfile(model=VOICE_MODEL, embedding=vector(index).tolist())


def capture(tmp_path, **kwargs):
    vad = Mock()
    vad.contains_speech.side_effect = lambda samples, _: bool(np.any(samples))
    return AudioIngestionPipeline(vad, tmp_path, START, **kwargs)


def test_silence_never_creates_a_recording(tmp_path):
    pipeline = capture(tmp_path)
    assert pipeline.push_audio(audio(25, 0)) == []
    assert pipeline.flush() == []
    assert list(tmp_path.iterdir()) == []


def test_ten_second_gate_retains_trigger_and_final_silence(tmp_path):
    pipeline = capture(tmp_path)
    assert pipeline.push_audio(audio(9)) == []
    pipeline._vad.contains_speech.assert_not_called()
    assert pipeline.push_audio(audio(1)) == []
    assert pipeline.state == ConversationState.ACCUMULATING
    [recording] = pipeline.push_audio(audio(10, 0))
    assert recording.started_at == START
    assert recording.duration == 20
    samples, rate = sf.read(recording.audio_path, dtype="float32")
    np.testing.assert_array_equal(samples, np.concatenate([audio(10), audio(10, 0)]))
    assert rate == SAMPLE_RATE
    assert Recording.model_validate_json(recording.audio_path.with_suffix(".json").read_text()) == recording


def test_short_speech_is_checked_on_flush(tmp_path):
    pipeline = capture(tmp_path)
    pipeline.push_audio(audio(3))
    [recording] = pipeline.flush()
    assert recording.duration == 3
    pipeline._vad.contains_speech.assert_called_once()
    assert pipeline.flush() == []


def test_partial_flush_preserves_next_recording_clock(tmp_path):
    pipeline = capture(tmp_path)
    pipeline.push_audio(audio(10, 0))
    pipeline.push_audio(audio(11))
    [first] = pipeline.flush()
    assert first.started_at == START + timedelta(seconds=10)
    assert first.duration == 11
    pipeline.push_audio(audio(10))
    [second] = pipeline.flush()
    assert second.started_at == START + timedelta(seconds=21)
    assert second.id != first.id


def test_overflow_retains_remaining_speech(tmp_path):
    pipeline = capture(tmp_path, max_conversation_seconds=20)
    [first] = pipeline.push_audio(audio(30))
    assert first.duration == 20
    assert pipeline.state == ConversationState.ACCUMULATING
    [second] = pipeline.flush()
    assert second.duration == 10
    assert second.started_at == START + timedelta(seconds=20)


def test_reset_discards_partial_samples_and_restarts_clock(tmp_path):
    pipeline = capture(tmp_path)
    pipeline.push_audio(audio(3))
    new_start = START + timedelta(hours=1)
    pipeline.reset(new_start)
    assert pipeline.flush() == []
    pipeline.push_audio(audio(2))
    assert pipeline.flush()[0].started_at == new_start


def test_capture_write_failure_does_not_discard_buffer(tmp_path, monkeypatch):
    pipeline = capture(tmp_path)
    pipeline.push_audio(audio(3))
    writer = sf.write
    monkeypatch.setattr(sf, "write", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        pipeline.flush()
    assert pipeline.conversation_duration_seconds == 3
    monkeypatch.setattr(sf, "write", writer)
    assert pipeline.flush()[0].duration == 3


def test_write_retry_keeps_capped_recording_separate_from_remaining_audio(tmp_path, monkeypatch):
    pipeline = capture(tmp_path, max_conversation_seconds=20)
    writer = sf.write
    monkeypatch.setattr(sf, "write", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        pipeline.push_audio(audio(35))
    monkeypatch.setattr(sf, "write", writer)
    first, second = pipeline.flush()
    assert [first.duration, second.duration] == [20, 15]
    assert second.started_at == START + timedelta(seconds=20)


def test_flush_after_vad_failure_drains_full_windows_before_partial(tmp_path):
    pipeline = capture(tmp_path, max_conversation_seconds=20)
    pipeline._vad.contains_speech.side_effect = RuntimeError("VAD unavailable")
    with pytest.raises(RuntimeError):
        pipeline.push_audio(audio(35))
    pipeline._vad.contains_speech.side_effect = lambda samples, _: bool(np.any(samples))
    first, second = pipeline.flush()
    assert [first.duration, second.duration] == [20, 15]
    assert second.started_at == START + timedelta(seconds=20)


@pytest.mark.parametrize("samples", [np.zeros((2, 2)), np.array([np.nan]), np.array([np.inf])])
def test_invalid_capture_audio_rejected(tmp_path, samples):
    with pytest.raises(ValueError):
        capture(tmp_path).push_audio(samples)


@pytest.mark.parametrize("options", [{"window_seconds": 0}, {"silence_windows_eoc": 0},
                                     {"max_conversation_seconds": 15}])
def test_invalid_capture_configuration_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        capture(tmp_path, **options)


@pytest.mark.parametrize("embedding", [[0] * 512, [1] * 256, [float("nan")] * 512])
def test_invalid_enrollment_vectors_rejected(embedding):
    with pytest.raises(ValueError):
        VoiceProfile(model=VOICE_MODEL, embedding=embedding)


def test_vectors_are_normalized():
    assert np.linalg.norm(VoiceProfile(model=VOICE_MODEL, embedding=[2] * 512).embedding) == pytest.approx(1)


def attribute(tmp_path, embeddings, user=None, person=None, config=None):
    recording = Recording(audio_path=tmp_path / "recording.wav", started_at=START, sample_count=10 * SAMPLE_RATE)
    person_id = uuid4()
    turns = [DiarizedTurn(index * 2, index * 2 + 1, speaker) for index, speaker in enumerate(embeddings)]
    segments = SpeakerAttributor(config).attribute(
        turns, embeddings, recording, embedding_model=VOICE_MODEL, user_profile=user,
        interlocutor_profile=person, interlocutor_id=person_id,
    )
    return segments, person_id


def test_both_enrolled_speakers_match_and_bystander_remains_unknown(tmp_path):
    segments, person_id = attribute(tmp_path, {"S0": vector(0), "S1": vector(1), "S2": vector(2)}, profile(0), profile(1))
    assert [segment.speaker_label for segment in segments] == ["user", "interlocutor", "unknown"]
    assert [segment.person_id for segment in segments] == [None, person_id, None]
    assert [segment.speaker_id for segment in segments] == ["S0", "S1", "S2"]


def test_no_enrollment_does_not_guess_wearer(tmp_path):
    segments, _ = attribute(tmp_path, {"S0": vector(0), "S1": vector(1)})
    assert all(segment.speaker_label == "unknown" for segment in segments)


def test_missing_interlocutor_profile_does_not_mean_everyone_else_is_partner(tmp_path):
    segments, _ = attribute(tmp_path, {"S0": vector(0), "S1": vector(1)}, user=profile(0))
    assert [segment.speaker_label for segment in segments] == ["user", "unknown"]


def test_ambiguous_profiles_abstain(tmp_path):
    segments, _ = attribute(tmp_path, {"S0": vector(0)}, profile(0), profile(0))
    assert segments[0].speaker_label == "unknown"


def test_missing_turn_embedding_still_preserves_the_turn(tmp_path):
    segments, _ = attribute(tmp_path, {"S0": None}, profile(0), profile(1))
    assert segments[0].speaker_label == "unknown"
    assert segments[0].speaker_similarity is None


def test_model_mismatch_rejected(tmp_path):
    other = VoiceProfile(model="different-model", embedding=vector(0).tolist())
    with pytest.raises(ValueError, match="model mismatch"):
        attribute(tmp_path, {"S0": vector(0)}, user=other)


def test_match_threshold_and_margin_are_both_required(tmp_path):
    near = vector(0) * 0.8 + vector(1) * 0.6
    segments, _ = attribute(tmp_path, {"S0": near}, profile(0), profile(1),
                            AttributionConfig(match_threshold=0.75, min_margin=0.25))
    assert segments[0].speaker_label == "unknown"


def test_community_output_object_is_unwrapped():
    annotation = Mock()
    annotation.itertracks.return_value = [(SimpleNamespace(start=1, end=3), None, "S0")]
    engine = DiarizationEngine("fixture")
    engine._pipeline = Mock(return_value=SimpleNamespace(speaker_diarization=annotation))
    assert engine.diarize(audio(5)) == [DiarizedTurn(1, 3, "S0")]
    assert engine._pipeline.call_args.args[0]["waveform"].shape == (1, 5 * SAMPLE_RATE)


def test_embedding_extraction_uses_crop_not_whole_file(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote.core", SimpleNamespace(Segment=lambda start, end: (start, end)))
    engine = DiarizationEngine("fixture")
    engine._embedding = Mock()
    engine._embedding.crop.return_value = vector(0)
    np.testing.assert_array_equal(engine.extract_speaker_embedding(audio(5), 1, 3), vector(0))
    engine._embedding.assert_not_called()
    assert engine._embedding.crop.call_args.args[1] == (1, 3)


def test_pyannote_loading_uses_current_token_argument(monkeypatch):
    pipeline = Mock()
    model = Mock()
    inference = Mock()
    monkeypatch.setitem(sys.modules, "pyannote.audio",
                        SimpleNamespace(Pipeline=pipeline, Model=model, Inference=inference))
    engine = DiarizationEngine("fixture-token")
    engine._ensure_pipeline()
    engine._ensure_embedding()
    pipeline.from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-community-1", token="fixture-token")
    model.from_pretrained.assert_called_once_with(VOICE_MODEL, token="fixture-token")
    assert inference.call_args.kwargs["window"] == "whole"


def test_overlapping_speech_is_excluded_from_voice_embedding():
    engine = DiarizationEngine("fixture")
    engine.extract_speaker_embedding = Mock(return_value=vector(0))
    turns = [DiarizedTurn(0, 2, "S0"), DiarizedTurn(1, 3, "S1"), DiarizedTurn(4, 6, "S0")]
    embeddings = engine.extract_per_speaker_embeddings(audio(6), turns)
    assert set(embeddings) == {"S0"}
    assert engine.extract_speaker_embedding.call_args.args[1:3] == (4, 6)


def test_silero_loads_once_and_handles_empty_audio(monkeypatch):
    load = Mock(return_value=object())
    timestamps = Mock(return_value=[{"start": 0, "end": 1000}])
    monkeypatch.setitem(sys.modules, "silero_vad", SimpleNamespace(
        load_silero_vad=load, get_speech_timestamps=timestamps))
    vad = SileroVAD()
    assert not vad.contains_speech(audio(0))
    load.assert_not_called()
    assert vad.contains_speech(audio(1))
    assert vad.contains_speech(audio(1))
    load.assert_called_once()
