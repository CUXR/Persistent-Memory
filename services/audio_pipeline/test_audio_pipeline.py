"""Tests for the audio segmentation and speaker attribution pipeline.

All ML models (Silero VAD, pyannote diarization, pyannote embedding) are
mocked so the suite runs without GPU, internet access, or HuggingFace tokens.

Coverage:
    - AudioSegment output contract
    - SileroVAD speech detection and gating logic
    - DiarizationEngine turn parsing and embedding extraction
    - SpeakerAttributor cosine-similarity labelling (user / interlocutor / uncertain)
    - Segment stitching across adjacent windows
    - AudioIngestionPipeline window state machine:
      - IDLE → ACCUMULATING on first speech window
      - Silent windows do NOT enter ACCUMULATING
      - ACCUMULATING → finalize after SILENCE_WINDOWS_EOC silent windows
      - Buffer overflow triggers finalization at MAX_CONVERSATION_SAMPLES
      - flush() forces conversation end regardless of state
      - Partial chunks smaller than a VAD window are held until window fills
    - Conversation offset is applied to all output timestamps
    - Error resilience: diarization failure returns empty list (no crash)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

# Make sure the services package is importable when run from repo root.
SERVICES_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

from audio_pipeline.segment import (
    SAMPLE_RATE,
    VAD_WINDOW_SAMPLES,
    MAX_CONVERSATION_SAMPLES,
    SILENCE_WINDOWS_EOC,
    AudioSegment,
)
from audio_pipeline.diarization import DiarizedTurn
from audio_pipeline.speaker_attribution import (
    AttributionConfig,
    SpeakerAttributor,
    _cosine_similarity,
    _stitch_segments,
)
from audio_pipeline.ingestion import AudioIngestionPipeline, ConversationState
from audio_pipeline.vad import SileroVAD


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _silence(n_samples: int) -> np.ndarray:
    """Return a zero-filled audio array."""
    return np.zeros(n_samples, dtype=np.float32)


def _noise(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return low-amplitude noise that passes as non-speech in mock tests."""
    rng = rng or np.random.default_rng(0)
    return (rng.random(n_samples) * 0.01).astype(np.float32)


def _vad_window() -> np.ndarray:
    """Exactly one VAD window of silence."""
    return _silence(VAD_WINDOW_SAMPLES)


def _make_vad(speech: bool = True) -> SileroVAD:
    """Return a SileroVAD whose contains_speech() always returns *speech*."""
    vad = MagicMock(spec=SileroVAD)
    vad.contains_speech.return_value = speech
    return vad


def _make_engine(turns: Optional[list[DiarizedTurn]] = None) -> MagicMock:
    """Return a mock DiarizationEngine."""
    engine = MagicMock()
    engine.diarize.return_value = turns or []
    engine.extract_per_speaker_embeddings.return_value = {}
    return engine


def _make_attributor(segments: Optional[list[AudioSegment]] = None) -> MagicMock:
    attributor = MagicMock(spec=SpeakerAttributor)
    attributor.attribute.return_value = segments or []
    return attributor


def _unit_vec(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_pipeline(
    *,
    vad_speech: bool = True,
    turns: Optional[list[DiarizedTurn]] = None,
    segments: Optional[list[AudioSegment]] = None,
    enrolled: Optional[np.ndarray] = None,
    silence_eoc: int = SILENCE_WINDOWS_EOC,
) -> AudioIngestionPipeline:
    """Convenience factory for AudioIngestionPipeline with mocked dependencies."""
    return AudioIngestionPipeline(
        vad=_make_vad(vad_speech),
        diarization_engine=_make_engine(turns),
        attributor=_make_attributor(segments),
        enrolled_user_embedding=enrolled,
        silence_windows_eoc=silence_eoc,
    )


# ---------------------------------------------------------------------------
# AudioSegment
# ---------------------------------------------------------------------------

class TestAudioSegment:
    def test_from_times_derives_sample_indices(self):
        seg = AudioSegment.from_times(1.0, 3.0, "user", 0.9)
        assert seg.start_sample == SAMPLE_RATE
        assert seg.end_sample == 3 * SAMPLE_RATE

    def test_audio_slice_returns_correct_subarray(self):
        audio = np.arange(SAMPLE_RATE * 4, dtype=np.float32)
        # 1.0–2.0 s = 1 second = SAMPLE_RATE samples
        seg = AudioSegment.from_times(1.0, 2.0, "interlocutor", 0.8)
        sliced = seg.audio_slice(audio)
        assert len(sliced) == SAMPLE_RATE
        # audio is arange so audio[start_sample] == start_sample value
        assert sliced[0] == pytest.approx(float(seg.start_sample))

    def test_repr_contains_label_and_times(self):
        seg = AudioSegment.from_times(0.5, 1.5, "uncertain", 0.0)
        r = repr(seg)
        assert "uncertain" in r
        assert "0.50" in r
        assert "1.50" in r

    def test_start_end_ordering(self):
        seg = AudioSegment.from_times(2.0, 5.0, "user", 0.95)
        assert seg.start_time < seg.end_time
        assert seg.start_sample < seg.end_sample

    def test_sample_rate_constant_is_16000(self):
        assert SAMPLE_RATE == 16_000


# ---------------------------------------------------------------------------
# SileroVAD
# ---------------------------------------------------------------------------

class TestSileroVAD:
    """Tests the SileroVAD wrapper using a mocked torch.hub.load."""

    def _make_silero_mocks(self, timestamps: list[dict]):
        """Return a (model_mock, utils_tuple) suitable for torch.hub.load."""
        mock_model = MagicMock()
        mock_get_speech_timestamps = MagicMock(return_value=timestamps)
        utils = (mock_get_speech_timestamps, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        return mock_model, utils

    def test_contains_speech_true_when_timestamps_present(self):
        vad = SileroVAD(threshold=0.5)
        model, utils = self._make_silero_mocks([{"start": 0, "end": 8000}])
        with patch("torch.hub.load", return_value=(model, utils)):
            result = vad.contains_speech(_silence(VAD_WINDOW_SAMPLES))
        assert result is True

    def test_contains_speech_false_when_no_timestamps(self):
        vad = SileroVAD(threshold=0.5)
        model, utils = self._make_silero_mocks([])
        with patch("torch.hub.load", return_value=(model, utils)):
            result = vad.contains_speech(_silence(VAD_WINDOW_SAMPLES))
        assert result is False

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SileroVAD(threshold=1.5)
        with pytest.raises(ValueError):
            SileroVAD(threshold=0.0)

    def test_model_loaded_once_on_repeated_calls(self):
        vad = SileroVAD(threshold=0.5)
        model, utils = self._make_silero_mocks([])
        with patch("torch.hub.load", return_value=(model, utils)) as mock_load:
            vad.contains_speech(_silence(100))
            vad.contains_speech(_silence(100))
        mock_load.assert_called_once()

    def test_speech_ratio_is_zero_for_silence(self):
        vad = SileroVAD(threshold=0.5)
        model, utils = self._make_silero_mocks([])
        with patch("torch.hub.load", return_value=(model, utils)):
            ratio = vad.speech_ratio(_silence(VAD_WINDOW_SAMPLES))
        assert ratio == pytest.approx(0.0)

    def test_speech_ratio_is_nonzero_for_speech(self):
        vad = SileroVAD(threshold=0.5)
        timestamps = [{"start": 0, "end": 8_000}]  # half of 16k window
        model, utils = self._make_silero_mocks(timestamps)
        with patch("torch.hub.load", return_value=(model, utils)):
            ratio = vad.speech_ratio(_silence(16_000))
        assert ratio == pytest.approx(0.5)

    def test_speech_ratio_capped_at_one(self):
        vad = SileroVAD(threshold=0.5)
        # Report more speech than the total audio length — should be capped.
        timestamps = [{"start": 0, "end": 200_000}]
        model, utils = self._make_silero_mocks(timestamps)
        with patch("torch.hub.load", return_value=(model, utils)):
            ratio = vad.speech_ratio(_silence(16_000))
        assert ratio == pytest.approx(1.0)

    def test_empty_audio_returns_false(self):
        vad = SileroVAD(threshold=0.5)
        model, utils = self._make_silero_mocks([])
        with patch("torch.hub.load", return_value=(model, utils)):
            result = vad.contains_speech(np.array([], dtype=np.float32))
        assert result is False


# ---------------------------------------------------------------------------
# DiarizedTurn
# ---------------------------------------------------------------------------

class TestDiarizedTurn:
    def test_duration_property(self):
        turn = DiarizedTurn(start=1.5, end=4.5, speaker_id="SPEAKER_00")
        assert turn.duration == pytest.approx(3.0)

    def test_zero_duration_turn(self):
        turn = DiarizedTurn(start=2.0, end=2.0, speaker_id="SPEAKER_01")
        assert turn.duration == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SpeakerAttributor — cosine similarity helpers
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self):
        v = _unit_vec(64)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors_have_similarity_minus_one(self):
        v = _unit_vec(64)
        assert _cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors_have_similarity_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(10, dtype=np.float32)
        b = _unit_vec(10)
        assert _cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SpeakerAttributor — labelling
# ---------------------------------------------------------------------------

class TestSpeakerAttributor:
    def _make_turns(self) -> list[DiarizedTurn]:
        return [
            DiarizedTurn(0.0, 5.0, "SPEAKER_00"),
            DiarizedTurn(5.5, 10.0, "SPEAKER_01"),
        ]

    def test_user_label_when_high_similarity(self):
        cfg = AttributionConfig(user_threshold=0.75, interlocutor_threshold=0.45)
        attributor = SpeakerAttributor(cfg)
        enrolled = _unit_vec(64)
        # SPEAKER_00 embedding ≈ enrolled (high similarity)
        embeddings = {
            "SPEAKER_00": enrolled.copy(),
            "SPEAKER_01": _unit_vec(64, seed=42),
        }
        # Make SPEAKER_01 orthogonal so similarity → 0
        orth = enrolled.copy()
        orth[0] = 0.0
        orth[1] = 1.0
        orth = orth / np.linalg.norm(orth)
        embeddings["SPEAKER_01"] = orth

        segments = attributor.attribute(self._make_turns(), embeddings, enrolled)
        labels = {s.speaker_id: s.speaker_label for s in segments}
        assert labels["SPEAKER_00"] == "user"

    def test_interlocutor_label_when_low_similarity(self):
        cfg = AttributionConfig(user_threshold=0.75, interlocutor_threshold=0.45)
        attributor = SpeakerAttributor(cfg)
        enrolled = _unit_vec(64)
        opposite = -enrolled
        embeddings = {
            "SPEAKER_00": enrolled.copy(),
            "SPEAKER_01": opposite,
        }
        segments = attributor.attribute(self._make_turns(), embeddings, enrolled)
        labels = {s.speaker_id: s.speaker_label for s in segments}
        assert labels["SPEAKER_01"] == "interlocutor"

    def test_uncertain_label_in_ambiguous_zone(self):
        cfg = AttributionConfig(user_threshold=0.85, interlocutor_threshold=0.15)
        attributor = SpeakerAttributor(cfg)
        enrolled = _unit_vec(64)
        # Create embedding with ~0.5 cosine similarity — lands in uncertain zone.
        mid = enrolled.copy()
        mid[: len(mid) // 2] *= -1
        mid = mid / np.linalg.norm(mid)
        embeddings = {"SPEAKER_00": mid, "SPEAKER_01": -enrolled}
        segments = attributor.attribute(self._make_turns(), embeddings, enrolled)
        labels = {s.speaker_id: s.speaker_label for s in segments}
        assert labels["SPEAKER_00"] == "uncertain"

    def test_no_enrolled_embedding_uses_relative_attribution(self):
        """With 2 speakers and no enrollment, one gets 'user' and one 'interlocutor'."""
        attributor = SpeakerAttributor()
        embeddings = {
            "SPEAKER_00": _unit_vec(64, seed=1),
            "SPEAKER_01": _unit_vec(64, seed=2),
        }
        segments = attributor.attribute(self._make_turns(), embeddings, enrolled_user_embedding=None)
        labels = {s.speaker_label for s in segments}
        assert "user" in labels
        assert "interlocutor" in labels

    def test_no_turns_returns_empty(self):
        attributor = SpeakerAttributor()
        result = attributor.attribute([], {}, None)
        assert result == []

    def test_missing_speaker_embedding_yields_uncertain(self):
        attributor = SpeakerAttributor()
        enrolled = _unit_vec(64)
        turns = [DiarizedTurn(0.0, 3.0, "SPEAKER_00")]
        # No embeddings at all
        segments = attributor.attribute(turns, {}, enrolled)
        assert segments[0].speaker_label == "uncertain"

    def test_turns_below_min_duration_are_dropped(self):
        cfg = AttributionConfig(min_turn_duration=1.0)
        attributor = SpeakerAttributor(cfg)
        short_turns = [DiarizedTurn(0.0, 0.05, "SPEAKER_00")]  # 50ms — below threshold
        result = attributor.attribute(short_turns, {}, None)
        assert result == []

    def test_conversation_offset_applied_to_timestamps(self):
        attributor = SpeakerAttributor()
        enrolled = _unit_vec(64)
        turns = [DiarizedTurn(0.0, 5.0, "SPEAKER_00")]
        embeddings = {"SPEAKER_00": enrolled.copy()}
        offset = 30.0  # 30 seconds into the recording

        segments = attributor.attribute(turns, embeddings, enrolled, conversation_offset_seconds=offset)
        assert segments[0].start_time == pytest.approx(30.0)
        assert segments[0].end_time == pytest.approx(35.0)
        assert segments[0].start_sample == int(30.0 * SAMPLE_RATE)

    def test_confidence_is_in_zero_to_one_range(self):
        attributor = SpeakerAttributor()
        enrolled = _unit_vec(64)
        turns = [DiarizedTurn(0.0, 5.0, "SPEAKER_00")]
        embeddings = {"SPEAKER_00": enrolled.copy()}
        segments = attributor.attribute(turns, embeddings, enrolled)
        assert 0.0 <= segments[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# Segment stitching
# ---------------------------------------------------------------------------

class TestStitchSegments:
    def _seg(self, start, end, label="user", sid="SPEAKER_00"):
        return AudioSegment.from_times(start, end, label, 0.9, speaker_id=sid)

    def test_adjacent_same_speaker_stitched(self):
        segments = [
            self._seg(0.0, 2.0),
            self._seg(2.1, 4.0),  # 0.1s gap — within default 0.3s threshold
        ]
        result = _stitch_segments(segments, gap_threshold=0.3)
        assert len(result) == 1
        assert result[0].start_time == pytest.approx(0.0)
        assert result[0].end_time == pytest.approx(4.0)

    def test_large_gap_not_stitched(self):
        segments = [
            self._seg(0.0, 2.0),
            self._seg(3.0, 5.0),  # 1s gap — exceeds threshold
        ]
        result = _stitch_segments(segments, gap_threshold=0.3)
        assert len(result) == 2

    def test_different_speakers_not_stitched(self):
        segments = [
            self._seg(0.0, 2.0, sid="SPEAKER_00"),
            self._seg(2.1, 4.0, sid="SPEAKER_01"),
        ]
        result = _stitch_segments(segments, gap_threshold=0.3)
        assert len(result) == 2

    def test_different_labels_not_stitched(self):
        seg_user = AudioSegment.from_times(0.0, 2.0, "user", 0.9, "SPEAKER_00")
        seg_inter = AudioSegment.from_times(2.1, 4.0, "interlocutor", 0.9, "SPEAKER_00")
        result = _stitch_segments([seg_user, seg_inter], gap_threshold=0.3)
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        assert _stitch_segments([], gap_threshold=0.3) == []

    def test_single_segment_unchanged(self):
        segments = [self._seg(0.0, 3.0)]
        result = _stitch_segments(segments, gap_threshold=0.3)
        assert len(result) == 1

    def test_stitched_confidence_is_maximum(self):
        s1 = AudioSegment.from_times(0.0, 2.0, "user", 0.6, "SPEAKER_00")
        s2 = AudioSegment.from_times(2.1, 4.0, "user", 0.9, "SPEAKER_00")
        result = _stitch_segments([s1, s2], gap_threshold=0.3)
        assert result[0].confidence == pytest.approx(0.9)

    def test_sample_indices_updated_after_stitch(self):
        s1 = AudioSegment.from_times(1.0, 2.0, "user", 0.9, "SPEAKER_00")
        s2 = AudioSegment.from_times(2.1, 3.0, "user", 0.9, "SPEAKER_00")
        result = _stitch_segments([s1, s2], gap_threshold=0.3)
        assert result[0].start_sample == int(1.0 * SAMPLE_RATE)
        assert result[0].end_sample == int(3.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# AudioIngestionPipeline — state machine
# ---------------------------------------------------------------------------

class TestPipelineStateTransitions:
    def test_initial_state_is_idle(self):
        pipeline = _make_pipeline()
        assert pipeline.state == ConversationState.IDLE

    def test_silent_window_stays_idle(self):
        pipeline = _make_pipeline(vad_speech=False)
        pipeline.push_audio(_vad_window())
        assert pipeline.state == ConversationState.IDLE

    def test_speech_window_enters_accumulating(self):
        pipeline = _make_pipeline(vad_speech=True)
        pipeline.push_audio(_vad_window())
        assert pipeline.state == ConversationState.ACCUMULATING

    def test_returns_to_idle_after_finalization(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=1)
        pipeline.push_audio(_vad_window())  # → ACCUMULATING
        # Now set VAD to return silence to trigger EOC
        pipeline._vad.contains_speech.return_value = False
        pipeline.push_audio(_vad_window())  # silent → EOC
        assert pipeline.state == ConversationState.IDLE

    def test_consecutive_silence_triggers_eoc(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=2)
        pipeline.push_audio(_vad_window())  # → ACCUMULATING (speech)
        pipeline._vad.contains_speech.return_value = False
        pipeline.push_audio(_vad_window())  # silent window 1
        assert pipeline.state == ConversationState.ACCUMULATING
        pipeline.push_audio(_vad_window())  # silent window 2 → EOC
        assert pipeline.state == ConversationState.IDLE

    def test_reset_returns_to_idle(self):
        pipeline = _make_pipeline(vad_speech=True)
        pipeline.push_audio(_vad_window())
        assert pipeline.state == ConversationState.ACCUMULATING
        pipeline.reset()
        assert pipeline.state == ConversationState.IDLE


class TestPipelineVADGating:
    def test_diarization_not_called_for_silent_windows(self):
        pipeline = _make_pipeline(vad_speech=False)
        for _ in range(5):
            pipeline.push_audio(_vad_window())
        pipeline._diarization_engine.diarize.assert_not_called()

    def test_diarization_called_after_eoc(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=1)
        pipeline.push_audio(_vad_window())  # speech → ACCUMULATING
        pipeline._vad.contains_speech.return_value = False
        pipeline.push_audio(_vad_window())  # silence → EOC → diarize
        pipeline._diarization_engine.diarize.assert_called_once()

    def test_vad_called_once_per_window(self):
        pipeline = _make_pipeline(vad_speech=False)
        pipeline.push_audio(_vad_window())
        pipeline.push_audio(_vad_window())
        assert pipeline._vad.contains_speech.call_count == 2

class TestPipelineWindowAccumulation:
    def test_partial_chunk_held_until_window_full(self):
        pipeline = _make_pipeline(vad_speech=False)
        half_window = _silence(VAD_WINDOW_SAMPLES // 2)
        pipeline.push_audio(half_window)
        # VAD not yet called — window not complete.
        pipeline._vad.contains_speech.assert_not_called()
        pipeline.push_audio(half_window)
        # Now a full window exists — VAD called.
        pipeline._vad.contains_speech.assert_called_once()

    def test_oversized_chunk_produces_multiple_vad_windows(self):
        pipeline = _make_pipeline(vad_speech=False)
        three_windows = _silence(VAD_WINDOW_SAMPLES * 3)
        pipeline.push_audio(three_windows)
        assert pipeline._vad.contains_speech.call_count == 3

    def test_conversation_duration_tracks_accumulation(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=999)
        for _ in range(4):
            pipeline.push_audio(_vad_window())
        # 4 windows × VAD_WINDOW_SECONDS
        expected = 4 * (VAD_WINDOW_SAMPLES / SAMPLE_RATE)
        assert pipeline.conversation_duration_seconds == pytest.approx(expected)

    def test_buffer_overflow_triggers_finalization(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=9999)
        # Push enough audio to overflow the 10-minute cap.
        overflow_samples = MAX_CONVERSATION_SAMPLES + VAD_WINDOW_SAMPLES * 3
        chunk = _silence(overflow_samples)
        pipeline.push_audio(chunk)
        # Pipeline should have processed and returned to IDLE.
        assert pipeline.state == ConversationState.IDLE

    def test_first_speech_window_included_in_conversation(self):
        """The triggering VAD window is not discarded when transitioning."""
        captured_audio: list[np.ndarray] = []

        def fake_diarize(audio, sr):
            captured_audio.append(audio.copy())
            return []

        pipeline = _make_pipeline(vad_speech=True, silence_eoc=1)
        pipeline._diarization_engine.diarize.side_effect = fake_diarize

        pipeline.push_audio(_vad_window())  # speech → ACCUMULATING
        pipeline._vad.contains_speech.return_value = False
        pipeline.push_audio(_vad_window())  # silence → EOC

        assert len(captured_audio) == 1
        # Buffer must contain at least the two windows pushed.
        assert len(captured_audio[0]) >= VAD_WINDOW_SAMPLES * 2


class TestPipelineFlush:
    def test_flush_idle_returns_empty(self):
        pipeline = _make_pipeline(vad_speech=False)
        result = pipeline.flush()
        assert result == []

    def test_flush_accumulating_returns_segments(self):
        seg = AudioSegment.from_times(0.0, 1.0, "user", 0.9)
        turns = [DiarizedTurn(0.0, 1.0, "SPEAKER_00")]
        pipeline = _make_pipeline(vad_speech=True, turns=turns, segments=[seg], silence_eoc=999)
        pipeline.push_audio(_vad_window())  # → ACCUMULATING
        result = pipeline.flush()
        assert len(result) == 1
        assert result[0].speaker_label == "user"

    def test_flush_resets_state_to_idle(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=999)
        pipeline.push_audio(_vad_window())
        assert pipeline.state == ConversationState.ACCUMULATING
        pipeline.flush()
        assert pipeline.state == ConversationState.IDLE

    def test_flush_partial_audio_included(self):
        """Partial chunk (< VAD_WINDOW_SAMPLES) is included when flush() is called."""
        captured: list[np.ndarray] = []

        def fake_diarize(audio, sr):
            captured.append(audio.copy())
            return []

        pipeline = _make_pipeline(vad_speech=True, silence_eoc=999)
        pipeline._diarization_engine.diarize.side_effect = fake_diarize

        pipeline.push_audio(_vad_window())  # speech → ACCUMULATING
        partial = _silence(1000)
        pipeline.push_audio(partial)   # partial — not a full VAD window
        pipeline.flush()

        assert len(captured) == 1
        assert len(captured[0]) >= VAD_WINDOW_SAMPLES + 1000


class TestPipelineSegmentOutput:
    def test_segments_returned_on_eoc(self):
        seg = AudioSegment.from_times(0.5, 3.0, "interlocutor", 0.88)
        turns = [DiarizedTurn(0.5, 3.0, "SPEAKER_00")]
        pipeline = _make_pipeline(vad_speech=True, turns=turns, segments=[seg], silence_eoc=1)
        pipeline.push_audio(_vad_window())
        pipeline._vad.contains_speech.return_value = False
        results = pipeline.push_audio(_vad_window())
        assert len(results) == 1
        assert results[0].speaker_label == "interlocutor"

    def test_no_segments_for_empty_diarization(self):
        pipeline = _make_pipeline(vad_speech=True, turns=[], segments=[], silence_eoc=1)
        pipeline.push_audio(_vad_window())
        pipeline._vad.contains_speech.return_value = False
        results = pipeline.push_audio(_vad_window())
        assert results == []

    def test_diarization_error_returns_empty_not_crash(self):
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=1)
        pipeline._diarization_engine.diarize.side_effect = RuntimeError("diarization failed")
        pipeline.push_audio(_vad_window())
        pipeline._vad.contains_speech.return_value = False
        results = pipeline.push_audio(_vad_window())
        assert results == []
        assert pipeline.state == ConversationState.IDLE  # recovered to IDLE
    def test_conversation_offset_forwarded_to_attributor(self):
        """Attribution receives the correct conversation_offset_seconds."""
        pipeline = _make_pipeline(vad_speech=True, silence_eoc=1)
        pipeline._diarization_engine.diarize.return_value = [DiarizedTurn(0.0, 2.0, "S0")]
        pipeline._diarization_engine.extract_per_speaker_embeddings.return_value = {}

        # Push enough silence first so the conversation starts at a non-zero offset.
        # (The VAD returns False for the first two windows.)
        pipeline._vad.contains_speech.return_value = False
        pipeline.push_audio(_vad_window())
        pipeline.push_audio(_vad_window())

        # Now speech starts.
        pipeline._vad.contains_speech.return_value = True
        pipeline.push_audio(_vad_window())

        pipeline._vad.contains_speech.return_value = False
        pipeline.push_audio(_vad_window())  # → EOC

        call_kwargs = pipeline._attributor.attribute.call_args
        offset = call_kwargs[1].get(
            "conversation_offset_seconds",
            call_kwargs[0][3] if len(call_kwargs[0]) > 3 else None,
        )
        # Two silent windows before speech → offset ≥ 2 × VAD_WINDOW_SECONDS
        assert offset is not None
        assert offset >= 2 * (VAD_WINDOW_SAMPLES / SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Integration-style: attributor + stitching end-to-end (no real models)
# ---------------------------------------------------------------------------

class TestAttributorStitchingIntegration:
    def test_multi_speaker_conversation_end_to_end(self):
        """End-to-end attribution with two speakers and stitching."""
        enrolled = _unit_vec(64)
        cfg = AttributionConfig(user_threshold=0.75, interlocutor_threshold=0.45, stitch_gap_seconds=0.5)
        attributor = SpeakerAttributor(cfg)

        turns = [
            DiarizedTurn(0.0, 3.0, "SPEAKER_00"),
            DiarizedTurn(3.1, 5.5, "SPEAKER_01"),
            DiarizedTurn(5.6, 8.0, "SPEAKER_00"),  # same speaker with small gap → stitched
        ]

        # SPEAKER_00 ≈ enrolled, SPEAKER_01 is opposite.
        embeddings = {
            "SPEAKER_00": enrolled.copy(),
            "SPEAKER_01": -enrolled,
        }

        segments = attributor.attribute(turns, embeddings, enrolled)
        labels = [s.speaker_label for s in segments]

        # SPEAKER_00 appears in turns 0 and 2 (gap 0.1s < 0.5s) → should stitch to 1 segment.
        user_segments = [s for s in segments if s.speaker_label == "user"]
        assert len(user_segments) == 1
        assert user_segments[0].start_time == pytest.approx(0.0)
        assert user_segments[0].end_time == pytest.approx(8.0)

        # SPEAKER_01 is one separate segment.
        interlocutor_segments = [s for s in segments if s.speaker_label == "interlocutor"]
        assert len(interlocutor_segments) == 1

    def test_three_speakers_with_no_enrollment(self):
        """Three speakers + no enrolled embedding → all uncertain (not enough for relative)."""
        attributor = SpeakerAttributor()
        turns = [
            DiarizedTurn(0.0, 2.0, "S0"),
            DiarizedTurn(2.5, 4.0, "S1"),
            DiarizedTurn(4.5, 6.0, "S2"),
        ]
        embeddings = {f"S{i}": _unit_vec(64, seed=i) for i in range(3)}
        segments = attributor.attribute(turns, embeddings, enrolled_user_embedding=None)
        assert all(s.speaker_label == "uncertain" for s in segments)
