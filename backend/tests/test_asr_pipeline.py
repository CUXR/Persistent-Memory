"""Tests for backend/app/services/asr.py — orchestration only.

Uses an in-test stub engine so no Whisper model is loaded. The engine
itself (asr_engine.py) is tested separately in test_asr_engine.py.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import sys

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schema.asr import RawTranscription, SpeechSegment
from app.services.asr import transcribe_segments
from app.services.asr_engine import WhisperEngine


# ── Helpers ──────────────────────────────────────────────────


def _seg(
    start: float,
    end: float,
    speaker: str = "user",
    audio_path: Path = Path("/tmp/conv.wav"),
) -> SpeechSegment:
    return SpeechSegment(
        start_time=start,
        end_time=end,
        speaker_label=speaker,  # type: ignore[arg-type]
        audio_path=audio_path,
    )


def _raw(
    segment: SpeechSegment,
    text: str,
    *,
    avg_logprob: float = -0.1,
    no_speech_prob: float = 0.0,
) -> RawTranscription:
    return RawTranscription(
        segment=segment,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


class StubEngine:
    """Returns canned RawTranscriptions keyed by segment object identity.

    Any segment listed in ``raises_on`` raises an exception when transcribed,
    exercising the orchestrator's per-segment failure handling.
    """

    def __init__(
        self,
        responses: dict[int, RawTranscription] | None = None,
        raises_on: set[int] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.raises_on = raises_on or set()
        self.calls: list[SpeechSegment] = []

    def transcribe(self, segment: SpeechSegment) -> RawTranscription:
        self.calls.append(segment)
        if id(segment) in self.raises_on:
            raise RuntimeError("simulated ASR failure")
        return self.responses[id(segment)]


# ── Tests ────────────────────────────────────────────────────


class TestTranscribeSegments:
    def test_empty_input_returns_empty_dialog_without_calling_engine(self) -> None:
        engine = StubEngine()
        dialog = transcribe_segments([], engine)
        assert dialog.turns == []
        assert engine.calls == []

    def test_happy_path_assembles_dialog(self) -> None:
        seg_a = _seg(0.0, 1.0, speaker="user")
        seg_b = _seg(1.2, 2.0, speaker="user")
        seg_c = _seg(2.5, 3.5, speaker="interlocutor")
        engine = StubEngine(
            responses={
                id(seg_a): _raw(seg_a, "hello"),
                id(seg_b): _raw(seg_b, "there"),
                id(seg_c): _raw(seg_c, "hi back"),
            }
        )

        dialog = transcribe_segments([seg_a, seg_b, seg_c], engine)

        assert [t.speaker for t in dialog.turns] == ["user", "interlocutor"]
        assert [t.text for t in dialog.turns] == ["hello there", "hi back"]
        assert dialog.turns[0].segment_count == 2
        assert dialog.turns[1].segment_count == 1
        # Order of engine calls matches input order.
        assert engine.calls == [seg_a, seg_b, seg_c]

    def test_engine_failure_on_one_segment_does_not_abort_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        seg_a = _seg(0.0, 1.0, speaker="user")
        seg_bad = _seg(1.2, 2.0, speaker="user")
        seg_c = _seg(2.5, 3.5, speaker="user")
        engine = StubEngine(
            responses={
                id(seg_a): _raw(seg_a, "before"),
                id(seg_c): _raw(seg_c, "after"),
            },
            raises_on={id(seg_bad)},
        )

        with caplog.at_level(logging.ERROR, logger="app.services.asr"):
            dialog = transcribe_segments([seg_a, seg_bad, seg_c], engine)

        # Surviving segments are still transcribed; failed one is skipped.
        # seg_a (0..1) and seg_c (2.5..3.5) — gap of 1.5s, both "user", so
        # they merge under the default 2.0s threshold.
        assert len(dialog.turns) == 1
        turn = dialog.turns[0]
        assert turn.text == "before after"
        assert turn.segment_count == 2
        # Failure was logged.
        assert any(
            "ASR engine failed" in record.message for record in caplog.records
        )

    def test_all_segments_failing_yields_empty_dialog(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        segs = [_seg(0.0, 1.0), _seg(1.5, 2.5)]
        engine = StubEngine(raises_on={id(s) for s in segs})

        with caplog.at_level(logging.ERROR, logger="app.services.asr"):
            dialog = transcribe_segments(segs, engine)

        assert dialog.turns == []
        # One log line per failed segment.
        failure_logs = [
            r for r in caplog.records if "ASR engine failed" in r.message
        ]
        assert len(failure_logs) == 2

    def test_all_silent_segments_yield_empty_dialog(self) -> None:
        # Engine returns transcriptions but Whisper flagged them as silence.
        seg_a = _seg(0.0, 1.0)
        seg_b = _seg(1.2, 2.0)
        engine = StubEngine(
            responses={
                id(seg_a): _raw(seg_a, "hallucinated", no_speech_prob=0.95),
                id(seg_b): _raw(seg_b, "", no_speech_prob=0.1),  # empty text
            }
        )

        dialog = transcribe_segments([seg_a, seg_b], engine)
        assert dialog.turns == []

    def test_unknown_speaker_passes_through(self) -> None:
        seg_a = _seg(0.0, 1.0, speaker="unknown")
        seg_b = _seg(1.1, 2.0, speaker="unknown")
        engine = StubEngine(
            responses={
                id(seg_a): _raw(seg_a, "uncertain"),
                id(seg_b): _raw(seg_b, "speaker"),
            }
        )

        dialog = transcribe_segments([seg_a, seg_b], engine)
        assert len(dialog.turns) == 1
        assert dialog.turns[0].speaker == "unknown"

    def test_max_gap_seconds_override_propagates_to_merge(self) -> None:
        seg_a = _seg(0.0, 1.0, speaker="user")
        seg_b = _seg(1.8, 2.5, speaker="user")  # 0.8s gap
        engine = StubEngine(
            responses={
                id(seg_a): _raw(seg_a, "first"),
                id(seg_b): _raw(seg_b, "second"),
            }
        )

        # Tight override (0.5s) splits what the default (2.0s) would merge.
        dialog = transcribe_segments([seg_a, seg_b], engine, max_gap_seconds=0.5)
        assert [t.text for t in dialog.turns] == ["first", "second"]

    def test_low_confidence_segment_kept_with_confidence_flag(self) -> None:
        seg = _seg(0.0, 1.0, speaker="user")
        engine = StubEngine(
            responses={
                id(seg): _raw(seg, "mumble", avg_logprob=-2.5, no_speech_prob=0.1),
            }
        )

        dialog = transcribe_segments([seg], engine)
        assert len(dialog.turns) == 1
        # Low confidence surfaces through asr_confidence; segment is not dropped.
        assert dialog.turns[0].asr_confidence == pytest.approx(math.exp(-2.5))


# ── End-to-end with real Whisper (slow, optional) ────────────


@pytest.mark.slow
class TestPipelineEndToEnd:
    """End-to-end smoke through the full pipeline: real WhisperEngine,
    real soundfile-loaded audio, real assembly.

    Skipped by default; run with `pytest --runslow`. Requires:
      - faster-whisper installed
      - ~75 MB of `tiny` model weights on first run

    What this test catches that the stubbed pipeline tests do not:
      - The orchestrator + engine + assembly modules actually compose
      - WhisperEngine outputs are shaped the way RawTranscription expects
      - File loading via soundfile produces arrays Whisper accepts
      - The audio cache is exercised by the orchestrator's per-segment loop

    What this test does NOT validate (synthetic silence, not real speech):
      - Transcription accuracy on actual conversation audio
      - Hallucination behavior on noisy non-speech
      For those, exercise the pipeline manually with a real recording.
    """

    def test_pipeline_runs_end_to_end_on_silent_segments(
        self, tmp_path: Path, write_synthetic_wav
    ) -> None:
        # One conversation file, three segments slicing into it, with a
        # speaker change in the middle to exercise both the merge and
        # the boundary-split paths.
        audio_path = write_synthetic_wav(
            tmp_path / "conv.wav", duration_seconds=4.0
        )
        segments = [
            SpeechSegment(
                start_time=0.0,
                end_time=1.5,
                speaker_label="user",
                audio_path=audio_path,
            ),
            SpeechSegment(
                start_time=1.5,
                end_time=2.5,
                speaker_label="interlocutor",
                audio_path=audio_path,
            ),
            SpeechSegment(
                start_time=2.5,
                end_time=4.0,
                speaker_label="user",
                audio_path=audio_path,
            ),
        ]

        engine = WhisperEngine(
            model_size="tiny", compute_type="int8", device="cpu"
        )
        dialog = transcribe_segments(segments, engine)

        # Pure silence — assembly's filter_empty_or_silent should drop
        # every transcription, leaving an empty Dialog.
        assert dialog.turns == []
        # Engine cache proves the orchestrator actually reached the engine
        # and the engine actually loaded the file (not bypassed by an
        # accidental short-circuit somewhere in the pipeline).
        assert str(audio_path) in engine._audio_cache


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
