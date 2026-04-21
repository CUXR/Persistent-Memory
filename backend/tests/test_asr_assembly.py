"""Tests for backend/app/services/asr_assembly.py (issue #24, Phase 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schema.asr import RawTranscription, SpeechSegment
from app.services.asr_assembly import SILENCE_THRESHOLD, filter_empty_or_silent


def _seg(start: float = 0.0, end: float = 1.0, speaker: str = "user") -> SpeechSegment:
    return SpeechSegment(
        start_time=start,
        end_time=end,
        speaker_label=speaker,  # type: ignore[arg-type]
        audio_path=Path("/tmp/fake.wav"),
    )


def _raw(
    text: str,
    *,
    no_speech_prob: float = 0.0,
    avg_logprob: float = -0.1,
    start: float = 0.0,
    end: float = 1.0,
    speaker: str = "user",
) -> RawTranscription:
    return RawTranscription(
        segment=_seg(start, end, speaker),
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


class TestFilterEmptyOrSilent:
    def test_drops_empty_string(self) -> None:
        result = filter_empty_or_silent([_raw("")])
        assert result == []

    def test_drops_whitespace_only(self) -> None:
        result = filter_empty_or_silent([_raw("   \t\n  ")])
        assert result == []

    def test_drops_above_silence_threshold(self) -> None:
        # 0.61 is above the 0.6 threshold.
        result = filter_empty_or_silent([_raw("hello", no_speech_prob=0.61)])
        assert result == []

    def test_keeps_at_silence_threshold_boundary(self) -> None:
        # Exactly at the threshold should be kept (condition is strict >).
        result = filter_empty_or_silent(
            [_raw("hello", no_speech_prob=SILENCE_THRESHOLD)]
        )
        assert len(result) == 1

    def test_keeps_low_confidence_but_non_silent(self) -> None:
        # Very negative logprob (low confidence) should still be retained —
        # low confidence is a flag, not a drop reason per the locked decisions.
        result = filter_empty_or_silent(
            [_raw("mumble", avg_logprob=-2.5, no_speech_prob=0.1)]
        )
        assert len(result) == 1

    def test_preserves_order_and_identity(self) -> None:
        a = _raw("first", start=0.0, end=1.0)
        b = _raw("", start=1.0, end=2.0)  # dropped
        c = _raw("third", start=2.0, end=3.0, no_speech_prob=0.95)  # dropped
        d = _raw("fourth", start=3.0, end=4.0)
        result = filter_empty_or_silent([a, b, c, d])
        assert result == [a, d]

    def test_empty_input(self) -> None:
        assert filter_empty_or_silent([]) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
