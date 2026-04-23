"""Tests for backend/app/services/asr_assembly.py (issue #24, Phase 1)."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schema.asr import RawTranscription, SpeechSegment
from app.services.asr_assembly import (
    MAX_MERGE_GAP_SECONDS,
    SILENCE_THRESHOLD,
    filter_empty_or_silent,
    merge_adjacent_turns,
    normalize_text,
)


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


class TestNormalizeText:
    def test_collapses_whitespace_and_trims(self) -> None:
        result = normalize_text("  hello   world \n  from\tasr  ")
        assert result == "hello world from asr"

    def test_removes_space_before_punctuation(self) -> None:
        result = normalize_text("hello , world ! Are you there ?")
        assert result == "hello, world! Are you there?"

    def test_empty_or_whitespace_becomes_empty_string(self) -> None:
        assert normalize_text(" \t\n ") == ""


class TestMergeAdjacentTurns:
    def test_merges_same_speaker_within_default_gap(self) -> None:
        raw = [
            _raw("hello", start=0.0, end=1.0, speaker="user"),
            _raw("there", start=1.4, end=2.0, speaker="user"),
        ]
        dialog = merge_adjacent_turns(raw)

        assert len(dialog.turns) == 1
        turn = dialog.turns[0]
        assert turn.speaker == "user"
        assert turn.text == "hello there"
        assert turn.start_time == 0.0
        assert turn.end_time == 2.0
        assert turn.segment_count == 2
        expected = (math.exp(-0.1) + math.exp(-0.1)) / 2
        assert turn.asr_confidence == pytest.approx(expected)

    def test_merges_at_gap_boundary(self) -> None:
        raw = [
            _raw("first", start=0.0, end=1.0, speaker="interlocutor"),
            _raw(
                "second",
                start=1.0 + MAX_MERGE_GAP_SECONDS,
                end=4.0,
                speaker="interlocutor",
            ),
        ]
        dialog = merge_adjacent_turns(raw)
        assert len(dialog.turns) == 1
        assert dialog.turns[0].text == "first second"

    def test_does_not_merge_when_gap_exceeds_threshold(self) -> None:
        raw = [
            _raw("first", start=0.0, end=1.0, speaker="user"),
            _raw(
                "second",
                start=1.0 + MAX_MERGE_GAP_SECONDS + 0.01,
                end=4.0,
                speaker="user",
            ),
        ]
        dialog = merge_adjacent_turns(raw)
        assert [t.text for t in dialog.turns] == ["first", "second"]
        assert [t.segment_count for t in dialog.turns] == [1, 1]

    def test_does_not_merge_across_speakers_even_with_small_gap(self) -> None:
        raw = [
            _raw("hi", start=0.0, end=1.0, speaker="user"),
            _raw("hello", start=1.1, end=2.0, speaker="interlocutor"),
        ]
        dialog = merge_adjacent_turns(raw)
        assert len(dialog.turns) == 2
        assert [t.speaker for t in dialog.turns] == ["user", "interlocutor"]

    def test_preserves_unknown_speaker_label(self) -> None:
        raw = [
            _raw("uncertain", start=0.0, end=1.0, speaker="unknown"),
            _raw("speaker", start=1.2, end=2.0, speaker="unknown"),
        ]
        dialog = merge_adjacent_turns(raw)
        assert len(dialog.turns) == 1
        assert dialog.turns[0].speaker == "unknown"
        assert dialog.turns[0].segment_count == 2

    def test_applies_normalization_when_building_turn_text(self) -> None:
        raw = [
            _raw(" hello ", start=0.0, end=1.0, speaker="user"),
            _raw("world !", start=1.3, end=2.0, speaker="user"),
        ]
        dialog = merge_adjacent_turns(raw)
        assert dialog.turns[0].text == "hello world!"

    def test_uses_custom_gap_override(self) -> None:
        raw = [
            _raw("a", start=0.0, end=1.0, speaker="user"),
            _raw("b", start=1.8, end=2.5, speaker="user"),
        ]
        dialog = merge_adjacent_turns(raw, max_gap_seconds=0.5)
        assert [t.text for t in dialog.turns] == ["a", "b"]

    def test_empty_input_returns_empty_dialog(self) -> None:
        dialog = merge_adjacent_turns([])
        assert dialog.turns == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
