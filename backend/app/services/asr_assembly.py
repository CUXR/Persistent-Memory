"""
backend/app/services/asr_assembly.py
──────────────
Deterministic assembly helpers for the ASR pipeline (issue #24).

These functions operate on already-transcribed segments (``RawTranscription``)
and contain no Whisper/model code, so they can be unit-tested without loading
model weights. The orchestrator in ``asr.py`` composes them in order:

    filter_empty_or_silent -> merge_adjacent_turns (which calls normalize_text)
"""

from __future__ import annotations

import math
import re

from app.schema.asr import Dialog, DialogTurn, RawTranscription

# Whisper's no_speech_prob above this threshold => treat segment as silence
# and drop it. Matches the locked decision in ISSUES.md #24.
SILENCE_THRESHOLD = 0.6

# Consecutive same-speaker segments within this gap (seconds) are merged into
# one DialogTurn. Larger gaps start a new turn even for the same speaker.
MAX_MERGE_GAP_SECONDS = 2.0

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:])")


def filter_empty_or_silent(
    transcriptions: list[RawTranscription],
) -> list[RawTranscription]:
    """Drop transcriptions that are empty/whitespace or that Whisper flagged
    as non-speech (``no_speech_prob > SILENCE_THRESHOLD``).

    Low-confidence but non-silent segments are kept — they surface through
    ``asr_confidence`` on the eventual DialogTurn.
    """
    return [
        t
        for t in transcriptions
        if t.text.strip() and t.no_speech_prob <= SILENCE_THRESHOLD
    ]


def normalize_text(text: str) -> str:
    """Collapse whitespace and tighten punctuation spacing. Pure function."""
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", collapsed)


def merge_adjacent_turns(
    transcriptions: list[RawTranscription],
    max_gap_seconds: float = MAX_MERGE_GAP_SECONDS,
) -> Dialog:
    """Fuse consecutive same-speaker transcriptions (gap ≤ ``max_gap_seconds``)
    into ``DialogTurn`` entries. Input is assumed to be in chronological order.
    """
    turns: list[DialogTurn] = []
    group: list[RawTranscription] = []

    def flush() -> None:
        if not group:
            return
        turns.append(_group_to_turn(group))
        group.clear()

    for t in transcriptions:
        if not group:
            group.append(t)
            continue
        prev = group[-1]
        gap = t.segment.start_time - prev.segment.end_time
        same_speaker = t.segment.speaker_label == prev.segment.speaker_label
        if same_speaker and gap <= max_gap_seconds:
            group.append(t)
        else:
            flush()
            group.append(t)
    flush()

    return Dialog(turns=turns)


def _group_to_turn(group: list[RawTranscription]) -> DialogTurn:
    text = normalize_text(" ".join(t.text for t in group))
    mean_prob = sum(math.exp(t.avg_logprob) for t in group) / len(group)
    return DialogTurn(
        speaker=group[0].segment.speaker_label,
        text=text,
        start_time=group[0].segment.start_time,
        end_time=group[-1].segment.end_time,
        asr_confidence=mean_prob,
        segment_count=len(group),
    )
