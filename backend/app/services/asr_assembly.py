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
# and drop it.
SILENCE_THRESHOLD = 0.6

# Consecutive same-speaker segments within this gap (seconds) are merged into
# one DialogTurn. Larger gaps start a new turn even for the same speaker.
MAX_MERGE_GAP_SECONDS = 2.0

# Minimum length of a same-token run before we treat it as a Whisper
# repetition hallucination and collapse it. Set to 3 deliberately so that
# legitimate doublings ("had had", "that that") survive untouched while
# pathological loops ("the the the the") get cleaned up.
MIN_REPETITION_RUN = 3

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


def collapse_repetitions(
    text: str, min_runs: int = MIN_REPETITION_RUN
) -> str:
    """Collapse runs of ``min_runs`` or more identical adjacent tokens to one.

    Targets a specific Whisper failure mode where the decoder loops on a
    short fragment ("the the the the", "thank you thank you thank you") and
    pollutes the transcript. Conservative by design:

      - Single-token runs only — phrase-level dedup ("I think I think") is
        more dangerous (legitimate phrase repetition exists) and not handled.
      - Exact, case-sensitive token comparison after whitespace split.
        Punctuation-attached tokens do NOT match the bare form ("hello,"
        does not collapse against "hello").
      - Below the ``min_runs`` threshold (default 3), repetition is preserved
        so legitimate English like "had had" or "that that" survives.

    Pure function. Idempotent: collapsing twice gives the same result.
    """
    if min_runs < 2:
        raise ValueError(f"min_runs must be >= 2, got {min_runs}")

    tokens = text.split()
    if len(tokens) < min_runs:
        return text

    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        # Find length of run starting at i where every token matches tokens[i].
        j = i + 1
        while j < n and tokens[j] == tokens[i]:
            j += 1
        run_len = j - i
        if run_len >= min_runs:
            out.append(tokens[i])  # collapse the entire run to one token
        else:
            out.extend(tokens[i:j])  # keep run as-is
        i = j

    return " ".join(out)


def normalize_text(text: str) -> str:
    """Collapse whitespace, tighten punctuation, and de-dupe Whisper
    repetition loops. Pure function.

    Pipeline:
      1. Collapse runs of whitespace, strip leading/trailing.
      2. Remove space before ``,.!?;:``.
      3. Collapse Whisper repetition runs (see ``collapse_repetitions``).
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    tightened = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", collapsed)
    return collapse_repetitions(tightened)


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
