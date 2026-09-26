"""
backend/app/services/asr.py
──────────────
Public entry point for the ASR + dialog-assembly pipeline (issue #24).

Composes three steps over a list of speaker-labeled segments:

    engine.transcribe (per segment)
        -> filter_empty_or_silent
        -> merge_adjacent_turns
        -> Dialog

The engine is dependency-injected via the ``ASREngine`` Protocol so the
orchestrator can be tested with stubs and the real ``WhisperEngine`` stays
out of the assembly/orchestration test suite (no model load).

Failure policy lives here, not in the engine. Per-segment ASR exceptions
are retained on Dialog.errors, alongside any successful turns. The memory
coordinator will not extract facts from an incomplete transcript.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.schema.asr import Dialog, RawTranscription, SpeechSegment, TranscriptionFailure
from app.services.asr_assembly import (
    MAX_MERGE_GAP_SECONDS,
    filter_empty_or_silent,
    merge_adjacent_turns,
)

logger = logging.getLogger("app.services.asr")


class ASREngine(Protocol):
    """Minimal contract for an ASR engine. Anything callable with the right
    shape satisfies it — no inheritance required.

    The real implementation is :class:`~app.services.asr_engine.WhisperEngine`.
    Tests inject lightweight stubs that return canned ``RawTranscription``
    objects.
    """

    def transcribe(self, segment: SpeechSegment) -> RawTranscription:
        ...


def transcribe_segments(
    segments: list[SpeechSegment],
    engine: ASREngine,
    *,
    max_gap_seconds: float = MAX_MERGE_GAP_SECONDS,
) -> Dialog:
    """Transcribe and assemble speaker-labeled segments into a ``Dialog``.

    Args:
        segments: Speaker-labeled segments from one finalized recording,
            sorted here by recording-relative start time.
        engine: Any object implementing the ``ASREngine`` protocol.
        max_gap_seconds: Override the default same-speaker merge gap.

    Returns:
        ``Dialog`` containing zero or more ``DialogTurn`` entries. Empty
        input and all-silent input yield an empty dialog. Failed segments
        are reported in errors and can be retried from the retained WAV.
    """
    if not segments:
        return Dialog(turns=[])

    raw: list[RawTranscription] = []
    failures: list[TranscriptionFailure] = []
    for segment in sorted(segments, key=lambda segment: segment.start_time):
        try:
            raw.append(engine.transcribe(segment))
        except Exception as exc:
            failures.append(TranscriptionFailure(segment=segment, error=str(exc)))
            logger.exception(
                "ASR engine failed on segment [%.3f, %.3f] in %s; error retained.",
                segment.start_time,
                segment.end_time,
                segment.audio_path,
            )

    surviving = filter_empty_or_silent(raw)
    dialog = merge_adjacent_turns(surviving, max_gap_seconds=max_gap_seconds)
    dialog.errors = failures
    return dialog
