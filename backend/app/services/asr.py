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
are logged and the failing segment is skipped — one bad segment does not
abort transcription of the rest of the conversation.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.schema.asr import Dialog, RawTranscription, SpeechSegment
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
        segments: Chronologically ordered speaker-labeled audio segments
            from the VAD/diarization pipeline (issue #23). Order is
            preserved through the pipeline; window-boundary stitching is
            assumed to have already happened upstream.
        engine: Any object implementing the ``ASREngine`` protocol.
        max_gap_seconds: Override the default same-speaker merge gap.

    Returns:
        ``Dialog`` containing zero or more ``DialogTurn`` entries. Empty
        input, all-silent input, and all-failed input each yield
        ``Dialog(turns=[])`` rather than raising.
    """
    if not segments:
        return Dialog(turns=[])

    raw: list[RawTranscription] = []
    for segment in segments:
        try:
            raw.append(engine.transcribe(segment))
        except Exception:  # noqa: BLE001 — policy is "log and skip"
            logger.exception(
                "ASR engine failed on segment [%.3f, %.3f] in %s; skipping.",
                segment.start_time,
                segment.end_time,
                segment.audio_path,
            )

    surviving = filter_empty_or_silent(raw)
    return merge_adjacent_turns(surviving, max_gap_seconds=max_gap_seconds)
