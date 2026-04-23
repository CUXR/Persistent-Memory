"""faster-whisper wrapper for the ASR pipeline (issue #24).

This is the only module that imports faster-whisper. Everything else in the
service tree depends on the abstract shape (a callable that turns a
SpeechSegment into a RawTranscription), not on Whisper itself. That keeps
the orchestrator and assembly tests free of model-load cost and makes
swapping engines (Whisper API, distil-whisper, etc.) a one-file change.

Design notes:
  - Model load is lazy. Constructing WhisperEngine() is cheap; weights are
    loaded on first transcribe() call. This lets test modules import the
    class without paying ~460MB.
  - Audio is loaded into a numpy array once per file and cached on the
    engine instance (LRU, bounded). For a typical conversation ingestion —
    many segments, one audio_path — this avoids re-reading the WAV per
    segment.
  - faster-whisper internally chunks input into ~30s windows and may return
    multiple internal segments per transcribe() call. We concatenate text
    and aggregate confidence so each input SpeechSegment maps cleanly to
    one RawTranscription.
  - This wrapper does NOT filter, normalize, or merge. It returns whatever
    Whisper said, raw. All policy lives in asr_assembly.py and asr.py.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.app.schema.asr import RawTranscription, SpeechSegment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Input audio is mono 16 kHz per the contract from #23.
SAMPLE_RATE = 16_000


class WhisperEngine:
    """Thin wrapper around faster-whisper's WhisperModel.

    Implements the ASREngine protocol consumed by transcribe_segments() in
    asr.py. Construct once per process and reuse — model load is the
    expensive operation, transcription itself is cheap by comparison.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",          # "cuda" | "cpu" | "auto"
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._model: "WhisperModel | None" = None

    # ── Public API ───────────────────────────────────────────

    def transcribe(self, segment: SpeechSegment) -> RawTranscription:
        """Transcribe a single SpeechSegment.

        Loads the model on first call. Loads (and caches) the audio file
        on first reference to that path. Raises whatever faster-whisper
        raises — failure policy is the orchestrator's call.
        """
        ...

    def transcribe_many(
        self, segments: list[SpeechSegment]
    ) -> list[RawTranscription]:
        """Transcribe an ordered list of SpeechSegments.

        Convenience for the common case (many segments, one or few
        audio files). Order of output matches order of input. Per-segment
        failures propagate; this method does not swallow exceptions.
        """
        ...

    # ── Internal ─────────────────────────────────────────────

    def _ensure_model(self) -> "WhisperModel":
        """Lazy-load WhisperModel on first use, then return the cached
        instance on every subsequent call."""
        ...

    def _load_audio(self, path: Path) -> np.ndarray:
        """Read the WAV at `path` as a mono float32 array at SAMPLE_RATE.
        Cached per-instance so repeated segments on the same file pay the
        I/O cost once."""
        ...

    def _slice(
        self, audio: np.ndarray, start_time: float, end_time: float
    ) -> np.ndarray:
        """Return the sub-array for [start_time, end_time]. Sample offsets
        are int(t * SAMPLE_RATE)."""
        ...

    @staticmethod
    def _aggregate(
        whisper_segments: list,
    ) -> tuple[str, float, float]:
        """Collapse multiple internal Whisper segments (Whisper chunks
        long audio into ~30s pieces internally) into one (text,
        avg_logprob, no_speech_prob) tuple.

          - text: " ".join of segment texts, raw — no normalization here
          - avg_logprob: duration-weighted mean across segments
          - no_speech_prob: from the first segment (Whisper's silence
            decision is made up-front per chunk)

        For the typical case of one Whisper segment in == one out, this
        is a passthrough.
        """
        ...