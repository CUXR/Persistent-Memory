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
    engine instance (FIFO, bounded). For a typical conversation ingestion —
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
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.schema.asr import RawTranscription, SpeechSegment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger("app.services.asr_engine")

# Input audio is mono 16 kHz per the contract from #23.
SAMPLE_RATE = 16_000

# Default audio cache size — bounds the engine's RAM footprint when called
# across multiple conversation files. A 10-minute mono 16 kHz float32 file
# is ~38 MB, so 2 entries ≈ 76 MB worst case.
DEFAULT_AUDIO_CACHE_SIZE = 2


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
        audio_cache_size: int = DEFAULT_AUDIO_CACHE_SIZE,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._cache_size = audio_cache_size
        self._model: "WhisperModel | None" = None
        # OrderedDict gives FIFO eviction with O(1) ops.
        self._audio_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    # ── Public API ───────────────────────────────────────────

    def transcribe(self, segment: SpeechSegment) -> RawTranscription:
        """Transcribe a single SpeechSegment.

        Loads the model on first call. Loads (and caches) the audio file
        on first reference to that path. Raises whatever faster-whisper
        raises — failure policy is the orchestrator's call.
        """
        model = self._ensure_model()
        audio = self._load_audio(segment.audio_path)
        clip = self._slice(audio, segment.start_time, segment.end_time)

        # faster-whisper returns (segments_iterator, transcription_info).
        # The iterator must be consumed to drive actual decoding.
        whisper_segments_iter, _info = model.transcribe(
            clip,
            language=self._language,
            beam_size=self._beam_size,
        )
        whisper_segments = list(whisper_segments_iter)

        text, avg_logprob, no_speech_prob = self._aggregate(whisper_segments)
        return RawTranscription(
            segment=segment,
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        )

    def transcribe_many(
        self, segments: list[SpeechSegment]
    ) -> list[RawTranscription]:
        """Transcribe an ordered list of SpeechSegments.

        Convenience for the common case (many segments, one or few
        audio files). Order of output matches order of input. Per-segment
        failures propagate; this method does not swallow exceptions —
        that policy lives in asr.transcribe_segments().
        """
        return [self.transcribe(s) for s in segments]

    # ── Internal ─────────────────────────────────────────────

    def _ensure_model(self) -> "WhisperModel":
        """Lazy-load WhisperModel on first use, then return the cached
        instance on every subsequent call."""
        if self._model is None:
            from faster_whisper import WhisperModel  # local import: pay the
            # import cost only when transcription actually happens.

            logger.info(
                "Loading WhisperModel(size=%s, device=%s, compute_type=%s); "
                "first run downloads weights to the HuggingFace cache.",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _load_audio(self, path: Path) -> np.ndarray:
        """Read the WAV at ``path`` as a mono float32 array at SAMPLE_RATE.
        Cached per-instance so repeated segments on the same file pay the
        I/O cost once.
        """
        key = str(path)
        cached = self._audio_cache.get(key)
        if cached is not None:
            # Refresh recency by re-inserting at the end.
            self._audio_cache.move_to_end(key)
            return cached

        import soundfile as sf  # local import keeps module import cheap.

        samples, sample_rate = sf.read(key, dtype="float32")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Audio file {path} has sample rate {sample_rate}, "
                f"expected {SAMPLE_RATE} Hz (mono 16 kHz, per #23 contract)."
            )
        if samples.ndim > 1:
            # Defensive: contract is mono, but if a stereo file slips through
            # we collapse to mono by averaging channels rather than failing.
            logger.warning(
                "Audio file %s has %d channels; collapsing to mono.",
                path,
                samples.shape[1],
            )
            samples = samples.mean(axis=1).astype(np.float32)

        self._audio_cache[key] = samples
        while len(self._audio_cache) > self._cache_size:
            self._audio_cache.popitem(last=False)  # FIFO eviction
        return samples

    @staticmethod
    def _slice(
        audio: np.ndarray, start_time: float, end_time: float
    ) -> np.ndarray:
        """Return the sub-array for [start_time, end_time]. Sample offsets
        are int(t * SAMPLE_RATE)."""
        start_sample = int(start_time * SAMPLE_RATE)
        end_sample = int(end_time * SAMPLE_RATE)
        return audio[start_sample:end_sample]

    @staticmethod
    def _aggregate(
        whisper_segments: list[Any],
    ) -> tuple[str, float, float]:
        """Collapse multiple internal Whisper segments into one
        (text, avg_logprob, no_speech_prob) tuple.

          - text: " ".join of segment texts, raw — no normalization here.
          - avg_logprob: duration-weighted mean across segments. A long
            confident chunk shouldn't get pulled down by a 0.2s mumbled tail.
          - no_speech_prob: from the first segment. Whisper's silence
            decision is made up-front per ~30s chunk; the first chunk's
            value is what tells us "is this even speech."

        Returns ("", 0.0, 1.0) when ``whisper_segments`` is empty — Whisper
        produces no segments when it decides the input is pure silence. That
        triple gets dropped by ``filter_empty_or_silent`` downstream.
        """
        if not whisper_segments:
            return "", 0.0, 1.0

        text = " ".join(s.text for s in whisper_segments)
        no_speech_prob = whisper_segments[0].no_speech_prob

        durations = [s.end - s.start for s in whisper_segments]
        total_duration = sum(durations)
        if total_duration > 0:
            avg_logprob = (
                sum(s.avg_logprob * d for s, d in zip(whisper_segments, durations))
                / total_duration
            )
        else:
            # All zero-duration segments — fall back to simple mean.
            avg_logprob = sum(s.avg_logprob for s in whisper_segments) / len(
                whisper_segments
            )

        return text, avg_logprob, no_speech_prob
