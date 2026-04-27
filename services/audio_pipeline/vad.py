"""Silero VAD wrapper for lightweight speech / no-speech gating.

The Silero model is loaded lazily on first use via ``torch.hub`` so the module
can be imported freely without triggering a download.  Subsequent calls reuse
the cached model.

Usage::

    vad = SileroVAD(threshold=0.5)
    if vad.contains_speech(audio_chunk):
        # hand off to diarization
        ...
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .segment import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Silero hub coordinates.  Force-reload only when you need a model update.
_SILERO_REPO = "snakers4/silero-vad"
_SILERO_MODEL = "silero_vad"


class SileroVAD:
    """Thin wrapper around the Silero VAD model.

    Args:
        threshold:      Speech probability threshold in [0, 1].
                        Higher values → more selective (fewer false positives).
        force_reload:   Pass ``True`` to re-download the model on next use.
    """

    def __init__(self, threshold: float = 0.5, force_reload: bool = False) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold
        self._force_reload = force_reload
        self._model = None
        self._get_speech_timestamps = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch  # deferred so import is optional at module level

        logger.info("Loading Silero VAD model from torch.hub …")
        model, utils = torch.hub.load(
            repo_or_dir=_SILERO_REPO,
            model=_SILERO_MODEL,
            force_reload=self._force_reload,
            trust_repo=True,
        )
        (
            get_speech_timestamps,
            _save_audio,
            _read_audio,
            _VADIterator,
            _collect_chunks,
        ) = utils

        self._model = model
        self._get_speech_timestamps = get_speech_timestamps
        logger.info("Silero VAD model loaded.")

    # ------------------------------------------------------------------
    def contains_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> bool:
        """Return ``True`` if at least one speech region is detected.

        Args:
            audio:          1-D float32 PCM array.
            sample_rate:    Audio sample rate in Hz (default 16 000).

        Returns:
            ``True`` when Silero detects any speech above *threshold*.
        """
        import torch

        self._ensure_loaded()
        tensor = torch.from_numpy(audio.astype(np.float32))
        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
        )
        has_speech = len(timestamps) > 0
        logger.debug(
            "VAD: %d speech region(s) detected in %.2fs window",
            len(timestamps),
            len(audio) / sample_rate,
        )
        return has_speech

    # ------------------------------------------------------------------
    def speech_ratio(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> float:
        """Return the fraction of the window that contains speech, in [0, 1].

        Useful for downstream logging and threshold experiments.

        Args:
            audio:          1-D float32 PCM array.
            sample_rate:    Audio sample rate in Hz.

        Returns:
            Float in [0, 1]; 0.0 when no speech is detected.
        """
        import torch

        self._ensure_loaded()
        total_samples = len(audio)
        if total_samples == 0:
            return 0.0

        tensor = torch.from_numpy(audio.astype(np.float32))
        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
        )

        speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        return min(1.0, speech_samples / total_samples)

    # ------------------------------------------------------------------
    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[dict]:
        """Return raw Silero timestamp dicts ``{start: int, end: int}`` in samples.

        Args:
            audio:          1-D float32 PCM array.
            sample_rate:    Audio sample rate in Hz.

        Returns:
            List of ``{"start": int, "end": int}`` sample-index dicts.
        """
        import torch

        self._ensure_loaded()
        tensor = torch.from_numpy(audio.astype(np.float32))
        return self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
        )
