"""Lazy Silero VAD using the packaged model rather than mutable torch.hub code."""
import numpy as np

from .segment import SAMPLE_RATE, validate_audio


class SileroVAD:
    def __init__(self, threshold: float = 0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold
        self._model = None
        self._timestamps = None

    def get_speech_timestamps(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[dict]:
        audio = validate_audio(audio, sample_rate)
        if not len(audio):
            return []
        import torch
        if self._model is None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._model = load_silero_vad()
            self._timestamps = get_speech_timestamps
        return self._timestamps(
            torch.from_numpy(audio), self._model, sampling_rate=sample_rate, threshold=self.threshold,
        )

    def contains_speech(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
        return bool(self.get_speech_timestamps(audio, sample_rate))
