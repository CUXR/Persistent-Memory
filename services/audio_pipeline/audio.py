"""Normalize file input once; capture and model stages only handle mono 16 kHz."""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from .segment import SAMPLE_RATE, validate_audio


def load_audio(path: Path, *, max_seconds: float | None = None) -> np.ndarray:
    with sf.SoundFile(path) as source:
        if max_seconds is not None and len(source) / source.samplerate > max_seconds:
            raise ValueError(f"Recording exceeds {max_seconds:g} seconds")
        sample_rate = source.samplerate
        audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    return validate_audio(audio)
