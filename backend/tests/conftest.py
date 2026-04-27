"""Pytest configuration and shared fixtures.

Environment variables are set here at the very top — before any app module is
imported — so that pydantic-settings can validate required fields during
collection.  These values are test-only stubs; the MemoryStore tests always
pass their own ``db_url`` directly, and LLM calls are fully mocked.
"""

from pathlib import Path
import os

import pytest

# Provide the required settings fields so pydantic-settings validates cleanly.
# Tests always override these at the MemoryStore/LLMClient constructor level,
# so these stubs are never actually used to open a connection or make API calls.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-tests")


# `--runslow` plumbing lives in the repo-root conftest.py because
# `pytest_addoption` must be registered above the rootdir.


# ── Shared audio fixture for ASR tests ───────────────────────


@pytest.fixture
def write_synthetic_wav():
    """Return a callable that writes a mono float32 WAV at the ASR sample rate.

    Used by ASR engine and pipeline tests to avoid checking binary fixtures
    into the repo. Defaults to silence; pass `frequency` for a sine tone.
    Pass `channels=2` to test the stereo-collapse path.
    """

    def _writer(
        path: Path,
        duration_seconds: float = 1.0,
        *,
        frequency: float | None = None,
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> Path:
        import numpy as np
        import soundfile as sf

        n_samples = int(duration_seconds * sample_rate)
        if frequency is None:
            samples = np.zeros(n_samples, dtype=np.float32)
        else:
            t = np.arange(n_samples, dtype=np.float32) / sample_rate
            samples = (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        if channels > 1:
            samples = np.stack([samples] * channels, axis=1)
        sf.write(str(path), samples, sample_rate)
        return path

    return _writer
