"""Pytest configuration and shared fixtures.

Environment variables are set here at the very top — before any app module is
imported — so that pydantic-settings can validate required fields during
collection.  These values are test-only stubs; the MemoryStore tests always
pass their own ``db_url`` directly, and LLM calls are fully mocked.
"""

from pathlib import Path
import os

import numpy as np
import pytest
import soundfile as sf

# Provide the required settings fields so pydantic-settings validates cleanly.
# Tests always override these at the MemoryStore/LLMClient constructor level,
# so these stubs are never actually used to open a connection or make API calls.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-tests")


# ── @pytest.mark.slow opt-in ─────────────────────────────────
#
# Tests marked `@pytest.mark.slow` (e.g. real Whisper integration in
# test_asr_engine.py) are skipped by default. Pass `--runslow` to include
# them. This is the canonical pytest recipe from the docs.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (load real ML models, etc.)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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
