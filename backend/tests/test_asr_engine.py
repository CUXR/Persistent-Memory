"""Tests for backend/app/services/asr_engine.py.

Most tests in this module exercise the engine WITHOUT loading Whisper:
the static aggregation logic, audio loading, slicing, lazy-load behavior,
and the FIFO audio cache.

The single `@pytest.mark.slow` test at the bottom actually constructs a
real WhisperModel and transcribes synthetic audio. It is skipped by default
(`pytest` ignores `-m slow` unless requested) and the first run will
download ~460 MB of model weights to the HuggingFace cache.

Audio fixtures are generated on the fly with soundfile + numpy rather than
checked into the repo. Mono 16 kHz float32 WAVs by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schema.asr import SpeechSegment
from app.services.asr_engine import (
    DEFAULT_AUDIO_CACHE_SIZE,
    SAMPLE_RATE,
    WhisperEngine,
)


# ── Helpers ──────────────────────────────────────────────────


@dataclass
class _FakeWhisperSegment:
    """Stand-in for faster_whisper.Segment with the attributes we use."""

    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float


def _write_synthetic_wav(
    path: Path,
    duration_seconds: float = 1.0,
    *,
    frequency: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    """Write a synthetic mono float32 WAV. Silence by default; a sine tone
    when `frequency` is given. Returns the path for chaining.
    """
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


# ── Static logic: _aggregate ─────────────────────────────────


class TestAggregate:
    def test_empty_returns_silence_signal(self) -> None:
        text, avg_logprob, no_speech = WhisperEngine._aggregate([])
        assert text == ""
        assert no_speech == 1.0  # forces filter_empty_or_silent to drop it
        assert avg_logprob == 0.0

    def test_single_segment_passthrough(self) -> None:
        seg = _FakeWhisperSegment("hello", 0.0, 1.0, -0.3, 0.05)
        text, avg_logprob, no_speech = WhisperEngine._aggregate([seg])
        assert text == "hello"
        assert avg_logprob == pytest.approx(-0.3)
        assert no_speech == pytest.approx(0.05)

    def test_text_join_uses_single_space(self) -> None:
        segs = [
            _FakeWhisperSegment("first", 0.0, 1.0, -0.1, 0.0),
            _FakeWhisperSegment("second", 1.0, 2.0, -0.1, 0.0),
        ]
        text, _, _ = WhisperEngine._aggregate(segs)
        assert text == "first second"

    def test_avg_logprob_is_duration_weighted(self) -> None:
        # 9 seconds at -0.1, 1 second at -3.0. Simple mean would give ~-1.55,
        # duration-weighted should give -0.39.
        segs = [
            _FakeWhisperSegment("clean", 0.0, 9.0, -0.1, 0.05),
            _FakeWhisperSegment("noise", 9.0, 10.0, -3.0, 0.05),
        ]
        _, avg_logprob, _ = WhisperEngine._aggregate(segs)
        expected = (-0.1 * 9.0 + -3.0 * 1.0) / 10.0
        assert avg_logprob == pytest.approx(expected)
        # Sanity: this is much closer to -0.1 than to a simple mean.
        assert avg_logprob > -1.0

    def test_no_speech_prob_taken_from_first_segment(self) -> None:
        segs = [
            _FakeWhisperSegment("a", 0.0, 1.0, -0.1, 0.10),
            _FakeWhisperSegment("b", 1.0, 2.0, -0.1, 0.95),  # ignored
        ]
        _, _, no_speech = WhisperEngine._aggregate(segs)
        assert no_speech == pytest.approx(0.10)

    def test_zero_duration_segments_fall_back_to_simple_mean(self) -> None:
        # All segments have start == end. Avoid divide-by-zero by simple mean.
        segs = [
            _FakeWhisperSegment("a", 1.0, 1.0, -0.2, 0.0),
            _FakeWhisperSegment("b", 1.0, 1.0, -0.4, 0.0),
        ]
        _, avg_logprob, _ = WhisperEngine._aggregate(segs)
        assert avg_logprob == pytest.approx(-0.3)


# ── Static logic: _slice ─────────────────────────────────────


class TestSlice:
    def test_extracts_correct_sample_range(self) -> None:
        # 4 seconds of ascending samples; slice [1.0, 2.5).
        audio = np.arange(4 * SAMPLE_RATE, dtype=np.float32)
        clip = WhisperEngine._slice(audio, 1.0, 2.5)
        assert clip.shape == (int(1.5 * SAMPLE_RATE),)
        assert clip[0] == SAMPLE_RATE  # sample at t=1.0
        assert clip[-1] == int(2.5 * SAMPLE_RATE) - 1

    def test_zero_length_slice(self) -> None:
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        clip = WhisperEngine._slice(audio, 0.5, 0.5)
        assert clip.shape == (0,)


# ── Audio loading + caching ──────────────────────────────────


class TestLoadAudio:
    def test_loads_mono_16khz_wav(self, tmp_path: Path) -> None:
        path = _write_synthetic_wav(tmp_path / "speech.wav", duration_seconds=2.0)
        engine = WhisperEngine()
        samples = engine._load_audio(path)
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert samples.shape == (int(2.0 * SAMPLE_RATE),)

    def test_rejects_wrong_sample_rate(self, tmp_path: Path) -> None:
        path = _write_synthetic_wav(
            tmp_path / "wrong_sr.wav", duration_seconds=0.5, sample_rate=8000
        )
        engine = WhisperEngine()
        with pytest.raises(ValueError, match="sample rate"):
            engine._load_audio(path)

    def test_collapses_stereo_to_mono(self, tmp_path: Path) -> None:
        path = _write_synthetic_wav(
            tmp_path / "stereo.wav", duration_seconds=0.5, channels=2
        )
        engine = WhisperEngine()
        samples = engine._load_audio(path)
        assert samples.ndim == 1
        assert samples.shape == (int(0.5 * SAMPLE_RATE),)

    def test_caches_repeat_loads(self, tmp_path: Path) -> None:
        path = _write_synthetic_wav(tmp_path / "cached.wav", duration_seconds=0.5)
        engine = WhisperEngine()
        first = engine._load_audio(path)
        second = engine._load_audio(path)
        # Cache returns the same array object, not a fresh load.
        assert first is second

    def test_evicts_oldest_when_cache_full(self, tmp_path: Path) -> None:
        engine = WhisperEngine(audio_cache_size=2)
        a = _write_synthetic_wav(tmp_path / "a.wav", duration_seconds=0.5)
        b = _write_synthetic_wav(tmp_path / "b.wav", duration_seconds=0.5)
        c = _write_synthetic_wav(tmp_path / "c.wav", duration_seconds=0.5)
        engine._load_audio(a)
        engine._load_audio(b)
        # Loading a third file evicts the oldest (a).
        engine._load_audio(c)
        assert str(a) not in engine._audio_cache
        assert str(b) in engine._audio_cache
        assert str(c) in engine._audio_cache

    def test_recently_used_entries_survive_eviction(self, tmp_path: Path) -> None:
        engine = WhisperEngine(audio_cache_size=2)
        a = _write_synthetic_wav(tmp_path / "a.wav", duration_seconds=0.5)
        b = _write_synthetic_wav(tmp_path / "b.wav", duration_seconds=0.5)
        c = _write_synthetic_wav(tmp_path / "c.wav", duration_seconds=0.5)
        engine._load_audio(a)
        engine._load_audio(b)
        engine._load_audio(a)  # touches a -> b is now oldest
        engine._load_audio(c)  # should evict b, not a
        assert str(a) in engine._audio_cache
        assert str(b) not in engine._audio_cache
        assert str(c) in engine._audio_cache


# ── Lazy model load ──────────────────────────────────────────


class TestLazyModel:
    def test_construction_does_not_load_model(self) -> None:
        engine = WhisperEngine()
        assert engine._model is None

    def test_default_audio_cache_size_is_documented_value(self) -> None:
        engine = WhisperEngine()
        assert engine._cache_size == DEFAULT_AUDIO_CACHE_SIZE


# ── Real Whisper integration (slow, optional) ────────────────


@pytest.mark.slow
class TestWhisperIntegration:
    """Loads a real WhisperModel and transcribes synthetic audio.

    Skipped by default; run with `pytest -m slow`. Requires:
      - faster-whisper installed (in requirements.txt)
      - ~460 MB of model weights on first run (cached afterwards)
      - working internet on first run only
    """

    def test_silence_yields_high_no_speech_prob_or_empty_text(
        self, tmp_path: Path
    ) -> None:
        # Pure silence — Whisper should either return no segments at all
        # (which our _aggregate maps to no_speech_prob=1.0) or flag the
        # result as non-speech. Either outcome is correct; the assembly
        # layer drops both.
        audio_path = _write_synthetic_wav(
            tmp_path / "silence.wav", duration_seconds=2.0
        )
        segment = SpeechSegment(
            start_time=0.0,
            end_time=2.0,
            speaker_label="user",
            audio_path=audio_path,
        )

        engine = WhisperEngine(model_size="tiny", compute_type="int8", device="cpu")
        result = engine.transcribe(segment)

        # Either Whisper called silence (high no_speech_prob) OR returned
        # nothing transcribable (empty text). We accept either.
        assert result.text.strip() == "" or result.no_speech_prob > 0.5
        # Segment passes through unchanged.
        assert result.segment is segment


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
