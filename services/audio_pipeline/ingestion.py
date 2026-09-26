"""Silero-gated capture. Finalize a durable WAV before running expensive models."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .segment import SAMPLE_RATE, Recording, validate_audio


class ConversationState(Enum):
    IDLE = auto()
    ACCUMULATING = auto()


class AudioIngestionPipeline:
    """One 16 kHz mono stream, with non-overlapping 10-second VAD checks.

    push_audio/flush return durable recordings, not ephemeral segment offsets.
    Capture and model processing are separate so callers can queue work.
    """

    def __init__(
        self, vad, recording_dir: Path, started_at: datetime, *,
        window_seconds: float = 10, silence_windows_eoc: int = 1,
        max_conversation_seconds: float = 600,
    ):
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("started_at must include a timezone")
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not np.isfinite(max_conversation_seconds) or max_conversation_seconds < window_seconds:
            raise ValueError("Conversation limit must be at least one VAD window")
        if silence_windows_eoc < 1:
            raise ValueError("silence_windows_eoc must be positive")
        self._vad = vad
        self._directory = Path(recording_dir).resolve()
        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._started_at = started_at
        self._window_samples = int(window_seconds * SAMPLE_RATE)
        if self._window_samples < 1:
            raise ValueError("VAD window is smaller than one sample")
        # Whole windows avoid truncating a window at the accumulation boundary.
        self._max_samples = int(max_conversation_seconds * SAMPLE_RATE)
        if self._max_samples % self._window_samples:
            raise ValueError("Conversation limit must be a multiple of the VAD window")
        self._silence_limit = silence_windows_eoc
        self._partial = np.empty(0, dtype=np.float32)
        self._samples_seen = 0
        self._clear_conversation()

    def _clear_conversation(self):
        self._buffer = []
        self._count = 0
        self._start_sample = 0
        self._silent_windows = 0
        self._recording_id = None
        self._needs_finalize = False
        self._state = ConversationState.IDLE

    @property
    def state(self):
        return self._state

    @property
    def conversation_duration_seconds(self):
        return self._count / SAMPLE_RATE

    def push_audio(self, chunk: np.ndarray) -> list[Recording]:
        chunk = validate_audio(chunk)
        self._partial = np.concatenate([self._partial, chunk])
        recordings = []
        if self._needs_finalize:
            recordings.append(self._finalize())
        while len(self._partial) >= self._window_samples:
            window = self._partial[:self._window_samples].copy()
            # Preserve the unconsumed input if VAD fails.
            has_speech = self._vad.contains_speech(window, SAMPLE_RATE)
            self._partial = self._partial[self._window_samples:]
            recordings.extend(self._consume(window, has_speech))
        return recordings

    def _consume(self, window, has_speech):
        start = self._samples_seen
        self._samples_seen += len(window)
        if self._state == ConversationState.IDLE and not has_speech:
            return []
        if self._state == ConversationState.IDLE:
            self._start_sample = start
            self._recording_id = uuid4()
            self._state = ConversationState.ACCUMULATING
        self._buffer.append(window)
        self._count += len(window)
        self._silent_windows = 0 if has_speech else self._silent_windows + 1
        if self._count >= self._max_samples or self._silent_windows >= self._silence_limit:
            return [self._finalize()]
        return []

    def flush(self) -> list[Recording]:
        # A previous VAD/write failure may have left full windows pending.
        recordings = self.push_audio(np.empty(0, dtype=np.float32))
        if len(self._partial):
            partial = self._partial.copy()
            has_speech = self._vad.contains_speech(partial, SAMPLE_RATE)
            self._partial = np.empty(0, dtype=np.float32)
            recordings.extend(self._consume(partial, has_speech))
        if self._state == ConversationState.ACCUMULATING:
            recordings.append(self._finalize())
        return recordings

    def reset(self, started_at: datetime) -> None:
        """Explicitly discard all pending samples and start a new stream clock."""
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("started_at must include a timezone")
        self._clear_conversation()
        self._partial = np.empty(0, dtype=np.float32)
        self._samples_seen = 0
        self._started_at = started_at

    def _finalize(self) -> Recording:
        self._needs_finalize = True
        recording = Recording(
            id=self._recording_id, audio_path=self._directory / f"{self._recording_id}.wav",
            started_at=self._started_at + timedelta(seconds=self._start_sample / SAMPLE_RATE),
            sample_count=self._count,
        )
        temporary = recording.audio_path.with_suffix(".wav.tmp")
        # FLOAT avoids additional quantization of the incoming float32 samples.
        with open(temporary, "wb") as handle:
            os.chmod(temporary, 0o600)
            sf.write(handle, np.concatenate(self._buffer), SAMPLE_RATE, format="WAV", subtype="FLOAT")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(recording.audio_path)
        manifest = recording.audio_path.with_suffix(".json")
        temporary_manifest = manifest.with_suffix(".json.tmp")
        with open(temporary_manifest, "w") as handle:
            os.chmod(temporary_manifest, 0o600)
            handle.write(recording.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temporary_manifest.replace(manifest)
        self._clear_conversation()
        return recording
