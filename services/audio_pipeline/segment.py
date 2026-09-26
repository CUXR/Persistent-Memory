"""Shared recording, speaker, and segment contracts used through ASR."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAMPLE_RATE = 16_000
VOICE_MODEL = "pyannote/embedding"
VOICE_DIMENSION = 512
SpeakerLabel = Literal["user", "interlocutor", "unknown"]
AttributionMethod = Literal["voice_match", "conversation", "unknown"]


def validate_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Audio must be normalized to 16 kHz before capture")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Audio must be finite mono PCM")
    return audio


class VoiceProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str = Field(min_length=1, max_length=100)
    embedding: list[float] = Field(min_length=VOICE_DIMENSION, max_length=VOICE_DIMENSION)

    @field_validator("embedding")
    @classmethod
    def normalize(cls, value: list[float]) -> list[float]:
        vector = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-10:
            raise ValueError("Voice embedding must be finite and nonzero")
        return (vector / norm).tolist()


class DiscoveredVoice(BaseModel):
    """A new voice learned from this recording, not an independent identity match."""
    speaker_id: str
    profile: VoiceProfile


class Recording(BaseModel):
    """Immutable WAV reference. All segment times are relative to this file."""
    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)
    audio_path: Path
    started_at: datetime
    sample_rate: Literal[16000] = SAMPLE_RATE
    sample_count: int = Field(gt=0)

    @field_validator("started_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recording start must include a timezone")
        return value

    @property
    def duration(self) -> float:
        return self.sample_count / self.sample_rate


class SpeechSegment(BaseModel):
    """One diarized turn; identity and raw similarity survive transcription."""
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    audio_path: Path
    speaker_id: str = Field(min_length=1)
    speaker_label: SpeakerLabel
    attribution_method: AttributionMethod = "unknown"
    speaker_similarity: float | None = Field(default=None, ge=-1, le=1)
    person_id: UUID | None = None

    @model_validator(mode="after")
    def ordered(self) -> "SpeechSegment":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if self.speaker_label != "interlocutor" and self.person_id is not None:
            raise ValueError("Only an identified interlocutor can carry person_id")
        return self
