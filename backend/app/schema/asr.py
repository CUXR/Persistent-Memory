"""
backend/app/schema/asr.py
──────────────
Pydantic models for the ASR + dialog-assembly pipeline (issue #24).

Pipeline shape:
    list[SpeechSegment]                    (input — from VAD/diarization, #23)
        -> engine.transcribe per segment
        -> list[RawTranscription]          (internal — Whisper output per segment)
        -> filter / normalize / merge
        -> Dialog                          (output — for downstream ingestion)

Design notes:
  - Input audio is assumed to be mono 16 kHz on disk. SpeechSegment carries
    a path and time range; sample offsets are derived as int(t * 16000).
    This matches the contract from issue #23 (pyannote community-1 requires
    16 kHz mono and segments reference offsets into a single conversation
    audio file).
  - Speaker labels include "unknown" for cases where #23 cannot confidently
    attribute a segment. We transcribe these and pass them through untouched
    so downstream code can decide what to do.
  - asr_confidence is exp(avg_logprob) from faster-whisper, i.e. the average
    per-token probability bounded in [0, 1]. This keeps the field meaningful
    across engines and easy to threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Speaker attribution from the upstream diarization step. "unknown" is the
# explicit fallback when the diarizer cannot confidently label a segment.
SpeakerLabel = Literal["user", "interlocutor", "unknown"]


# ── Input ────────────────────────────────────────────────────


class SpeechSegment(BaseModel):
    """A single speaker-labeled audio segment produced by the VAD/diarization
    pipeline (issue #23).

    The segment is a slice into ``audio_path`` between ``start_time`` and
    ``end_time`` (seconds from the start of the file). Audio is assumed to be
    mono 16 kHz; sample offsets are ``int(t * 16000)``.
    """

    start_time: float = Field(
        ge=0.0,
        description="Segment start, seconds from the beginning of audio_path.",
    )
    end_time: float = Field(
        gt=0.0,
        description="Segment end, seconds from the beginning of audio_path.",
    )
    speaker_label: SpeakerLabel = Field(
        description="Speaker attribution from upstream diarization.",
    )
    audio_path: Path = Field(
        description="Path to the conversation audio file (mono 16 kHz WAV).",
    )

    @model_validator(mode="after")
    def _check_time_range(self) -> "SpeechSegment":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than "
                f"start_time ({self.start_time})"
            )
        return self


# ── Internal (engine output, pre-assembly) ───────────────────


class RawTranscription(BaseModel):
    """One Whisper transcription result for a single SpeechSegment, before
    filtering / normalization / merging. Internal to the service.
    """

    segment: SpeechSegment = Field(
        description="The input segment this transcription corresponds to.",
    )
    text: str = Field(
        description="Raw transcribed text from the ASR engine, unnormalized.",
    )
    avg_logprob: float = Field(
        description=(
            "Average per-token log probability from Whisper. Negative; closer "
            "to 0 is more confident. Used internally for confidence checks."
        ),
    )
    no_speech_prob: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Whisper's estimate that the segment contains no speech. "
            "Segments above the silence threshold are dropped during assembly."
        ),
    )


# ── Output ───────────────────────────────────────────────────


class DialogTurn(BaseModel):
    """One speaker turn in the assembled dialog. May represent one input
    segment or several adjacent same-speaker segments merged together.
    """

    speaker: SpeakerLabel = Field(
        description="Speaker attribution carried through from the input.",
    )
    text: str = Field(
        description="Normalized transcript text for this turn.",
    )
    start_time: float = Field(
        ge=0.0,
        description="Turn start in seconds (earliest constituent segment).",
    )
    end_time: float = Field(
        gt=0.0,
        description="Turn end in seconds (latest constituent segment).",
    )
    asr_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "exp(avg_logprob) averaged across constituent segments — the mean "
            "per-token probability in [0, 1]."
        ),
    )
    segment_count: int = Field(
        ge=1,
        description="Number of input SpeechSegments merged into this turn.",
    )

    @model_validator(mode="after")
    def _check_time_range(self) -> "DialogTurn":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than "
                f"start_time ({self.start_time})"
            )
        return self


class Dialog(BaseModel):
    """Final assembled output of the ASR pipeline. An ordered list of
    speaker-attributed turns ready for downstream ingestion.
    """

    turns: list[DialogTurn] = Field(
        default_factory=list,
        description="Dialog turns in chronological order.",
    )
