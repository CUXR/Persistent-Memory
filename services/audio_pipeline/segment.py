"""Output contract for the audio segmentation pipeline.

Every downstream consumer (ASR, ingestion) works exclusively against
``AudioSegment`` objects produced by this module.  Nothing upstream should
leak raw pyannote or torch types through the public surface.

Sample-index convention
-----------------------
Audio is assumed to be **mono PCM float32 at SAMPLE_RATE Hz** throughout the
pipeline.  The ``start_sample`` / ``end_sample`` fields are integer offsets
into the flat 1-D numpy array that represents the full conversation recording.
To recover the raw waveform slice for a segment:

    waveform_slice = conversation_audio[seg.start_sample : seg.end_sample]

This is equivalent to ``seg.audio_slice(conversation_audio)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Expected sample rate throughout the pipeline (Hz).
SAMPLE_RATE: int = 16_000

# Rolling VAD window length in seconds.
VAD_WINDOW_SECONDS: float = 5.0

# Maximum conversation accumulation buffer length in seconds (~10 min).
MAX_CONVERSATION_SECONDS: float = 600.0

# Number of consecutive silent VAD windows that signals end-of-conversation.
SILENCE_WINDOWS_EOC: int = 2

# Samples per VAD window.
VAD_WINDOW_SAMPLES: int = int(VAD_WINDOW_SECONDS * SAMPLE_RATE)

# Samples per conversation window ceiling.
MAX_CONVERSATION_SAMPLES: int = int(MAX_CONVERSATION_SECONDS * SAMPLE_RATE)

SpeakerLabel = Literal["user", "interlocutor", "uncertain"]


@dataclass
class AudioSegment:
    """One speaker-labeled speech segment within a conversation.

    Attributes:
        start_time:     Seconds from the start of the conversation recording.
        end_time:       Seconds from the start of the conversation recording.
        speaker_label:  ``"user"`` (the wearer), ``"interlocutor"`` (the other
                        party), or ``"uncertain"`` when similarity scoring is
                        inconclusive.
        start_sample:   Integer sample index into the conversation audio array.
                        Always ``int(start_time * SAMPLE_RATE)``.
        end_sample:     Integer sample index (exclusive upper bound).
                        Always ``int(end_time * SAMPLE_RATE)``.
        confidence:     Cosine-similarity score used for attribution, in [0, 1].
                        ``0.0`` for ``"uncertain"`` segments.
        speaker_id:     Raw diarization speaker label (e.g. ``"SPEAKER_00"``).
                        Useful for debugging; not meaningful across conversations.
    """

    start_time: float
    end_time: float
    speaker_label: SpeakerLabel
    start_sample: int
    end_sample: int
    confidence: float
    speaker_id: str = ""

    # ------------------------------------------------------------------
    def audio_slice(self, conversation_audio: np.ndarray) -> np.ndarray:
        """Return the raw waveform for this segment.

        Args:
            conversation_audio: 1-D float32 array for the full conversation
                at :data:`SAMPLE_RATE` Hz.

        Returns:
            A numpy view (or copy if out-of-bounds) of the segment audio.
        """
        return conversation_audio[self.start_sample : self.end_sample]

    # ------------------------------------------------------------------
    @classmethod
    def from_times(
        cls,
        start_time: float,
        end_time: float,
        speaker_label: SpeakerLabel,
        confidence: float,
        speaker_id: str = "",
    ) -> "AudioSegment":
        """Convenience constructor that derives sample indices from timestamps."""
        return cls(
            start_time=start_time,
            end_time=end_time,
            speaker_label=speaker_label,
            start_sample=int(start_time * SAMPLE_RATE),
            end_sample=int(end_time * SAMPLE_RATE),
            confidence=confidence,
            speaker_id=speaker_id,
        )

    def __repr__(self) -> str:
        return (
            f"AudioSegment({self.start_time:.2f}s–{self.end_time:.2f}s "
            f"[{self.speaker_label}] conf={self.confidence:.3f})"
        )
