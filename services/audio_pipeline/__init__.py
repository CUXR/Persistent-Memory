"""Audio segmentation and speaker attribution pipeline.

Public API
----------
- :class:`AudioIngestionPipeline`  — main entry point; feed audio chunks, receive segments.
- :class:`ConversationState`       — observable state enum.
- :class:`AudioSegment`            — output contract (downstream ASR contract).
- :class:`SileroVAD`               — lightweight VAD wrapper.
- :class:`DiarizationEngine`       — pyannote diarization wrapper.
- :class:`SpeakerAttributor`       — cosine-similarity attribution.
- :class:`AttributionConfig`       — threshold tuning.
- :data:`SAMPLE_RATE`              — expected sample rate (16 000 Hz).
"""

from .segment import (
    SAMPLE_RATE,
    VAD_WINDOW_SECONDS,
    MAX_CONVERSATION_SECONDS,
    AudioSegment,
    SpeakerLabel,
)
from .vad import SileroVAD
from .diarization import DiarizationEngine, DiarizedTurn
from .speaker_attribution import SpeakerAttributor, AttributionConfig
from .ingestion import AudioIngestionPipeline, ConversationState

__all__ = [
    "SAMPLE_RATE",
    "VAD_WINDOW_SECONDS",
    "MAX_CONVERSATION_SECONDS",
    "AudioSegment",
    "SpeakerLabel",
    "SileroVAD",
    "DiarizationEngine",
    "DiarizedTurn",
    "SpeakerAttributor",
    "AttributionConfig",
    "AudioIngestionPipeline",
    "ConversationState",
]
