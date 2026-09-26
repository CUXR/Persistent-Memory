from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from audio_pipeline.segment import DiscoveredVoice, Recording
from .asr import Dialog
from .ingestion import IngestionResult


class AudioJobOut(BaseModel):
    id: UUID
    person_id: UUID
    recording: Recording
    status: Literal["pending", "processing", "failed", "needs_review", "no_speech", "complete"]
    stage: str
    error: str | None = None
    attempt_id: UUID | None = None
    dialog: Dialog | None = None
    voice_discovery: DiscoveredVoice | None = None
    transcript: str = ""
    result: IngestionResult | None = None
