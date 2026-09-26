"""ASR adds text to the shared audio segment without discarding identity."""
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from audio_pipeline.segment import AttributionMethod, SpeakerLabel, SpeechSegment


class RawTranscription(BaseModel):
    segment: SpeechSegment
    text: str
    avg_logprob: float = Field(le=0, allow_inf_nan=False)
    no_speech_prob: float = Field(ge=0, le=1)


class DialogTurn(BaseModel):
    speaker: SpeakerLabel
    speaker_id: str = Field(min_length=1)
    attribution_method: AttributionMethod = "unknown"
    person_id: UUID | None = None
    speaker_similarity: float | None = Field(default=None, ge=-1, le=1)
    text: str
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    asr_confidence: float = Field(ge=0, le=1)
    segment_count: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


class TranscriptionFailure(BaseModel):
    segment: SpeechSegment
    error: str


class Dialog(BaseModel):
    turns: list[DialogTurn] = Field(default_factory=list)
    errors: list[TranscriptionFailure] = Field(default_factory=list)

    def to_transcript(self, wearer_name: str, interlocutor_name: str) -> str:
        names = {"user": wearer_name, "interlocutor": interlocutor_name}
        return "\n".join(
            f"{names.get(turn.speaker, f'Unknown ({turn.speaker_id})')}: {turn.text}"
            for turn in self.turns
        )
