"""Enroll an explicitly identified speaker from a clean single-speaker recording."""
from pathlib import Path
from uuid import UUID

from audio_pipeline.audio import load_audio
from audio_pipeline.segment import SAMPLE_RATE, VoiceProfile

from ..crud.voice import VoiceStore


class VoiceEnrollmentService:
    def __init__(self, voices: VoiceStore, diarizer, min_speech_seconds: float = 5):
        if min_speech_seconds <= 0:
            raise ValueError("Minimum speech duration must be positive")
        self.voices = voices
        self.diarizer = diarizer
        self.min_speech_seconds = min_speech_seconds

    def enroll(self, audio_path: Path, *, person_id: UUID | None = None) -> VoiceProfile:
        self.voices.validate_subject(person_id)
        audio = load_audio(audio_path, max_seconds=120)
        turns = self.diarizer.diarize(audio, SAMPLE_RATE)
        speakers = {turn.speaker_id for turn in turns}
        if len(speakers) != 1:
            raise ValueError("Enrollment requires exactly one speaker")
        eligible = [turn for turn in turns if turn.duration >= 1]
        # Count unique speech time; overlapping intervals must not inflate duration.
        speech_seconds, end = 0.0, 0.0
        for turn in sorted(eligible, key=lambda turn: turn.start):
            speech_seconds += max(0, turn.end - max(end, turn.start))
            end = max(end, turn.end)
        if speech_seconds < self.min_speech_seconds:
            raise ValueError(f"Enrollment requires at least {self.min_speech_seconds:g}s of clear speech")
        embeddings = self.diarizer.extract_per_speaker_embeddings(audio, eligible, SAMPLE_RATE)
        vector = embeddings.get(next(iter(speakers)))
        if vector is None:
            raise ValueError("Unable to extract a reliable enrollment embedding")
        result = VoiceProfile(model=self.diarizer.embedding_model, embedding=vector.tolist())
        self.voices.save(result, person_id=person_id)
        return result
