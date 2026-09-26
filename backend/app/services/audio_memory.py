"""Coordinate retained recordings through voice matching, ASR, and memory."""
from datetime import timedelta
from uuid import UUID
import asyncio

import soundfile as sf

from audio_pipeline.segment import Recording, SAMPLE_RATE, validate_audio
from audio_pipeline.speaker_attribution import SpeakerAttributor
from ..crud.audio_jobs import AudioJobStore
from ..crud.memory_store import MemoryStore
from ..crud.voice import VoiceStore
from ..schema.audio_job import AudioJobOut
from .asr import transcribe_segments
from .conversation_ingestion import ingest_conversation


class AudioMemoryService:
    def __init__(self, store: MemoryStore, diarizer, asr_engine, *, llm_client=None,
                 attributor: SpeakerAttributor | None = None):
        self.store = store
        self.jobs = AudioJobStore(store)
        self.voices = VoiceStore(store)
        self.diarizer = diarizer
        self.asr_engine = asr_engine
        self.llm_client = llm_client
        self.attributor = attributor or SpeakerAttributor()

    async def process(
        self, recording: Recording, person_id: UUID, *, retry: bool = False,
        recover_interrupted: bool = False,
    ) -> AudioJobOut:
        job = self.jobs.register(recording, person_id)
        person_id = job.person_id
        if job.status in ("complete", "no_speech"):
            return job
        attempt = self.jobs.claim(recording.id, retry=retry, recover_interrupted=recover_interrupted)
        stage = "loading"
        try:
            voices = self.voices.load(person_id)
            audio, sample_rate = sf.read(recording.audio_path, dtype="float32")
            audio = validate_audio(audio, sample_rate)
            if len(audio) != recording.sample_count:
                raise ValueError("WAV length no longer matches its recording manifest")
            stage = "diarization"
            self.jobs.update(recording.id, attempt, stage=stage)
            turns = await asyncio.to_thread(self.diarizer.diarize, audio, SAMPLE_RATE)
            if not turns:
                self.jobs.update(recording.id, attempt, status="no_speech", stage="complete",
                                 dialog={"turns": [], "errors": []}, transcript="")
                return self.jobs.get(recording.id)
            stage = "attribution"
            self.jobs.update(recording.id, attempt, stage=stage)
            embeddings = await asyncio.to_thread(
                self.diarizer.extract_per_speaker_embeddings, audio, turns, SAMPLE_RATE,
            )
            segments = self.attributor.attribute(
                turns, embeddings, recording, embedding_model=self.diarizer.embedding_model,
                user_profile=voices.user, interlocutor_profile=voices.interlocutor,
                interlocutor_id=person_id,
            )
            learned_here = job.voice_discovery is not None
            if voices.interlocutor is None:
                discovery = self.attributor.discover_interlocutor(
                    segments, embeddings, user_profile=voices.user,
                    embedding_model=self.diarizer.embedding_model,
                )
                if discovery is not None:
                    profile, created = self.voices.learn_from_conversation(
                        discovery, person_id=person_id, recording_id=recording.id, attempt_id=attempt,
                    )
                    if created:
                        learned_here = True
                        segments = [segment.model_copy(update={
                            "speaker_label": "interlocutor", "person_id": person_id,
                        }) if segment.speaker_id == discovery.speaker_id else segment for segment in segments]
                    else:
                        # Another recording filled the profile first. Match against
                        # that profile instead of overwriting or assuming identity.
                        segments = self.attributor.attribute(
                            turns, embeddings, recording, embedding_model=self.diarizer.embedding_model,
                            user_profile=voices.user, interlocutor_profile=profile, interlocutor_id=person_id,
                        )
            if learned_here:
                # Comparing a discovery recording to its own embedding is not
                # independent identity evidence, including when retrying this job.
                segments = [segment.model_copy(update={
                    "attribution_method": "conversation", "speaker_similarity": None,
                }) if segment.speaker_label == "interlocutor" else segment for segment in segments]
            stage = "asr"
            self.jobs.update(recording.id, attempt, stage=stage)
            dialog = await asyncio.to_thread(transcribe_segments, segments, self.asr_engine)
            transcript = dialog.to_transcript(voices.wearer_name, voices.interlocutor_name)
            self.jobs.update(recording.id, attempt, dialog=dialog.model_dump(mode="json"), transcript=transcript)
            if dialog.errors:
                raise RuntimeError(f"ASR failed for {len(dialog.errors)} segment(s); recording retained")
            overlap = any(
                left.speaker_id != right.speaker_id and left.start < right.end and right.start < left.end
                for index, left in enumerate(turns) for right in turns[index + 1:]
            )
            if not dialog.turns:
                self.jobs.update(recording.id, attempt, status="no_speech", stage="complete")
            elif (overlap or any(segment.speaker_label == "unknown" for segment in segments)
                  or not any(turn.speaker == "interlocutor" for turn in dialog.turns)):
                self.jobs.update(
                    recording.id, attempt, status="needs_review", stage="attribution",
                    error="Unknown speaker, overlapping speech, or no confirmed interlocutor; memories not extracted",
                )
            else:
                stage = "memory"
                self.jobs.update(recording.id, attempt, stage=stage)
                await ingest_conversation(
                    transcript, voices.wearer_name, voices.interlocutor_name,
                    recording.started_at, recording.started_at + timedelta(seconds=recording.duration),
                    self.store, llm_client=self.llm_client, person_id=person_id,
                    recording_id=recording.id, attempt_id=attempt,
                )
        except Exception as exc:
            # A recovered attempt fences out the old worker, including its error updates.
            self.jobs.update(recording.id, attempt, status="failed", stage=stage, error=str(exc))
        return self.jobs.get(recording.id)
