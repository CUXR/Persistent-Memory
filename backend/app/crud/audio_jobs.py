"""Owner-scoped job state with atomic claims and attempt fencing for retries."""
from uuid import UUID, uuid4

from sqlalchemy import update

from audio_pipeline.segment import Recording
from ..models.audio_job import AudioJob
from ..schema.audio_job import AudioJobOut
from .memory_store import MemoryStore


class AudioJobStore:
    def __init__(self, store: MemoryStore):
        self.store = store

    def register(self, recording: Recording, person_id: UUID) -> AudioJobOut:
        with self.store.Session.begin() as session:
            self.store._assert_person_exists(session, person_id, self.store.owner_user_id)
            row = session.get(AudioJob, recording.id)
            if row is None:
                session.add(AudioJob(
                    id=recording.id, user_id=self.store.owner_user_id, person_id=person_id,
                    recording=recording.model_dump(mode="json"),
                ))
            elif (row.user_id != self.store.owner_user_id or row.person_id != person_id
                  or row.recording != recording.model_dump(mode="json")):
                raise ValueError("Recording is already registered with different ownership or metadata")
        return self.get(recording.id)

    def get(self, job_id: UUID) -> AudioJobOut:
        with self.store.Session() as session:
            row = session.get(AudioJob, job_id)
            if row is None or row.user_id != self.store.owner_user_id:
                raise ValueError("Audio job not found for this owner")
            return AudioJobOut(**{field: getattr(row, field) for field in AudioJobOut.model_fields})

    def claim(self, job_id: UUID, *, retry=False, recover_interrupted=False) -> UUID:
        allowed = ["pending"]
        if retry:
            allowed += ["failed", "needs_review"]
        if recover_interrupted:
            allowed += ["processing"]
        attempt_id = uuid4()
        with self.store.Session.begin() as session:
            result = session.execute(update(AudioJob).where(
                AudioJob.id == job_id, AudioJob.user_id == self.store.owner_user_id,
                AudioJob.status.in_(allowed),
            ).values(status="processing", stage="loading", error=None, attempt_id=attempt_id))
            if result.rowcount != 1:
                raise ValueError("Job is already processing or terminal; inspect its status before retrying")
        return attempt_id

    def update(self, job_id: UUID, attempt_id: UUID, **fields):
        with self.store.Session.begin() as session:
            result = session.execute(update(AudioJob).where(
                AudioJob.id == job_id, AudioJob.user_id == self.store.owner_user_id,
                AudioJob.attempt_id == attempt_id, AudioJob.status == "processing",
            ).values(**fields))
            if result.rowcount != 1:
                raise ValueError("Audio processing attempt is no longer current")
