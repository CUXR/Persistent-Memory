"""Owner-scoped voice enrollment and retrieval for both conversation parties."""
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from audio_pipeline.segment import DiscoveredVoice, VoiceProfile

from .memory_store import MemoryStore
from ..models.person import Person
from ..models.user import User
from ..models.audio_job import AudioJob


@dataclass(frozen=True)
class ConversationVoices:
    wearer_name: str
    interlocutor_name: str
    user: VoiceProfile | None
    interlocutor: VoiceProfile | None


class VoiceStore:
    def __init__(self, store: MemoryStore):
        self.store = store

    def _subject(self, session, person_id):
        if person_id is None:
            subject = session.get(User, self.store.owner_user_id)
        else:
            subject = session.get(Person, person_id)
            if subject is not None and subject.user_id != self.store.owner_user_id:
                subject = None
        if subject is None:
            raise ValueError("Voice enrollment subject not found for this owner")
        return subject

    def save(self, profile: VoiceProfile, *, person_id: UUID | None = None) -> None:
        """None targets the wearer; an explicit owned person ID targets their partner."""
        with self.store.Session.begin() as session:
            subject = self._subject(session, person_id)
            subject.voice_embedding = profile.embedding
            subject.voice_embedding_model = profile.model

    def validate_subject(self, person_id: UUID | None = None) -> None:
        with self.store.Session() as session:
            self._subject(session, person_id)

    def learn_from_conversation(
        self, discovery: DiscoveredVoice, *, person_id: UUID, recording_id: UUID, attempt_id: UUID,
    ) -> tuple[VoiceProfile, bool]:
        """Fill a missing partner profile once, fenced by the active audio job.

        Return the stored profile and whether this call created it. A
        competing recording or explicit replacement must never be overwritten.
        """
        with self.store.Session.begin() as session:
            person = session.scalar(select(Person).where(
                Person.id == person_id, Person.user_id == self.store.owner_user_id,
            ).with_for_update())
            job = session.scalar(select(AudioJob).where(
                AudioJob.id == recording_id, AudioJob.user_id == self.store.owner_user_id,
            ).with_for_update())
            if person is None or job is None or job.person_id != person_id:
                raise ValueError("Voice discovery does not belong to this conversation")
            if job.status != "processing" or job.attempt_id != attempt_id:
                raise ValueError("Audio processing attempt is no longer current")
            if person.voice_embedding is not None:
                return VoiceProfile(model=person.voice_embedding_model,
                                    embedding=list(person.voice_embedding)), False
            person.voice_embedding = discovery.profile.embedding
            person.voice_embedding_model = discovery.profile.model
            job.voice_discovery = discovery.model_dump(mode="json")
            return discovery.profile, True

    def load(self, person_id: UUID) -> ConversationVoices:
        def profile(subject):
            if subject.voice_embedding is None:
                return None
            if not subject.voice_embedding_model:
                raise ValueError("Legacy voice embedding has no model; re-enrollment required")
            return VoiceProfile(model=subject.voice_embedding_model, embedding=list(subject.voice_embedding))

        with self.store.Session() as session:
            user = self._subject(session, None)
            person = self._subject(session, person_id)
            return ConversationVoices(
                wearer_name=user.display_name or f"{user.first_name} {user.last_name}".strip(),
                interlocutor_name=person.display_name or f"{person.first_name} {person.last_name}".strip(),
                user=profile(user), interlocutor=profile(person),
            )
