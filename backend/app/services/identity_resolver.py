from __future__ import annotations

from dataclasses import dataclass
import logging
from uuid import UUID, uuid4

from ..crud.memory_store import MemoryStore

logger = logging.getLogger("app.services.identity_resolver")


@dataclass(frozen=True, slots=True)
class IdentitySignal:
    """Observed identity features for the current interlocutor."""

    face_key: str | None = None


class IdentityResolver:
    """Resolve the active interlocutor from observed identity signals."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self.active_interlocutor_id: UUID | None = None

    def resolve(self, signal: IdentitySignal | str | None) -> UUID | None:
        """Return the matched or created person id for the current signal."""

        face_key = self._extract_face_key(signal)
        if not face_key:
            self._set_active_interlocutor(None)
            return None

        person = self._store.resolve_person_by_face_key(face_key)
        if person is None:
            person = self._store.upsert_person(
                name=self._placeholder_name(),
                aliases=[],
                face_key=face_key,
            )
            logger.info("Created new person id=%s for face_key=%s", person.id, face_key)

        self._set_active_interlocutor(person.id)
        return person.id

    def _set_active_interlocutor(self, person_id: UUID | None) -> None:
        previous = self.active_interlocutor_id
        if previous == person_id:
            return

        self.active_interlocutor_id = person_id
        logger.info("Active interlocutor changed from %s to %s", previous, person_id)

    @staticmethod
    def _placeholder_name() -> str:
        return f"Unknown Person {uuid4().hex[:12]}"

    @staticmethod
    def _extract_face_key(signal: IdentitySignal | str | None) -> str:
        if isinstance(signal, IdentitySignal):
            return (signal.face_key or "").strip()
        return (signal or "").strip()
