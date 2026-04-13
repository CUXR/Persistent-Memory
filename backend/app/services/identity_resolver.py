"""Identity resolution for the current interlocutor.

This service converts observed biometric signals into a stable ``person_id`` in
the memory store. v1 resolves only on ``face_key``; ``voice_key`` can be added
later without changing the calling pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from uuid import UUID

from ..crud.memory_store import MemoryStore
from ..schema.memory import PersonOut

logger = logging.getLogger("app.services.identity_resolver")


def _clean_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _default_person_name(face_key: str) -> str:
    digest = hashlib.sha1(face_key.encode("utf-8")).hexdigest()[:8]
    return f"Unknown Interlocutor {digest}"


@dataclass(frozen=True, slots=True)
class IdentitySignal:
    """Observed identity signal for the current speaker."""

    face_key: str | None = None
    observed_name: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityResolutionResult:
    """Resolved identity outcome."""

    person_id: UUID | None
    person: PersonOut | None
    created_new_person: bool


class IdentityResolver:
    """Resolve the primary conversation partner into a stored person record."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self.active_interlocutor_id: UUID | None = None

    def resolve(self, signal: IdentitySignal) -> IdentityResolutionResult:
        """Resolve an observed signal to an existing or newly created person."""

        face_key = _clean_key(signal.face_key)
        if face_key is None:
            self._set_active_interlocutor(None)
            return IdentityResolutionResult(
                person_id=None,
                person=None,
                created_new_person=False,
            )

        person = self._store.resolve_person_by_face_key(face_key)
        created_new_person = False

        if person is None:
            name = signal.observed_name.strip() if signal.observed_name and signal.observed_name.strip() else _default_person_name(face_key)
            person = self._store.create_person(
                name=name,
                aliases=[],
                face_key=face_key,
            )
            created_new_person = True

        self._set_active_interlocutor(person.id)
        return IdentityResolutionResult(
            person_id=person.id,
            person=person,
            created_new_person=created_new_person,
        )

    def _set_active_interlocutor(self, person_id: UUID | None) -> None:
        previous = self.active_interlocutor_id
        if previous == person_id:
            return

        self.active_interlocutor_id = person_id
        logger.info(
            "active_interlocutor_id changed from %s to %s",
            previous,
            person_id,
        )
