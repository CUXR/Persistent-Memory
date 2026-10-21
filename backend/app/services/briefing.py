"""Bounded person briefings for the face path and the name path."""
from __future__ import annotations

from uuid import UUID

from ..crud.briefing_queries import get_owned_person, last_interaction, top_facts
from ..crud.memory_store import MemoryStore
from ..crud.person_resolver import PersonResolver, is_unnamed
from ..schema.briefing import Brief, BriefAmbiguous, BriefFound, BriefNotFound, BriefResult

DEFAULT_MAX_FACTS = 5


class BriefingService:
    def __init__(self, store: MemoryStore, resolver: PersonResolver | None = None) -> None:
        self.store = store
        self.resolver = resolver or PersonResolver(store)

    def brief(self, person_id: UUID, *, max_facts: int = DEFAULT_MAX_FACTS) -> Brief:
        """Face path: brief the person in view, identified by face recognition.

        Raises PersonNotFoundError for an unknown or other owner's ID.
        """
        return self._build(person_id, max_facts=max_facts, include_visual=False)

    def brief_by_name(self, name: str, *, max_facts: int = DEFAULT_MAX_FACTS) -> BriefResult:
        """Name path: resolve a spoken name, then brief or ask to disambiguate."""
        _check_max_facts(max_facts)
        resolved = self.resolver.resolve_person_from_query(name)
        if resolved.is_ambiguous:
            return BriefAmbiguous(candidates=resolved.candidates)
        if resolved.person_id is None:
            return BriefNotFound(message=f"There is no one in your memory named {name.strip()}.")
        brief = self._build(resolved.person_id, max_facts=max_facts, include_visual=True)
        return BriefFound(brief=brief)

    def _build(self, person_id: UUID, *, max_facts: int, include_visual: bool) -> Brief:
        _check_max_facts(max_facts)
        owner_id = self.store.owner_user_id
        with self.store.Session() as session:
            person = get_owned_person(session, owner_id, person_id)
            unnamed = is_unnamed(person)
            return Brief(
                person_id=person.id,
                display_name=None if unnamed else (
                    person.display_name or f"{person.first_name} {person.last_name}".strip()
                ),
                is_unnamed=unnamed,
                hobbies=top_facts(session, person.id, "hobby", max_facts),
                affiliations=top_facts(session, person.id, "affiliation", max_facts),
                visual_descriptors=(
                    top_facts(session, person.id, "visual_descriptor", max_facts) if include_visual else []
                ),
                last_interaction=last_interaction(session, owner_id, person.id),
            )


def _check_max_facts(max_facts: int) -> None:
    if max_facts < 0:
        raise ValueError("max_facts must be non-negative")


__all__ = ["BriefingService", "DEFAULT_MAX_FACTS"]
