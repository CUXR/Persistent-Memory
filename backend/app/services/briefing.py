"""Person briefings: by person ID, or by name with disambiguation.

Every brief carries all fact categories; clients decide what to display.
"""
from __future__ import annotations

from uuid import UUID

from ..crud.briefing_queries import (
    FactCursor, InvalidCursorError, PersonNotFoundError, fact_page, get_owned_person, last_interaction,
)
from ..crud.memory_store import MemoryStore
from ..crud.person_resolver import PersonResolver, is_unnamed
from ..schema.briefing import (
    Brief, BriefAmbiguous, BriefFound, BriefNotFound, BriefResult, FactCategory, FactPageResponse,
)

DEFAULT_MAX_FACTS = 5
# Upper bound on one page. Totals are unbounded; callers page with cursors.
MAX_PAGE_SIZE = 50


class BriefingService:
    def __init__(self, store: MemoryStore, resolver: PersonResolver | None = None) -> None:
        self.store = store
        self.resolver = resolver or PersonResolver(store)

    def brief(self, person_id: UUID, *, max_facts: int = DEFAULT_MAX_FACTS) -> Brief:
        """Brief a known person, e.g. one face recognition identified.

        Raises PersonNotFoundError for an unknown or other owner's ID.
        """
        _check_page_size(max_facts)
        with self.store.Session() as session:
            return self._build(session, person_id, max_facts=max_facts)

    def brief_by_name(self, name: str, *, max_facts: int = DEFAULT_MAX_FACTS) -> BriefResult:
        """Resolve a spoken name. One match returns the same brief as brief();
        several return candidates for the wearer to choose between by ID."""
        _check_page_size(max_facts)
        resolved = self.resolver.resolve_person_from_query(name)
        if resolved.is_ambiguous:
            # Hints arrive unbounded and newest first; keep the top max_facts per category.
            candidates = [
                c.model_copy(update={"hints": {
                    category: texts[:max_facts] for category, texts in c.hints.items() if max_facts
                }})
                for c in resolved.candidates
            ]
            return BriefAmbiguous(candidates=candidates)
        if resolved.person_id is None:
            return BriefNotFound(message=f"There is no one in your memory named {name.strip()}.")
        return BriefFound(brief=self.brief(resolved.person_id, max_facts=max_facts))

    def facts(
        self, person_id: UUID, category: FactCategory, *, cursor: str | None = None,
        limit: int = DEFAULT_MAX_FACTS,
    ) -> FactPageResponse:
        """One page of a person's facts in one category: the first page, or the
        page after `cursor`.

        Raises PersonNotFoundError for an unknown or other owner's ID, and
        InvalidCursorError for a malformed cursor or one issued for a different
        person or category.
        """
        _check_page_size(limit)
        after = FactCursor.decode(cursor) if cursor is not None else None
        with self.store.Session() as session:
            get_owned_person(session, self.store.owner_user_id, person_id)
            if after is not None and (after.person_id, after.category) != (person_id, category):
                raise InvalidCursorError("Cursor belongs to a different person or category")
            page = fact_page(session, person_id, category, limit, after)
        return FactPageResponse(person_id=person_id, category=category, **page.model_dump())

    def _build(self, session, person_id: UUID, *, max_facts: int) -> Brief:
        person = get_owned_person(session, self.store.owner_user_id, person_id)
        unnamed = is_unnamed(person)
        return Brief(
            person_id=person.id,
            display_name=None if unnamed else (
                person.display_name or f"{person.first_name} {person.last_name}".strip()
            ),
            is_unnamed=unnamed,
            hobbies=fact_page(session, person.id, "hobby", max_facts),
            affiliations=fact_page(session, person.id, "affiliation", max_facts),
            visual_descriptors=fact_page(session, person.id, "visual_descriptor", max_facts),
            last_interaction=last_interaction(session, self.store.owner_user_id, person.id),
        )


def _check_page_size(size: int) -> None:
    if size < 0:
        raise ValueError("max_facts must be non-negative")
    if size > MAX_PAGE_SIZE:
        raise ValueError(f"max_facts must be at most {MAX_PAGE_SIZE}")


__all__ = ["BriefingService", "DEFAULT_MAX_FACTS", "MAX_PAGE_SIZE", "PersonNotFoundError"]
