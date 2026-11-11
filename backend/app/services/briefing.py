"""Bounded person briefings for the face path and the name path."""
from __future__ import annotations

from uuid import UUID

from ..crud.briefing_queries import (
    FactCursor, PersonNotFoundError, fact_page, get_owned_person, last_interaction,
)
from ..crud.display_settings import get_display_mode, set_display_mode
from ..crud.memory_store import MemoryStore
from ..crud.person_resolver import PersonResolver, is_unnamed
from ..schema.briefing import (
    Brief, BriefAmbiguous, BriefFound, BriefNotFound, BriefResult, DisplayMode, FactPageResponse,
)

DEFAULT_MAX_FACTS = 5
# Upper bound on one page. Totals are unbounded; callers page with cursors.
MAX_PAGE_SIZE = 50


class BriefingService:
    def __init__(self, store: MemoryStore, resolver: PersonResolver | None = None) -> None:
        self.store = store
        self.resolver = resolver or PersonResolver(store)

    def brief(
        self, person_id: UUID, *, max_facts: int = DEFAULT_MAX_FACTS,
        display_mode: DisplayMode | None = None,
    ) -> Brief:
        """Face path: brief the person in view, identified by face recognition.

        `display_mode` defaults to the owner's stored glasses setting.
        Raises PersonNotFoundError for an unknown or other owner's ID.
        """
        _check_max_facts(max_facts)
        with self.store.Session() as session:
            mode = display_mode or get_display_mode(session, self.store.owner_user_id)
            return self._build(session, person_id, max_facts=max_facts, mode=mode, include_visual=False)

    def brief_by_name(self, name: str, *, max_facts: int = DEFAULT_MAX_FACTS) -> BriefResult:
        """Name path: resolve a spoken name, then brief or ask to disambiguate.

        The wearer asked explicitly, so the glasses display mode does not apply.
        """
        _check_max_facts(max_facts)
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
        with self.store.Session() as session:
            brief = self._build(session, resolved.person_id, max_facts=max_facts,
                                mode="name_and_facts", include_visual=True)
        return BriefFound(brief=brief)

    def more_facts(self, cursor: str, *, limit: int = DEFAULT_MAX_FACTS) -> FactPageResponse:
        """Continue a category from a `next_cursor` issued by a brief or an earlier page.

        Raises InvalidCursorError for a malformed cursor and PersonNotFoundError
        if the cursor's person is not this owner's.
        """
        _check_max_facts(limit)
        after = FactCursor.decode(cursor)
        with self.store.Session() as session:
            get_owned_person(session, self.store.owner_user_id, after.person_id)
            page = fact_page(session, after.person_id, after.category, limit, after)
        return FactPageResponse(person_id=after.person_id, category=after.category, **page.model_dump())

    def display_mode(self) -> DisplayMode:
        with self.store.Session() as session:
            return get_display_mode(session, self.store.owner_user_id)

    def set_display_mode(self, mode: DisplayMode) -> DisplayMode:
        with self.store.Session.begin() as session:
            return set_display_mode(session, self.store.owner_user_id, mode)

    def _build(self, session, person_id: UUID, *, max_facts: int, mode: DisplayMode,
               include_visual: bool) -> Brief:
        person = get_owned_person(session, self.store.owner_user_id, person_id)
        unnamed = is_unnamed(person)
        brief = Brief(person_id=person.id, display_mode=mode, is_unnamed=unnamed)
        if mode == "nothing":
            return brief
        if not unnamed:
            brief.display_name = person.display_name or f"{person.first_name} {person.last_name}".strip()
        if mode == "name":
            return brief
        brief.hobbies = fact_page(session, person.id, "hobby", max_facts)
        brief.affiliations = fact_page(session, person.id, "affiliation", max_facts)
        if include_visual:
            brief.visual_descriptors = fact_page(session, person.id, "visual_descriptor", max_facts)
        brief.last_interaction = last_interaction(session, self.store.owner_user_id, person.id)
        return brief


def _check_max_facts(max_facts: int) -> None:
    if max_facts < 0:
        raise ValueError("max_facts must be non-negative")
    if max_facts > MAX_PAGE_SIZE:
        raise ValueError(f"max_facts must be at most {MAX_PAGE_SIZE}")


__all__ = ["BriefingService", "DEFAULT_MAX_FACTS", "MAX_PAGE_SIZE", "PersonNotFoundError"]
