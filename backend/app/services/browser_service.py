"""Read-only services behind the memory browser UI.

Every function receives the request-scoped SQLAlchemy session (used only for its
engine) and the authenticated ``owner_user_id``. All data access is delegated to
an owner-bound :class:`~app.crud.memory_store.MemoryStore`, so every query is
scoped to that owner.
"""

from __future__ import annotations

from typing import cast
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..crud.memory_store import MemoryStore
from ..crud.person_resolver import PersonResolver
from ..schema.browser import (
    ConversationDetailOut,
    PeopleDirectoryOut,
    PersonContextOut,
    PersonListItemOut,
    PersonProfileOut,
    PersonSummaryOut,
    ProfileContextOut,
    RecentConversationOut,
)
from ..schema.person_resolver import ResolveResult
from .retrieval_service import retrieve_person_context


def _owner_store(session: Session, owner_user_id: UUID) -> MemoryStore:
    return MemoryStore.from_engine(cast(Engine, session.get_bind()), owner_user_id=owner_user_id)


def list_recent_conversations(
    session: Session,
    owner_user_id: UUID,
    *,
    limit: int = 20,
) -> list[RecentConversationOut]:
    """Load the recent episode list for one authenticated owner user."""

    return _owner_store(session, owner_user_id).list_recent_conversations(limit=limit)


def get_conversation_detail(
    session: Session,
    owner_user_id: UUID,
    episode_id: UUID,
) -> ConversationDetailOut:
    """Load one conversation episode plus participant overview cards.

    Raises ``ValueError`` when the episode does not exist for this owner.
    """

    return _owner_store(session, owner_user_id).get_conversation_detail(episode_id)


def list_people_directory(
    session: Session,
    owner_user_id: UUID,
    *,
    limit: int = 100,
    query: str | None = None,
) -> PeopleDirectoryOut:
    """Load the people list for one authenticated owner user.

    Without a query this is a single batched read ordered by most recently seen.
    With a query the existing :class:`PersonResolver` runs first; when it cannot
    resolve or disambiguate, a lexical match over names, aliases, and top facts
    is used as a fallback.
    """

    store = _owner_store(session, owner_user_id)
    cleaned_query = (query or "").strip()

    if not cleaned_query:
        return PeopleDirectoryOut(items=store.list_people_overview(limit=limit), query=None, resolution=None)

    resolution = PersonResolver(store).resolve_person_from_query(cleaned_query)
    items = _resolve_directory_items(store, cleaned_query, resolution)
    return PeopleDirectoryOut(items=items[:limit], query=cleaned_query, resolution=resolution)


def get_person_profile(
    session: Session,
    owner_user_id: UUID,
    person_id: UUID,
) -> PersonProfileOut:
    """Load one person plus their full stored profile bundle.

    Raises ``ValueError`` when the person does not exist for this owner.
    """

    store = _owner_store(session, owner_user_id)
    return PersonProfileOut(
        person=PersonSummaryOut.from_person(store.get_person(person_id)),
        profile=ProfileContextOut.from_profile(store.get_profile_context(person_id)),
    )


def get_person_retrieval_context(
    session: Session,
    owner_user_id: UUID,
    person_id: UUID,
    query: str,
) -> PersonContextOut:
    """Load query-relevant stored memory for one person.

    Raises ``ValueError`` when the person does not exist for this owner.
    """

    store = _owner_store(session, owner_user_id)
    return PersonContextOut.from_context(retrieve_person_context(person_id, query, store=store))


def _resolve_directory_items(
    store: MemoryStore,
    query: str,
    resolution: ResolveResult,
) -> list[PersonListItemOut]:
    if resolution.person_id is not None:
        return store.list_people_overview(limit=None, person_ids=[resolution.person_id])

    if resolution.is_ambiguous:
        candidate_ids = [candidate.person_id for candidate in resolution.candidates]
        cards_by_id = {card.id: card for card in store.list_people_overview(limit=None, person_ids=candidate_ids)}
        return [cards_by_id[person_id] for person_id in candidate_ids if person_id in cards_by_id]

    return _lexical_filter(store.list_people_overview(limit=None), query)


def _lexical_filter(items: list[PersonListItemOut], query: str) -> list[PersonListItemOut]:
    """Rank people whose name, aliases, or top facts mention any query token."""

    tokens = [token for token in query.lower().split() if token]
    if not tokens:
        return items

    def score(item: PersonListItemOut) -> tuple[int, int]:
        haystack = " ".join([item.name, *item.aliases, *item.top_facts]).lower()
        matched = sum(1 for token in tokens if token in haystack)
        return matched, len(item.top_facts)

    ranked = [(item, score(item)) for item in items]
    return [item for item, item_score in sorted(ranked, key=lambda row: row[1], reverse=True) if item_score[0] > 0]
