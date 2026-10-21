"""Read-only, owner-scoped queries backing person briefings."""
from __future__ import annotations

from datetime import timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.episode import Episode
from ..models.memory import EpisodeParticipant
from ..models.person import Person, PersonFact
from ..schema.briefing import LastInteraction


class PersonNotFoundError(ValueError):
    """The person does not exist or belongs to another owner."""


def get_owned_person(session: Session, owner_id: UUID, person_id: UUID) -> Person:
    person = session.get(Person, person_id)
    # Same error either way, so callers can't probe other owners' IDs.
    if person is None or person.user_id != owner_id:
        raise PersonNotFoundError(f"Person id={person_id} not found")
    return person


def top_facts(session: Session, person_id: UUID, category: str, limit: int) -> list[str]:
    """Return the highest-ranked fact texts in one category.

    Facts are treated as already ranked by importance. Until a ranking
    exists, newest first stands in for it; swap the ordering here.
    """
    if limit <= 0:
        return []
    return list(session.scalars(
        select(PersonFact.fact_text)
        .where(PersonFact.person_id == person_id, PersonFact.fact_category == category)
        .order_by(PersonFact.created_at.desc(), PersonFact.id)
        .limit(limit)
    ))


def last_interaction(session: Session, owner_id: UUID, person_id: UUID) -> LastInteraction | None:
    episode = session.scalar(
        select(Episode)
        .join(EpisodeParticipant, EpisodeParticipant.episode_id == Episode.id)
        .where(Episode.user_id == owner_id, EpisodeParticipant.person_id == person_id)
        .order_by(Episode.start_time.desc())
        .limit(1)
    )
    if episode is None:
        return None
    started = episode.start_time
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return LastInteraction(date=started, summary=episode.dialogue_summary or None)


__all__ = ["PersonNotFoundError", "get_owned_person", "last_interaction", "top_facts"]
