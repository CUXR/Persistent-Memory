"""Read-only, owner-scoped queries backing person briefings."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import get_args
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from ..models.episode import Episode
from ..models.memory import EpisodeParticipant
from ..models.person import Person, PersonFact
from ..schema.briefing import BriefFact, FactCategory, FactPage, LastInteraction

# Bump when the fact ordering changes so old cursors are rejected, not misread.
CURSOR_VERSION = 1


class PersonNotFoundError(ValueError):
    """The person does not exist or belongs to another owner."""


class InvalidCursorError(ValueError):
    """The cursor is malformed or from an incompatible ordering."""


@dataclass(frozen=True)
class FactCursor:
    """Position after the last fact of a page, plus what is being paged."""

    person_id: UUID
    category: FactCategory
    created_at: datetime
    fact_id: UUID

    def encode(self) -> str:
        payload = {
            "v": CURSOR_VERSION, "p": str(self.person_id), "c": self.category,
            "t": self.created_at.isoformat(), "i": str(self.fact_id),
        }
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    @classmethod
    def decode(cls, cursor: str) -> FactCursor:
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            payload = json.loads(raw)
            if payload["v"] != CURSOR_VERSION or payload["c"] not in get_args(FactCategory):
                raise InvalidCursorError("Invalid cursor")
            return cls(
                person_id=UUID(payload["p"]), category=payload["c"],
                created_at=datetime.fromisoformat(payload["t"]), fact_id=UUID(payload["i"]),
            )
        except InvalidCursorError:
            raise
        except (binascii.Error, UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise InvalidCursorError("Invalid cursor") from exc


def get_owned_person(session: Session, owner_id: UUID, person_id: UUID) -> Person:
    person = session.get(Person, person_id)
    # Same error either way, so callers can't probe other owners' IDs.
    if person is None or person.user_id != owner_id:
        raise PersonNotFoundError(f"Person id={person_id} not found")
    return person


def fact_page(
    session: Session, person_id: UUID, category: FactCategory, limit: int,
    after: FactCursor | None = None,
) -> FactPage:
    """Return one page of a category's facts, highest ranked first.

    This is the single place facts are ordered. Facts are treated as already
    ranked by importance; until a ranking exists, newest first stands in for
    it, with the fact ID breaking ties so pages never skip or repeat. A new
    ranking changes the ordering and the cursor keys here, and CURSOR_VERSION.
    """
    if limit <= 0:
        return FactPage()
    query = select(PersonFact.id, PersonFact.fact_text, PersonFact.created_at).where(
        PersonFact.person_id == person_id, PersonFact.fact_category == category,
    )
    if after is not None:
        query = query.where(or_(
            PersonFact.created_at < after.created_at,
            and_(PersonFact.created_at == after.created_at, PersonFact.id < after.fact_id),
        ))
    rows = session.execute(
        query.order_by(PersonFact.created_at.desc(), PersonFact.id.desc()).limit(limit + 1)
    ).all()
    page, more = rows[:limit], len(rows) > limit
    next_cursor = None
    if more:
        last = page[-1]
        next_cursor = FactCursor(person_id, category, last.created_at, last.id).encode()
    return FactPage(items=[BriefFact(id=row.id, text=row.fact_text) for row in page],
                    next_cursor=next_cursor)


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


__all__ = [
    "CURSOR_VERSION",
    "FactCursor",
    "InvalidCursorError",
    "PersonNotFoundError",
    "fact_page",
    "get_owned_person",
    "last_interaction",
]
