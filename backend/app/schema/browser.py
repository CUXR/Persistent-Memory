from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from .memory import PersonOut, ProfileContext
from .person_resolver import ResolveResult


class ConversationParticipantOut(BaseModel):
    """One person shown in the recent-conversations list."""

    id: UUID
    name: str


class RecentConversationOut(BaseModel):
    """List item returned by the recent-conversations API."""

    id: UUID
    started_at: str
    ended_at: str | None = None
    summary: str
    participants: list[ConversationParticipantOut] = Field(default_factory=list)

class PersonListItemOut(BaseModel):
    """List item returned by the people-browse API."""

    id: UUID
    name: str
    aliases: list[str] = Field(default_factory=list)
    last_seen_at: str | None = None
    top_facts: list[str] = Field(default_factory=list)
    fact_count: int = 0
    summary_count: int = 0
    relationship_count: int = 0


class ConversationDetailOut(BaseModel):
    """Full conversation payload returned by the conversation detail API."""

    id: UUID
    started_at: str
    ended_at: str | None = None
    summary: str
    transcript: str
    participants: list[PersonListItemOut] = Field(default_factory=list)


class PeopleDirectoryOut(BaseModel):
    """People browse payload, optionally filtered by a backend search query."""

    items: list[PersonListItemOut] = Field(default_factory=list)
    query: str | None = None
    resolution: ResolveResult | None = None


class PersonProfileOut(BaseModel):
    """Full person/profile payload returned by the people detail API."""

    person: PersonOut
    profile: ProfileContext
