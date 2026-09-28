from __future__ import annotations

from uuid import UUID

from typing import Optional

from pydantic import BaseModel, Field

from .memory import EdgeOut, FactOut, PersonOut, PrefOut, ProfileContext, RetrievedPersonContext, SummaryOut
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


class PersonSummaryOut(BaseModel):
    """Person record for the browser: identity only, no biometric keys or vectors."""

    id: UUID
    name: str
    aliases: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @classmethod
    def from_person(cls, person: PersonOut) -> "PersonSummaryOut":
        return cls(
            id=person.id,
            name=person.name,
            aliases=person.aliases,
            created_at=person.created_at,
            updated_at=person.updated_at,
        )


class FactSummaryOut(BaseModel):
    """Stored fact for the browser, without its retrieval embedding."""

    id: UUID
    fact_category: str
    fact_text: str
    confidence: float
    source: Optional[UUID] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    created_at: str

    @classmethod
    def from_fact(cls, fact: FactOut) -> "FactSummaryOut":
        return cls(
            id=fact.id,
            fact_category=fact.fact_category,
            fact_text=fact.fact_text,
            confidence=fact.confidence,
            source=fact.source,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            created_at=fact.created_at,
        )


class ProfileContextOut(BaseModel):
    """Profile bundle for the browser (facts, prefs, summaries, relationships)."""

    facts: list[FactSummaryOut] = Field(default_factory=list)
    prefs: list[PrefOut] = Field(default_factory=list)
    summaries: list[SummaryOut] = Field(default_factory=list)
    edges_from: list[EdgeOut] = Field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: ProfileContext) -> "ProfileContextOut":
        return cls(
            facts=[FactSummaryOut.from_fact(fact) for fact in profile.facts],
            prefs=profile.prefs,
            summaries=profile.summaries,
            edges_from=profile.edges_from,
        )


class PersonProfileOut(BaseModel):
    """Full person/profile payload returned by the people detail API."""

    person: PersonSummaryOut
    profile: ProfileContextOut


class PersonContextOut(BaseModel):
    """Query-relevant memory for one person, returned by the people context API."""

    person_id: UUID
    facts: list[FactSummaryOut] = Field(default_factory=list)
    summaries: list[SummaryOut] = Field(default_factory=list)
    edges: list[EdgeOut] = Field(default_factory=list)

    @classmethod
    def from_context(cls, context: RetrievedPersonContext) -> "PersonContextOut":
        return cls(
            person_id=context.person_id,
            facts=[FactSummaryOut.from_fact(fact) for fact in context.facts],
            summaries=context.summaries,
            edges=context.edges,
        )
