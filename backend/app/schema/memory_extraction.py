"""Pydantic schemas for the memory extraction pipeline (issue #29).

Extraction only — turning a transcript into structured, validated candidate
memories. Persisting candidates to the store is a separate concern (see
app/services/conversation_ingestion.py and app/crud/memory_store.py).

Two layers, mirroring schema/ingestion.py's convention:
  - "LLM response" schemas — the structured-output target sent to OpenAI.
    The model decides text/category/confidence; nothing else.
  - "Candidate" schemas — the service's return type. Adds ``needs_review``,
    computed deterministically from confidence, so review-flagging logic
    lives in our code, not the prompt.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Below this confidence, a candidate is flagged for human review rather than
# silently trusted. Callers may override per-extraction.
DEFAULT_REVIEW_CONFIDENCE_THRESHOLD = 0.5

# Mirrors the DB CHECK constraint on PersonFact.fact_category.
FactCategory = Literal["visual_descriptor", "affiliation", "hobby"]
PreferencePolarity = Literal["like", "dislike", "neutral"]


# ── LLM response schemas (OpenAI structured-output targets) ─────


class RawFact(BaseModel):
    """A candidate fact as decided by the model — pre-review-flagging."""

    fact_text: str = Field(description="The fact in a short, declarative sentence.")
    category: FactCategory = Field(description="Semantic category of the fact.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence the fact is accurate and worth storing.")


class RawPreference(BaseModel):
    """A candidate like/dislike/opinion as decided by the model."""

    preference_text: str = Field(description="The preference in a short, declarative sentence.")
    polarity: PreferencePolarity = Field(description="Whether this is a like, dislike, or neutral opinion.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence the preference is accurate and worth storing.")


class RawRelationship(BaseModel):
    """A candidate directed relationship to another named person or org."""

    target_name: str = Field(description="Display name of the target person or organisation.")
    relation: str = Field(description="Relationship label in snake_case, e.g. 'works_at', 'knows'.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence the relationship is accurate.")


class RawEvent(BaseModel):
    """A candidate specific occurrence — past or planned — mentioned in conversation."""

    event_text: str = Field(description="Short declarative description of the event.")
    occurred_at: Optional[str] = Field(
        default=None,
        description="Free-text temporal reference exactly as implied, e.g. 'last weekend', 'next March'. Null if no time was mentioned.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence the event is accurate and worth storing.")


class MemoryExtractionLLMResponse(BaseModel):
    """Structured output from the memory-extraction LLM call."""

    facts: list[RawFact] = Field(default_factory=list)
    preferences: list[RawPreference] = Field(default_factory=list)
    relationships: list[RawRelationship] = Field(default_factory=list)
    events: list[RawEvent] = Field(default_factory=list)


# ── Candidate schemas (service return type) ──────────────────


class FactCandidate(BaseModel):
    fact_text: str
    category: FactCategory
    confidence: float
    needs_review: bool


class PreferenceCandidate(BaseModel):
    pref_text: str
    polarity: PreferencePolarity
    confidence: float
    needs_review: bool


class RelationshipCandidate(BaseModel):
    target_name: str
    relation: str
    confidence: float
    needs_review: bool


class EventCandidate(BaseModel):
    event_text: str
    occurred_at: Optional[str]
    confidence: float
    needs_review: bool


class MemoryExtractionResult(BaseModel):
    """Typed memory candidates extracted from one transcript."""

    facts: list[FactCandidate] = Field(default_factory=list)
    preferences: list[PreferenceCandidate] = Field(default_factory=list)
    relationships: list[RelationshipCandidate] = Field(default_factory=list)
    events: list[EventCandidate] = Field(default_factory=list)


__all__ = [
    "DEFAULT_REVIEW_CONFIDENCE_THRESHOLD",
    "EventCandidate",
    "FactCandidate",
    "FactCategory",
    "MemoryExtractionLLMResponse",
    "MemoryExtractionResult",
    "PreferenceCandidate",
    "PreferencePolarity",
    "RawEvent",
    "RawFact",
    "RawPreference",
    "RawRelationship",
    "RelationshipCandidate",
]
