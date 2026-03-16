"""Pydantic schemas for the conversation ingestion pipeline.

Serves double duty: typed return values for the service layer *and* structured
output schemas sent to the OpenAI API via ``response_format``.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM response schemas (used as OpenAI structured-output targets)
# ---------------------------------------------------------------------------


class EpisodeSummaryLLMResponse(BaseModel):
    """Structured output from the episode summary LLM call."""

    summary: str = Field(description="Concise summary of the conversation episode.")
    importance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How significant is this episode? "
            "0.0 = trivial small talk, 1.0 = life-changing event."
        ),
    )


# Must align with the DB CHECK constraint on PersonFact.fact_category
FactCategory = Literal["visual_descriptor", "affiliation", "hobby"]


class ExtractedFact(BaseModel):
    """A single candidate fact about the interlocutor extracted by the LLM."""

    fact_text: str = Field(description="The fact in a short, declarative sentence.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this fact is accurate and worth storing.",
    )
    category: FactCategory = Field(
        description="Semantic category of the fact.",
    )
    is_shared_interest: bool = Field(
        default=False,
        description="True when this interest/hobby is also held by the wearer.",
    )


class ExtractedEdge(BaseModel):
    """A directed relationship extracted from the conversation."""

    relation: str = Field(
        description=(
            "Relationship label, e.g. 'works_at', 'member_of', 'knows'. "
            "Use snake_case."
        )
    )
    target_name: str = Field(
        description="Display name of the target person or organisation.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this relationship is accurate.",
    )


class FactExtractionLLMResponse(BaseModel):
    """Structured output from the fact-extraction LLM call."""

    facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="New facts about the interlocutor not already stored.",
    )
    edges: list[ExtractedEdge] = Field(
        default_factory=list,
        description="Directed relationships involving the interlocutor.",
    )


# ---------------------------------------------------------------------------
# Service return type
# ---------------------------------------------------------------------------


class IngestionResult(BaseModel):
    """Return value from :func:`ingest_conversation`."""

    episode_id: UUID
    person_id: UUID
    summary: str
    importance_score: float
    facts_written: list[UUID] = Field(
        default_factory=list,
        description="UUIDs of PersonFact rows created during this ingestion.",
    )
    facts_skipped_as_duplicate: list[str] = Field(
        default_factory=list,
        description="fact_text values that matched existing facts and were skipped.",
    )
    edges_written: list[UUID] = Field(
        default_factory=list,
        description="UUIDs of Edge rows created during this ingestion.",
    )
