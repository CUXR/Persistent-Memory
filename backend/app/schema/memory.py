"""
backend/app/schema/memory.py
──────────────
Pydantic models for the EgoMem memory store.

Every public method on MemoryStore accepts and returns these models.
This gives us:
  - runtime input validation (replaces Zod from the TS world)
  - auto-generated JSON schemas (useful for FastAPI)
  - a single place to see every data shape in the system

Design notes:
  - "In" models  = what you pass INTO a method  (write inputs)
  - "Out" models = what you get BACK from a method (read outputs)
  - ProfileContext = the composite shape returned by get_profile_context()
                     This maps directly to the Level-1 MemChunk content
                     that gets injected into the dialog model (Eq. 2: p_t).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# Allowed categories for PersonFact.fact_category (mirrors DB CHECK constraint)
PersonFactCategory = Literal["visual_descriptor", "affiliation", "hobby"]


# ── Shared validators ────────────────────────────────────────

def _check_confidence(v: float) -> float:
    """Confidence must be between 0 and 1 inclusive."""
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {v}")
    return v


def _check_persona90(v: list[float]) -> list[float]:
    """persona90 must be exactly 90 floats (or empty)."""
    if len(v) != 0 and len(v) != 90:
        raise ValueError(f"persona90 must have 0 or 90 elements, got {len(v)}")
    return v


#input schemas

#validates name length and persona length rule
class PersonIn(BaseModel):
    """Input for upsert_person()."""
    name: str = Field(..., min_length=1, max_length=200)
    face_key: Optional[str] = None
    voice_key: Optional[str] = None
    persona90: list[float] = Field(default_factory=list)

    @field_validator("persona90")
    @classmethod
    def validate_persona90(cls, v: list[float]) -> list[float]:
        return _check_persona90(v)


class EpisodeIn(BaseModel):
    """Input for write_episode()."""
    time_start: datetime
    time_end: datetime
    transcript: str = ""
    summary: str = ""
    participant_ids: list[UUID] = Field(default_factory=list)

    @field_validator("time_end")
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        start = info.data.get("time_start")
        if start and v < start:
            raise ValueError("time_end must be >= time_start")
        return v


class FactIn(BaseModel):
    """Input for write_fact()."""
    person_id: UUID
    fact_text: str = Field(..., min_length=1)
    confidence: float = 1.0
    fact_category: Optional[PersonFactCategory] = None
    episode_id: Optional[UUID] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return _check_confidence(v)

    @field_validator("valid_to")
    @classmethod
    def valid_range(cls, v: Optional[datetime], info) -> Optional[datetime]:
        start = info.data.get("valid_from")
        if start and v and v < start:
            raise ValueError("valid_to must be >= valid_from")
        return v


class SummaryIn(BaseModel):
    """Input for write_summary()."""
    person_id: UUID
    summary_text: str = Field(..., min_length=1)
    episode_time_start: Optional[datetime] = None
    episode_time_end: Optional[datetime] = None
    episode_id: Optional[UUID] = None

    @field_validator("episode_time_end")
    @classmethod
    def summary_end_after_start(cls, v: Optional[datetime], info) -> Optional[datetime]:
        start = info.data.get("episode_time_start")
        if start and v and v < start:
            raise ValueError("episode_time_end must be >= episode_time_start")
        return v

#enforces non-empty relation and forbids self edges
class EdgeIn(BaseModel):
    """Input for write_edge()."""
    src_id: UUID
    relation: str = Field(..., min_length=1, max_length=100)
    dst_id: UUID
    confidence: float = 1.0
    episode_id: Optional[UUID] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return _check_confidence(v)

    @field_validator("dst_id")
    @classmethod
    def no_self_edges(cls, v: UUID, info) -> UUID:
        if info.data.get("src_id") == v:
            raise ValueError("self-edges are not allowed (src_id == dst_id)")
        return v


#outputs schemas

class PersonOut(BaseModel):
    id: UUID
    name: str
    face_key: Optional[str]
    voice_key: Optional[str]
    persona90: list[float]
    created_at: str
    updated_at: str


class FactOut(BaseModel):
    id: UUID
    fact_text: str
    confidence: float
    fact_category: Optional[str] = None
    episode_id: Optional[UUID]
    valid_from: Optional[str]
    valid_to: Optional[str]
    created_at: str


class SummaryOut(BaseModel):
    id: UUID
    summary_text: str
    episode_time_start: Optional[str]
    episode_time_end: Optional[str]
    episode_id: Optional[UUID]
    created_at: str


class EdgeOut(BaseModel):
    id: UUID
    relation: str
    dst_id: UUID
    dst_name: str
    confidence: float
    episode_id: Optional[UUID]
    created_at: str


class ProfileContext(BaseModel):
    """
    The composite view returned by get_profile_context().

    This is the Python equivalent of the Level-1 MemChunk content
    from the paper. When the retrieval process identifies a user,
    it calls get_profile_context(person_id) and the result gets
    serialized into the text channel of the MemChunk (Eq. 2: p_t).

    Shape matches the issue spec:
        { facts, summaries, edges_from, persona90 }
    """
    facts: list[FactOut] = Field(default_factory=list)
    summaries: list[SummaryOut] = Field(default_factory=list)
    edges_from: list[EdgeOut] = Field(default_factory=list)
    persona90: list[float] = Field(default_factory=list)
