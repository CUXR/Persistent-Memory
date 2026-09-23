"""Endpoints for writing extracted memories and retrieving ranked recall.

The write endpoints are a thin HTTP surface over MemoryStore.write_fact /
write_summary, intended for a memory-extraction pipeline to call once it has
produced structured output for a conversation. The relevant-memories endpoint
implements the re-encounter path: bounded, ranked recall for a known person.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from pydantic import BaseModel, Field

from ...crud.memory_store import MemoryStore
from ...schema.memory import FactOut, PersonFactCategory, RelevantMemories, SummaryOut
from ..deps import get_memory_store

router = APIRouter()


class FactWriteRequest(BaseModel):
    """Body for POST /people/{person_id}/facts (person_id comes from the path)."""

    fact_text: str = Field(..., min_length=1)
    confidence: float = 1.0
    fact_category: Optional[PersonFactCategory] = None
    episode_id: Optional[UUID] = None
    embedding: Optional[list[float]] = None


class SummaryWriteRequest(BaseModel):
    """Body for POST /people/{person_id}/summaries (person_id comes from the path)."""

    summary_text: str = Field(..., min_length=1)
    episode_id: Optional[UUID] = None
    embedding: Optional[list[float]] = None


class RelevantMemoriesRequest(BaseModel):
    """Body for POST /people/{person_id}/memories/relevant."""

    limit: int = Field(default=5, ge=1, le=50)
    query_embedding: Optional[list[float]] = None


@router.post("/{person_id}/facts", response_model=FactOut, status_code=201)
def write_fact(
    person_id: UUID,
    body: FactWriteRequest,
    store: MemoryStore = Depends(get_memory_store),
) -> FactOut:
    try:
        fact_id = store.write_fact(
            person_id=person_id,
            fact_text=body.fact_text,
            confidence=body.confidence,
            fact_category=body.fact_category,
            episode_id=body.episode_id,
            embedding=body.embedding,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    profile = store.get_profile_context(person_id)
    return next(f for f in profile.facts if f.id == fact_id)


@router.post("/{person_id}/summaries", response_model=SummaryOut, status_code=201)
def write_summary(
    person_id: UUID,
    body: SummaryWriteRequest,
    store: MemoryStore = Depends(get_memory_store),
) -> SummaryOut:
    try:
        summary_id = store.write_summary(
            person_id=person_id,
            summary_text=body.summary_text,
            episode_id=body.episode_id,
            embedding=body.embedding,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    profile = store.get_profile_context(person_id)
    return next(s for s in profile.summaries if s.id == summary_id)


@router.post("/{person_id}/memories/relevant", response_model=RelevantMemories)
def get_relevant_memories(
    person_id: UUID,
    body: Optional[RelevantMemoriesRequest] = None,
    store: MemoryStore = Depends(get_memory_store),
) -> RelevantMemories:
    """Bounded, ranked memories for a person the wearer just re-encountered."""

    query = body or RelevantMemoriesRequest()
    try:
        return store.get_relevant_memories(
            person_id,
            limit=query.limit,
            query_embedding=query.query_embedding,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
