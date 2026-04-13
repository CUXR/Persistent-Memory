from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ResolveCandidate(BaseModel):
    """One possible person match returned by the query resolver."""

    person_id: UUID
    name: str = Field(..., min_length=1, max_length=200)
    hints: dict[str, list[str]] = Field(default_factory=dict)


class ResolveResult(BaseModel):
    """Typed result returned by PersonResolver.resolve_person_from_query()."""

    person_id: Optional[UUID] = None
    is_ambiguous: bool = False
    candidates: list[ResolveCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "ResolveResult":
        if self.person_id is not None and self.is_ambiguous:
            raise ValueError("resolved result cannot be ambiguous")
        if self.person_id is not None and self.candidates:
            raise ValueError("resolved result should not include ambiguity candidates")
        return self


__all__ = ["ResolveCandidate", "ResolveResult"]
