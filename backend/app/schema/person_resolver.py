from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


def _check_confidence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {value}")
    return value


class ResolveCandidate(BaseModel):
    """One possible person match returned by the query resolver."""

    person_id: UUID
    name: str = Field(..., min_length=1, max_length=200)
    confidence: float

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return _check_confidence(value)


class ResolveResult(BaseModel):
    """Typed result returned by PersonResolver.resolve_person_from_query()."""

    person_id: Optional[UUID] = None
    confidence: float = 0.0
    is_ambiguous: bool = False
    candidates: list[ResolveCandidate] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return _check_confidence(value)

    @model_validator(mode="after")
    def validate_consistency(self) -> "ResolveResult":
        if self.person_id is not None and self.is_ambiguous:
            raise ValueError("resolved result cannot be ambiguous")
        if self.person_id is not None and self.candidates:
            raise ValueError("resolved result should not include ambiguity candidates")
        return self


__all__ = ["ResolveCandidate", "ResolveResult"]
