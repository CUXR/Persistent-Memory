from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from .person_resolver import ResolveCandidate

FactCategory = Literal["hobby", "affiliation", "visual_descriptor"]


class LastInteraction(BaseModel):
    """Most recent conversation with a person."""

    date: datetime
    summary: Optional[str] = None


class BriefFact(BaseModel):
    id: UUID
    text: str


class FactPage(BaseModel):
    """One page of a category's facts, best first.

    `next_cursor` is opaque; pass it to GET /people/{id}/facts for the next
    page. It is null on the last page.
    """

    items: list[BriefFact] = Field(default_factory=list)
    next_cursor: Optional[str] = None


class Brief(BaseModel):
    """Everything to remember about one person. The client decides what to show."""

    person_id: UUID
    # Null when the person is unnamed; never the "Unknown person <uuid>" placeholder.
    display_name: Optional[str] = None
    is_unnamed: bool
    hobbies: FactPage
    affiliations: FactPage
    visual_descriptors: FactPage
    last_interaction: Optional[LastInteraction] = None


class FactPageResponse(FactPage):
    """A page from GET /people/{id}/facts."""

    person_id: UUID
    category: FactCategory


class BriefFound(BaseModel):
    status: Literal["found"] = "found"
    brief: Brief


class BriefAmbiguous(BaseModel):
    status: Literal["ambiguous"] = "ambiguous"
    candidates: list[ResolveCandidate]


class BriefNotFound(BaseModel):
    status: Literal["not_found"] = "not_found"
    message: str


BriefResult = Annotated[
    Union[BriefFound, BriefAmbiguous, BriefNotFound],
    Field(discriminator="status"),
]


__all__ = [
    "Brief",
    "BriefAmbiguous",
    "BriefFact",
    "BriefFound",
    "BriefNotFound",
    "BriefResult",
    "FactCategory",
    "FactPage",
    "FactPageResponse",
    "LastInteraction",
]
