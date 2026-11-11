from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from .person_resolver import ResolveCandidate

# What the glasses show for the person in view. There is no one-fact mode.
DisplayMode = Literal["nothing", "name", "name_and_facts"]
DEFAULT_DISPLAY_MODE: DisplayMode = "name_and_facts"

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

    `next_cursor` is opaque; pass it to GET /briefings/facts for the next page.
    It is null on the last page.
    """

    items: list[BriefFact] = Field(default_factory=list)
    next_cursor: Optional[str] = None


class Brief(BaseModel):
    """Bounded "what to remember" view of one person.

    A fact section is null when this mode or path does not include it, and an
    empty page when it is included but the person has no such facts.
    """

    person_id: UUID
    display_mode: DisplayMode
    # Null when the person is unnamed or the mode is "nothing".
    display_name: Optional[str] = None
    is_unnamed: bool
    hobbies: Optional[FactPage] = None
    affiliations: Optional[FactPage] = None
    # Name path only; on the face path the person is in view.
    visual_descriptors: Optional[FactPage] = None
    last_interaction: Optional[LastInteraction] = None


class FactPageResponse(FactPage):
    """A continuation page from GET /briefings/facts."""

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


class DisplaySettings(BaseModel):
    display_mode: DisplayMode


__all__ = [
    "DEFAULT_DISPLAY_MODE",
    "Brief",
    "BriefAmbiguous",
    "BriefFact",
    "BriefFound",
    "BriefNotFound",
    "BriefResult",
    "DisplayMode",
    "DisplaySettings",
    "FactCategory",
    "FactPage",
    "FactPageResponse",
    "LastInteraction",
]
