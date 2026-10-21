from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from .person_resolver import ResolveCandidate


class LastInteraction(BaseModel):
    """Most recent conversation with a person."""

    date: datetime
    summary: Optional[str] = None


class Brief(BaseModel):
    """Bounded "what to remember" view of one person."""

    person_id: UUID
    display_name: Optional[str] = None
    is_unnamed: bool
    hobbies: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    # Only filled on the name path; on the face path the person is in view.
    visual_descriptors: list[str] = Field(default_factory=list)
    last_interaction: Optional[LastInteraction] = None


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
    "BriefFound",
    "BriefNotFound",
    "BriefResult",
    "LastInteraction",
]
