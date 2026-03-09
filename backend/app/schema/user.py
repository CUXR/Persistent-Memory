from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UserFactRead(BaseModel):
    """Serialized user fact."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    fact_text: str
    source: str | None = None
    confidence: Decimal | None = None


class UserRead(BaseModel):
    """Serialized user record."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    first_name: str
    last_name: str
    display_name: str | None = None
    username: str
    preferences: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    facts: list[UserFactRead] = Field(default_factory=list)


class WearerStateRead(BaseModel):
    """Serialized wearer state consumed by the interlocutor tracker."""

    wearer_person_id: UUID
