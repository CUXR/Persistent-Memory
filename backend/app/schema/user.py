from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer


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

    @field_serializer("created_at", "updated_at")
    def _serialize_utc(self, value: datetime) -> str:
        # SQLite returns naive datetimes; present every timestamp as aware UTC ISO-8601.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
