from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ..core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .episode import Episode
    from .person import Person


PREFERENCES_TYPE = JSON().with_variant(JSONB, "postgresql")


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Primary user wearing the smart glasses."""

    __tablename__ = "users"

    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    oauth_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    oauth_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    preferences: Mapped[dict[str, Any]] = mapped_column(PREFERENCES_TYPE, nullable=False, default=dict)

    facts: Mapped[list["UserFact"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    episodes: Mapped[list["Episode"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    people: Mapped[list["Person"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class UserFact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """User-level facts unrelated to a specific interlocutor."""

    __tablename__ = "user_facts"

    user_id: Mapped[Any] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    user: Mapped["User"] = relationship(back_populates="facts")
