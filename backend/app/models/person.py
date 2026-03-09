from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin

settings = get_settings()

if TYPE_CHECKING:
    from app.models.episode import Episode
    from app.models.user import User


class Person(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "people"

    user_id: Mapped[Any] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    face_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )
    voice_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )
    face_embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    voice_embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    owner: Mapped["User"] = relationship(back_populates="people")
    facts: Mapped[list["PersonFact"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )
    episodes: Mapped[list["Episode"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )


class PersonFact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "person_facts"

    person_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_episode_id: Mapped[Any | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    person: Mapped["Person"] = relationship(back_populates="facts")
    source_episode: Mapped["Episode"] = relationship(back_populates="introduced_facts")
