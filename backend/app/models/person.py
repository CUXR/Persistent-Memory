from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import ARRAY, CheckConstraint, DateTime, Float, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ..core.config import get_settings
from ..core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin

settings = get_settings()

try:
    from pgvector.sqlalchemy import Vector

    VECTOR_TYPE = Vector(settings.embedding_dimension).with_variant(JSON(), "sqlite")
except ModuleNotFoundError:
    VECTOR_TYPE = JSON()

PERSONA_TYPE = ARRAY(Float).with_variant(JSON(), "sqlite")

if TYPE_CHECKING:
    from .episode import Episode
    from .memory import Alias, Edge, EpisodeParticipant, Pref, Summary
    from .user import User


class Person(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Interlocutor known to the wearer."""

    __tablename__ = "people"

    user_id: Mapped[Any] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    face_embedding: Mapped[list[float] | None] = mapped_column(VECTOR_TYPE, nullable=True)
    voice_embedding: Mapped[list[float] | None] = mapped_column(VECTOR_TYPE, nullable=True)
    face_embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    voice_embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    face_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    voice_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    persona90: Mapped[list[float]] = mapped_column(PERSONA_TYPE, nullable=False, default=list)
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
    aliases: Mapped[list["Alias"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )
    prefs: Mapped[list["Pref"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )
    summaries: Mapped[list["Summary"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )
    edges_from: Mapped[list["Edge"]] = relationship(
        foreign_keys="Edge.src_id",
        back_populates="src_person",
        cascade="all, delete-orphan",
    )
    edges_to: Mapped[list["Edge"]] = relationship(
        foreign_keys="Edge.dst_id",
        back_populates="dst_person",
    )
    episode_links: Mapped[list["EpisodeParticipant"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )


class PersonFact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Structured fact remembered about an interlocutor."""

    __tablename__ = "person_facts"
    __table_args__ = (
        CheckConstraint(
            "fact_category IN ('visual_descriptor', 'affiliation', 'hobby')",
            name="ck_person_facts_fact_category",
        ),
    )

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
    fact_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    person: Mapped["Person"] = relationship(back_populates="facts")
    source_episode: Mapped["Episode"] = relationship(back_populates="introduced_facts")
