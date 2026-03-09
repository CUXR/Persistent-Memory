"""Memory-specific tables layered on top of the shared user/person/episode schema."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .episode import Episode
    from .person import Person


class Alias(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Alternate case-insensitive names for a person."""

    __tablename__ = "person_aliases"

    person_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alias: Mapped[str] = mapped_column(String(200), nullable=False)

    __table_args__ = (
        Index("ix_person_aliases_alias_ci_unique", func.lower(alias), unique=True),
    )

    person: Mapped["Person"] = relationship(back_populates="aliases")


class EpisodeParticipant(Base):
    """Join table capturing every person involved in an episode."""

    __tablename__ = "episode_participants"

    episode_id: Mapped[Any] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    person_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    episode: Mapped["Episode"] = relationship(back_populates="participants")
    person: Mapped["Person"] = relationship(back_populates="episode_links")


class Pref(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Preference remembered about a person."""

    __tablename__ = "person_prefs"

    person_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pref_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    episode_id: Mapped[Any | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    person: Mapped["Person"] = relationship(back_populates="prefs")
    episode: Mapped["Episode"] = relationship(back_populates="prefs")


class Summary(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Summary slice associated with a person."""

    __tablename__ = "person_summaries"

    person_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    episode_time_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    episode_time_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    episode_id: Mapped[Any | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    person: Mapped["Person"] = relationship(back_populates="summaries")
    episode: Mapped["Episode"] = relationship(back_populates="summaries")


class Edge(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Directed relationship between two people."""

    __tablename__ = "person_edges"
    __table_args__ = (
        UniqueConstraint("src_id", "relation", "dst_id", name="uq_person_edges_src_relation_dst"),
    )

    src_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation: Mapped[str] = mapped_column(String(100), nullable=False)
    dst_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    episode_id: Mapped[Any | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    src_person: Mapped["Person"] = relationship(
        foreign_keys=[src_id],
        back_populates="edges_from",
    )
    dst_person: Mapped["Person"] = relationship(
        foreign_keys=[dst_id],
        back_populates="edges_to",
    )
    episode: Mapped["Episode"] = relationship(back_populates="edges")


__all__ = [
    "Alias",
    "Edge",
    "EpisodeParticipant",
    "Pref",
    "Summary",
]
