from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.person import Person, PersonFact
    from app.models.user import User


class Episode(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "episodes"
    __table_args__ = (
        Index("ix_episodes_user_id_start_time", "user_id", "start_time"),
        Index("ix_episodes_person_id_start_time", "person_id", "start_time"),
    )

    user_id: Mapped[Any] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    person_id: Mapped[Any] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dialogue_summary: Mapped[str] = mapped_column(Text, nullable=False)
    summary_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    importance_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    user: Mapped["User"] = relationship(back_populates="episodes")
    person: Mapped["Person"] = relationship(back_populates="episodes")
    introduced_facts: Mapped[list["PersonFact"]] = relationship(back_populates="source_episode")
