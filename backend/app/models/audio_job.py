"""Durable processing state and structured dialog for retained recordings."""
from typing import Any

from sqlalchemy import ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from ..core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AudioJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "audio_jobs"

    user_id: Mapped[Any] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    person_id: Mapped[Any] = mapped_column(ForeignKey("people.id"), nullable=False, index=True)
    recording: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    stage: Mapped[str] = mapped_column(String(30), default="capture", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt_id: Mapped[Any | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    dialog: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    voice_discovery: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    episode_id: Mapped[Any | None] = mapped_column(ForeignKey("episodes.id"), nullable=True)
