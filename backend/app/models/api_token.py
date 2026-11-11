"""Bearer credentials that map a device or client to its owner."""
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ApiToken(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One issued credential. Only the SHA-256 of the token is stored."""

    __tablename__ = "api_tokens"

    user_id: Mapped[Any] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
