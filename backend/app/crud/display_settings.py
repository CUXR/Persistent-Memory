"""The owner's glasses display mode, kept in users.preferences."""
from __future__ import annotations

from typing import get_args
from uuid import UUID

from sqlalchemy.orm import Session

from ..models.user import User
from ..schema.briefing import DEFAULT_DISPLAY_MODE, DisplayMode

PREFERENCE_KEY = "glasses_display_mode"


def get_display_mode(session: Session, owner_id: UUID) -> DisplayMode:
    user = session.get(User, owner_id)
    stored = (user.preferences or {}).get(PREFERENCE_KEY) if user is not None else None
    # An unset or unrecognized value falls back rather than failing the HUD.
    return stored if stored in get_args(DisplayMode) else DEFAULT_DISPLAY_MODE


def set_display_mode(session: Session, owner_id: UUID, mode: DisplayMode) -> DisplayMode:
    if mode not in get_args(DisplayMode):
        raise ValueError(f"Unknown display mode {mode!r}")
    user = session.get(User, owner_id)
    if user is None:
        raise ValueError(f"User id={owner_id} not found")
    # Reassign so SQLAlchemy sees the JSON column change.
    user.preferences = {**(user.preferences or {}), PREFERENCE_KEY: mode}
    return mode


__all__ = ["PREFERENCE_KEY", "get_display_mode", "set_display_mode"]
