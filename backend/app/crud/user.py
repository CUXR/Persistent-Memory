from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models.user import User


def get_user_by_id(session: Session, user_id: UUID) -> User | None:
    """Return one user and their saved user-level facts."""

    return session.scalar(
        select(User)
        .options(selectinload(User.facts))
        .where(User.id == user_id)
    )
