from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.user import User


def get_or_create_wearer(session: Session) -> User:
    """Return the owner/wearer user, creating a default row when absent."""

    wearer = session.scalar(select(User).order_by(User.created_at.asc()).limit(1))
    if wearer is None:
        wearer = User(
            first_name="Memory",
            last_name="Owner",
            display_name="Memory Store Owner",
            username="memory-store-owner",
        )
        session.add(wearer)
        session.commit()
        session.refresh(wearer)
    return wearer
