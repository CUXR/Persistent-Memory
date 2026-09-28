from collections.abc import Generator
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.user import User

settings = get_settings()


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def get_authenticated_owner_user_id(
    request: Request,
    session: Session = Depends(get_db_session),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> UUID:
    """Return the owner user id that every memory read must be scoped to.

    Resolution order:

    1. ``request.state.user_id`` — the integration point for the session/auth
       middleware delivered by the separate account-creation ticket.
    2. Local development only, when ``ALLOW_DEV_AUTH_FALLBACK=true``: the
       ``X-User-Id`` header, or the single existing user when no header is sent.

    The dev fallback is off by default, so a deployment that never sets the flag
    cannot be driven by a client-asserted header. Whatever the source, the id
    must be a valid UUID that exists in ``users`` or the request is rejected.
    """

    candidate = getattr(request.state, "user_id", None)

    if candidate is None and settings.allow_dev_auth_fallback:
        candidate = x_user_id
        if candidate is None:
            owner_ids = list(
                session.scalars(select(User.id).order_by(User.created_at.asc()).limit(2)).all()
            )
            if len(owner_ids) == 1:
                candidate = owner_ids[0]
            elif len(owner_ids) > 1:
                raise _unauthorized(
                    "Authenticated user required; multiple owner users exist. "
                    "Send X-User-Id or use the real session."
                )
            else:
                raise _unauthorized("Authenticated user required; no owner users exist yet.")

    if candidate is None:
        raise _unauthorized("Authenticated user required")

    try:
        owner_user_id = UUID(str(candidate))
    except ValueError as exc:
        raise _unauthorized("Invalid authenticated user id") from exc

    if session.get(User, owner_user_id) is None:
        raise _unauthorized("Authenticated user was not found")

    return owner_user_id
