"""Issue, verify, and revoke bearer tokens. Raw tokens are never stored."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import secrets
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.api_token import ApiToken
from ..models.user import User

TOKEN_PREFIX = "pm_"


def _hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def issue_token(session: Session, user_id: UUID, label: str) -> tuple[ApiToken, str]:
    """Create a token for an existing user. The raw token is returned only here."""
    if session.get(User, user_id) is None:
        raise ValueError(f"User id={user_id} not found")
    if not label.strip():
        raise ValueError("A token label is required")
    raw = TOKEN_PREFIX + secrets.token_urlsafe(32)
    token = ApiToken(user_id=user_id, label=label.strip(), token_hash=_hash(raw))
    session.add(token)
    session.flush()
    return token, raw


def owner_for_token(session: Session, raw_token: str) -> UUID | None:
    """Return the owner of an active token, or None if it is unknown or revoked."""
    token = session.scalar(select(ApiToken).where(ApiToken.token_hash == _hash(raw_token)))
    if token is None or token.revoked_at is not None:
        return None
    return token.user_id


def list_tokens(session: Session, user_id: UUID) -> list[ApiToken]:
    return list(session.scalars(
        select(ApiToken).where(ApiToken.user_id == user_id).order_by(ApiToken.created_at)
    ))


def revoke_token(session: Session, token_id: UUID) -> bool:
    """Revoke a token by ID. Returns False if no such token exists."""
    token = session.get(ApiToken, token_id)
    if token is None:
        return False
    if token.revoked_at is None:
        token.revoked_at = datetime.now(timezone.utc)
    return True


__all__ = ["TOKEN_PREFIX", "issue_token", "list_tokens", "owner_for_token", "revoke_token"]
