from collections.abc import Generator
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, sessionmaker

from app.core import database
from app.core.database import get_db
from app.crud.api_tokens import owner_for_token
from app.crud.memory_store import MemoryStore
from app.services.briefing import BriefingService

_bearer = HTTPBearer(auto_error=False)


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_session_factory() -> sessionmaker[Session]:
    """The app-wide session factory. Tests override this dependency."""
    if database.SessionLocal is None:
        raise RuntimeError("Database driver not available for configured database_url")
    return database.SessionLocal


def get_current_owner(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> UUID:
    """Derive the owner from a bearer token. This is the one owner-scoping seam.

    Clients never send an owner ID; every owner-scoped route depends on this.
    """
    owner_id = None
    if credentials is not None and credentials.scheme.lower() == "bearer":
        with session_factory() as session:
            owner_id = owner_for_token(session, credentials.credentials)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return owner_id


def get_briefing_service(
    owner_id: UUID = Depends(get_current_owner),
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> BriefingService:
    store = MemoryStore.from_session_factory(session_factory, owner_user_id=owner_id)
    return BriefingService(store)
