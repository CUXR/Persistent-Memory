from collections.abc import Generator
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session, sessionmaker

from app.core import database
from app.core.database import get_db
from app.crud.memory_store import MemoryStore
from app.models.user import User
from app.services.briefing import BriefingService


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_session_factory() -> sessionmaker[Session]:
    """The app-wide session factory. Tests override this dependency."""
    if database.SessionLocal is None:
        raise RuntimeError("Database driver not available for configured database_url")
    return database.SessionLocal


def get_current_owner(
    x_owner_id: str | None = Header(None, alias="X-Owner-Id"),
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> UUID:
    """The requesting owner. Every owner-scoped route depends on this.

    Development stand-in (#35): the client names its owner in the X-Owner-Id
    header, which is not authentication. Real user accounts replace this
    function, and routes stay unchanged.
    """
    try:
        owner_id = UUID(x_owner_id) if x_owner_id else None
    except ValueError:
        owner_id = None
    if owner_id is not None:
        with session_factory() as session:
            if session.get(User, owner_id) is None:
                owner_id = None
    if owner_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="X-Owner-Id must name an existing user")
    return owner_id


def get_briefing_service(
    owner_id: UUID = Depends(get_current_owner),
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> BriefingService:
    store = MemoryStore.from_session_factory(session_factory, owner_user_id=owner_id)
    return BriefingService(store)
