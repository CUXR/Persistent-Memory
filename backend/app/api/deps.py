from collections.abc import Generator
from functools import lru_cache
from uuid import UUID

from fastapi import Header
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.crud.memory_store import MemoryStore


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


@lru_cache
def _memory_store_for_owner(owner_user_id: UUID) -> MemoryStore:
    """Build (and cache) one MemoryStore engine per owner.

    The project has no auth/session layer yet — it targets a single wearer
    per deployment (see scripts/live_ingestion.py). The owner id is passed
    explicitly via header until real authentication is added.
    """

    store = MemoryStore(owner_user_id=owner_user_id)
    store.initialize(create_schema=False)
    return store


def get_memory_store(x_owner_user_id: UUID = Header(..., alias="X-Owner-User-Id")) -> MemoryStore:
    """Resolve the MemoryStore bound to the requesting owner."""

    return _memory_store_for_owner(x_owner_user_id)
