from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..deps import get_authenticated_owner_user_id, get_db_session
from ...schema.browser import PeopleDirectoryOut, PersonContextOut, PersonProfileOut
from ...services.browser_service import (
    get_person_profile,
    get_person_retrieval_context,
    list_people_directory,
)

router = APIRouter()


@router.get("", response_model=PeopleDirectoryOut)
def get_people(
    owner_user_id: Annotated[UUID, Depends(get_authenticated_owner_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=250)] = 100,
    query: Annotated[str | None, Query(max_length=200)] = None,
) -> PeopleDirectoryOut:
    """Return the authenticated user's people directory."""

    return list_people_directory(session, owner_user_id, limit=limit, query=query)


@router.get("/{person_id}/profile", response_model=PersonProfileOut)
def get_people_profile(
    person_id: UUID,
    owner_user_id: Annotated[UUID, Depends(get_authenticated_owner_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
) -> PersonProfileOut:
    """Return one person's stored profile bundle for the authenticated owner."""

    try:
        return get_person_profile(session, owner_user_id, person_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found",
        ) from exc


@router.get("/{person_id}/context", response_model=PersonContextOut)
def get_people_context(
    person_id: UUID,
    owner_user_id: Annotated[UUID, Depends(get_authenticated_owner_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
    query: Annotated[str, Query(min_length=1, max_length=400)],
) -> PersonContextOut:
    """Return relevant stored memory for one person and one user query."""

    try:
        return get_person_retrieval_context(session, owner_user_id, person_id, query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found",
        ) from exc
