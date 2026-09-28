from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..deps import get_authenticated_owner_user_id, get_db_session
from ...crud.user import get_user_by_id
from ...schema.user import UserRead

router = APIRouter()


@router.get("/me", response_model=UserRead)
def get_current_user(
    owner_user_id: Annotated[UUID, Depends(get_authenticated_owner_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
) -> UserRead:
    """Return the authenticated owner user record for the frontend shell."""

    user = get_user_by_id(session, owner_user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user was not found",
        )
    return UserRead.model_validate(user)
