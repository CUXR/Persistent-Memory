from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..deps import get_authenticated_owner_user_id, get_db_session
from ...schema.browser import ConversationDetailOut, RecentConversationOut
from ...services.browser_service import get_conversation_detail, list_recent_conversations

router = APIRouter()


@router.get("/recent", response_model=list[RecentConversationOut])
def get_recent_conversations(
    owner_user_id: Annotated[UUID, Depends(get_authenticated_owner_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[RecentConversationOut]:
    """Return the authenticated user's most recent conversation episodes."""

    return list_recent_conversations(session, owner_user_id, limit=limit)


@router.get("/{episode_id}", response_model=ConversationDetailOut)
def get_conversation(
    episode_id: UUID,
    owner_user_id: Annotated[UUID, Depends(get_authenticated_owner_user_id)],
    session: Annotated[Session, Depends(get_db_session)],
) -> ConversationDetailOut:
    """Return one conversation episode with linked participant detail cards."""

    try:
        return get_conversation_detail(session, owner_user_id, episode_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from exc
