"""Owner-scoped person briefings, mounted under /people.

Contract and fixtures: docs/contracts/briefings/.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_briefing_service
from app.crud.briefing_queries import InvalidCursorError, PersonNotFoundError
from app.schema.briefing import Brief, BriefResult, FactCategory, FactPageResponse
from app.services.briefing import DEFAULT_MAX_FACTS, MAX_PAGE_SIZE, BriefingService

router = APIRouter()


def _page_size():
    # A fresh Query per parameter: FastAPI binds a shared instance to the first name.
    return Query(DEFAULT_MAX_FACTS, ge=1, le=MAX_PAGE_SIZE, description="Facts per category on this page")


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")


@router.get("/briefing", response_model=BriefResult)
def brief_by_name(
    name: str = Query(..., min_length=1, max_length=200), max_facts: int = _page_size(),
    service: BriefingService = Depends(get_briefing_service),
):
    """Look someone up by name: a brief for one match, candidates for several."""
    if not name.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="name is blank")
    return service.brief_by_name(name, max_facts=max_facts)


# `:uuid` keeps these paths from matching words like "briefing" as a person ID.
@router.get("/{person_id:uuid}/briefing", response_model=Brief)
def brief_person(
    person_id: UUID, max_facts: int = _page_size(),
    service: BriefingService = Depends(get_briefing_service),
) -> Brief:
    """Everything to remember about a known person. The client decides what to show."""
    try:
        return service.brief(person_id, max_facts=max_facts)
    except PersonNotFoundError as exc:
        raise _not_found() from exc


@router.get("/{person_id:uuid}/facts", response_model=FactPageResponse)
def person_facts(
    person_id: UUID, category: FactCategory,
    cursor: str | None = Query(None, min_length=1, max_length=1000), limit: int = _page_size(),
    service: BriefingService = Depends(get_briefing_service),
) -> FactPageResponse:
    """One page of a category: the first page, or the next one after `cursor`."""
    try:
        return service.facts(person_id, category, cursor=cursor, limit=limit)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor") from exc
    except PersonNotFoundError as exc:
        raise _not_found() from exc
