"""Owner-scoped person briefings for the glasses and the phone.

Contract and fixtures: docs/contracts/briefings/.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_briefing_service
from app.crud.briefing_queries import InvalidCursorError, PersonNotFoundError
from app.schema.briefing import Brief, BriefResult, DisplaySettings, FactPageResponse
from app.services.briefing import DEFAULT_MAX_FACTS, MAX_PAGE_SIZE, BriefingService

router = APIRouter()


def _page_size():
    # A fresh Query per parameter: FastAPI binds a shared instance to the first name.
    return Query(DEFAULT_MAX_FACTS, ge=1, le=MAX_PAGE_SIZE, description="Facts per category on this page")


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")


@router.get("/people/{person_id}", response_model=Brief)
def brief_person(
    person_id: UUID, max_facts: int = _page_size(),
    service: BriefingService = Depends(get_briefing_service),
) -> Brief:
    """Face path: the person in view. Honors the owner's display mode; never visual descriptors."""
    try:
        return service.brief(person_id, max_facts=max_facts)
    except PersonNotFoundError as exc:
        raise _not_found() from exc


@router.get("/by-name", response_model=BriefResult)
def brief_by_name(
    name: str = Query(..., min_length=1, max_length=200), max_facts: int = _page_size(),
    service: BriefingService = Depends(get_briefing_service),
):
    """Name path: a person the wearer asked about. Always the full brief."""
    if not name.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="name is blank")
    return service.brief_by_name(name, max_facts=max_facts)


@router.get("/facts", response_model=FactPageResponse)
def more_facts(
    cursor: str = Query(..., min_length=1, max_length=1000), limit: int = _page_size(),
    service: BriefingService = Depends(get_briefing_service),
) -> FactPageResponse:
    """Next page of one category, from a `next_cursor`."""
    try:
        return service.more_facts(cursor, limit=limit)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor") from exc
    except PersonNotFoundError as exc:
        raise _not_found() from exc


@router.get("/display-mode", response_model=DisplaySettings)
def get_display_mode(service: BriefingService = Depends(get_briefing_service)) -> DisplaySettings:
    return DisplaySettings(display_mode=service.display_mode())


@router.put("/display-mode", response_model=DisplaySettings)
def put_display_mode(
    body: DisplaySettings, service: BriefingService = Depends(get_briefing_service),
) -> DisplaySettings:
    return DisplaySettings(display_mode=service.set_display_mode(body.display_mode))
