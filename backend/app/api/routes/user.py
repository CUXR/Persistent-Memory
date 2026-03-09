from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db_session
from ...crud.user import get_or_create_wearer
from ...schema.user import WearerStateRead

router = APIRouter()


@router.get("/wearer-state", response_model=WearerStateRead)
def read_wearer_state(db: Session = Depends(get_db_session)) -> WearerStateRead:
    wearer = get_or_create_wearer(db)
    return WearerStateRead(wearer_person_id=wearer.id)
