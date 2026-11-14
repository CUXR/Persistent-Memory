from fastapi import FastAPI

from .api.routes import briefing, user
from .core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(user.router, prefix="/users", tags=["users"])
# Before any other /people router: a route like /people/{person_id} without a
# :uuid converter would otherwise capture /people/briefing.
app.include_router(briefing.router, prefix="/people", tags=["briefings"])
