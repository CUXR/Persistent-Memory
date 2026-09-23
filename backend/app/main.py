from fastapi import FastAPI

from .api.routes import memory, user
from .core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(user.router, prefix="/users", tags=["users"])
app.include_router(memory.router, prefix="/people", tags=["memory"])
