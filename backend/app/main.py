from fastapi import FastAPI

from app.api.routes import users
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(users.router, prefix="/users", tags=["users"])
