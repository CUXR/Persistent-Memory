from fastapi import FastAPI

from .core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)
