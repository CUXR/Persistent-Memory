from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded exclusively from environment variables / .env.

    Fields with **no default** are required — the application will refuse to
    start if they are absent from the environment.  Fields with a default are
    optional tuning knobs whose values are unlikely to differ between
    deployments; override them in .env when needed.

    See ``.env.example`` for the full list of supported variables.
    """

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #
    app_name: str = "Persistent Memory"

    # ------------------------------------------------------------------ #
    # Database — required, no default                                     #
    # ------------------------------------------------------------------ #
    database_url: str  # e.g. postgresql+psycopg://user:pass@host:5432/db

    # ------------------------------------------------------------------ #
    # Embeddings                                                           #
    # ------------------------------------------------------------------ #
    embedding_dimension: int = 512
    db_echo: bool = False

    # ------------------------------------------------------------------ #
    # OpenAI / LLM — api key required; model and retries have defaults    #
    # ------------------------------------------------------------------ #
    openai_api_key: str  # required — set OPENAI_API_KEY in .env
    openai_model: str = "gpt-4o-mini"
    openai_max_retries: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
