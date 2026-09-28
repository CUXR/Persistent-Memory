from functools import lru_cache
import json

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors_origins(value: object) -> list[str]:
    """Parse a comma-separated string or JSON list of origins into a clean list."""

    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        text = str(value or "").strip()
        if text.startswith("["):
            decoded = json.loads(text)
            if not isinstance(decoded, list):
                raise ValueError("cors_allowed_origins JSON value must be a list")
            items = [str(item) for item in decoded]
        else:
            items = text.split(",")
    return [item.strip() for item in items if item.strip()]


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
    retrieval_embedding_dimension: int = 1024
    retrieval_bi_encoder_model: str = "BAAI/bge-m3"
    retrieval_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_bi_encoder_top_k: int = 100
    retrieval_bi_encoder_min_score: float = 0.0
    retrieval_reranker_top_k: int = 20
    retrieval_reranker_min_score: float = 0.0
    db_echo: bool = False

    # ------------------------------------------------------------------ #
    # Memory browser API (frontend integration)                           #
    # ------------------------------------------------------------------ #
    # Local development only: when true, requests may identify the owner
    # through the ``X-User-Id`` header, or fall back to the single existing
    # user when no header is sent. Never enable this on a shared deployment;
    # the real session middleware sets ``request.state.user_id`` instead.
    allow_dev_auth_fallback: bool = False
    # Browser origins allowed to call the API, as a comma-separated string or a
    # JSON list, e.g. ``http://localhost:5173,http://127.0.0.1:5173``. Kept as a
    # plain string because pydantic-settings 2.1 JSON-decodes ``list`` fields
    # at the environment layer, before any validator can split on commas.
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("cors_allowed_origins")
    @classmethod
    def _validate_cors_origins(cls, value: str) -> str:
        origins = parse_cors_origins(value)
        if "*" in origins:
            raise ValueError(
                "cors_allowed_origins must list explicit origins; '*' is not allowed with credentials"
            )
        return ",".join(origins)

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """The configured CORS origins as a list."""

        return parse_cors_origins(self.cors_allowed_origins)

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
