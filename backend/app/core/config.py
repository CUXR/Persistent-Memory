from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "Persistent Memory"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/persistent_memory"
    embedding_dimension: int = 512
    retrieval_embedding_dimension: int = 1024
    retrieval_bi_encoder_model: str = "BAAI/bge-m3"
    retrieval_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_bi_encoder_top_k: int = 100
    retrieval_bi_encoder_min_score: float = 0.0
    retrieval_reranker_top_k: int = 20
    retrieval_reranker_min_score: float = 0.0
    db_echo: bool = False

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
