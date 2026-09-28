"""Service-layer helpers for backend application workflows."""

from ..schema.ingestion import IngestionResult
from .browser_service import (
    get_conversation_detail,
    get_person_profile,
    get_person_retrieval_context,
    list_people_directory,
    list_recent_conversations,
)
from .conversation_ingestion import ingest_conversation
from .embedding import EmbeddingProvider
from .retrieval_service import retrieve_person_context

__all__ = [
    "EmbeddingProvider",
    "IngestionResult",
    "get_conversation_detail",
    "get_person_profile",
    "get_person_retrieval_context",
    "ingest_conversation",
    "list_people_directory",
    "list_recent_conversations",
    "retrieve_person_context",
]
