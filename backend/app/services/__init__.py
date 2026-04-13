"""Services layer — business logic above the CRUD layer.

Public surface:

* :func:`ingest_conversation` — full conversation ingestion pipeline
* :class:`IngestionResult` — typed return value from the pipeline
* :class:`EmbeddingProvider` — protocol for plugging in a RAG embedding backend
* :class:`IdentityResolver` — resolves the active speaker into a stored person
"""

from .conversation_ingestion import ingest_conversation
from .embedding import EmbeddingProvider
from .identity_resolver import IdentityResolutionResult, IdentityResolver, IdentitySignal
from ..schema.ingestion import IngestionResult

__all__ = [
    "ingest_conversation",
    "IngestionResult",
    "EmbeddingProvider",
    "IdentityResolver",
    "IdentityResolutionResult",
    "IdentitySignal",
]
