"""EmbeddingProvider protocol — the extensibility hook for RAG semantic search.

No implementation is shipped in v1. Future providers (OpenAI text-embedding-3,
local sentence-transformers, etc.) implement this protocol and are injected into
:func:`~app.services.conversation_ingestion.ingest_conversation` via the
``embedding_provider`` parameter.

Example future usage::

    class OpenAIEmbeddingProvider:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            response = await client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return [d.embedding for d in response.data]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async interface for text embedding backends.

    Implementations must be stateless with respect to individual calls —
    each :meth:`embed` invocation should be independently retryable.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, same length and order as *texts*.
        """
        ...
