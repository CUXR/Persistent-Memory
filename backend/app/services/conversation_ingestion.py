"""Conversation ingestion pipeline — the main public entry-point for this module.

Accepts a raw transcript string, drives two concurrent LLM calls (summary and
fact extraction), deduplicates candidate facts against the existing store, then
persists everything through :class:`~app.crud.memory_store.MemoryStore`.

Transcript format
-----------------
Each turn on its own line, speaker followed by a colon::

    Alice: Hey, how's the project going?
    Bob: Really well — we shipped the first milestone yesterday.

Extensibility hooks
-------------------
* **EmbeddingProvider** — inject an async embedding backend to generate and
  store vector representations of the transcript and extracted facts, enabling
  semantic/RAG retrieval in a future iteration.
* **LLMClient** — pass a custom :class:`~app.services.llm_client.LLMClient`
  instance (e.g. pointing at a different model or a test double) via the
  ``llm_client`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional
from uuid import UUID

from ..crud.memory_store import MemoryStore
from ..schema.ingestion import IngestionResult
from .embedding import EmbeddingProvider
from .llm_client import LLMClient

logger = logging.getLogger("app.services.conversation_ingestion")


def _normalise(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace for dedup comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


async def ingest_conversation(
    transcript: str,
    wearer_name: str,
    interlocutor_name: str,
    time_start: datetime,
    time_end: datetime,
    store: MemoryStore,
    interlocutor_aliases: Optional[list[str]] = None,
    llm_client: Optional[LLMClient] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> IngestionResult:
    """Ingest a conversation transcript into persistent memory.

    Orchestrates the full pipeline:

    1. Resolve or create the interlocutor :class:`~app.models.person.Person`.
    2. Load existing profile context and wearer facts for LLM context.
    3. Concurrently call the LLM for an episode summary and fact extraction.
    4. Write the episode record and a person-level summary slice.
    5. Deduplicate and write new facts; write resolved relationship edges.
    6. Optionally generate and store embeddings when a provider is supplied.

    Args:
        transcript: Raw conversation in ``Speaker: utterance`` per-line format.
        wearer_name: Display name of the smart-glasses wearer (used in prompts).
        interlocutor_name: Primary conversation partner's name.  Used for person
            resolution and LLM context.
        time_start: UTC start time of the conversation episode.
        time_end: UTC end time of the conversation episode.
        store: Initialised :class:`~app.crud.memory_store.MemoryStore` bound to
            the wearer's owner account.
        interlocutor_aliases: Optional extra names to register on the person
            record (only applied when a *new* person is created).
        llm_client: Optional pre-configured :class:`~app.services.llm_client.LLMClient`.
            A default client is constructed from application settings when omitted.
        embedding_provider: Optional async embedding backend.  When supplied,
            embeddings are generated for the transcript and each new fact and
            passed to the provider.  No-op in v1 (no vector columns on these
            tables yet; hook is present for forward compatibility).

    Returns:
        :class:`~app.schema.ingestion.IngestionResult` summarising what was
        written to the store.

    Raises:
        ValueError: If the store returns an unexpected error during persistence.
    """

    client = llm_client or LLMClient()

    # ------------------------------------------------------------------
    # 1. Resolve or create the interlocutor
    # ------------------------------------------------------------------
    person_out = store.resolve_person_by_name_or_alias(interlocutor_name)
    if person_out is None:
        logger.info("ingest_conversation: creating new person '%s'", interlocutor_name)
        person_out = store.upsert_person(
            name=interlocutor_name,
            aliases=interlocutor_aliases or [],
        )
    else:
        logger.info("ingest_conversation: resolved existing person id=%s", person_out.id)

    person_id: UUID = person_out.id

    # ------------------------------------------------------------------
    # 2. Load existing context for LLM deduplication / shared-interest prompts
    # ------------------------------------------------------------------
    profile = store.get_profile_context(person_id)
    existing_fact_texts: list[str] = [f.fact_text for f in profile.facts]
    user_fact_texts: list[str] = store.get_user_facts()

    # ------------------------------------------------------------------
    # 3. Concurrent LLM calls — summary and fact extraction are independent
    # ------------------------------------------------------------------
    logger.debug("ingest_conversation: dispatching concurrent LLM calls")
    summary_response, extraction_response = await asyncio.gather(
        client.generate_summary(transcript, wearer_name, interlocutor_name),
        client.extract_facts(
            transcript,
            wearer_name,
            interlocutor_name,
            existing_fact_texts,
            user_fact_texts,
        ),
    )

    # ------------------------------------------------------------------
    # 4. Persist the episode and person-level summary
    # ------------------------------------------------------------------
    episode_id = store.write_episode(
        time_start=time_start,
        time_end=time_end,
        transcript=transcript,
        summary=summary_response.summary,
        participants=[person_id],
    )
    logger.debug("ingest_conversation: wrote episode id=%s", episode_id)

    store.write_summary(
        person_id=person_id,
        summary_text=summary_response.summary,
        episode_time_start=time_start,
        episode_time_end=time_end,
        episode_id=episode_id,
    )

    # ------------------------------------------------------------------
    # 5. Deduplicate and write facts
    # ------------------------------------------------------------------
    existing_normalised = {_normalise(t) for t in existing_fact_texts}
    facts_written: list[UUID] = []
    facts_skipped: list[str] = []

    for extracted in extraction_response.facts:
        if _normalise(extracted.fact_text) in existing_normalised:
            logger.debug("ingest_conversation: skipping duplicate fact '%s'", extracted.fact_text)
            facts_skipped.append(extracted.fact_text)
            continue

        fact_id = store.write_fact(
            person_id=person_id,
            fact_text=extracted.fact_text,
            confidence=extracted.confidence,
            fact_category=extracted.category,
            episode_id=episode_id,
        )
        facts_written.append(fact_id)
        # Add to the seen set so duplicates within the same batch are also caught
        existing_normalised.add(_normalise(extracted.fact_text))

    logger.debug(
        "ingest_conversation: %d facts written, %d skipped as duplicates",
        len(facts_written),
        len(facts_skipped),
    )

    # ------------------------------------------------------------------
    # 6. Write relationship edges (skip edges to unknown targets)
    # ------------------------------------------------------------------
    edges_written: list[UUID] = []

    for extracted_edge in extraction_response.edges:
        target = store.resolve_person_by_name_or_alias(extracted_edge.target_name)
        if target is None:
            logger.warning(
                "ingest_conversation: skipping edge '%s' -> '%s' (target not found in store)",
                interlocutor_name,
                extracted_edge.target_name,
            )
            continue

        edge_id = store.write_edge(
            src_id=person_id,
            relation=extracted_edge.relation,
            dst_id=target.id,
            confidence=extracted_edge.confidence,
            episode_id=episode_id,
        )
        edges_written.append(edge_id)

    logger.debug("ingest_conversation: %d edges written", len(edges_written))

    # ------------------------------------------------------------------
    # 7. Optional embedding generation (RAG extensibility hook)
    # ------------------------------------------------------------------
    if embedding_provider is not None:
        texts_to_embed = [transcript] + [
            f.fact_text
            for f in extraction_response.facts
            if _normalise(f.fact_text) not in {_normalise(s) for s in facts_skipped}
        ]
        embeddings = await embedding_provider.embed(texts_to_embed)
        # v1: vectors are generated but not yet persisted — no embedding columns
        # exist on Episode or PersonFact yet.  Log the shapes for observability.
        logger.info(
            "ingest_conversation: generated %d embeddings (dim=%d) — "
            "storage deferred to a future migration",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
        )

    # ------------------------------------------------------------------
    # 8. Return result summary
    # ------------------------------------------------------------------
    return IngestionResult(
        episode_id=episode_id,
        person_id=person_id,
        summary=summary_response.summary,
        importance_score=summary_response.importance_score,
        facts_written=facts_written,
        facts_skipped_as_duplicate=facts_skipped,
        edges_written=edges_written,
    )
