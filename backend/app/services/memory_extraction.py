"""Memory extraction pipeline — transcript in, typed memory candidates out (issue #29).

Pure extraction: calls the LLM once for structured output covering all four
memory types, then validates and normalizes the raw response into
:class:`~app.schema.memory_extraction.MemoryExtractionResult`. Persisting
candidates to the store is a separate concern (see
app/services/conversation_ingestion.py and app/crud/memory_store.py).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..schema.memory_extraction import (
    DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
    EventCandidate,
    FactCandidate,
    MemoryExtractionLLMResponse,
    MemoryExtractionResult,
    PreferenceCandidate,
    RelationshipCandidate,
)
from .llm_client import LLMClient

logger = logging.getLogger("app.services.memory_extraction")


def _clean(text: str) -> str:
    """Collapse internal whitespace and strip; used for text field normalization."""
    return re.sub(r"\s+", " ", text.strip())


async def extract_memory_candidates(
    transcript: str,
    wearer_name: str,
    interlocutor_name: str,
    llm_client: Optional[LLMClient] = None,
    review_confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> MemoryExtractionResult:
    """Extract typed memory candidates about ``interlocutor_name`` from a transcript.

    Args:
        transcript: Raw conversation in ``Speaker: utterance`` per-line format.
        wearer_name: Display name of the smart-glasses wearer.
        interlocutor_name: Display name of the conversation partner being
            extracted about.
        llm_client: Optional pre-configured :class:`~app.services.llm_client.LLMClient`.
            A default client is constructed from application settings when omitted.
        review_confidence_threshold: Candidates with ``confidence`` strictly
            below this value are flagged ``needs_review=True`` instead of
            being silently trusted.

    Returns:
        A :class:`~app.schema.memory_extraction.MemoryExtractionResult` with
        empty lists when the transcript has no content worth extracting.
    """

    if not transcript.strip():
        logger.debug("extract_memory_candidates: empty transcript, skipping LLM call")
        return MemoryExtractionResult()

    client = llm_client or LLMClient()

    raw = await client.extract_memories(
        transcript=transcript,
        wearer_name=wearer_name,
        interlocutor_name=interlocutor_name,
    )

    result = normalize_extraction(raw, review_confidence_threshold)

    logger.info(
        "extract_memory_candidates: %d facts, %d preferences, %d relationships, %d events "
        "(%d flagged for review)",
        len(result.facts),
        len(result.preferences),
        len(result.relationships),
        len(result.events),
        sum(
            c.needs_review
            for group in (result.facts, result.preferences, result.relationships, result.events)
            for c in group
        ),
    )
    return result


def normalize_extraction(
    raw: MemoryExtractionLLMResponse,
    review_confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> MemoryExtractionResult:
    """Validate and normalize a raw LLM response into typed candidates.

    Pure function (no I/O) so it can be exercised directly against fixture
    data without mocking the LLM client. Drops items whose primary text
    field is empty after whitespace-normalization; flags the rest for
    review when ``confidence < review_confidence_threshold``.
    """

    return MemoryExtractionResult(
        facts=[
            FactCandidate(
                fact_text=text,
                category=f.category,
                confidence=f.confidence,
                needs_review=f.confidence < review_confidence_threshold,
            )
            for f in raw.facts
            if (text := _clean(f.fact_text))
        ],
        preferences=[
            PreferenceCandidate(
                pref_text=text,
                polarity=p.polarity,
                confidence=p.confidence,
                needs_review=p.confidence < review_confidence_threshold,
            )
            for p in raw.preferences
            if (text := _clean(p.preference_text))
        ],
        relationships=[
            RelationshipCandidate(
                target_name=name,
                relation=_clean(r.relation),
                confidence=r.confidence,
                needs_review=r.confidence < review_confidence_threshold,
            )
            for r in raw.relationships
            if (name := _clean(r.target_name)) and _clean(r.relation)
        ],
        events=[
            EventCandidate(
                event_text=text,
                occurred_at=_clean(e.occurred_at) if e.occurred_at and _clean(e.occurred_at) else None,
                confidence=e.confidence,
                needs_review=e.confidence < review_confidence_threshold,
            )
            for e in raw.events
            if (text := _clean(e.event_text))
        ],
    )


__all__ = ["extract_memory_candidates", "normalize_extraction"]
