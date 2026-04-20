"""Tests for the conversation ingestion pipeline.

All LLM calls are mocked — no real OpenAI network traffic is made.  The store
uses an in-memory SQLite database, mirroring the pattern in test_memory_store.py.

Coverage:
  - Person creation when interlocutor is unknown
  - Person reuse when interlocutor is already stored
  - Episode and per-person summary written correctly
  - Facts extracted from LLM response are persisted
  - Duplicate facts (case-insensitive) are skipped, not re-written
  - Intra-batch duplicate facts are caught before the second write
  - Shared-interest flag is respected (facts still written)
  - Edges written when the target person exists in the store
  - Edges skipped (with log warning) when the target is unknown
  - Both LLM calls are awaited concurrently (asyncio.gather)
  - IngestionResult shape has all expected fields
  - Embedding provider hook is called when supplied
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from uuid import UUID

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.models.user import UserFact
from app.schema.ingestion import (
    EpisodeSummaryLLMResponse,
    ExtractedEdge,
    ExtractedFact,
    FactExtractionLLMResponse,
    IngestionResult,
)
from app.services.conversation_ingestion import ingest_conversation
from app.services.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 16, 10, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 3, 16, 10, 30, 0, tzinfo=timezone.utc)

TRANSCRIPT = (
    "Alice: Hi Bob, good to see you!\n"
    "Bob: Hey Alice! I've been enjoying rock climbing lately.\n"
    "Alice: Me too! I also do bouldering.\n"
    "Bob: I work at Acme Corp as a senior engineer.\n"
)


def _summary_response(summary: str = "A friendly catch-up.", importance: float = 0.3) -> EpisodeSummaryLLMResponse:
    return EpisodeSummaryLLMResponse(summary=summary, importance_score=importance)


def _facts_response(
    facts: Optional[list[ExtractedFact]] = None,
    edges: Optional[list[ExtractedEdge]] = None,
) -> FactExtractionLLMResponse:
    return FactExtractionLLMResponse(
        facts=facts or [],
        edges=edges or [],
    )


def _mock_llm(
    summary: EpisodeSummaryLLMResponse | None = None,
    extraction: FactExtractionLLMResponse | None = None,
) -> LLMClient:
    """Return a LLMClient with both async methods mocked."""
    client = MagicMock(spec=LLMClient)
    client.generate_summary = AsyncMock(return_value=summary or _summary_response())
    client.extract_facts = AsyncMock(return_value=extraction or _facts_response())
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    """Fresh in-memory SQLite MemoryStore for each test."""
    s = MemoryStore("sqlite+pysqlite:///:memory:")
    s.initialize()
    yield s
    s.close()


@pytest.fixture()
def bob(store: MemoryStore):
    """Pre-existing interlocutor 'Bob Smith'."""
    return store.upsert_person(name="Bob Smith", aliases=["Bobby"])


@pytest.fixture()
def alice_facts(store: MemoryStore):
    """Seed user (wearer) facts so shared-interest detection has material."""
    with store.Session() as session:
        with session.begin():
            from sqlalchemy import select
            from app.models.user import User
            owner = session.scalar(select(User).limit(1))
            if owner is None:
                # trigger owner creation via any store method first
                pass
    # Use a write to trigger owner creation
    store.upsert_person(name="__seed__", aliases=[])
    with store.Session() as session:
        with session.begin():
            from sqlalchemy import select
            from app.models.user import User
            owner = session.scalar(select(User).limit(1))
            session.add(UserFact(user_id=owner.id, fact_text="Enjoys rock climbing"))
            session.add(UserFact(user_id=owner.id, fact_text="Likes hiking"))


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPersonResolution:
    """Person creation and reuse behaviour."""

    @pytest.mark.asyncio
    async def test_creates_person_when_unknown(self, store: MemoryStore):
        """A new person record is created when the interlocutor has no prior entry."""
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(),
        )
        resolved = store.resolve_person_by_name_or_alias("Bob Smith")
        assert resolved is not None
        assert resolved.id == result.person_id

    @pytest.mark.asyncio
    async def test_reuses_existing_person(self, store: MemoryStore, bob):
        """An existing person is reused — no duplicate is created."""
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(),
        )
        assert result.person_id == bob.id

    @pytest.mark.asyncio
    async def test_alias_resolution(self, store: MemoryStore, bob):
        """Person is resolved via alias, not just display name."""
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bobby",  # alias of bob
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(),
        )
        assert result.person_id == bob.id


class TestEpisodeAndSummary:
    """Episode record and per-person summary slice are written correctly."""

    @pytest.mark.asyncio
    async def test_episode_written_with_correct_content(self, store: MemoryStore, bob):
        llm = _mock_llm(summary=_summary_response("They discussed climbing and work."))
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=llm,
        )
        assert isinstance(result.episode_id, UUID)
        assert result.summary == "They discussed climbing and work."

    @pytest.mark.asyncio
    async def test_person_summary_slice_written(self, store: MemoryStore, bob):
        """A Summary row is created for the interlocutor linked to the episode."""
        from sqlalchemy import select
        from app.models.memory import Summary

        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(summary=_summary_response("Quick chat.")),
        )
        with store.Session() as session:
            summaries = session.scalars(
                select(Summary).where(Summary.person_id == bob.id)
            ).all()
        assert len(summaries) == 1
        assert summaries[0].summary_text == "Quick chat."
        assert summaries[0].episode_id == result.episode_id

    @pytest.mark.asyncio
    async def test_importance_score_propagated(self, store: MemoryStore, bob):
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(summary=_summary_response(importance=0.85)),
        )
        assert result.importance_score == pytest.approx(0.85)


class TestFactExtraction:
    """Facts from the LLM are persisted; duplicates are skipped."""

    @pytest.mark.asyncio
    async def test_new_facts_written(self, store: MemoryStore, bob):
        facts = [
            ExtractedFact(fact_text="Likes rock climbing", confidence=0.9, category="hobby"),
            ExtractedFact(fact_text="Works at Acme Corp", confidence=0.95, category="affiliation"),
        ]
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(extraction=_facts_response(facts=facts)),
        )
        assert len(result.facts_written) == 2
        assert result.facts_skipped_as_duplicate == []

    @pytest.mark.asyncio
    async def test_duplicate_fact_skipped(self, store: MemoryStore, bob):
        """A fact already in the store is not re-written (case-insensitive match)."""
        # Pre-store a fact for bob
        store.write_fact(person_id=bob.id, fact_text="Likes rock climbing", confidence=0.9)

        facts = [
            ExtractedFact(fact_text="likes rock climbing", confidence=0.8, category="hobby"),
            ExtractedFact(fact_text="Works at Acme Corp", confidence=0.9, category="affiliation"),
        ]
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(extraction=_facts_response(facts=facts)),
        )
        assert len(result.facts_written) == 1
        assert len(result.facts_skipped_as_duplicate) == 1
        assert result.facts_skipped_as_duplicate[0] == "likes rock climbing"

    @pytest.mark.asyncio
    async def test_intra_batch_duplicate_skipped(self, store: MemoryStore, bob):
        """Two identical facts in the same LLM response are deduplicated."""
        facts = [
            ExtractedFact(fact_text="Enjoys hiking", confidence=0.9, category="hobby"),
            ExtractedFact(fact_text="ENJOYS HIKING", confidence=0.7, category="hobby"),
        ]
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(extraction=_facts_response(facts=facts)),
        )
        assert len(result.facts_written) == 1
        assert len(result.facts_skipped_as_duplicate) == 1

    @pytest.mark.asyncio
    async def test_shared_interest_fact_still_written(self, store: MemoryStore, bob):
        """A shared-interest fact is written to the store (not merely flagged)."""
        facts = [
            ExtractedFact(
                fact_text="Loves rock climbing",
                confidence=0.9,
                category="hobby",
                is_shared_interest=True,
            ),
        ]
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(extraction=_facts_response(facts=facts)),
        )
        assert len(result.facts_written) == 1


class TestEdges:
    """Relationship edges are written when targets exist; skipped otherwise."""

    @pytest.mark.asyncio
    async def test_edge_written_for_known_target(self, store: MemoryStore, bob):
        """Edge is created when the target person is already in the store."""
        acme = store.upsert_person(name="Acme Corp", aliases=[])
        edges = [
            ExtractedEdge(relation="works_at", target_name="Acme Corp", confidence=0.95),
        ]
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(extraction=_facts_response(edges=edges)),
        )
        assert len(result.edges_written) == 1

    @pytest.mark.asyncio
    async def test_edge_skipped_for_unknown_target(self, store: MemoryStore, bob, caplog):
        """Edge is silently skipped (with a warning) when the target is unknown."""
        edges = [
            ExtractedEdge(relation="knows", target_name="Unknown Person XYZ", confidence=0.5),
        ]
        with caplog.at_level(logging.WARNING, logger="app.services.conversation_ingestion"):
            result = await ingest_conversation(
                transcript=TRANSCRIPT,
                wearer_name="Alice",
                interlocutor_name="Bob Smith",
                time_start=_T0,
                time_end=_T1,
                store=store,
                llm_client=_mock_llm(extraction=_facts_response(edges=edges)),
            )
        assert result.edges_written == []
        assert any("Unknown Person XYZ" in r.message for r in caplog.records)


class TestConcurrency:
    """LLM calls are dispatched concurrently via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_both_llm_calls_awaited(self, store: MemoryStore, bob):
        """Both generate_summary and extract_facts are called exactly once."""
        llm = _mock_llm()
        await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=llm,
        )
        llm.generate_summary.assert_awaited_once()
        llm.extract_facts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_receives_wearer_facts_context(self, store: MemoryStore, alice_facts, bob):
        """extract_facts is called with the wearer's stored facts as context."""
        llm = _mock_llm()
        await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=llm,
        )
        _, kwargs = llm.extract_facts.call_args
        # user_facts is passed as a keyword argument
        user_facts = kwargs.get("user_facts", llm.extract_facts.call_args[0][4] if llm.extract_facts.call_args[0] else [])
        assert any("rock climbing" in f.lower() for f in user_facts)


class TestIngestionResult:
    """IngestionResult has all expected fields and correct types."""

    @pytest.mark.asyncio
    async def test_result_shape(self, store: MemoryStore, bob):
        facts = [ExtractedFact(fact_text="Enjoys hiking", confidence=0.8, category="hobby")]
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(extraction=_facts_response(facts=facts)),
        )
        assert isinstance(result, IngestionResult)
        assert isinstance(result.episode_id, UUID)
        assert isinstance(result.person_id, UUID)
        assert isinstance(result.summary, str)
        assert 0.0 <= result.importance_score <= 1.0
        assert isinstance(result.facts_written, list)
        assert isinstance(result.facts_skipped_as_duplicate, list)
        assert isinstance(result.edges_written, list)


class TestEmbeddingProviderHook:
    """EmbeddingProvider is invoked when supplied and gracefully skipped otherwise."""

    @pytest.mark.asyncio
    async def test_embedding_provider_called_when_supplied(self, store: MemoryStore, bob):
        provider = MagicMock()
        provider.embed = AsyncMock(return_value=[[0.1] * 512])

        await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(),
            embedding_provider=provider,
        )
        provider.embed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_error_without_embedding_provider(self, store: MemoryStore, bob):
        """Pipeline completes normally when no embedding provider is given."""
        result = await ingest_conversation(
            transcript=TRANSCRIPT,
            wearer_name="Alice",
            interlocutor_name="Bob Smith",
            time_start=_T0,
            time_end=_T1,
            store=store,
            llm_client=_mock_llm(),
            embedding_provider=None,
        )
        assert result.episode_id is not None


class TestGetUserFacts:
    """MemoryStore.get_user_facts() returns the wearer's fact texts."""

    def test_returns_empty_when_no_facts(self, store: MemoryStore):
        # Trigger owner creation
        store.upsert_person(name="Seed", aliases=[])
        assert store.get_user_facts() == []

    def test_returns_stored_facts(self, store: MemoryStore):
        store.upsert_person(name="Seed", aliases=[])
        with store.Session() as session:
            with session.begin():
                from sqlalchemy import select
                from app.models.user import User
                owner = session.scalar(select(User).limit(1))
                session.add(UserFact(user_id=owner.id, fact_text="Loves hiking"))
                session.add(UserFact(user_id=owner.id, fact_text="Plays guitar"))

        facts = store.get_user_facts()
        assert "Loves hiking" in facts
        assert "Plays guitar" in facts
        assert len(facts) == 2
