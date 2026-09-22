"""
Tests for the memory extraction pipeline (GitHub issue #29).

Acceptance Criteria:
  - Extraction service returns structured, typed memory candidates from a transcript
  - Data model for all four memory types (facts, preferences, relationships, events) defined
  - Tests execute against fixtures without calling live LLM

All LLM interaction is mocked — no network calls are made.
"""

from pathlib import Path
import sys
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schema.memory_extraction import (
    DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
    MemoryExtractionLLMResponse,
    RawEvent,
    RawFact,
    RawPreference,
    RawRelationship,
)
from app.services.llm_client import LLMClient
from app.services.memory_extraction import extract_memory_candidates, normalize_extraction

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcripts"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


# ── Data model tests (facts, preferences, relationships, events) ──


class TestSchemaValidation:
    def test_fact_confidence_must_be_in_range(self):
        with pytest.raises(ValidationError):
            RawFact(fact_text="likes climbing", category="hobby", confidence=1.5)

    def test_fact_category_must_be_known_value(self):
        with pytest.raises(ValidationError):
            RawFact(fact_text="likes climbing", category="not_a_category", confidence=0.9)

    def test_preference_polarity_must_be_known_value(self):
        with pytest.raises(ValidationError):
            RawPreference(preference_text="loves coffee", polarity="obsessed", confidence=0.9)

    def test_relationship_requires_target_and_relation(self):
        rel = RawRelationship(target_name="Sam Rivera", relation="works_with", confidence=0.8)
        assert rel.target_name == "Sam Rivera"
        assert rel.relation == "works_with"

    def test_event_occurred_at_is_optional(self):
        event = RawEvent(event_text="finished a triathlon", confidence=0.7)
        assert event.occurred_at is None

    def test_extraction_response_defaults_to_empty_lists(self):
        response = MemoryExtractionLLMResponse()
        assert response.facts == []
        assert response.preferences == []
        assert response.relationships == []
        assert response.events == []


# ── normalize_extraction() — pure function, no I/O ────────────


class TestNormalizeExtraction:
    def test_maps_all_four_types(self):
        raw = MemoryExtractionLLMResponse(
            facts=[RawFact(fact_text="works at Cornell XR", category="affiliation", confidence=0.9)],
            preferences=[RawPreference(preference_text="loves climbing", polarity="like", confidence=0.85)],
            relationships=[RawRelationship(target_name="Sam Rivera", relation="works_with", confidence=0.8)],
            events=[RawEvent(event_text="finished a triathlon", occurred_at="last weekend", confidence=0.75)],
        )

        result = normalize_extraction(raw)

        assert len(result.facts) == 1
        assert result.facts[0].fact_text == "works at Cornell XR"
        assert result.facts[0].category == "affiliation"

        assert len(result.preferences) == 1
        assert result.preferences[0].pref_text == "loves climbing"
        assert result.preferences[0].polarity == "like"

        assert len(result.relationships) == 1
        assert result.relationships[0].target_name == "Sam Rivera"
        assert result.relationships[0].relation == "works_with"

        assert len(result.events) == 1
        assert result.events[0].event_text == "finished a triathlon"
        assert result.events[0].occurred_at == "last weekend"

    @pytest.mark.parametrize(
        "confidence,expected_needs_review",
        [(0.9, False), (0.5, False), (0.49, True), (0.1, True)],
    )
    def test_needs_review_flag_at_default_threshold(self, confidence, expected_needs_review):
        raw = MemoryExtractionLLMResponse(
            facts=[RawFact(fact_text="likes climbing", category="hobby", confidence=confidence)]
        )
        result = normalize_extraction(raw)
        assert result.facts[0].needs_review is expected_needs_review
        assert result.facts[0].confidence == confidence

    def test_custom_review_threshold(self):
        raw = MemoryExtractionLLMResponse(
            facts=[RawFact(fact_text="likes climbing", category="hobby", confidence=0.6)]
        )
        result = normalize_extraction(raw, review_confidence_threshold=0.7)
        assert result.facts[0].needs_review is True

    def test_whitespace_is_normalized(self):
        raw = MemoryExtractionLLMResponse(
            facts=[RawFact(fact_text="  likes   climbing\n\n", category="hobby", confidence=0.9)]
        )
        result = normalize_extraction(raw)
        assert result.facts[0].fact_text == "likes climbing"

    def test_blank_text_items_are_dropped(self):
        raw = MemoryExtractionLLMResponse(
            facts=[RawFact(fact_text="   ", category="hobby", confidence=0.9)],
            preferences=[RawPreference(preference_text="", polarity="like", confidence=0.9)],
            relationships=[RawRelationship(target_name="  ", relation="knows", confidence=0.9)],
            events=[RawEvent(event_text="\t", confidence=0.9)],
        )
        result = normalize_extraction(raw)
        assert result.facts == []
        assert result.preferences == []
        assert result.relationships == []
        assert result.events == []

    def test_blank_occurred_at_becomes_none(self):
        raw = MemoryExtractionLLMResponse(
            events=[RawEvent(event_text="ran a marathon", occurred_at="   ", confidence=0.9)]
        )
        result = normalize_extraction(raw)
        assert result.events[0].occurred_at is None

    def test_empty_response_yields_empty_result(self):
        result = normalize_extraction(MemoryExtractionLLMResponse())
        assert result.facts == []
        assert result.preferences == []
        assert result.relationships == []
        assert result.events == []


# ── extract_memory_candidates() — mocked LLM, fixture transcripts ──


class TestExtractMemoryCandidates:
    @pytest.fixture
    def mock_llm_client(self):
        client = AsyncMock(spec=LLMClient)
        client.extract_memories.return_value = MemoryExtractionLLMResponse(
            facts=[
                RawFact(fact_text="works at Cornell XR", category="affiliation", confidence=0.95),
                RawFact(fact_text="prefers climbing to running", category="hobby", confidence=0.4),
            ],
            preferences=[
                RawPreference(preference_text="loves climbing", polarity="like", confidence=0.85),
            ],
            relationships=[
                RawRelationship(target_name="Sam Rivera", relation="works_with", confidence=0.8),
            ],
            events=[
                RawEvent(event_text="finished a triathlon", occurred_at="last weekend", confidence=0.9),
                RawEvent(event_text="starting a new job", occurred_at="next March", confidence=0.7),
            ],
        )
        return client

    @pytest.mark.asyncio
    async def test_returns_structured_typed_candidates(self, mock_llm_client):
        transcript = _load_fixture("cornell_xr_demo.txt")

        result = await extract_memory_candidates(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            llm_client=mock_llm_client,
        )

        assert len(result.facts) == 2
        assert len(result.preferences) == 1
        assert len(result.relationships) == 1
        assert len(result.events) == 2

    @pytest.mark.asyncio
    async def test_low_confidence_item_flagged_for_review(self, mock_llm_client):
        transcript = _load_fixture("cornell_xr_demo.txt")

        result = await extract_memory_candidates(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            llm_client=mock_llm_client,
        )

        by_text = {f.fact_text: f for f in result.facts}
        assert by_text["works at Cornell XR"].needs_review is False
        assert by_text["prefers climbing to running"].needs_review is True

    @pytest.mark.asyncio
    async def test_calls_llm_with_transcript_and_names(self, mock_llm_client):
        transcript = _load_fixture("cornell_xr_demo.txt")

        await extract_memory_candidates(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            llm_client=mock_llm_client,
        )

        mock_llm_client.extract_memories.assert_awaited_once_with(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
        )

    @pytest.mark.asyncio
    async def test_empty_transcript_short_circuits_without_calling_llm(self, mock_llm_client):
        transcript = _load_fixture("empty.txt")

        result = await extract_memory_candidates(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            llm_client=mock_llm_client,
        )

        assert result.facts == []
        assert result.preferences == []
        assert result.relationships == []
        assert result.events == []
        mock_llm_client.extract_memories.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_small_talk_transcript_can_yield_no_candidates(self):
        transcript = _load_fixture("small_talk.txt")
        client = AsyncMock(spec=LLMClient)
        client.extract_memories.return_value = MemoryExtractionLLMResponse()

        result = await extract_memory_candidates(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Casey",
            llm_client=client,
        )

        assert result.facts == []
        assert result.preferences == []
        assert result.relationships == []
        assert result.events == []

    @pytest.mark.asyncio
    async def test_null_parsed_result_raises(self):
        client = AsyncMock(spec=LLMClient)
        client.extract_memories.side_effect = ValueError(
            "OpenAI returned a null parsed result for memory extraction"
        )

        with pytest.raises(ValueError, match="null parsed result"):
            await extract_memory_candidates(
                transcript="Alex: hi\nJordan: hi",
                wearer_name="Alex",
                interlocutor_name="Jordan",
                llm_client=client,
            )

    @pytest.mark.asyncio
    async def test_custom_review_threshold_propagates(self, mock_llm_client):
        transcript = _load_fixture("cornell_xr_demo.txt")

        result = await extract_memory_candidates(
            transcript=transcript,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            llm_client=mock_llm_client,
            review_confidence_threshold=0.9,
        )

        # Only the 0.95-confidence fact clears a 0.9 threshold.
        flagged = {f.fact_text for f in result.facts if f.needs_review}
        assert "prefers climbing to running" in flagged
        assert "works at Cornell XR" not in flagged

    def test_default_review_threshold_constant(self):
        assert DEFAULT_REVIEW_CONFIDENCE_THRESHOLD == 0.5
