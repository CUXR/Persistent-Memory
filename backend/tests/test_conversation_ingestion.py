"""Exercise ingestion against the real memory store with mocked LLM responses."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.crud.memory_store import MemoryStore
from app.models.episode import Episode
from app.models.user import User
from app.schema.ingestion import (
    EpisodeSummaryLLMResponse,
    ExtractedEdge,
    ExtractedFact,
    FactExtractionLLMResponse,
)
from app.services.conversation_ingestion import ingest_conversation
from app.services.llm_client import LLMClient


@pytest.fixture
def store():
    owner_id = uuid4()
    store = MemoryStore("sqlite+pysqlite:///:memory:", owner_user_id=owner_id)
    store.initialize()
    try:
        with store.Session.begin() as session:
            session.add(User(id=owner_id, first_name="Alex", username="test-owner"))
        yield store
    finally:
        store.close()


@pytest.fixture
def llm_client():
    client = Mock(spec=LLMClient)
    client.generate_summary.return_value = EpisodeSummaryLLMResponse(
        summary="Alex and Jordan discussed rock climbing.", importance_score=0.5
    )
    client.extract_facts.return_value = FactExtractionLLMResponse(
        facts=[ExtractedFact(fact_text="Enjoys rock climbing", confidence=0.9, category="hobby")]
    )
    return client


def ingest(store, llm_client, name="Jordan Lee"):
    start = datetime(2026, 9, 26, tzinfo=timezone.utc)
    return asyncio.run(ingest_conversation(
        transcript="Alex: What do you enjoy?\nJordan: Rock climbing.",
        wearer_name="Alex",
        interlocutor_name=name,
        time_start=start,
        time_end=start + timedelta(minutes=1),
        store=store,
        llm_client=llm_client,
    ))


def test_new_person_conversation_is_persisted(store, llm_client):
    result = ingest(store, llm_client)

    person = store.resolve_person_by_name("Jordan Lee")
    assert person is not None
    assert result.person_id == person.id
    assert len(store.list_people()) == 1
    with store.Session() as session:
        episode = session.get(Episode, result.episode_id)
        assert episode.user_id == store.owner_user_id
        assert episode.person_id == person.id
        assert "Rock climbing" in episode.transcript
        assert episode.dialogue_summary == result.summary
    profile = store.get_profile_context(person.id)
    assert [fact.id for fact in profile.facts] == result.facts_written
    assert profile.facts[0].episode_id == result.episode_id
    assert profile.summaries[0].episode_id == result.episode_id
    llm_client.generate_summary.assert_awaited_once()
    llm_client.extract_facts.assert_awaited_once()


def test_existing_person_is_reused_by_name_and_facts_are_deduplicated(store, llm_client):
    first = ingest(store, llm_client)
    second = ingest(store, llm_client, name="  JORDAN LEE  ")

    assert second.person_id == first.person_id
    assert second.episode_id != first.episode_id
    assert len(store.list_people()) == 1
    assert second.facts_written == []
    assert second.facts_skipped_as_duplicate == ["Enjoys rock climbing"]
    profile = store.get_profile_context(first.person_id)
    assert len(profile.facts) == 1
    assert len(profile.summaries) == 2
    assert llm_client.extract_facts.call_args.args[3] == ["Enjoys rock climbing"]


@pytest.mark.parametrize("target_name, should_link", [
    ("  PRIYA SHARMA  ", True),
    ("Priya", False),
    ("Unknown Person", False),
])
def test_relationship_targets_resolve_only_by_full_name(store, llm_client, target_name, should_link):
    target = store.upsert_person("Priya Sharma")
    llm_client.extract_facts.return_value = FactExtractionLLMResponse(
        edges=[ExtractedEdge(relation="knows", target_name=target_name, confidence=0.9)]
    )

    result = ingest(store, llm_client)

    edges = store.get_profile_context(result.person_id).edges_from
    assert len(edges) == int(should_link)
    assert [edge.id for edge in edges] == result.edges_written
    if should_link:
        assert edges[0].dst_id == target.id
        assert edges[0].episode_id == result.episode_id
    assert len(store.list_people()) == 2
