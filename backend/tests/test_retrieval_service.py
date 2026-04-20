from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.services.retrieval_service import retrieve_person_context


class StaticScorer:
    """Deterministic scorer used to test ranking behavior without ML models."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores_by_text = scores_by_text

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self._scores_by_text.get(text, 0.0) for text in texts]


@pytest.fixture
def store():
    s = MemoryStore("sqlite+pysqlite:///:memory:")
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def people(store: MemoryStore):
    emily = store.upsert_person(
        name="Emily Chen",
        aliases=["Em"],
        persona90=[0.25] * 90,
    )
    john = store.upsert_person(
        name="John Rivera",
        aliases=["Johnny"],
    )
    return emily, john


def test_retrieve_person_context_returns_ranked_facts_summaries_and_edges(store: MemoryStore, people):
    emily, john = people
    now = datetime.now(timezone.utc)
    episode_id = store.write_episode(
        time_start=now - timedelta(minutes=20),
        time_end=now - timedelta(minutes=10),
        transcript="Emily talked about tennis and her work with John.",
        summary="Discussion covered tennis and a work project.",
        participants=[emily.id, john.id],
    )

    fact_text = "Emily plays tennis every Saturday"
    fact_candidate_text = f"hobby {fact_text}"
    summary_text = "Emily planned a tennis session this weekend."
    edge_text = "colleague John Rivera"

    store.write_fact(
        person_id=emily.id,
        fact_text=fact_text,
        fact_category="hobby",
        source=episode_id,
    )
    store.write_fact(person_id=emily.id, fact_text="Emily has a dog named Max", fact_category="biographical")
    store.write_pref(person_id=emily.id, pref_text="Favorite drink: oat milk latte", episode_id=episode_id)
    store.write_summary(person_id=emily.id, summary_text=summary_text, episode_id=episode_id)
    store.write_edge(src_id=emily.id, relation="colleague", dst_id=john.id, confidence=0.9, episode_id=episode_id)

    scorer = StaticScorer(
        {
            fact_candidate_text: 0.95,
            summary_text: 0.90,
            edge_text: 0.85,
        }
    )
    result = retrieve_person_context(
        emily.id,
        "What do we know about Emily's tennis plans and coworkers?",
        store=store,
        bi_encoder=scorer,
        reranker=scorer,
    )

    assert result.person_id == emily.id
    assert [fact.fact_text for fact in result.facts] == [fact_text]
    assert [summary.summary_text for summary in result.summaries] == [summary_text]
    assert [edge.dst_name for edge in result.edges] == ["John Rivera"]
    assert "prefs" not in result.model_dump(mode="json")


def test_retrieve_person_context_uses_reranker_for_final_cut(store: MemoryStore, people):
    emily, john = people

    fact_text = "Emily is training for a marathon"
    fact_candidate_text = f"hobby {fact_text}"
    summary_text = "Emily discussed an upcoming marathon training block."
    edge_text = "teammate John Rivera"

    store.write_fact(person_id=emily.id, fact_text=fact_text, fact_category="hobby")
    store.write_summary(person_id=emily.id, summary_text=summary_text)
    store.write_edge(src_id=emily.id, relation="teammate", dst_id=john.id)

    bi_encoder = StaticScorer(
        {
            fact_candidate_text: 0.99,
            summary_text: 0.98,
            edge_text: 0.97,
        }
    )
    reranker = StaticScorer(
        {
            summary_text: 0.91,
            edge_text: 0.89,
            fact_candidate_text: 0.10,
        }
    )
    result = retrieve_person_context(
        emily.id,
        "Who is Emily training with for the marathon?",
        store=store,
        bi_encoder=bi_encoder,
        reranker=reranker,
        bi_encoder_top_k=3,
        reranker_top_k=2,
    )

    assert result.facts == []
    assert [summary.summary_text for summary in result.summaries] == [summary_text]
    assert [edge.relation for edge in result.edges] == ["teammate"]


def test_retrieve_person_context_blank_query_returns_available_memories(store: MemoryStore, people):
    emily, john = people

    store.write_fact(person_id=emily.id, fact_text="Emily likes sailing", fact_category="hobby")
    store.write_summary(person_id=emily.id, summary_text="Emily discussed weekend sailing plans.")
    store.write_edge(src_id=emily.id, relation="friend", dst_id=john.id)

    result = retrieve_person_context(emily.id, "   ", store=store)

    assert len(result.facts) == 1
    assert len(result.summaries) == 1
    assert len(result.edges) == 1
