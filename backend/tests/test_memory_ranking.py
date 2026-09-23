"""
Tests for ranked memory retrieval (GitHub issue #30).

Acceptance Criteria:
  - Memories persist per-person with source encounter metadata
  - API returns bounded, ranked relevant memories for individuals
  - Retrieval logic tested for ranking, recency, and output limits
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from uuid import uuid4

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.models.user import User
from app.services import memory_ranking


# ── Pure scoring function tests ─────────────────────────────


class TestRecencyScore:
    def test_zero_age_is_full_score(self):
        now = datetime.now(timezone.utc)
        assert memory_ranking.recency_score(now, now) == pytest.approx(1.0)

    def test_half_life_halves_score(self):
        now = datetime.now(timezone.utc)
        anchor = now - timedelta(days=14)
        assert memory_ranking.recency_score(anchor, now, half_life_days=14.0) == pytest.approx(0.5)

    def test_older_scores_lower(self):
        now = datetime.now(timezone.utc)
        recent = memory_ranking.recency_score(now - timedelta(days=1), now)
        old = memory_ranking.recency_score(now - timedelta(days=100), now)
        assert recent > old

    def test_future_anchor_clamped_to_full_score(self):
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=10)
        assert memory_ranking.recency_score(future, now) == pytest.approx(1.0)

    def test_naive_datetime_treated_as_utc(self):
        now = datetime.now(timezone.utc)
        naive_anchor = (now - timedelta(days=14)).replace(tzinfo=None)
        assert memory_ranking.recency_score(naive_anchor, now, half_life_days=14.0) == pytest.approx(0.5)

    def test_non_positive_half_life_rejected(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError):
            memory_ranking.recency_score(now, now, half_life_days=0)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert memory_ranking.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert memory_ranking.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert memory_ranking.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_mismatched_length_returns_zero(self):
        assert memory_ranking.cosine_similarity([1.0, 0.0], [1.0]) == 0.0

    def test_empty_vector_returns_zero(self):
        assert memory_ranking.cosine_similarity([], [1.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert memory_ranking.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestScoreMemory:
    def test_without_similarity_redistributes_weight(self):
        score = memory_ranking.score_memory(recency=1.0, relevance=0.0, similarity=None)
        assert score == pytest.approx(
            memory_ranking.DEFAULT_RECENCY_WEIGHT
            / (memory_ranking.DEFAULT_RECENCY_WEIGHT + memory_ranking.DEFAULT_RELEVANCE_WEIGHT)
        )

    def test_with_similarity_uses_all_three_weights(self):
        score = memory_ranking.score_memory(recency=1.0, relevance=1.0, similarity=1.0)
        assert score == pytest.approx(1.0)

    def test_higher_similarity_scores_higher(self):
        low = memory_ranking.score_memory(recency=0.5, relevance=0.5, similarity=0.0)
        high = memory_ranking.score_memory(recency=0.5, relevance=0.5, similarity=1.0)
        assert high > low


class TestRelativeTimePhrase:
    @pytest.mark.parametrize(
        "delta,expected",
        [
            (timedelta(hours=1), "today"),
            (timedelta(days=1, hours=2), "yesterday"),
            (timedelta(days=3), "3 days ago"),
            (timedelta(days=14), "2 weeks ago"),
            (timedelta(days=60), "2 months ago"),
            (timedelta(days=400), "1 year ago"),
        ],
    )
    def test_buckets(self, delta, expected):
        now = datetime.now(timezone.utc)
        assert memory_ranking.relative_time_phrase(now - delta, now) == expected


class TestBuildInteractionContext:
    def test_no_prior_encounters(self):
        now = datetime.now(timezone.utc)
        ctx = memory_ranking.build_interaction_context(first_met_at=None, first_met_summary=None, now=now)
        assert "no prior encounters" in ctx.lower()

    def test_includes_relative_time_and_summary(self):
        now = datetime.now(timezone.utc)
        ctx = memory_ranking.build_interaction_context(
            first_met_at=now - timedelta(days=14),
            first_met_summary="Met at Cornell XR demo",
            now=now,
        )
        assert "2 weeks ago" in ctx
        assert "Cornell XR demo" in ctx

    def test_falls_back_without_summary(self):
        now = datetime.now(timezone.utc)
        ctx = memory_ranking.build_interaction_context(
            first_met_at=now - timedelta(days=1), first_met_summary="", now=now
        )
        assert ctx == "First met yesterday."


# ── MemoryStore.get_relevant_memories integration tests ─────


def _make_store() -> MemoryStore:
    owner_id = uuid4()
    store = MemoryStore("sqlite+pysqlite:///:memory:", owner_user_id=owner_id)
    store.initialize()
    with store.Session() as session:
        with session.begin():
            session.add(User(
                id=owner_id,
                first_name="Test",
                last_name="Owner",
                display_name="Test Owner",
                username="test-owner",
            ))
    return store


@pytest.fixture
def store():
    s = _make_store()
    yield s
    s.close()


@pytest.fixture
def emily(store: MemoryStore):
    return store.upsert_person(name="Emily Chen")


class TestGetRelevantMemories:
    def test_nonexistent_person_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.get_relevant_memories(uuid4())

    def test_bad_limit_rejected(self, store, emily):
        with pytest.raises(ValueError, match="limit"):
            store.get_relevant_memories(emily.id, limit=0)

    def test_empty_profile(self, store, emily):
        result = store.get_relevant_memories(emily.id)
        assert result.facts == []
        assert result.summaries == []
        assert "no prior encounters" in result.interaction_context.lower()

    def test_output_is_bounded_by_limit(self, store, emily):
        for i in range(10):
            store.write_fact(person_id=emily.id, fact_text=f"fact {i}", confidence=0.5)

        result = store.get_relevant_memories(emily.id, limit=3)
        assert len(result.facts) == 3

    def test_more_recent_fact_outranks_older_at_equal_confidence(self, store, emily):
        now = datetime.now(timezone.utc)
        store.write_fact(
            person_id=emily.id, fact_text="old", confidence=0.8, valid_from=now - timedelta(days=180)
        )
        store.write_fact(
            person_id=emily.id, fact_text="new", confidence=0.8, valid_from=now - timedelta(days=1)
        )

        result = store.get_relevant_memories(emily.id, limit=5)
        assert result.facts[0].fact_text == "new"
        assert result.facts[0].score > result.facts[1].score

    def test_higher_confidence_outranks_lower_at_equal_recency(self, store, emily):
        now = datetime.now(timezone.utc)
        store.write_fact(
            person_id=emily.id, fact_text="low conf", confidence=0.1, valid_from=now - timedelta(days=5)
        )
        store.write_fact(
            person_id=emily.id, fact_text="high conf", confidence=0.95, valid_from=now - timedelta(days=5)
        )

        result = store.get_relevant_memories(emily.id, limit=5)
        assert result.facts[0].fact_text == "high conf"

    def test_summary_relevance_uses_linked_episode_importance(self, store, emily):
        now = datetime.now(timezone.utc)
        important_episode = store.write_episode(
            time_start=now - timedelta(days=5),
            time_end=now - timedelta(days=5),
            participants=[emily.id],
            importance_score=0.95,
        )
        trivial_episode = store.write_episode(
            time_start=now - timedelta(days=5),
            time_end=now - timedelta(days=5),
            participants=[emily.id],
            importance_score=0.05,
        )
        store.write_summary(
            person_id=emily.id,
            summary_text="life-changing news",
            episode_time_end=now - timedelta(days=5),
            episode_id=important_episode,
        )
        store.write_summary(
            person_id=emily.id,
            summary_text="small talk",
            episode_time_end=now - timedelta(days=5),
            episode_id=trivial_episode,
        )

        result = store.get_relevant_memories(emily.id, limit=5)
        assert result.summaries[0].summary_text == "life-changing news"

    def test_summary_without_episode_gets_neutral_relevance(self, store, emily):
        sid = store.write_summary(person_id=emily.id, summary_text="standalone summary")
        result = store.get_relevant_memories(emily.id, limit=5)
        assert result.summaries[0].id == sid
        assert result.summaries[0].relevance_score == pytest.approx(memory_ranking.NEUTRAL_RELEVANCE)

    def test_query_embedding_boosts_similar_fact(self, store, emily):
        now = datetime.now(timezone.utc)
        store.write_fact(
            person_id=emily.id,
            fact_text="matches the query",
            confidence=0.5,
            valid_from=now - timedelta(days=30),
            embedding=[1.0, 0.0, 0.0],
        )
        store.write_fact(
            person_id=emily.id,
            fact_text="unrelated but more recent",
            confidence=0.5,
            valid_from=now - timedelta(days=1),
            embedding=[0.0, 1.0, 0.0],
        )

        result = store.get_relevant_memories(emily.id, limit=5, query_embedding=[1.0, 0.0, 0.0])
        assert result.facts[0].fact_text == "matches the query"
        assert result.facts[0].similarity_score == pytest.approx(1.0)
        assert result.facts[1].similarity_score == pytest.approx(0.0)

    def test_fact_without_stored_embedding_has_no_similarity_score(self, store, emily):
        store.write_fact(person_id=emily.id, fact_text="no embedding", confidence=0.5)
        result = store.get_relevant_memories(emily.id, limit=5, query_embedding=[1.0, 0.0])
        assert result.facts[0].similarity_score is None

    def test_interaction_context_reflects_first_episode(self, store, emily):
        now = datetime.now(timezone.utc)
        store.write_episode(
            time_start=now - timedelta(days=14),
            time_end=now - timedelta(days=14),
            summary="Met at Cornell XR demo",
            participants=[emily.id],
        )
        store.write_episode(
            time_start=now - timedelta(days=1),
            time_end=now - timedelta(days=1),
            summary="Follow-up chat",
            participants=[emily.id],
        )

        result = store.get_relevant_memories(emily.id)
        assert "2 weeks ago" in result.interaction_context
        assert "Cornell XR demo" in result.interaction_context

    def test_write_episode_persists_importance_score(self, store, emily):
        now = datetime.now(timezone.utc)
        eid = store.write_episode(
            time_start=now, time_end=now, participants=[emily.id], importance_score=0.42
        )
        with store.Session() as session:
            from app.models.episode import Episode
            episode = session.get(Episode, eid)
            assert float(episode.importance_score) == pytest.approx(0.42)

    def test_bad_importance_score_rejected(self, store, emily):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError):
            store.write_episode(
                time_start=now, time_end=now, participants=[emily.id], importance_score=1.5
            )
