"""
Tests for the MemoryStore service.

Issue Acceptance Criterion:
  Script can insert mock data via MemoryStore
  Profile context retrieval works
  No raw SQL used outside MemoryStore
  All functions typed (enforced by Pydantic)
  Input validation catches bad data
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
from app.models.memory import Edge, Summary
from app.models.person import PersonFact
from app.models.user import User


def _make_store() -> MemoryStore:
    """Create an initialized MemoryStore with a pre-seeded owner user."""

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
    """Fresh in-memory database for each test."""

    s = _make_store()
    yield s
    s.close()


@pytest.fixture
def emily(store: MemoryStore):
    """Create a test person named Emily."""

    return store.upsert_person(
        name="Emily Chen",
        face_key="face_emily_001",
        voice_key="voice_emily_001",
        persona90=[0.5] * 90,
    )


@pytest.fixture
def john(store: MemoryStore):
    """Create a second test person named John."""

    return store.upsert_person(
        name="John Rivera",
        face_key="face_john_001",
        voice_key="voice_john_001",
    )


class TestUpsertPerson:
    def test_insert_new_person(self, store, emily):
        assert emily.id
        assert emily.name == "Emily Chen"
        assert emily.face_key == "face_emily_001"
        assert emily.voice_key == "voice_emily_001"
        assert len(emily.persona90) == 90

    def test_upsert_updates_existing(self, store, emily):
        updated = store.upsert_person(
            name="Emily Chen",
            face_key="face_emily_002",
        )
        assert updated.id == emily.id
        assert updated.face_key == "face_emily_002"

    def test_persona90_must_be_0_or_90(self, store):
        with pytest.raises(ValueError, match="0 or 90"):
            store.upsert_person(name="Bad Person", persona90=[1.0] * 50)

    def test_empty_name_rejected(self, store):
        with pytest.raises(ValueError):
            store.upsert_person(name="")


class TestWriteEpisode:
    def test_write_episode_with_participants(self, store, emily, john):
        now = datetime.now(timezone.utc)
        eid = store.write_episode(
            time_start=now,
            time_end=now + timedelta(minutes=5),
            transcript="Emily: Hi John!\nJohn: Hey Emily!",
            summary="Emily greeted John.",
            participants=[emily.id, john.id],
        )
        assert eid

    def test_write_episode_requires_participant(self, store):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="requires at least one participant"):
            store.write_episode(
                time_start=now,
                time_end=now + timedelta(minutes=1),
                participants=[],
            )

    def test_end_before_start_rejected(self, store):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="time_end"):
            store.write_episode(
                time_start=now,
                time_end=now - timedelta(minutes=1),
                participants=[uuid4()],
            )


class TestWriteFact:
    def test_write_and_read_fact(self, store, emily):
        fid = store.write_fact(
            person_id=emily.id,
            fact_text="the user is a student born in 2013",
            confidence=0.95,
        )
        assert fid

        profile = store.get_profile_context(emily.id)
        assert len(profile.facts) == 1
        assert profile.facts[0].fact_text == "the user is a student born in 2013"
        assert profile.facts[0].confidence == 0.95

    def test_temporal_fact(self, store, emily):
        now = datetime.now(timezone.utc)
        store.write_fact(
            person_id=emily.id,
            fact_text="user lives in Beijing",
            valid_from=now - timedelta(days=365),
            valid_to=now,
        )
        profile = store.get_profile_context(emily.id)
        assert profile.facts[0].valid_to is not None

    def test_bad_confidence_rejected(self, store, emily):
        with pytest.raises(ValueError, match="confidence"):
            store.write_fact(
                person_id=emily.id,
                fact_text="test",
                confidence=1.5,
            )

    def test_nonexistent_person_rejected(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.write_fact(person_id=uuid4(), fact_text="test")


class TestWriteSummary:
    def test_write_summary(self, store, emily):
        sid = store.write_summary(
            person_id=emily.id,
            summary_text="2024-05-13: the user asked about swimming.",
            episode_time_start=datetime(2024, 5, 13, 10, 0),
            episode_time_end=datetime(2024, 5, 13, 10, 15),
        )
        assert sid

        profile = store.get_profile_context(emily.id)
        assert len(profile.summaries) == 1


class TestWriteEdge:
    def test_write_edge(self, store, emily, john):
        eid = store.write_edge(
            src_id=emily.id,
            relation="colleague",
            dst_id=john.id,
            confidence=0.9,
        )
        assert eid

        profile = store.get_profile_context(emily.id)
        assert len(profile.edges_from) == 1
        assert profile.edges_from[0].relation == "colleague"
        assert profile.edges_from[0].dst_name == "John Rivera"

    def test_upsert_edge_updates_confidence(self, store, emily, john):
        store.write_edge(
            src_id=emily.id, relation="colleague", dst_id=john.id, confidence=0.5
        )
        store.write_edge(
            src_id=emily.id, relation="colleague", dst_id=john.id, confidence=0.95
        )
        profile = store.get_profile_context(emily.id)
        assert len(profile.edges_from) == 1
        assert profile.edges_from[0].confidence == 0.95

    def test_self_edge_rejected(self, store, emily):
        with pytest.raises(ValueError, match="self-edges"):
            store.write_edge(
                src_id=emily.id, relation="friend", dst_id=emily.id
            )


class TestProfileContext:
    def test_full_profile(self, store, emily, john):
        now = datetime.now(timezone.utc)
        eid = store.write_episode(
            time_start=now - timedelta(minutes=10),
            time_end=now,
            transcript="Emily asked about tennis",
            summary="Discussion about sports",
            participants=[emily.id, john.id],
        )

        store.write_fact(
            person_id=emily.id,
            fact_text="the user is a student born in 2013",
            episode_id=eid,
        )
        store.write_fact(
            person_id=emily.id,
            fact_text="the user has a pet dog named Max",
            confidence=0.8,
        )
        store.write_summary(
            person_id=emily.id,
            summary_text="2024-05-13: the user asked about swimming.",
            episode_time_start=now - timedelta(minutes=10),
            episode_time_end=now,
            episode_id=eid,
        )
        store.write_edge(
            src_id=emily.id,
            relation="colleague",
            dst_id=john.id,
            confidence=0.9,
        )

        profile = store.get_profile_context(emily.id)

        assert len(profile.facts) == 2
        assert len(profile.summaries) == 1
        assert len(profile.edges_from) == 1
        assert len(profile.persona90) == 90

        dumped = profile.model_dump(mode="json")
        assert isinstance(dumped["facts"], list)
        assert isinstance(dumped["persona90"], list)

    def test_empty_profile(self, store, emily):
        profile = store.get_profile_context(emily.id)
        assert profile.facts == []
        assert profile.summaries == []
        assert profile.edges_from == []
        assert len(profile.persona90) == 90

    def test_nonexistent_person_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.get_profile_context(uuid4())


class TestLifecycleAndErrors:
    def test_session_access_before_initialize_raises(self):
        store = MemoryStore("sqlite+pysqlite:///:memory:", owner_user_id=uuid4())
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = store.Session

    def test_close_is_idempotent(self):
        store = MemoryStore("sqlite+pysqlite:///:memory:", owner_user_id=uuid4())
        store.close()
        store.initialize()
        store.close()
        store.close()

    def test_write_episode_missing_participant_rejected(self, store):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="Participant person id\\(s\\) not found"):
            store.write_episode(
                time_start=now,
                time_end=now + timedelta(minutes=1),
                participants=[uuid4()],
            )

    def test_write_fact_bad_episode_rejected(self, store, emily):
        with pytest.raises(ValueError, match="Episode id=.* not found"):
            store.write_fact(
                person_id=emily.id,
                fact_text="test",
                episode_id=uuid4(),
            )

    def test_write_summary_bad_episode_rejected(self, store, emily):
        with pytest.raises(ValueError, match="Episode id=.* not found"):
            store.write_summary(
                person_id=emily.id,
                summary_text="summary",
                episode_id=uuid4(),
            )

    def test_write_edge_bad_episode_rejected(self, store, emily, john):
        with pytest.raises(ValueError, match="Episode id=.* not found"):
            store.write_edge(
                src_id=emily.id,
                relation="colleague",
                dst_id=john.id,
                episode_id=uuid4(),
            )


class TestOrderingAndSerialization:
    def test_profile_context_returns_desc_created_order(self, store, emily, john):
        now = datetime.now(timezone.utc)

        older_fact = store.write_fact(person_id=emily.id, fact_text="old fact")
        newer_fact = store.write_fact(person_id=emily.id, fact_text="new fact")

        older_summary = store.write_summary(person_id=emily.id, summary_text="old summary")
        newer_summary = store.write_summary(person_id=emily.id, summary_text="new summary")

        older_edge = store.write_edge(src_id=emily.id, relation="old-rel", dst_id=john.id)
        newer_edge = store.write_edge(src_id=emily.id, relation="new-rel", dst_id=john.id)

        with store.Session() as session:
            with session.begin():
                session.get(PersonFact, older_fact).created_at = now - timedelta(minutes=4)
                session.get(PersonFact, newer_fact).created_at = now - timedelta(minutes=1)

                session.get(Summary, older_summary).created_at = now - timedelta(minutes=6)
                session.get(Summary, newer_summary).created_at = now - timedelta(minutes=3)

                session.get(Edge, older_edge).created_at = now - timedelta(minutes=7)
                session.get(Edge, newer_edge).created_at = now

        profile = store.get_profile_context(emily.id)

        assert profile.facts[0].fact_text == "new fact"
        assert profile.facts[1].fact_text == "old fact"
        assert profile.summaries[0].summary_text == "new summary"
        assert profile.summaries[1].summary_text == "old summary"
        assert profile.edges_from[0].relation == "new-rel"
        assert profile.edges_from[1].relation == "old-rel"

    def test_naive_datetime_serialized_with_utc_offset(self, store, emily):
        store.write_summary(
            person_id=emily.id,
            summary_text="naive dt summary",
            episode_time_start=datetime(2024, 1, 1, 12, 0, 0),
            episode_time_end=datetime(2024, 1, 1, 12, 30, 0),
        )
        profile = store.get_profile_context(emily.id)

        assert profile.summaries[0].episode_time_start is not None
        assert profile.summaries[0].episode_time_start.endswith("+00:00")
