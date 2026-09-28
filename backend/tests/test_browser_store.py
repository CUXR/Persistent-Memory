"""MemoryStore read paths that power the memory browser (people overview, recent conversations)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from uuid import uuid4

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.models.person import Person
from app.models.user import User


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    owner_id = uuid4()
    memory_store = MemoryStore(f"sqlite+pysqlite:///{tmp_path / 'store.sqlite'}", owner_user_id=owner_id)
    memory_store.initialize()
    with memory_store.Session() as session:
        with session.begin():
            session.add(User(id=owner_id, first_name="Test", last_name="Owner", username="test-owner"))
    yield memory_store
    memory_store.close()


def _last_seen(store: MemoryStore, person_id):
    with store.Session() as session:
        person = session.get(Person, person_id)
        return None if person is None or person.last_seen_at is None else _aware(person.last_seen_at)


def test_write_episode_sets_last_seen_at_to_episode_end(store):
    emily = store.upsert_person(name="Emily Chen")
    bystander = store.upsert_person(name="Sam Bystander")
    start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)

    store.write_episode(time_start=start, time_end=end, participants=[emily.id])

    assert _last_seen(store, emily.id) == end
    assert _last_seen(store, bystander.id) is None


def test_write_episode_never_regresses_last_seen_at(store):
    emily = store.upsert_person(name="Emily Chen")
    newer_end = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store.write_episode(time_start=newer_end - timedelta(hours=1), time_end=newer_end, participants=[emily.id])

    older_end = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    store.write_episode(time_start=older_end - timedelta(hours=1), time_end=older_end, participants=[emily.id])

    assert _last_seen(store, emily.id) == newer_end


def test_list_people_overview_orders_by_last_seen_and_batches_metadata(store):
    emily = store.upsert_person(name="Emily Chen", aliases=["Em"])
    john = store.upsert_person(name="John Rivera")
    never_seen = store.upsert_person(name="Never Seen")
    base = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    episode = store.write_episode(time_start=base, time_end=base + timedelta(minutes=10), participants=[john.id])
    store.write_episode(
        time_start=base + timedelta(days=1),
        time_end=base + timedelta(days=1, minutes=10),
        participants=[emily.id],
    )
    store.write_fact(emily.id, "Emily leads the robotics club", fact_category="affiliation")
    store.write_fact(emily.id, "Emily runs marathons", fact_category="hobby")
    store.write_fact(emily.id, "Emily grew up in Lisbon", fact_category="biographical")
    store.write_summary(emily.id, "Talked about robotics.", episode_id=episode)
    store.write_edge(emily.id, "friend", john.id)

    overview = store.list_people_overview()

    assert [item.name for item in overview] == ["Emily Chen", "John Rivera", "Never Seen"]
    emily_card = overview[0]
    assert emily_card.aliases == ["Em"]
    assert emily_card.top_facts == ["Emily grew up in Lisbon", "Emily runs marathons"]
    assert emily_card.fact_count == 3
    assert emily_card.summary_count == 1
    assert emily_card.relationship_count == 1
    assert emily_card.last_seen_at == (base + timedelta(days=1, minutes=10)).isoformat()
    assert overview[2].last_seen_at is None
    assert overview[2].top_facts == []

    assert [item.name for item in store.list_people_overview(limit=1)] == ["Emily Chen"]
    subset = store.list_people_overview(limit=None, person_ids=[never_seen.id, emily.id, uuid4()])
    assert [item.name for item in subset] == ["Emily Chen", "Never Seen"]
    assert store.list_people_overview(limit=None, person_ids=[]) == []


def test_list_people_overview_falls_back_to_episode_time_when_last_seen_unset(store):
    legacy = store.upsert_person(name="Legacy Person")
    fresh = store.upsert_person(name="Fresh Person")
    base = datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)
    store.write_episode(time_start=base + timedelta(days=2), time_end=base + timedelta(days=2, minutes=5), participants=[legacy.id])
    store.write_episode(time_start=base, time_end=base + timedelta(minutes=5), participants=[fresh.id])

    # Simulate a row written before last_seen_at was maintained on the write path.
    with store.Session() as session:
        with session.begin():
            session.get(Person, legacy.id).last_seen_at = None

    overview = store.list_people_overview()

    assert [item.name for item in overview] == ["Legacy Person", "Fresh Person"]
    assert overview[0].last_seen_at == (base + timedelta(days=2)).isoformat()


def test_list_recent_conversations_orders_participants_deterministically(store):
    zoe = store.upsert_person(name="Zoe Alder")
    carl = store.upsert_person(name="Carl Dent")
    amy = store.upsert_person(name="Amy Brook")
    base = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
    # Insertion order (zoe, carl, amy) differs from alphabetical order so a
    # clock-dependent sort would be caught.
    episode_id = store.write_episode(
        time_start=base,
        time_end=base + timedelta(minutes=10),
        summary="Zoe led the discussion.",
        participants=[zoe.id, carl.id, amy.id],
    )

    recent = store.list_recent_conversations()
    detail = store.get_conversation_detail(episode_id)

    assert [item.id for item in recent] == [episode_id]
    # Primary participant first, then the rest alphabetically, on both read paths.
    assert [p.name for p in recent[0].participants] == ["Zoe Alder", "Amy Brook", "Carl Dent"]
    assert [p.name for p in detail.participants] == ["Zoe Alder", "Amy Brook", "Carl Dent"]
    assert recent[0].started_at == detail.started_at == base.isoformat()
    assert detail.participants[0].last_seen_at == (base + timedelta(minutes=10)).isoformat()


def test_episode_times_with_non_utc_offsets_are_stored_as_utc(store):
    emily = store.upsert_person(name="Emily Chen")
    plus_two = timezone(timedelta(hours=2))
    start = datetime(2026, 9, 1, 12, 0, tzinfo=plus_two)  # 10:00 UTC
    end = datetime(2026, 9, 1, 12, 30, tzinfo=plus_two)  # 10:30 UTC

    store.write_episode(time_start=start, time_end=end, participants=[emily.id])

    recent = store.list_recent_conversations()[0]
    assert recent.started_at == "2026-09-01T10:00:00+00:00"
    assert recent.ended_at == "2026-09-01T10:30:00+00:00"
    assert store.list_people_overview()[0].last_seen_at == "2026-09-01T10:30:00+00:00"
    assert _last_seen(store, emily.id) == end


def test_recent_conversation_reads_reject_bad_limits_and_unknown_episodes(store):
    with pytest.raises(ValueError):
        store.list_recent_conversations(limit=0)
    with pytest.raises(ValueError):
        store.list_people_overview(limit=0)
    with pytest.raises(ValueError, match="not found"):
        store.get_conversation_detail(uuid4())
    assert store.list_recent_conversations() == []
    assert store.list_people_overview() == []


def test_from_engine_requires_an_owner(store):
    with pytest.raises(ValueError, match="owner_user_id"):
        MemoryStore.from_engine(store.Session.kw["bind"], owner_user_id=None)  # type: ignore[arg-type]
