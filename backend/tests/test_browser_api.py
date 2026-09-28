from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api import deps
from app.api.deps import get_authenticated_owner_user_id
from app.api.routes.conversations import get_conversation, get_recent_conversations
from app.api.routes.people import get_people, get_people_context, get_people_profile
from app.api.routes.user import get_current_user
from app.core.database import Base, make_engine, make_session_factory
from app.crud.memory_store import MemoryStore
from app.models.user import User


@pytest.fixture
def browser_db(tmp_path):
    db_path = tmp_path / "browser-api.sqlite"
    db_url = f"sqlite+pysqlite:///{db_path}"
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = make_session_factory(engine)

    owner_a = uuid4()
    owner_b = uuid4()
    with SessionLocal() as session:
        with session.begin():
            session.add(
                User(
                    id=owner_a,
                    first_name="Avery",
                    last_name="Owner",
                    display_name="Avery Owner",
                    username="avery-owner",
                )
            )
            session.add(
                User(
                    id=owner_b,
                    first_name="Blair",
                    last_name="Owner",
                    display_name="Blair Owner",
                    username="blair-owner",
                )
            )

    store_a = MemoryStore(db_url, owner_user_id=owner_a)
    store_a.initialize(create_schema=False)
    emily = store_a.upsert_person(name="Emily Chen", aliases=["Em"])
    john = store_a.upsert_person(name="John Rivera")
    now = datetime.now(timezone.utc)
    episode_id = store_a.write_episode(
        time_start=now - timedelta(hours=2),
        time_end=now - timedelta(hours=1, minutes=30),
        transcript="Emily and John talked through a project update.",
        summary="Project update with Emily and John.",
        participants=[emily.id, john.id],
    )
    store_a.write_fact(
        person_id=emily.id,
        fact_text="Emily leads the robotics club",
        fact_category="affiliation",
        source=episode_id,
    )
    store_a.write_fact(
        person_id=emily.id,
        fact_text="Emily loves long-distance running",
        fact_category="hobby",
    )
    store_a.write_fact(
        person_id=john.id,
        fact_text="John works on backend reliability",
        fact_category="affiliation",
    )

    store_b = MemoryStore(db_url, owner_user_id=owner_b)
    store_b.initialize(create_schema=False)
    other_person = store_b.upsert_person(name="Taylor Guest")
    store_b.write_episode(
        time_start=now - timedelta(days=1),
        time_end=now - timedelta(days=1, minutes=-20),
        transcript="Taylor met briefly with the wearer.",
        summary="Short catch-up with Taylor.",
        participants=[other_person.id],
    )

    try:
        yield SessionLocal, owner_a, owner_b, episode_id
    finally:
        store_a.close()
        store_b.close()
        engine.dispose()


def _request(user_id=None) -> SimpleNamespace:
    state = SimpleNamespace()
    if user_id is not None:
        state.user_id = user_id
    return SimpleNamespace(state=state)


def test_owner_dependency_rejects_missing_identity_when_multiple_owners_exist(browser_db):
    SessionLocal, _, _, _ = browser_db
    with SessionLocal() as session:
        with pytest.raises(HTTPException) as exc_info:
            get_authenticated_owner_user_id(_request(), session, None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == (
        "Authenticated user required; multiple owner users exist. "
        "Send X-User-Id or use the real session."
    )


def test_owner_dependency_falls_back_to_single_local_owner(tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'single-owner.sqlite'}")
    Base.metadata.create_all(engine)
    SessionLocal = make_session_factory(engine)
    owner_id = uuid4()
    with SessionLocal() as session:
        with session.begin():
            session.add(User(id=owner_id, first_name="Solo", last_name="Owner", username="solo-owner"))

    try:
        with SessionLocal() as session:
            assert get_authenticated_owner_user_id(_request(), session, None) == owner_id
    finally:
        engine.dispose()


def test_owner_dependency_uses_dev_header(browser_db):
    SessionLocal, owner_a, _, _ = browser_db
    with SessionLocal() as session:
        assert get_authenticated_owner_user_id(_request(), session, str(owner_a)) == owner_a


def test_owner_dependency_prefers_request_state_over_header(browser_db):
    SessionLocal, owner_a, owner_b, _ = browser_db
    with SessionLocal() as session:
        resolved = get_authenticated_owner_user_id(_request(user_id=owner_a), session, str(owner_b))

    assert resolved == owner_a


def test_owner_dependency_ignores_header_when_dev_fallback_disabled(browser_db, monkeypatch):
    SessionLocal, owner_a, _, _ = browser_db
    monkeypatch.setattr(deps.settings, "allow_dev_auth_fallback", False)

    with SessionLocal() as session:
        with pytest.raises(HTTPException) as exc_info:
            get_authenticated_owner_user_id(_request(), session, str(owner_a))
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authenticated user required"

        # The real session middleware path keeps working with the flag off.
        assert get_authenticated_owner_user_id(_request(user_id=owner_a), session, None) == owner_a


def test_owner_dependency_rejects_malformed_and_unknown_ids(browser_db):
    SessionLocal, _, _, _ = browser_db
    with SessionLocal() as session:
        with pytest.raises(HTTPException) as malformed:
            get_authenticated_owner_user_id(_request(), session, "not-a-uuid")
        with pytest.raises(HTTPException) as unknown:
            get_authenticated_owner_user_id(_request(), session, str(uuid4()))

    assert malformed.value.status_code == 401
    assert malformed.value.detail == "Invalid authenticated user id"
    assert unknown.value.status_code == 401
    assert unknown.value.detail == "Authenticated user was not found"


def test_recent_conversations_are_scoped_to_owner(browser_db):
    SessionLocal, owner_a, _, _ = browser_db
    with SessionLocal() as session:
        payload = get_recent_conversations(owner_a, session)

    assert len(payload) == 1
    assert payload[0].summary == "Project update with Emily and John."
    assert [participant.name for participant in payload[0].participants] == [
        "Emily Chen",
        "John Rivera",
    ]


def test_current_user_endpoint_uses_existing_user_model(browser_db):
    SessionLocal, owner_a, _, _ = browser_db
    with SessionLocal() as session:
        payload = get_current_user(owner_a, session)

    assert payload.id == owner_a
    assert payload.display_name == "Avery Owner"
    assert payload.username == "avery-owner"


def test_people_endpoint_returns_last_seen_and_top_facts(browser_db):
    SessionLocal, owner_a, owner_b, _ = browser_db
    with SessionLocal() as session:
        payload = get_people(owner_a, session)

    assert payload.query is None
    assert [person.name for person in payload.items] == ["John Rivera", "Emily Chen"]
    assert payload.items[0].top_facts == ["John works on backend reliability"]
    assert payload.items[0].fact_count == 1
    assert payload.items[1].aliases == ["Em"]
    assert payload.items[1].top_facts == [
        "Emily loves long-distance running",
        "Emily leads the robotics club",
    ]
    assert payload.items[1].last_seen_at is not None
    assert payload.items[1].summary_count == 0

    with SessionLocal() as session:
        other_payload = get_people(owner_b, session)
    assert [person.name for person in other_payload.items] == ["Taylor Guest"]


def test_people_endpoint_search_uses_resolver_and_filters_results(browser_db):
    SessionLocal, owner_a, _, _ = browser_db
    with SessionLocal() as session:
        payload = get_people(owner_a, session, query="Emily")

    assert payload.query == "Emily"
    assert payload.resolution is not None
    assert payload.resolution.person_id is not None
    assert [person.name for person in payload.items] == ["Emily Chen"]


def test_people_profile_endpoint_uses_profile_context(browser_db):
    SessionLocal, owner_a, _, _ = browser_db
    with SessionLocal() as session:
        people = get_people(owner_a, session)
        emily = next(person for person in people.items if person.name == "Emily Chen")

    with SessionLocal() as session:
        payload = get_people_profile(emily.id, owner_a, session)

    assert payload.person.name == "Emily Chen"
    assert payload.person.aliases == ["Em"]
    assert [fact.fact_text for fact in payload.profile.facts] == [
        "Emily loves long-distance running",
        "Emily leads the robotics club",
    ]


def test_people_context_endpoint_uses_existing_retrieval_service(browser_db):
    SessionLocal, owner_a, _, _ = browser_db
    with SessionLocal() as session:
        people = get_people(owner_a, session)
        emily = next(person for person in people.items if person.name == "Emily Chen")

    with SessionLocal() as session:
        payload = get_people_context(emily.id, owner_a, session, "robotics club")

    assert payload.person_id == emily.id
    assert [fact.fact_text for fact in payload.facts] == [
        "Emily leads the robotics club",
    ]


def test_conversation_detail_endpoint_returns_participants_and_transcript(browser_db):
    SessionLocal, owner_a, _, episode_id = browser_db

    with SessionLocal() as session:
        payload = get_conversation(episode_id, owner_a, session)

    assert payload.id == episode_id
    assert payload.summary == "Project update with Emily and John."
    assert payload.transcript == "Emily and John talked through a project update."
    assert [participant.name for participant in payload.participants] == [
        "Emily Chen",
        "John Rivera",
    ]
    assert payload.participants[0].fact_count == 2


def test_conversation_detail_endpoint_returns_404_for_unknown_episode(browser_db):
    SessionLocal, owner_a, _, _ = browser_db

    with SessionLocal() as session:
        with pytest.raises(HTTPException) as exc_info:
            get_conversation(uuid4(), owner_a, session)

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail
