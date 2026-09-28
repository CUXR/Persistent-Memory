"""HTTP-level tests for the memory browser API (conversations, people, users/me).

These go through FastAPI's TestClient so they exercise real dependency wiring,
status codes, JSON serialization, query validation, CORS, and route ordering,
which the direct route-function tests in ``test_browser_api.py`` cannot cover.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.deps import get_db_session
from app.core.database import Base, make_engine, make_session_factory
from app.crud.memory_store import MemoryStore
from app.main import app
from app.models.user import User


@pytest.fixture
def api(tmp_path):
    """TestClient bound to a file-backed SQLite DB seeded for two owners."""

    db_url = f"sqlite+pysqlite:///{tmp_path / 'browser-http.sqlite'}"
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = make_session_factory(engine)

    owner_a, owner_b, owner_empty = uuid4(), uuid4(), uuid4()
    with SessionLocal() as session, session.begin():
        for user_id, first, username in (
            (owner_a, "Avery", "avery-owner"),
            (owner_b, "Blair", "blair-owner"),
            (owner_empty, "Casey", "casey-owner"),
        ):
            session.add(
                User(id=user_id, first_name=first, last_name="Owner", display_name=f"{first} Owner", username=username)
            )

    now = datetime.now(timezone.utc)
    store_a = MemoryStore(db_url, owner_user_id=owner_a)
    store_a.initialize(create_schema=False)
    emily = store_a.upsert_person(name="Emily Chen", aliases=["Em"])
    john = store_a.upsert_person(name="John Rivera")
    episode_a = store_a.write_episode(
        time_start=now - timedelta(hours=2),
        time_end=now - timedelta(hours=1),
        transcript="Emily and John talked through a project update.",
        summary="Project update with Emily and John.",
        participants=[emily.id, john.id],
    )
    store_a.write_fact(person_id=emily.id, fact_text="Emily leads the robotics club", fact_category="affiliation")
    john_park = store_a.upsert_person(name="John Park")
    store_a.write_fact(person_id=john_park.id, fact_text="John Park coaches the rowing team", fact_category="affiliation")
    older_episode = store_a.write_episode(
        time_start=now - timedelta(days=3),
        time_end=now - timedelta(days=3) + timedelta(minutes=15),
        transcript="John Park introduced the rowing schedule.",
        summary="Rowing schedule with John Park.",
        participants=[john_park.id],
    )

    store_b = MemoryStore(db_url, owner_user_id=owner_b)
    store_b.initialize(create_schema=False)
    taylor = store_b.upsert_person(name="Taylor Guest")
    episode_b = store_b.write_episode(
        time_start=now - timedelta(days=1),
        time_end=now - timedelta(days=1) + timedelta(minutes=20),
        transcript="Taylor met briefly with the wearer.",
        summary="Short catch-up with Taylor.",
        participants=[taylor.id],
    )

    def override_db_session():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db_session] = override_db_session
    client = TestClient(app)
    try:
        yield {
            "client": client,
            "owner_a": owner_a,
            "owner_b": owner_b,
            "owner_empty": owner_empty,
            "emily_id": str(emily.id),
            "john_park_id": str(john_park.id),
            "episode_a": str(episode_a),
            "older_episode": str(older_episode),
            "episode_b": str(episode_b),
        }
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        client.close()
        store_a.close()
        store_b.close()
        engine.dispose()


def _headers(owner_id) -> dict[str, str]:
    return {"X-User-Id": str(owner_id)}


def test_recent_conversations_http_shape(api):
    response = api["client"].get("/conversations/recent", headers=_headers(api["owner_a"]))

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload] == [api["episode_a"], api["older_episode"]]
    item = payload[0]
    assert item["id"] == api["episode_a"]
    assert item["summary"] == "Project update with Emily and John."
    assert [p["name"] for p in item["participants"]] == ["Emily Chen", "John Rivera"]
    # Timestamps must be timezone-aware ISO strings even when the DB is SQLite.
    assert item["started_at"].endswith("+00:00")
    assert item["ended_at"].endswith("+00:00")


def test_recent_conversations_limit_is_validated(api):
    client, headers = api["client"], _headers(api["owner_a"])

    assert client.get("/conversations/recent", params={"limit": 0}, headers=headers).status_code == 422
    assert client.get("/conversations/recent", params={"limit": 101}, headers=headers).status_code == 422
    limited = client.get("/conversations/recent", params={"limit": 1}, headers=headers)
    assert limited.status_code == 200
    assert [item["id"] for item in limited.json()] == [api["episode_a"]]


def test_missing_owner_header_is_unauthorized_when_multiple_owners_exist(api):
    response = api["client"].get("/conversations/recent")

    assert response.status_code == 401
    assert "Authenticated user required" in response.json()["detail"]


def test_malformed_owner_header_is_unauthorized(api):
    response = api["client"].get("/people", headers={"X-User-Id": "not-a-uuid"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authenticated user id"


def test_unknown_owner_id_is_rejected_not_500(api):
    response = api["client"].get("/people", headers=_headers(uuid4()))

    assert response.status_code == 401
    assert "not found" in response.json()["detail"].lower()


def test_empty_states_for_owner_without_data(api):
    client, headers = api["client"], _headers(api["owner_empty"])

    assert client.get("/conversations/recent", headers=headers).json() == []
    people = client.get("/people", headers=headers).json()
    assert people == {"items": [], "query": None, "resolution": None}


def test_conversation_detail_is_scoped_to_owner(api):
    client = api["client"]

    own = client.get(f"/conversations/{api['episode_a']}", headers=_headers(api["owner_a"]))
    assert own.status_code == 200
    assert own.json()["transcript"] == "Emily and John talked through a project update."
    assert own.json()["started_at"].endswith("+00:00")
    assert own.json()["participants"][0]["last_seen_at"].endswith("+00:00")

    cross = client.get(f"/conversations/{api['episode_b']}", headers=_headers(api["owner_a"]))
    assert cross.status_code == 404

    bad = client.get("/conversations/not-a-uuid", headers=_headers(api["owner_a"]))
    assert bad.status_code == 422


def test_people_list_is_scoped_and_serialized(api):
    client = api["client"]

    a = client.get("/people", headers=_headers(api["owner_a"])).json()
    # Most recently seen first; John Park's only episode is three days old.
    assert [p["name"] for p in a["items"]] == ["John Rivera", "Emily Chen", "John Park"]
    emily = a["items"][1]
    assert emily["aliases"] == ["Em"]
    assert emily["top_facts"] == ["Emily leads the robotics club"]
    assert emily["fact_count"] == 1
    assert emily["last_seen_at"].endswith("+00:00")

    b = client.get("/people", headers=_headers(api["owner_b"])).json()
    assert [p["name"] for p in b["items"]] == ["Taylor Guest"]


def test_people_search_resolves_a_known_name(api):
    response = api["client"].get("/people", params={"query": "Emily"}, headers=_headers(api["owner_a"]))

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "Emily"
    assert payload["resolution"]["person_id"] == api["emily_id"]
    assert [p["name"] for p in payload["items"]] == ["Emily Chen"]


def test_person_profile_is_scoped_to_owner(api):
    client = api["client"]

    own = client.get(f"/people/{api['emily_id']}/profile", headers=_headers(api["owner_a"]))
    assert own.status_code == 200
    assert own.json()["person"]["name"] == "Emily Chen"
    assert [f["fact_text"] for f in own.json()["profile"]["facts"]] == ["Emily leads the robotics club"]

    cross = client.get(f"/people/{api['emily_id']}/profile", headers=_headers(api["owner_b"]))
    assert cross.status_code == 404

    missing = client.get(f"/people/{uuid4()}/profile", headers=_headers(api["owner_a"]))
    assert missing.status_code == 404


def test_person_context_is_scoped_to_owner(api):
    client = api["client"]

    own = client.get(f"/people/{api['emily_id']}/context", params={"query": "robotics"}, headers=_headers(api["owner_a"]))
    assert own.status_code == 200
    assert own.json()["person_id"] == api["emily_id"]
    assert [f["fact_text"] for f in own.json()["facts"]] == ["Emily leads the robotics club"]
    assert "embedding" not in own.json()["facts"][0]

    cross = client.get(f"/people/{api['emily_id']}/context", params={"query": "robotics"}, headers=_headers(api["owner_b"]))
    assert cross.status_code == 404
    assert cross.json()["detail"] == "Person not found"

    assert client.get(f"/people/{uuid4()}/context", params={"query": "x"}, headers=_headers(api["owner_a"])).status_code == 404
    assert client.get(f"/people/{api['emily_id']}/context", params={"query": ""}, headers=_headers(api["owner_a"])).status_code == 422
    assert client.get(f"/people/{api['emily_id']}/context", headers=_headers(api["owner_a"])).status_code == 422


def test_profile_payload_omits_vectors_and_biometric_keys(api):
    payload = api["client"].get(f"/people/{api['emily_id']}/profile", headers=_headers(api["owner_a"])).json()

    assert set(payload["person"]) == {"id", "name", "aliases", "created_at", "updated_at"}
    assert set(payload["profile"]) == {"facts", "prefs", "summaries", "edges_from"}
    assert "embedding" not in payload["profile"]["facts"][0]
    assert payload["person"]["created_at"].endswith("+00:00")


def test_current_user_endpoint(api):
    client = api["client"]

    ok = client.get("/users/me", headers=_headers(api["owner_a"]))
    assert ok.status_code == 200
    assert ok.json()["username"] == "avery-owner"
    assert ok.json()["created_at"].endswith("+00:00")
    assert ok.json()["updated_at"].endswith("+00:00")

    assert client.get("/users/me", headers=_headers(uuid4())).status_code in (401, 404)


def test_cors_preflight_allows_frontend_origin_and_owner_header(api):
    response = api["client"].options(
        "/people",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-user-id",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "x-user-id" in response.headers["access-control-allow-headers"].lower()


def test_cors_rejects_unknown_origin(api):
    response = api["client"].get(
        "/people",
        headers={**_headers(api["owner_a"]), "Origin": "http://evil.example"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_list_and_detail_serialize_timestamps_identically(api):
    client, headers = api["client"], _headers(api["owner_a"])

    recent = client.get("/conversations/recent", headers=headers).json()[0]
    detail = client.get(f"/conversations/{api['episode_a']}", headers=headers).json()

    assert recent["started_at"] == detail["started_at"]
    assert recent["ended_at"] == detail["ended_at"]
    assert [p["id"] for p in recent["participants"]] == [p["id"] for p in detail["participants"]]


def test_people_limit_is_applied_and_validated(api):
    client, headers = api["client"], _headers(api["owner_a"])

    limited = client.get("/people", params={"limit": 2}, headers=headers)
    assert limited.status_code == 200
    assert [p["name"] for p in limited.json()["items"]] == ["John Rivera", "Emily Chen"]

    assert client.get("/people", params={"limit": 0}, headers=headers).status_code == 422
    assert client.get("/people", params={"limit": 251}, headers=headers).status_code == 422
    assert client.get("/people", params={"query": "x" * 201}, headers=headers).status_code == 422


def test_people_search_returns_ambiguity_candidates(api):
    response = api["client"].get("/people", params={"query": "John"}, headers=_headers(api["owner_a"]))

    assert response.status_code == 200
    payload = response.json()
    resolution = payload["resolution"]
    assert resolution["person_id"] is None
    assert resolution["is_ambiguous"] is True
    assert [c["name"] for c in resolution["candidates"]] == ["John Park", "John Rivera"]
    assert [p["id"] for p in payload["items"]] == [c["person_id"] for c in resolution["candidates"]]
    assert resolution["candidates"][0]["hints"] == {"affiliation": ["John Park coaches the rowing team"]}


def test_people_search_falls_back_to_lexical_match_on_facts(api):
    response = api["client"].get("/people", params={"query": "robotics"}, headers=_headers(api["owner_a"]))

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolution"] == {"person_id": None, "is_ambiguous": False, "candidates": []}
    assert [p["name"] for p in payload["items"]] == ["Emily Chen"]


def test_people_search_with_no_match_returns_empty_items(api):
    payload = api["client"].get("/people", params={"query": "zzzz"}, headers=_headers(api["owner_a"])).json()

    assert payload["items"] == []
    assert payload["query"] == "zzzz"


def test_dev_header_is_ignored_when_fallback_disabled(api, monkeypatch):
    from app.api import deps

    monkeypatch.setattr(deps.settings, "allow_dev_auth_fallback", False)

    response = api["client"].get("/people", headers=_headers(api["owner_a"]))

    assert response.status_code == 401
    assert response.json()["detail"] == "Authenticated user required"


def test_api_is_read_only(api):
    client, headers = api["client"], _headers(api["owner_a"])

    assert client.post("/people", headers=headers).status_code == 405
    assert client.post("/conversations/recent", headers=headers).status_code == 405
    assert client.delete(f"/people/{api['emily_id']}/profile", headers=headers).status_code == 405
