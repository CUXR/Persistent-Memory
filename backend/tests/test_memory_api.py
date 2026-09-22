"""
Tests for the memory HTTP API (GitHub issue #30).

Covers the write endpoints (facts/summaries) and the bounded, ranked
relevant-memories endpoint used on re-encountering a person.
"""

from pathlib import Path
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api import deps
from app.crud.memory_store import MemoryStore
from app.main import app
from app.models.user import User


@pytest.fixture
def owner_id():
    return uuid4()


@pytest.fixture
def client(owner_id, tmp_path):
    # FastAPI's TestClient drives the app from a worker thread; a `:memory:`
    # SQLite DB is only visible to the connection that created it, so each
    # thread would see an empty database. A temp file avoids that.
    db_path = tmp_path / "memory_api_test.db"
    store = MemoryStore(f"sqlite+pysqlite:///{db_path}", owner_user_id=owner_id)
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

    app.dependency_overrides[deps.get_memory_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()
    store.close()


def _headers(owner_id):
    return {"X-Owner-User-Id": str(owner_id)}


def _make_person(name: str = "Emily Chen"):
    # People are created through the store fixture directly in these tests
    # (no /people POST endpoint exists yet), so reach into the app's store.
    store: MemoryStore = app.dependency_overrides[deps.get_memory_store]()
    return store.upsert_person(name=name)


class TestWriteFact:
    def test_write_fact_returns_created_fact(self, client, owner_id):
        person = _make_person()

        resp = client.post(
            f"/people/{person.id}/facts",
            json={"fact_text": "likes climbing", "confidence": 0.9, "fact_category": "hobby"},
            headers=_headers(owner_id),
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["fact_text"] == "likes climbing"
        assert body["confidence"] == pytest.approx(0.9)
        assert body["fact_category"] == "hobby"

    def test_write_fact_for_unknown_person_404s(self, client, owner_id):
        resp = client.post(
            f"/people/{uuid4()}/facts",
            json={"fact_text": "test"},
            headers=_headers(owner_id),
        )
        assert resp.status_code == 404

    def test_missing_owner_header_rejected(self):
        # Uses a bare TestClient (no dependency override) so the real
        # X-Owner-User-Id header requirement on get_memory_store applies.
        assert not app.dependency_overrides
        resp = TestClient(app).post(
            f"/people/{uuid4()}/facts",
            json={"fact_text": "test"},
        )
        assert resp.status_code == 422


class TestWriteSummary:
    def test_write_summary_returns_created_summary(self, client, owner_id):
        person = _make_person()

        resp = client.post(
            f"/people/{person.id}/summaries",
            json={"summary_text": "Talked about climbing"},
            headers=_headers(owner_id),
        )

        assert resp.status_code == 201
        assert resp.json()["summary_text"] == "Talked about climbing"


class TestRelevantMemories:
    def test_returns_bounded_ranked_memories(self, client, owner_id):
        person = _make_person()
        for i in range(5):
            client.post(
                f"/people/{person.id}/facts",
                json={"fact_text": f"fact {i}", "confidence": i / 5},
                headers=_headers(owner_id),
            )

        resp = client.post(
            f"/people/{person.id}/memories/relevant",
            json={"limit": 2},
            headers=_headers(owner_id),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["facts"]) == 2
        # Highest confidence (most relevant) fact should rank first.
        assert body["facts"][0]["fact_text"] == "fact 4"

    def test_defaults_to_limit_five_with_empty_body(self, client, owner_id):
        person = _make_person()
        resp = client.post(f"/people/{person.id}/memories/relevant", headers=_headers(owner_id))
        assert resp.status_code == 200
        assert resp.json()["interaction_context"]

    def test_unknown_person_404s(self, client, owner_id):
        resp = client.post(
            f"/people/{uuid4()}/memories/relevant",
            headers=_headers(owner_id),
        )
        assert resp.status_code == 404
