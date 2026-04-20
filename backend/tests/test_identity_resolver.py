from __future__ import annotations

from pathlib import Path
import sys

import pytest
from sqlalchemy import func, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.models.person import Person
from app.services.identity_resolver import IdentityResolver, IdentitySignal


@pytest.fixture
def store():
    s = MemoryStore("sqlite+pysqlite:///:memory:")
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def existing_person(store: MemoryStore):
    return store.create_person(
        name="Jordan Lee",
        aliases=["Jordan"],
        face_key="face_jordan_001",
    )


class TestIdentityResolver:
    def test_resolves_existing_person_from_face_key(self, store: MemoryStore, existing_person):
        resolver = IdentityResolver(store)

        result = resolver.resolve(IdentitySignal(face_key="face_jordan_001"))

        assert result.person_id == existing_person.id
        assert result.person is not None
        assert result.person.id == existing_person.id
        assert result.created_new_person is False
        assert resolver.active_interlocutor_id == existing_person.id

    def test_creates_person_when_face_key_is_unknown(self, store: MemoryStore):
        resolver = IdentityResolver(store)

        result = resolver.resolve(
            IdentitySignal(
                face_key="face_new_001",
                observed_name="Sam Carter",
            )
        )

        assert result.person_id is not None
        assert result.person is not None
        assert result.person.name == "Sam Carter"
        assert result.created_new_person is True
        assert resolver.active_interlocutor_id == result.person_id
        assert store.resolve_person_by_face_key("face_new_001").id == result.person_id

    def test_repeated_resolution_calls_reuse_created_person(self, store: MemoryStore):
        resolver = IdentityResolver(store)

        first = resolver.resolve(IdentitySignal(face_key="face_repeat_001"))
        second = resolver.resolve(IdentitySignal(face_key="face_repeat_001"))

        assert first.person_id == second.person_id
        assert first.created_new_person is True
        assert second.created_new_person is False

        with store.Session() as session:
            people_count = session.scalar(select(func.count()).select_from(Person))

        assert people_count == 1

    def test_logs_only_when_active_identity_changes(self, store: MemoryStore, existing_person, caplog):
        resolver = IdentityResolver(store)
        other_person = store.create_person(
            name="Taylor Brooks",
            aliases=[],
            face_key="face_taylor_001",
        )

        caplog.set_level("INFO", logger="app.services.identity_resolver")

        resolver.resolve(IdentitySignal(face_key="face_jordan_001"))
        resolver.resolve(IdentitySignal(face_key="face_jordan_001"))
        resolver.resolve(IdentitySignal(face_key="face_taylor_001"))

        change_logs = [
            record.message
            for record in caplog.records
            if "active_interlocutor_id changed" in record.message
        ]

        assert change_logs == [
            f"active_interlocutor_id changed from None to {existing_person.id}",
            f"active_interlocutor_id changed from {existing_person.id} to {other_person.id}",
        ]

    def test_missing_face_key_clears_active_interlocutor(self, store: MemoryStore, existing_person):
        resolver = IdentityResolver(store)
        resolver.resolve(IdentitySignal(face_key="face_jordan_001"))

        result = resolver.resolve(IdentitySignal(face_key="  "))

        assert result.person_id is None
        assert result.person is None
        assert result.created_new_person is False
        assert resolver.active_interlocutor_id is None
