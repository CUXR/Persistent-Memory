from pathlib import Path
import sys
import logging

from sqlalchemy import select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.models.person import Person
from app.services.identity_resolver import IdentityResolver, IdentitySignal


def _make_store() -> MemoryStore:
    store = MemoryStore("sqlite+pysqlite:///:memory:")
    store.initialize()
    return store


def test_matches_existing_person_and_updates_active_interlocutor(caplog):
    store = _make_store()
    try:
        emily = store.upsert_person(
            name="Emily Chen",
            aliases=["Em"],
            face_key="face_emily_001",
        )
        resolver = IdentityResolver(store)

        with caplog.at_level(logging.INFO, logger="app.services.identity_resolver"):
            person_id = resolver.resolve(IdentitySignal(face_key="face_emily_001"))

        assert person_id == emily.id
        assert resolver.active_interlocutor_id == emily.id
        assert "Active interlocutor changed from None to" in caplog.text
    finally:
        store.close()


def test_creates_new_person_when_face_key_is_unknown():
    store = _make_store()
    try:
        resolver = IdentityResolver(store)

        person_id = resolver.resolve(IdentitySignal(face_key="face_new_001"))

        assert person_id is not None
        assert resolver.active_interlocutor_id == person_id

        created = store.resolve_person_by_face_key("face_new_001")
        assert created is not None
        assert created.id == person_id
    finally:
        store.close()


def test_repeated_resolution_calls_do_not_duplicate_people():
    store = _make_store()
    try:
        resolver = IdentityResolver(store)

        first_id = resolver.resolve(IdentitySignal(face_key="face_repeat_001"))
        second_id = resolver.resolve(IdentitySignal(face_key="face_repeat_001"))
        third_id = resolver.resolve(IdentitySignal(face_key="face_repeat_001"))

        assert first_id == second_id == third_id
        assert resolver.active_interlocutor_id == first_id

        with store.Session() as session:
            count = len(
                session.scalars(
                    select(Person).where(Person.face_key == "face_repeat_001")
                ).all()
            )

        assert count == 1
    finally:
        store.close()


def test_blank_signal_clears_active_interlocutor(caplog):
    store = _make_store()
    try:
        store.upsert_person(
            name="Emily Chen",
            aliases=[],
            face_key="face_emily_001",
        )
        resolver = IdentityResolver(store)
        resolver.resolve(IdentitySignal(face_key="face_emily_001"))

        with caplog.at_level(logging.INFO, logger="app.services.identity_resolver"):
            result = resolver.resolve(IdentitySignal(face_key="  "))

        assert result is None
        assert resolver.active_interlocutor_id is None
        assert "Active interlocutor changed from" in caplog.text
    finally:
        store.close()
