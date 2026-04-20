"""
Tests for the person_resolver module.

Issue #8 Acceptance Criteria:
  - Query referencing known name resolves correctly
  - Ambiguous queries return candidates with fact hints
  - No console errors
"""

from pathlib import Path
import sys
from uuid import UUID, uuid4

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore
from app.models.user import User
from app.crud.person_resolver import PersonResolver


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def store():
    """Fresh in-memory database for each test."""
    owner_id = uuid4()
    s = MemoryStore("sqlite+pysqlite:///:memory:", owner_user_id=owner_id)
    s.initialize()
    with s.Session() as session:
        with session.begin():
            session.add(User(
                id=owner_id,
                first_name="Test",
                last_name="Owner",
                display_name="Test Owner",
                username="test-owner",
            ))
    yield s
    s.close()


@pytest.fixture
def resolver(store: MemoryStore):
    """PersonResolver backed by the in-memory store."""
    return PersonResolver(store)


@pytest.fixture
def emily(store: MemoryStore):
    """Person: Emily Chen."""
    return store.upsert_person(
        name="Emily Chen",
        face_key="face_emily_001",
        voice_key="voice_emily_001",
        persona90=[0.5] * 90,
    )


@pytest.fixture
def john(store: MemoryStore):
    """Person: John Rivera."""
    return store.upsert_person(
        name="John Rivera",
        face_key="face_john_001",
        voice_key="voice_john_001",
    )


@pytest.fixture
def sarah(store: MemoryStore):
    """Person: Sarah Martinez."""
    return store.upsert_person(
        name="Sarah Martinez",
    )


# ── Name resolution ─────────────────────────────────────────


class TestResolveByName:
    """Query referencing known name resolves correctly."""

    def test_exact_name_match(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert result.person_id == emily.id

    def test_case_insensitive_name(self, resolver, emily):
        result = resolver.resolve_person_from_query("emily chen")
        assert result.person_id == emily.id

    def test_mixed_case_name(self, resolver, emily):
        result = resolver.resolve_person_from_query("EMILY CHEN")
        assert result.person_id == emily.id

    def test_name_embedded_in_sentence(self, resolver, emily):
        result = resolver.resolve_person_from_query("What does Emily Chen like?")
        assert result.person_id == emily.id

    def test_first_name_only_when_unambiguous(self, resolver, emily):
        """First name alone should resolve when only one person matches."""
        result = resolver.resolve_person_from_query("Emily")
        assert result.person_id == emily.id

    def test_unknown_name_returns_no_match(self, resolver, emily):
        result = resolver.resolve_person_from_query("Zara Thompson")
        assert result.person_id is None
        assert result.is_ambiguous is False


# ── Ambiguous queries ───────────────────────────────────────


class TestAmbiguousQueries:
    """Ambiguous queries return candidates for disambiguation."""

    def test_shared_first_name_is_ambiguous(self, store, resolver):
        store.upsert_person(name="Alex Smith")
        store.upsert_person(name="Alex Johnson")
        result = resolver.resolve_person_from_query("Alex")
        assert result.is_ambiguous is True
        assert result.person_id is None
        assert len(result.candidates) >= 2

    def test_ambiguous_result_contains_candidate_ids(self, store, resolver):
        p1 = store.upsert_person(name="Jordan Lee")
        p2 = store.upsert_person(name="Jordan Park")
        result = resolver.resolve_person_from_query("Jordan")
        assert result.is_ambiguous is True
        candidate_ids = {c.person_id for c in result.candidates}
        assert p1.id in candidate_ids
        assert p2.id in candidate_ids

    def test_full_name_disambiguates(self, store, resolver):
        store.upsert_person(name="Jordan Lee")
        store.upsert_person(name="Jordan Park")
        result = resolver.resolve_person_from_query("Jordan Lee")
        assert result.is_ambiguous is False
        assert result.person_id is not None

    def test_completely_vague_query(self, resolver, emily, john):
        result = resolver.resolve_person_from_query("What's the weather?")
        assert result.person_id is None
        assert result.is_ambiguous is False


# ── Disambiguation hints ────────────────────────────────────


class TestAmbiguousWithHints:
    """Ambiguous results include categorized fact hints."""

    def test_hints_populated_from_facts(self, store, resolver):
        p1 = store.upsert_person(name="Alex Smith")
        p2 = store.upsert_person(name="Alex Johnson")
        store.write_fact(p1.id, "tall with brown hair", fact_category="visual_descriptor")
        store.write_fact(p1.id, "works at Google", fact_category="affiliation")
        store.write_fact(p2.id, "short with glasses", fact_category="visual_descriptor")
        store.write_fact(p2.id, "likes tennis", fact_category="hobby")

        result = resolver.resolve_person_from_query("Alex")
        assert result.is_ambiguous is True

        by_name = {c.name: c for c in result.candidates}
        smith = by_name["Alex Smith"]
        johnson = by_name["Alex Johnson"]

        assert "tall with brown hair" in smith.hints.get("visual_descriptor", [])
        assert "works at Google" in smith.hints.get("affiliation", [])
        assert "short with glasses" in johnson.hints.get("visual_descriptor", [])
        assert "likes tennis" in johnson.hints.get("hobby", [])

    def test_hints_empty_when_no_facts(self, store, resolver):
        store.upsert_person(name="Alex Smith")
        store.upsert_person(name="Alex Johnson")
        result = resolver.resolve_person_from_query("Alex")
        assert result.is_ambiguous is True
        for candidate in result.candidates:
            assert candidate.hints == {}

    def test_hints_only_include_categorized_facts(self, store, resolver):
        p1 = store.upsert_person(name="Alex Smith")
        store.upsert_person(name="Alex Johnson")
        store.write_fact(p1.id, "uncategorized fact")
        store.write_fact(p1.id, "plays guitar", fact_category="hobby")

        result = resolver.resolve_person_from_query("Alex")
        smith = next(c for c in result.candidates if c.name == "Alex Smith")
        assert "plays guitar" in smith.hints.get("hobby", [])
        assert not any("uncategorized" in t for texts in smith.hints.values() for t in texts)


# ── Edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self, resolver):
        result = resolver.resolve_person_from_query("")
        assert result.person_id is None

    def test_whitespace_only(self, resolver):
        result = resolver.resolve_person_from_query("   ")
        assert result.person_id is None

    def test_special_characters(self, resolver, emily):
        result = resolver.resolve_person_from_query("@#$%^&*()")
        assert result.person_id is None

    def test_very_long_query(self, resolver, emily):
        long_text = "tell me about " + "Emily Chen " * 100
        result = resolver.resolve_person_from_query(long_text)
        assert result.person_id == emily.id

    def test_name_with_extra_whitespace(self, resolver, emily):
        result = resolver.resolve_person_from_query("  Emily   Chen  ")
        assert result.person_id == emily.id


# ── Result shape ────────────────────────────────────────────


class TestResultShape:
    def test_resolved_result_has_person_id(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.person_id, UUID)

    def test_result_has_is_ambiguous(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.is_ambiguous, bool)

    def test_result_has_candidates_list(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.candidates, list)

    def test_unresolved_result_shape(self, resolver):
        result = resolver.resolve_person_from_query("Nobody")
        assert result.person_id is None
        assert result.is_ambiguous is False
        assert result.candidates == []


# ── Multiple-person scenarios ───────────────────────────────


class TestMultiplePeople:
    def test_resolve_different_people_independently(self, resolver, emily, john):
        r1 = resolver.resolve_person_from_query("Emily Chen")
        r2 = resolver.resolve_person_from_query("John Rivera")
        assert r1.person_id == emily.id
        assert r2.person_id == john.id
        assert r1.person_id != r2.person_id
