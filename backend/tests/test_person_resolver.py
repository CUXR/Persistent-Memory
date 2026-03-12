"""
Tests for the person_resolver module.

Issue #2 Acceptance Criteria:
  - Query referencing current interlocutor resolves correctly
  - Query referencing known name resolves correctly
  - Alias resolution works
  - Ambiguous queries result in requests for further information
  - No console errors
  - Module only resolves identity (no retrieval)
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from uuid import UUID, uuid4

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.crud.memory_store import MemoryStore

# The module under test -- will fail until person_resolver.py is created.
from app.crud.person_resolver import PersonResolver


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def store():
    """Fresh in-memory database for each test."""
    s = MemoryStore("sqlite+pysqlite:///:memory:")
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def resolver(store: MemoryStore):
    """PersonResolver backed by the in-memory store."""
    return PersonResolver(store)


@pytest.fixture
def emily(store: MemoryStore):
    """Person: Emily Chen with aliases Em, Dr. Chen."""
    return store.upsert_person(
        name="Emily Chen",
        aliases=["Em", "Dr. Chen"],
        face_key="face_emily_001",
        voice_key="voice_emily_001",
        persona90=[0.5] * 90,
    )


@pytest.fixture
def john(store: MemoryStore):
    """Person: John Rivera with alias Johnny."""
    return store.upsert_person(
        name="John Rivera",
        aliases=["Johnny"],
        face_key="face_john_001",
        voice_key="voice_john_001",
    )


@pytest.fixture
def sarah(store: MemoryStore):
    """Person: Sarah Martinez with alias S."""
    return store.upsert_person(
        name="Sarah Martinez",
        aliases=["S"],
    )


@pytest.fixture
def recent_episode(store, emily, john):
    """Episode where Emily and John participated recently."""
    now = datetime.now(timezone.utc)
    return store.write_episode(
        time_start=now - timedelta(minutes=10),
        time_end=now,
        transcript="Emily: Hi John!\nJohn: Hey Emily!",
        summary="Casual greeting between Emily and John.",
        participants=[emily.id, john.id],
    )


# ── Canonical name resolution ───────────────────────────────


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
        """Extract name from a natural-language query."""
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


# ── Alias resolution ────────────────────────────────────────


class TestResolveByAlias:
    """Alias resolution works."""

    def test_alias_exact(self, resolver, emily):
        result = resolver.resolve_person_from_query("Em")
        assert result.person_id == emily.id

    def test_alias_case_insensitive(self, resolver, emily):
        result = resolver.resolve_person_from_query("dr. chen")
        assert result.person_id == emily.id

    def test_alias_in_sentence(self, resolver, emily):
        result = resolver.resolve_person_from_query("Tell me about Dr. Chen")
        assert result.person_id == emily.id

    def test_alias_for_second_person(self, resolver, john):
        result = resolver.resolve_person_from_query("Johnny")
        assert result.person_id == john.id


# ── Active interlocutor resolution ──────────────────────────


class TestResolveActiveInterlocutor:
    """Query referencing current interlocutor resolves correctly."""

    def test_the_person_i_was_talking_to(self, resolver, emily, recent_episode):
        resolver.set_active_interlocutor(emily.id)
        result = resolver.resolve_person_from_query(
            "the person I was talking to"
        )
        assert result.person_id == emily.id

    def test_this_person(self, resolver, emily, recent_episode):
        resolver.set_active_interlocutor(emily.id)
        result = resolver.resolve_person_from_query("this person")
        assert result.person_id == emily.id

    def test_them(self, resolver, emily, recent_episode):
        resolver.set_active_interlocutor(emily.id)
        result = resolver.resolve_person_from_query("What do they like?")
        assert result.person_id == emily.id

    def test_current_speaker(self, resolver, john, recent_episode):
        resolver.set_active_interlocutor(john.id)
        result = resolver.resolve_person_from_query(
            "the person I'm speaking with"
        )
        assert result.person_id == john.id

    def test_no_active_interlocutor_set(self, resolver):
        """Relative reference without an active interlocutor returns no match."""
        result = resolver.resolve_person_from_query(
            "the person I was talking to"
        )
        assert result.person_id is None


# ── Ambiguous queries ───────────────────────────────────────


class TestAmbiguousQueries:
    """Ambiguous queries result in requests for further information."""

    def test_shared_first_name_is_ambiguous(self, store, resolver):
        """Two people named 'Alex' should produce an ambiguous result."""
        store.upsert_person(name="Alex Smith", aliases=[])
        store.upsert_person(name="Alex Johnson", aliases=[])
        result = resolver.resolve_person_from_query("Alex")
        assert result.is_ambiguous is True
        assert result.person_id is None
        assert len(result.candidates) >= 2

    def test_ambiguous_result_contains_candidate_ids(self, store, resolver):
        p1 = store.upsert_person(name="Jordan Lee", aliases=["JL"])
        p2 = store.upsert_person(name="Jordan Park", aliases=["JP"])
        result = resolver.resolve_person_from_query("Jordan")
        assert result.is_ambiguous is True
        candidate_ids = {c.person_id for c in result.candidates}
        assert p1.id in candidate_ids
        assert p2.id in candidate_ids

    def test_full_name_disambiguates(self, store, resolver):
        """Using the full name should not be ambiguous."""
        store.upsert_person(name="Jordan Lee", aliases=[])
        store.upsert_person(name="Jordan Park", aliases=[])
        result = resolver.resolve_person_from_query("Jordan Lee")
        assert result.is_ambiguous is False
        assert result.person_id is not None

    def test_completely_vague_query(self, resolver, emily, john):
        """A query with no person reference at all."""
        result = resolver.resolve_person_from_query("What's the weather?")
        assert result.person_id is None
        assert result.is_ambiguous is False


# ── Confidence threshold ────────────────────────────────────


class TestConfidenceThreshold:
    """Return resolved person_id only if above certain confidence threshold."""

    def test_exact_match_above_threshold(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert result.confidence >= 0.8

    def test_alias_match_above_threshold(self, resolver, emily):
        result = resolver.resolve_person_from_query("Em")
        assert result.confidence >= 0.5

    def test_no_match_confidence_is_zero(self, resolver, emily):
        result = resolver.resolve_person_from_query("Nonexistent Person")
        assert result.confidence == 0.0

    def test_vague_reference_low_confidence(self, resolver, emily, john):
        """A vague query should not produce high confidence."""
        result = resolver.resolve_person_from_query("someone")
        assert result.confidence < 0.5


# ── Edge cases and input validation ─────────────────────────


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
    """Verify the resolver returns a well-typed result object."""

    def test_resolved_result_has_person_id(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.person_id, UUID)

    def test_result_has_confidence(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_is_ambiguous(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.is_ambiguous, bool)

    def test_result_has_candidates_list(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert isinstance(result.candidates, list)

    def test_unresolved_result_shape(self, resolver):
        result = resolver.resolve_person_from_query("Nobody")
        assert result.person_id is None
        assert result.confidence == 0.0
        assert result.is_ambiguous is False
        assert result.candidates == []


# ── Identity-only boundary ──────────────────────────────────


class TestIdentityOnlyBoundary:
    """Module only resolves identity; no retrieval should occur."""

    def test_resolve_does_not_return_facts(self, resolver, store, emily):
        store.write_fact(person_id=emily.id, fact_text="Likes cats")
        result = resolver.resolve_person_from_query("Emily Chen")
        assert not hasattr(result, "facts")

    def test_resolve_does_not_return_profile(self, resolver, emily):
        result = resolver.resolve_person_from_query("Emily Chen")
        assert not hasattr(result, "profile")
        assert not hasattr(result, "prefs")
        assert not hasattr(result, "summaries")


# ── Multiple-person scenarios ───────────────────────────────


class TestMultiplePeople:
    def test_resolve_different_people_independently(self, resolver, emily, john):
        r1 = resolver.resolve_person_from_query("Emily Chen")
        r2 = resolver.resolve_person_from_query("John Rivera")
        assert r1.person_id == emily.id
        assert r2.person_id == john.id
        assert r1.person_id != r2.person_id

    def test_resolve_by_alias_when_multiple_people_exist(
        self, resolver, emily, john, sarah
    ):
        result = resolver.resolve_person_from_query("Johnny")
        assert result.person_id == john.id

    def test_switching_active_interlocutor(self, resolver, emily, john, recent_episode):
        resolver.set_active_interlocutor(emily.id)
        r1 = resolver.resolve_person_from_query("this person")
        assert r1.person_id == emily.id

        resolver.set_active_interlocutor(john.id)
        r2 = resolver.resolve_person_from_query("this person")
        assert r2.person_id == john.id
