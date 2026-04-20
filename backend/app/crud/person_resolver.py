from __future__ import annotations

import re
from uuid import UUID

from .memory_store import MemoryStore
from ..models.person import Person
from ..schema.person_resolver import ResolveCandidate, ResolveResult

WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> str:
    """Lowercase, trim, and collapse internal whitespace."""

    return WHITESPACE_RE.sub(" ", text.strip().lower())


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return True when a phrase appears with word boundaries."""

    normalized_text = _normalize(text)
    normalized_phrase = _normalize(phrase)
    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    return re.search(pattern, normalized_text) is not None


def _person_name(person: Person) -> str:
    """Return the best available display name for a person."""

    return person.display_name or f"{person.first_name} {person.last_name}".strip()


class PersonResolver:
    """Resolve the subject of a user query to a known person.

    Given a query such as "what did Bob say last week", determines which
    person in the memory store is being referenced. When multiple people
    match, returns candidates with categorized fact hints for disambiguation
    by a downstream model (GPT 5 nano).
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def resolve_person_from_query(self, query_text: str) -> ResolveResult:
        """Determine which person a user query is referencing.

        Resolution:
        1. Display-name match (handles both exact and embedded in sentence)
        2. First-name token fallback

        At each step: 0 matches continues, 1 match resolves, 2+ returns
        ambiguous candidates with fact hints for disambiguation.

        Args:
            query_text: Natural-language query containing a person name.

        Returns:
            ResolveResult with person_id when a unique match is found,
            or is_ambiguous=True with candidates when multiple match.
        """

        cleaned = _normalize(query_text)
        if not cleaned:
            return _no_match()

        people = self.store.list_people()
        if not people:
            return _no_match()

        # Pass 1: display-name match
        name_matches = [p for p in people if _contains_phrase(cleaned, _person_name(p))]
        result = self._branch(name_matches)
        if result is not None:
            return result

        # Pass 2: first-name token fallback
        tokens = set(WORD_RE.findall(cleaned))
        first_matches = [
            p for p in people
            if _normalize(p.first_name) in tokens
        ]
        result = self._branch(first_matches)
        if result is not None:
            return result

        return _no_match()

    def _branch(self, matches: list[Person]) -> ResolveResult | None:
        """Apply the 0/1/N branching logic for a set of matches."""

        if not matches:
            return None

        # Deduplicate (same person matched via different paths)
        seen: dict[UUID, Person] = {}
        for p in matches:
            if p.id not in seen:
                seen[p.id] = p
        unique = list(seen.values())

        if len(unique) == 1:
            return ResolveResult(
                person_id=unique[0].id,
                is_ambiguous=False,
                candidates=[],
            )

        candidates = [
            ResolveCandidate(
                person_id=p.id,
                name=_person_name(p),
                hints=self.store.get_disambiguation_hints(p.id),
            )
            for p in sorted(unique, key=lambda p: _person_name(p).lower())
        ]
        return ResolveResult(
            person_id=None,
            is_ambiguous=True,
            candidates=candidates,
        )


def _no_match() -> ResolveResult:
    """Build a no-match result."""

    return ResolveResult(
        person_id=None,
        is_ambiguous=False,
        candidates=[],
    )


__all__ = ["PersonResolver"]
