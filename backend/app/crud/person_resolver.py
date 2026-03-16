from __future__ import annotations

import re
from uuid import UUID

from .memory_store import MemoryStore
from ..schema.person_resolver import ResolveCandidate, ResolveResult

ACTIVE_INTERLOCUTOR_PHRASES = {
    "this person",
    "the person i was talking to",
    "the person i'm speaking with",
    "the person i am speaking with",
}

PRONOUN_PATTERNS = {
    "they",
    "them",
    "him",
    "her",
}

# Confidence tiers for resolution results
ACTIVE_INTERLOCUTOR_CONFIDENCE = 0.95
EXACT_MATCH_CONFIDENCE = 0.95
SENTENCE_NAME_CONFIDENCE = 0.8
SENTENCE_ALIAS_CONFIDENCE = 0.7
FIRST_NAME_CONFIDENCE = 0.6
AMBIGUOUS_CONFIDENCE = 0.4
NO_MATCH_CONFIDENCE = 0.0

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


def _is_active_interlocutor_reference(text: str) -> bool:
    """Detect relative references that should use active interlocutor state."""

    return any(
        _contains_phrase(text, phrase)
        for phrase in ACTIVE_INTERLOCUTOR_PHRASES | PRONOUN_PATTERNS
    )


def _resolved(person_id: UUID, confidence: float) -> ResolveResult:
    """Build a resolved, non-ambiguous match result."""

    return ResolveResult(
        person_id=person_id,
        confidence=confidence,
        is_ambiguous=False,
        candidates=[],
    )


def _candidate(person_id: UUID, name: str, confidence: float) -> ResolveCandidate:
    """Build one ambiguity candidate entry."""

    return ResolveCandidate(
        person_id=person_id,
        name=name,
        confidence=confidence,
    )


def _ambiguous(
    candidates: list[ResolveCandidate],
    confidence: float = AMBIGUOUS_CONFIDENCE,
) -> ResolveResult:
    """Build an ambiguous result with candidate options."""

    return ResolveResult(
        person_id=None,
        confidence=confidence,
        is_ambiguous=True,
        candidates=candidates,
    )


def _no_match() -> ResolveResult:
    """Build a no-match result."""

    return ResolveResult(
        person_id=None,
        confidence=NO_MATCH_CONFIDENCE,
        is_ambiguous=False,
        candidates=[],
    )


class PersonResolver:
    """Resolve a referenced person from a user query."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._active_interlocutor_id: UUID | None = None

    def set_active_interlocutor(self, person_id: UUID | None) -> None:
        self._active_interlocutor_id = person_id

    def _match_name_or_alias_in_sentence(self, query_text: str) -> ResolveResult | None:
        """Resolve a full display name or alias mentioned inside longer text."""

        matches: dict[UUID, tuple[str, float]] = {}

        for person in self.store.list_people():
            display_name = person.display_name or person.first_name
            if display_name and _contains_phrase(query_text, display_name):
                matches[person.id] = (display_name, SENTENCE_NAME_CONFIDENCE)

            for alias in person.aliases:
                if _contains_phrase(query_text, alias.alias):
                    current = matches.get(person.id)
                    alias_confidence = SENTENCE_ALIAS_CONFIDENCE
                    if current is None or alias_confidence > current[1]:
                        matches[person.id] = (display_name, alias_confidence)

        if not matches:
            return None

        if len(matches) == 1:
            person_id, (_, confidence) = next(iter(matches.items()))
            return _resolved(person_id, confidence)

        candidates = [
            _candidate(person_id, name, confidence)
            for person_id, (name, confidence) in sorted(
                matches.items(),
                key=lambda item: (-item[1][1], item[1][0].lower()),
            )
        ]
        return _ambiguous(candidates)

    def _match_first_name(self, query_text: str) -> ResolveResult | None:
        """Resolve or disambiguate first-name references in the query text."""

        tokens = set(WORD_RE.findall(_normalize(query_text)))
        if not tokens:
            return None

        matches: list[tuple[UUID, str]] = []

        for person in self.store.list_people():
            first_name = _normalize(person.first_name)
            if first_name and first_name in tokens:
                matches.append((person.id, person.display_name or person.first_name))

        if not matches:
            return None

        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            person_id, _ = unique_matches[0]
            return _resolved(person_id, FIRST_NAME_CONFIDENCE)

        candidates = [
            _candidate(person_id, name, AMBIGUOUS_CONFIDENCE)
            for person_id, name in sorted(unique_matches, key=lambda item: item[1].lower())
        ]
        return _ambiguous(candidates, confidence=AMBIGUOUS_CONFIDENCE)

    def resolve_person_from_query(self, query_text: str) -> ResolveResult:
        """Determine which person a user query is referencing.

        Resolution happens in the following priority order:

        1. Exact display-name and alias matches (case-insensitive)
        2. Embedded full-name and alias mentions inside longer text
        3. First-name token matches (ambiguous if multiple people share a name)
        4. Active interlocutor phrases / pronoun references

        Name-based matching is attempted before pronoun or relative-reference
        resolution so that queries like ``"What did Emily tell him?"`` resolve
        to Emily rather than the active interlocutor.

        This function resolves identity only and does not retrieve profile data.

        Args:
            query_text: Natural-language query that may contain a person
                reference by name, alias, or relative pronoun.

        Returns:
            ResolveResult with person_id and confidence when a unique
            match is found, or is_ambiguous=True with candidates when
            multiple people match.
        """

        cleaned = _normalize(query_text)
        if not cleaned:
            return _no_match()

        # 1. Exact display-name / alias
        exact = self.store.resolve_person_by_name_or_alias(cleaned)
        if exact is not None:
            return _resolved(exact.id, confidence=EXACT_MATCH_CONFIDENCE)

        # 2. Name or alias embedded in a longer sentence
        sentence_match = self._match_name_or_alias_in_sentence(cleaned)
        if sentence_match is not None:
            return sentence_match

        # 3. First-name token match
        first_name_match = self._match_first_name(cleaned)
        if first_name_match is not None:
            return first_name_match

        # 4. Relative / pronoun reference to the active interlocutor
        if _is_active_interlocutor_reference(cleaned):
            if self._active_interlocutor_id is not None:
                return _resolved(
                    self._active_interlocutor_id,
                    confidence=ACTIVE_INTERLOCUTOR_CONFIDENCE,
                )

        return _no_match()


__all__ = ["PersonResolver"]
