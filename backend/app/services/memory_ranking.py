"""Pure scoring functions for ranking a person's stored memories.

Used by :meth:`~app.crud.memory_store.MemoryStore.get_relevant_memories` to
decide which facts and summaries are worth surfacing when a person is
re-encountered. Kept dependency-free (no ORM/session access) so the scoring
math can be unit tested in isolation from the database.

Score = weighted blend of:
  - recency:   exponential decay from the memory's anchor timestamp
  - relevance: fact confidence, or the source episode's importance_score
  - similarity: cosine similarity against a caller-supplied query embedding
                (optional — omitted when no embedding is available, and the
                remaining weights are redistributed proportionally)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Sequence

DEFAULT_HALF_LIFE_DAYS = 14.0
DEFAULT_RECENCY_WEIGHT = 0.4
DEFAULT_RELEVANCE_WEIGHT = 0.3
DEFAULT_SIMILARITY_WEIGHT = 0.3

# Default relevance for a memory with no confidence/importance signal at all.
NEUTRAL_RELEVANCE = 0.5


def ensure_aware(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC (SQLite drops tzinfo on round-trip)."""

    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def recency_score(anchor: datetime, now: datetime, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Exponential recency decay in (0, 1] — 1.0 for "just happened".

    Halves every ``half_life_days``; clamps future-dated anchors to age 0
    rather than boosting them above 1.0.
    """

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    age_days = max((now - ensure_aware(anchor)).total_seconds(), 0.0) / 86400.0
    return 0.5 ** (age_days / half_life_days)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for empty, mismatched, or zero vectors."""

    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def score_memory(
    *,
    recency: float,
    relevance: float,
    similarity: Optional[float],
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
    similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT,
) -> float:
    """Blend recency, relevance, and optional similarity into one score.

    When ``similarity`` is ``None`` (no query embedding, or the item has no
    stored embedding), its weight is redistributed proportionally across the
    other two components so scores stay comparable whether or not a query
    embedding was supplied.
    """

    if similarity is None:
        remaining = recency_weight + relevance_weight
        if remaining <= 0:
            return 0.0
        return (recency_weight / remaining) * recency + (relevance_weight / remaining) * relevance

    return recency_weight * recency + relevance_weight * relevance + similarity_weight * similarity


def relative_time_phrase(anchor: datetime, now: datetime) -> str:
    """Human-readable relative time, e.g. "two weeks ago" -> "2 weeks ago"."""

    age_days = max((now - ensure_aware(anchor)).total_seconds(), 0.0) / 86400.0

    if age_days < 1:
        return "today"
    if age_days < 2:
        return "yesterday"
    if age_days < 7:
        n = int(age_days)
        return f"{n} day{'s' if n != 1 else ''} ago"
    if age_days < 30:
        n = round(age_days / 7)
        return f"{n} week{'s' if n != 1 else ''} ago"
    if age_days < 365:
        n = round(age_days / 30)
        return f"{n} month{'s' if n != 1 else ''} ago"
    n = round(age_days / 365)
    return f"{n} year{'s' if n != 1 else ''} ago"


def build_interaction_context(
    *,
    first_met_at: Optional[datetime],
    first_met_summary: Optional[str],
    now: datetime,
) -> str:
    """Short human-readable note on when/how this person was first encountered."""

    if first_met_at is None:
        return "No prior encounters on record."

    phrase = relative_time_phrase(first_met_at, now)
    if first_met_summary:
        return f"Met {phrase}: {first_met_summary}"
    return f"First met {phrase}."


__all__ = [
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_RECENCY_WEIGHT",
    "DEFAULT_RELEVANCE_WEIGHT",
    "DEFAULT_SIMILARITY_WEIGHT",
    "NEUTRAL_RELEVANCE",
    "build_interaction_context",
    "cosine_similarity",
    "ensure_aware",
    "recency_score",
    "relative_time_phrase",
    "score_memory",
]
