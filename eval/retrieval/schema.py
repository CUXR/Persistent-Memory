"""Dataset record types shared by every eval/retrieval module.

These mirror the committed JSONL files in ``eval/retrieval/data``. Keep this
module dependency-free (stdlib only) so it imports in any environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

QUERY_TYPES: tuple[str, ...] = (
    "lookup",  # single-conversation lookup
    "paraphrase",  # single conversation; the question avoids the summary's wording
    "detail_only",  # the answer is in the dialogue but not in the summary
    "multi_conversation",  # needs two or more conversations with one person
    "cross_person",  # no name in the query ("who did I talk to about climbing?")
    "temporal",  # carries a time expression ("what did Sam say last week?")
    "unanswerable",  # the person exists but no conversation answers it (system eval only)
    "unknown_name",  # nobody by that name exists (system eval only)
)
ANSWERABLE_TYPES: tuple[str, ...] = QUERY_TYPES[:6]
SYSTEM_ONLY_TYPES: tuple[str, ...] = QUERY_TYPES[6:]
POOL_BUCKETS: tuple[str, ...] = ("<=20", "21-100", ">100")
OWNER_SPEAKER = "Me"

# Vocabulary of the system eval, spelled once here and used by the pipeline
# (which produces the values), the judges and the metrics (which consume them).
# The stages of the query flow, in order: keys of a system record's
# ``usage_by_stage`` / ``latency_by_stage``.
SYSTEM_STAGES: tuple[str, ...] = ("extraction", "rank", "answer_summary", "answer_dialogue")
# Where the flow ended for one query.
SYSTEM_PATHS: tuple[str, ...] = (
    "unknown_name",  # the extracted name matches nobody: the exact error message, no ranking
    "ambiguous",  # the extracted name matches several people: a clarification, no ranking
    "summary",  # answered from the top-k summaries
    "fallback",  # answered from the top-k' full dialogues after the summaries could not
    "declined",  # neither attempt could answer: the decline answer
    "empty_pool",  # the filters left nothing to rank: the decline answer, no LLM call
)
MISROUTED_PATHS: frozenset[str] = frozenset({"unknown_name", "ambiguous"})  # stopped before ranking
DECLINE_PATHS: frozenset[str] = frozenset({"declined", "empty_pool"})  # ended without an answer
# The answer judge's grades.
VERDICTS: tuple[str, ...] = ("correct", "partial", "incorrect")


def parse_dt(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime (naive means UTC)."""

    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_dt(value: datetime) -> str:
    """Serialize a datetime as ISO-8601 UTC with a ``Z`` suffix."""

    return parse_dt(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def pool_bucket(pool_size: int) -> str:
    """Map a filtered-pool size to its reporting bucket."""

    if pool_size <= 20:
        return POOL_BUCKETS[0]
    if pool_size <= 100:
        return POOL_BUCKETS[1]
    return POOL_BUCKETS[2]


@dataclass(frozen=True)
class TimeRange:
    """Closed interval ``[start, end]`` of aware UTC datetimes.

    A conversation falls inside the range when its ``start_time`` does.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", parse_dt(self.start))
        object.__setattr__(self, "end", parse_dt(self.end))
        if self.end < self.start:
            raise ValueError(f"time range ends before it starts: {self.start} > {self.end}")

    def contains(self, moment: datetime) -> bool:
        return self.start <= parse_dt(moment) <= self.end

    def to_json(self) -> dict[str, str]:
        return {"start": format_dt(self.start), "end": format_dt(self.end)}

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> "TimeRange | None":
        if not data:
            return None
        return cls(start=parse_dt(data["start"]), end=parse_dt(data["end"]))


@dataclass(frozen=True)
class Person:
    person_id: str
    first_name: str
    last_name: str
    display_name: str
    relationship: str = ""
    is_heavy: bool = False
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "display_name": self.display_name,
            "relationship": self.relationship,
            "is_heavy": self.is_heavy,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Person":
        return cls(
            person_id=str(data["person_id"]),
            first_name=data["first_name"],
            last_name=data.get("last_name", ""),
            display_name=data["display_name"],
            relationship=data.get("relationship", ""),
            is_heavy=bool(data.get("is_heavy", False)),
            meta=dict(data.get("meta") or {}),
        )


@dataclass(frozen=True)
class Turn:
    speaker: str  # the other person's first name, or OWNER_SPEAKER ("Me") for the owner
    text: str


@dataclass(frozen=True)
class ConversationHeader:
    """What the answering step sees on the summary path: no dialogue."""

    episode_id: str
    person_id: str
    person_name: str
    start_time: datetime
    summary: str


@dataclass
class Conversation:
    """One memory: a dialogue, its summary, and the existing episode metadata."""

    episode_id: str
    person_id: str
    start_time: datetime
    end_time: datetime
    dialogue: list[Turn]
    summary: str
    meta: dict[str, Any] = field(default_factory=dict)

    def dialogue_text(self) -> str:
        """Render the structured dialogue as ``Sam: "..."`` / ``Me: "..."`` lines."""

        return "\n".join(f'{turn.speaker}: "{turn.text}"' for turn in self.dialogue)

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "person_id": self.person_id,
            "start_time": format_dt(self.start_time),
            "end_time": format_dt(self.end_time),
            "dialogue": [{"speaker": turn.speaker, "text": turn.text} for turn in self.dialogue],
            "summary": self.summary,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Conversation":
        return cls(
            episode_id=str(data["episode_id"]),
            person_id=str(data["person_id"]),
            start_time=parse_dt(data["start_time"]),
            end_time=parse_dt(data["end_time"]),
            dialogue=[Turn(speaker=t["speaker"], text=t["text"]) for t in data.get("dialogue", [])],
            summary=data.get("summary", ""),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class Query:
    """One benchmark query with its gold labels.

    ``gold_name`` is the name as a correct extraction returns it (None when the
    query names nobody). ``gold_person_id`` is the person that name resolves to
    (None for no-name and unknown-name queries). ``gold_time_range`` is the
    unpadded range a correct extraction returns (None without a time expression).
    """

    query_id: str
    query: str
    type: str
    gold_name: str | None
    gold_time_range: TimeRange | None
    gold_episode_ids: list[str]
    gold_answer: str | None
    split: str = "test"  # "dev" | "test"
    gold_person_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in QUERY_TYPES:
            raise ValueError(f"unknown query type: {self.type!r}")

    @property
    def answerable(self) -> bool:
        return self.type in ANSWERABLE_TYPES

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "type": self.type,
            "gold_name": self.gold_name,
            "gold_time_range": self.gold_time_range.to_json() if self.gold_time_range else None,
            "gold_episode_ids": list(self.gold_episode_ids),
            "gold_answer": self.gold_answer,
            "split": self.split,
            "gold_person_id": self.gold_person_id,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Query":
        return cls(
            query_id=str(data["query_id"]),
            query=data["query"],
            type=data["type"],
            gold_name=data.get("gold_name"),
            gold_time_range=TimeRange.from_json(data.get("gold_time_range")),
            gold_episode_ids=[str(e) for e in data.get("gold_episode_ids", [])],
            gold_answer=data.get("gold_answer"),
            split=data.get("split", "test"),
            gold_person_id=data.get("gold_person_id"),
            meta=dict(data.get("meta") or {}),
        )
