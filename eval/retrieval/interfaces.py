"""Protocols and result records that connect the benchmark's modules.

Rankers, the system pipeline, and the runner depend on these protocols rather
than on concrete clients, so unit tests can inject fakes and never touch the
network, a database, or a model download.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .schema import Conversation, ConversationHeader, Person, TimeRange


@dataclass
class Usage:
    """Hosted-API usage. ``latency_s`` sums the latency of the individual calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
            calls=self.calls + other.calls,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "calls": self.calls,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "Usage":
        data = data or {}
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            latency_s=float(data.get("latency_s", 0.0)),
            calls=int(data.get("calls", 0)),
        )


@dataclass
class LLMResult:
    data: dict[str, Any]  # the parsed JSON object the model returned
    usage: Usage  # as measured on the original (uncached) call
    # The model id the gateway reports as having served the call (a gateway may
    # pin a request for "x" to a dated snapshot "x-2026-10-01"); the requested
    # id when the response names none. Same convention as DecisionResult.model.
    model: str
    cached: bool = False
    raw_text: str = ""


@runtime_checkable
class ChatLLM(Protocol):
    """Chat model that returns one JSON object per call. Must be thread-safe."""

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        namespace: str = "default",
        salt: str = "",
    ) -> LLMResult:
        """``namespace`` groups cache entries; ``salt`` forces a distinct cache key."""
        ...


@dataclass
class DecisionResult:
    """Answers from a decision model, normalized to the OpenRouter shape.

    noul:    {"type": "noul", "noul": p}
    score:   {"type": "score", "score": s, "probabilities": {"0": p0, ...}, "confidence": c}
    choice:  {"type": "choice", "choice": name, "probabilities": {name: p, ...}, "confidence": c}
    refusal: {"type": "refusal"} when the model declined to answer that question

    ``confidence`` is None for answers that came through the Vercel gateway,
    which does not report it. ``model`` follows the convention of ``LLMResult``.
    """

    answers: dict[str, dict[str, Any]]
    usage: Usage
    model: str  # the dated snapshot that served the request
    cached: bool = False


@runtime_checkable
class DecisionClient(Protocol):
    """Hosted decision model (Jev): one shared ``state``, many typed questions.

    ``decide`` must be thread-safe (cache writes and the spend cap included):
    rankers call it from several threads at once.
    """

    model: str

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        salt: str = "",
    ) -> DecisionResult:
        ...


@runtime_checkable
class Embedder(Protocol):
    """Bi-encoder. Vectors are L2-normalized, so a dot product is the cosine."""

    model_name: str

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@runtime_checkable
class PairScorer(Protocol):
    """Cross-encoder: one relevance score per (query, text) pair, higher is better."""

    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        ...


@runtime_checkable
class CorpusStore(Protocol):
    """Read access to one owner's conversations.

    ``time_range`` is the already-padded filter; a conversation matches when
    ``time_range.start <= start_time <= time_range.end``.
    """

    def list_people(self, owner_id: str) -> list[Person]:
        """Everyone the owner has talked to, ordered by name, then person id.

        The name is the display name, else first and last name. Stores sort in
        Python, so the order does not depend on a database collation.
        """
        ...

    def pool_ids(
        self,
        owner_id: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[str]:
        """Episode ids in the filtered pool, newest first (ties by episode id)."""
        ...

    def vector_search(
        self,
        owner_id: str,
        query_embedding: Sequence[float],
        person_id: str | None = None,
        time_range: TimeRange | None = None,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Top ``limit`` (episode_id, cosine similarity) in the filtered pool.

        Ordered by cosine descending, then episode id, so every store cuts the
        list at the same place when similarities tie.
        """
        ...

    def get_summaries(self, episode_ids: Sequence[str]) -> dict[str, str]:
        """Summary text only: the single view of a conversation that rankers get."""
        ...

    def get_headers(self, episode_ids: Sequence[str]) -> dict[str, ConversationHeader]:
        """Summary plus date and person name, for the answering step's summary path."""
        ...

    def get_conversations(self, episode_ids: Sequence[str]) -> dict[str, Conversation]:
        """Full records, including the dialogue (used only by the answer fallback and judges).

        Every record carries ``meta["person_name"]``, the name of the person it
        is with: the answer fallback labels a dialogue with it when none of the
        summaries it saw is with that person, and the relevance judge shows it.
        Unknown ids are left out.
        """
        ...

    def corpus_fingerprint(self) -> str | None:
        """Which corpus is served: ``db.dataset_fingerprint`` of the people and conversations.

        A store that computes it from what it holds returns it; one that reads
        what a load recorded returns None when nothing was recorded. The run
        states it in its report, and a ``pg`` run checks it against the
        committed dataset before ranking.
        """
        ...


@dataclass
class RankTrace:
    """Diagnostics for one ``rank`` call."""

    ranker: str
    pool_size: int
    latency_s: float  # wall clock for the whole call
    stage1_ids: list[str] | None = None  # what bge-m3 handed to the second stage (None: no stage 1)
    stage_latency_s: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)  # hosted-model usage; zero for local models
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankResult:
    ranked: list[tuple[str, float]]  # (episode_id, score), best first, at most k items
    trace: RankTrace


@runtime_checkable
class Ranker(Protocol):
    """Reduce the filtered pool to the top k conversations. Sees summaries only."""

    name: str

    def rank(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[tuple[str, float]]:
        ...

    def rank_with_trace(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> RankResult:
        ...
