"""Shared building blocks of the four rankers.

Every ranker honours one contract (``interfaces.Ranker``): it reads candidate
text only through ``store.get_summaries`` and returns at most ``k``
``(episode_id, score)`` pairs, best first, in an order that does not depend on
``k``. The runner can therefore rank once at the deepest cut-off and take
prefixes of the result for the smaller ones.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from .. import config
from ..interfaces import CorpusStore, Embedder, RankResult
from ..schema import TimeRange

DEFAULT_K = max(config.RECALL_KS)  # the deepest cut-off the report needs


class StageTimer:
    """Wall clock of one rank call: the whole algorithm and each named stage."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Record how long the ``with`` block takes under ``name``."""

        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = time.perf_counter() - started

    def elapsed(self) -> float:
        """Seconds since the timer was created."""

        return time.perf_counter() - self._started


def top_k(episode_ids: Sequence[str], scores: Sequence[float], k: int) -> list[tuple[str, float]]:
    """The ``k`` best ``(episode_id, score)`` pairs, best first.

    The sort is stable, so candidates with equal scores keep the order they
    were given in: stage-1 order, or pool order for the full-pool ranker.
    """

    if len(scores) != len(episode_ids):
        raise ValueError(f"got {len(scores)} scores for {len(episode_ids)} candidates")
    pairs = [(episode_id, float(score)) for episode_id, score in zip(episode_ids, scores)]
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)[:k]


def fetch_summaries(store: CorpusStore, episode_ids: Sequence[str]) -> list[str]:
    """Summaries of ``episode_ids``, in that order: the only candidate text a ranker reads."""

    if not episode_ids:
        return []
    summaries = store.get_summaries(episode_ids)
    missing = [episode_id for episode_id in episode_ids if episode_id not in summaries]
    if missing:
        raise LookupError(
            f"the store has no summary for {len(missing)} of {len(episode_ids)} candidates "
            f"(first: {missing[0]})"
        )
    return [summaries[episode_id] for episode_id in episode_ids]


class BaseRanker(ABC):
    """What all rankers share: the store, ``k``, and ``rank`` as a view of ``rank_with_trace``."""

    name: str

    def __init__(self, store: CorpusStore, k: int = DEFAULT_K) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.store = store
        self.k = k

    def rank(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[tuple[str, float]]:
        """The top ``k`` of the filtered pool as ``(episode_id, score)`` pairs, best first."""

        return self.rank_with_trace(owner_id, query, person_id, time_range).ranked

    @abstractmethod
    def rank_with_trace(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> RankResult:
        """``rank`` plus the diagnostics the benchmark reports (latency, stage 1, usage)."""


class Stage1Ranker(BaseRanker):
    """A ranker whose first stage is the bge-m3 vector search (rankers 1-3)."""

    def __init__(
        self,
        store: CorpusStore,
        embedder: Embedder,
        k: int = DEFAULT_K,
        stage1_k: int = config.STAGE1_TOP_K,
    ) -> None:
        super().__init__(store, k)
        if stage1_k < 1:
            raise ValueError(f"stage1_k must be at least 1, got {stage1_k}")
        self.embedder = embedder
        self.stage1_k = stage1_k

    def _stage1(
        self,
        timer: StageTimer,
        owner_id: str,
        query: str,
        person_id: str | None,
        time_range: TimeRange | None,
    ) -> list[tuple[str, float]]:
        """The ``stage1_k`` summaries nearest to the query, as ``(episode_id, cosine)``.

        This is the whole filtered pool when it holds fewer than ``stage1_k``
        conversations. Equal cosines are ordered by episode id, so the stage-1
        order (the tie-break of the second stage) does not depend on the store.
        """

        with timer.stage("embed_query"):
            query_embedding = self.embedder.embed_query(query)
        with timer.stage("vector_search"):
            hits = self.store.vector_search(
                owner_id,
                query_embedding,
                person_id=person_id,
                time_range=time_range,
                limit=self.stage1_k,
            )
        ordered = sorted(
            ((episode_id, float(cosine)) for episode_id, cosine in hits),
            key=lambda hit: (-hit[1], hit[0]),
        )
        return ordered[: self.stage1_k]

    def _pool_size(
        self,
        owner_id: str,
        person_id: str | None,
        time_range: TimeRange | None,
    ) -> int:
        """Size of the filtered pool, for the trace.

        These rankers never need the pool itself, so this is a diagnostic:
        callers take it after they have read the clock, which keeps it out of
        the reported latency and stops it warming the store for the search.
        """

        return len(self.store.pool_ids(owner_id, person_id=person_id, time_range=time_range))
