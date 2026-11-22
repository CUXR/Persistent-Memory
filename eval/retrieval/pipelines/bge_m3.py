"""Ranker 1: bge-m3 only."""

from __future__ import annotations

from ..interfaces import RankResult, RankTrace
from ..schema import TimeRange
from .base import Stage1Ranker, StageTimer


class BgeM3Ranker(Stage1Ranker):
    """Top ``stage1_k`` summaries by cosine to the query, truncated to ``k``.

    This is the baseline: the other bi-encoder rankers start from the same
    search and differ only in what they do with its result.
    """

    name = "bge_m3"

    def rank_with_trace(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> RankResult:
        timer = StageTimer()
        ranked = self._stage1(timer, owner_id, query, person_id, time_range)[: self.k]
        latency_s = timer.elapsed()
        trace = RankTrace(
            ranker=self.name,
            pool_size=self._pool_size(owner_id, person_id, time_range),
            latency_s=latency_s,
            stage_latency_s=timer.stages,
        )
        return RankResult(ranked=ranked, trace=trace)
