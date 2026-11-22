"""Ranker 4: Jev over the whole filtered pool, with no bi-encoder stage."""

from __future__ import annotations

from ..interfaces import CorpusStore, DecisionClient, RankResult, RankTrace
from ..schema import TimeRange
from .base import DEFAULT_K, BaseRanker, StageTimer, fetch_summaries, top_k
from .jev_scoring import DEFAULT_SCORING, JevScoringConfig, score_candidates, trace_extra


class JevFullPoolRanker(BaseRanker):
    """Jev scores every conversation in the filtered pool; the top ``k`` are kept.

    There is no embedder here on purpose: this ranker measures what Jev does
    when nothing has pre-selected its candidates. Reading the pool is its first
    step, so it is timed. Candidates Jev scores equally stay in pool order
    (newest first); a candidate Jev gave no answer for ranks last.
    """

    name = "jev_full_pool"

    def __init__(
        self,
        store: CorpusStore,
        jev: DecisionClient,
        scoring: JevScoringConfig = DEFAULT_SCORING,
        k: int = DEFAULT_K,
    ) -> None:
        super().__init__(store, k)
        self.jev = jev
        self.scoring = scoring

    def rank_with_trace(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> RankResult:
        timer = StageTimer()
        with timer.stage("fetch_pool"):
            pool = self.store.pool_ids(owner_id, person_id=person_id, time_range=time_range)
        with timer.stage("fetch_summaries"):
            summaries = fetch_summaries(self.store, pool)
        with timer.stage("rerank"):
            jev = score_candidates(self.jev, query, list(zip(pool, summaries)), self.scoring)
        ranked = top_k(pool, [jev.scores[episode_id] for episode_id in pool], self.k)
        trace = RankTrace(
            ranker=self.name,
            pool_size=len(pool),
            latency_s=timer.elapsed(),
            stage_latency_s=timer.stages,
            usage=jev.usage,
            extra=trace_extra(jev, self.scoring),
        )
        return RankResult(ranked=ranked, trace=trace)
