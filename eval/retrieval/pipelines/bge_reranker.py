"""Ranker 2: bge-m3 stage 1, reranked by the bge-reranker-v2-m3 cross-encoder."""

from __future__ import annotations

from .. import config
from ..interfaces import CorpusStore, Embedder, PairScorer, RankResult, RankTrace
from ..schema import TimeRange
from .base import DEFAULT_K, Stage1Ranker, StageTimer, fetch_summaries, top_k


class BgeRerankerRanker(Stage1Ranker):
    """bge-m3 top ``stage1_k`` (the whole pool if it is smaller), reranked, top ``k`` kept.

    The cross-encoder scores each (query, summary) pair; candidates it scores
    equally stay in stage-1 order.
    """

    name = "bge_m3_reranker"

    def __init__(
        self,
        store: CorpusStore,
        embedder: Embedder,
        scorer: PairScorer,
        k: int = DEFAULT_K,
        stage1_k: int = config.STAGE1_TOP_K,
    ) -> None:
        super().__init__(store, embedder, k, stage1_k)
        self.scorer = scorer

    def rank_with_trace(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> RankResult:
        timer = StageTimer()
        stage1 = self._stage1(timer, owner_id, query, person_id, time_range)
        stage1_ids = [episode_id for episode_id, _ in stage1]
        with timer.stage("fetch_summaries"):
            summaries = fetch_summaries(self.store, stage1_ids)
        with timer.stage("rerank"):
            scores = self.scorer.score(query, summaries) if summaries else []
        ranked = top_k(stage1_ids, scores, self.k)
        latency_s = timer.elapsed()
        trace = RankTrace(
            ranker=self.name,
            pool_size=self._pool_size(owner_id, person_id, time_range),
            latency_s=latency_s,
            stage1_ids=stage1_ids,
            stage_latency_s=timer.stages,
        )
        return RankResult(ranked=ranked, trace=trace)
