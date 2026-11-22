"""Ranker 3: bge-m3 stage 1, reranked by Jev."""

from __future__ import annotations

from .. import config
from ..interfaces import CorpusStore, DecisionClient, Embedder, RankResult, RankTrace
from ..schema import TimeRange
from .base import DEFAULT_K, Stage1Ranker, StageTimer, fetch_summaries, top_k
from .jev_scoring import DEFAULT_SCORING, JevScoringConfig, score_candidates, trace_extra


class BgeJevRanker(Stage1Ranker):
    """bge-m3 top ``stage1_k`` (the whole pool if it is smaller), scored by Jev, top ``k`` kept.

    ``scoring`` says how Jev is applied. Candidates Jev scores equally stay in
    stage-1 order; a candidate Jev gave no answer for ranks last.
    """

    name = "bge_m3_jev"

    def __init__(
        self,
        store: CorpusStore,
        embedder: Embedder,
        jev: DecisionClient,
        scoring: JevScoringConfig = DEFAULT_SCORING,
        k: int = DEFAULT_K,
        stage1_k: int = config.STAGE1_TOP_K,
    ) -> None:
        super().__init__(store, embedder, k, stage1_k)
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
        stage1 = self._stage1(timer, owner_id, query, person_id, time_range)
        stage1_ids = [episode_id for episode_id, _ in stage1]
        with timer.stage("fetch_summaries"):
            summaries = fetch_summaries(self.store, stage1_ids)
        with timer.stage("rerank"):
            jev = score_candidates(self.jev, query, list(zip(stage1_ids, summaries)), self.scoring)
        ranked = top_k(stage1_ids, [jev.scores[episode_id] for episode_id in stage1_ids], self.k)
        latency_s = timer.elapsed()
        trace = RankTrace(
            ranker=self.name,
            pool_size=self._pool_size(owner_id, person_id, time_range),
            latency_s=latency_s,
            stage1_ids=stage1_ids,
            stage_latency_s=timer.stages,
            usage=jev.usage,
            extra=trace_extra(jev, self.scoring),
        )
        return RankResult(ranked=ranked, trace=trace)
