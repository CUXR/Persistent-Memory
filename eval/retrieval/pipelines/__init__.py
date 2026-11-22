"""The four rankers the benchmark compares, and ``build_ranker`` to make one by name."""

from __future__ import annotations

from typing import TypeVar

from .. import config
from ..interfaces import CorpusStore, DecisionClient, Embedder, PairScorer, Ranker
from .base import DEFAULT_K
from .bge_jev import BgeJevRanker
from .bge_m3 import BgeM3Ranker
from .bge_reranker import BgeRerankerRanker
from .jev_full import JevFullPoolRanker
from .jev_scoring import DEFAULT_SCORING, JevScores, JevScoringConfig, score_candidates

__all__ = [
    "DEFAULT_K",
    "BgeJevRanker",
    "BgeM3Ranker",
    "BgeRerankerRanker",
    "JevFullPoolRanker",
    "JevScores",
    "JevScoringConfig",
    "build_ranker",
    "score_candidates",
]

_T = TypeVar("_T")


def build_ranker(
    name: str,
    *,
    store: CorpusStore,
    embedder: Embedder | None = None,
    scorer: PairScorer | None = None,
    jev: DecisionClient | None = None,
    jev_scoring: JevScoringConfig | None = None,
    k: int = DEFAULT_K,
) -> Ranker:
    """Build the ranker called ``name`` (one of ``config.RANKER_NAMES``).

    A caller can pass every dependency it has: each ranker takes only the ones
    it needs, and one that is needed but missing is reported by name here
    instead of failing in the middle of a run. ``jev_scoring`` defaults to
    ``JevScoringConfig()`` and matters only to the two Jev rankers.
    """

    if name not in config.RANKER_NAMES:
        expected = ", ".join(config.RANKER_NAMES)
        raise ValueError(f"unknown ranker {name!r}; expected one of: {expected}")
    scoring = DEFAULT_SCORING if jev_scoring is None else jev_scoring
    if name == BgeM3Ranker.name:
        return BgeM3Ranker(store, _needed(name, "embedder", embedder), k=k)
    if name == BgeRerankerRanker.name:
        return BgeRerankerRanker(
            store,
            _needed(name, "embedder", embedder),
            _needed(name, "scorer", scorer),
            k=k,
        )
    if name == BgeJevRanker.name:
        return BgeJevRanker(
            store,
            _needed(name, "embedder", embedder),
            _needed(name, "jev", jev),
            scoring,
            k=k,
        )
    return JevFullPoolRanker(store, _needed(name, "jev", jev), scoring, k=k)


def _needed(ranker: str, argument: str, value: _T | None) -> _T:
    """``value``, or a clear error naming the dependency ``ranker`` was not given."""

    if value is None:
        raise ValueError(f"ranker {ranker!r} needs {argument}=, which was not given")
    return value
