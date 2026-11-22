"""Per-query result records, retrieval metrics, bootstrap statistics, aggregation.

The runner produces one ``RankerRecord`` per (query, ranker, run) and one
``SystemRecord`` per (query, ranker, k, k'). The aggregators turn those into the
nested tables that ``report.py`` renders:

    ranker_comparison(records)[ranker]        -> overall / by_type / by_pool slices,
                                                 recall_gain, recall_diff_vs, n_missing, failures
    jev_variance(records)[ranker]             -> run-to-run statistics, snapshots per run
    jev_snapshots(records)                    -> the Jev snapshot ids the records report
    system_eval(records)[ranker][k][k_prime]  -> overall / by_type / by_pool slices, failures,
                                                 next_k, next_k_prime, accuracy_diff_vs
    recommend(ranker_table, system_table)     -> ranker, k, k_prime, failure_modes

Table keys are always strings (``"10"``, not ``10``) so a table is the same
before and after a JSON round trip. Every proportion is an ``Estimate``: the
mean over queries with a percentile-bootstrap confidence interval. Records are
ordered by query id before resampling and the bootstrap is seeded from
``config.SEED``, so the same records always give the same numbers, whatever
order the runner produced them in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, TypeGuard, TypeVar

import numpy as np

from . import config
from .interfaces import Usage
from .pipelines.jev_scoring import MISSING_SCORE
from .schema import (
    ANSWERABLE_TYPES,
    DECLINE_PATHS,
    MISROUTED_PATHS,
    POOL_BUCKETS,
    QUERY_TYPES,
    SYSTEM_PATHS,
    SYSTEM_STAGES,
    VERDICTS,
    TimeRange,
    pool_bucket,
)

# SYSTEM_PATHS, SYSTEM_STAGES, DECLINE_PATHS, MISROUTED_PATHS and VERDICTS are
# schema's and stay importable from here, where the aggregates use them.
# The failure categories in the order the flow meets them: extraction, routing,
# ranking, answering. ``_system_failures`` says what each one counts.
FAILURE_MODES: tuple[str, ...] = (
    "extraction_name_errors",
    "missed_time_ranges",
    "time_filter_misses",
    "spurious_time_filters",
    "misrouted",
    "empty_pools",
    "stage1_misses",
    "gold_not_in_top_k",
    "false_declines_gold_retrieved",
    "wrong_answers_gold_retrieved",
    "wrong_answers_from_summaries",
    "unjudged",
)
MIN_GAIN = 0.02  # a step to the next k or k' must add at least 2 points to be worth taking
REFERENCE_K = 10  # rankers are compared on Recall@10 and on answer accuracy at k=10
VARIANCE_TOP_N = 10  # depth of the top-n sets compared between repeated runs
_CI_LABEL = f"{round(100 * config.CI_LEVEL)}% CI"

T = TypeVar("T")
R = TypeVar("R", "RankerRecord", "SystemRecord")
SystemGroups = Mapping[tuple[str, int, int], Mapping[str, "SystemRecord"]]


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #
def _check_type(query_type: str) -> None:
    if query_type not in QUERY_TYPES:
        raise ValueError(f"unknown query type: {query_type!r}")


@dataclass
class RankerRecord:
    """One ranker's output for one answerable query under the gold filters."""

    query_id: str
    type: str
    pool_size: int
    gold_ids: list[str]
    ranker: str
    run: int  # 0-based; above 0 only for repeated Jev runs
    ranked_ids: list[str]  # best first, at most the deepest cut-off the report needs
    scores: list[float]  # parallel to ranked_ids
    stage1_ids: list[str] | None  # what bge-m3 handed to the second stage; None without one
    latency_s: float
    stage_latency_s: dict[str, float]
    usage: Usage
    # RankTrace.extra. The Jev rankers report n_requests, n_missing, n_cached, jev_model, scoring.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_type(self.type)
        if len(self.ranked_ids) != len(self.scores):
            raise ValueError(
                f"{self.ranker}/{self.query_id}: {len(self.ranked_ids)} ranked ids "
                f"but {len(self.scores)} scores"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "type": self.type,
            "pool_size": self.pool_size,
            "gold_ids": list(self.gold_ids),
            "ranker": self.ranker,
            "run": self.run,
            "ranked_ids": list(self.ranked_ids),
            "scores": [float(score) for score in self.scores],
            "stage1_ids": None if self.stage1_ids is None else list(self.stage1_ids),
            "latency_s": self.latency_s,
            "stage_latency_s": dict(self.stage_latency_s),
            "usage": self.usage.to_json(),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RankerRecord":
        stage1_ids = data.get("stage1_ids")
        stage_latency = data.get("stage_latency_s") or {}
        return cls(
            query_id=str(data["query_id"]),
            type=data["type"],
            pool_size=int(data["pool_size"]),
            gold_ids=[str(item) for item in data["gold_ids"]],
            ranker=data["ranker"],
            run=int(data.get("run", 0)),
            ranked_ids=[str(item) for item in data["ranked_ids"]],
            scores=[float(score) for score in data["scores"]],
            stage1_ids=None if stage1_ids is None else [str(item) for item in stage1_ids],
            latency_s=float(data["latency_s"]),
            stage_latency_s={str(stage): float(value) for stage, value in stage_latency.items()},
            usage=Usage.from_json(data.get("usage")),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class SystemRecord:
    """The end-to-end outcome of one query at one (ranker, k, k') setting."""

    query_id: str
    type: str
    pool_size: int | None  # pool under the gold filters; None for unknown_name
    ranker: str
    k: int
    k_prime: int
    path: str  # one of SYSTEM_PATHS: where the flow ended
    needed_dialogue: bool  # the summary attempt could not answer
    final_answer: str
    verdict: str | None  # one of VERDICTS; None when the answer was not judged
    gold_ids: list[str]
    retrieved_ids: list[str]  # the top-k ids the answerer saw
    gold_name: str | None
    extracted_name: str | None
    name_correct: bool
    person_correct: bool
    extracted_time_range: TimeRange | None  # unpadded
    time_filter_ok: bool | None  # None when the query has no gold time range
    unknown_name_exact: bool | None  # None unless type is unknown_name
    usage_by_stage: dict[str, Usage]  # keyed by SYSTEM_STAGES; a stage that did not run is zero
    latency_by_stage: dict[str, float]  # seconds
    rank_cached: bool = False  # the rank stage replayed cached Jev answers, so its latency is a cache read

    def __post_init__(self) -> None:
        _check_type(self.type)
        if self.path not in SYSTEM_PATHS:
            raise ValueError(f"unknown path: {self.path!r}")
        if self.verdict is not None and self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict: {self.verdict!r}")
        unknown = sorted((set(self.usage_by_stage) | set(self.latency_by_stage)) - set(SYSTEM_STAGES))
        if unknown:
            raise ValueError(f"unknown stages {unknown}; expected a subset of {SYSTEM_STAGES}")

    def to_json(self) -> dict[str, Any]:
        extracted = self.extracted_time_range
        return {
            "query_id": self.query_id,
            "type": self.type,
            "pool_size": self.pool_size,
            "ranker": self.ranker,
            "k": self.k,
            "k_prime": self.k_prime,
            "path": self.path,
            "needed_dialogue": self.needed_dialogue,
            "final_answer": self.final_answer,
            "verdict": self.verdict,
            "gold_ids": list(self.gold_ids),
            "retrieved_ids": list(self.retrieved_ids),
            "gold_name": self.gold_name,
            "extracted_name": self.extracted_name,
            "name_correct": self.name_correct,
            "person_correct": self.person_correct,
            "extracted_time_range": None if extracted is None else extracted.to_json(),
            "time_filter_ok": self.time_filter_ok,
            "unknown_name_exact": self.unknown_name_exact,
            "usage_by_stage": {s: usage.to_json() for s, usage in self.usage_by_stage.items()},
            "latency_by_stage": dict(self.latency_by_stage),
            "rank_cached": self.rank_cached,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "SystemRecord":
        pool_size = data.get("pool_size")
        usage_by_stage = data.get("usage_by_stage") or {}
        latency_by_stage = data.get("latency_by_stage") or {}
        return cls(
            query_id=str(data["query_id"]),
            type=data["type"],
            pool_size=None if pool_size is None else int(pool_size),
            ranker=data["ranker"],
            k=int(data["k"]),
            k_prime=int(data["k_prime"]),
            path=data["path"],
            needed_dialogue=bool(data["needed_dialogue"]),
            final_answer=data.get("final_answer") or "",
            verdict=data.get("verdict"),
            gold_ids=[str(item) for item in data.get("gold_ids") or []],
            retrieved_ids=[str(item) for item in data.get("retrieved_ids") or []],
            gold_name=data.get("gold_name"),
            extracted_name=data.get("extracted_name"),
            name_correct=bool(data["name_correct"]),
            person_correct=bool(data["person_correct"]),
            extracted_time_range=TimeRange.from_json(data.get("extracted_time_range")),
            time_filter_ok=data.get("time_filter_ok"),
            unknown_name_exact=data.get("unknown_name_exact"),
            usage_by_stage={str(s): Usage.from_json(usage) for s, usage in usage_by_stage.items()},
            latency_by_stage={str(s): float(value) for s, value in latency_by_stage.items()},
            rank_cached=bool(data.get("rank_cached", False)),
        )


# --------------------------------------------------------------------------- #
# Estimates and bootstrap                                                      #
# --------------------------------------------------------------------------- #
def nan_to_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def _none_to_nan(value: float | None) -> float:
    return math.nan if value is None else float(value)


@dataclass(frozen=True)
class Estimate:
    """A mean over ``n`` queries with its bootstrap confidence interval."""

    mean: float
    lo: float
    hi: float
    n: int

    @property
    def defined(self) -> bool:
        """False for the all-NaN estimate of an empty sample."""

        return self.n > 0 and not math.isnan(self.mean)

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": nan_to_none(self.mean),
            "lo": nan_to_none(self.lo),
            "hi": nan_to_none(self.hi),
            "n": self.n,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Estimate":
        return cls(
            mean=_none_to_nan(data.get("mean")),
            lo=_none_to_nan(data.get("lo")),
            hi=_none_to_nan(data.get("hi")),
            n=int(data.get("n") or 0),
        )


def as_estimate(value: Any) -> Estimate | None:
    """Accept an ``Estimate`` or its JSON form (a table read back from a results file)."""

    if isinstance(value, Estimate):
        return value
    if isinstance(value, Mapping) and {"mean", "lo", "hi", "n"} <= set(value):
        return Estimate.from_json(value)
    return None


def _finite(values: Iterable[float]) -> np.ndarray:
    data = np.fromiter(values, dtype=float)
    return data[~np.isnan(data)]


def bootstrap_ci(
    values: Iterable[float],
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    level: float = config.CI_LEVEL,
    seed: int = config.SEED,
) -> Estimate:
    """Mean of ``values`` with a percentile-bootstrap CI, resampling queries with replacement.

    Undefined (NaN) entries are dropped, so ``n`` counts the queries that
    contribute. An empty sample gives NaNs with ``n == 0``. The resample indices
    depend only on ``(seed, n, n_resamples)``, so metrics computed over the same
    queries share their resamples.
    """

    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    data = _finite(values)
    n = int(data.size)
    if n == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    rng = np.random.default_rng(seed)
    resample_means = data[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(resample_means, [tail, 1.0 - tail])
    return Estimate(float(data.mean()), float(lo), float(hi), n)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    level: float = config.CI_LEVEL,
    seed: int = config.SEED,
) -> Estimate:
    """Mean of ``a - b`` over the same queries, with a CI that resamples the pairs.

    Pairing removes the between-query variance both sides share, which is what
    makes a small difference between two rankers detectable. Pairs with an
    undefined side are dropped.
    """

    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} vs {len(b)}")
    diffs = (x - y for x, y in zip(a, b))
    return bootstrap_ci(diffs, n_resamples=n_resamples, level=level, seed=seed)


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    """p50, p95 and mean of per-query latencies (NaNs for an empty sample)."""

    data = _finite(values)
    if data.size == 0:
        return {"p50": math.nan, "p95": math.nan, "mean": math.nan, "n": 0}
    p50, p95 = np.percentile(data, [50, 95])
    return {"p50": float(p50), "p95": float(p95), "mean": float(data.mean()), "n": int(data.size)}


# --------------------------------------------------------------------------- #
# Retrieval primitives                                                         #
# --------------------------------------------------------------------------- #
def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of the gold conversations that appear in the top ``k``.

    NaN without gold: recall is undefined there and the aggregates skip it. A
    repeated id in ``ranked_ids`` uses up a slot but earns credit once.
    """

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    gold = set(gold_ids)
    if not gold:
        return math.nan
    return len(gold.intersection(ranked_ids[:k])) / len(gold)


def all_found_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """1.0 when every gold conversation is in the top ``k``, else 0.0 (NaN without gold)."""

    recall = recall_at_k(ranked_ids, gold_ids, k)
    return recall if math.isnan(recall) else float(recall == 1.0)


def stage1_recall(stage1_ids: Sequence[str] | None, gold_ids: Iterable[str]) -> float:
    """Fraction of the gold conversations that stage 1 passed on.

    NaN without gold or for a ranker without a stage 1.
    """

    gold = set(gold_ids)
    if stage1_ids is None or not gold:
        return math.nan
    return len(gold.intersection(stage1_ids)) / len(gold)


# --------------------------------------------------------------------------- #
# Shared aggregation helpers                                                   #
# --------------------------------------------------------------------------- #
def ordered_rankers(names: Iterable[str]) -> list[str]:
    """Known rankers in ``config.RANKER_NAMES`` order, then any others alphabetically."""

    rank = {name: position for position, name in enumerate(config.RANKER_NAMES)}
    return sorted(set(names), key=lambda name: (rank.get(name, len(rank)), name))


def numeric_keys(keys: Iterable[str]) -> list[str]:
    """The integer-like keys of a table level (k or k' values), in numeric order."""

    return sorted({key for key in keys if str(key).isdigit()}, key=int)


def dig(tree: Any, *keys: str | None) -> Any:
    """Follow ``keys`` through nested mappings; None as soon as one is missing."""

    for key in keys:
        if not isinstance(tree, Mapping) or key not in tree:
            return None
        tree = tree[key]
    return tree


def mean(values: Iterable[float]) -> float:
    """Mean of the defined entries, summed exactly so their order cannot matter; NaN if none."""

    defined = [float(value) for value in values if not math.isnan(value)]
    return math.fsum(defined) / len(defined) if defined else math.nan


def _group(records: Iterable[R], key: Callable[[R], Hashable]) -> dict[Any, dict[str, R]]:
    """Group records by ``key`` and index each group by query id, in query-id order.

    A repeated query within a group is an error: it would be double counted, and
    it is what a resumed run that appends twice looks like.
    """

    groups: dict[Any, dict[str, R]] = {}
    for record in records:
        group = groups.setdefault(key(record), {})
        if record.query_id in group:
            raise ValueError(f"duplicate record for query {record.query_id!r} in {key(record)!r}")
        group[record.query_id] = record
    return {name: dict(sorted(group.items())) for name, group in groups.items()}


def _sliced(
    items: Sequence[T],
    summarize: Callable[[list[T]], Any],
    record: Callable[[T], Any] = lambda item: item,
) -> dict[str, Any]:
    """Apply ``summarize`` overall, per query type, and per pool-size bucket.

    Only slices that have queries are present. Queries without a pool (unknown
    names) belong to no pool bucket. ``record`` maps an item to the record that
    carries its type and pool size (items may be pairs of records).
    """

    by_type: dict[str, list[T]] = {name: [] for name in QUERY_TYPES}
    by_pool: dict[str, list[T]] = {bucket: [] for bucket in POOL_BUCKETS}
    for item in items:
        by_type[record(item).type].append(item)
        pool_size = record(item).pool_size
        if pool_size is not None:
            by_pool[pool_bucket(pool_size)].append(item)
    return {
        "overall": summarize(list(items)),
        "by_type": {name: summarize(rows) for name, rows in by_type.items() if rows},
        "by_pool": {bucket: summarize(rows) for bucket, rows in by_pool.items() if rows},
    }


def _rate(rows: Sequence[T], hit: Callable[[T], bool]) -> Estimate:
    """Share of ``rows`` for which ``hit`` holds, with its bootstrap CI."""

    return bootstrap_ci(1.0 if hit(row) else 0.0 for row in rows)


def _negated(tree: Any) -> Any:
    """Flip the sign of every estimate in a nested table of paired differences."""

    if isinstance(tree, Estimate):
        return Estimate(0.0 - tree.mean, 0.0 - tree.hi, 0.0 - tree.lo, tree.n)
    return {key: _negated(value) for key, value in tree.items()}


# --------------------------------------------------------------------------- #
# Ranker comparison                                                            #
# --------------------------------------------------------------------------- #
def _recall(row: RankerRecord, k: int) -> float:
    return recall_at_k(row.ranked_ids, row.gold_ids, k)


def from_cache(row: RankerRecord) -> bool:
    """A response of this call came from the cache, so its wall clock is not the model's."""

    return (row.extra.get("n_cached") or 0) > 0


def missing_scores(rows: Sequence[RankerRecord]) -> int | None:
    """Candidates left without a usable score over ``rows``; None if no record counts them."""

    counts = [row.extra["n_missing"] for row in rows if row.extra.get("n_missing") is not None]
    return int(sum(counts)) if counts else None


def _stage_p50(rows: Sequence[RankerRecord]) -> dict[str, float]:
    """Median latency of each ranker stage, over the queries that report it."""

    by_stage: dict[str, list[float]] = {}
    for row in rows:
        for stage, seconds in row.stage_latency_s.items():
            by_stage.setdefault(stage, []).append(seconds)
    return {stage: float(np.median(values)) for stage, values in by_stage.items()}


def _ranker_slice(rows: Sequence[RankerRecord], ks: Sequence[int]) -> dict[str, Any]:
    multi_gold = [row for row in rows if len(set(row.gold_ids)) >= 2]
    multi_type = [row for row in rows if row.type == "multi_conversation"]
    staged = [row for row in rows if row.stage1_ids is not None]
    large = [row for row in staged if row.pool_size > config.STAGE1_TOP_K]
    timed = [row for row in rows if not from_cache(row)]

    def all_found(subset: Sequence[RankerRecord]) -> dict[str, Estimate]:
        return {
            str(k): bootstrap_ci(all_found_at_k(row.ranked_ids, row.gold_ids, k) for row in subset)
            for k in ks
        }

    def stage1(subset: Sequence[RankerRecord]) -> Estimate | None:
        if not staged:
            return None
        return bootstrap_ci(stage1_recall(row.stage1_ids, row.gold_ids) for row in subset)

    return {
        "n": len(rows),
        "recall": {str(k): bootstrap_ci(_recall(row, k) for row in rows) for k in ks},
        "all_found": all_found(multi_gold),
        "all_found_multi_conversation": all_found(multi_type),
        "stage1_recall": stage1(large),
        "stage1_recall_all_pools": stage1(staged),
        "latency": latency_summary(row.latency_s for row in timed),
        "stage_latency_p50": _stage_p50(timed),
        "cost_per_1k_usd": 1000.0 * mean(row.usage.cost_usd for row in rows),
    }


def ranker_comparison(
    records: Iterable[RankerRecord], ks: Sequence[int] = config.RECALL_KS
) -> dict[str, Any]:
    """Compare rankers on their first run (``run == 0``) under the gold filters.

    For each ranker, ``overall``, ``by_type[type]`` and ``by_pool[bucket]`` hold:
    ``n``; ``recall[k]``; ``all_found[k]`` over queries with two or more gold
    conversations and ``all_found_multi_conversation[k]`` over that query type
    alone; ``stage1_recall`` over pools larger than the stage-1 cut-off and
    ``stage1_recall_all_pools`` for reference (both None for rankers without a
    stage 1); ``latency`` (p50/p95/mean/n), ``stage_latency_p50`` and
    ``cost_per_1k_usd``. Latency leaves out the records with ``extra["n_cached"]``
    above zero, because a cached call times a cache read and not the model;
    ``latency["n"]`` is the number of records the latency figures rest on. Cost
    covers every record: a cached call still reports what the original call cost.
    Beside the slices each ranker has:

    - ``n_missing``: candidates the scorer left without a usable score, summed
      over the ranker's queries (None for rankers that do not count them).
    - ``recall_gain[k][deeper_k]``: paired gain in Recall from k to a deeper k.
    - ``recall_diff_vs[other]``: its Recall@k minus the other ranker's over the
      queries both ran, as ``overall[k]``, ``by_type[type][k]``, ``by_pool[bucket][k]``.
    - ``failures``: ``stage1_misses`` (queries where stage 1 dropped a gold
      conversation; None without a stage 1) and ``gold_not_in_top_k[k]``.
    """

    cutoffs = sorted(set(ks))
    groups = _group(records, lambda record: (record.ranker, record.run))
    names = ordered_rankers(name for name, run in groups if run == 0)
    first_runs = {name: groups[(name, 0)] for name in names}
    table: dict[str, Any] = {}
    for name, by_query in first_runs.items():
        rows = list(by_query.values())
        recalls = {k: [_recall(row, k) for row in rows] for k in cutoffs}
        staged = [row for row in rows if row.stage1_ids is not None]
        entry = _sliced(rows, lambda subset: _ranker_slice(subset, cutoffs))
        entry["recall_gain"] = {
            str(low): {
                str(high): paired_bootstrap_diff(recalls[high], recalls[low])
                for high in cutoffs
                if high > low
            }
            for low in cutoffs[:-1]
        }
        entry["recall_diff_vs"] = {}
        entry["n_missing"] = missing_scores(rows)
        entry["failures"] = {
            "stage1_misses": (
                sum(stage1_recall(row.stage1_ids, row.gold_ids) < 1.0 for row in staged)
                if staged
                else None
            ),
            "gold_not_in_top_k": {
                str(k): sum(value < 1.0 for value in recalls[k]) for k in cutoffs
            },
        }
        table[name] = entry

    def recall_diffs(pairs: Sequence[tuple[RankerRecord, RankerRecord]]) -> dict[str, Estimate]:
        return {
            str(k): paired_bootstrap_diff(
                [_recall(left, k) for left, _ in pairs], [_recall(right, k) for _, right in pairs]
            )
            for k in cutoffs
        }

    for earlier, later in combinations(names, 2):
        base = first_runs[earlier]
        pairs = [(row, base[qid]) for qid, row in first_runs[later].items() if qid in base]
        diff = _sliced(pairs, recall_diffs, record=lambda pair: pair[0])
        table[later]["recall_diff_vs"][earlier] = diff
        table[earlier]["recall_diff_vs"][later] = _negated(diff)
    return table


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Overlap of two id sets; two empty sets are identical, hence 1.0."""

    union = set(a) | set(b)
    return len(set(a) & set(b)) / len(union) if union else 1.0


def jev_snapshots(records: Iterable[RankerRecord]) -> list[str]:
    """The dated Jev snapshot ids the API reported, from ``extra["jev_model"]``.

    Only the Jev rankers set the key, so any records can be passed; a trace
    comma-joins the snapshots when one query's requests hit several.
    """

    found: set[str] = set()
    for record in records:
        for name in str(record.extra.get("jev_model", "")).split(","):
            if name:
                found.add(name)
    return sorted(found)


def jev_variance(
    records: Iterable[RankerRecord], ks: Sequence[int] = config.RECALL_KS
) -> dict[str, Any]:
    """Run-to-run variation for every ranker that was run more than once.

    Over the queries present in every run: ``recall[k]`` with the value
    ``per_run`` and its ``mean`` and sample standard deviation ``std`` across
    runs; ``mean_abs_score_diff``, the mean absolute score difference per
    (query, conversation) between two runs, over the conversations both runs
    returned and scored (a pair where either run left the conversation at
    ``MISSING_SCORE`` is counted in ``n_unscored_pairs`` instead, so a refusal
    does not pass for drift); and ``top10_jaccard``, the mean Jaccard overlap
    of the two runs' top-10 sets. Those average over every pair of runs.
    ``snapshots_per_run`` lists, parallel to ``runs``, the Jev snapshot ids each
    run's records report, so a drift can be told from a model change.
    """

    groups = _group(records, lambda record: (record.ranker, record.run))
    table: dict[str, Any] = {}
    for name in ordered_rankers(ranker for ranker, _ in groups):
        runs = {run: groups[(ranker, run)] for ranker, run in sorted(groups) if ranker == name}
        if len(runs) < 2:
            continue
        shared = sorted(set.intersection(*(set(by_query) for by_query in runs.values())))
        recall: dict[str, Any] = {}
        for k in sorted(set(ks)):
            per_run = [
                mean(_recall(by_query[qid], k) for qid in shared) for by_query in runs.values()
            ]
            recall[str(k)] = {
                "per_run": per_run,
                "mean": mean(per_run),
                "std": float(np.std(per_run, ddof=1)),
            }
        score_diffs: list[float] = []
        n_unscored_pairs = 0
        overlaps: list[float] = []
        for first, second in combinations(runs.values(), 2):
            for qid in shared:
                ours, theirs = first[qid], second[qid]
                their_scores = dict(zip(theirs.ranked_ids, theirs.scores))
                for episode_id, score in zip(ours.ranked_ids, ours.scores):
                    if episode_id not in their_scores:
                        continue
                    if MISSING_SCORE in (score, their_scores[episode_id]):
                        n_unscored_pairs += 1
                    else:
                        score_diffs.append(abs(score - their_scores[episode_id]))
                overlaps.append(
                    _jaccard(ours.ranked_ids[:VARIANCE_TOP_N], theirs.ranked_ids[:VARIANCE_TOP_N])
                )
        table[name] = {
            "runs": list(runs),
            "n_queries": len(shared),
            "snapshots_per_run": [jev_snapshots(by_query.values()) for by_query in runs.values()],
            "recall": recall,
            "mean_abs_score_diff": mean(score_diffs),
            "n_unscored_pairs": n_unscored_pairs,
            "top10_jaccard": mean(overlaps),
        }
    return table


# --------------------------------------------------------------------------- #
# System eval                                                                  #
# --------------------------------------------------------------------------- #
def _answerable(row: SystemRecord) -> bool:
    return row.type in ANSWERABLE_TYPES


def _correct(row: SystemRecord) -> bool:
    return row.verdict == "correct"


def _declined(row: SystemRecord) -> bool:
    return row.path in DECLINE_PATHS


def _gold_retrieved(row: SystemRecord) -> bool:
    """Every gold conversation was among the top-k ids the answerer saw."""

    return bool(row.gold_ids) and set(row.gold_ids) <= set(row.retrieved_ids)


def _input_tokens(row: SystemRecord) -> int:
    return sum(usage.input_tokens for usage in row.usage_by_stage.values())


def _output_tokens(row: SystemRecord) -> int:
    return sum(usage.output_tokens for usage in row.usage_by_stage.values())


def _spend(row: SystemRecord) -> float:
    return math.fsum(usage.cost_usd for usage in row.usage_by_stage.values())


def _cost(rows: Sequence[SystemRecord]) -> dict[str, Any]:
    """Per-query averages of tokens and dollars, and latency percentiles, summed over the stages.

    The latency summary leaves out the rows whose rank stage replayed cached
    Jev answers (``rank_cached``): a cache read is not the ranker's latency,
    as the ranker comparison leaves out its cached records. ``n_timed`` is
    the number of rows the latency summary covers.
    """

    timed = [row for row in rows if not row.rank_cached]
    return {
        "n": len(rows),
        "input_tokens": mean(_input_tokens(row) for row in rows),
        "output_tokens": mean(_output_tokens(row) for row in rows),
        "cost_usd": mean(_spend(row) for row in rows),
        "latency": latency_summary(math.fsum(row.latency_by_stage.values()) for row in timed),
        "n_timed": len(timed),
    }


def _system_slice(rows: Sequence[SystemRecord]) -> dict[str, Any]:
    answerable = [row for row in rows if _answerable(row)]
    gold_retrieved = [row for row in answerable if _gold_retrieved(row)]
    temporal = [row for row in rows if row.type == "temporal" and row.time_filter_ok is not None]
    no_gold_range = [row for row in rows if row.time_filter_ok is None]
    unknown_name = [row for row in rows if row.type == "unknown_name"]
    return {
        "n": len(rows),
        "answer_accuracy": _rate(answerable, _correct),
        "partial_rate": _rate(answerable, lambda row: row.verdict == "partial"),
        "fallback_rate": _rate(answerable, lambda row: row.needed_dialogue),
        "summary_path_accuracy": _rate([r for r in answerable if r.path == "summary"], _correct),
        "fallback_path_accuracy": _rate([r for r in answerable if r.path == "fallback"], _correct),
        "needed_dialogue_accuracy": _rate([r for r in answerable if r.needed_dialogue], _correct),
        "correct_decline_rate": _rate([r for r in rows if r.type == "unanswerable"], _declined),
        "false_decline_rate": _rate(answerable, _declined),
        "false_decline_rate_gold_retrieved": _rate(gold_retrieved, _declined),
        "unknown_name_exact_rate": _rate(
            unknown_name, lambda row: row.unknown_name_exact is True
        ),
        "name_accuracy": _rate(rows, lambda row: row.name_correct),
        "person_accuracy": _rate(rows, lambda row: row.person_correct),
        "time_range_extracted_rate": _rate(temporal, lambda row: row.extracted_time_range is not None),
        "time_filter_recall": _rate(temporal, lambda row: row.time_filter_ok is True),
        "spurious_time_range_rate": _rate(
            no_gold_range, lambda row: row.extracted_time_range is not None
        ),
        "misrouted_rate": _rate(answerable, lambda row: row.path in MISROUTED_PATHS),
        "cost": {
            "overall": _cost(rows),
            "summary": _cost([row for row in rows if row.path == "summary"]),
            "fallback": _cost([row for row in rows if row.needed_dialogue]),
        },
    }


def _system_failures(rows: Sequence[SystemRecord]) -> dict[str, int]:
    """Query counts per failure category (``FAILURE_MODES``); a query can fall into several.

    Each count charges the stage that failed: ``missed_time_ranges`` is a
    temporal query the extraction returned no time range for, and
    ``time_filter_misses`` one whose padded range dropped a gold conversation.
    ``gold_not_in_top_k`` covers only the answerable queries the answerer saw a
    ranking for; one that stopped at an unknown or ambiguous name is
    ``misrouted`` and one whose filters left nothing to rank is an
    ``empty_pools``. A decline with the gold retrieved is a
    ``false_declines_gold_retrieved`` and not a ``wrong_answers_gold_retrieved``,
    although the judge grades the refusal incorrect. ``unjudged`` is an
    answerable query whose final answer the judge gave no usable verdict for
    (``verdict`` None): it counts as not correct in the accuracy rates, so the
    count says how much of the error is the judge's.
    """

    answerable = [row for row in rows if _answerable(row)]
    ranked = [
        row for row in answerable if row.path not in MISROUTED_PATHS and row.path != "empty_pool"
    ]
    retrieved = [row for row in answerable if _gold_retrieved(row)]
    temporal = [row for row in rows if row.type == "temporal" and row.time_filter_ok is not None]
    return {
        "extraction_name_errors": sum(not row.name_correct for row in rows),
        "missed_time_ranges": sum(row.extracted_time_range is None for row in temporal),
        "time_filter_misses": sum(
            row.time_filter_ok is False and row.extracted_time_range is not None for row in rows
        ),
        "spurious_time_filters": sum(
            row.time_filter_ok is None and row.extracted_time_range is not None for row in rows
        ),
        "misrouted": sum(row.path in MISROUTED_PATHS for row in answerable),
        "empty_pools": sum(row.path == "empty_pool" for row in answerable),
        "gold_not_in_top_k": sum(not _gold_retrieved(row) for row in ranked),
        "false_declines_gold_retrieved": sum(_declined(row) for row in retrieved),
        "wrong_answers_gold_retrieved": sum(
            row.verdict == "incorrect" and not _declined(row) for row in retrieved
        ),
        "wrong_answers_from_summaries": sum(
            row.path == "summary" and row.verdict == "incorrect" for row in answerable
        ),
        "unjudged": sum(row.verdict is None for row in answerable),
    }


def _answerable_pairs(
    left: Mapping[str, SystemRecord], right: Mapping[str, SystemRecord]
) -> list[tuple[SystemRecord, SystemRecord]]:
    """Answerable queries present in both settings, in query-id order."""

    return [(row, right[qid]) for qid, row in left.items() if qid in right and _answerable(row)]


def _accuracy_diff(pairs: Sequence[tuple[SystemRecord, SystemRecord]]) -> Estimate:
    """Paired difference in answer accuracy, first setting minus second."""

    return paired_bootstrap_diff(
        [float(_correct(left)) for left, _ in pairs], [float(_correct(right)) for _, right in pairs]
    )


def _next_k(groups: SystemGroups, ranker: str, k: int, k_prime: int) -> dict[str, Any] | None:
    """Paired gain in answer accuracy from the next larger k at the same k'."""

    larger = [other for name, other, kp in groups if name == ranker and kp == k_prime and other > k]
    if not larger:
        return None
    pairs = _answerable_pairs(groups[(ranker, min(larger), k_prime)], groups[(ranker, k, k_prime)])
    return {"k": str(min(larger)), "answer_accuracy_gain": _accuracy_diff(pairs)}


def _next_k_prime(groups: SystemGroups, ranker: str, k: int, k_prime: int) -> dict[str, Any] | None:
    """Paired gain from the next larger k' at the same k, over the queries that needed dialogue.

    Only those queries reach the dialogue step, so they are the ones k' can
    change; the extra input tokens are what the deeper fallback costs them.
    """

    larger = [other for name, kk, other in groups if name == ranker and kk == k and other > k_prime]
    if not larger:
        return None
    pairs = [
        (deeper, current)
        for deeper, current in _answerable_pairs(
            groups[(ranker, k, min(larger))], groups[(ranker, k, k_prime)]
        )
        if deeper.needed_dialogue and current.needed_dialogue
    ]
    return {
        "k_prime": str(min(larger)),
        "needed_dialogue_accuracy_gain": _accuracy_diff(pairs),
        "extra_input_tokens": mean(
            _input_tokens(deeper) - _input_tokens(current) for deeper, current in pairs
        ),
    }


def system_eval(records: Iterable[SystemRecord]) -> dict[str, Any]:
    """Aggregate the end-to-end eval into ``table[ranker][k][k_prime]``.

    Each setting has ``overall``, ``by_type[type]`` and ``by_pool[bucket]``
    slices with these estimates (the population is in brackets):

    - ``answer_accuracy``, ``partial_rate`` [answerable]: verdict correct /
      partial. Unjudged answers (declined, misrouted) count as not correct.
    - ``fallback_rate`` [answerable]: the summary attempt could not answer.
    - ``summary_path_accuracy`` / ``fallback_path_accuracy`` [answerable whose
      final answer came from summaries / from dialogue].
    - ``needed_dialogue_accuracy`` [answerable that needed dialogue, including
      those that then declined]: the number k' can move.
    - ``correct_decline_rate`` [unanswerable], ``false_decline_rate``
      [answerable], ``false_decline_rate_gold_retrieved`` [answerable with every
      gold conversation retrieved]: the flow ended in "declined" or "empty_pool".
    - ``unknown_name_exact_rate`` [unknown_name]: the exact error message.
    - ``name_accuracy``, ``person_accuracy`` [all queries].
    - ``time_range_extracted_rate`` [temporal]: the extraction returned a time range.
    - ``time_filter_recall`` [temporal]: a range was extracted and, padded, kept
      every gold conversation; a missed range is a miss here too.
    - ``spurious_time_range_rate`` [no gold time range]: a range was extracted anyway.
    - ``misrouted_rate`` [answerable]: ended as "unknown_name" or "ambiguous".
    - ``cost``: per-query ``input_tokens``, ``output_tokens``, ``cost_usd`` and
      ``latency`` (all summed over the stages), for ``overall``, the ``summary`` path
      (answered from summaries) and the ``fallback`` path (every query whose
      summary attempt failed, since it pays for the dialogue call even when it
      then declines).

    Beside the slices: ``failures`` (query counts per category), ``next_k`` and
    ``next_k_prime`` (paired gain in answer accuracy from moving to the next
    larger k or k'; None at the largest value), and ``accuracy_diff_vs[other]``
    (paired answer-accuracy difference against another ranker at the same k and k').
    """

    groups = _group(records, lambda record: (record.ranker, record.k, record.k_prime))
    table: dict[str, Any] = {name: {} for name in ordered_rankers(name for name, _, _ in groups)}
    for ranker, k, k_prime in sorted(groups):
        rows = list(groups[(ranker, k, k_prime)].values())
        cell = _sliced(rows, _system_slice)
        cell["failures"] = _system_failures(rows)
        cell["next_k"] = _next_k(groups, ranker, k, k_prime)
        cell["next_k_prime"] = _next_k_prime(groups, ranker, k, k_prime)
        cell["accuracy_diff_vs"] = {}
        table[ranker].setdefault(str(k), {})[str(k_prime)] = cell
    for earlier, later in combinations(table, 2):
        for k, by_k_prime in table[later].items():
            for k_prime, cell in by_k_prime.items():
                other = dig(table, earlier, k, k_prime)
                if other is None:
                    continue
                setting = (int(k), int(k_prime))
                diff = _accuracy_diff(
                    _answerable_pairs(groups[(later, *setting)], groups[(earlier, *setting)])
                )
                cell["accuracy_diff_vs"][earlier] = diff
                other["accuracy_diff_vs"][later] = _negated(diff)
    return table


# --------------------------------------------------------------------------- #
# Recommendation                                                               #
# --------------------------------------------------------------------------- #
def _defined(estimate: Estimate | None) -> TypeGuard[Estimate]:
    return estimate is not None and estimate.defined


def _real_gain(gain: Estimate | None, min_gain: float) -> bool:
    """A step pays when it adds at least ``min_gain`` and its paired CI excludes zero."""

    return _defined(gain) and gain.mean >= min_gain and gain.lo > 0.0


def _or_inf(value: float | None) -> float:
    return math.inf if value is None or math.isnan(value) else float(value)


def _points(gain: Estimate | None) -> str:
    """A paired difference in percentage points with its CI, for a rationale sentence."""

    if not _defined(gain):
        return "an unmeasured amount"
    middle, lo, hi = (round(100.0 * value, 1) + 0.0 for value in (gain.mean, gain.lo, gain.hi))
    return f"{middle:+.1f} points ({_CI_LABEL} {lo:+.1f} to {hi:+.1f})"


def _reference_key(keys: Iterable[str]) -> str | None:
    """``REFERENCE_K`` when it was measured, otherwise the deepest cut-off available."""

    ordered = numeric_keys(keys)
    if not ordered:
        return None
    return str(REFERENCE_K) if str(REFERENCE_K) in ordered else ordered[-1]


def _largest_key(keys: Iterable[str]) -> str | None:
    ordered = numeric_keys(keys)
    return ordered[-1] if ordered else None


# Why a ranker is out of contention on a metric; ``{on}`` names the metric(s) in the rationale.
_SIGNIFICANTLY_BEHIND = "significantly below the best on {on}"
_BEHIND_ON_POINT_ESTIMATE = "below the best on {on} by point estimate (no paired difference to test it)"
_NOT_MEASURED = "not measured on {on}"


def _standing(
    names: Sequence[str],
    estimates: Mapping[str, Estimate | None],
    paired_diff: Callable[[str, str], Any],
) -> dict[str, str | None]:
    """Why each ranker is out of contention on one metric, or None while it is tied with the best.

    A ranker stays in while the paired CI of (best - ranker) includes zero;
    without a paired estimate the point estimates decide. Rankers with no value
    drop out, unless nobody has one, in which case the metric excludes no one.
    The reasons are the templates above, in ``names`` order.
    """

    scored = [name for name in names if _defined(estimates.get(name))]
    if not scored:
        return dict.fromkeys(names)
    means = {name: estimates[name].mean for name in scored}
    best = max(scored, key=means.__getitem__)
    standing: dict[str, str | None] = {}
    for name in names:
        if name not in scored:
            standing[name] = _NOT_MEASURED
        elif name == best:
            standing[name] = None
        elif _defined(diff := as_estimate(paired_diff(best, name))):
            standing[name] = _SIGNIFICANTLY_BEHIND if diff.lo > 0.0 else None
        else:
            standing[name] = _BEHIND_ON_POINT_ESTIMATE if means[best] > means[name] else None
    return standing


def _choose_ranker(
    ranker_table: Mapping[str, Any], system_table: Mapping[str, Any]
) -> dict[str, Any]:
    names = ordered_rankers([*ranker_table, *system_table])
    if not names:
        return {
            "choice": None,
            "rationale": "No ranker results to choose from.",
            "tied": [],
            "tied_on_recall": [],
            "tied_on_accuracy": [],
            "reference": {},
        }
    recall_k = _reference_key(
        k for name in ranker_table for k in dig(ranker_table, name, "overall", "recall") or {}
    )
    k = _reference_key(k for name in system_table for k in system_table[name])
    k_prime = _largest_key(
        kp for name in system_table for kp in dig(system_table, name, k) or {}
    )
    recall = {
        name: as_estimate(dig(ranker_table, name, "overall", "recall", recall_k)) for name in names
    }
    accuracy = {
        name: as_estimate(dig(system_table, name, k, k_prime, "overall", "answer_accuracy"))
        for name in names
    }
    recall_label = f"Recall@{recall_k}"
    accuracy_label = f"answer accuracy at k={k}, k'={k_prime}"
    recall_standing = _standing(
        names,
        recall,
        lambda best, other: dig(ranker_table, best, "recall_diff_vs", other, "overall", recall_k),
    )
    accuracy_standing = _standing(
        names,
        accuracy,
        lambda best, other: dig(system_table, best, k, k_prime, "accuracy_diff_vs", other),
    )
    recall_tied = [name for name in names if recall_standing[name] is None]
    accuracy_tied = [name for name in names if accuracy_standing[name] is None]
    # End-to-end accuracy has the last word if no ranker is tied with the best on both metrics.
    on_both = [name for name in names if name in recall_tied and name in accuracy_tied]
    tied = on_both or accuracy_tied

    def expense(name: str) -> tuple[float, float]:
        return (
            _or_inf(dig(ranker_table, name, "overall", "latency", "p50")),
            _or_inf(dig(ranker_table, name, "overall", "cost_per_1k_usd")),
        )

    def numbers(name: str) -> str:
        parts = []
        if _defined(value := recall[name]):
            parts.append(f"{recall_label} {100.0 * value.mean:.1f}%")
        if _defined(value := accuracy[name]):
            parts.append(f"answer accuracy {100.0 * value.mean:.1f}%")
        latency, cost = expense(name)
        if math.isfinite(latency):
            parts.append(f"p50 latency {latency:.3f} s")
        if math.isfinite(cost):
            parts.append(f"${cost:.2f} per 1k queries")
        return ", ".join(parts) or "no metrics"

    def shortfall(name: str) -> str:
        """Why ``name`` is out of contention: only the metrics that actually excluded it."""

        labels_by_reason: dict[str, list[str]] = {}
        for label, standing in ((recall_label, recall_standing), (accuracy_label, accuracy_standing)):
            if standing[name] is not None:
                labels_by_reason.setdefault(standing[name], []).append(label)
        reasons = " and ".join(
            reason.format(on=" and on ".join(labels)) for reason, labels in labels_by_reason.items()
        )
        return f"{name} ({numbers(name)}), {reasons}"

    choice = min(tied, key=expense)
    basis = []
    if on_both and any(_defined(value) for value in recall.values()):
        basis.append(recall_label)
    if any(_defined(value) for value in accuracy.values()):
        basis.append(accuracy_label)
    on = " and ".join(basis) or "the available metrics"
    rivals = "; ".join(f"{name} ({numbers(name)})" for name in tied if name != choice)
    excluded = [name for name in names if name not in tied]
    if len(names) == 1:
        rationale = f"The only ranker with results ({numbers(choice)})."
    elif rivals:
        rationale = (
            f"Statistically tied on {on} with {rivals} (paired {_CI_LABEL} includes zero) and "
            f"chosen for the lowest p50 latency, then cost, among them ({numbers(choice)})."
        )
    else:
        rationale = f"The only ranker left in contention, with {numbers(choice)}."
    if excluded:
        significance = (
            f" (significantly: the paired {_CI_LABEL} of the difference excludes zero)"
            if any(
                _SIGNIFICANTLY_BEHIND in (recall_standing[name], accuracy_standing[name])
                for name in excluded
            )
            else ""
        )
        rationale += (
            f" Out of contention: {'; '.join(shortfall(name) for name in excluded)}{significance}."
        )
    return {
        "choice": choice,
        "rationale": rationale,
        "tied": tied,
        "tied_on_recall": recall_tied,
        "tied_on_accuracy": accuracy_tied,
        "reference": {"recall_k": recall_k, "k": k, "k_prime": k_prime},
    }


def _choose_k(
    ranker_table: Mapping[str, Any],
    system_table: Mapping[str, Any],
    ranker: str | None,
    min_gain: float,
) -> dict[str, Any]:
    ks = numeric_keys(dig(system_table, ranker) or {}) or numeric_keys(
        dig(ranker_table, ranker, "overall", "recall") or {}
    )
    if not ks:
        return {"choice": None, "rationale": "No results to choose k from.", "steps": []}
    steps: list[dict[str, Any]] = []
    choice, flattened = int(ks[-1]), False
    for low, high in zip(ks, ks[1:]):
        # Judge k at the deepest fallback, where a shallow k' cannot mask what a larger k adds.
        k_prime = _largest_key(dig(system_table, ranker, low) or {})
        following = dig(system_table, ranker, low, k_prime, "next_k") or {}
        if following.get("k") != high:
            following = {}
        step = {
            "from": int(low),
            "to": int(high),
            "k_prime": None if k_prime is None else int(k_prime),
            "recall_gain": as_estimate(dig(ranker_table, ranker, "recall_gain", low, high)),
            "answer_accuracy_gain": as_estimate(following.get("answer_accuracy_gain")),
        }
        steps.append(step)
        pays = _real_gain(step["recall_gain"], min_gain) or _real_gain(
            step["answer_accuracy_gain"], min_gain
        )
        if not pays:
            choice, flattened = int(low), True
            break
    if not steps:
        rationale = f"k={choice} is the only value evaluated."
        return {"choice": choice, "rationale": rationale, "steps": []}
    last = steps[-1]
    gains = (
        f"Recall@k gains {_points(last['recall_gain'])} and answer accuracy gains "
        f"{_points(last['answer_accuracy_gain'])}"
    )
    if flattened:
        rationale = (
            f"From k={last['from']} to k={last['to']}, {gains}; neither is a significant gain "
            f"of at least {100.0 * min_gain:g} points."
        )
    else:
        rationale = (
            f"k={choice} is the largest value evaluated and the step from k={last['from']} "
            f"still pays: {gains}."
        )
    return {"choice": choice, "rationale": rationale, "steps": steps}


def _choose_k_prime(
    system_table: Mapping[str, Any], ranker: str | None, k: int | None, min_gain: float
) -> dict[str, Any]:
    cells = dig(system_table, ranker, str(k)) or {}
    k_primes = numeric_keys(cells)
    if not k_primes:
        return {
            "choice": None,
            "rationale": "No system eval results to choose k' from.",
            "steps": [],
        }
    steps: list[dict[str, Any]] = []
    choice, flattened = int(k_primes[-1]), False
    for low, high in zip(k_primes, k_primes[1:]):
        following = dig(cells, low, "next_k_prime") or {}
        if following.get("k_prime") != high:
            following = {}
        step = {
            "from": int(low),
            "to": int(high),
            "needed_dialogue_accuracy_gain": as_estimate(
                following.get("needed_dialogue_accuracy_gain")
            ),
            "extra_input_tokens": following.get("extra_input_tokens"),
        }
        steps.append(step)
        if not _real_gain(step["needed_dialogue_accuracy_gain"], min_gain):
            choice, flattened = int(low), True
            break
    if not steps:
        rationale = f"k'={choice} is the only value evaluated."
        return {"choice": choice, "rationale": rationale, "steps": []}
    last = steps[-1]
    tokens = last["extra_input_tokens"]
    price = "an unmeasured number of" if tokens is None or math.isnan(tokens) else f"{tokens:+,.0f}"
    gain = (
        f"answer accuracy among the queries that needed dialogue gains "
        f"{_points(last['needed_dialogue_accuracy_gain'])} for {price} input tokens per "
        f"fallback query"
    )
    if flattened:
        rationale = (
            f"From k'={last['from']} to k'={last['to']}, {gain}; not a significant gain of at "
            f"least {100.0 * min_gain:g} points."
        )
    else:
        rationale = (
            f"k'={choice} is the largest value evaluated and the step from k'={last['from']} "
            f"still pays: {gain}."
        )
    return {"choice": choice, "rationale": rationale, "steps": steps}


def _failure_modes(
    ranker_table: Mapping[str, Any],
    system_table: Mapping[str, Any],
    ranker: str | None,
    k: int | None,
    k_prime: int | None,
) -> dict[str, int | None]:
    """Query counts per failure category at the chosen setting (None where nothing measured it)."""

    measured = dig(system_table, ranker, str(k), str(k_prime), "failures") or {}
    counts: dict[str, int | None] = {mode: measured.get(mode) for mode in FAILURE_MODES}
    counts["stage1_misses"] = dig(ranker_table, ranker, "failures", "stage1_misses")
    if counts["gold_not_in_top_k"] is None:
        counts["gold_not_in_top_k"] = dig(
            ranker_table, ranker, "failures", "gold_not_in_top_k", str(k)
        )
    return counts


def recommend(
    ranker_table: Mapping[str, Any], system_table: Mapping[str, Any], min_gain: float = MIN_GAIN
) -> dict[str, Any]:
    """Draft the ranker, k and k' to ship from the two tables, with the evidence.

    - ``ranker``: rankers stay in contention while they are not significantly
      below the best on Recall@10 and on answer accuracy (k=10, deepest k'),
      "significantly" meaning the paired CI of the difference excludes zero;
      when no ranker is tied with the best on both, answer accuracy decides.
      Among those the lowest p50 latency wins, then the lowest cost. Beside
      ``tied`` (the contention set) the draft lists ``tied_on_recall`` and
      ``tied_on_accuracy``, and its rationale names, per excluded ranker, only
      the metric(s) that excluded it.
    - ``k``: the smallest k from which the step to the next k is not a real gain
      in Recall@k nor in answer accuracy. A real gain is at least ``min_gain``
      with a paired CI that excludes zero.
    - ``k_prime``: the same rule on answer accuracy among the queries that
      needed dialogue, reported with the extra input tokens each step costs.
    - ``failure_modes``: query counts per ``FAILURE_MODES`` category at the
      chosen setting; a query can fall into several. Stage-1 misses come from
      the ranker comparison, the rest from the system eval
      (``_system_failures`` says what each category counts).

    Each choice is ``{"choice", "rationale", ...}``; the rationale is one
    sentence with the numbers behind it. With one table empty the draft uses the
    other and leaves what it cannot decide as None.
    """

    ranker = _choose_ranker(ranker_table, system_table)
    k = _choose_k(ranker_table, system_table, ranker["choice"], min_gain)
    k_prime = _choose_k_prime(system_table, ranker["choice"], k["choice"], min_gain)
    failure_modes = _failure_modes(
        ranker_table, system_table, ranker["choice"], k["choice"], k_prime["choice"]
    )
    return {"ranker": ranker, "k": k, "k_prime": k_prime, "failure_modes": failure_modes}
