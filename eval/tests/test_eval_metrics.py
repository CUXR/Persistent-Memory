"""Retrieval metrics, bootstrap statistics, the aggregators and the recommendation rules.

The aggregate tests run on small hand-built record sets; the comments beside
each fixture give the per-query values the expected numbers are worked out from.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from eval.retrieval import config, records
from eval.retrieval.interfaces import Usage
from eval.retrieval.metrics import (
    FAILURE_MODES,
    Estimate,
    RankerRecord,
    SystemRecord,
    all_found_at_k,
    as_estimate,
    bootstrap_ci,
    dig,
    jev_snapshots,
    jev_variance,
    latency_summary,
    numeric_keys,
    ordered_rankers,
    paired_bootstrap_diff,
    ranker_comparison,
    recall_at_k,
    recommend,
    stage1_recall,
    system_eval,
)
from eval.retrieval.pipelines.jev_scoring import MISSING_SCORE
from eval.retrieval.schema import Query, TimeRange

RANGE = TimeRange(
    datetime(2026, 10, 26, tzinfo=timezone.utc), datetime(2026, 11, 1, 23, 59, 59, tzinfo=timezone.utc)
)


def plain(value: Any) -> Any:
    """A table as JSON-like data (estimates as dicts, NaN as None), so two tables compare with ==."""

    if isinstance(value, Estimate):
        return value.to_json()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def check(estimate: Estimate, mean: float, n: int) -> None:
    """The estimate has this mean over ``n`` queries and an interval around it."""

    assert estimate.n == n
    assert estimate.mean == pytest.approx(mean)
    assert estimate.lo - 1e-12 <= estimate.mean <= estimate.hi + 1e-12


# --------------------------------------------------------------------------- #
# Retrieval primitives                                                         #
# --------------------------------------------------------------------------- #
def test_recall_at_k_counts_the_gold_in_the_top_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["a", "c"], 1) == 0.5
    assert recall_at_k(ranked, ["a", "c"], 2) == 0.5
    assert recall_at_k(ranked, ["a", "c"], 3) == 1.0
    assert recall_at_k(ranked, ["z"], 4) == 0.0
    assert recall_at_k(ranked, ["a", "b", "z"], 4) == pytest.approx(2 / 3)


def test_recall_at_k_when_k_exceeds_the_list():
    assert recall_at_k(["a", "b"], ["b"], 30) == 1.0
    assert recall_at_k(["a", "b"], ["b", "c"], 30) == 0.5
    assert recall_at_k([], ["a"], 5) == 0.0


def test_recall_at_k_is_undefined_without_gold():
    assert math.isnan(recall_at_k(["a", "b"], [], 5))
    assert math.isnan(recall_at_k([], [], 5))


def test_recall_at_k_with_duplicate_ids():
    # A repeated ranked id uses up a slot but is credited once.
    assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 0.5
    assert recall_at_k(["a", "a", "b"], ["a", "b"], 3) == 1.0
    # A repeated gold id is one conversation.
    assert recall_at_k(["a", "x"], ["a", "a"], 1) == 1.0
    assert recall_at_k(["x", "a"], ["a", "a", "b"], 2) == 0.5


def test_recall_at_k_rejects_a_non_positive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(["a"], ["a"], 0)


def test_all_found_at_k_needs_every_gold_conversation():
    ranked = ["a", "b", "c"]
    assert all_found_at_k(ranked, ["a", "c"], 2) == 0.0
    assert all_found_at_k(ranked, ["a", "c"], 3) == 1.0
    assert all_found_at_k(ranked, ["a", "c"], 30) == 1.0
    assert all_found_at_k(ranked, ["a", "z"], 30) == 0.0
    assert all_found_at_k(["a", "a", "b"], ["a", "b"], 2) == 0.0
    assert all_found_at_k(["a"], ["a", "a"], 1) == 1.0
    assert math.isnan(all_found_at_k(ranked, [], 3))


def test_stage1_recall():
    assert stage1_recall(["a", "b", "c"], ["a", "z"]) == 0.5
    assert stage1_recall(["a", "b", "c"], ["c", "a"]) == 1.0
    assert stage1_recall(["a", "a"], ["a", "b", "b"]) == 0.5
    assert stage1_recall([], ["a"]) == 0.0
    assert math.isnan(stage1_recall(None, ["a"]))  # the ranker has no stage 1
    assert math.isnan(stage1_recall(["a"], []))


# --------------------------------------------------------------------------- #
# Bootstrap                                                                    #
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_is_deterministic():
    values = [0.0, 1.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.75, 1.0, 0.5]
    first = bootstrap_ci(values)
    assert first == bootstrap_ci(values)
    assert first == bootstrap_ci(iter(values))  # any iterable
    assert first == bootstrap_ci(values, seed=config.SEED, n_resamples=config.BOOTSTRAP_RESAMPLES)
    other_seed = bootstrap_ci(values, seed=config.SEED + 1)
    assert other_seed.mean == first.mean
    assert (other_seed.lo, other_seed.hi) != (first.lo, first.hi)


def test_bootstrap_ci_is_the_percentile_bootstrap_of_the_mean():
    values = [0.0, 1.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.75]
    data = np.array(values)
    index = np.random.default_rng(5).integers(0, len(values), size=(400, len(values)))
    lo, hi = np.quantile(data[index].mean(axis=1), [0.05, 0.95])

    estimate = bootstrap_ci(values, n_resamples=400, level=0.9, seed=5)

    assert estimate == Estimate(float(data.mean()), float(lo), float(hi), 8)


def test_bootstrap_ci_brackets_the_mean_and_matches_the_normal_approximation():
    values = [0.0, 1.0] * 200  # p = 0.5 over 400 queries: standard error 0.025
    estimate = bootstrap_ci(values)
    assert estimate.mean == 0.5
    assert estimate.n == 400
    assert estimate.lo < estimate.mean < estimate.hi
    assert estimate.mean - estimate.lo == pytest.approx(1.96 * 0.025, abs=0.006)
    assert estimate.hi - estimate.mean == pytest.approx(1.96 * 0.025, abs=0.006)


def test_bootstrap_ci_widens_with_the_level():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0] * 3
    narrow, wide = bootstrap_ci(values, level=0.5), bootstrap_ci(values, level=0.95)
    assert wide.lo < narrow.lo <= narrow.mean <= narrow.hi < wide.hi


def test_bootstrap_ci_degenerate_inputs():
    empty = bootstrap_ci([])
    assert empty.n == 0 and not empty.defined
    assert math.isnan(empty.mean) and math.isnan(empty.lo) and math.isnan(empty.hi)

    assert bootstrap_ci([0.25]) == Estimate(0.25, 0.25, 0.25, 1)
    assert bootstrap_ci([1.0] * 7) == Estimate(1.0, 1.0, 1.0, 7)
    assert bootstrap_ci([0.0, 0.0]) == Estimate(0.0, 0.0, 0.0, 2)


def test_bootstrap_ci_drops_undefined_values():
    values = [1.0, math.nan, 0.0, 1.0, math.nan]
    assert bootstrap_ci(values) == bootstrap_ci([1.0, 0.0, 1.0])
    assert bootstrap_ci(values).n == 3
    assert bootstrap_ci([math.nan, math.nan]).n == 0


def test_bootstrap_ci_shares_resamples_between_metrics_over_the_same_queries():
    hits = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    hit_rate = bootstrap_ci(hits)
    miss_rate = bootstrap_ci([1.0 - hit for hit in hits])
    assert miss_rate.mean == pytest.approx(1.0 - hit_rate.mean)
    assert miss_rate.lo == pytest.approx(1.0 - hit_rate.hi)
    assert miss_rate.hi == pytest.approx(1.0 - hit_rate.lo)


def test_bootstrap_ci_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([1.0], n_resamples=0)
    for level in (0.0, 1.0, 95):
        with pytest.raises(ValueError, match="level"):
            bootstrap_ci([1.0], level=level)


def test_paired_bootstrap_diff_is_the_mean_difference():
    a = [1.0, 1.0, 1.0, 0.0, 0.5]
    b = [0.0, 1.0, 0.0, 0.0, 1.0]
    diff = paired_bootstrap_diff(a, b)
    check(diff, 0.3, 5)  # (1 + 0 + 1 + 0 - 0.5) / 5
    assert diff == bootstrap_ci([1.0, 0.0, 1.0, 0.0, -0.5])
    assert diff == paired_bootstrap_diff(a, b)

    mirrored = paired_bootstrap_diff(b, a)
    assert mirrored.mean == pytest.approx(-diff.mean)
    assert mirrored.lo == pytest.approx(-diff.hi)
    assert mirrored.hi == pytest.approx(-diff.lo)


def test_paired_bootstrap_diff_removes_the_variance_both_sides_share():
    b = [0.0, 0.2, 0.4, 0.6, 0.8] * 4
    a = [value + 0.1 for value in b]
    # Unpaired, the two intervals overlap almost entirely ...
    assert bootstrap_ci(a).lo < bootstrap_ci(b).hi
    # ... paired, every query agrees on the difference.
    diff = paired_bootstrap_diff(a, b)
    assert diff.n == 20
    assert diff.mean == pytest.approx(0.1)
    assert diff.lo == pytest.approx(0.1)
    assert diff.hi == pytest.approx(0.1)


def test_paired_bootstrap_diff_degenerate_inputs():
    same = [0.0, 1.0, 0.5]
    assert paired_bootstrap_diff(same, same) == Estimate(0.0, 0.0, 0.0, 3)
    assert not paired_bootstrap_diff([], []).defined
    # A pair with an undefined side is dropped.
    assert paired_bootstrap_diff([1.0, math.nan, 1.0], [0.0, 0.0, math.nan]) == Estimate(1.0, 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="differ in length"):
        paired_bootstrap_diff([1.0, 0.0], [1.0])


def test_latency_summary():
    summary = latency_summary([5.0, 1.0, 3.0, 2.0, 4.0])
    assert summary == pytest.approx({"p50": 3.0, "p95": 4.8, "mean": 3.0, "n": 5})
    assert latency_summary(iter([0.25])) == {"p50": 0.25, "p95": 0.25, "mean": 0.25, "n": 1}

    empty = latency_summary([])
    assert empty["n"] == 0
    assert all(math.isnan(empty[key]) for key in ("p50", "p95", "mean"))


def test_estimate_json_round_trip():
    estimate = Estimate(0.5, 0.25, 0.75, 12)
    assert estimate.to_json() == {"mean": 0.5, "lo": 0.25, "hi": 0.75, "n": 12}
    assert Estimate.from_json(estimate.to_json()) == estimate
    assert as_estimate(estimate) is estimate
    assert as_estimate(estimate.to_json()) == estimate

    undefined = bootstrap_ci([])
    assert undefined.to_json() == {"mean": None, "lo": None, "hi": None, "n": 0}
    assert not Estimate.from_json(undefined.to_json()).defined

    assert as_estimate({"p50": 1.0, "p95": 2.0, "mean": 1.5, "n": 3}) is None  # a latency summary
    assert as_estimate(None) is None
    assert as_estimate(0.5) is None


def test_table_helpers():
    names = ["zeta", "jev_full_pool", "bge_m3", "alpha", "bge_m3"]
    assert ordered_rankers(names) == ["bge_m3", "jev_full_pool", "alpha", "zeta"]
    assert ordered_rankers(reversed(config.RANKER_NAMES)) == list(config.RANKER_NAMES)
    assert numeric_keys(["10", "5", "overall", "20", "5"]) == ["5", "10", "20"]

    tree = {"a": {"b": 1}}
    assert dig(tree, "a", "b") == 1
    assert dig(tree, "a") == {"b": 1}
    assert dig(tree, "a", "x") is None
    assert dig(tree, "a", "b", "c") is None
    assert dig(tree, None) is None


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #
def ranked(
    query_id: str,
    ranked_ids: list[str],
    gold_ids: list[str],
    *,
    ranker: str = "bge_m3",
    type: str = "lookup",
    pool_size: int = 10,
    run: int = 0,
    scores: list[float] | None = None,
    stage1_ids: list[str] | None = None,
    latency_s: float = 1.0,
    stage_latency_s: dict[str, float] | None = None,
    cost_usd: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> RankerRecord:
    return RankerRecord(
        query_id=query_id,
        type=type,
        pool_size=pool_size,
        gold_ids=list(gold_ids),
        ranker=ranker,
        run=run,
        ranked_ids=list(ranked_ids),
        scores=list(scores) if scores is not None else [1.0 - 0.1 * i for i in range(len(ranked_ids))],
        stage1_ids=stage1_ids,
        latency_s=latency_s,
        stage_latency_s=dict(stage_latency_s or {}),
        usage=Usage(cost_usd=cost_usd),
        extra=dict(extra or {}),
    )


def test_ranker_record_json_round_trip():
    record = ranked(
        "q1",
        ["a", "b"],
        ["a"],
        ranker="bge_m3_jev",
        type="temporal",
        pool_size=120,
        run=2,
        scores=[0.9, 0.25],
        stage1_ids=["a", "b", "c"],
        latency_s=1.5,
        stage_latency_s={"rerank": 1.25},
        cost_usd=0.002,
        extra={"n_requests": 3, "n_missing": 1, "n_cached": 0, "jev_model": "jev-x", "scoring": "pointwise"},
    )
    data = json.loads(json.dumps(record.to_json()))
    assert data["extra"]["n_missing"] == 1
    assert RankerRecord.from_json(data) == record

    without_stage1 = ranked("q2", ["a"], ["a"])
    data = json.loads(json.dumps(without_stage1.to_json()))
    assert data["stage1_ids"] is None and data["extra"] == {}
    assert RankerRecord.from_json(data) == without_stage1


def test_ranker_record_extra_defaults_to_an_empty_dict():
    fields = dict(
        query_id="q1",
        type="lookup",
        pool_size=3,
        gold_ids=["a"],
        ranker="bge_m3",
        run=0,
        ranked_ids=["a"],
        scores=[1.0],
        stage1_ids=None,
        latency_s=0.1,
        stage_latency_s={},
        usage=Usage(),
    )
    first, second = RankerRecord(**fields), RankerRecord(**fields)
    assert first.extra == {}
    first.extra["n_missing"] = 1
    assert second.extra == {}  # not shared between records

    data = first.to_json()
    del data["extra"]  # a record written before the field existed
    assert RankerRecord.from_json(data).extra == {}


def test_ranker_record_validation():
    with pytest.raises(ValueError, match="2 ranked ids but 1 scores"):
        ranked("q1", ["a", "b"], ["a"], scores=[1.0])
    with pytest.raises(ValueError, match="unknown query type"):
        ranked("q1", ["a"], ["a"], type="trivia")


def system_record(**overrides: Any) -> SystemRecord:
    fields: dict[str, Any] = dict(
        query_id="s1",
        type="temporal",
        pool_size=42,
        ranker="bge_m3_reranker",
        k=10,
        k_prime=5,
        path="fallback",
        needed_dialogue=True,
        final_answer="On Tuesday.",
        verdict="correct",
        gold_ids=["g1", "g2"],
        retrieved_ids=["g1", "x", "g2"],
        gold_name="Sam",
        extracted_name="Sam",
        name_correct=True,
        person_correct=True,
        extracted_time_range=RANGE,
        time_filter_ok=True,
        unknown_name_exact=None,
        usage_by_stage={"extraction": Usage(100, 10, 0.001, 0.5, 1), "rank": Usage()},
        latency_by_stage={"extraction": 0.5, "rank": 0.125},
    )
    return SystemRecord(**{**fields, **overrides})


def test_system_record_json_round_trip():
    record = system_record()
    data = json.loads(json.dumps(record.to_json()))
    assert data["extracted_time_range"] == {"start": "2026-10-26T00:00:00Z", "end": "2026-11-01T23:59:59Z"}
    assert SystemRecord.from_json(data) == record

    early = system_record(
        type="unknown_name",
        pool_size=None,
        path="unknown_name",
        needed_dialogue=False,
        verdict=None,
        gold_ids=[],
        retrieved_ids=[],
        extracted_time_range=None,
        time_filter_ok=None,
        unknown_name_exact=True,
    )
    data = json.loads(json.dumps(early.to_json()))
    assert data["pool_size"] is None and data["verdict"] is None and data["time_filter_ok"] is None
    assert SystemRecord.from_json(data) == early


def test_system_record_keeps_whether_the_ranking_was_a_cache_replay():
    assert system_record().rank_cached is False
    cached = system_record(rank_cached=True)
    data = json.loads(json.dumps(cached.to_json()))
    assert data["rank_cached"] is True
    assert SystemRecord.from_json(data) == cached
    # A record written before the flag existed reads as timed.
    del data["rank_cached"]
    assert SystemRecord.from_json(data).rank_cached is False


def test_system_record_validation():
    with pytest.raises(ValueError, match="unknown path"):
        system_record(path="guessed")
    with pytest.raises(ValueError, match="unknown verdict"):
        system_record(verdict="maybe")
    with pytest.raises(ValueError, match="unknown query type"):
        system_record(type="trivia")


# --------------------------------------------------------------------------- #
# Ranker comparison                                                            #
# --------------------------------------------------------------------------- #
KS = (1, 3)
QUERIES: dict[str, tuple[str, int, list[str]]] = {  # query id -> (type, pool size, gold ids)
    "q1": ("lookup", 10, ["g1"]),
    "q2": ("lookup", 50, ["g2"]),
    "q3": ("multi_conversation", 150, ["g3a", "g3b"]),
    "q4": ("multi_conversation", 150, ["g4a", "g4b", "g4c"]),
    "q5": ("temporal", 8, ["g5a", "g5b"]),
    "q6": ("cross_person", 800, ["g6"]),
}
BGE_RANKED = {  # recall@1, recall@3
    "q1": ["g1", "x1", "x2"],  # 1, 1
    "q2": ["x1", "g2", "x2"],  # 0, 1
    "q3": ["g3a", "x1", "g3b"],  # 1/2, 1
    "q4": ["x1", "g4a", "x2"],  # 0, 1/3
    "q5": ["g5a", "g5b", "x1"],  # 1/2, 1
    "q6": ["x1", "x2", "x3"],  # 0, 0
}
JEV_RANKED = {  # recall@1, recall@3
    "q1": ["x1", "g1", "x2"],  # 0, 1
    "q2": ["g2", "x1", "x2"],  # 1, 1
    "q3": ["g3a", "g3b", "x1"],  # 1/2, 1
    "q4": ["g4a", "g4b", "x1"],  # 1/3, 2/3
    "q5": ["x1", "g5a", "x2"],  # 0, 1/2
    "q6": ["g6", "x1", "x2"],  # 1, 1
}
JEV_STAGE1 = {  # stage-1 recall
    "q1": ["g1", "x1", "x2"],  # 1
    "q2": ["g2", "x1", "x2"],  # 1
    "q3": ["g3a", "g3b", "x1"],  # 1
    "q4": ["g4a", "g4b", "x1", "x2"],  # 2/3: stage 1 dropped g4c
    "q5": ["x1", "g5a", "x2", "g5b"],  # 1
    "q6": ["g6", "x1", "x2"],  # 1
}
JEV_CACHED = {"q2": 2, "q5": 1}  # requests answered from the cache
JEV_MISSING = {"q2": 1, "q4": 2}  # candidates left without a score


def comparison_records() -> list[RankerRecord]:
    """Two rankers over six queries: bge_m3 takes 0.1 s .. 0.6 s, bge_m3_jev 1 s .. 6 s."""

    records = []
    for index, (query_id, (query_type, pool_size, gold)) in enumerate(QUERIES.items(), start=1):
        shared = {"type": query_type, "pool_size": pool_size}
        records.append(ranked(query_id, BGE_RANKED[query_id], gold, latency_s=0.1 * index, **shared))
        records.append(
            ranked(
                query_id,
                JEV_RANKED[query_id],
                gold,
                ranker="bge_m3_jev",
                stage1_ids=JEV_STAGE1[query_id],
                latency_s=float(index),
                stage_latency_s={"vector_search": 0.5, "rerank": index - 0.5},
                cost_usd=0.001 * index,
                extra={
                    "n_requests": 2,
                    "n_missing": JEV_MISSING.get(query_id, 0),
                    "n_cached": JEV_CACHED.get(query_id, 0),
                    "jev_model": "jev-test",
                    "scoring": "pointwise",
                },
                **shared,
            )
        )
    return records


@pytest.fixture(scope="module")
def comparison() -> dict[str, Any]:
    return ranker_comparison(comparison_records(), ks=KS)


def test_ranker_comparison_recall(comparison):
    assert list(comparison) == ["bge_m3", "bge_m3_jev"]
    bge, jev = comparison["bge_m3"]["overall"], comparison["bge_m3_jev"]["overall"]
    assert bge["n"] == jev["n"] == 6
    check(bge["recall"]["1"], 1 / 3, 6)  # (1 + 0 + 1/2 + 0 + 1/2 + 0) / 6
    check(bge["recall"]["3"], 13 / 18, 6)  # (1 + 1 + 1 + 1/3 + 1 + 0) / 6
    check(jev["recall"]["1"], 17 / 36, 6)  # (0 + 1 + 1/2 + 1/3 + 0 + 1) / 6
    check(jev["recall"]["3"], 31 / 36, 6)  # (1 + 1 + 1 + 2/3 + 1/2 + 1) / 6
    # Each estimate is the bootstrap over the per-query values in query-id order.
    assert bge["recall"]["1"] == bootstrap_ci([1, 0, 0.5, 0, 0.5, 0])
    assert jev["recall"]["3"] == bootstrap_ci([1, 1, 1, 2 / 3, 0.5, 1])


def test_ranker_comparison_uses_the_report_cutoffs_by_default():
    table = ranker_comparison(comparison_records())
    assert list(table["bge_m3"]["overall"]["recall"]) == ["5", "10", "20", "30"]
    check(table["bge_m3"]["overall"]["recall"]["30"], 13 / 18, 6)  # lists hold three ids


def test_ranker_comparison_all_found(comparison):
    bge, jev = comparison["bge_m3"]["overall"], comparison["bge_m3_jev"]["overall"]
    # Two or more gold conversations: q3, q4, q5 (q5 is not of the multi_conversation type).
    check(bge["all_found"]["1"], 0.0, 3)
    check(bge["all_found"]["3"], 2 / 3, 3)  # q3 and q5
    check(jev["all_found"]["3"], 1 / 3, 3)  # q3
    # The multi_conversation type alone: q3, q4.
    check(bge["all_found_multi_conversation"]["3"], 1 / 2, 2)
    check(jev["all_found_multi_conversation"]["3"], 1 / 2, 2)
    check(jev["all_found_multi_conversation"]["1"], 0.0, 2)
    # A slice without such queries has nothing to average.
    lookup = comparison["bge_m3"]["by_type"]["lookup"]
    assert lookup["all_found"]["3"].n == 0 and not lookup["all_found"]["3"].defined


def test_ranker_comparison_stage1_recall(comparison):
    jev = comparison["bge_m3_jev"]
    check(jev["overall"]["stage1_recall"], 8 / 9, 3)  # pools over 100: q3, q4, q6
    check(jev["overall"]["stage1_recall_all_pools"], 17 / 18, 6)
    check(jev["by_pool"][">100"]["stage1_recall"], 8 / 9, 3)
    small = jev["by_pool"]["<=20"]
    assert small["stage1_recall"].n == 0 and not small["stage1_recall"].defined
    check(small["stage1_recall_all_pools"], 1.0, 2)
    # A ranker without a stage 1 reports none.
    bge = comparison["bge_m3"]["overall"]
    assert bge["stage1_recall"] is None and bge["stage1_recall_all_pools"] is None


def test_ranker_comparison_stage1_recall_counts_pools_over_the_cutoff_only():
    records = [
        ranked("q1", ["g"], ["g"], ranker="bge_m3_reranker", pool_size=config.STAGE1_TOP_K, stage1_ids=["g"]),
        ranked("q2", ["x"], ["g"], ranker="bge_m3_reranker", pool_size=config.STAGE1_TOP_K + 1, stage1_ids=["x"]),
    ]
    overall = ranker_comparison(records)["bge_m3_reranker"]["overall"]
    check(overall["stage1_recall"], 0.0, 1)  # stage 1 can only drop gold from a pool it has to cut
    check(overall["stage1_recall_all_pools"], 0.5, 2)


def test_ranker_comparison_slices_by_type_and_pool(comparison):
    bge = comparison["bge_m3"]
    assert list(bge["by_type"]) == ["lookup", "multi_conversation", "cross_person", "temporal"]
    assert list(bge["by_pool"]) == ["<=20", "21-100", ">100"]
    assert {name: entry["n"] for name, entry in bge["by_type"].items()} == {
        "lookup": 2,
        "multi_conversation": 2,
        "cross_person": 1,
        "temporal": 1,
    }
    check(bge["by_type"]["lookup"]["recall"]["1"], 1 / 2, 2)  # q1, q2
    check(bge["by_type"]["multi_conversation"]["recall"]["1"], 1 / 4, 2)  # q3, q4
    check(bge["by_type"]["multi_conversation"]["recall"]["3"], 2 / 3, 2)
    check(bge["by_type"]["cross_person"]["recall"]["3"], 0.0, 1)  # q6
    check(bge["by_type"]["temporal"]["recall"]["1"], 1 / 2, 1)  # q5

    assert {name: entry["n"] for name, entry in bge["by_pool"].items()} == {"<=20": 2, "21-100": 1, ">100": 3}
    check(bge["by_pool"]["<=20"]["recall"]["1"], 3 / 4, 2)  # q1, q5
    check(bge["by_pool"]["21-100"]["recall"]["1"], 0.0, 1)  # q2
    check(bge["by_pool"][">100"]["recall"]["1"], 1 / 6, 3)  # q3, q4, q6
    check(bge["by_pool"][">100"]["recall"]["3"], 4 / 9, 3)
    check(comparison["bge_m3_jev"]["by_pool"][">100"]["recall"]["1"], 11 / 18, 3)  # (1/2 + 1/3 + 1) / 3


def test_ranker_comparison_latency_leaves_out_cached_records(comparison):
    bge, jev = comparison["bge_m3"], comparison["bge_m3_jev"]
    # No cache in play: all six records, 0.1 s .. 0.6 s.
    assert bge["overall"]["latency"] == pytest.approx({"p50": 0.35, "p95": 0.575, "mean": 0.35, "n": 6})
    assert bge["overall"]["stage_latency_p50"] == {}
    # q2 and q5 were served from the cache: the figures rest on q1, q3, q4, q6 = 1, 3, 4, 6 s.
    assert jev["overall"]["n"] == 6
    assert jev["overall"]["latency"] == pytest.approx({"p50": 3.5, "p95": 5.7, "mean": 3.5, "n": 4})
    assert jev["overall"]["stage_latency_p50"] == pytest.approx({"vector_search": 0.5, "rerank": 3.0})

    assert jev["by_pool"][">100"]["latency"] == pytest.approx({"p50": 4.0, "p95": 5.8, "mean": 13 / 3, "n": 3})
    assert jev["by_pool"]["<=20"]["latency"] == pytest.approx({"p50": 1.0, "p95": 1.0, "mean": 1.0, "n": 1})
    only_cached = jev["by_pool"]["21-100"]  # q2 alone
    assert only_cached["n"] == 1
    assert only_cached["latency"]["n"] == 0 and math.isnan(only_cached["latency"]["p50"])
    assert only_cached["stage_latency_p50"] == {}


def test_ranker_comparison_cost_covers_every_record(comparison):
    assert comparison["bge_m3"]["overall"]["cost_per_1k_usd"] == 0.0
    jev = comparison["bge_m3_jev"]
    assert jev["overall"]["cost_per_1k_usd"] == pytest.approx(3.5)  # mean of $0.001 .. $0.006
    assert jev["by_pool"]["21-100"]["cost_per_1k_usd"] == pytest.approx(2.0)  # the cached q2
    assert jev["by_pool"][">100"]["cost_per_1k_usd"] == pytest.approx(13 / 3)  # q3, q4, q6


def test_ranker_comparison_totals_the_missing_scores(comparison):
    assert comparison["bge_m3_jev"]["n_missing"] == 3
    assert comparison["bge_m3"]["n_missing"] is None  # the ranker does not count them

    none_missing = [record for record in comparison_records() if record.query_id not in JEV_MISSING]
    assert ranker_comparison(none_missing, ks=KS)["bge_m3_jev"]["n_missing"] == 0


def test_ranker_comparison_paired_differences(comparison):
    diff = comparison["bge_m3_jev"]["recall_diff_vs"]["bge_m3"]
    # bge_m3_jev minus bge_m3 per query, at 1: -1, 1, 0, 1/3, -1/2, 1; at 3: 0, 0, 0, 1/3, -1/2, 1.
    check(diff["overall"]["1"], 5 / 36, 6)
    check(diff["overall"]["3"], 5 / 36, 6)
    assert diff["overall"]["1"] == paired_bootstrap_diff([0, 1, 0.5, 1 / 3, 0, 1], [1, 0, 0.5, 0, 0.5, 0])
    check(diff["by_type"]["lookup"]["1"], 0.0, 2)  # -1 and 1
    check(diff["by_pool"][">100"]["1"], 4 / 9, 3)  # 0, 1/3, 1
    check(diff["by_pool"]["<=20"]["3"], -1 / 4, 2)  # 0, -1/2

    mirrored = comparison["bge_m3"]["recall_diff_vs"]["bge_m3_jev"]
    for scope, k in (("overall", "1"), ("overall", "3")):
        ours, theirs = mirrored[scope][k], diff[scope][k]
        assert (ours.mean, ours.lo, ours.hi, ours.n) == (-theirs.mean, -theirs.hi, -theirs.lo, theirs.n)
    assert list(mirrored["by_pool"]) == list(diff["by_pool"])


def test_ranker_comparison_pairs_only_the_queries_both_rankers_ran():
    records = [
        record
        for record in comparison_records()
        if not (record.ranker == "bge_m3_jev" and record.query_id == "q6")
    ]
    table = ranker_comparison(records, ks=KS)
    assert table["bge_m3"]["overall"]["n"] == 6 and table["bge_m3_jev"]["overall"]["n"] == 5
    check(table["bge_m3_jev"]["recall_diff_vs"]["bge_m3"]["overall"]["1"], -1 / 30, 5)


def test_ranker_comparison_recall_gain_and_failures(comparison):
    bge, jev = comparison["bge_m3"], comparison["bge_m3_jev"]
    assert list(bge["recall_gain"]) == ["1"] and list(bge["recall_gain"]["1"]) == ["3"]
    check(bge["recall_gain"]["1"]["3"], 7 / 18, 6)  # 0, 1, 1/2, 1/3, 1/2, 0
    check(jev["recall_gain"]["1"]["3"], 7 / 18, 6)  # 1, 0, 1/2, 1/3, 1/2, 0
    assert bge["failures"] == {"stage1_misses": None, "gold_not_in_top_k": {"1": 5, "3": 2}}
    assert jev["failures"] == {"stage1_misses": 1, "gold_not_in_top_k": {"1": 4, "3": 2}}


def test_ranker_comparison_ignores_record_order_and_repeated_runs(comparison):
    records = comparison_records()
    random.Random(3).shuffle(records)
    records.append(ranked("q1", ["x1", "x2", "x3"], ["g1"], ranker="bge_m3_jev", run=1))
    records.append(ranked("q1", ["g1"], ["g1"], ranker="jev_full_pool", run=1))  # no run 0 at all

    table = ranker_comparison(records, ks=KS)

    assert plain(table) == plain(comparison)


def test_ranker_comparison_rejects_a_query_recorded_twice():
    records = comparison_records()
    with pytest.raises(ValueError, match="duplicate record for query 'q1'"):
        ranker_comparison([*records, records[0]], ks=KS)


def test_ranker_comparison_without_records():
    assert ranker_comparison([]) == {}


# --------------------------------------------------------------------------- #
# Jev run-to-run variance                                                      #
# --------------------------------------------------------------------------- #
def variance_records() -> list[RankerRecord]:
    def jev(query_id: str, run: int, ranked_ids: list[str], scores: list[float], gold: str) -> RankerRecord:
        return ranked(query_id, ranked_ids, [gold], ranker="jev_full_pool", run=run, scores=scores)

    return [
        # recall@1 per run: (1 + 0) / 2, (0 + 1) / 2, (1 + 1) / 2; recall@2 is 1 everywhere.
        jev("q1", 0, ["a", "b", "c"], [0.9, 0.5, 0.1], "a"),
        jev("q2", 0, ["d", "e"], [0.8, 0.2], "e"),
        jev("q1", 1, ["b", "a", "c"], [0.7, 0.6, 0.1], "a"),
        jev("q2", 1, ["e", "d"], [0.6, 0.4], "e"),
        jev("q1", 2, ["a", "b", "c"], [0.9, 0.4, 0.2], "a"),
        jev("q2", 2, ["e", "f"], [0.9, 0.1], "e"),
        # q3 is missing from run 2, so it is left out of every statistic.
        jev("q3", 0, ["g"], [0.5], "g"),
        jev("q3", 1, ["h"], [0.5], "g"),
        # A ranker that ran once has no variance to report.
        ranked("q1", ["a", "b", "c"], ["a"], ranker="bge_m3"),
        ranked("q2", ["d", "e"], ["e"], ranker="bge_m3"),
    ]


def test_jev_variance_statistics():
    table = jev_variance(variance_records(), ks=(1, 2))

    assert list(table) == ["jev_full_pool"]
    entry = table["jev_full_pool"]
    assert entry["runs"] == [0, 1, 2]
    assert entry["n_queries"] == 2
    assert entry["snapshots_per_run"] == [[], [], []]  # the records above carry no jev_model
    assert entry["recall"]["1"]["per_run"] == [0.5, 0.5, 1.0]
    assert entry["recall"]["1"]["mean"] == pytest.approx(2 / 3)
    assert entry["recall"]["1"]["std"] == pytest.approx(math.sqrt(1 / 12))  # sample std of .5, .5, 1
    assert entry["recall"]["2"] == {"per_run": [1.0, 1.0, 1.0], "mean": 1.0, "std": 0.0}
    # |score differences| over the conversations two runs both returned:
    #   runs 0, 1: q1 .3 + .2 + 0, q2 .4 + .4      -> 1.3 over 5
    #   runs 0, 2: q1 0 + .1 + .1, q2 .7 (e only)  -> 0.9 over 4
    #   runs 1, 2: q1 .3 + .3 + .1, q2 .3 (e only) -> 1.0 over 4
    assert entry["mean_abs_score_diff"] == pytest.approx(3.2 / 13)
    assert entry["n_unscored_pairs"] == 0
    # Jaccard of the top-10 sets: 1 for q1 in every pair; for q2: 1, 1/3 and 1/3.
    assert entry["top10_jaccard"] == pytest.approx((3 + 1 + 1 / 3 + 1 / 3) / 6)


def test_jev_variance_leaves_out_the_pairs_a_run_did_not_score():
    # Run 1 refused b: its MISSING_SCORE sentinel must not pass for a score drift of 1.8.
    records = [
        ranked("q1", ["a", "b", "c"], ["a"], ranker="jev_full_pool", scores=[0.9, 0.8, 0.1]),
        ranked("q1", ["a", "c", "b"], ["a"], ranker="jev_full_pool", run=1, scores=[0.9, 0.1, MISSING_SCORE]),
    ]

    entry = jev_variance(records, ks=(1,))["jev_full_pool"]

    assert entry["mean_abs_score_diff"] == 0.0  # a and c did not move
    assert entry["n_unscored_pairs"] == 1  # b
    assert entry["top10_jaccard"] == 1.0

    # With every shared conversation unscored in one of the runs there is no drift to report.
    unscored = [records[0], ranked("q1", ["a", "b", "c"], ["a"], ranker="jev_full_pool", run=1, scores=[MISSING_SCORE] * 3)]
    entry = jev_variance(unscored, ks=(1,))["jev_full_pool"]
    assert math.isnan(entry["mean_abs_score_diff"]) and entry["n_unscored_pairs"] == 3


def test_jev_variance_reports_the_snapshot_each_run_hit():
    def jev(run: int, snapshot: str | None) -> RankerRecord:
        extra = {} if snapshot is None else {"jev_model": snapshot}
        return ranked("q1", ["a", "b"], ["a"], ranker="jev_full_pool", run=run, extra=extra)

    records = [jev(0, "jev-1.14-20261014"), jev(1, "jev-1.14-20261014,jev-1.14-20261101"), jev(2, None)]
    entry = jev_variance(records, ks=(1,))["jev_full_pool"]
    assert entry["runs"] == [0, 1, 2]
    assert entry["snapshots_per_run"] == [["jev-1.14-20261014"], ["jev-1.14-20261014", "jev-1.14-20261101"], []]
    assert jev_snapshots(records) == ["jev-1.14-20261014", "jev-1.14-20261101"]
    assert jev_snapshots([ranked("q1", ["a"], ["a"])]) == []


def test_jev_variance_compares_the_top_ten_only():
    ids = [f"e{i}" for i in range(12)]
    scores = [1.0 - 0.05 * i for i in range(12)]
    swapped = [*ids[:9], ids[11], ids[9], ids[10]]  # e11 moves into the top ten, e9 out of it
    reordered_tail = [*ids[:10], ids[11], ids[10]]  # same top ten

    def table_for(second_run: list[str]) -> dict[str, Any]:
        records = [
            ranked("q1", ids, ["e0"], ranker="bge_m3_jev", scores=scores),
            ranked("q1", second_run, ["e0"], ranker="bge_m3_jev", run=1, scores=scores),
        ]
        return jev_variance(records)["bge_m3_jev"]

    assert table_for(swapped)["top10_jaccard"] == pytest.approx(9 / 11)
    assert table_for(reordered_tail)["top10_jaccard"] == 1.0
    identical = table_for(ids)
    assert identical["mean_abs_score_diff"] == 0.0
    assert identical["recall"]["5"] == {"per_run": [1.0, 1.0], "mean": 1.0, "std": 0.0}


def test_jev_variance_needs_more_than_one_run():
    assert jev_variance(comparison_records()) == {}
    assert jev_variance([]) == {}


# --------------------------------------------------------------------------- #
# System eval                                                                  #
# --------------------------------------------------------------------------- #
def outcome(
    type: str,
    pool_size: int | None,
    path: str,
    verdict: str | None = None,
    gold: str = "",
    retrieved: str = "",
    **flags: Any,
) -> dict[str, Any]:
    """One query's base outcome; ``gold`` and ``retrieved`` are space-separated episode ids."""

    return dict(
        type=type,
        pool_size=pool_size,
        path=path,
        verdict=verdict,
        gold_ids=gold.split(),
        retrieved_ids=retrieved.split(),
        **flags,
    )


SYSTEM_BASE: dict[str, dict[str, Any]] = {
    # Answerable (nine queries).
    "s01": outcome("lookup", 10, "summary", "correct", gold="g1", retrieved="g1 x"),
    "s02": outcome("lookup", 50, "fallback", "correct", gold="g2", retrieved="g2"),
    "s03": outcome("paraphrase", 150, "summary", "incorrect", gold="g3", retrieved="g3"),
    "s04": outcome("detail_only", 150, "fallback", "partial", gold="g4a g4b", retrieved="g4a x"),
    # The judge grades a refusal as incorrect; the runner judges every answerable query's answer.
    "s05": outcome("multi_conversation", 150, "declined", "incorrect", gold="g5a g5b", retrieved="g5b x g5a"),
    # Stopping before the ranking still ends in a final answer the judge grades (incorrect).
    "s06": outcome("temporal", 8, "empty_pool", "incorrect", gold="g6", extracted_time_range=RANGE, time_filter_ok=False),
    "s07": outcome(
        "temporal", 8, "summary", "correct", gold="g7", retrieved="g7", extracted_time_range=RANGE, time_filter_ok=True
    ),
    "s08": outcome(
        "cross_person",
        800,
        "unknown_name",
        "incorrect",
        gold="g8",
        extracted_name="Sam",
        name_correct=False,
        person_correct=False,
        extracted_time_range=RANGE,
    ),
    "s13": outcome("lookup", 5, "ambiguous", "incorrect", gold="g13", extracted_name="Alex", person_correct=False),
    # System-eval-only types.
    "s09": outcome("unanswerable", 20, "declined", retrieved="x"),
    "s10": outcome("unanswerable", 20, "summary", retrieved="x"),
    "s11": outcome("unknown_name", None, "unknown_name", extracted_name="Zed", unknown_name_exact=True),
    "s12": outcome("unknown_name", None, "summary", unknown_name_exact=False, name_correct=False, person_correct=False),
}


def stage_figures(path: str, needed_dialogue: bool, dialogue_input: int) -> tuple[dict[str, Usage], dict[str, float]]:
    """Usage and latency per stage, given where the flow ended.

    Totals per query: 1,100 tokens in / 60 out / $0.011 / 1.6 s when answered
    from summaries; 6,100 / 160 / $0.061 / 3.6 s when dialogue was needed;
    100 / 10 / $0.001 for an empty pool (0.6 s) or an early stop (0.5 s).
    """

    usage = {stage: Usage() for stage in ("extraction", "rank", "answer_summary", "answer_dialogue")}
    usage["extraction"] = Usage(100, 10, 0.001, 0.5, 1)
    latency = {"extraction": 0.5, "rank": 0.0, "answer_summary": 0.0, "answer_dialogue": 0.0}
    if path not in ("unknown_name", "ambiguous"):
        latency["rank"] = 0.1
    if path in ("summary", "fallback", "declined"):
        usage["answer_summary"] = Usage(1000, 50, 0.01, 1.0, 1)
        latency["answer_summary"] = 1.0
    if needed_dialogue:
        usage["answer_dialogue"] = Usage(dialogue_input, 100, 0.05, 2.0, 1)
        latency["answer_dialogue"] = 2.0
    return usage, latency


def system_records(
    ranker: str = "bge_m3",
    k: int = 5,
    k_prime: int = 3,
    changes: dict[str, dict[str, Any]] | None = None,
    dialogue_input: int = 5000,
) -> list[SystemRecord]:
    """The thirteen base queries at one setting, with per-query ``changes`` to the base outcome."""

    records = []
    for query_id, base in SYSTEM_BASE.items():
        spec = {**base, **(changes or {}).get(query_id, {})}
        needed_dialogue = spec["path"] in ("fallback", "declined")
        usage, latency = stage_figures(spec["path"], needed_dialogue, dialogue_input)
        records.append(
            SystemRecord(
                query_id=query_id,
                type=spec["type"],
                pool_size=spec["pool_size"],
                ranker=ranker,
                k=k,
                k_prime=k_prime,
                path=spec["path"],
                needed_dialogue=needed_dialogue,
                final_answer="an answer",
                verdict=spec["verdict"],
                gold_ids=list(spec["gold_ids"]),
                retrieved_ids=list(spec["retrieved_ids"]),
                gold_name=None,
                extracted_name=spec.get("extracted_name"),
                name_correct=spec.get("name_correct", True),
                person_correct=spec.get("person_correct", True),
                extracted_time_range=spec.get("extracted_time_range"),
                time_filter_ok=spec.get("time_filter_ok"),
                unknown_name_exact=spec.get("unknown_name_exact"),
                usage_by_stage=usage,
                latency_by_stage=latency,
            )
        )
    return records


BASE_FAILURES = {
    "extraction_name_errors": 2,  # s08, s12
    "missed_time_ranges": 0,  # both temporal queries had a range extracted
    "time_filter_misses": 1,  # s06
    "spurious_time_filters": 1,  # s08
    "misrouted": 2,  # s08, s13
    "empty_pools": 1,  # s06
    "gold_not_in_top_k": 1,  # s04 (one of its two); s06, s08 and s13 were never ranked
    "false_declines_gold_retrieved": 1,  # s05
    "wrong_answers_gold_retrieved": 1,  # s03; the declined s05 is not a wrong answer
    "wrong_answers_from_summaries": 1,  # s03
    "unjudged": 0,  # every answerable answer got a verdict
}
ANSWERED_AT_DEEPER_SETTING = {"path": "fallback", "verdict": "correct"}


def all_system_records() -> list[SystemRecord]:
    """bge_m3 at (k, k') = (5, 3), (10, 3) and (5, 5), and bge_m3_jev at (5, 3)."""

    deeper_k = {"s03": {"verdict": "correct"}, "s05": ANSWERED_AT_DEEPER_SETTING}
    deeper_fallback = {"s04": {"verdict": "correct"}, "s05": ANSWERED_AT_DEEPER_SETTING}
    other_ranker = {"s01": {"verdict": "incorrect"}, **deeper_k}
    return [
        *system_records(),
        *system_records(k=10, changes=deeper_k),
        *system_records(k_prime=5, changes=deeper_fallback, dialogue_input=8000),
        *system_records(ranker="bge_m3_jev", changes=other_ranker),
    ]


@pytest.fixture(scope="module")
def system_table() -> dict[str, Any]:
    return system_eval(all_system_records())


def test_system_eval_is_keyed_by_ranker_k_and_k_prime(system_table):
    assert list(system_table) == ["bge_m3", "bge_m3_jev"]
    assert list(system_table["bge_m3"]) == ["5", "10"]
    assert list(system_table["bge_m3"]["5"]) == ["3", "5"]
    assert list(system_table["bge_m3"]["10"]) == ["3"]
    assert list(system_table["bge_m3_jev"]) == ["5"]


def test_system_eval_rates(system_table):
    overall = system_table["bge_m3"]["5"]["3"]["overall"]
    assert overall["n"] == 13
    expected = {  # metric -> (mean, size of its population)
        "answer_accuracy": (3 / 9, 9),  # s01, s02, s07
        "partial_rate": (1 / 9, 9),  # s04
        "fallback_rate": (3 / 9, 9),  # s02, s04, s05
        "summary_path_accuracy": (2 / 3, 3),  # s01 and s07 of s01, s03, s07
        "fallback_path_accuracy": (1 / 2, 2),  # s02 of s02, s04
        "needed_dialogue_accuracy": (1 / 3, 3),  # s02 of s02, s04, s05
        "correct_decline_rate": (1 / 2, 2),  # s09 of s09, s10
        "false_decline_rate": (2 / 9, 9),  # s05, s06
        "false_decline_rate_gold_retrieved": (1 / 5, 5),  # s05 of s01, s02, s03, s05, s07
        "unknown_name_exact_rate": (1 / 2, 2),  # s11 of s11, s12
        "name_accuracy": (11 / 13, 13),  # all but s08, s12
        "person_accuracy": (10 / 13, 13),  # all but s08, s12, s13
        "time_range_extracted_rate": (2 / 2, 2),  # s06, s07
        "time_filter_recall": (1 / 2, 2),  # s07 of s06, s07
        "spurious_time_range_rate": (1 / 11, 11),  # s08 of the queries without a gold range
        "misrouted_rate": (2 / 9, 9),  # s08, s13
    }
    for metric, (mean, n) in expected.items():
        check(overall[metric], mean, n)
    assert set(overall) == {"n", "cost", *expected}
    # Answerable queries in query-id order: s01 .. s08, s13.
    assert overall["answer_accuracy"] == bootstrap_ci([1, 1, 0, 0, 0, 0, 1, 0, 0])


def check_cost(group: dict[str, Any], n: int, tokens_in: float, tokens_out: float, usd: float, p50: float) -> None:
    assert group["n"] == n and group["n_timed"] == n and group["latency"]["n"] == n
    assert group["input_tokens"] == pytest.approx(tokens_in)
    assert group["output_tokens"] == pytest.approx(tokens_out)
    assert group["cost_usd"] == pytest.approx(usd)
    assert group["latency"]["p50"] == pytest.approx(p50)


def test_system_eval_cost_split_by_path(system_table):
    cost = system_table["bge_m3"]["5"]["3"]["overall"]["cost"]
    assert list(cost) == ["overall", "summary", "fallback"]
    # Five queries answered from summaries, four that needed dialogue, four that stopped earlier.
    check_cost(cost["overall"], 13, 30300 / 13, 980 / 13, 0.303 / 13, 1.6)
    assert cost["overall"]["latency"]["p95"] == pytest.approx(3.6)
    assert cost["overall"]["latency"]["mean"] == pytest.approx(24.5 / 13)
    check_cost(cost["summary"], 5, 1100, 60, 0.011, 1.6)  # s01, s03, s07, s10, s12
    check_cost(cost["fallback"], 4, 6100, 160, 0.061, 3.6)  # s02, s04, s05, s09

    deeper = system_table["bge_m3"]["5"]["5"]["overall"]["cost"]
    check_cost(deeper["fallback"], 4, 9100, 160, 0.061, 3.6)
    check_cost(deeper["summary"], 5, 1100, 60, 0.011, 1.6)


def test_system_eval_latency_leaves_out_the_queries_whose_ranking_was_a_cache_replay():
    """A rank stage that replayed cached Jev answers timed a cache read, not the ranker."""

    rows = [
        system_record(query_id="s1", latency_by_stage={"extraction": 0.5, "rank": 0.125}),
        system_record(query_id="s2", latency_by_stage={"extraction": 0.5, "rank": 4.0}, rank_cached=True),
        system_record(query_id="s3", latency_by_stage={"extraction": 0.5, "rank": 0.625}),
    ]
    cost = system_eval(rows)["bge_m3_reranker"]["10"]["5"]["overall"]["cost"]["overall"]
    assert cost["n"] == 3 and cost["n_timed"] == 2
    assert cost["latency"]["n"] == 2
    assert cost["latency"]["mean"] == pytest.approx((0.625 + 1.125) / 2)
    assert cost["latency"]["p95"] < 4.0  # the 4.5 s cache replay is not among the percentiles
    # Tokens and dollars still cover every query: the cache replay cost what it cost.
    assert cost["input_tokens"] == pytest.approx(100.0)
    assert cost["cost_usd"] == pytest.approx(0.001)


def test_system_eval_slices_by_type_and_pool(system_table):
    cell = system_table["bge_m3"]["5"]["3"]
    by_type, by_pool = cell["by_type"], cell["by_pool"]
    assert {name: entry["n"] for name, entry in by_type.items()} == {
        "lookup": 3,
        "paraphrase": 1,
        "detail_only": 1,
        "multi_conversation": 1,
        "cross_person": 1,
        "temporal": 2,
        "unanswerable": 2,
        "unknown_name": 2,
    }
    check(by_type["lookup"]["answer_accuracy"], 2 / 3, 3)  # s01, s02 of s01, s02, s13
    check(by_type["lookup"]["misrouted_rate"], 1 / 3, 3)  # s13
    check(by_type["temporal"]["answer_accuracy"], 1 / 2, 2)
    check(by_type["temporal"]["time_filter_recall"], 1 / 2, 2)
    check(by_type["temporal"]["false_decline_rate"], 1 / 2, 2)
    check(by_type["unanswerable"]["correct_decline_rate"], 1 / 2, 2)
    assert by_type["unanswerable"]["answer_accuracy"].n == 0  # nothing answerable to grade
    check(by_type["unknown_name"]["unknown_name_exact_rate"], 1 / 2, 2)
    check(by_type["unknown_name"]["name_accuracy"], 1 / 2, 2)
    assert by_type["lookup"]["unknown_name_exact_rate"].n == 0
    check_cost(by_type["unknown_name"]["cost"]["summary"], 1, 1100, 60, 0.011, 1.6)  # s12

    # Unknown-name queries have no pool, so they are in no pool bucket.
    assert {name: entry["n"] for name, entry in by_pool.items()} == {"<=20": 6, "21-100": 1, ">100": 4}
    check(by_pool["<=20"]["answer_accuracy"], 2 / 4, 4)  # s01, s07 of s01, s06, s07, s13
    check(by_pool["<=20"]["correct_decline_rate"], 1 / 2, 2)
    check(by_pool["21-100"]["answer_accuracy"], 1.0, 1)  # s02
    check(by_pool[">100"]["answer_accuracy"], 0.0, 4)  # s03, s04, s05, s08
    check(by_pool[">100"]["fallback_rate"], 2 / 4, 4)  # s04, s05


def test_system_eval_failure_counts(system_table):
    failures = system_table["bge_m3"]["5"]["3"]["failures"]
    assert list(failures) == [mode for mode in FAILURE_MODES if mode != "stage1_misses"]
    assert failures == BASE_FAILURES
    # At k = 10 the paraphrase is answered correctly and the declined query is answered.
    assert system_table["bge_m3"]["10"]["3"]["failures"] == {
        **BASE_FAILURES,
        "false_declines_gold_retrieved": 0,
        "wrong_answers_gold_retrieved": 0,
        "wrong_answers_from_summaries": 0,
    }


def test_system_eval_charges_each_failure_to_the_stage_that_failed():
    # s05 declined with its gold retrieved and the judge graded the refusal incorrect: a false
    # decline, not a wrong answer. s06 (empty pool), s08 and s13 (misrouted) were never ranked,
    # so they are not gold-not-in-top-k failures; a ranked query that missed its gold still is.
    ranked_without_gold = {"s01": {"retrieved_ids": ["x"]}}
    failures = system_eval(system_records(changes=ranked_without_gold))["bge_m3"]["5"]["3"]["failures"]
    assert failures == {**BASE_FAILURES, "gold_not_in_top_k": 2}

    # A declined query whose answer the judge could not grade is still a false decline, and
    # the missing verdict is counted on its own, so the judge's share of the error is visible.
    unjudged = {"s05": {"verdict": None}}
    cell = system_eval(system_records(changes=unjudged))["bge_m3"]["5"]["3"]
    assert cell["failures"] == {**BASE_FAILURES, "unjudged": 1}
    check(cell["overall"]["answer_accuracy"], 3 / 9, 9)  # the unjudged answer is not a correct one
    # An unanswerable query has nothing to judge: its missing verdict is not an unjudged answer.
    assert system_eval(system_records())["bge_m3"]["5"]["3"]["failures"]["unjudged"] == 0


def test_system_eval_counts_a_temporal_query_without_a_range_as_a_time_filter_miss():
    # The extraction returned no range for s07: no padded range contains its gold conversation.
    missed = {"s07": {"extracted_time_range": None, "time_filter_ok": False}}
    cell = system_eval(system_records(changes=missed))["bge_m3"]["5"]["3"]

    check(cell["overall"]["time_range_extracted_rate"], 1 / 2, 2)  # s06 of s06, s07
    check(cell["overall"]["time_filter_recall"], 0.0, 2)
    check(cell["by_type"]["temporal"]["time_filter_recall"], 0.0, 2)
    assert cell["failures"] == {**BASE_FAILURES, "missed_time_ranges": 1}  # s06 is still the one dropped gold


def temporal_query(gold: list[str]) -> Query:
    return Query("q", "What did Dana say last week?", "temporal", "Dana", RANGE, gold, "An answer.")


def test_record_time_filter_ok_needs_a_filter_that_keeps_every_gold_conversation():
    inside = datetime(2026, 10, 28, 9, tzinfo=timezone.utc)
    outside = datetime(2026, 11, 3, 9, tzinfo=timezone.utc)
    starts = {"e1": inside, "e2": outside}
    query = temporal_query(["e1"])

    assert records.time_filter_ok(query, RANGE, starts) is True
    assert records.time_filter_ok(temporal_query(["e1", "e2"]), RANGE, starts) is False
    assert records.time_filter_ok(query, None, starts) is False  # no range extracted: nothing contains the gold
    without_gold_range = Query("q", "What did Dana say?", "lookup", "Dana", None, ["e1"], "An answer.")
    assert records.time_filter_ok(without_gold_range, None, starts) is None
    assert records.time_filter_ok(without_gold_range, RANGE, starts) is None  # spurious, but not a filter miss


def test_system_eval_gain_to_the_next_k(system_table):
    step = system_table["bge_m3"]["5"]["3"]["next_k"]
    assert step["k"] == "10"
    check(step["answer_accuracy_gain"], 2 / 9, 9)  # s03 and s05 turn correct
    check(system_table["bge_m3"]["10"]["3"]["overall"]["answer_accuracy"], 5 / 9, 9)
    assert system_table["bge_m3"]["10"]["3"]["next_k"] is None  # the largest k
    assert system_table["bge_m3"]["5"]["5"]["next_k"] is None  # no larger k at k' = 5


def test_system_eval_gain_to_the_next_k_prime(system_table):
    step = system_table["bge_m3"]["5"]["3"]["next_k_prime"]
    assert step["k_prime"] == "5"
    # Over the answerable queries that needed dialogue at both settings: s02, s04, s05.
    check(step["needed_dialogue_accuracy_gain"], 2 / 3, 3)
    assert step["extra_input_tokens"] == pytest.approx(3000)  # 8,000 instead of 5,000 dialogue tokens
    assert system_table["bge_m3"]["5"]["5"]["next_k_prime"] is None
    assert system_table["bge_m3"]["10"]["3"]["next_k_prime"] is None


def test_system_eval_paired_accuracy_difference_between_rankers(system_table):
    check(system_table["bge_m3_jev"]["5"]["3"]["overall"]["answer_accuracy"], 4 / 9, 9)
    diff = system_table["bge_m3_jev"]["5"]["3"]["accuracy_diff_vs"]["bge_m3"]
    check(diff, 1 / 9, 9)  # s01 lost, s03 and s05 gained
    assert diff == paired_bootstrap_diff([0, 1, 1, 0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0, 0])
    mirrored = system_table["bge_m3"]["5"]["3"]["accuracy_diff_vs"]["bge_m3_jev"]
    assert (mirrored.mean, mirrored.lo, mirrored.hi, mirrored.n) == (-diff.mean, -diff.hi, -diff.lo, 9)
    # Settings the other ranker did not run have nothing to compare against.
    assert system_table["bge_m3"]["10"]["3"]["accuracy_diff_vs"] == {}
    assert system_table["bge_m3"]["5"]["5"]["accuracy_diff_vs"] == {}


def test_system_eval_ignores_record_order(system_table):
    records = all_system_records()
    random.Random(11).shuffle(records)
    assert plain(system_eval(records)) == plain(system_table)


def test_system_eval_rejects_a_query_recorded_twice():
    records = system_records()
    with pytest.raises(ValueError, match="duplicate record for query 's01'"):
        system_eval([*records, records[0]])
    assert system_eval([]) == {}


# --------------------------------------------------------------------------- #
# Recommendation                                                               #
# --------------------------------------------------------------------------- #
def est(mean: float, lo: float | None = None, hi: float | None = None, n: int = 100) -> Estimate:
    return Estimate(mean, mean - 0.03 if lo is None else lo, mean + 0.03 if hi is None else hi, n)


def ranker_entry(
    recall: dict[int, float],
    p50: float,
    cost: float,
    *,
    diff_vs: dict[str, dict[int, Estimate]] | None = None,
    gains: dict[tuple[int, int], Estimate] | None = None,
    stage1_misses: int | None = None,
    not_in_top_k: dict[int, int] | None = None,
) -> dict[str, Any]:
    """One ranker's entry in a ranker_comparison table, with only what ``recommend`` reads."""

    return {
        "overall": {
            "recall": {str(k): est(value) for k, value in recall.items()},
            "latency": {"p50": p50},
            "cost_per_1k_usd": cost,
        },
        "recall_diff_vs": {
            other: {"overall": {str(k): diff for k, diff in diffs.items()}}
            for other, diffs in (diff_vs or {}).items()
        },
        "recall_gain": {str(low): {str(high): gain} for (low, high), gain in (gains or {}).items()},
        "failures": {
            "stage1_misses": stage1_misses,
            "gold_not_in_top_k": {str(k): count for k, count in (not_in_top_k or {}).items()},
        },
    }


def system_cell(
    accuracy: float,
    *,
    diff_vs: dict[str, Estimate] | None = None,
    next_k: dict[str, Any] | None = None,
    next_k_prime: dict[str, Any] | None = None,
    failures: dict[str, int] | None = None,
) -> dict[str, Any]:
    """One (ranker, k, k') cell of a system_eval table, with only what ``recommend`` reads."""

    return {
        "overall": {"answer_accuracy": est(accuracy)},
        "accuracy_diff_vs": dict(diff_vs or {}),
        "next_k": next_k,
        "next_k_prime": next_k_prime,
        "failures": dict(failures or {}),
    }


def ranker_tables(
    jev_minus_reranker_recall: Estimate,
    jev_minus_reranker_accuracy: Estimate,
    reranker_p50: float = 0.8,
    reranker_cost: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Three rankers at k = 10, k' = 10. bge_m3_jev is best on both metrics and dearest;
    bge_m3 is significantly behind it; how far bge_m3_reranker is behind is the argument.
    Recall@30, where bge_m3 leads, is there to show that the comparison is made at 10."""

    ranker_table = {
        "bge_m3": ranker_entry({10: 0.70, 30: 0.99}, p50=0.03, cost=0.0),
        "bge_m3_reranker": ranker_entry({10: 0.85, 30: 0.95}, p50=reranker_p50, cost=reranker_cost),
        "bge_m3_jev": ranker_entry(
            {10: 0.87, 30: 0.95},
            p50=1.5,
            cost=0.6,
            diff_vs={"bge_m3": {10: est(0.17, 0.10, 0.24)}, "bge_m3_reranker": {10: jev_minus_reranker_recall}},
        ),
    }
    system_table = {
        "bge_m3": {"10": {"10": system_cell(0.60)}},
        "bge_m3_reranker": {"10": {"10": system_cell(0.72)}},
        "bge_m3_jev": {
            "10": {
                "10": system_cell(
                    0.74, diff_vs={"bge_m3": est(0.14, 0.06, 0.22), "bge_m3_reranker": jev_minus_reranker_accuracy}
                )
            }
        },
    }
    return ranker_table, system_table


TIED = est(0.02, -0.02, 0.06)  # the paired CI includes zero
AHEAD = est(0.05, 0.01, 0.09)  # the paired CI excludes zero


BGE_NUMBERS = "bge_m3 (Recall@10 70.0%, answer accuracy 60.0%, p50 latency 0.030 s, $0.00 per 1k queries)"
RERANKER_NUMBERS = "bge_m3_reranker (Recall@10 85.0%, answer accuracy 72.0%, p50 latency 0.800 s, $0.00 per 1k queries)"
SIGNIFICANTLY = " (significantly: the paired 95% CI of the difference excludes zero)."


def test_recommend_prefers_the_cheaper_ranker_among_statistical_ties():
    choice = recommend(*ranker_tables(TIED, TIED))["ranker"]

    assert choice["choice"] == "bge_m3_reranker"
    assert choice["tied"] == ["bge_m3_reranker", "bge_m3_jev"]
    assert choice["tied_on_recall"] == choice["tied_on_accuracy"] == ["bge_m3_reranker", "bge_m3_jev"]
    assert choice["reference"] == {"recall_k": "10", "k": "10", "k_prime": "10"}
    rationale = choice["rationale"]
    assert rationale.count(".") >= 1 and "\n" not in rationale
    for number in ("Recall@10 85.0%", "answer accuracy 72.0%", "p50 latency 0.800 s", "$0.00 per 1k queries"):
        assert number in rationale
    assert rationale.startswith("Statistically tied on Recall@10 and answer accuracy at k=10, k'=10 with ")
    assert "bge_m3_jev (Recall@10 87.0%, answer accuracy 74.0%, p50 latency 1.500 s, $0.60 per 1k" in rationale
    assert rationale.endswith(
        f"Out of contention: {BGE_NUMBERS}, significantly below the best on Recall@10 and on answer "
        f"accuracy at k=10, k'=10{SIGNIFICANTLY}"
    )


def test_recommend_keeps_the_best_ranker_when_the_others_are_significantly_behind():
    choice = recommend(*ranker_tables(AHEAD, AHEAD))["ranker"]

    assert choice["choice"] == "bge_m3_jev"
    assert choice["tied"] == choice["tied_on_recall"] == choice["tied_on_accuracy"] == ["bge_m3_jev"]
    assert choice["rationale"] == (
        "The only ranker left in contention, with Recall@10 87.0%, answer accuracy 74.0%, p50 latency "
        f"1.500 s, $0.60 per 1k queries. Out of contention: {BGE_NUMBERS}, significantly below the best "
        f"on Recall@10 and on answer accuracy at k=10, k'=10; {RERANKER_NUMBERS}, significantly below "
        f"the best on Recall@10 and on answer accuracy at k=10, k'=10{SIGNIFICANTLY}"
    )


def test_recommend_requires_a_tie_on_both_metrics():
    # Tied on answer accuracy but significantly behind on Recall@10, and the other way round.
    behind_on_recall = recommend(*ranker_tables(AHEAD, TIED))["ranker"]
    assert behind_on_recall["choice"] == "bge_m3_jev"
    assert behind_on_recall["tied"] == ["bge_m3_jev"]
    assert behind_on_recall["tied_on_recall"] == ["bge_m3_jev"]
    assert behind_on_recall["tied_on_accuracy"] == ["bge_m3_reranker", "bge_m3_jev"]
    assert behind_on_recall["rationale"].endswith(
        f"; {RERANKER_NUMBERS}, significantly below the best on Recall@10{SIGNIFICANTLY}"
    )

    behind_on_accuracy = recommend(*ranker_tables(TIED, AHEAD))["ranker"]
    assert behind_on_accuracy["choice"] == "bge_m3_jev"
    assert behind_on_accuracy["tied"] == ["bge_m3_jev"]
    assert behind_on_accuracy["tied_on_recall"] == ["bge_m3_reranker", "bge_m3_jev"]
    assert behind_on_accuracy["tied_on_accuracy"] == ["bge_m3_jev"]
    # The rationale blames the one metric whose paired CI excluded the reranker, not both.
    assert behind_on_accuracy["rationale"].endswith(
        f"; {RERANKER_NUMBERS}, significantly below the best on answer accuracy at k=10, k'=10{SIGNIFICANTLY}"
    )


def test_recommend_breaks_latency_ties_on_cost_and_ranks_unmeasured_latency_last():
    same_latency = recommend(*ranker_tables(TIED, TIED, reranker_p50=1.5, reranker_cost=0.9))["ranker"]
    assert same_latency["choice"] == "bge_m3_jev"  # $0.60 against $0.90 per 1k at the same p50
    slower_but_free = recommend(*ranker_tables(TIED, TIED, reranker_p50=2.0))["ranker"]
    assert slower_but_free["choice"] == "bge_m3_jev"  # latency is compared before cost

    # Every reranker record came from the cache, so it has no latency to compete with.
    unmeasured = recommend(*ranker_tables(TIED, TIED, reranker_p50=math.nan))["ranker"]
    assert unmeasured["choice"] == "bge_m3_jev"
    assert unmeasured["tied"] == ["bge_m3_reranker", "bge_m3_jev"]


def test_recommend_lets_answer_accuracy_decide_when_no_ranker_leads_on_both():
    ranker_table = {
        "bge_m3_reranker": ranker_entry({10: 0.90}, p50=0.8, cost=0.0, diff_vs={"bge_m3_jev": {10: AHEAD}}),
        "bge_m3_jev": ranker_entry({10: 0.85}, p50=1.5, cost=0.6),
    }
    system_table = {
        "bge_m3_reranker": {"10": {"5": system_cell(0.70)}},
        "bge_m3_jev": {"10": {"5": system_cell(0.75, diff_vs={"bge_m3_reranker": AHEAD})}},
    }
    choice = recommend(ranker_table, system_table)["ranker"]
    assert choice["choice"] == "bge_m3_jev"
    assert choice["tied"] == ["bge_m3_jev"]
    assert choice["tied_on_recall"] == ["bge_m3_reranker"]
    assert choice["tied_on_accuracy"] == ["bge_m3_jev"]
    # The reranker leads on Recall@10, so only answer accuracy is held against it.
    assert choice["rationale"].endswith(
        "Out of contention: bge_m3_reranker (Recall@10 90.0%, answer accuracy 70.0%, p50 latency 0.800 s, "
        f"$0.00 per 1k queries), significantly below the best on answer accuracy at k=10, k'=5{SIGNIFICANTLY}"
    )


def test_recommend_falls_back_to_point_estimates_without_paired_differences():
    ranker_table = {
        "bge_m3": ranker_entry({10: 0.70}, p50=0.03, cost=0.0),
        "bge_m3_reranker": ranker_entry({10: 0.71}, p50=0.8, cost=0.0),
    }
    result = recommend(ranker_table, {})
    assert result["ranker"]["choice"] == "bge_m3_reranker"
    assert result["ranker"]["reference"] == {"recall_k": "10", "k": None, "k_prime": None}
    assert result["ranker"]["tied_on_recall"] == ["bge_m3_reranker"]
    assert result["ranker"]["tied_on_accuracy"] == ["bge_m3", "bge_m3_reranker"]  # nobody has a figure
    assert result["ranker"]["rationale"] == (
        "The only ranker left in contention, with Recall@10 71.0%, p50 latency 0.800 s, $0.00 per 1k "
        "queries. Out of contention: bge_m3 (Recall@10 70.0%, p50 latency 0.030 s, $0.00 per 1k queries), "
        "below the best on Recall@10 by point estimate (no paired difference to test it)."
    )


def k_tables(
    recall_gains: dict[tuple[int, int], Estimate],
    accuracy_gains: dict[tuple[int, int], Estimate],
    ks: tuple[int, ...] = (5, 10, 20),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One ranker measured at ``ks`` (k' = 10), with the paired gains between neighbouring ks."""

    ranker_table = {"bge_m3": ranker_entry({k: 0.8 for k in ks}, p50=0.1, cost=0.0, gains=recall_gains)}
    cells = {}
    for low, high in zip(ks, [*ks[1:], None]):
        next_k = None if high is None else {"k": str(high), "answer_accuracy_gain": accuracy_gains.get((low, high))}
        cells[str(low)] = {"10": system_cell(0.7, next_k=next_k)}
    return ranker_table, {"bge_m3": cells}


PAYS = est(0.05, 0.02, 0.08)  # at least 2 points and significant
SMALL = est(0.01, 0.002, 0.018)  # significant, but under 2 points
NOISY = est(0.03, -0.01, 0.07)  # over 2 points, but the CI includes zero


def test_recommend_k_stops_where_neither_recall_nor_accuracy_gains():
    tables = k_tables({(5, 10): PAYS, (10, 20): SMALL}, {(5, 10): NOISY, (10, 20): NOISY})
    result = recommend(*tables)

    assert result["ranker"]["choice"] == "bge_m3"
    assert "only ranker" in result["ranker"]["rationale"]
    choice = result["k"]
    assert choice["choice"] == 10
    assert [(step["from"], step["to"]) for step in choice["steps"]] == [(5, 10), (10, 20)]
    assert choice["steps"][1]["recall_gain"] == SMALL and choice["steps"][1]["answer_accuracy_gain"] == NOISY
    assert choice["rationale"] == (
        "From k=10 to k=20, Recall@k gains +1.0 points (95% CI +0.2 to +1.8) and answer accuracy "
        "gains +3.0 points (95% CI -1.0 to +7.0); neither is a significant gain of at least 2 points."
    )


def test_recommend_k_continues_when_answer_accuracy_alone_gains():
    tables = k_tables({(5, 10): SMALL, (10, 20): SMALL}, {(5, 10): PAYS, (10, 20): NOISY})
    assert recommend(*tables)["k"]["choice"] == 10


def test_recommend_k_is_the_smallest_k_whose_next_step_does_not_pay():
    # The step from 10 to 20 would pay, but the rule stops at the first step that does not.
    tables = k_tables({(5, 10): NOISY, (10, 20): PAYS}, {(5, 10): SMALL, (10, 20): PAYS})
    choice = recommend(*tables)["k"]
    assert choice["choice"] == 5
    assert len(choice["steps"]) == 1


def test_recommend_k_takes_the_largest_k_while_every_step_pays():
    tables = k_tables({(5, 10): PAYS, (10, 20): PAYS}, {})
    choice = recommend(*tables)["k"]
    assert choice["choice"] == 20
    assert "largest value evaluated" in choice["rationale"]
    assert "+5.0 points (95% CI +2.0 to +8.0)" in choice["rationale"]
    assert "an unmeasured amount" in choice["rationale"]  # no accuracy gain was given


def test_recommend_min_gain_sets_the_size_of_a_step_that_pays():
    tables = k_tables({(5, 10): SMALL}, {}, ks=(5, 10))
    assert recommend(*tables)["k"]["choice"] == 5
    assert recommend(*tables, min_gain=0.005)["k"]["choice"] == 10
    exactly_two_points = k_tables({(5, 10): est(0.02, 0.001, 0.04)}, {}, ks=(5, 10))
    assert recommend(*exactly_two_points)["k"]["choice"] == 10  # "under 2 points" excludes 2.0


def test_recommend_with_a_single_k_and_k_prime():
    result = recommend(*k_tables({}, {}, ks=(10,)))
    assert result["k"] == {"choice": 10, "rationale": "k=10 is the only value evaluated.", "steps": []}
    assert result["k_prime"] == {"choice": 10, "rationale": "k'=10 is the only value evaluated.", "steps": []}


def k_prime_table(steps: dict[tuple[int, int], tuple[Estimate, float]]) -> dict[str, Any]:
    """One ranker at k = 10 and k' in 3, 5, 10, with the gain and extra tokens of each step."""

    cells = {}
    for low, high in ((3, 5), (5, 10), (10, None)):
        step = None
        if high is not None:
            gain, tokens = steps[(low, high)]
            step = {"k_prime": str(high), "needed_dialogue_accuracy_gain": gain, "extra_input_tokens": tokens}
        cells[str(low)] = system_cell(0.7, next_k_prime=step)
    return {"bge_m3": {"10": cells}}


def test_recommend_k_prime_stops_where_dialogue_accuracy_stops_gaining():
    system_table = k_prime_table({(3, 5): (est(0.10, 0.02, 0.18, n=40), 3000.0), (5, 10): (NOISY, 6500.0)})
    result = recommend({}, system_table)

    assert result["k"]["choice"] == 10
    choice = result["k_prime"]
    assert choice["choice"] == 5
    assert [(step["from"], step["to"], step["extra_input_tokens"]) for step in choice["steps"]] == [
        (3, 5, 3000.0),
        (5, 10, 6500.0),
    ]
    assert choice["rationale"] == (
        "From k'=5 to k'=10, answer accuracy among the queries that needed dialogue gains +3.0 points "
        "(95% CI -1.0 to +7.0) for +6,500 input tokens per fallback query; not a significant gain of "
        "at least 2 points."
    )


def test_recommend_k_prime_extremes():
    # The first step that does not pay ends the search, whatever a later step would add.
    for later_gain in (PAYS, NOISY):
        first_step_fails = k_prime_table({(3, 5): (SMALL, 3000.0), (5, 10): (later_gain, 6500.0)})
        choice = recommend({}, first_step_fails)["k_prime"]
        assert choice["choice"] == 3
        assert len(choice["steps"]) == 1

    always_pays = k_prime_table({(3, 5): (PAYS, 3000.0), (5, 10): (PAYS, 6500.0)})
    choice = recommend({}, always_pays)["k_prime"]
    assert choice["choice"] == 10
    assert "largest value evaluated" in choice["rationale"] and "+6,500 input tokens" in choice["rationale"]


def test_recommend_failure_modes_at_the_chosen_setting():
    system_failures = {mode: index for index, mode in enumerate(FAILURE_MODES, start=1) if mode != "stage1_misses"}
    ranker_table = {"bge_m3_jev": ranker_entry({10: 0.9}, p50=1.0, cost=0.5, stage1_misses=4, not_in_top_k={10: 99})}
    system_table = {"bge_m3_jev": {"10": {"5": system_cell(0.8, failures=system_failures)}}}

    modes = recommend(ranker_table, system_table)["failure_modes"]

    assert list(modes) == list(FAILURE_MODES)
    assert modes == {**system_failures, "stage1_misses": 4}  # the system eval's own top-k count wins

    # Without a system eval only the ranker comparison's categories are known.
    modes = recommend(ranker_table, {})["failure_modes"]
    assert modes == {**dict.fromkeys(FAILURE_MODES), "stage1_misses": 4, "gold_not_in_top_k": 99}


def test_recommend_without_results():
    result = recommend({}, {})
    assert list(result) == ["ranker", "k", "k_prime", "failure_modes"]
    for key in ("ranker", "k", "k_prime"):
        assert result[key]["choice"] is None
        assert result[key]["rationale"].startswith("No ")
    assert result["failure_modes"] == dict.fromkeys(FAILURE_MODES)


def test_recommend_reads_tables_in_their_json_form():
    tables = [k_tables({(5, 10): PAYS, (10, 20): SMALL}, {(5, 10): NOISY, (10, 20): NOISY}), ranker_tables(TIED, AHEAD)]
    tables.append(({}, k_prime_table({(3, 5): (PAYS, 3000.0), (5, 10): (NOISY, 6500.0)})))
    for ranker_table, system_table in tables:
        from_json = recommend(json.loads(json.dumps(plain(ranker_table))), json.loads(json.dumps(plain(system_table))))
        assert plain(from_json) == plain(recommend(ranker_table, system_table))


def test_recommend_on_the_aggregated_tables(comparison, system_table):
    result = recommend(comparison, system_table)

    # bge_m3_jev was not run at the reference setting (k = 10), so bge_m3 is the only candidate.
    assert result["ranker"]["choice"] == "bge_m3"
    assert result["ranker"]["reference"] == {"recall_k": "3", "k": "10", "k_prime": "3"}
    # No measured gain from k = 5 to 10 at the deepest k' (5); two of three fallback queries
    # gaining from k' = 3 to 5 is not significant.
    assert result["k"]["choice"] == 5
    assert result["k_prime"]["choice"] == 3
    step = result["k_prime"]["steps"][0]
    check(step["needed_dialogue_accuracy_gain"], 2 / 3, 3)
    assert step["needed_dialogue_accuracy_gain"].lo == 0.0
    assert "+3,000 input tokens" in result["k_prime"]["rationale"]
    assert result["failure_modes"] == {**BASE_FAILURES, "stage1_misses": None}
    assert list(result["failure_modes"]) == list(FAILURE_MODES)
    # bge_m3_jev is out for want of a figure, not because its paired difference excluded zero.
    assert result["ranker"]["tied"] == ["bge_m3"]
    assert result["ranker"]["tied_on_recall"] == ["bge_m3", "bge_m3_jev"]
    assert result["ranker"]["tied_on_accuracy"] == ["bge_m3"]
    assert result["ranker"]["rationale"] == (
        "The only ranker left in contention, with Recall@3 72.2%, answer accuracy 55.6%, p50 latency "
        "0.350 s, $0.00 per 1k queries. Out of contention: bge_m3_jev (Recall@3 86.1%, p50 latency "
        "3.500 s, $3.50 per 1k queries), not measured on answer accuracy at k=10, k'=3."
    )
