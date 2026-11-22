"""The report writer: formatting helpers, the Markdown report, and the two result files."""

from __future__ import annotations

import copy
import inspect
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.retrieval import config
from eval.retrieval.interfaces import Usage
from eval.retrieval.metrics import (
    Estimate,
    RankerRecord,
    SystemRecord,
    bootstrap_ci,
    jev_variance,
    paired_bootstrap_diff,
    ranker_comparison,
    recommend,
    system_eval,
)
from eval.retrieval.report import (
    DASH,
    fmt_count,
    fmt_estimate,
    fmt_float,
    fmt_pct,
    fmt_seconds,
    fmt_usd,
    fmt_value,
    md_table,
    render_markdown,
    to_jsonable,
    write_report,
)
from eval.retrieval.schema import TimeRange

NAN = math.nan
RANGE = TimeRange(
    datetime(2026, 10, 26, tzinfo=timezone.utc), datetime(2026, 11, 1, 23, 59, 59, tzinfo=timezone.utc)
)


# --------------------------------------------------------------------------- #
# Formatting helpers                                                           #
# --------------------------------------------------------------------------- #
def test_fmt_pct():
    assert fmt_pct(0.912) == "91.2"
    assert fmt_pct(1.0) == "100.0"
    assert fmt_pct(0.0) == "0.0"
    assert fmt_pct(0.05, signed=True) == "+5.0"
    assert fmt_pct(-0.123, signed=True) == "-12.3"
    assert fmt_pct(-0.0001) == "0.0"  # never "-0.0"
    assert fmt_pct(-0.0001, signed=True) == "+0.0"
    assert fmt_pct(None) == DASH
    assert fmt_pct(NAN) == DASH


def test_fmt_estimate():
    estimate = Estimate(0.912, 0.88, 0.941, 50)
    assert fmt_estimate(estimate) == "91.2 [88.0, 94.1]"
    assert fmt_estimate(estimate, with_n=True) == "91.2 [88.0, 94.1] (n=50)"
    assert fmt_estimate(estimate.to_json()) == "91.2 [88.0, 94.1]"  # as read back from the JSON file
    assert fmt_estimate(Estimate(NAN, NAN, NAN, 0)) == DASH
    assert fmt_estimate(Estimate(NAN, NAN, NAN, 0), with_n=True) == DASH
    assert fmt_estimate({"mean": None, "lo": None, "hi": None, "n": 0}) == DASH
    assert fmt_estimate(None) == DASH
    assert fmt_estimate(0.5) == DASH  # not an estimate


def test_fmt_estimate_marks_a_difference_whose_interval_excludes_zero():
    assert fmt_estimate(Estimate(0.05, 0.01, 0.09, 40), signed=True) == "**+5.0 [+1.0, +9.0]**"
    assert fmt_estimate(Estimate(-0.05, -0.09, -0.01, 40), signed=True) == "**-5.0 [-9.0, -1.0]**"
    assert fmt_estimate(Estimate(0.02, -0.01, 0.05, 40), signed=True) == "+2.0 [-1.0, +5.0]"
    assert fmt_estimate(Estimate(0.02, 0.0, 0.05, 40), signed=True) == "+2.0 [+0.0, +5.0]"  # zero is inside
    assert fmt_estimate(Estimate(0.05, 0.01, 0.09, 40), signed=True, with_n=True) == "**+5.0 [+1.0, +9.0]** (n=40)"


def test_fmt_seconds_picks_the_unit():
    assert fmt_seconds(0.0042) == "4.2 ms"
    assert fmt_seconds(0.0) == "0.0 ms"
    assert fmt_seconds(0.01) == "10 ms"
    assert fmt_seconds(0.042) == "42 ms"
    assert fmt_seconds(0.312) == "312 ms"
    assert fmt_seconds(0.9994) == "999 ms"
    assert fmt_seconds(0.9996) == "1.00 s"
    assert fmt_seconds(2.414) == "2.41 s"
    assert fmt_seconds(None) == DASH
    assert fmt_seconds(NAN) == DASH


def test_fmt_usd():
    assert fmt_usd(0) == "$0"
    assert fmt_usd(1234.5) == "$1,234.50"
    assert fmt_usd(3.5) == "$3.50"
    assert fmt_usd(0.4) == "$0.40"
    assert fmt_usd(0.63) == "$0.63"
    assert fmt_usd(0.042) == "$0.042"  # two significant digits under ten cents
    assert fmt_usd(0.00042) == "$0.00042"
    assert fmt_usd(0.00005) == "<$0.0001"
    assert fmt_usd(None) == DASH
    assert fmt_usd(NAN) == DASH


def test_fmt_count_and_fmt_float():
    assert fmt_count(1234.4) == "1,234"
    assert fmt_count(0) == "0"
    assert fmt_count(None) == DASH
    assert fmt_count(NAN) == DASH
    assert fmt_float(0.24615) == "0.246"
    assert fmt_float(1) == "1.000"
    assert fmt_float(NAN) == DASH
    assert fmt_float(None) == DASH


def test_fmt_value():
    assert fmt_value(True) == "yes"
    assert fmt_value(False) == "no"
    assert fmt_value(20261104) == "20261104"
    assert fmt_value(0.123456) == "0.1235"
    assert fmt_value("openai/gpt-5-nano") == "openai/gpt-5-nano"
    assert fmt_value("two\n  lines") == "two lines"
    assert fmt_value("  ") == DASH
    assert fmt_value({"b": 1, "a": "x"}) == "a=x, b=1"
    assert fmt_value(["mps", 2]) == "mps, 2"
    assert fmt_value(Estimate(0.9, 0.8, 1.0, 60)) == "90.0 [80.0, 100.0] (n=60)"
    for nothing in (None, NAN, {}, []):
        assert fmt_value(nothing) == DASH


def test_md_table():
    table = md_table(["Ranker", "Pool", "Recall"], [["bge_m3", "all", "91.2"], ["jev", ">100", 3]], left=2)
    assert table.splitlines() == [
        "| Ranker | Pool | Recall |",
        "| :--- | :--- | ---: |",
        "| bge_m3 | all | 91.2 |",
        "| jev | >100 | 3 |",
    ]
    assert md_table(["A"], []).splitlines() == ["| A |", "| :--- |"]


def test_md_table_prints_a_dash_for_missing_cells_and_keeps_the_grid_intact():
    table = md_table(["Name", "Value"], [["a", None], ["b", NAN], ["c", ""], ["d", 0], ["x|y", "two\nlines"]])
    assert table.splitlines()[2:] == [
        "| a | - |",
        "| b | - |",
        "| c | - |",
        "| d | 0 |",
        "| x\\|y | two lines |",
    ]
    with pytest.raises(ValueError, match="row has 1 cells for 2 columns"):
        md_table(["Name", "Value"], [["only one"]])


def test_to_jsonable():
    value = {
        "estimate": Estimate(0.5, NAN, 0.75, 4),
        "nan": NAN,
        "infinite": math.inf,
        "numpy": [np.float64(0.25), np.int64(3), np.float64("nan")],
        "tuple": (1, "two", None, True),
        "when": datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc),
        "where": Path("eval/retrieval/results"),
        "range": RANGE,
        "usage": Usage(10, 2, 0.5, 1.5, 1),
        5: {10: "int keys become strings"},
    }
    plain = to_jsonable(value)
    assert plain == {
        "estimate": {"mean": 0.5, "lo": None, "hi": 0.75, "n": 4},
        "nan": None,
        "infinite": None,
        "numpy": [0.25, 3, None],
        "tuple": [1, "two", None, True],
        "when": "2026-11-04T18:00:00Z",
        "where": "eval/retrieval/results",
        "range": {"start": "2026-10-26T00:00:00Z", "end": "2026-11-01T23:59:59Z"},
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.5, "latency_s": 1.5, "calls": 1},
        "5": {"10": "int keys become strings"},
    }
    assert type(plain["numpy"][1]) is int and plain["tuple"][3] is True
    json.dumps(plain, allow_nan=False)  # nothing left that JSON cannot spell
    with pytest.raises(TypeError, match="cannot write set"):
        to_jsonable({"ids": {"a", "b"}})


# --------------------------------------------------------------------------- #
# A small full run                                                             #
# --------------------------------------------------------------------------- #
RANKED_QUERIES: dict[str, tuple[str, int, list[str]]] = {  # query id -> (type, pool size, gold ids)
    "q1": ("lookup", 10, ["g1"]),
    "q2": ("multi_conversation", 150, ["g2a", "g2b"]),
    "q3": ("temporal", 50, ["g3"]),
    "q4": ("cross_person", 800, ["g4"]),
}


def ranker_records() -> list[RankerRecord]:
    """bge_m3 (4 ms, free) finds the gold of q1 and q3 only; bge_m3_jev (1.5 s, $0.0005 a
    query) finds it everywhere in both of its runs, leaves three candidates of q2 unscored
    and had q3 answered from the cache."""

    records = []
    for ranker, runs in (("bge_m3", 1), ("bge_m3_jev", 2)):
        jev = ranker == "bge_m3_jev"
        for run in range(runs):
            for index, (query_id, (query_type, pool_size, gold)) in enumerate(RANKED_QUERIES.items()):
                ranked_ids = [*gold, "x1", "x2"] if jev or index % 2 == 0 else ["x1", "x2", "x3"]
                extra = {
                    "n_requests": 2,
                    "n_missing": 3 if query_id == "q2" else 0,
                    "n_cached": 1 if query_id == "q3" else 0,
                    "jev_model": "jev-test",
                    "scoring": "pointwise",
                }
                records.append(
                    RankerRecord(
                        query_id=query_id,
                        type=query_type,
                        pool_size=pool_size,
                        gold_ids=gold,
                        ranker=ranker,
                        run=run,
                        ranked_ids=ranked_ids,
                        scores=[0.9 - 0.1 * position - 0.01 * run for position in range(len(ranked_ids))],
                        stage1_ids=[*gold, "x1", "x2", "x3"] if jev else None,
                        latency_s=1.5 if jev else 0.004,
                        stage_latency_s={"vector_search": 0.004, "rerank": 1.4} if jev else {"vector_search": 0.004},
                        usage=Usage(cost_usd=0.0005 if jev else 0.0),
                        extra=extra if jev else {},
                    )
                )
    return records


SYSTEM_QUERIES: dict[str, tuple[str, int | None, list[str], str]] = {  # id -> (type, pool, gold, path)
    "s1": ("lookup", 10, ["g1"], "summary"),
    "s2": ("multi_conversation", 150, ["g2a", "g2b"], "fallback"),
    "s3": ("temporal", 50, ["g3"], "declined"),
    "s4": ("unanswerable", 20, [], "declined"),
    "s5": ("unknown_name", None, [], "unknown_name"),
}


def system_records() -> list[SystemRecord]:
    """Both rankers at k in 5, 10 and k' in 3, 5. s1 is answered from summaries (1,100 tokens
    in, 1.5 s, $0.011); s2, s3 and s4 need dialogue (5,100 tokens in, 3.5 s, $0.051) and only
    bge_m3_jev answers s2 correctly; s5 stops at the unknown name."""

    records = []
    for ranker in ("bge_m3", "bge_m3_jev"):
        for k in (5, 10):
            for k_prime in (3, 5):
                for query_id, (query_type, pool_size, gold, path) in SYSTEM_QUERIES.items():
                    needed_dialogue = path in ("fallback", "declined")
                    usage = {"extraction": Usage(100, 10, 0.001, 0.5, 1)}
                    latency = {"extraction": 0.5}
                    if path != "unknown_name":
                        usage["answer_summary"] = Usage(1000, 50, 0.01, 1.0, 1)
                        latency["answer_summary"] = 1.0
                    if needed_dialogue:
                        usage["answer_dialogue"] = Usage(4000, 100, 0.04, 2.0, 1)
                        latency["answer_dialogue"] = 2.0
                    # The runner judges every answerable query's final answer, the refusal of s3 included.
                    verdict = {"s1": "correct", "s2": "correct" if ranker == "bge_m3_jev" else "incorrect", "s3": "incorrect"}
                    records.append(
                        SystemRecord(
                            query_id=query_id,
                            type=query_type,
                            pool_size=pool_size,
                            ranker=ranker,
                            k=k,
                            k_prime=k_prime,
                            path=path,
                            needed_dialogue=needed_dialogue,
                            final_answer="an answer",
                            verdict=verdict.get(query_id),
                            gold_ids=gold,
                            retrieved_ids=gold,
                            gold_name=None,
                            extracted_name=None,
                            name_correct=True,
                            person_correct=True,
                            extracted_time_range=RANGE if query_type == "temporal" else None,
                            time_filter_ok=True if query_type == "temporal" else None,
                            unknown_name_exact=True if query_type == "unknown_name" else None,
                            usage_by_stage=usage,
                            latency_by_stage=latency,
                        )
                    )
    return records


def build_results() -> dict[str, Any]:
    """What the runner hands to the report: every section, built by the real aggregators."""

    ranker_table = ranker_comparison(ranker_records())
    system_table = system_eval(system_records())
    return {
        "meta": {
            "date": "2026-11-06",
            "seed": config.SEED,
            "split": "test",
            "store": "memory",
            "counts": {"people": 40, "conversations": 800, "queries": {"lookup": 45, "temporal": 36}},
            "models": {"embedding": config.EMBEDDING_MODEL, "reranker": config.RERANKER_MODEL},
            "jev": {"model": "jev-test", "scoring": {"mode": "pointwise", "concurrency": 8}},
            "hardware": {"device": "mps", "machine": "arm64"},
            "label_pool": {"depth": 30},
        },
        "ranker_comparison": ranker_table,
        "jev_variance": jev_variance(ranker_records()),
        "system_eval": system_table,
        "judge_validation": {  # the summary judge_validation.agreement returns
            "n": 60,
            "agreement_rate": 0.95,
            "disagreements": [
                {
                    "key": "bge_m3|5|3|s2",
                    "query": "What did Sam say about the move?",
                    "gold_answer": "He is moving to Leeds in March.",
                    "system_answer": "Sam is moving.\nNo date was given | no city either.",
                    "judge_verdict": "incorrect",
                    "human_label": "partial",
                }
            ],
        },
        "recommendation": recommend(ranker_table, system_table),
        "notes": "Synthetic numbers.\n",
    }


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    return build_results()


@pytest.fixture(scope="module")
def markdown(results) -> str:
    return render_markdown(results)


def block(markdown: str, heading: str) -> list[str]:
    """The non-empty lines under ``heading``, up to the next heading of the same or a higher level."""

    lines = markdown.splitlines()
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        hashes = len(line) - len(line.lstrip("#"))
        if 0 < hashes <= level and line[hashes : hashes + 1] == " ":
            break
        if line:
            body.append(line)
    return body


def table_rows(markdown: str, heading: str) -> list[str]:
    """The data rows (no header, no alignment row) of the table under ``heading``."""

    return [line for line in block(markdown, heading) if line.startswith("|")][2:]


def assert_well_formed(markdown: str) -> None:
    """Every table is a full grid with no empty cell, and no raw missing value leaks into the text."""

    assert not re.search(r"\b(nan|NaN|None|inf)\b", markdown)
    width = None
    for line in markdown.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
        assert all(cells), f"empty cell in {line!r}"
        assert width in (None, len(cells)), f"ragged table row {line!r}"
        width = len(cells)


def test_full_report_has_every_section_in_order(markdown):
    assert_well_formed(markdown)
    assert markdown.endswith("\n") and not markdown.endswith("\n\n")
    assert [line for line in markdown.splitlines() if line.startswith("# ")] == [
        "# Retrieval ranker benchmark (2026-11-06, test split)"
    ]
    assert [line for line in markdown.splitlines() if line.startswith("## ")] == [
        "## Summary",
        "## Setup",
        "## Ranker comparison",
        "## Jev run-to-run variance",
        "## System eval",
        "## Judge validation",
        "## Recommendation",
        "## Notes",
    ]
    assert [line for line in markdown.splitlines() if line.startswith("### ")] == [
        "### Dataset",
        "### Models",
        "### Hardware (local models)",
        "### Jev",  # a section the run does not write: titled from its key, after the known ones
        "### Label pool",
        "### What was measured",
        "### Recall@k",
        "### All-found@k",
        "### Stage 1 Recall@100",
        "### Latency and cost by pool size",
        "### Recall@k by query type",
        "### Recall@k by pool size",
        "### Paired differences in Recall@k",
        "### bge_m3_jev",
        "### Answer accuracy by ranker, k and k'",
        "### Fallback rate and accuracy by path",
        "### Decline rates",
        "### Extraction and unknown-name handling",
        "### Tokens, latency and cost by path",
        "### Slices at the focus setting",
        "### Answer accuracy by query type, per ranker",
        "### Answer accuracy by pool size, per ranker",
        "### Disagreements",
        "### Evidence for the ranker",
        "### Evidence for k",
        "### Evidence for k'",
        "### Failure modes",
    ]


def test_summary_is_the_recommendation_in_three_lines(results, markdown):
    recommendation = results["recommendation"]
    # The rankers are statistically tied on four queries, so the faster, free one is drafted.
    assert (recommendation["ranker"]["choice"], recommendation["k"]["choice"]) == ("bge_m3", 5)
    assert recommendation["k_prime"]["choice"] == 3
    assert block(markdown, "## Summary") == [
        f"- **Ranker:** `bge_m3`. {recommendation['ranker']['rationale']}",
        f"- **k:** `5`. {recommendation['k']['rationale']}",
        f"- **k':** `3`. {recommendation['k_prime']['rationale']}",
    ]
    assert markdown.index("## Summary") < markdown.index("## Setup")
    assert block(markdown, "## Recommendation")[:3] == block(markdown, "## Summary")


def test_setup_lists_the_run_and_how_it_was_measured(markdown):
    setup = block(markdown, "## Setup")
    assert setup[:4] == [
        "- **date:** 2026-11-06",
        "- **split:** test",
        f"- **seed:** {config.SEED}",
        "- **store:** memory",
    ]
    assert block(markdown, "### Dataset") == [
        "- **conversations:** 800",
        "- **people:** 40",
        "- **queries:**",
        "  - **lookup:** 45",
        "  - **temporal:** 36",
    ]
    assert "- **embedding:** BAAI/bge-m3" in block(markdown, "### Models")
    assert block(markdown, "### Jev") == [
        "- **model:** jev-test",
        "- **scoring:**",
        "  - **concurrency:** 8",
        "  - **mode:** pointwise",
    ]
    assert block(markdown, "### Hardware (local models)") == ["- **device:** mps", "- **machine:** arm64"]
    measured = block(markdown, "### What was measured")
    assert [line.split(".**")[0] for line in measured] == [
        "- **Ranker comparison",
        "- **System eval",
        "- **Statistics",
        "- **Slices",
    ]
    assert f"percentile bootstrap (5,000 resamples of queries, seed {config.SEED})" in measured[2]


def test_ranker_comparison_tables(markdown):
    perfect = "100.0 [100.0, 100.0]"
    half = fmt_estimate(bootstrap_ci([1, 0, 1, 0]))  # bge_m3: q1 and q3 found, q2 and q4 missed
    assert half.startswith("50.0 [")
    assert table_rows(markdown, "### Recall@k") == [
        f"| bge_m3 | 4 | {half} | {half} | {half} | {half} | - |",
        f"| bge_m3_jev | 4 | {perfect} | {perfect} | {perfect} | {perfect} | 3 |",
    ]
    assert [line for line in block(markdown, "### Recall@k") if line.startswith("| Ranker")] == [
        "| Ranker | n | @5 | @10 | @20 | @30 | Unscored |"
    ]

    missed = "0.0 [0.0, 0.0]"  # q2 is the only query with two gold conversations
    assert table_rows(markdown, "### All-found@k") == [
        f"| bge_m3 | 2+ gold conversations | 1 | {missed} | {missed} | {missed} | {missed} |",
        f"| bge_m3 | multi_conversation type | 1 | {missed} | {missed} | {missed} | {missed} |",
        f"| bge_m3_jev | 2+ gold conversations | 1 | {perfect} | {perfect} | {perfect} | {perfect} |",
        f"| bge_m3_jev | multi_conversation type | 1 | {perfect} | {perfect} | {perfect} | {perfect} |",
    ]
    # Only the ranker with a first stage is listed: q2 and q4 have pools over 100.
    assert table_rows(markdown, "### Stage 1 Recall@100") == [
        f"| bge_m3_jev | {perfect} (n=2) | {perfect} (n=4) |"
    ]

    by_type = block(markdown, "### Recall@k by query type")
    assert by_type[0] == (
        "| Ranker | k | lookup (n=1) | multi_conversation (n=1) | cross_person (n=1) | temporal (n=1) |"
    )
    assert "| bge_m3 | 5 | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |" in by_type
    by_pool = block(markdown, "### Recall@k by pool size")
    assert by_pool[0] == "| Ranker | k | <=20 (n=1) | 21-100 (n=1) | >100 (n=2) |"
    assert "| bge_m3 | 30 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] |" in by_pool


def test_latency_table_says_how_many_records_were_timed(markdown):
    rows = table_rows(markdown, "### Latency and cost by pool size")
    assert rows == [
        "| bge_m3 | all | 4 | 4 | 4.0 ms | 4.0 ms | $0 | vector_search 4.0 ms |",
        "| bge_m3 | <=20 | 1 | 1 | 4.0 ms | 4.0 ms | $0 | vector_search 4.0 ms |",
        "| bge_m3 | 21-100 | 1 | 1 | 4.0 ms | 4.0 ms | $0 | vector_search 4.0 ms |",
        "| bge_m3 | >100 | 2 | 2 | 4.0 ms | 4.0 ms | $0 | vector_search 4.0 ms |",
        # q3, the only query with a pool of 21-100, came from the cache: nothing was timed there.
        "| bge_m3_jev | all | 4 | 3 | 1.50 s | 1.50 s | $0.50 | rerank 1.40 s, vector_search 4.0 ms |",
        "| bge_m3_jev | <=20 | 1 | 1 | 1.50 s | 1.50 s | $0.50 | rerank 1.40 s, vector_search 4.0 ms |",
        "| bge_m3_jev | 21-100 | 1 | 0 | - | - | $0.50 | - |",
        "| bge_m3_jev | >100 | 2 | 2 | 1.50 s | 1.50 s | $0.50 | rerank 1.40 s, vector_search 4.0 ms |",
    ]
    text = " ".join(block(markdown, "### Latency and cost by pool size"))
    assert "| Ranker | Pool | Queries | Timed | p50 | p95 | $ / 1k queries | Stage p50 |" in text
    assert "served from the cache" in text and "is left out" in text


def test_paired_difference_table(markdown):
    rows = table_rows(markdown, "### Paired differences in Recall@k")
    overall = fmt_estimate(paired_bootstrap_diff([1, 1, 1, 1], [1, 0, 1, 0]), signed=True)
    assert overall.startswith("+50.0 [") and "**" not in overall  # four queries cannot exclude zero
    assert rows[0] == f"| all | bge_m3_jev - bge_m3 | 4 | {overall} | {overall} | {overall} | {overall} |"
    # Both queries with a pool over 100 are found by one ranker only, so that slice is significant.
    assert rows[-1] == "| >100 | bge_m3_jev - bge_m3 | 2 | " + " | ".join(["**+100.0 [+100.0, +100.0]**"] * 4) + " |"
    assert [row.split(" | ")[0] for row in rows] == ["| all", "| <=20", "| 21-100", "| >100"]


def test_variance_section(markdown):
    assert block(markdown, "## Jev run-to-run variance") == [
        "### bge_m3_jev",
        "2 runs over the same 4 queries. Recall@k in percent; the standard deviation is across runs, in points.",
        "| Recall | run 0 | run 1 | mean | std |",
        "| :--- | ---: | ---: | ---: | ---: |",
        "| @5 | 100.0 | 100.0 | 100.0 | 0.0 |",
        "| @10 | 100.0 | 100.0 | 100.0 | 0.0 |",
        "| @20 | 100.0 | 100.0 | 100.0 | 0.0 |",
        "| @30 | 100.0 | 100.0 | 100.0 | 0.0 |",
        "- Mean absolute score difference per (query, conversation) between runs: 0.010",
        "- Pairs left out of that difference because a run had no usable score: 0",
        "- Mean Jaccard overlap of the top-10 sets between runs: 1.000",
        "- Jev snapshot(s) served: run 0: jev-test; run 1: jev-test",
    ]


def test_variance_section_flags_runs_that_hit_different_snapshots(results):
    variance = copy.deepcopy(results["jev_variance"])
    variance["bge_m3_jev"]["snapshots_per_run"] = [["jev-test"], ["jev-test", "jev-test-2"]]
    variance["bge_m3_jev"]["n_unscored_pairs"] = 3
    rendered = render_markdown({"jev_variance": variance})
    assert block(rendered, "### bge_m3_jev")[-2:] == [
        "- Mean Jaccard overlap of the top-10 sets between runs: 1.000",
        "- Jev snapshot(s) served: run 0: jev-test; run 1: jev-test, jev-test-2 (the runs hit different "
        "snapshots, so the drift above includes a model change)",
    ]
    assert "- Pairs left out of that difference because a run had no usable score: 3" in rendered


def test_setup_warns_when_several_jev_snapshots_served_the_run(results):
    meta = copy.deepcopy(results["meta"])
    meta["models"]["jev_snapshots"] = ["jev-test-20261014"]
    assert block(render_markdown({"meta": meta}), "### Models") == [
        "- **embedding:** BAAI/bge-m3",
        "- **jev_snapshots:** jev-test-20261014",
        f"- **reranker:** {config.RERANKER_MODEL}",
    ]
    meta["models"]["jev_snapshots"] = ["jev-test-20261014", "jev-test-20261101"]
    assert block(render_markdown({"meta": meta}), "### Models") == [
        "- **embedding:** BAAI/bge-m3",
        "- **jev_snapshots:** jev-test-20261014, jev-test-20261101",
        f"- **reranker:** {config.RERANKER_MODEL}",
        "- **Warning:** 2 Jev snapshots served this run, so the Jev figures mix model versions.",
    ]


def test_system_eval_tables(markdown):
    third = fmt_estimate(bootstrap_ci([1, 0, 0]))  # bge_m3: s1 of the answerable s1, s2, s3
    two_thirds = fmt_estimate(bootstrap_ci([1, 1, 0]))
    assert third.startswith("33.3 [") and two_thirds.startswith("66.7 [")
    assert table_rows(markdown, "### Answer accuracy by ranker, k and k'") == [
        f"| bge_m3 | 5 | 3 | {third} | {third} |",
        f"| bge_m3 | 10 | 3 | {third} | {third} |",
        f"| bge_m3_jev | 5 | 3 | {two_thirds} | {two_thirds} |",
        f"| bge_m3_jev | 10 | 3 | {two_thirds} | {two_thirds} |",
    ]
    assert block(markdown, "### Answer accuracy by ranker, k and k'")[1] == "| Ranker | k | n | k'=3 | k'=5 |"

    paths = table_rows(markdown, "### Fallback rate and accuracy by path")
    assert len(paths) == 8  # 2 rankers x 2 k x 2 k'
    # bge_m3: no partial answers; s2 and s3 needed dialogue; the summary path (s1) is right,
    # the fallback path (s2) is wrong, and so are both queries that needed dialogue.
    assert paths[0] == (
        f"| bge_m3 | 5 | 3 | 0.0 [0.0, 0.0] | {two_thirds} | 100.0 [100.0, 100.0] (n=1) "
        "| 0.0 [0.0, 0.0] (n=1) | 0.0 [0.0, 0.0] (n=2) |"
    )

    declines = table_rows(markdown, "### Decline rates")
    # s4 is declined correctly; s3 is a false decline with its gold retrieved; nothing is misrouted.
    assert declines[0] == (
        f"| bge_m3 | 5 | 3 | 100.0 [100.0, 100.0] (n=1) | {third} | {third} (n=3) | 0.0 [0.0, 0.0] |"
    )

    assert block(markdown, "### Extraction and unknown-name handling") == [
        "Measured at bge_m3, k=5, k'=3. Extraction and name resolution run before ranking, so these do "
        "not depend on the ranker, k or k'.",
        "| Metric | % [95% CI] | n |",
        "| :--- | ---: | ---: |",
        "| Name extraction accuracy (all queries) | 100.0 [100.0, 100.0] | 5 |",
        "| Person resolution accuracy (all queries) | 100.0 [100.0, 100.0] | 5 |",
        "| Time range extracted (temporal queries) | 100.0 [100.0, 100.0] | 1 |",
        "| Time-filter recall (temporal queries) | 100.0 [100.0, 100.0] | 1 |",
        "| Spurious time range (queries without a gold range) | 0.0 [0.0, 0.0] | 4 |",
        "| Exact unknown-name message (unknown_name queries) | 100.0 [100.0, 100.0] | 1 |",
    ]


def test_cost_table_splits_by_path(markdown):
    rows = table_rows(markdown, "### Tokens, latency and cost by path")
    assert len(rows) == 24  # 8 settings x (all, summary, fallback)
    assert rows[:3] == [
        # (100 + 1,100 + 3 x 5,100) / 5 tokens in; latencies 0.5, 1.5, 3.5, 3.5, 3.5 s.
        "| bge_m3 | 5 | 3 | all | 5 | 3,300 | 110 | 5 | 3.50 s | 3.50 s | $33.00 |",
        "| bge_m3 | 5 | 3 | summary | 1 | 1,100 | 60 | 1 | 1.50 s | 1.50 s | $11.00 |",
        "| bge_m3 | 5 | 3 | fallback | 3 | 5,100 | 160 | 3 | 3.50 s | 3.50 s | $51.00 |",
    ]
    assert block(markdown, "### Tokens, latency and cost by path")[1] == (
        "| Ranker | k | k' | Path | n | Tokens in | Tokens out | n timed | p50 | p95 | $ / 1k queries |"
    )


def test_system_slices_at_the_focus_setting(markdown):
    slices = block(markdown, "### Slices at the focus setting")
    assert slices[0] == "bge_m3, k=5, k'=3, by query type and by pool size."
    rows = table_rows(markdown, "### Slices at the focus setting")
    assert [row.split(" | ")[0] for row in rows] == [
        "| all",
        "| lookup",
        "| multi_conversation",
        "| temporal",
        "| unanswerable",
        "| unknown_name",
        "| <=20",
        "| 21-100",
        "| >100",
    ]
    # The unanswerable query: nothing answerable to grade, declined correctly, no time range.
    assert rows[4] == "| unanswerable | 1 | - | - | - | - | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | - |"
    # The unknown name has no pool, so the pool buckets hold the other four queries.
    assert [row.split(" | ")[1] for row in rows[-3:]] == ["2", "1", "1"]

    by_type = table_rows(markdown, "### Answer accuracy by query type, per ranker")
    assert by_type == [
        "| bge_m3 | 100.0 [100.0, 100.0] | - | - | 0.0 [0.0, 0.0] | - | 0.0 [0.0, 0.0] |",
        "| bge_m3_jev | 100.0 [100.0, 100.0] | - | - | 100.0 [100.0, 100.0] | - | 0.0 [0.0, 0.0] |",
    ]
    by_pool = block(markdown, "### Answer accuracy by pool size, per ranker")
    assert by_pool[0] == "At k=5, k'=3."
    assert by_pool[-1] == "| bge_m3_jev | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |"


def test_detail_tables_follow_the_recommended_setting(results):
    def focused_on(ranker: str, k: int, k_prime: int) -> str:
        recommendation = copy.deepcopy(results["recommendation"])
        for key, choice in (("ranker", ranker), ("k", k), ("k_prime", k_prime)):
            recommendation[key]["choice"] = choice
        return render_markdown({**results, "recommendation": recommendation})

    rendered = focused_on("bge_m3_jev", 10, 5)
    extraction = block(rendered, "### Extraction and unknown-name handling")
    assert extraction[0].startswith("Measured at bge_m3_jev, k=10, k'=5.")
    slices = block(rendered, "### Slices at the focus setting")
    assert slices[0] == "bge_m3_jev, k=10, k'=5, by query type and by pool size."
    assert slices[3].startswith("| all | 5 | 66.7 [")
    assert block(rendered, "### Answer accuracy by pool size, per ranker")[0] == "At k=10, k'=5."

    # A recommendation for a setting that was not measured falls back to the first one that was.
    rendered = focused_on("jev_full_pool", 20, 10)
    slices = block(rendered, "### Slices at the focus setting")
    assert slices[0] == "bge_m3, k=5, k'=3, by query type and by pool size."


def test_judge_validation_and_notes(results, markdown):
    assert block(markdown, "## Judge validation") == [
        "The judge's verdict matched the human label on 95.0% of the 60 answers checked.",
        "### Disagreements",
        "| Query | Gold answer | System answer | Judge | Human |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| What did Sam say about the move? | He is moving to Leeds in March. "
        "| Sam is moving. No date was given \\| no city either. | incorrect | partial |",
    ]
    assert block(markdown, "## Notes") == ["Synthetic numbers."]

    # Any other shape is listed as it is, and full agreement needs no table.
    validation = {"labelled_by": "two\nannotators", "kappa": 0.8123456, "rounds": {"first": 40}}
    other = render_markdown({"judge_validation": validation})
    assert block(other, "## Judge validation") == [
        "- **kappa:** 0.8123",
        "- **labelled_by:** two annotators",
        "- **rounds:**",
        "  - **first:** 40",
    ]
    agreed = render_markdown({"judge_validation": {"n": 12, "agreement_rate": 1.0, "disagreements": []}})
    assert block(agreed, "## Judge validation") == [
        "The judge's verdict matched the human label on 100.0% of the 12 answers checked."
    ]
    assert block(render_markdown({"judge_validation": "not run on the dev split"}), "## Judge validation") == [
        "not run on the dev split"
    ]

    without = {key: value for key, value in results.items() if key not in ("judge_validation", "notes")}
    rendered = render_markdown(without)
    assert "## Judge validation" not in rendered and "## Notes" not in rendered
    assert rendered.rstrip().endswith("| Answer not graded (the judge gave no usable verdict) | 0 |")


def test_recommendation_section(results, markdown):
    recommendation = results["recommendation"]
    half = fmt_estimate(bootstrap_ci([1, 0, 1, 0]))
    assert table_rows(markdown, "### Evidence for the ranker") == [
        f"| bge_m3 | {half} | {fmt_estimate(bootstrap_ci([1, 0, 0]))} | 4.0 ms | $0 | yes | yes |",
        f"| bge_m3_jev | 100.0 [100.0, 100.0] | {fmt_estimate(bootstrap_ci([1, 1, 0]))} | 1.50 s | $0.50 | yes | yes |",
    ]
    assert block(markdown, "### Evidence for the ranker")[1] == (
        "| Ranker | Recall@10 | Answer accuracy (k=10, k'=5) | p50 latency | $ / 1k queries "
        "| Tied on Recall@10 | Tied on accuracy |"
    )
    # Going from k = 5 to 10, or from k' = 3 to 5, changes no answer in this run.
    assert table_rows(markdown, "### Evidence for k") == ["| 5 to 10 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |"]
    assert table_rows(markdown, "### Evidence for k'") == ["| 3 to 5 | +0.0 [+0.0, +0.0] (n=2) | 0 |"]
    assert recommendation["failure_modes"]["false_declines_gold_retrieved"] == 1
    assert table_rows(markdown, "### Failure modes") == [
        "| Extraction returned the wrong name | 0 |",
        "| No time range extracted for a temporal query | 0 |",
        "| Padded time filter dropped a gold conversation | 0 |",
        "| Time range extracted where the query has none | 0 |",
        "| Stopped before ranking (unknown or ambiguous name) | 0 |",
        "| Filters left nothing to rank | 0 |",
        "| Gold conversation missing from the bge-m3 top 100 (gold filters) | - |",  # bge_m3 has no stage 1
        "| Gold conversation missing from the top k the answerer saw | 0 |",
        "| Declined although every gold conversation was retrieved | 1 |",
        "| Wrong answer although every gold conversation was retrieved | 1 |",
        "| Wrong answer given from summaries alone | 0 |",
        "| Answer not graded (the judge gave no usable verdict) | 0 |",
    ]


def test_ranker_evidence_shows_which_metric_excluded_a_ranker(results):
    # The reranker is tied on Recall@10 but significantly behind on answer accuracy.
    choice = copy.deepcopy(results["recommendation"]["ranker"])
    choice.update(
        {
            "choice": "bge_m3_jev",
            "tied": ["bge_m3_jev"],
            "tied_on_recall": ["bge_m3", "bge_m3_jev"],
            "tied_on_accuracy": ["bge_m3_jev"],
        }
    )
    rendered = render_markdown({**results, "recommendation": {**results["recommendation"], "ranker": choice}})
    rows = table_rows(rendered, "### Evidence for the ranker")
    assert [row.split(" | ")[0] for row in rows] == ["| bge_m3", "| bge_m3_jev"]
    assert [row.split(" | ")[-2:] for row in rows] == [["yes", "no |"], ["yes", "yes |"]]
    intro = block(rendered, "### Evidence for the ranker")[0]
    assert intro.startswith("A ranker is tied on a metric while it is not significantly below the best on it")
    assert "if there is none, those tied on answer accuracy are" in intro


def test_ranker_evidence_reads_the_reference_cells():
    recall = {"10": Estimate(0.8, 0.7, 0.9, 50), "20": Estimate(0.95, 0.9, 1.0, 50)}
    ranker_table = {"bge_m3": {"overall": {"recall": recall, "latency": {"p50": 0.02}, "cost_per_1k_usd": 0.0}}}
    system_table = {"bge_m3": {"20": {"5": {"overall": {"answer_accuracy": Estimate(0.6, 0.5, 0.7, 50)}}}}}
    choice = {
        "choice": "bge_m3",
        "rationale": "The only ranker with results.",
        "tied": [],
        "reference": {"recall_k": "10", "k": "20", "k_prime": "5"},
    }

    rendered = render_markdown(
        {"ranker_comparison": ranker_table, "system_eval": system_table, "recommendation": {"ranker": choice}}
    )

    assert_well_formed(rendered)  # the tables around it hold little more than dashes
    assert block(rendered, "### Evidence for the ranker")[1:] == [
        "| Ranker | Recall@10 | Answer accuracy (k=20, k'=5) | p50 latency | $ / 1k queries "
        "| Tied on Recall@10 | Tied on accuracy |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| bge_m3 | 80.0 [70.0, 90.0] | 60.0 [50.0, 70.0] | 20 ms | $0 | no | no |",  # no tied sets recorded
    ]
    assert block(rendered, "## Summary") == [
        "- **Ranker:** `bge_m3`. The only ranker with results.",
        "- **k:** -. No rationale was recorded.",
        "- **k':** -. No rationale was recorded.",
    ]


def test_render_markdown_leaves_the_results_untouched(results):
    before = to_jsonable(results)
    render_markdown(results)
    assert to_jsonable(results) == before


# --------------------------------------------------------------------------- #
# Partial results                                                              #
# --------------------------------------------------------------------------- #
def test_empty_results_still_render():
    rendered = render_markdown({})
    assert_well_formed(rendered)
    assert rendered.splitlines()[0] == "# Retrieval ranker benchmark"
    assert [line for line in rendered.splitlines() if line.startswith("## ")] == [
        "## Summary",
        "## Setup",
        "## Ranker comparison",
        "## Jev run-to-run variance",
        "## System eval",
        "## Recommendation",
    ]
    assert block(rendered, "## Summary") == ["_No recommendation was drafted for this run._"]
    for heading in ("## Ranker comparison", "## Jev run-to-run variance", "## System eval", "## Recommendation"):
        assert block(rendered, heading) == ["_Not run._"]
    assert block(rendered, "## Setup")[0] == "### What was measured"


@pytest.mark.parametrize("section", ["ranker_comparison", "jev_variance", "system_eval", "recommendation", "meta"])
def test_report_renders_with_one_section_missing(results, markdown, section):
    partial = {key: value for key, value in results.items() if key != section}
    rendered = render_markdown(partial)
    assert_well_formed(rendered)
    kept = {"ranker_comparison": "### Recall@k", "system_eval": "### Decline rates", "jev_variance": "### bge_m3_jev"}
    for other, heading in kept.items():
        if other != section:
            assert block(rendered, heading) == block(markdown, heading)
    if section in kept:
        assert kept[section] not in rendered
        assert "_Not run._" in rendered


def test_report_renders_with_one_section_alone(results):
    for section in ("ranker_comparison", "jev_variance", "system_eval", "recommendation", "judge_validation"):
        rendered = render_markdown({section: results[section]})
        assert_well_formed(rendered)
        assert rendered.count("_Not run._") >= 2


def test_recommendation_without_results_prints_dashes():
    rendered = render_markdown({"recommendation": recommend({}, {})})
    assert_well_formed(rendered)
    assert block(rendered, "## Summary") == [
        "- **Ranker:** -. No ranker results to choose from.",
        "- **k:** -. No results to choose k from.",
        "- **k':** -. No system eval results to choose k' from.",
    ]
    assert "### Evidence for" not in rendered
    assert set(row.rsplit(" | ", 1)[1] for row in table_rows(rendered, "### Failure modes")) == {"- |"}


def test_missing_and_undefined_cells_print_a_dash(results):
    damaged = json.loads(json.dumps(to_jsonable(results)))  # the JSON form, free to edit
    bge = damaged["ranker_comparison"]["bge_m3"]
    bge["overall"]["recall"]["5"] = {"mean": None, "lo": None, "hi": None, "n": 0}
    del bge["overall"]["recall"]["30"]
    del bge["overall"]["latency"]
    del bge["by_pool"]
    bge["overall"]["cost_per_1k_usd"] = None
    del damaged["ranker_comparison"]["bge_m3_jev"]["recall_diff_vs"]
    del damaged["jev_variance"]["bge_m3_jev"]["mean_abs_score_diff"]
    del damaged["jev_variance"]["bge_m3_jev"]["n_unscored_pairs"]
    damaged["jev_variance"]["bge_m3_jev"]["snapshots_per_run"] = [["jev-test"]]
    damaged["jev_variance"]["bge_m3_jev"]["recall"]["5"]["per_run"] = [1.0]
    cell = damaged["system_eval"]["bge_m3"]["5"]["3"]
    del cell["overall"]["answer_accuracy"]
    del cell["overall"]["cost"]["fallback"]
    del cell["by_type"]
    del damaged["system_eval"]["bge_m3_jev"]["10"]["5"]
    del damaged["recommendation"]["k"]["steps"][0]["recall_gain"]
    del damaged["recommendation"]["failure_modes"]["time_filter_misses"]

    rendered = render_markdown(damaged)

    assert_well_formed(rendered)
    half = fmt_estimate(bootstrap_ci([1, 0, 1, 0]))
    assert table_rows(rendered, "### Recall@k")[0] == f"| bge_m3 | 4 | - | {half} | {half} | - | - |"
    latency = table_rows(rendered, "### Latency and cost by pool size")
    assert latency[0] == "| bge_m3 | all | 4 | - | - | - | - | vector_search 4.0 ms |"
    assert len(latency) == 5  # bge_m3 lost its pool slices
    assert block(rendered, "### Paired differences in Recall@k")[-1] == "_No data._"
    variance = block(rendered, "### bge_m3_jev")
    assert "| @5 | 100.0 | - | 100.0 | 0.0 |" in variance
    assert variance[-4].endswith("between runs: -")
    assert variance[-3].endswith("no usable score: -")
    assert variance[-1] == (
        "- Jev snapshot(s) served: run 0: jev-test; run 1: - (the runs hit different snapshots, so the "
        "drift above includes a model change)"
    )
    accuracy = table_rows(rendered, "### Answer accuracy by ranker, k and k'")
    assert accuracy[0].startswith("| bge_m3 | 5 | 3 | - | 33.3 [")
    assert accuracy[3].startswith("| bge_m3_jev | 10 | 3 | 66.7 [") and accuracy[3].endswith("| - |")
    assert "| bge_m3 | 5 | 3 | fallback | - | - | - | - | - | - |" in rendered
    assert table_rows(rendered, "### Evidence for k") == ["| 5 to 10 | - | +0.0 [+0.0, +0.0] |"]
    assert "| Padded time filter dropped a gold conversation | - |" in rendered


# --------------------------------------------------------------------------- #
# Files                                                                        #
# --------------------------------------------------------------------------- #
def strict_json(text: str) -> Any:
    """Parse as standard JSON: NaN and Infinity are not part of it."""

    def reject(constant: str) -> None:
        raise AssertionError(f"{constant} is not valid JSON")

    return json.loads(text, parse_constant=reject)


def test_write_report_writes_both_files(results, markdown, tmp_path):
    out_dir = tmp_path / "results" / "nested"  # created on demand

    md_path, json_path = write_report(results, "2026-11-06", out_dir=out_dir)

    assert (md_path, json_path) == (out_dir / "2026-11-06.md", out_dir / "2026-11-06.json")
    assert sorted(path.name for path in out_dir.iterdir()) == ["2026-11-06.json", "2026-11-06.md"]
    assert md_path.read_text(encoding="utf-8") == markdown

    text = json_path.read_text(encoding="utf-8")
    loaded = strict_json(text)
    assert text == json.dumps(loaded, sort_keys=True, indent=2, ensure_ascii=False) + "\n"  # keys are sorted
    assert set(loaded) == set(results)
    assert loaded["meta"]["seed"] == config.SEED
    assert loaded == json.loads(json.dumps(to_jsonable(results)))
    # NaN becomes null: an estimate over no queries, and the latency of the all-cached slice.
    lookup = loaded["ranker_comparison"]["bge_m3"]["by_type"]["lookup"]
    assert lookup["all_found"]["5"] == {"mean": None, "lo": None, "hi": None, "n": 0}
    cached = loaded["ranker_comparison"]["bge_m3_jev"]["by_pool"]["21-100"]["latency"]
    assert cached == {"p50": None, "p95": None, "mean": None, "n": 0}
    assert loaded["ranker_comparison"]["bge_m3_jev"]["n_missing"] == 3
    assert loaded["recommendation"]["ranker"]["choice"] == "bge_m3"


def test_json_file_renders_the_same_report(results, markdown, tmp_path):
    _, json_path = write_report(results, "run", out_dir=tmp_path)
    assert render_markdown(strict_json(json_path.read_text(encoding="utf-8"))) == markdown


def test_write_report_overwrites_an_earlier_run(results, tmp_path):
    write_report({"notes": "first"}, "2026-11-06", out_dir=tmp_path)
    md_path, json_path = write_report({"notes": "second"}, "2026-11-06", out_dir=tmp_path)
    assert "second" in md_path.read_text(encoding="utf-8")
    assert strict_json(json_path.read_text(encoding="utf-8")) == {"notes": "second"}


def test_write_report_writes_nothing_when_the_results_cannot_be_serialized(tmp_path):
    with pytest.raises(TypeError, match="cannot write set"):
        write_report({"meta": {"rankers": {"bge_m3"}}}, "2026-11-06", out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_write_report_defaults_to_the_results_directory():
    assert inspect.signature(write_report).parameters["out_dir"].default == config.RESULTS_DIR


def test_partial_results_write_valid_files(results, tmp_path):
    partial = copy.copy(results)
    del partial["system_eval"], partial["recommendation"]
    md_path, json_path = write_report(partial, "partial", out_dir=tmp_path)
    assert_well_formed(md_path.read_text(encoding="utf-8"))
    assert "system_eval" not in strict_json(json_path.read_text(encoding="utf-8"))
