"""Render the benchmark results as a Markdown report and a JSON file.

``results`` is the one dict the runner assembles:

    meta               date, seed, split, counts, models, jev, hardware, store;
                       models.jev_snapshots lists the Jev snapshot ids the run was
                       served (a warning is printed when there is more than one)
    ranker_comparison  metrics.ranker_comparison(...)
    jev_variance       metrics.jev_variance(...)
    system_eval        metrics.system_eval(...)
    judge_validation   optional: how the LLM judge was checked
    recommendation     metrics.recommend(...)
    notes              optional free text

Every section is optional: a missing one is reported as not run and a missing
or NaN cell prints a dash, so a partial run still gives a readable report.
Estimates are accepted as ``Estimate`` objects or in their JSON form, and
nothing depends on dict order, so the JSON file re-renders to the same report.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from . import config
from .metrics import FAILURE_MODES, as_estimate, dig, numeric_keys, ordered_rankers
from .schema import ANSWERABLE_TYPES, POOL_BUCKETS, QUERY_TYPES, format_dt

DASH = "-"
NOT_RUN = "_Not run._"
NO_DATA = "_No data._"
_CI = f"{round(100 * config.CI_LEVEL)}% CI"

_META_SCALARS = ("date", "split", "seed", "store", "corpus_fingerprint")
_META_SECTIONS = {
    "counts": "Dataset",
    "models": "Models",
    "hardware": "Hardware (local models)",
}
_CHOICES = (("ranker", "Ranker"), ("k", "k"), ("k_prime", "k'"))
_FAILURE_LABELS = {
    "extraction_name_errors": "Extraction returned the wrong name",
    "missed_time_ranges": "No time range extracted for a temporal query",
    "time_filter_misses": "Padded time filter dropped a gold conversation",
    "spurious_time_filters": "Time range extracted where the query has none",
    "misrouted": "Stopped before ranking (unknown or ambiguous name)",
    "empty_pools": "Filters left nothing to rank",
    "stage1_misses": f"Gold conversation missing from the bge-m3 top {config.STAGE1_TOP_K} (gold filters)",
    "gold_not_in_top_k": "Gold conversation missing from the top k the answerer saw",
    "false_declines_gold_retrieved": "Declined although every gold conversation was retrieved",
    "wrong_answers_gold_retrieved": "Wrong answer although every gold conversation was retrieved",
    "wrong_answers_from_summaries": "Wrong answer given from summaries alone",
    "unjudged": "Answer not graded (the judge gave no usable verdict)",
}
_JEV_SNAPSHOTS = "jev_snapshots"  # meta["models"] key: the dated snapshot ids the Jev API served
_PATH_COLUMNS = (
    ("partial_rate", "Partial", False),
    ("fallback_rate", "Fallback rate", False),
    ("summary_path_accuracy", "Summary-path accuracy", True),
    ("fallback_path_accuracy", "Fallback-path accuracy", True),
    ("needed_dialogue_accuracy", "Accuracy when dialogue was needed", True),
)
_DECLINE_COLUMNS = (
    ("correct_decline_rate", "Correct decline (unanswerable)", True),
    ("false_decline_rate", "False decline (answerable)", False),
    ("false_decline_rate_gold_retrieved", "False decline, gold retrieved", True),
    ("misrouted_rate", "Misrouted (answerable)", False),
)
_EXTRACTION_ROWS = (
    ("name_accuracy", "Name extraction accuracy (all queries)"),
    ("person_accuracy", "Person resolution accuracy (all queries)"),
    ("time_range_extracted_rate", "Time range extracted (temporal queries)"),
    ("time_filter_recall", "Time-filter recall (temporal queries)"),
    ("spurious_time_range_rate", "Spurious time range (queries without a gold range)"),
    ("unknown_name_exact_rate", "Exact unknown-name message (unknown_name queries)"),
)
_COST_PATHS = (("overall", "all"), ("summary", "summary"), ("fallback", "fallback"))
_DISAGREEMENT_COLUMNS = (
    ("query", "Query"),
    ("gold_answer", "Gold answer"),
    ("system_answer", "System answer"),
    ("judge_verdict", "Judge"),
    ("human_label", "Human"),
)
_SLICE_COLUMNS = (
    ("answer_accuracy", "Answer accuracy"),
    ("partial_rate", "Partial"),
    ("fallback_rate", "Fallback rate"),
    ("false_decline_rate", "False decline"),
    ("correct_decline_rate", "Correct decline"),
    ("name_accuracy", "Name accuracy"),
    ("time_filter_recall", "Time-filter recall"),
)


# --------------------------------------------------------------------------- #
# Formatting helpers                                                           #
# --------------------------------------------------------------------------- #
def _missing(value: Any) -> bool:
    """None or NaN: there is nothing to print."""

    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt_pct(value: float | None, signed: bool = False) -> str:
    """A proportion as a percentage figure with one decimal (``91.2``), or a dash."""

    if _missing(value):
        return DASH
    figure = round(100.0 * value, 1) + 0.0  # adding 0.0 turns a rounded -0.0 into 0.0
    return f"{figure:+.1f}" if signed else f"{figure:.1f}"


def fmt_estimate(value: Any, signed: bool = False, with_n: bool = False) -> str:
    """``91.2 [88.0, 94.1]`` for an estimate, or a dash when it is undefined.

    ``signed`` is for paired differences: figures carry their sign and the cell
    is bold when the interval excludes zero. ``with_n`` appends the sample size,
    for metrics whose population differs from the table's.
    """

    estimate = as_estimate(value)
    if estimate is None or not estimate.defined:
        return DASH
    low, high = fmt_pct(estimate.lo, signed), fmt_pct(estimate.hi, signed)
    text = f"{fmt_pct(estimate.mean, signed)} [{low}, {high}]"
    if signed and (estimate.lo > 0.0 or estimate.hi < 0.0):
        text = f"**{text}**"
    return f"{text} (n={estimate.n})" if with_n else text


def fmt_seconds(value: float | None) -> str:
    """A latency with the unit that fits its size: ``4.2 ms``, ``312 ms``, ``2.41 s``."""

    if _missing(value):
        return DASH
    if value < 0.01:
        return f"{1000.0 * value:.1f} ms"
    if value < 0.9995:
        return f"{1000.0 * value:.0f} ms"
    return f"{value:.2f} s"


def fmt_usd(value: float | None) -> str:
    """Dollars to the cent, or to two significant digits for amounts under ten cents."""

    if _missing(value):
        return DASH
    if value == 0:
        return "$0"
    if value < 0.0001:
        return "<$0.0001"
    decimals = max(2, 1 - math.floor(math.log10(value)))
    return f"${value:,.{decimals}f}"


def fmt_count(value: float | None) -> str:
    """A count or a token average as a whole number with thousands separators."""

    return DASH if _missing(value) else f"{round(value):,}"


def fmt_float(value: float | None) -> str:
    """A score-scale figure (a score difference, a Jaccard overlap) to three decimals."""

    return DASH if _missing(value) else f"{value:.3f}"


def fmt_value(value: Any) -> str:
    """A setup or judge-validation value on one line."""

    estimate = as_estimate(value)
    if estimate is not None:
        return fmt_estimate(estimate, with_n=True)
    if _missing(value):
        return DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={fmt_value(value[key])}" for key in sorted(value)) or DASH
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt_value(item) for item in value) or DASH
    return " ".join(str(value).split()) or DASH  # free text may span lines


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], left: int = 1) -> str:
    """A GitHub-flavoured Markdown table.

    The first ``left`` columns are left-aligned labels and the rest are
    right-aligned figures. None, NaN and empty cells print a dash, and pipes and
    line breaks are neutralized so a value can never break the table.
    """

    def cell(value: Any) -> str:
        text = DASH if _missing(value) or value == "" else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join(":---" if i < left else "---:" for i in range(len(headers))) + " |",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells for {len(headers)} columns: {row!r}")
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    """Reduce results to plain JSON types.

    Estimates and records go through their ``to_json``; NaN and infinities
    become null, because JSON has no spelling for them.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return format_dt(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_json"):
        return to_jsonable(value.to_json())
    raise TypeError(f"cannot write {type(value).__name__} to the results file")


# --------------------------------------------------------------------------- #
# Small shared pieces                                                          #
# --------------------------------------------------------------------------- #
def _is_section(value: Any) -> bool:
    """A non-empty mapping that is not an estimate gets its own nested block."""

    return isinstance(value, Mapping) and bool(value) and as_estimate(value) is None


def _in_order(keys: Any, first: Sequence[str]) -> list[str]:
    """The ``first`` keys that are present, then the rest alphabetically."""

    return [key for key in first if key in keys] + sorted(key for key in keys if key not in first)


def _bullets(mapping: Mapping[str, Any], depth: int = 0) -> list[str]:
    """A nested mapping as an indented Markdown list, keys sorted, leaves on one line."""

    lines: list[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if _is_section(value):
            lines.append(f"{'  ' * depth}- **{key}:**")
            lines.extend(_bullets(value, depth + 1))
        else:
            lines.append(f"{'  ' * depth}- **{key}:** {fmt_value(value)}")
    return lines


def _sample_size(estimates: Sequence[Any]) -> int | None:
    """The n shared by a row of estimates over one population (None if all are missing)."""

    return next((found.n for found in map(as_estimate, estimates) if found is not None), None)


def _scopes() -> list[tuple[str, tuple[str, ...]]]:
    """Row scopes of the by-pool tables: every query, then each pool-size bucket."""

    return [("all", ("overall",)), *((bucket, ("by_pool", bucket)) for bucket in POOL_BUCKETS)]


def _present(table: Mapping[str, Any], names: Sequence[str], dimension: str) -> list[str]:
    """The slices of one dimension that any ranker has, in reporting order."""

    order = QUERY_TYPES if dimension == "by_type" else POOL_BUCKETS
    found = {key for name in names for key in dig(table, name, dimension) or {}}
    return [key for key in order if key in found]


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #
def _title(meta: Mapping[str, Any]) -> str:
    details = [str(meta["date"])] if meta.get("date") else []
    if meta.get("split"):
        details.append(f"{meta['split']} split")
    return "# Retrieval ranker benchmark" + (f" ({', '.join(details)})" if details else "")


def _choice_lines(recommendation: Mapping[str, Any]) -> str:
    """The recommendation in three lines: ranker, k, k'."""

    lines = []
    for key, label in _CHOICES:
        choice = dig(recommendation, key, "choice")
        rationale = dig(recommendation, key, "rationale") or "No rationale was recorded."
        lines.append(f"- **{label}:** {DASH if choice is None else f'`{choice}`'}. {rationale}")
    return "\n".join(lines)


def _summary_section(recommendation: Mapping[str, Any]) -> list[str]:
    if not recommendation:
        return ["## Summary", "_No recommendation was drafted for this run._"]
    return ["## Summary", _choice_lines(recommendation)]


def _method(meta: Mapping[str, Any]) -> str:
    seed = meta.get("seed", config.SEED)
    return "\n".join(
        [
            "- **Ranker comparison.** Answerable queries only. The pool is filtered with the "
            "gold name and the padded gold time range, so ranking is isolated from extraction "
            "errors. Rankers see summaries only. Recall@k is the fraction of a query's gold "
            "conversations in the top k, averaged over queries. All-found@k is the share of "
            "queries whose gold conversations are all in the top k. Stage 1 "
            f"Recall@{config.STAGE1_TOP_K} is the fraction of gold conversations that bge-m3 "
            "hands to the second stage. Latency is wall clock per query and cost is the "
            "hosted-API spend.",
            "- **System eval.** Every query goes through the real flow: the extraction call, "
            "PersonResolver, time-range padding, the ranker, and the answering model, which "
            "reads the top k summaries first and falls back to the full dialogue of the top k' "
            "conversations. An LLM judge grades each answer against the gold answer. Answer "
            "accuracy is the share of answerable queries judged correct; a declined or "
            "misrouted query counts as wrong.",
            f"- **Statistics.** Proportions are means over queries with a {_CI} from a "
            f"percentile bootstrap ({config.BOOTSTRAP_RESAMPLES:,} resamples of queries, seed "
            f"{seed}), printed in percent as `mean [low, high]`. Differences are paired: both "
            "sides are resampled over the same queries, and a difference is bold when its "
            "interval excludes zero.",
            "- **Slices.** By query type and by pool size, the number of conversations left "
            f"after the gold filters ({', '.join(POOL_BUCKETS)}).",
        ]
    )


def _setup_section(meta: Mapping[str, Any]) -> list[str]:
    blocks = ["## Setup"]
    scalars = [key for key in meta if not _is_section(meta[key])]
    if scalars:
        blocks.append(
            "\n".join(
                f"- **{key}:** {fmt_value(meta[key])}" for key in _in_order(scalars, _META_SCALARS)
            )
        )
    sections = [key for key in meta if _is_section(meta[key])]
    for key in _in_order(sections, tuple(_META_SECTIONS)):
        title = _META_SECTIONS.get(key) or key.replace("_", " ").capitalize()
        lines = _bullets(meta[key])
        served = meta[key].get(_JEV_SNAPSHOTS) if key == "models" else None
        if isinstance(served, (list, tuple)) and len(served) > 1:
            lines.append(
                f"- **Warning:** {len(served)} Jev snapshots served this run, so the Jev "
                "figures mix model versions."
            )
        blocks += [f"### {title}", "\n".join(lines)]
    return blocks + ["### What was measured", _method(meta)]


def _ranker_section(table: Mapping[str, Any]) -> list[str]:
    names = ordered_rankers(table)
    if not names:
        return ["## Ranker comparison", NOT_RUN]
    ks = numeric_keys(k for name in names for k in dig(table, name, "overall", "recall") or {})
    at = [f"@{k}" for k in ks]
    blocks = [
        "## Ranker comparison",
        f"Answerable queries under the gold name and time range. Cells are percent with a {_CI}.",
    ]

    rows = [
        [
            name,
            dig(table, name, "overall", "n"),
            *(fmt_estimate(dig(table, name, "overall", "recall", k)) for k in ks),
            fmt_count(dig(table, name, "n_missing")),
        ]
        for name in names
    ]
    blocks += [
        "### Recall@k",
        "Unscored: candidates, summed over the queries, for which the hosted scorer returned "
        "no usable score (they sort last); a dash for rankers that do not count them.",
        md_table(["Ranker", "n", *at, "Unscored"], rows),
    ]

    rows = []
    for name in names:
        for key, label in (
            ("all_found", "2+ gold conversations"),
            ("all_found_multi_conversation", "multi_conversation type"),
        ):
            found = [dig(table, name, "overall", key, k) for k in ks]
            rows.append([name, label, _sample_size(found), *(fmt_estimate(e) for e in found)])
    blocks += [
        "### All-found@k",
        "Share of queries whose gold conversations are all in the top k.",
        md_table(["Ranker", "Queries", "n", *at], rows, left=2),
    ]

    rows = [
        [
            name,
            fmt_estimate(dig(table, name, "overall", "stage1_recall"), with_n=True),
            fmt_estimate(dig(table, name, "overall", "stage1_recall_all_pools"), with_n=True),
        ]
        for name in names
        if dig(table, name, "overall", "stage1_recall_all_pools") is not None
    ]
    headers = ["Ranker", f"Pools > {config.STAGE1_TOP_K}", "All pools"]
    blocks += [
        f"### Stage 1 Recall@{config.STAGE1_TOP_K}",
        md_table(headers, rows) if rows else "_No ranker with a first stage was run._",
    ]

    rows = []
    for name in names:
        for scope, path in _scopes():
            entry = dig(table, name, *path)
            if entry is None:
                continue
            stages = entry.get("stage_latency_p50") or {}
            rows.append(
                [
                    name,
                    scope,
                    entry.get("n"),
                    dig(entry, "latency", "n"),
                    fmt_seconds(dig(entry, "latency", "p50")),
                    fmt_seconds(dig(entry, "latency", "p95")),
                    fmt_usd(entry.get("cost_per_1k_usd")),
                    ", ".join(f"{stage} {fmt_seconds(stages[stage])}" for stage in sorted(stages)),
                ]
            )
    headers = ["Ranker", "Pool", "Queries", "Timed", "p50", "p95", "$ / 1k queries", "Stage p50"]
    blocks += [
        "### Latency and cost by pool size",
        "Latency is wall clock per query over the Timed queries only: a query with a scoring "
        "response served from the cache times a cache read, not the model, and is left out. "
        "Cost covers every query.",
        md_table(headers, rows, left=2),
    ]

    for dimension, title in (("by_type", "query type"), ("by_pool", "pool size")):
        columns = _present(table, names, dimension)
        sizes = [
            next((n for name in names if (n := dig(table, name, dimension, c, "n"))), None)
            for c in columns
        ]
        headers = ["Ranker", "k", *(f"{c} (n={fmt_count(n)})" for c, n in zip(columns, sizes))]
        rows = [
            [name, k, *(fmt_estimate(dig(table, name, dimension, c, "recall", k)) for c in columns)]
            for name in names
            for k in ks
        ]
        blocks += [f"### Recall@k by {title}", md_table(headers, rows) if columns else NO_DATA]

    rows = []
    for scope, path in _scopes():
        for earlier, later in combinations(names, 2):
            diffs = [dig(table, later, "recall_diff_vs", earlier, *path, k) for k in ks]
            if _sample_size(diffs) is not None:
                cells = (fmt_estimate(diff, signed=True) for diff in diffs)
                rows.append([scope, f"{later} - {earlier}", _sample_size(diffs), *cells])
    blocks += [
        "### Paired differences in Recall@k",
        f"Mean over the same queries of Recall@k(A) - Recall@k(B), in points with a {_CI}; "
        "bold when the interval excludes zero.",
        md_table(["Pool", "A - B", "n", *at], rows, left=2) if rows else NO_DATA,
    ]
    return blocks


def _variance_section(table: Mapping[str, Any]) -> list[str]:
    blocks = ["## Jev run-to-run variance"]
    names = ordered_rankers(table)
    if not names:
        return blocks + [NOT_RUN]
    for name in names:
        entry = table[name]
        runs = list(entry.get("runs") or [])
        rows = []
        for k in numeric_keys(entry.get("recall") or {}):
            stats = entry["recall"][k]
            per_run = list(stats.get("per_run") or [])
            cells = [fmt_pct(per_run[i] if i < len(per_run) else None) for i in range(len(runs))]
            rows.append([f"@{k}", *cells, fmt_pct(stats.get("mean")), fmt_pct(stats.get("std"))])
        headers = ["Recall", *(f"run {run}" for run in runs), "mean", "std"]
        blocks += [
            f"### {name}",
            f"{len(runs)} runs over the same {fmt_count(entry.get('n_queries'))} queries. "
            "Recall@k in percent; the standard deviation is across runs, in points.",
            md_table(headers, rows),
            "- Mean absolute score difference per (query, conversation) between runs: "
            f"{fmt_float(entry.get('mean_abs_score_diff'))}\n"
            "- Pairs left out of that difference because a run had no usable score: "
            f"{fmt_count(entry.get('n_unscored_pairs'))}\n"
            "- Mean Jaccard overlap of the top-10 sets between runs: "
            f"{fmt_float(entry.get('top10_jaccard'))}\n"
            f"- {_snapshots_line(runs, entry.get('snapshots_per_run'))}",
        ]
    return blocks


def _snapshots_line(runs: Sequence[Any], snapshots_per_run: Any) -> str:
    """Which Jev snapshot served each run, flagged when the runs differ."""

    per_run = list(snapshots_per_run or [])
    served = [fmt_value(per_run[i] if i < len(per_run) else None) for i in range(len(runs))]
    text = "; ".join(f"run {run}: {ids}" for run, ids in zip(runs, served))
    if len(set(served)) > 1:
        text += " (the runs hit different snapshots, so the drift above includes a model change)"
    return f"Jev snapshot(s) served: {text}"


def _settings(table: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Every measured (ranker, k, k') in reporting order."""

    return [
        (name, k, k_prime)
        for name in ordered_rankers(table)
        for k in numeric_keys(table[name])
        for k_prime in numeric_keys(table[name][k])
    ]


def _focus(
    table: Mapping[str, Any], recommendation: Mapping[str, Any]
) -> tuple[str, str, str] | None:
    """The setting the detail tables zoom in on: the recommended one if it was measured."""

    settings = _settings(table)
    wanted = tuple(str(dig(recommendation, key, "choice")) for key, _ in _CHOICES)
    if wanted in settings:
        return wanted
    return settings[0] if settings else None


def _cost_cells(group: Mapping[str, Any]) -> list[Any]:
    """n, tokens in and out, the timed n, p50 and p95 latency, and dollars per 1k queries for one path."""

    spend = group.get("cost_usd")
    return [
        group.get("n"),
        fmt_count(group.get("input_tokens")),
        fmt_count(group.get("output_tokens")),
        group.get("n_timed"),
        fmt_seconds(dig(group, "latency", "p50")),
        fmt_seconds(dig(group, "latency", "p95")),
        fmt_usd(None if _missing(spend) else 1000.0 * spend),
    ]


def _system_section(table: Mapping[str, Any], recommendation: Mapping[str, Any]) -> list[str]:
    settings = _settings(table)
    focus = _focus(table, recommendation)
    if focus is None:
        return ["## System eval", NOT_RUN]
    names = ordered_rankers(table)
    k_primes = numeric_keys(k_prime for _, _, k_prime in settings)
    blocks = [
        "## System eval",
        "All queries through the real extraction, person resolution, padding, ranking and "
        f"answering. Cells are percent with a {_CI}; (n=...) is the population when it "
        "differs from the row's.",
    ]

    rows = []
    for name, k in dict.fromkeys((name, k) for name, k, _ in settings):
        cells = [dig(table, name, k, kp, "overall", "answer_accuracy") for kp in k_primes]
        rows.append([name, k, _sample_size(cells), *(fmt_estimate(cell) for cell in cells)])
    blocks += [
        "### Answer accuracy by ranker, k and k'",
        "Share of answerable queries judged correct.",
        md_table(["Ranker", "k", "n", *(f"k'={kp}" for kp in k_primes)], rows),
    ]

    def per_setting(columns: Sequence[tuple[str, str, bool]]) -> str:
        body = [
            [
                name,
                k,
                kp,
                *(
                    fmt_estimate(dig(table, name, k, kp, "overall", key), with_n=with_n)
                    for key, _, with_n in columns
                ),
            ]
            for name, k, kp in settings
        ]
        return md_table(["Ranker", "k", "k'", *(label for _, label, _ in columns)], body)

    blocks += ["### Fallback rate and accuracy by path", per_setting(_PATH_COLUMNS)]
    blocks += ["### Decline rates", per_setting(_DECLINE_COLUMNS)]

    name, k, kp = focus
    overall = dig(table, name, k, kp, "overall") or {}
    rows = [
        [label, fmt_estimate(overall.get(key)), _sample_size([overall.get(key)])]
        for key, label in _EXTRACTION_ROWS
    ]
    blocks += [
        "### Extraction and unknown-name handling",
        f"Measured at {name}, k={k}, k'={kp}. Extraction and name resolution run before "
        "ranking, so these do not depend on the ranker, k or k'.",
        md_table(["Metric", f"% [{_CI}]", "n"], rows),
    ]

    rows = [
        [*setting, label, *_cost_cells(dig(table, *setting, "overall", "cost", path) or {})]
        for setting in settings
        for path, label in _COST_PATHS
    ]
    headers = ["Ranker", "k", "k'", "Path", "n", "Tokens in", "Tokens out", "n timed", "p50", "p95"]
    blocks += [
        "### Tokens, latency and cost by path",
        "Per query, summed over extraction, ranking and answering. The summary path is every "
        "query answered from summaries; the fallback path is every query whose summary attempt "
        "failed, whether the dialogue then answered it or not. The latencies cover the n timed "
        "queries: a query whose ranking replayed cached Jev answers is left out of them, as its "
        "rank stage measured a cache read.",
        md_table([*headers, "$ / 1k queries"], rows, left=4),
    ]

    rows = []
    for scope, path in (
        ("all", ("overall",)),
        *((t, ("by_type", t)) for t in QUERY_TYPES),
        *((bucket, ("by_pool", bucket)) for bucket in POOL_BUCKETS),
    ):
        entry = dig(table, name, k, kp, *path)
        if entry is not None:
            cells = (fmt_estimate(entry.get(key)) for key, _ in _SLICE_COLUMNS)
            rows.append([scope, entry.get("n"), *cells])
    blocks += [
        "### Slices at the focus setting",
        f"{name}, k={k}, k'={kp}, by query type and by pool size.",
        md_table(["Slice", "n", *(label for _, label in _SLICE_COLUMNS)], rows),
    ]

    for dimension, columns, title in (
        ("by_type", ANSWERABLE_TYPES, "query type"),
        ("by_pool", POOL_BUCKETS, "pool size"),
    ):
        rows = [
            [
                ranker,
                *(
                    fmt_estimate(dig(table, ranker, k, kp, dimension, c, "answer_accuracy"))
                    for c in columns
                ),
            ]
            for ranker in names
        ]
        blocks += [
            f"### Answer accuracy by {title}, per ranker",
            f"At k={k}, k'={kp}.",
            md_table(["Ranker", *columns], rows),
        ]
    return blocks


def _judge_section(validation: Any) -> list[str]:
    """How far the LLM judge agrees with human labels, then every answer they disagree on.

    Reads the summary that ``judge_validation.agreement`` returns (``n``,
    ``agreement_rate``, ``disagreements``); whatever else the section holds is
    listed as it is.
    """

    blocks = ["## Judge validation"]
    if not isinstance(validation, Mapping):
        return blocks + [fmt_value(validation)]
    rest = dict(validation)
    disagreements = rest.get("disagreements")
    tabular = isinstance(disagreements, (list, tuple)) and all(
        isinstance(item, Mapping) for item in disagreements
    )
    if tabular:
        del rest["disagreements"]
    if not _missing(rest.get("agreement_rate")):
        rate, n = rest.pop("agreement_rate"), rest.pop("n", None)
        blocks.append(
            f"The judge's verdict matched the human label on {fmt_pct(rate)}% of the "
            f"{fmt_count(n)} answers checked."
        )
    if rest:
        blocks.append("\n".join(_bullets(rest)))
    if tabular and disagreements:
        headers = [label for _, label in _DISAGREEMENT_COLUMNS]
        rows = [
            [fmt_value(item.get(key)) for key, _ in _DISAGREEMENT_COLUMNS] for item in disagreements
        ]
        blocks += ["### Disagreements", md_table(headers, rows, left=len(headers))]
    return blocks


def _ranker_evidence(
    recommendation: Mapping[str, Any],
    ranker_table: Mapping[str, Any],
    system_table: Mapping[str, Any],
) -> list[str]:
    """The figures the ranker choice rests on, one row per ranker."""

    names = ordered_rankers([*ranker_table, *system_table])
    if not names:
        return []
    reference = dig(recommendation, "ranker", "reference") or {}
    recall_k, k, k_prime = (reference.get(key) for key in ("recall_k", "k", "k_prime"))
    tied_on_recall = dig(recommendation, "ranker", "tied_on_recall") or []
    tied_on_accuracy = dig(recommendation, "ranker", "tied_on_accuracy") or []
    rows = [
        [
            name,
            fmt_estimate(dig(ranker_table, name, "overall", "recall", recall_k)),
            fmt_estimate(dig(system_table, name, k, k_prime, "overall", "answer_accuracy")),
            fmt_seconds(dig(ranker_table, name, "overall", "latency", "p50")),
            fmt_usd(dig(ranker_table, name, "overall", "cost_per_1k_usd")),
            "yes" if name in tied_on_recall else "no",
            "yes" if name in tied_on_accuracy else "no",
        ]
        for name in names
    ]
    recall_label = "Recall@k" if recall_k is None else f"Recall@{recall_k}"
    headers = [
        "Ranker",
        recall_label,
        "Answer accuracy" if k is None else f"Answer accuracy (k={k}, k'={k_prime})",
        "p50 latency",
        "$ / 1k queries",
        f"Tied on {recall_label}",
        "Tied on accuracy",
    ]
    return [
        "### Evidence for the ranker",
        "A ranker is tied on a metric while it is not significantly below the best on it (the "
        f"paired {_CI} of the difference includes zero). The rankers tied on both metrics are "
        "in contention; if there is none, those tied on answer accuracy are. Among those the "
        "lowest p50 latency wins, then the lowest cost.",
        md_table(headers, rows),
    ]


def _recommendation_section(
    recommendation: Mapping[str, Any],
    ranker_table: Mapping[str, Any],
    system_table: Mapping[str, Any],
) -> list[str]:
    blocks = ["## Recommendation"]
    if not recommendation:
        return blocks + [NOT_RUN]
    blocks.append(_choice_lines(recommendation))
    blocks += _ranker_evidence(recommendation, ranker_table, system_table)

    steps = dig(recommendation, "k", "steps") or []
    if steps:
        rows = [
            [
                f"{step.get('from')} to {step.get('to')}",
                fmt_estimate(step.get("recall_gain"), signed=True),
                fmt_estimate(step.get("answer_accuracy_gain"), signed=True),
            ]
            for step in steps
        ]
        blocks += [
            "### Evidence for k",
            f"Paired gain from each k to the next, in points with a {_CI}.",
            md_table(["Step", "Recall@k", "Answer accuracy"], rows),
        ]

    steps = dig(recommendation, "k_prime", "steps") or []
    if steps:
        rows = [
            [
                f"{step.get('from')} to {step.get('to')}",
                fmt_estimate(step.get("needed_dialogue_accuracy_gain"), signed=True, with_n=True),
                fmt_count(step.get("extra_input_tokens")),
            ]
            for step in steps
        ]
        headers = ["Step", "Accuracy when dialogue was needed", "Extra input tokens per query"]
        blocks += [
            "### Evidence for k'",
            f"Paired gain from each k' to the next over the queries that needed dialogue, in "
            f"points with a {_CI}, and what the deeper fallback costs those queries.",
            md_table(headers, rows),
        ]

    counts = recommendation.get("failure_modes") or {}
    rows = [[_FAILURE_LABELS[mode], fmt_count(counts.get(mode))] for mode in FAILURE_MODES]
    blocks += [
        "### Failure modes",
        "Queries per category at the recommended setting. A query can fall into several "
        "categories; a dash means the category was not measured.",
        md_table(["Failure mode", "Queries"], rows),
    ]
    return blocks


# --------------------------------------------------------------------------- #
# Entry points                                                                 #
# --------------------------------------------------------------------------- #
def render_markdown(results: Mapping[str, Any]) -> str:
    """Render the whole report; see the module docstring for what ``results`` holds."""

    meta = results.get("meta") or {}
    recommendation = results.get("recommendation") or {}
    ranker_table = results.get("ranker_comparison") or {}
    system_table = results.get("system_eval") or {}
    blocks = [
        _title(meta),
        *_summary_section(recommendation),
        *_setup_section(meta),
        *_ranker_section(ranker_table),
        *_variance_section(results.get("jev_variance") or {}),
        *_system_section(system_table, recommendation),
    ]
    if results.get("judge_validation"):
        blocks += _judge_section(results["judge_validation"])
    blocks += _recommendation_section(recommendation, ranker_table, system_table)
    if results.get("notes"):
        blocks += ["## Notes", str(results["notes"]).strip()]
    return "\n\n".join(blocks) + "\n"


def write_report(
    results: Mapping[str, Any], date_str: str, out_dir: Path = config.RESULTS_DIR
) -> tuple[Path, Path]:
    """Write ``<date>.md`` and ``<date>.json`` into ``out_dir`` and return both paths.

    The JSON has sorted keys and null in place of NaN, so two runs diff cleanly
    and any JSON parser can read it. Both texts are built before either file is
    written: results that cannot be serialized leave no half-written pair.
    """

    markdown = render_markdown(results)
    payload = json.dumps(
        to_jsonable(results), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{date_str}.md"
    json_path = out_dir / f"{date_str}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(payload + "\n", encoding="utf-8")
    return md_path, json_path
