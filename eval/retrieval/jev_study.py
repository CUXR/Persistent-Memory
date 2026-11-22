"""Dev-set study of how to apply Jev, and the pin the test run reads.

The ticket leaves open how Jev turns a candidate list into scores. This
command settles it on the dev split. For every Jev model id, every
:class:`JevScoringConfig` of a grid over (mode, primitive, wording, batch
size) and each Jev ranker, it ranks the answerable queries under the gold
filters, ``--runs`` times with distinct salts so that run-to-run variance
shows. It prints one table ranked by mean Recall@10, writes every number and
every per-query ranking to a JSON file, and with ``--pin`` writes
``eval/retrieval/jev_scoring.json`` (the chosen model id and variant), which
:mod:`eval.retrieval.runtime` reads when it builds the Jev rankers of the
test run.

Listwise scoring is one choice question over all candidates, which Jev caps
at ``MAX_CHOICE_OPTIONS`` options. ``bge_m3_jev`` hands Jev at most
``config.STAGE1_TOP_K`` candidates, so listwise is studied there; the pools
of ``jev_full_pool`` go past the cap, so listwise is skipped for it (the
output says so) and is never pinned unless ``--allow-listwise``.

Every Jev request is cached by (model, request, salt), so a study that was
interrupted resumes where it stopped when rerun with the same arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, metrics
from . import jev as jev_module
from . import runtime as runtime_module
from .dataset import load_queries
from .metrics import jev_snapshots, mean, nan_to_none
from .pipelines.jev_scoring import MAX_CHOICE_OPTIONS, PRIMITIVES, WORDINGS, JevScoringConfig
from .runs import run_ranker_over_queries
from .runtime import Runtime
from .schema import Query

# The gateways' ids for the two Jev versions the study compares.
DEFAULT_MODELS: dict[str, tuple[str, ...]] = {
    "openrouter": ("typesafe/jev-1.13", "typesafe/jev-1.14"),
    "vercel": ("typesafe-ai/jev-1.13.0", "typesafe-ai/jev-1.14.0"),
}
DEFAULT_BATCH_SIZES: tuple[int, ...] = (16, 64)
DEFAULT_RUNS = 2
DEFAULT_OUT = config.DATA_DIR / "jev_study.json"
STUDY_KS: tuple[int, ...] = (5, 10, 20)
PIN_K = 10  # the table is ranked, and the pin chosen, by Recall at this cut-off
ALL_FOUND_K = 10
RANKER_K = max(config.RECALL_KS)  # how deep every ranker ranks
# The ordering of the table and of the pin: what each key component compares.
_ORDERING = (f"Recall@{PIN_K}", "$ per query", "p50 latency")


# --------------------------------------------------------------------------- #
# The grid                                                                     #
# --------------------------------------------------------------------------- #
def default_models(gateway: str) -> list[str]:
    """The ids of Jev 1.13 and 1.14 on ``gateway``."""

    if gateway not in DEFAULT_MODELS:
        raise ValueError(f"no default Jev model ids for gateway {gateway!r}; pass --models")
    return list(DEFAULT_MODELS[gateway])


def study_variants(batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES) -> list[JevScoringConfig]:
    """The grid: pair and batched (per batch size) x primitive x wording, then listwise x wording."""

    sizes = sorted(set(batch_sizes))
    if not sizes or sizes[0] < 1:
        raise ValueError(f"batch sizes must be positive, got {list(batch_sizes)}")
    variants = [
        JevScoringConfig(mode="pair", primitive=primitive, wording=wording)
        for primitive in PRIMITIVES
        for wording in WORDINGS
    ]
    variants += [
        JevScoringConfig(mode="batched", primitive=primitive, wording=wording, batch_size=size)
        for primitive in PRIMITIVES
        for wording in WORDINGS
        for size in sizes
    ]
    variants += [JevScoringConfig(mode="listwise", wording=wording) for wording in WORDINGS]
    return variants


def select_variants(
    variants: Sequence[JevScoringConfig], labels: Iterable[str]
) -> list[JevScoringConfig]:
    """The variants whose labels are listed, in grid order; an unknown label is an error."""

    wanted = set(labels)
    known = [variant.label for variant in variants]
    unknown = sorted(wanted - set(known))
    if unknown:
        raise ValueError(f"unknown variant label(s) {unknown}; the grid has: {', '.join(known)}")
    return [variant for variant in variants if variant.label in wanted]


def supports_listwise(ranker: str) -> bool:
    """Whether every candidate list ``ranker`` hands Jev fits in one choice question."""

    return ranker == "bge_m3_jev" and config.STAGE1_TOP_K <= MAX_CHOICE_OPTIONS


def listwise_skip_reason(ranker: str) -> str:
    return (
        f"listwise scoring asks one choice question over the candidates, which Jev caps at "
        f"{MAX_CHOICE_OPTIONS} options, and {ranker} scores the whole filtered pool, which can "
        f"be larger"
    )


def study_salt(model: str, run: int) -> str:
    """Distinct per (model, run): repeated runs are fresh calls, which is what measures variance."""

    return f"study-{model}-run{run}"


def pin_config(scoring: JevScoringConfig) -> JevScoringConfig:
    """What the pin records: how Jev is asked, with salt and concurrency at their defaults.

    Both are per-run settings that ``Runtime.jev_scoring`` fills in, so the
    committed pin does not carry this machine's concurrency or a study salt.
    """

    return JevScoringConfig(
        mode=scoring.mode,
        primitive=scoring.primitive,
        wording=scoring.wording,
        batch_size=scoring.batch_size,
    )


# --------------------------------------------------------------------------- #
# Cells and their numbers                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    """One (ranker, model, variant, run) of the grid: its records and, derived, its numbers."""

    ranker: str
    model: str
    scoring: JevScoringConfig  # as it ran: with this run's salt and the machine's concurrency
    run: int
    records: list[metrics.RankerRecord]

    @property
    def variant(self) -> str:
        return self.scoring.label

    def describe(self) -> str:
        return f"{self.ranker} {self.model} {self.variant} run{self.run}"

    def to_json(self) -> dict[str, Any]:
        return {
            "ranker": self.ranker,
            "model": self.model,
            "variant": self.variant,
            "run": self.run,
            "salt": self.scoring.salt,
            "scoring": self.scoring.to_json(),
            "summary": aggregate([self]),
            "records": [record.to_json() for record in self.records],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Cell":
        return cls(
            ranker=data["ranker"],
            model=data["model"],
            scoring=JevScoringConfig.from_json(data["scoring"]),
            run=int(data["run"]),
            records=[metrics.RankerRecord.from_json(record) for record in data["records"]],
        )


def _recall(k: int) -> Callable[[metrics.RankerRecord], float]:
    return lambda record: metrics.recall_at_k(record.ranked_ids, record.gold_ids, k)


def _all_found(record: metrics.RankerRecord) -> float:
    """All-found@k over the queries with two or more gold conversations (NaN elsewhere)."""

    if len(record.gold_ids) < 2:
        return math.nan
    return metrics.all_found_at_k(record.ranked_ids, record.gold_ids, ALL_FOUND_K)


def _per_query_means(
    cells: Sequence[Cell], value: Callable[[metrics.RankerRecord], float]
) -> list[float]:
    """One number per query: the mean of ``value`` over every record of that query in ``cells``.

    Averaging over runs (and rankers) first keeps a query as one bootstrap
    unit, so the CI is over queries. Undefined (NaN) values are left out.
    Queries are in id order, so every metric resamples the same queries.
    """

    by_query: dict[str, list[float]] = {}
    for cell in cells:
        for record in cell.records:
            number = value(record)
            if not math.isnan(number):
                by_query.setdefault(record.query_id, []).append(number)
    return [mean(values) for _, values in sorted(by_query.items())]


def aggregate(cells: Sequence[Cell]) -> dict[str, Any]:
    """The numbers over ``cells``: a run, a variant's runs, or (for the pin) its runs on every ranker.

    Recall and all-found are, per query, the mean over the cells, then a
    bootstrap over queries. Latency leaves out records with ``n_cached > 0``:
    a cache replay's wall clock says nothing about Jev. Cost and requests are
    per-query means over every record; ``n_missing`` is a total.
    """

    records = [record for cell in cells for record in cell.records]
    runs = sorted({cell.run for cell in cells})
    by_run = {
        str(run): nan_to_none(
            mean(_per_query_means([c for c in cells if c.run == run], _recall(PIN_K)))
        )
        for run in runs
    }
    defined = [value for value in by_run.values() if value is not None]
    latency = metrics.latency_summary(
        record.latency_s for record in records if not metrics.from_cache(record)
    )
    return {
        "n_cells": len(cells),
        "n_runs": len(runs),
        "n_queries": len({record.query_id for record in records}),
        "recall": {
            str(k): metrics.bootstrap_ci(_per_query_means(cells, _recall(k))).to_json()
            for k in STUDY_KS
        },
        "recall_by_run": by_run,
        "run_spread": max(defined) - min(defined) if len(defined) > 1 else None,
        "all_found": {
            str(ALL_FOUND_K): metrics.bootstrap_ci(_per_query_means(cells, _all_found)).to_json()
        },
        "latency": {
            key: int(value) if key == "n" else nan_to_none(float(value))
            for key, value in latency.items()
        },
        "cost_per_query": nan_to_none(mean(record.usage.cost_usd for record in records)),
        "requests_per_query": nan_to_none(
            mean(float(record.extra.get("n_requests", 0)) for record in records)
        ),
        "n_missing": metrics.missing_scores(records) or 0,
        "n_cached_records": sum(1 for record in records if metrics.from_cache(record)),
        "jev_models": jev_snapshots(records),
    }


def _ordering_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Higher Recall@PIN_K first, then lower $ per query, then lower p50; unknowns last.

    Rounded so that two variants with the same per-query results compare
    equal whatever order their sums were taken in.
    """

    recall = row["recall"][str(PIN_K)]["mean"]
    cost = row["cost_per_query"]
    p50 = row["latency"]["p50"]
    return (
        -round(recall, 9) if recall is not None else math.inf,
        round(cost, 12) if cost is not None else math.inf,
        round(p50, 9) if p50 is not None else math.inf,
    )


def _decided_by(best: Mapping[str, Any], runner_up: Mapping[str, Any]) -> str:
    """Which key component separated the winner from the runner-up."""

    for name, mine, theirs in zip(_ORDERING, _ordering_key(best), _ordering_key(runner_up)):
        if mine != theirs:
            return name
    return "grid order (a complete tie)"


def table_rows(cells: Sequence[Cell]) -> list[dict[str, Any]]:
    """One row per (ranker, model, variant) over its runs, best first.

    Ranked by mean Recall@PIN_K, then $ per query, then p50; a p50 that is
    unknown (every record was a cache replay) ranks last, and what is still
    tied keeps grid order.
    """

    groups: dict[tuple[str, str, str], list[Cell]] = {}
    for cell in cells:
        groups.setdefault((cell.ranker, cell.model, cell.variant), []).append(cell)
    rows = [
        {"ranker": ranker, "model": model, "variant": variant, **aggregate(group)}
        for (ranker, model, variant), group in groups.items()
    ]
    rows.sort(key=_ordering_key)
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    return rows


def choose_pin(cells: Sequence[Cell], allow_listwise: bool = False) -> dict[str, Any]:
    """The (model, variant) with the best mean Recall@PIN_K over every ranker and run it ran on.

    Ties go to the lower $ per query, then the lower p50. Listwise variants
    are left out unless ``allow_listwise``: the pin serves ``jev_full_pool``
    too, and listwise cannot. The result carries the numbers that decided it
    and a one-paragraph ``reason``.
    """

    eligible = [cell for cell in cells if allow_listwise or cell.scoring.mode != "listwise"]
    if not eligible:
        raise ValueError(
            "nothing to pin: no cells, or only listwise variants (which cannot serve "
            "jev_full_pool) and --allow-listwise was not given"
        )
    groups: dict[tuple[str, str], list[Cell]] = {}
    for cell in eligible:
        groups.setdefault((cell.model, cell.variant), []).append(cell)
    candidates = [
        {
            "model": model,
            "variant": variant,
            "scoring": pin_config(group[0].scoring).to_json(),
            "rankers": sorted({cell.ranker for cell in group}),
            **aggregate(group),
        }
        for (model, variant), group in groups.items()
    ]
    candidates.sort(key=_ordering_key)
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    reason = f"{best['variant']} on {best['model']} wins: {_numbers(best)}."
    if runner_up is not None:
        reason += (
            f" Runner-up {runner_up['variant']} on {runner_up['model']}: {_numbers(runner_up)}; "
            f"decided by {_decided_by(best, runner_up)}."
        )
    return {**best, "runner_up": runner_up, "reason": reason}


def write_pin(choice: Mapping[str, Any], gateway: str, path: Path | None = None) -> Path:
    """Write the pin where ``Runtime`` reads it (``JEV_SCORING_PATH`` by default).

    The pin is ``{"model": id, "gateway": name, "served_models": [...], **scoring}``:
    the model id is the ``gateway``'s own spelling, so ``Runtime`` applies it on
    that gateway only, and ``served_models`` are the dated snapshot ids the API
    served the study, which the benchmark report compares with its own run's.
    ``scoring`` is what ``pin_config`` keeps; the per-run ``salt`` and this
    machine's ``concurrency`` stay out of the committed file.
    """

    if gateway not in config.GATEWAYS:
        raise ValueError(f"unknown gateway {gateway!r}; expected one of {sorted(config.GATEWAYS)}")
    target = Path(path) if path is not None else runtime_module.JEV_SCORING_PATH
    scoring = {
        key: value
        for key, value in choice["scoring"].items()
        if key not in ("concurrency", "salt")
    }
    payload = {
        "model": choice["model"],
        "gateway": gateway,
        "served_models": list(choice["jev_models"]),
        **scoring,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Running the grid                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Study:
    """Everything one invocation measured: the grid's cells, what was skipped, and the pin."""

    meta: dict[str, Any]
    cells: list[Cell] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    pin: dict[str, Any] | None = None

    def table(self) -> list[dict[str, Any]]:
        return table_rows(self.cells)

    def to_json(self) -> dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "cells": [cell.to_json() for cell in self.cells],
            "skipped": [dict(skip) for skip in self.skipped],
            "table": self.table(),
            "pin": self.pin,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Study":
        return cls(
            meta=dict(data["meta"]),
            cells=[Cell.from_json(cell) for cell in data["cells"]],
            skipped=[dict(skip) for skip in data.get("skipped", [])],
            pin=data.get("pin"),
        )


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _meta(
    runtime: Runtime,
    queries: Sequence[Query],
    models: Sequence[str],
    variants: Sequence[JevScoringConfig],
    runs: int,
    rankers: Sequence[str],
    split: str,
) -> dict[str, Any]:
    from .local_models import describe_hardware  # stage 1 of bge_m3_jev runs on this machine

    return {
        "split": split,
        "store": runtime.store_kind,
        "gateway": runtime.settings.gateway,
        "models": list(models),
        "variants": [variant.label for variant in variants],
        "runs": runs,
        "rankers": list(rankers),
        "n_queries": len(queries),
        "query_ids": [query.query_id for query in queries],
        "ks": list(STUDY_KS),
        "pin_k": PIN_K,
        "all_found_k": ALL_FOUND_K,
        "ranker_k": RANKER_K,
        "seed": config.SEED,
        "hardware": describe_hardware(),
    }


def run_study(
    runtime: Runtime,
    queries: Sequence[Query],
    *,
    models: Sequence[str],
    variants: Sequence[JevScoringConfig],
    runs: int = DEFAULT_RUNS,
    rankers: Sequence[str] = config.JEV_RANKERS,
    split: str = "dev",
    log: Callable[[str], None] = _log,
) -> Study:
    """Rank the answerable ``queries`` under the gold filters with every cell of the grid.

    One Jev client per model id, built through ``jev.build_jev`` with the
    runtime's cache and spend cap, serves every ranker and variant of that
    model; each cell gets the variant with the machine's concurrency and the
    salt of its (model, run). The queries run through the benchmark's own
    loop (``runs.run_ranker_over_queries``, without its warm-up: every cell's
    first call is a Jev call like the others), so a store that hands stage 1
    too few hits stops the study as it stops the benchmark. ``log`` gets one
    line per finished cell.
    """

    queries = [query for query in queries if query.answerable]
    if not queries:
        raise ValueError("the study needs answerable queries")
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    if not models:
        raise ValueError("the study needs at least one Jev model id")
    unknown = sorted(set(rankers) - set(config.JEV_RANKERS))
    if unknown:
        raise ValueError(f"not Jev rankers: {unknown}; expected ones of {config.JEV_RANKERS}")
    clients = {
        model: jev_module.build_jev(
            dataclasses.replace(runtime.settings, jev_model=model),
            cache=runtime.cache,
            spend=runtime.spend,
        )
        for model in models
    }
    study = Study(meta=_meta(runtime, queries, models, variants, runs, rankers, split))
    plan: list[tuple[str, JevScoringConfig, str, int]] = []
    for ranker_name in rankers:
        for variant in variants:
            if variant.mode == "listwise" and not supports_listwise(ranker_name):
                study.skipped.append(
                    {
                        "ranker": ranker_name,
                        "variant": variant.label,
                        "reason": listwise_skip_reason(ranker_name),
                    }
                )
                continue
            plan.extend(
                (ranker_name, variant, model, run) for model in models for run in range(runs)
            )
    for index, (ranker_name, variant, model, run) in enumerate(plan, 1):
        scoring = dataclasses.replace(
            variant, concurrency=runtime.settings.jev_concurrency, salt=study_salt(model, run)
        )
        ranker = runtime.ranker(ranker_name, k=RANKER_K, jev_scoring=scoring, jev=clients[model])
        records = run_ranker_over_queries(ranker, queries, run, warm_up=False)
        cell = Cell(ranker_name, model, scoring, run, records)
        study.cells.append(cell)
        log(f"[{index}/{len(plan)}] {cell.describe()}: {_numbers(aggregate([cell]))}")
    return study


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _numbers(row: Mapping[str, Any]) -> str:
    """The deciding numbers of a table row or pin candidate, in one clause."""

    recall = row["recall"][str(PIN_K)]
    return (
        f"Recall@{PIN_K} {_fmt(recall['mean'])} [{_fmt(recall['lo'])}, {_fmt(recall['hi'])}] "
        f"over {row['n_cells']} cell(s), ${_fmt(row['cost_per_query'], 6)}/query, "
        f"p50 {_fmt(row['latency']['p50'], 2)} s"
    )


def render_table(study: Study) -> str:
    """The ranked table as fixed-width text: a legend, then a line per (ranker, model, variant)."""

    meta = study.meta
    legend = (
        f"Jev study on {meta['n_queries']} answerable {meta['split']} queries, "
        f"{meta['runs']} run(s) per cell; ranked by Recall@{PIN_K} (mean over runs), "
        f"then $/query, then p50. "
        f"spread = max - min of per-run Recall@{PIN_K}; p50/p95 leave out cached calls; "
        f"AF = all-found over queries with 2+ gold conversations."
    )
    headers = [
        "#",
        "ranker",
        "model",
        "variant",
        f"R@{STUDY_KS[0]}",
        f"R@{PIN_K}",
        "95% CI",
        "spread",
        f"R@{STUDY_KS[-1]}",
        f"AF@{ALL_FOUND_K}",
        "p50 s",
        "p95 s",
        "$/query",
        "req/query",
        "missing",
        "snapshots",
    ]
    lines: list[list[str]] = []
    for row in study.table():
        recall, pinned = row["recall"], row["recall"][str(PIN_K)]
        lines.append(
            [
                str(row["rank"]),
                row["ranker"],
                row["model"],
                row["variant"],
                _fmt(recall[str(STUDY_KS[0])]["mean"]),
                _fmt(pinned["mean"]),
                f"[{_fmt(pinned['lo'])}, {_fmt(pinned['hi'])}]",
                _fmt(row["run_spread"]),
                _fmt(recall[str(STUDY_KS[-1])]["mean"]),
                _fmt(row["all_found"][str(ALL_FOUND_K)]["mean"]),
                _fmt(row["latency"]["p50"], 2),
                _fmt(row["latency"]["p95"], 2),
                _fmt(row["cost_per_query"], 6),
                _fmt(row["requests_per_query"], 1),
                str(row["n_missing"]),
                ",".join(row["jev_models"]) or "-",
            ]
        )
    widths = [max(len(cell) for cell in column) for column in zip(headers, *lines)]
    text_columns = 4

    def render(cells: Sequence[str]) -> str:
        return "  ".join(
            cell.ljust(width) if index < text_columns else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        ).rstrip()

    return "\n".join([legend, render(headers), *(render(line) for line in lines)])


def write_study(path: Path, study: Study) -> None:
    """The whole study as JSON with sorted keys (and no NaN: undefined numbers are null)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(study.to_json(), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _comma_list(text: str) -> list[str]:
    items: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if item and item not in items:
            items.append(item)
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _batch_sizes(text: str) -> tuple[int, ...]:
    try:
        sizes = sorted({int(item) for item in _comma_list(text)})
    except ValueError as error:
        message = f"expected comma-separated integers, got {text!r}"
        raise argparse.ArgumentTypeError(message) from error
    if sizes[0] < 1:
        raise argparse.ArgumentTypeError(f"batch sizes must be positive, got {text!r}")
    return tuple(sizes)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.jev_study",
        description=(
            "Compare the ways of applying Jev (and its versions) on the dev split, "
            "and optionally pin the best one for the test run."
        ),
    )
    parser.add_argument(
        "--split", default="dev", choices=("dev", "test"), help="queries to rank (default: dev)"
    )
    parser.add_argument(
        "--store", default="pg", choices=("pg", "memory"), help="corpus store (default: pg)"
    )
    parser.add_argument(
        "--models",
        type=_comma_list,
        help="comma-separated Jev model ids (default: the gateway's ids for Jev 1.13 and 1.14)",
    )
    parser.add_argument(
        "--variants",
        type=_comma_list,
        help=(
            "comma-separated variant labels to study, e.g. pair-noul-v1,batched-score-v2-b16 "
            "(default: the whole grid)"
        ),
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=DEFAULT_RUNS,
        help=f"repeats per cell, with distinct salts (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="comma-separated batch sizes for the batched mode (default: 16,64)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write the study (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--pin",
        action="store_true",
        help=f"write the best variant and model to {runtime_module.JEV_SCORING_PATH}",
    )
    parser.add_argument(
        "--allow-listwise",
        action="store_true",
        help="let a listwise variant be pinned (it cannot serve jev_full_pool)",
    )
    return parser


def main(argv: Sequence[str] | None = None, runtime: Runtime | None = None) -> int:
    """Run the study; input that cannot be used ends the process with one line, before any call."""

    args = build_parser().parse_args(argv)
    try:
        if runtime is None:
            runtime = Runtime(store_kind=args.store)
        if not runtime.settings.jev_url:
            raise ValueError(jev_module.no_jev_message(runtime.settings.gateway))
        models = args.models or default_models(runtime.settings.gateway)
        variants = study_variants(args.batch_sizes)
        if args.variants:
            variants = select_variants(variants, args.variants)
        if args.pin and not args.allow_listwise and all(v.mode == "listwise" for v in variants):
            raise ValueError(
                "--pin needs a variant that is not listwise (or --allow-listwise): "
                "listwise cannot serve jev_full_pool"
            )
        queries = [query for query in load_queries(args.split) if query.answerable]
        if not queries:
            raise ValueError(f"no answerable queries in the {args.split} split")
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    study = run_study(
        runtime, queries, models=models, variants=variants, runs=args.runs, split=args.split
    )
    if args.pin:
        study.pin = choose_pin(study.cells, allow_listwise=args.allow_listwise)
    print(render_table(study))
    for skip in study.skipped:
        print(f"skipped {skip['ranker']} x {skip['variant']}: {skip['reason']}")
    write_study(args.out, study)
    print(f"Wrote {args.out}")
    if study.pin is not None:
        path = write_pin(study.pin, runtime.settings.gateway)
        print(study.pin["reason"])
        print(f"Pinned {study.pin['variant']} on {study.pin['model']} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
