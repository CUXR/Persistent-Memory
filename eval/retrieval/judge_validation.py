"""A quick human check of the answer judge.

``sample_items`` draws a small stratified sample of judged system answers, a
person labels each one by the judge's own rubric, and ``agreement`` compares
those labels with the judge's verdicts.

    python -m eval.retrieval.judge_validation sample --records <system records .jsonl> --out sample.json
    python -m eval.retrieval.judge_validation score --sample sample.json --labels labels.json \\
        [--out summary.json]

``sample.json`` is the list of items. ``labels.json`` is one JSON object that
maps each item's ``key`` to "correct", "partial" or "incorrect". ``score``
writes its agreement summary to ``config.JUDGE_VALIDATION_PATH`` unless
``--out`` says otherwise; the benchmark report (``run.py``) includes the
summary found at that path and says so when there is none.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import config
from .dataset import load_queries, read_jsonl
from .judge import ANSWER_RUBRIC
from .schema import DECLINE_PATHS, VERDICTS, Query
from .timeutil import describe_moment

STRATA: tuple[str, ...] = ("correct", "partial_incorrect", "declined")
# Declined answers are all the same refusal, so a few are enough to see the
# judge's refusal rule at work: they get half the share of the other strata.
_STRATUM_WEIGHTS: dict[str, int] = {"correct": 2, "partial_incorrect": 2, "declined": 1}
_STRATUM_LABELS: dict[str, str] = {
    "correct": "correct",
    "partial_incorrect": "partial/incorrect",
    "declined": "declined",
}
# What the person labelling needs to grade like the judge: its rubric and its clock.
_LABELLING_GUIDE = f"""\
Label every item by the judge's rubric, and decide before you read its "judge_verdict".
The questions were asked at {describe_moment(config.NOW)}.
{ANSWER_RUBRIC}
Save the labels as one JSON object that maps each item's "key" to "correct", "partial" or
"incorrect", then run:
  python -m eval.retrieval.judge_validation score --sample <sample.json> --labels <labels.json>"""


def item_key(record: Mapping[str, Any]) -> str:
    """Identify one system answer: its query and the (ranker, k, k') setting that produced it."""

    return f"{record['query_id']}/{record['ranker']}/k{record['k']}/kp{record['k_prime']}"


def sample_items(
    system_records: Iterable[Mapping[str, Any]],
    queries: Iterable[Query],
    n: int = 10,
    seed: int = config.SEED,
) -> list[dict[str, Any]]:
    """Draw up to ``n`` judged system answers for a person to label, at most one per query.

    Only an answer that the judge graded, to a query that has a gold answer,
    can be drawn. Each one belongs to a stratum: "declined" when the flow ended
    without an answer, otherwise "correct" or "partial_incorrect" by the judge's
    verdict. The slots are split 2:2:1 over correct, partial/incorrect and
    declined, and a stratum with too few queries passes its slots on. Every
    stratum that is given a slot appears in the sample whenever the records
    allow it, even when the strata share queries.

    The draw depends on the records and ``seed`` only, not on the order of
    ``system_records``. Items come back sorted by key, so their order does not
    give the strata away. Fewer than ``n`` items come back when fewer queries
    qualify.
    """

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    queries_by_id = {query.query_id: query for query in queries}
    # stratum -> query id -> that query's records in the stratum
    candidates: dict[str, dict[str, list[Mapping[str, Any]]]] = {stratum: {} for stratum in STRATA}
    for record in system_records:
        stratum = _stratum(record)
        query = queries_by_id.get(record["query_id"])
        if stratum is None or query is None or not query.gold_answer:
            continue
        candidates[stratum].setdefault(query.query_id, []).append(record)

    rng = random.Random(seed)
    draw_order: dict[str, list[str]] = {}
    for stratum in STRATA:
        draw_order[stratum] = sorted(candidates[stratum])
        rng.shuffle(draw_order[stratum])
    items = []
    for query_id, stratum in sorted(_allocate(draw_order, n).items()):
        record = rng.choice(sorted(candidates[stratum][query_id], key=item_key))
        items.append(_item(record, queries_by_id[query_id]))
    return sorted(items, key=lambda item: item["key"])


def agreement(items: Sequence[Mapping[str, Any]], human_labels: Mapping[str, str]) -> dict[str, Any]:
    """Compare the human labels with the judge's verdicts on the sampled items.

    Returns ``{"n", "agreement_rate", "disagreements"}``: the number of items,
    the share on which the label equals the verdict (None without items), and
    every item on which they differ, with both labels. Every item must have a
    label, because a rate over the items someone chose to label says little;
    labels for keys outside ``items`` are ignored.
    """

    unlabelled = [item["key"] for item in items if item["key"] not in human_labels]
    if unlabelled:
        raise ValueError(f"no human label for {len(unlabelled)} item(s): {unlabelled}")
    disagreements: list[dict[str, Any]] = []
    for item in items:
        label = human_labels[item["key"]]
        if label not in VERDICTS:
            raise ValueError(f"the label for {item['key']} must be one of {VERDICTS}, got {label!r}")
        if label != item["judge_verdict"]:
            disagreements.append(
                {
                    "key": item["key"],
                    "query": item["query"],
                    "gold_answer": item["gold_answer"],
                    "system_answer": item["system_answer"],
                    "judge_verdict": item["judge_verdict"],
                    "human_label": label,
                }
            )
    n = len(items)
    return {
        "n": n,
        "agreement_rate": (n - len(disagreements)) / n if n else None,
        "disagreements": disagreements,
    }


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #
def _stratum(record: Mapping[str, Any]) -> str | None:
    """The stratum of a system record, or None when the judge did not grade it."""

    verdict = record.get("verdict")
    if verdict not in VERDICTS:
        return None
    if record["path"] in DECLINE_PATHS:
        return "declined"
    return "correct" if verdict == "correct" else "partial_incorrect"


def _item(record: Mapping[str, Any], query: Query) -> dict[str, Any]:
    """One sample item. The verdict comes last, below what the person reads to form a label."""

    return {
        "key": item_key(record),
        "query_id": query.query_id,
        "query": query.query,
        "gold_answer": query.gold_answer,
        "system_answer": record["final_answer"],
        "ranker": record["ranker"],
        "k": record["k"],
        "k_prime": record["k_prime"],
        "judge_verdict": record["verdict"],
    }


def _allocate(draw_order: Mapping[str, Sequence[str]], n: int) -> dict[str, str]:
    """Fill up to ``n`` slots with distinct queries: ``{query id: stratum}``.

    ``draw_order`` lists each stratum's queries in the order to draw them. A
    query can sit in several strata (correct at one setting, declined at
    another), so drawing stratum by stratum could spend the few queries of a
    small stratum on a large one. The slots are instead filled as a bipartite
    matching by augmenting paths: a slot whose queries are all taken makes
    another slot move to a different query of its own stratum, and a filled
    slot stays filled. Each stratum's first slot is filled before any second
    one, so as many strata as possible appear; then the quotas are filled as
    far as the records allow, and the slots that a short stratum leaves over
    go round the others.
    """

    present = [stratum for stratum in STRATA if draw_order[stratum]]
    if not present:
        return {}
    quotas = _quotas(n, present)
    holder: dict[str, str] = {}  # query id -> the stratum whose slot it fills

    def claim(stratum: str, tried: set[str]) -> bool:
        """Fill one more slot of ``stratum``, moving other slots to other queries if need be."""

        options = [query_id for query_id in draw_order[stratum] if query_id not in tried]
        for query_id in options:
            if query_id not in holder:
                holder[query_id] = stratum
                return True
        for query_id in options:
            if query_id in tried:
                continue
            tried.add(query_id)
            if claim(holder[query_id], tried):
                holder[query_id] = stratum
                return True
        return False

    for round_index in range(max(quotas.values())):
        for stratum in present:
            if round_index < quotas[stratum]:
                claim(stratum, set())
    progressed = True
    while progressed and len(holder) < n:
        progressed = False
        for stratum in present:
            if len(holder) < n and claim(stratum, set()):
                progressed = True
    return holder


def _quotas(n: int, strata: Sequence[str]) -> dict[str, int]:
    """Split ``n`` slots over ``strata`` by weight: largest remainder, ties to the earlier stratum."""

    total = sum(_STRATUM_WEIGHTS[stratum] for stratum in strata)
    quotas = {stratum: n * _STRATUM_WEIGHTS[stratum] // total for stratum in strata}
    by_remainder = sorted(strata, key=lambda stratum: -(n * _STRATUM_WEIGHTS[stratum] % total))
    for stratum in by_remainder[: n - sum(quotas.values())]:
        quotas[stratum] += 1
    return quotas


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """Run ``sample`` or ``score``; input that cannot be used ends the process with one line."""

    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.judge_validation",
        description="Human check of the answer judge: draw a sample to label, then score the labels.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="write a stratified sample of judged answers to label")
    sample.add_argument("--records", type=Path, required=True, help="system records of a run (JSONL)")
    sample.add_argument("--out", type=Path, required=True, help="where to write the sample (JSON)")
    sample.set_defaults(run=_sample_command)
    score = commands.add_parser("score", help="compare the human labels with the judge's verdicts")
    score.add_argument("--sample", type=Path, required=True, help="the sample that was labelled (JSON)")
    score.add_argument("--labels", type=Path, required=True, help="JSON object: item key -> label")
    score.add_argument(
        "--out",
        type=Path,
        default=config.JUDGE_VALIDATION_PATH,
        help="the summary the report includes (default: %(default)s)",
    )
    score.set_defaults(run=_score_command)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError) as error:  # unusable input: one line instead of a traceback
        raise SystemExit(f"error: {error}") from error


def _sample_command(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    items = sample_items(records, load_queries())
    if not items:
        raise ValueError(f"{args.records} has no judged answers to queries that have a gold answer")
    _write_json(args.out, items)
    records_by_key = {item_key(record): record for record in records}
    drawn = Counter(_stratum(records_by_key[item["key"]]) for item in items)
    counts = ", ".join(f"{drawn[stratum]} {_STRATUM_LABELS[stratum]}" for stratum in STRATA)
    print(f"Wrote {len(items)} items to {args.out}: {counts}.")
    print(_LABELLING_GUIDE)
    return 0


def _score_command(args: argparse.Namespace) -> int:
    items = _read_json(args.sample)
    labels = _read_json(args.labels)
    if not isinstance(items, list) or not isinstance(labels, dict):
        raise ValueError("--sample must hold a JSON list of items and --labels a JSON object of labels")
    summary = agreement(items, labels)
    print(_describe(summary))
    _write_json(args.out, summary)
    print(f"Wrote the summary to {args.out}; the report includes it from there.")
    return 0


def _describe(summary: Mapping[str, Any]) -> str:
    """The agreement summary as text: the headline, then each disagreement in full."""

    n = summary["n"]
    if not n:
        return "No items to score."
    agreed = n - len(summary["disagreements"])
    lines = [f"Judge and human agree on {agreed} of {n} items ({summary['agreement_rate']:.0%})."]
    for item in summary["disagreements"]:
        lines += [
            f"- {item['key']}: judge {item['judge_verdict']}, human {item['human_label']}",
            f"    question: {item['query']}",
            f"    gold answer: {item['gold_answer']}",
            f"    system answer: {item['system_answer']}",
        ]
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    """Load a JSON file; the labels are written by hand, so a syntax error names the file."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
