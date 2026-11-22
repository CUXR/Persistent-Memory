"""Add the gold labels the query generator missed, by pooling the rankers' top results.

    python -m eval.retrieval.data.pool_labels [--split dev|test|all] [--store pg|memory]
        [--workers N] [--depth 30] [--dry-run] [--queries PATH]

Every answerable and every unanswerable query is in scope; unknown-name
queries name nobody, so they have no pool. For each query every ranker the
gateway serves (all four, or the two local ones on ``EVAL_GATEWAY=openai``) runs
under the gold filters, their top ``depth`` results are pooled (a pool smaller
than the depth is pooled whole), and every pooled conversation that is not
already gold goes to the relevance judge.

* Answerable query: a confirmed hit joins ``gold_episode_ids`` (kept in
  start-time order) and the gold answer is rewritten over the full set.
* Unanswerable query: a confirmed hit means the query is not unanswerable
  after all. It is left as it is, flagged in its meta, and the command exits
  with code 2 once everything else is written, printing the command that
  regenerates the flagged queries.

``meta["pooling"]`` records what pooling did to a changed query. The run is
idempotent: a label added once is gold on the next run, the LLM cache makes a
repeated verdict free, and the file is rewritten only when a query changed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .. import config, dataset, judge, records
from ..interfaces import CorpusStore, Ranker, Usage
from ..runtime import Runtime
from ..schema import ANSWERABLE_TYPES, Conversation, Query
from ..spend import SpendLimitExceeded

POOLING_SALT = "pooling"  # the Jev rankers' cache salt: pooling never replays a benchmark run's scores
EXIT_FLAGGED = 2
REGENERATE_COMMAND = "python -m eval.retrieval.data.queries --regenerate"


def in_scope(query: Query) -> bool:
    """Whether pooling applies: the query has a pool (it is answerable or unanswerable)."""

    return query.answerable or query.type == "unanswerable"


@dataclass(frozen=True)
class Pool:
    """Every ranker's top results for one query, pooled."""

    pool_size: int
    candidates: tuple[str, ...]  # pooled ids that are not gold yet, in the order first seen


def pool_query(query: Query, rankers: Mapping[str, Ranker], owner_id: str = config.OWNER_ID) -> Pool:
    """Run every ranker under the gold filters and take the union of the rankings."""

    person_id, time_range = records.gold_filters(query)
    seen: dict[str, None] = {}
    pool_size = 0
    for ranker in rankers.values():
        result = ranker.rank_with_trace(owner_id, query.query, person_id, time_range)
        record = records.ranker_record(query, result)
        pool_size = record.pool_size
        for episode_id in record.ranked_ids:
            seen.setdefault(episode_id, None)
    gold = set(query.gold_episode_ids)
    return Pool(pool_size, tuple(episode_id for episode_id in seen if episode_id not in gold))


@dataclass
class Outcome:
    """What pooling did to one query in scope."""

    query: Query  # the record to write: the input record itself when nothing changed
    pool_size: int
    judged: int  # candidates sent to the judge (those outside the time window settle without a call)
    added: list[str] = field(default_factory=list)  # answerable: joined the gold set, oldest first
    flagged: list[str] = field(default_factory=list)  # unanswerable: relevant after all, oldest first
    usage: Usage = field(default_factory=Usage)  # the judge's and the gold-answer writer's calls

    @property
    def changed(self) -> bool:
        return bool(self.added or self.flagged)


def pool_labels(
    queries: Sequence[Query],
    runtime: Runtime,
    *,
    depth: int = config.LABEL_POOL_DEPTH,
    workers: int = 1,
) -> list[Outcome]:
    """Pool, judge and apply, for the queries in scope, in the order given. Writes nothing.

    The rankers run one query at a time (the Jev rankers parallelize inside);
    the relevance verdicts and the gold-answer rewrites run ``workers`` at a
    time, and their results are applied in query order, so the outcome does
    not depend on the thread timing.
    """

    names = runtime.settings.rankers
    rankers = {name: runtime.ranker(name, k=depth, salt=POOLING_SALT) for name in names}
    scoped = [query for query in queries if in_scope(query)]
    pools = [pool_query(query, rankers) for query in scoped]
    conversations = _fetch(
        runtime.store,
        (
            episode_id
            for query, pool in zip(scoped, pools, strict=True)
            for episode_id in (*query.gold_episode_ids, *pool.candidates)
        ),
    )
    model = runtime.settings.judge_model
    pairs = [
        (query, conversations[episode_id])
        for query, pool in zip(scoped, pools, strict=True)
        for episode_id in pool.candidates
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        verdicts = list(
            executor.map(lambda pair: _relevance(pair[0], pair[1], runtime.llm, model), pairs)
        )
        outcomes: list[Outcome] = []
        position = 0
        for query, pool in zip(scoped, pools, strict=True):
            own = verdicts[position : position + len(pool.candidates)]
            position += len(pool.candidates)
            relevant = [
                episode_id for episode_id, verdict in zip(pool.candidates, own, strict=True) if verdict.relevant
            ]
            usage = sum((verdict.usage for verdict in own), Usage())
            outcomes.append(
                _outcome(query, pool, _oldest_first(relevant, conversations), usage, depth, names, conversations)
            )
        # The gold answer of every grown set, written over the full set with the
        # old answer as the starting point (the outcome still carries it here).
        grown = [outcome for outcome in outcomes if outcome.added]
        rewrites = executor.map(
            lambda outcome: judge.write_gold_answer(
                outcome.query,
                [conversations[episode_id] for episode_id in outcome.query.gold_episode_ids],
                runtime.llm,
                model,
            ),
            grown,
        )
        for outcome, (answer, usage) in zip(grown, rewrites, strict=True):
            outcome.query.gold_answer = answer
            outcome.usage = outcome.usage + usage
    return outcomes


def _relevance(query: Query, conversation: Conversation, llm: Any, model: str) -> judge.RelevanceVerdict:
    """The judge's verdict on one candidate; one it cannot grade is not relevant, with a warning.

    The judge retries an unusable reply, the last time past the cache, so an
    error here is persistent for that pair; it must not stop the pass over
    every other query. The candidate stays out of the gold set, as it would
    with a "not relevant" verdict, and the warning says which pair to look at.
    """

    try:
        return judge.judge_relevance(query, conversation, llm, model)
    except judge.JudgeOutputError as error:
        print(
            f"warning: {query.query_id}: the relevance of {conversation.episode_id} was not judged "
            f"({error}); treated as not relevant",
            file=sys.stderr,
            flush=True,
        )
        return judge.RelevanceVerdict(relevant=False, reason=f"not judged: {error}", usage=Usage())


def _outcome(
    query: Query,
    pool: Pool,
    relevant: list[str],
    usage: Usage,
    depth: int,
    rankers: Sequence[str],
    conversations: Mapping[str, Conversation],
) -> Outcome:
    """Apply the verdicts to one query: a new record when they change it, else the record itself."""

    judged = len(pool.candidates)
    if not relevant:
        return Outcome(query, pool.pool_size, judged, usage=usage)
    pooling: dict[str, Any] = {"depth": depth, "rankers": list(rankers), "judged": judged}
    if not query.answerable:
        pooling["flagged"] = list(relevant)
        flagged = replace(query, meta={**query.meta, "pooling": pooling})
        return Outcome(flagged, pool.pool_size, judged, flagged=relevant, usage=usage)
    # Over repeated runs the additions accumulate and ``previous_gold_answer``
    # stays the generator's original, so the record always says what pooling
    # contributed to the labels as they now stand.
    previous = query.meta.get("pooling") or {}
    earlier = [episode_id for episode_id in previous.get("added", ()) if episode_id in query.gold_episode_ids]
    pooling["added"] = _oldest_first([*earlier, *relevant], conversations)
    pooling["previous_gold_answer"] = previous.get("previous_gold_answer", query.gold_answer)
    grown = replace(
        query,
        gold_episode_ids=_oldest_first([*query.gold_episode_ids, *relevant], conversations),
        meta={**query.meta, "pooling": pooling},
    )
    return Outcome(grown, pool.pool_size, judged, added=relevant, usage=usage)


def _oldest_first(episode_ids: Iterable[str], conversations: Mapping[str, Conversation]) -> list[str]:
    """Distinct episode ids by start time, then id: the order gold sets are kept in."""

    return sorted(set(episode_ids), key=lambda episode_id: (conversations[episode_id].start_time, episode_id))


def _fetch(store: CorpusStore, episode_ids: Iterable[str]) -> dict[str, Conversation]:
    """Every conversation named, in one store call; a missing id is a broken dataset."""

    wanted = sorted(set(episode_ids))
    found = store.get_conversations(wanted)
    missing = [episode_id for episode_id in wanted if episode_id not in found]
    if missing:
        raise ValueError(f"the store has no conversation for {len(missing)} pooled or gold ids: {missing[:5]}")
    return found


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def describe_change(outcome: Outcome, dry_run: bool) -> str:
    """One line per changed query, in the past tense once written, conditional on a dry run."""

    head = f"  {outcome.query.query_id} ({outcome.query.type}, pool {outcome.pool_size}, judged {outcome.judged}): "
    if outcome.flagged:
        verb = "would flag" if dry_run else "flagged"
        return head + f"{verb} {', '.join(outcome.flagged)}"
    verb = "would add" if dry_run else "added"
    return head + f"{verb} {', '.join(outcome.added)}; gold answer {'to be ' if dry_run else ''}rewritten"


def summary(outcomes: Sequence[Outcome], depth: int, rankers: Sequence[str], charged_usd: float) -> str:
    """The run's totals. ``charged_usd`` is what the spend tracker recorded (cached replies are free)."""

    usage = sum((outcome.usage for outcome in outcomes), Usage())
    per_type: Counter[str] = Counter()
    for outcome in outcomes:
        per_type[outcome.query.type] += len(outcome.added)
    by_type = ", ".join(f"{qtype} {per_type[qtype]}" for qtype in ANSWERABLE_TYPES if per_type[qtype])
    lines = [
        f"Label pooling at depth {depth} over {', '.join(rankers)}:",
        f"  queries in scope        {len(outcomes)}",
        f"  candidates judged       {sum(outcome.judged for outcome in outcomes)}",
        f"  labels added            {sum(per_type.values())}" + (f" ({by_type})" if by_type else ""),
        f"  gold answers rewritten  {sum(1 for outcome in outcomes if outcome.added)}",
        f"  flagged unanswerable    {sum(1 for outcome in outcomes if outcome.flagged)}",
        f"  hosted spend            ${charged_usd:.4f} charged this run; {usage.calls} LLM calls "
        f"(relevance judge and gold-answer writer), {usage.input_tokens} in / {usage.output_tokens} out tokens, "
        f"${usage.cost_usd:.4f} at list price (cached replies included)",
    ]
    return "\n".join(lines)


def regenerate_command(query_ids: Sequence[str], queries_path: Path) -> str:
    """The exact ``queries`` invocation that replaces the flagged queries in ``queries_path``."""

    command = f"{REGENERATE_COMMAND} {' '.join(query_ids)}"
    if Path(queries_path).resolve() != config.QUERIES_PATH.resolve():
        command += f" --out {queries_path}"
    return command


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None, runtime: Runtime | None = None) -> int:
    """Pool, judge, write the query file back in place. Exit 2 when unanswerable queries were flagged."""

    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.data.pool_labels",
        description="Find gold labels the query generator missed and add them to the query file.",
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all", help="queries to pool (default: all)")
    parser.add_argument("--store", choices=("pg", "memory"), default="pg", help="corpus store the rankers read")
    parser.add_argument(
        "--workers", type=int, default=None, help="parallel judge calls (default: EVAL_LLM_CONCURRENCY)"
    )
    parser.add_argument(
        "--depth", type=int, default=config.LABEL_POOL_DEPTH, help="results pooled per ranker (default: %(default)s)"
    )
    parser.add_argument("--dry-run", action="store_true", help="judge and print the would-be changes, write nothing")
    parser.add_argument("--queries", type=Path, default=config.QUERIES_PATH, help="query file to update in place")
    args = parser.parse_args(argv)
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if runtime is None:
        runtime = Runtime(store_kind=args.store)
    workers = args.workers or max(1, runtime.settings.llm_concurrency)

    queries = dataset.load_queries(path=args.queries)
    selected = [query for query in queries if args.split == "all" or query.split == args.split]
    try:
        outcomes = pool_labels(selected, runtime, depth=args.depth, workers=workers)
    except (judge.JudgeOutputError, SpendLimitExceeded) as error:
        print(f"error: {error}; {args.queries} was not written", file=sys.stderr)
        return 1

    changed = [outcome for outcome in outcomes if outcome.changed]
    if changed:
        print("Would change:" if args.dry_run else "Changed:")
        for outcome in changed:
            print(describe_change(outcome, args.dry_run))
    print(summary(outcomes, args.depth, runtime.settings.rankers, runtime.spend.total))

    # A query flagged on an earlier run comes out of this one identical: no write then.
    originals = {query.query_id: query for query in queries}
    updated = {
        outcome.query.query_id: outcome.query
        for outcome in changed
        if outcome.query.to_json() != originals[outcome.query.query_id].to_json()
    }
    if args.dry_run:
        print(f"dry run: {args.queries} was not written ({len(updated)} queries would change)")
    elif updated:
        dataset.save_queries([updated.get(query.query_id, query) for query in queries], args.queries)
        print(f"wrote {len(queries)} queries to {args.queries} ({len(updated)} changed)")
    else:
        print(f"no changes: {args.queries} was left as it is")

    flagged = [outcome.query.query_id for outcome in outcomes if outcome.flagged]
    if flagged:
        print(
            f"{len(flagged)} unanswerable queries have a relevant conversation after all; "
            "regenerate them with:"
        )
        print(f"  {regenerate_command(flagged, args.queries)}")
        return EXIT_FLAGGED
    return 0


if __name__ == "__main__":
    sys.exit(main())
