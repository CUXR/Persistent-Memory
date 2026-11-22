"""Run the retrieval benchmark: ``python -m eval.retrieval.run --split test``.

Three stages, each resumable from the records of an earlier invocation:

* ``rankers``: every answerable query of the split through each ranker under
  the gold filters (the gold person and the padded gold time range), at
  k = max(RECALL_KS), so ranking is measured apart from extraction. Jev
  rankers run ``--jev-runs`` times under distinct salts, so every run makes
  fresh Jev calls while an interrupted run resumes from the cache. Rankers run
  one query at a time, so the recorded latency is one ranking at a time.
* ``system``: every query of the split through the real flow (the extraction
  call, PersonResolver, padding, ranking, summary-first answering with the
  dialogue fallback) with each ranker at every k in SYSTEM_KS and k' in
  FALLBACK_KS, each final answer graded by the LLM judge. A query is prepared
  once and ranked once per ranker; a ranking from the ``rankers`` stage is
  reused when the system's filters equal the gold filters. Queries run in a
  thread pool; ranking runs under one lock so local-model latency stays
  sequential and comparable.
* ``report``: the aggregates of ``metrics`` over the records of the run,
  written by ``report.write_report`` to RESULTS_DIR/<date>.{md,json}.

Records stream to CACHE_DIR/runs/<date>-<split>/{ranker,system}_records.jsonl
as they complete, and a rerun skips the keys already there. The run directory
is bound to the dataset it was recorded against (``dataset.sha256``: the
queries and conversations files), so a changed dataset under the same
``--date`` is refused rather than aggregated with stale gold ids, and it keeps
``meta.json``: the hardware and model settings in force when its first
measuring stage ran, which the report states instead of whatever is in force
when the report is written. A ``pg`` run first checks that the database holds
the committed dataset (the fingerprint ``load_db`` recorded against the files'
own), since the stable episode ids would otherwise rank stale rows under the
right ids. The spend cap and Ctrl-C end the run cleanly with those records
saved and print the exact command that resumes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from . import config, dataset, db, judge, metrics, report
from .interfaces import Ranker, RankResult, RankTrace
from .jev import no_jev_message
from .metrics import RankerRecord, SystemRecord
from .records import gold_filters, system_record
from .runs import RecordLog, gold_pool_size, run_ranker_over_queries, salt_for, warm_up_salt
from .data.pool_labels import regenerate_command
from .runtime import Runtime, load_jev_pin
from .schema import QUERY_TYPES, Query
from .spend import SpendLimitExceeded
from .system.pipeline import Prepared, SystemPipeline, SystemResult, compose_system_result

STAGE_CHOICES: tuple[str, ...] = ("rankers", "system", "report", "all")
RANKER_RECORDS = "ranker_records.jsonl"
SYSTEM_RECORDS = "system_records.jsonl"
DATASET_MARKER = "dataset.sha256"  # the digest of the dataset the run dir's records were made against
RUN_META = "meta.json"  # the hardware and model settings in force when the run's records were made
PROGRESS_EVERY = 25  # ranker-stage progress line every this many queries


def run_dir(date: str, split: str) -> Path:
    """Where the records of one run live (read at call time, so tests can redirect it)."""

    return config.CACHE_DIR / "runs" / f"{date}-{split}"


def judge_validation_path() -> Path:
    """Where ``judge_validation score`` writes its summary, which the report includes when it exists."""

    return config.JUDGE_VALIDATION_PATH


def dataset_digest() -> str:
    """SHA-256 over the queries file and the conversations file, as they are now.

    Query ids are stable across regenerations and label pooling rewrites the
    queries file in place, so the ids alone cannot tell one version of the
    dataset from another; the bytes can. A conversations file that is absent
    (a ``pg`` run on a machine without the corpus files) digests as absent, so
    its appearance later counts as a change too.
    """

    digest = hashlib.sha256()
    for path in (config.QUERIES_PATH, config.CONVERSATIONS_PATH):
        data = path.read_bytes() if path.exists() else None
        digest.update(f"{path.name}:{-1 if data is None else len(data)}\n".encode("utf-8"))
        if data is not None:
            digest.update(data)
    return digest.hexdigest()


def rank_result_from_record(record: RankerRecord, k: int) -> RankResult:
    """Rebuild a ranker's result from its stored record, cut to the top ``k``.

    A ranker's top ``k`` is the first ``k`` of its deeper ranking, so the
    ``rankers``-stage record (k = max(RECALL_KS)) serves the system stage
    (k = max(SYSTEM_KS)).
    """

    trace = RankTrace(
        ranker=record.ranker,
        pool_size=record.pool_size,
        latency_s=record.latency_s,
        stage1_ids=None if record.stage1_ids is None else list(record.stage1_ids),
        stage_latency_s=dict(record.stage_latency_s),
        usage=record.usage,
        extra=dict(record.extra),
    )
    return RankResult(ranked=list(zip(record.ranked_ids, record.scores))[:k], trace=trace)


# --------------------------------------------------------------------------- #
# The run                                                                      #
# --------------------------------------------------------------------------- #
class Benchmark:
    """One (date, split) run: its queries, its record logs and the three stages."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        split: str,
        date: str,
        rankers: Sequence[str] | None = None,
        jev_runs: int = config.JEV_VARIANCE_RUNS,
        limit: int | None = None,
        workers: int | None = None,
        notes: Path | None = None,
        out: TextIO | None = None,
    ) -> None:
        if rankers is None:  # every ranker the gateway can run
            rankers = runtime.settings.rankers
        unknown = [name for name in rankers if name not in config.RANKER_NAMES]
        if unknown:
            raise ValueError(
                f"unknown rankers {unknown}; expected a subset of {config.RANKER_NAMES}"
            )
        if jev_runs < 1:
            raise ValueError(f"jev_runs must be at least 1, got {jev_runs}")
        self.runtime = runtime
        self.split = split
        self.date = date
        self.rankers = [name for name in config.RANKER_NAMES if name in rankers]  # config order
        self.jev_runs = jev_runs
        self.limit = limit
        self.workers = workers if workers is not None else runtime.settings.llm_concurrency
        self.notes_path = notes
        self.out = out if out is not None else sys.stdout
        queries = dataset.load_queries(split)
        self.queries: list[Query] = queries[:limit] if limit is not None else queries
        self._refuse_flagged_queries()
        self.run_dir = run_dir(date, split)
        self.ranker_log: RecordLog[RankerRecord] = RecordLog(
            self.run_dir / RANKER_RECORDS, RankerRecord.from_json
        )
        self.system_log: RecordLog[SystemRecord] = RecordLog(
            self.run_dir / SYSTEM_RECORDS, SystemRecord.from_json
        )
        self._bind_dataset()
        self._rank_lock = threading.Lock()
        self._system_rankers: dict[str, Ranker] = {}

    def _refuse_flagged_queries(self) -> None:
        """An unanswerable query that label pooling flagged is not a benchmark query yet.

        Pooling found a conversation that answers it, so declining it is not
        the correct outcome, and it must not enter the decline rates; it is
        replaced by regenerating it, which pooling printed the command for.
        """

        flagged = [query.query_id for query in self.queries if (query.meta.get("pooling") or {}).get("flagged")]
        if flagged:
            raise SystemExit(
                f"error: {len(flagged)} {self.split} quer{'y is' if len(flagged) == 1 else 'ies are'} "
                f"flagged by label pooling as answerable after all ({', '.join(flagged)}); regenerate "
                f"them first: {regenerate_command(flagged, config.QUERIES_PATH)}"
            )

    def _bind_dataset(self) -> None:
        """Bind the run directory to the dataset: refuse records made against another one.

        A directory without records is (re)bound to the dataset as it is now.
        One with records must carry the marker of the same dataset; otherwise
        its records hold stale gold ids, which a resume would keep and a
        report would aggregate, so the run stops here with no record touched.
        """

        digest = dataset_digest()
        marker = self.run_dir / DATASET_MARKER
        if not self.ranker_log and not self.system_log:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(digest + "\n", encoding="utf-8")
            return
        recorded = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
        if recorded != digest:
            what = "was recorded against a different dataset" if recorded else "has records but no dataset marker"
            raise SystemExit(
                f"error: run directory {self.run_dir} {what} ({len(self.ranker_log)} ranker records, "
                f"{len(self.system_log)} system records); its gold ids may be stale, so it is not "
                f"resumed or reported: pass a new --date, or delete the directory to rerun"
            )

    def close(self) -> None:
        self.ranker_log.close()
        self.system_log.close()

    def say(self, message: str) -> None:
        print(message, file=self.out, flush=True)

    @property
    def spend(self) -> float:
        return self.runtime.spend.total

    def _ranker(self, name: str, k: int, run: int, *, warm_up: bool = False) -> Ranker:
        """Build ``name`` at ``k``; Jev rankers get this run's salt, local rankers need none.

        With ``warm_up`` a Jev ranker gets the run's warm-up salt instead, so
        its answers never land in the cache the run's first query then reads.
        """

        salt = ""
        if name in config.JEV_RANKERS:
            salt = salt_for(self.date, self.split, run)
            if warm_up:
                salt = warm_up_salt(salt)
        return self.runtime.ranker(name, k=k, salt=salt)

    # Stage 1: the ranker comparison --------------------------------------- #
    def _ranker_sink(self, label: str, total: int) -> Callable[[RankerRecord], None]:
        """Persist each record as it completes and print progress every few queries."""

        done = 0

        def on_record(record: RankerRecord) -> None:
            nonlocal done
            self.ranker_log.append(record)
            done += 1
            if done % PROGRESS_EVERY == 0 or done == total:
                self.say(
                    f"{label}: {done}/{total} queries; {len(self.ranker_log)} ranker records; "
                    f"spend ${self.spend:.4f}"
                )

        return on_record

    def run_rankers(self) -> None:
        """Every answerable query through each ranker under the gold filters."""

        queries = [query for query in self.queries if query.answerable]
        k = max(config.RECALL_KS)
        self._check_corpus()
        for name in self.rankers:
            runs = self.jev_runs if name in config.JEV_RANKERS else 1
            for run in range(runs):
                label = f"[rankers] {name} run {run}"
                todo = [q for q in queries if (q.query_id, name, run) not in self.ranker_log]
                if not todo:
                    self.say(f"{label}: all {len(queries)} queries already recorded")
                    continue
                self.say(f"{label}: {len(todo)} of {len(queries)} queries to rank")
                self._record_environment()
                ranker = self._ranker(name, k=k, run=run)
                jev = name in config.JEV_RANKERS
                if jev:
                    # The loop's own warm-up would rank the first query under the
                    # run's salt and so put its Jev answers in the cache, making
                    # the first record a replay; warm up under a throwaway salt.
                    first = todo[0]
                    warm = self._ranker(name, k=k, run=run, warm_up=True)
                    warm.rank_with_trace(config.OWNER_ID, first.query, *gold_filters(first))
                run_ranker_over_queries(
                    ranker, todo, run, warm_up=not jev, on_record=self._ranker_sink(label, len(todo))
                )

    # What was in force when the records were made ------------------------- #
    def _check_corpus(self) -> None:
        """A ``pg`` run ranks over the committed dataset or not at all.

        The episode ids are stable across regenerations, so a database loaded
        from an older ``conversations.jsonl`` serves old summaries and
        dialogues under the gold ids and nothing downstream would notice. The
        fingerprint ``load_db`` recorded must be the files' own; the memory
        store is built from the files and needs no check.
        """

        if self.runtime.store_kind != "pg":
            return
        expected = db.dataset_fingerprint(dataset.load_people(), dataset.load_conversations())
        try:
            db.check_fingerprint(self.runtime.store.corpus_fingerprint(), expected)
        except db.StaleCorpusError as error:
            raise SystemExit(f"error: {error}") from error

    def environment(self) -> dict[str, Any]:
        """The hardware and model settings a measuring stage runs under, as the report states them.

        ``corpus_fingerprint`` says which corpus the store served (see
        ``db.dataset_fingerprint``): for a ``pg`` run the one its database
        recorded, which ``_check_corpus`` has matched to the files.
        """

        from .local_models import describe_hardware

        settings = self.runtime.settings
        jev = bool(settings.jev_url)  # a gateway without Jev has no model or scoring to state
        return {
            "store": self.runtime.store_kind,
            "corpus_fingerprint": self.runtime.store.corpus_fingerprint(),
            "models": {
                "gen": settings.gen_model,
                "summary": settings.summary_model,
                "system": settings.system_model,
                "judge": settings.judge_model,
                "jev": self.runtime.jev_model if jev else None,
                "jev_scoring": self.runtime.jev_scoring().label if jev else None,
                "embedding": config.EMBEDDING_MODEL,
                "reranker": config.RERANKER_MODEL,
            },
            "jev_pin": load_jev_pin(),
            "hardware": describe_hardware(),
        }

    def _record_environment(self) -> None:
        """Write ``meta.json`` before the run's first record, so the report can state what ran.

        Written once, when the directory holds no record yet: the hardware,
        the device and the Jev settings can all differ when the report is
        written later, on another machine or after the pin was rewritten.
        """

        path = self.run_dir / RUN_META
        if path.exists() or self.ranker_log or self.system_log:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def recorded_environment(self) -> dict[str, Any] | None:
        """What ``meta.json`` holds, or None when the run directory has none."""

        path = self.run_dir / RUN_META
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # Stage 2: the system eval --------------------------------------------- #
    def _cells(self, query: Query) -> list[tuple[str, int, int]]:
        """The (ranker, k, k') cells of ``query`` with no record yet."""

        return [
            (name, k, k_prime)
            for name in self.rankers
            for k in config.SYSTEM_KS
            for k_prime in config.FALLBACK_KS
            if (query.query_id, name, k, k_prime) not in self.system_log
        ]

    def run_system(self) -> None:
        """Every query through the real flow with each ranker at every (k, k')."""

        todo = [query for query in self.queries if self._cells(query)]
        per_query = len(self.rankers) * len(config.SYSTEM_KS) * len(config.FALLBACK_KS)
        self.say(
            f"[system] {len(todo)} of {len(self.queries)} queries to run ({per_query} cells each) "
            f"with {self.workers} workers"
        )
        if not todo:
            return
        self._check_corpus()
        self._record_environment()
        pipeline = SystemPipeline(
            self.runtime.store, self.runtime.llm, self.runtime.settings.system_model
        )
        self._system_rankers = {
            name: self._ranker(name, k=max(config.SYSTEM_KS), run=0) for name in self.rankers
        }
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._system_query, query, pipeline) for query in todo]
            try:
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    self.say(
                        f"[system] {done}/{len(todo)} queries; {len(self.system_log)} system "
                        f"records; spend ${self.spend:.4f}"
                    )
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    def _gold_start_times(self, query: Query) -> dict[str, datetime]:
        if not query.gold_episode_ids:
            return {}
        headers = self.runtime.store.get_headers(query.gold_episode_ids)
        return {episode_id: header.start_time for episode_id, header in headers.items()}

    def _ranking(self, query: Query, name: str, prepared: Prepared) -> RankResult:
        """The ranking for one (query, ranker); its trace carries the rank stage's latency.

        The ``rankers``-stage record is reused when the system applied the gold
        filters (same person, same padded range) and the record exists; else
        the ranker runs now, under the lock that keeps rankings sequential.
        Either way the latency is the ranker's own ``trace.latency_s``, the
        quantity the record stores.
        """

        if (prepared.person_id, prepared.time_range) == gold_filters(query):
            record = self.ranker_log.get((query.query_id, name, 0))
            if record is not None:
                return rank_result_from_record(record, max(config.SYSTEM_KS))
        ranker = self._system_rankers[name]
        with self._rank_lock:
            return ranker.rank_with_trace(
                config.OWNER_ID, query.query, prepared.person_id, prepared.time_range
            )

    def _system_query(self, query: Query, pipeline: SystemPipeline) -> int:
        """Prepare ``query`` once, then fill every missing cell; returns the records written."""

        cells = self._cells(query)
        if not cells:
            return 0
        prepared = pipeline.prepare(query.query)
        pool_size = gold_pool_size(self.runtime.store, query)
        start_times = self._gold_start_times(query)
        verdicts: dict[str, str | None] = {}

        def verdict_for(final_answer: str) -> str | None:
            # Every final answer of a query with a gold answer is judged,
            # declined and misrouted ones included; the same answer is judged once.
            # An answer the judge gives no usable verdict for, after its retries,
            # is recorded unjudged (verdict None) rather than stopping the stage;
            # the report counts those.
            if not query.gold_answer:
                return None
            if final_answer not in verdicts:
                try:
                    verdicts[final_answer] = judge.judge_answer(
                        query.query,
                        query.gold_answer,
                        final_answer,
                        self.runtime.llm,
                        self.runtime.settings.judge_model,
                    ).verdict
                except judge.JudgeOutputError as error:
                    self.say(f"[system] {query.query_id}: unjudged ({error})")
                    verdicts[final_answer] = None
            return verdicts[final_answer]

        def record(name: str, k: int, k_prime: int, result: SystemResult) -> None:
            verdict = verdict_for(result.final_answer)
            self.system_log.append(
                system_record(query, name, k, k_prime, result, pool_size, verdict, start_times)
            )

        if prepared.early_answer is not None:  # the same early result fills every cell
            result = compose_system_result(query.query, prepared)
            for name, k, k_prime in cells:
                record(name, k, k_prime, result)
            return len(cells)
        for name in self.rankers:
            ranker_cells = [(k, k_prime) for cell_name, k, k_prime in cells if cell_name == name]
            if not ranker_cells:
                continue
            rank_result = self._ranking(query, name, prepared)
            ranked_ids = [episode_id for episode_id, _ in rank_result.ranked]
            for k, k_prime in ranker_cells:
                answered = pipeline.answer(query.query, ranked_ids, k, k_prime)
                result = compose_system_result(query.query, prepared, rank_result, answered)
                record(name, k, k_prime, result)
        return len(cells)

    # Stage 3: the report -------------------------------------------------- #
    def _meta(
        self,
        ranker_records: Sequence[RankerRecord],
        system_records: Sequence[SystemRecord],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        """The report's Setup: the run's arguments, counts, and the environment the records were made in.

        ``models.jev`` is the id the run asked the gateway for; ``models.jev_snapshots``
        the dated snapshot ids the API said it served, read from the ranker records.
        """

        by_type = Counter(query.type for query in self.queries)
        models = dict(environment["models"])
        models["jev_snapshots"] = metrics.jev_snapshots(ranker_records)
        return {
            "date": self.date,
            "split": self.split,
            "seed": config.SEED,
            "store": environment["store"],
            "corpus_fingerprint": environment.get("corpus_fingerprint"),
            "limit": self.limit,
            "jev_runs": self.jev_runs,
            "workers": self.workers,
            "n_ranker_records": len(ranker_records),
            "n_system_records": len(system_records),
            "counts": {
                "queries": len(self.queries),
                "queries_by_type": {t: by_type[t] for t in QUERY_TYPES if by_type[t]},
            },
            "models": models,
            "hardware": dict(environment["hardware"]),
        }

    def _notes(
        self,
        left_out: tuple[int, int],
        environment_recorded: bool,
        jev_snapshots: Sequence[str],
        pinned_snapshots: Sequence[str],
    ) -> str:
        parts: list[str] = []
        if self.limit is not None:
            parts.append(
                f"Smoke run: only the first {len(self.queries)} queries of the {self.split} split "
                f"were used (`--limit {self.limit}`), so these numbers do not stand for the split."
            )
        if any(left_out):
            parts.append(
                f"The run directory also holds {left_out[0]} ranker records and {left_out[1]} "
                f"system records of queries outside this selection; they were left out."
            )
        if not environment_recorded:
            parts.append(
                "The hardware and model settings above were read when this report was written, "
                "not when the records were made: the run directory has no `meta.json`."
            )
        if len(jev_snapshots) > 1:
            parts.append(
                f"The Jev API served {len(jev_snapshots)} model snapshots during the ranker stage "
                f"({', '.join(jev_snapshots)}), so the Jev figures mix model versions; the variance "
                "section says which run hit which."
            )
        if jev_snapshots and pinned_snapshots and set(jev_snapshots) != set(pinned_snapshots):
            parts.append(
                f"The dev-set study that chose the Jev pin was served {', '.join(pinned_snapshots)}; "
                f"this run was served {', '.join(jev_snapshots)}, so its Jev figures come from a "
                "different model version than the one the pin was chosen on."
            )
        if self.notes_path is not None:
            parts.append(self.notes_path.read_text(encoding="utf-8").strip())
        return "\n\n".join(part for part in parts if part)

    def _records_of_current_queries(self) -> tuple[list[RankerRecord], list[SystemRecord], tuple[int, int]]:
        """The records of the queries this run covers, and how many others the logs hold.

        A ``--limit`` rerun on a directory with a full run reports the limited
        set only. A record whose gold ids differ from the query's cannot be
        aggregated: the dataset marker should have caught that, so it is an
        error, not a silent skip.
        """

        wanted = {query.query_id: query for query in self.queries}
        ranker_records = [record for record in self.ranker_log if record.query_id in wanted]
        system_records = [record for record in self.system_log if record.query_id in wanted]
        for record in (*ranker_records, *system_records):
            expected = list(wanted[record.query_id].gold_episode_ids)
            if list(record.gold_ids) != expected:
                raise ValueError(
                    f"{self.run_dir}: the record of {record.query_id!r} for {record.ranker} holds "
                    f"gold ids {list(record.gold_ids)} but the query now has {expected}; the run "
                    f"directory was recorded against another dataset"
                )
        left_out = (
            len(self.ranker_log) - len(ranker_records),
            len(self.system_log) - len(system_records),
        )
        return ranker_records, system_records, left_out

    def write_report(self) -> tuple[Path, Path]:
        """Aggregate the run's records and write ``<date>.md`` and ``<date>.json``."""

        ranker_records, system_records, left_out = self._records_of_current_queries()
        recorded = self.recorded_environment()
        environment = recorded if recorded is not None else self.environment()
        ranker_table = metrics.ranker_comparison(ranker_records, config.RECALL_KS)
        system_table = metrics.system_eval(system_records)
        meta = self._meta(ranker_records, system_records, environment)
        results: dict[str, Any] = {
            "meta": meta,
            "ranker_comparison": ranker_table,
            "jev_variance": metrics.jev_variance(ranker_records, config.RECALL_KS),
            "system_eval": system_table,
            "recommendation": metrics.recommend(ranker_table, system_table),
        }
        validation = judge_validation_path()
        if validation.exists():
            results["judge_validation"] = json.loads(validation.read_text(encoding="utf-8"))
        else:
            self.say(
                f"[report] no judge validation summary at {validation}; run "
                "`python -m eval.retrieval.judge_validation score` to include it"
            )
        # Without records there is nothing the live environment could misdescribe.
        has_records = bool(ranker_records or system_records)
        notes = self._notes(
            left_out,
            environment_recorded=recorded is not None or not has_records,
            jev_snapshots=meta["models"]["jev_snapshots"],
            pinned_snapshots=list((environment.get("jev_pin") or {}).get("served_models") or []),
        )
        if notes:
            results["notes"] = notes
        md_path, json_path = report.write_report(results, self.date, out_dir=config.RESULTS_DIR)
        self.say(
            f"[report] {len(ranker_records)} ranker records and {len(system_records)} system "
            f"records aggregated"
            + (f" ({left_out[0]} ranker and {left_out[1]} system records of other queries left out)" if any(left_out) else "")
        )
        self.say(f"[report] markdown: {md_path}")
        self.say(f"[report] json: {json_path}")
        return md_path, json_path


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _ranker_list(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in config.RANKER_NAMES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown rankers {unknown or [value]}; choose from {', '.join(config.RANKER_NAMES)}"
        )
    return names


def _date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a date as YYYY-MM-DD, got {value!r}") from None


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a number of at least 1, got {number}")
    return number


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.run",
        description="Run the retrieval ranker benchmark and write its report.",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test", help="which queries")
    parser.add_argument("--stage", choices=STAGE_CHOICES, default="all", help="what to run")
    parser.add_argument(
        "--rankers",
        type=_ranker_list,
        default=None,
        metavar="A,B,...",
        help=f"comma-separated subset of {', '.join(config.RANKER_NAMES)} to run (default: all "
        "the gateway serves, so no Jev ranker on EVAL_GATEWAY=openai; the report covers every "
        "record of the run)",
    )
    parser.add_argument("--store", choices=("pg", "memory"), default="pg", help="corpus store")
    parser.add_argument(
        "--date",
        type=_date,
        default=today(),
        metavar="YYYY-MM-DD",
        help="names the run directory and the report (default: today, UTC); a run started on "
        "another day resumes only with its own date",
    )
    parser.add_argument(
        "--jev-runs",
        type=_positive,
        default=config.JEV_VARIANCE_RUNS,
        metavar="N",
        help="repeated runs of the Jev rankers for the variance table",
    )
    parser.add_argument(
        "--limit",
        type=_positive,
        default=None,
        metavar="N",
        help="smoke run: only the first N queries of the split, in file order",
    )
    parser.add_argument(
        "--workers",
        type=_positive,
        default=None,
        metavar="N",
        help="threads over queries in the system stage (default: EVAL_LLM_CONCURRENCY)",
    )
    parser.add_argument(
        "--notes", type=Path, default=None, metavar="PATH", help="free text for the report's Notes"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="bypass the hosted-API response cache"
    )
    return parser.parse_args(argv)


def resume_command(args: argparse.Namespace) -> str:
    """The command that resumes this run: every argument spelled out, ``--date`` included.

    The default ``--date`` is today in UTC, so "the same command" after
    midnight UTC would open a new, empty run directory under new Jev salts.
    """

    parts = [
        "python -m eval.retrieval.run",
        f"--split {args.split}",
        f"--stage {args.stage}",
        f"--date {args.date}",
        f"--rankers {','.join(args.rankers)}",
        f"--store {args.store}",
        f"--jev-runs {args.jev_runs}",
    ]
    if args.limit is not None:
        parts.append(f"--limit {args.limit}")
    if args.workers is not None:
        parts.append(f"--workers {args.workers}")
    if args.notes is not None:
        parts.append(f"--notes {args.notes}")
    if args.no_cache:
        parts.append("--no-cache")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None, runtime: Runtime | None = None) -> int:
    args = parse_args(argv)
    if runtime is None:
        runtime = Runtime(store_kind=args.store, use_cache=not args.no_cache)
    settings = runtime.settings
    if args.rankers is None:
        args.rankers = list(settings.rankers)
    if not settings.jev_url and set(args.rankers) & set(config.JEV_RANKERS):
        raise SystemExit(f"error: {no_jev_message(settings.gateway)}")
    bench = Benchmark(
        runtime,
        split=args.split,
        date=args.date,
        rankers=args.rankers,
        jev_runs=args.jev_runs,
        limit=args.limit,
        workers=args.workers,
        notes=args.notes,
    )
    stages = STAGE_CHOICES[:3] if args.stage == "all" else (args.stage,)
    try:
        if "rankers" in stages:
            bench.run_rankers()
        if "system" in stages:
            bench.run_system()
        if "report" in stages:
            bench.write_report()
    except SpendLimitExceeded as error:
        _stopped(bench, f"stopped: {error}", resume_command(args))
        return 2
    except KeyboardInterrupt:
        _stopped(bench, "interrupted", resume_command(args))
        return 130
    finally:
        bench.close()
    bench.say(f"hosted spend this command: ${runtime.spend.total:.4f}")
    return 0


def _stopped(bench: Benchmark, why: str, command: str) -> None:
    print(
        f"{why}; {len(bench.ranker_log)} ranker records and {len(bench.system_log)} system records "
        f"are saved under {bench.run_dir}; hosted spend this command: ${bench.spend:.4f}; "
        f"to resume, rerun: {command}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
