"""Reusable pieces of a benchmark run: the record log, the ranking loop, run salts.

The runner (``run.py``), the label-pooling script and the Jev study all rank
queries under the gold filters and keep per-query records on disk between
invocations, so those pieces live here rather than in any one CLI.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Protocol, TextIO, TypeVar

from . import config
from .interfaces import CorpusStore, Ranker, RankResult
from .metrics import RankerRecord, SystemRecord
from .records import gold_filters, ranker_record
from .schema import Query

Key = tuple[Any, ...]


class JsonRecord(Protocol):
    def to_json(self) -> dict[str, Any]: ...


R = TypeVar("R", bound=JsonRecord)


class Stage1Shortfall(RuntimeError):
    """A ranker's first stage handed fewer candidates on than the pool allows.

    Rankers 2-3 score the bge-m3 top ``stage1_k`` (the whole pool when it is
    smaller). A store that returns fewer hits, for example because some
    conversations have no embedding, would not fail: it would quietly lower
    Recall and Stage 1 Recall for every query it touches. So the loop checks
    every trace and stops the run instead.
    """


def record_key(record: RankerRecord | SystemRecord) -> Key:
    """The identity of a record: what one query contributes once to an aggregate."""

    if isinstance(record, RankerRecord):
        return (record.query_id, record.ranker, record.run)
    if isinstance(record, SystemRecord):
        return (record.query_id, record.ranker, record.k, record.k_prime)
    raise TypeError(f"not a benchmark record: {type(record).__name__}")


def salt_for(date: str, split: str, run: int) -> str:
    """The cache salt of one Jev run.

    Distinct per date, split and run index, so every run of the variance study
    makes fresh Jev calls, and stable across invocations, so an interrupted run
    resumes from the cache instead of paying again.
    """

    if run < 0:
        raise ValueError(f"run index must be non-negative, got {run}")
    return f"{date}:{split}:run{run}"


def warm_up_salt(salt: str) -> str:
    """The salt of a Jev ranker's warm-up call: a cache key no recorded call shares.

    Warming up under the run's own salt would put the first query's answers in
    the cache, so its record would be a cache replay (near-zero latency, left
    out of the latency tables) instead of a timed call.
    """

    return f"{salt}:warmup"


class RecordLog(Generic[R]):
    """A JSONL file of records that a run appends to and a rerun picks up from.

    Records are written one per line and flushed as they complete, so an
    interrupted run keeps everything finished so far. On open, the existing
    lines are parsed with ``from_json`` and their keys are known, so a caller
    skips the work already done. A duplicate key is refused, because the
    aggregates count each (query, setting) once. A last line without its
    newline is a write that a crash cut short: it is dropped on open when it
    is not a whole record and kept when it is, and the file is terminated
    either way, so the next ``append`` starts a line of its own. ``append``
    is thread-safe; the file is opened for appending on the first ``append``.
    """

    def __init__(
        self,
        path: Path,
        from_json: Callable[[Mapping[str, Any]], R],
        key: Callable[[Any], Key] = record_key,
    ) -> None:
        self.path = Path(path)
        self._from_json = from_json
        self._key = key
        self._lock = threading.Lock()
        self._handle: TextIO | None = None
        self.records: list[R] = []
        self._by_key: dict[Key, R] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        text = self.path.read_text(encoding="utf-8")
        complete, _, tail = text.rpartition("\n")
        lines = complete.split("\n") if complete else []
        if tail.strip():
            try:
                json.loads(tail)
            except json.JSONDecodeError:  # a write that did not finish: drop it
                pass
            else:
                lines.append(tail)
            # Terminate the file: appending after an unterminated line would
            # glue two records on one line, which no later open could read.
            self.path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{self.path}:{number}: not a JSON record ({error.msg})") from None
            self._add(self._from_json(row))

    def _add(self, record: R) -> Key:
        key = self._key(record)
        if key in self._by_key:
            raise ValueError(f"{self.path}: duplicate record {key}")
        self._by_key[key] = record
        self.records.append(record)
        return key

    def append(self, record: R) -> None:
        """Write ``record`` as one line, flush it, and remember its key."""

        with self._lock:
            key = self._key(record)
            if key in self._by_key:
                raise ValueError(f"{self.path}: duplicate record {key}")
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8")
            line = json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True)
            self._handle.write(line + "\n")
            self._handle.flush()
            self._by_key[key] = record
            self.records.append(record)

    def get(self, key: Key) -> R | None:
        return self._by_key.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[R]:
        return iter(list(self.records))

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "RecordLog[R]":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def check_stage1(ranker: Ranker, result: RankResult) -> None:
    """Raise :class:`Stage1Shortfall` unless stage 1 handed on ``min(stage1_k, pool_size)`` ids.

    Rankers without a stage 1 (``stage1_ids`` is None) pass. ``stage1_k`` is
    the ranker's own cut-off when it has one, else ``config.STAGE1_TOP_K``.
    """

    stage1_ids = result.trace.stage1_ids
    if stage1_ids is None:
        return
    stage1_k = int(getattr(ranker, "stage1_k", config.STAGE1_TOP_K))
    expected = min(stage1_k, result.trace.pool_size)
    if len(stage1_ids) != expected:
        raise Stage1Shortfall(
            f"{result.trace.ranker}: stage 1 returned {len(stage1_ids)} candidates for a pool of "
            f"{result.trace.pool_size} (expected {expected}); the store returned too few hits"
        )


def run_ranker_over_queries(
    ranker: Ranker,
    queries: Sequence[Query],
    run: int = 0,
    warm_up: bool = True,
    on_record: Callable[[RankerRecord], None] | None = None,
    owner_id: str = config.OWNER_ID,
    now: datetime = config.NOW,
) -> list[RankerRecord]:
    """Rank every query under its gold filters, one at a time, and record each.

    ``queries`` must be answerable: a ranker record for any other query would
    pollute the aggregates, so one raises. Queries run sequentially, so the
    recorded latency is one ranking at a time (a Jev ranker's own concurrency
    is inside the call). With ``warm_up`` the first query is ranked once more
    before the loop and that result is discarded, so model loading and store
    warm-up are not charged to the first record; a Jev ranker is warmed up by
    the caller under :func:`warm_up_salt` instead, since under its own salt the
    warm-up's answers would make the first record a cache replay. ``on_record``
    is called with each record as it completes, so a caller can persist it
    before the next query runs. Every trace is checked with :func:`check_stage1`.
    """

    pending = list(queries)
    for query in pending:
        if not query.answerable:
            raise ValueError(
                f"{query.query_id} is a {query.type} query; "
                "ranker records are for answerable queries only"
            )
    records: list[RankerRecord] = []
    if not pending:
        return records
    if warm_up:
        first = pending[0]
        ranker.rank_with_trace(owner_id, first.query, *gold_filters(first, now))
    for query in pending:
        person_id, padded = gold_filters(query, now)
        result = ranker.rank_with_trace(owner_id, query.query, person_id, padded)
        check_stage1(ranker, result)
        record = ranker_record(query, result, run)
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def gold_pool_size(
    store: CorpusStore,
    query: Query,
    owner_id: str = config.OWNER_ID,
    now: datetime = config.NOW,
) -> int | None:
    """Conversations left under the gold filters; None for unknown-name queries (no pool)."""

    if query.type == "unknown_name":
        return None
    person_id, padded = gold_filters(query, now)
    return len(store.pool_ids(owner_id, person_id, padded))
