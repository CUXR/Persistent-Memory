"""The benchmark runner (``eval.retrieval.run``) and its reusable pieces (``runs``).

Everything runs offline on the integration fixture: its corpus in an
``InMemoryCorpusStore``, its fakes for the models and gateways, injected
through a ``Runtime`` whose attributes are assigned before first use. Cache,
results and dataset paths are redirected to a temporary directory.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_eval_integration import (
    ANSWERABLE,
    CONVERSATIONS,
    DANA,
    EXPECTED_PATH,
    OWNER,
    PEOPLE,
    QUERIES,
    SYSTEM_MODEL,
    FakeEmbedder,
    FakeJev,
    FakeLLM,
    FakeScorer,
    bag_of_words,
    make_store,
)
from test_eval_integration import settings as make_settings

from eval.retrieval import config, dataset, db, run, runs
from eval.retrieval.data.pool_labels import regenerate_command
from eval.retrieval import runtime as runtime_module
from eval.retrieval.interfaces import LLMResult, RankResult, Usage
from eval.retrieval.metrics import RankerRecord, SystemRecord
from eval.retrieval.runtime import Runtime
from eval.retrieval.schema import SYSTEM_STAGES
from eval.retrieval.spend import SpendLimitExceeded
from eval.retrieval.store import InMemoryCorpusStore
from eval.retrieval.system.pipeline import SystemPipeline, compose_system_result

DATE = "2026-11-06"
SPLIT = "test"
N_CELLS = len(config.SYSTEM_KS) * len(config.FALLBACK_KS)
EARLY_EXIT = ("q-priya", "q-sam")  # unknown name, ambiguous name: no ranking
NOT_ANSWERABLE_RANKED = ("q-car", "q-yesterday")  # ranked by the system, no ranker record


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #
class StoreProxy:
    """Delegates to the fixture store; subclasses alter one method."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class ShortStore(StoreProxy):
    """A store whose vector search returns one hit however large the pool: the stage-1 defect."""

    def vector_search(self, *args: Any, **kwargs: Any) -> list[tuple[str, float]]:
        return self._store.vector_search(*args, **kwargs)[:1]


class InterruptingStore(StoreProxy):
    """Raises KeyboardInterrupt on the vector search after ``after`` calls."""

    def __init__(self, store: Any, after: int) -> None:
        super().__init__(store)
        self.after = after
        self.calls = 0

    def vector_search(self, *args: Any, **kwargs: Any) -> list[tuple[str, float]]:
        self.calls += 1
        if self.calls > self.after:
            raise KeyboardInterrupt
        return self._store.vector_search(*args, **kwargs)


class MisnamingLLM(FakeLLM):
    """The fixture model, except that it extracts "Dana" from the climbing-gym query.

    That query names nobody, so the system's person filter then differs from
    the gold filter and the runner must rank it afresh.
    """

    def complete_json(self, *, model: str, system: str, user: str, **kwargs: Any) -> LLMResult:
        question = user.rsplit("Question: ", 1)[-1].strip()
        if system.startswith("You extract search filters") and question == "Who told me about a climbing gym?":
            self.calls.append((kwargs.get("namespace", "default"), system, user))
            data = {"name": "Dana", "time_range": None}
            return LLMResult(data=data, usage=Usage(calls=1), model=model, raw_text=json.dumps(data))
        return super().complete_json(model=model, system=system, user=user, **kwargs)


class UngradableLLM(FakeLLM):
    """The fixture model, except that the judge never gives a usable verdict for the budget query."""

    def complete_json(self, *, model: str, system: str, user: str, **kwargs: Any) -> LLMResult:
        if system.startswith("You grade answers") and "Q4 budget" in user.split("</question>", 1)[0]:
            self.calls.append((kwargs.get("namespace", "default"), system, user))
            data = {"verdict": "unsure", "reason": "not a verdict the rubric allows"}
            return LLMResult(data=data, usage=Usage(calls=1), model=model, raw_text=json.dumps(data))
        return super().complete_json(model=model, system=system, user=user, **kwargs)


class CappedLLM(FakeLLM):
    """The fixture model with a spend cap: the call after ``limit`` calls raises."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def complete_json(self, **kwargs: Any) -> LLMResult:
        if len(self.calls) >= self.limit:
            raise SpendLimitExceeded("hosted-API spend passed the cap")
        return super().complete_json(**kwargs)


def count_rank_calls(live: Runtime) -> Counter[str]:
    """Count ``rank_with_trace`` calls per ranker name on the rankers ``live`` builds."""

    calls: Counter[str] = Counter()
    build = live.ranker

    def ranker(name: str, **kwargs: Any) -> Any:
        built = build(name, **kwargs)
        inner = built.rank_with_trace

        def counted(*args: Any, **kw: Any) -> Any:
            calls[name] += 1
            return inner(*args, **kw)

        built.rank_with_trace = counted
        return built

    live.ranker = ranker  # type: ignore[method-assign]
    return calls


def pin_rank_latency(live: Runtime, latency_s: float) -> None:
    """Make every trace of the rankers ``live`` builds report ``latency_s``, whatever the clock says."""

    build = live.ranker

    def ranker(name: str, **kwargs: Any) -> Any:
        built = build(name, **kwargs)
        inner = built.rank_with_trace

        def pinned(*args: Any, **kw: Any) -> RankResult:
            result = inner(*args, **kw)
            result.trace.latency_s = latency_s
            return result

        built.rank_with_trace = pinned
        return built

    live.ranker = ranker  # type: ignore[method-assign]


def make_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM | None = None, store: Any = None
) -> Runtime:
    """A ``Runtime`` over the fixture, with every path the runner touches under ``tmp_path``."""

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    queries_path = tmp_path / "data" / "queries.jsonl"
    dataset.save_queries(QUERIES, queries_path)
    monkeypatch.setattr(config, "QUERIES_PATH", queries_path)
    monkeypatch.setattr(config, "JUDGE_VALIDATION_PATH", tmp_path / "data" / "judge_validation.json")
    monkeypatch.setattr(runtime_module, "JEV_SCORING_PATH", tmp_path / "jev_scoring.json")
    monkeypatch.delenv("EVAL_JEV_MODEL", raising=False)
    live = Runtime(make_settings(llm_concurrency=2), store_kind="memory", use_cache=False)
    live.store = store if store is not None else make_store()
    live.llm = llm if llm is not None else FakeLLM()
    live.jev = FakeJev()
    live.embedder = FakeEmbedder()
    live.scorer = FakeScorer()
    return live


def main(live: Runtime, *args: str) -> int:
    return run.main(["--split", SPLIT, "--date", DATE, *args], runtime=live)


def read_records(path: Path, from_json: Callable[[dict[str, Any]], Any]) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        return [from_json(json.loads(line)) for line in handle if line.strip()]


def ranker_records() -> list[RankerRecord]:
    return read_records(run.run_dir(DATE, SPLIT) / run.RANKER_RECORDS, RankerRecord.from_json)


def system_records() -> list[SystemRecord]:
    return read_records(run.run_dir(DATE, SPLIT) / run.SYSTEM_RECORDS, SystemRecord.from_json)


def expected_ranker_keys(rankers: Sequence[str], jev_runs: int) -> set[tuple[str, str, int]]:
    return {
        (query.query_id, name, run)
        for query in ANSWERABLE
        for name in rankers
        for run in range(jev_runs if name in config.JEV_RANKERS else 1)
    }


def expected_system_keys(rankers: Sequence[str], queries: Sequence[Any] = QUERIES) -> set[tuple]:
    return {
        (query.query_id, name, k, k_prime)
        for query in queries
        for name in rankers
        for k in config.SYSTEM_KS
        for k_prime in config.FALLBACK_KS
    }


# --------------------------------------------------------------------------- #
# runs.py                                                                      #
# --------------------------------------------------------------------------- #
def test_salt_is_distinct_per_run_and_stable() -> None:
    salts = {runs.salt_for(DATE, SPLIT, r) for r in range(3)}
    assert len(salts) == 3
    assert runs.salt_for(DATE, SPLIT, 1) == runs.salt_for(DATE, SPLIT, 1)
    assert runs.salt_for(DATE, "dev", 1) != runs.salt_for(DATE, SPLIT, 1)
    assert runs.salt_for("2026-11-07", SPLIT, 1) != runs.salt_for(DATE, SPLIT, 1)
    with pytest.raises(ValueError):
        runs.salt_for(DATE, SPLIT, -1)


def test_gold_pool_size_is_the_gold_filter_pool_and_none_for_unknown_names() -> None:
    store = make_store()
    by_id = {query.query_id: query for query in QUERIES}
    assert runs.gold_pool_size(store, by_id["q-priya"]) is None
    assert runs.gold_pool_size(store, by_id["q-budget"]) == len(store.pool_ids(OWNER, DANA.person_id))
    assert runs.gold_pool_size(store, by_id["q-gym"]) == len(store.pool_ids(OWNER))
    assert runs.gold_pool_size(store, by_id["q-yesterday"]) == 0


def test_run_ranker_over_queries_records_each_query_and_discards_the_warm_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    calls = count_rank_calls(live)
    seen: list[RankerRecord] = []

    records = runs.run_ranker_over_queries(
        live.ranker("bge_m3_reranker", k=30), ANSWERABLE, run=2, on_record=seen.append
    )

    assert [r.query_id for r in records] == [q.query_id for q in ANSWERABLE]
    assert records == seen and {r.run for r in records} == {2}
    assert calls["bge_m3_reranker"] == len(ANSWERABLE) + 1  # one warm-up call, not recorded
    assert all(r.stage1_ids is not None and r.gold_ids for r in records)
    assert runs.run_ranker_over_queries(live.ranker("bge_m3", k=30), [], warm_up=True) == []
    assert calls["bge_m3"] == 0
    runs.run_ranker_over_queries(live.ranker("bge_m3", k=30), ANSWERABLE[:1], warm_up=False)
    assert calls["bge_m3"] == 1
    unanswerable = next(query for query in QUERIES if not query.answerable)
    with pytest.raises(ValueError, match="answerable"):
        runs.run_ranker_over_queries(live.ranker("bge_m3", k=30), [unanswerable])


def test_stage1_check_fires_on_a_store_that_returns_too_few_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch, store=ShortStore(make_store()))
    budget = next(query for query in QUERIES if query.query_id == "q-budget")

    with pytest.raises(runs.Stage1Shortfall, match="stage 1 returned 1 candidates"):
        runs.run_ranker_over_queries(live.ranker("bge_m3_reranker", k=30), [budget])
    # Rankers without a stage 1 are not checked; a store that fills stage 1 passes.
    assert len(runs.run_ranker_over_queries(live.ranker("bge_m3", k=30), [budget])) == 1
    live.store = make_store()
    assert len(runs.run_ranker_over_queries(live.ranker("bge_m3_jev", k=30), [budget])) == 1


def test_record_log_resumes_refuses_duplicates_and_drops_a_partial_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    records = runs.run_ranker_over_queries(live.ranker("bge_m3", k=30), ANSWERABLE, warm_up=False)
    path = tmp_path / "log" / "ranker_records.jsonl"

    log: runs.RecordLog[RankerRecord] = runs.RecordLog(path, RankerRecord.from_json)
    assert len(log) == 0 and not path.exists()
    for record in records[:2]:
        log.append(record)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2  # flushed per record
    with pytest.raises(ValueError, match="duplicate"):
        log.append(records[0])
    log.close()

    with path.open("a", encoding="utf-8") as handle:  # a write that never finished
        handle.write('{"query_id": "q-partial", "ran')
    reopened: runs.RecordLog[RankerRecord] = runs.RecordLog(path, RankerRecord.from_json)
    assert [r.query_id for r in reopened] == [r.query_id for r in records[:2]]
    assert {runs.record_key(r) for r in reopened} == {runs.record_key(r) for r in records[:2]}
    assert runs.record_key(records[0]) in reopened and reopened.get(runs.record_key(records[1])) is not None
    reopened.append(records[2])
    reopened.close()
    assert [r.query_id for r in runs.RecordLog(path, RankerRecord.from_json)] == [
        r.query_id for r in records[:3]
    ]

    # A whole last record whose newline never made it to disk is kept, and the
    # file is terminated on open, so the next append does not share its line.
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    unterminated: runs.RecordLog[RankerRecord] = runs.RecordLog(path, RankerRecord.from_json)
    assert [r.query_id for r in unterminated] == [r.query_id for r in records[:3]]
    assert path.read_text(encoding="utf-8").endswith("\n")
    unterminated.append(records[3])
    unterminated.close()
    assert [r.query_id for r in runs.RecordLog(path, RankerRecord.from_json)] == [
        r.query_id for r in records[:4]
    ]

    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1: not a JSON record"):
        runs.RecordLog(path, RankerRecord.from_json)
    with pytest.raises(TypeError):
        runs.record_key("record")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Composing system results                                                     #
# --------------------------------------------------------------------------- #
def test_compose_system_result_matches_pipeline_run() -> None:
    store, llm = make_store(), FakeLLM()
    pipeline = SystemPipeline(store, llm, SYSTEM_MODEL)
    live = Runtime(make_settings(), store_kind="memory", use_cache=False)
    live.store, live.embedder, live.scorer, live.jev = store, FakeEmbedder(), FakeScorer(), FakeJev()
    ranker = live.ranker("bge_m3", k=20)
    by_id = {query.query_id: query for query in QUERIES}

    for query_id in ("q-multi", "q-pottery", "q-car", "q-yesterday"):
        text = by_id[query_id].query
        whole = pipeline.run(text, ranker, 10, 5)
        prepared = pipeline.prepare(text)
        rank = ranker.rank_with_trace(OWNER, text, prepared.person_id, prepared.time_range)
        rank.trace.latency_s = 0.25  # the ranker's own timing is the rank stage, whatever the clock says
        answered = pipeline.answer(text, [episode_id for episode_id, _ in rank.ranked], 10, 5)
        composed = compose_system_result(text, prepared, rank, answered)
        assert (composed.path, composed.final_answer) == (whole.path, whole.final_answer)
        assert composed.usage_by_stage == whole.usage_by_stage
        assert set(composed.latency_by_stage) == set(SYSTEM_STAGES)
        assert composed.latency_by_stage["rank"] == 0.25
        assert whole.latency_by_stage["rank"] == whole.rank.trace.latency_s
        assert composed.latency_s == pytest.approx(sum(composed.latency_by_stage.values()))
        assert composed.usage == sum(composed.usage_by_stage.values(), Usage())

    early = pipeline.prepare(by_id["q-priya"].query)
    composed = compose_system_result(by_id["q-priya"].query, early)
    assert composed.path == "unknown_name" and composed.rank is None and composed.answered is None
    assert composed.usage_by_stage["rank"] == Usage() and composed.latency_by_stage["rank"] == 0.0
    with pytest.raises(ValueError, match="go together"):
        compose_system_result(text, prepared, rank, None)
    with pytest.raises(ValueError, match="did not stop early"):
        compose_system_result(text, prepared)


def test_rank_result_from_record_cuts_the_stored_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    record = runs.run_ranker_over_queries(live.ranker("bge_m3_reranker", k=30), ANSWERABLE[:1], warm_up=False)[0]

    result = run.rank_result_from_record(record, 3)

    assert result.ranked == list(zip(record.ranked_ids, record.scores))[:3]
    assert result.trace.ranker == "bge_m3_reranker" and result.trace.pool_size == record.pool_size
    assert result.trace.stage1_ids == record.stage1_ids and result.trace.latency_s == record.latency_s


# --------------------------------------------------------------------------- #
# The rankers stage                                                            #
# --------------------------------------------------------------------------- #
def test_rankers_stage_records_every_answerable_query_per_ranker_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    calls = count_rank_calls(live)

    assert main(live, "--stage", "rankers", "--jev-runs", "2") == 0

    records = ranker_records()
    assert {runs.record_key(r) for r in records} == expected_ranker_keys(config.RANKER_NAMES, 2)
    assert len(records) == len({runs.record_key(r) for r in records})
    # One discarded warm-up call per (ranker, run), on top of one call per query.
    assert calls == {
        "bge_m3": len(ANSWERABLE) + 1,
        "bge_m3_reranker": len(ANSWERABLE) + 1,
        "bge_m3_jev": 2 * (len(ANSWERABLE) + 1),
        "jev_full_pool": 2 * (len(ANSWERABLE) + 1),
    }
    run_salts = {runs.salt_for(DATE, SPLIT, 0), runs.salt_for(DATE, SPLIT, 1)}
    salts = {salt for _, _, salt in live.jev.requests}
    assert salts == run_salts | {runs.warm_up_salt(salt) for salt in run_salts}
    assert all(r.ranked_ids and len(r.ranked_ids) <= max(config.RECALL_KS) for r in records)
    out = capsys.readouterr().out
    assert "[rankers] jev_full_pool run 1" in out and "hosted spend this command" in out
    logs = run.run_dir(DATE, SPLIT)
    assert (logs / run.DATASET_MARKER).read_text(encoding="utf-8").strip() == run.dataset_digest()
    assert json.loads((logs / run.RUN_META).read_text(encoding="utf-8"))["models"]["jev"] == live.jev_model


def test_a_gateway_without_jev_runs_the_local_rankers_and_refuses_the_jev_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # EVAL_GATEWAY=openai serves no Jev: an omitted --rankers means the two local
    # rankers, and naming a Jev ranker stops the command before it does any work.
    live = make_runtime(tmp_path, monkeypatch)
    live.settings = replace(live.settings, gateway="openai", jev_url="", jev_api_key=None, jev_model="")

    with pytest.raises(SystemExit, match="Jev is not served on the openai gateway"):
        main(live, "--stage", "rankers", "--rankers", "bge_m3,bge_m3_jev")
    assert not run.run_dir(DATE, SPLIT).exists()

    assert main(live, "--stage", "rankers") == 0
    assert {runs.record_key(r) for r in ranker_records()} == expected_ranker_keys(config.LOCAL_RANKERS, 1)
    assert live.jev.requests == [] and "jev" not in capsys.readouterr().out
    meta = json.loads((run.run_dir(DATE, SPLIT) / run.RUN_META).read_text(encoding="utf-8"))
    assert (meta["models"]["jev"], meta["models"]["jev_scoring"]) == (None, None)
    assert main(live, "--stage", "report") == 0  # the report states a run without Jev


def test_jev_rankers_warm_up_under_a_throwaway_salt_so_the_first_record_is_a_real_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    first = ANSWERABLE[0]

    assert main(live, "--stage", "rankers", "--rankers", "jev_full_pool", "--jev-runs", "1") == 0

    salt = runs.salt_for(DATE, SPLIT, 0)
    by_salt: dict[str, list[Any]] = {}
    for state, _, request_salt in live.jev.requests:
        by_salt.setdefault(request_salt, []).append(state["query"])
    # The warm-up asks about the first query only, under its own salt; the run's
    # salt then sees every query once, the first included, so no record is a replay.
    assert set(by_salt) == {runs.warm_up_salt(salt), salt}
    assert set(by_salt[runs.warm_up_salt(salt)]) == {first.query}
    assert set(by_salt[salt]) == {query.query for query in ANSWERABLE}
    assert live.jev.requests[0][2] == runs.warm_up_salt(salt)
    assert all(not record.extra.get("n_cached") for record in ranker_records())


def test_rankers_stage_resumes_from_the_log_without_new_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    calls = count_rank_calls(live)
    assert main(live, "--stage", "rankers", "--jev-runs", "2") == 0
    before = (dict(calls), len(live.jev.requests), ranker_records())

    assert main(live, "--stage", "rankers", "--jev-runs", "2") == 0

    assert (dict(calls), len(live.jev.requests), ranker_records()) == before
    # Drop one ranker's records: only those are redone (with one new warm-up call).
    path = run.run_dir(DATE, SPLIT) / run.RANKER_RECORDS
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if '"ranker": "bge_m3"' not in line]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert main(live, "--stage", "rankers", "--jev-runs", "2") == 0
    assert calls["bge_m3"] == before[0]["bge_m3"] + len(ANSWERABLE) + 1
    assert {name: n for name, n in calls.items() if name != "bge_m3"} == {
        name: n for name, n in before[0].items() if name != "bge_m3"
    }
    assert {runs.record_key(r) for r in ranker_records()} == expected_ranker_keys(config.RANKER_NAMES, 2)


# --------------------------------------------------------------------------- #
# The system stage                                                             #
# --------------------------------------------------------------------------- #
def test_system_stage_fills_every_cell_and_early_exits_fill_identical_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    rankers = ("bge_m3", "jev_full_pool")

    assert main(live, "--stage", "system", "--rankers", ",".join(rankers), "--workers", "2") == 0

    records = system_records()
    assert {runs.record_key(r) for r in records} == expected_system_keys(rankers)
    assert len(records) == len(QUERIES) * len(rankers) * N_CELLS
    by_query: dict[str, list[SystemRecord]] = {}
    for record in records:
        by_query.setdefault(record.query_id, []).append(record)
    for query in QUERIES:
        rows = by_query[query.query_id]
        assert {row.path for row in rows} == {EXPECTED_PATH[query.query_id]}, query.query_id
        assert all(set(row.usage_by_stage) == set(SYSTEM_STAGES) for row in rows)
        if query.gold_answer:
            assert all(row.verdict is not None for row in rows), query.query_id
        else:
            assert all(row.verdict is None for row in rows), query.query_id
        if query.type == "unknown_name":
            assert all(row.pool_size is None and row.unknown_name_exact for row in rows)
        else:
            assert all(isinstance(row.pool_size, int) for row in rows)
    for query_id in EARLY_EXIT:
        cells = {
            (row.path, row.final_answer, row.verdict, row.retrieved_ids == [], row.needed_dialogue,
             json.dumps({s: u.to_json() for s, u in row.usage_by_stage.items()}, sort_keys=True),
             json.dumps(row.latency_by_stage, sort_keys=True))
            for row in by_query[query_id]
        }
        assert len(cells) == 1, query_id
        assert by_query[query_id][0].usage_by_stage["rank"] == Usage()
    assert {row.verdict for row in by_query["q-budget"]} == {"correct"}
    assert {row.verdict for row in by_query["q-sam"]} == {"incorrect"}  # the clarification is graded
    assert all(row.needed_dialogue for row in by_query["q-pottery"])
    assert {row.k for row in records} == set(config.SYSTEM_KS)
    assert {row.k_prime for row in records} == set(config.FALLBACK_KS)


def test_an_answer_the_judge_cannot_grade_is_recorded_unjudged_and_the_stage_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = make_runtime(tmp_path, monkeypatch, llm=UngradableLLM())

    assert main(live, "--stage", "system", "--rankers", "bge_m3", "--workers", "2") == 0
    assert main(live, "--stage", "report", "--rankers", "bge_m3") == 0

    records = system_records()
    assert {runs.record_key(r) for r in records} == expected_system_keys(["bge_m3"])  # nothing was skipped
    budget = [row for row in records if row.query_id == "q-budget"]
    assert budget and all(row.verdict is None for row in budget)
    answerable = {query.query_id for query in ANSWERABLE}
    others = [row for row in records if row.query_id != "q-budget" and row.query_id in answerable]
    assert others and all(row.verdict is not None for row in others)
    out = capsys.readouterr().out
    assert "[system] q-budget: unjudged (" in out
    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["recommendation"]["failure_modes"]["unjudged"] == 1
    for k in config.SYSTEM_KS:
        for k_prime in config.FALLBACK_KS:
            assert results["system_eval"]["bge_m3"][str(k)][str(k_prime)]["failures"]["unjudged"] == 1
    markdown = (tmp_path / "results" / f"{DATE}.md").read_text(encoding="utf-8")
    assert "| Answer not graded (the judge gave no usable verdict) | 1 |" in markdown


def test_system_stage_ranks_every_query_that_is_not_an_early_exit_without_the_rankers_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    calls = count_rank_calls(live)

    assert main(live, "--stage", "system", "--rankers", "bge_m3_reranker,bge_m3_jev", "--workers", "1") == 0

    ranked = len(QUERIES) - len(EARLY_EXIT)
    assert calls == {"bge_m3_reranker": ranked, "bge_m3_jev": ranked}
    assert {salt for _, _, salt in live.jev.requests} == {runs.salt_for(DATE, SPLIT, 0)}
    assert not (run.run_dir(DATE, SPLIT) / run.RANKER_RECORDS).exists()


def test_system_stage_reuses_gold_filter_rankings_only_when_the_filters_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch, llm=MisnamingLLM())
    calls = count_rank_calls(live)
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3") == 0
    after_rankers = calls["bge_m3"]

    assert main(live, "--stage", "system", "--rankers", "bge_m3", "--workers", "2") == 0

    # Ranked now: the two queries without a ranker record, and the one whose
    # extracted person differs from the gold filters. The rest are reused.
    assert calls["bge_m3"] == after_rankers + len(NOT_ANSWERABLE_RANKED) + 1
    stored = {record.query_id: record for record in ranker_records()}
    rows = system_records()
    reused = [row for row in rows if row.query_id in stored and row.query_id not in (*EARLY_EXIT, "q-gym")]
    assert {row.query_id for row in reused} == {q.query_id for q in ANSWERABLE} - {*EARLY_EXIT, "q-gym"}
    for row in reused:
        assert row.latency_by_stage["rank"] == stored[row.query_id].latency_s
        assert row.retrieved_ids == stored[row.query_id].ranked_ids[: row.k]
    gym = [row for row in rows if row.query_id == "q-gym"]
    assert all(not row.person_correct and row.extracted_name == "Dana" for row in gym)
    assert all(row.path != "unknown_name" for row in gym)


def test_system_stage_resumes_without_new_llm_or_ranker_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    calls = count_rank_calls(live)
    assert main(live, "--stage", "system", "--rankers", "bge_m3,jev_full_pool") == 0
    before = (len(live.llm.calls), len(live.jev.requests), dict(calls), len(system_records()))

    assert main(live, "--stage", "system", "--rankers", "bge_m3,jev_full_pool") == 0

    assert (len(live.llm.calls), len(live.jev.requests), dict(calls), len(system_records())) == before
    # Drop the cells of one query: only that query is redone.
    path = run.run_dir(DATE, SPLIT) / run.SYSTEM_RECORDS
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if '"query_id": "q-budget"' not in line]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert main(live, "--stage", "system", "--rankers", "bge_m3,jev_full_pool") == 0
    assert len(system_records()) == before[3]
    new_calls = live.llm.calls[before[0]:]
    extractions = [user for _, system, user in new_calls if system.startswith("You extract search filters")]
    assert len(extractions) == 1 and extractions[0].endswith("What did Dana say about the Q4 budget?")
    assert all("What did Dana say about the Q4 budget?" in user for _, _, user in new_calls)
    assert dict(calls) == {"bge_m3": before[2]["bge_m3"] + 1, "jev_full_pool": before[2]["jev_full_pool"] + 1}


# --------------------------------------------------------------------------- #
# Limits, the report and the whole run                                         #
# --------------------------------------------------------------------------- #
def test_limit_keeps_the_first_queries_and_the_report_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    first = QUERIES[:3]

    assert main(live, "--rankers", "bge_m3", "--limit", "3") == 0

    assert {runs.record_key(r) for r in ranker_records()} == {
        (query.query_id, "bge_m3", 0) for query in first if query.answerable
    }
    assert {runs.record_key(r) for r in system_records()} == expected_system_keys(["bge_m3"], first)
    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["limit"] == 3 and results["meta"]["counts"]["queries"] == 3
    assert results["meta"]["counts"]["queries_by_type"] == {"lookup": 1, "detail_only": 1, "unanswerable": 1}
    markdown = (tmp_path / "results" / f"{DATE}.md").read_text(encoding="utf-8")
    assert "Smoke run: only the first 3 queries" in markdown


def test_all_stages_write_a_report_that_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    validation = {"n": 2, "agreement_rate": 1.0, "disagreements": []}
    run.judge_validation_path().write_text(json.dumps(validation), encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_text("Dev queries were read by hand before this run.\n", encoding="utf-8")
    rankers = ("bge_m3_reranker", "jev_full_pool")

    assert main(live, "--rankers", ",".join(rankers), "--jev-runs", "2", "--notes", str(notes)) == 0

    md_path, json_path = tmp_path / "results" / f"{DATE}.md", tmp_path / "results" / f"{DATE}.json"
    assert md_path.exists() and json_path.exists()
    results = json.loads(json_path.read_text(encoding="utf-8"))
    meta = results["meta"]
    assert (meta["date"], meta["split"], meta["seed"], meta["store"]) == (DATE, SPLIT, config.SEED, "memory")
    assert (meta["limit"], meta["jev_runs"], meta["workers"]) == (None, 2, 2)
    assert meta["n_ranker_records"] == len(expected_ranker_keys(rankers, 2))
    assert meta["n_system_records"] == len(QUERIES) * len(rankers) * N_CELLS
    assert meta["counts"]["queries"] == len(QUERIES)
    assert meta["models"] == {
        "gen": "fake/gen",
        "summary": "fake/summary",
        "system": SYSTEM_MODEL,
        "judge": live.settings.judge_model,
        "jev": live.jev_model,
        "jev_snapshots": [live.jev.model],  # what the API said it served, from the ranker records
        "jev_scoring": live.jev_scoring().label,
        "embedding": config.EMBEDDING_MODEL,
        "reranker": config.RERANKER_MODEL,
    }
    assert "machine" in meta["hardware"] and "cpu_count" in meta["hardware"]
    assert set(results["ranker_comparison"]) == set(rankers)
    assert set(results["jev_variance"]) == {"jev_full_pool"}
    assert set(results["system_eval"]) == set(rankers)
    assert results["recommendation"]["ranker"]["choice"] in rankers
    assert results["judge_validation"] == validation
    assert "Dev queries were read by hand" in results["notes"]
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Retrieval ranker benchmark") and "## Notes" in markdown
    out = capsys.readouterr().out
    assert str(md_path) in out and str(json_path) in out and "hosted spend this command: $" in out
    # The report stage alone re-renders the same aggregates from the logs.
    assert main(live, "--stage", "report", "--jev-runs", "2") == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["ranker_comparison"] == results["ranker_comparison"]


def test_report_stage_without_records_still_writes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = make_runtime(tmp_path, monkeypatch)

    assert main(live, "--stage", "report") == 0

    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["ranker_comparison"] == {} and results["system_eval"] == {}
    assert results["meta"]["n_ranker_records"] == 0 and results["meta"]["n_system_records"] == 0
    assert "judge_validation" not in results and "notes" not in results
    # The report says where it looked for the judge-validation summary and how to make one.
    out = capsys.readouterr().out
    assert f"[report] no judge validation summary at {run.judge_validation_path()}" in out
    assert "judge_validation score" in out
    assert run.judge_validation_path() == config.JUDGE_VALIDATION_PATH


def test_report_names_every_jev_snapshot_the_api_served_and_warns_when_they_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requested id is an alias; the report states the dated snapshots that answered,
    read from the ranker records, and warns when the runs did not all hit the same one."""

    live = make_runtime(tmp_path, monkeypatch)
    first = live.jev.model
    assert main(live, "--stage", "rankers", "--rankers", "jev_full_pool", "--jev-runs", "1") == 0
    live.jev.model = second = "typesafe/jev-1.14-20261101"  # the API rolls to a new snapshot
    assert main(live, "--stage", "rankers", "--rankers", "jev_full_pool", "--jev-runs", "2") == 0

    assert main(live, "--stage", "report", "--rankers", "jev_full_pool", "--jev-runs", "2") == 0

    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["models"]["jev"] == live.jev_model  # the alias the run asked for
    assert results["meta"]["models"]["jev_snapshots"] == sorted([first, second])
    assert results["jev_variance"]["jev_full_pool"]["snapshots_per_run"] == [[first], [second]]
    assert "served 2 model snapshots during the ranker stage" in results["notes"]
    assert first in results["notes"] and second in results["notes"]
    markdown = (tmp_path / "results" / f"{DATE}.md").read_text(encoding="utf-8")
    assert "**Warning:** 2 Jev snapshots served this run" in markdown
    assert "mix model versions" in markdown


def test_report_notes_when_the_run_was_served_another_snapshot_than_the_pins_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    pin = {"model": live.jev_model, "gateway": "openrouter", "served_models": ["typesafe/jev-1.14-20260901"]}
    (tmp_path / "jev_scoring.json").write_text(json.dumps(pin), encoding="utf-8")

    assert main(live, "--stage", "rankers", "--rankers", "jev_full_pool", "--jev-runs", "1") == 0
    assert main(live, "--stage", "report", "--rankers", "jev_full_pool", "--jev-runs", "1") == 0

    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["models"]["jev_snapshots"] == [live.jev.model]
    note = results["notes"]
    assert "study that chose the Jev pin was served typesafe/jev-1.14-20260901" in note
    assert f"this run was served {live.jev.model}" in note
    assert "served 2 model snapshots" not in note  # one snapshot served this run: no mixing

    # The pin that names the run's own snapshot draws no note.
    pin["served_models"] = [live.jev.model]
    (tmp_path / "jev_scoring.json").write_text(json.dumps(pin), encoding="utf-8")
    (run.run_dir(DATE, SPLIT) / run.RUN_META).unlink()  # re-read the pin at report time
    assert main(live, "--stage", "report", "--rankers", "jev_full_pool", "--jev-runs", "1") == 0
    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert "Jev pin" not in results.get("notes", "")


# --------------------------------------------------------------------------- #
# Stopping cleanly                                                             #
# --------------------------------------------------------------------------- #
def test_spend_cap_stops_the_system_stage_with_records_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = make_runtime(tmp_path, monkeypatch, llm=CappedLLM(limit=5))

    assert main(live, "--stage", "system", "--rankers", "bge_m3", "--workers", "1") == 2

    saved = system_records()
    assert 0 < len(saved) < len(QUERIES) * N_CELLS
    err = capsys.readouterr().err
    assert "stopped: hosted-API spend passed the cap" in err
    assert (
        "to resume, rerun: python -m eval.retrieval.run --split test --stage system "
        f"--date {DATE} --rankers bge_m3 --store pg --jev-runs {config.JEV_VARIANCE_RUNS} --workers 1"
    ) in err
    # A rerun with the cap lifted finishes the grid from where it stopped.
    live.llm = FakeLLM()
    assert main(live, "--stage", "system", "--rankers", "bge_m3", "--workers", "1") == 0
    assert {runs.record_key(r) for r in system_records()} == expected_system_keys(["bge_m3"])


def test_keyboard_interrupt_stops_the_rankers_stage_with_records_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InterruptingStore(make_store(), after=3)  # the warm-up, two queries, then Ctrl-C
    live = make_runtime(tmp_path, monkeypatch, store=store)

    assert main(live, "--stage", "rankers", "--rankers", "bge_m3") == 130

    assert [r.query_id for r in ranker_records()] == [q.query_id for q in ANSWERABLE[:2]]
    err = capsys.readouterr().err
    assert "interrupted" in err and f"--date {DATE}" in err


def test_resume_command_spells_out_the_defaulted_date_and_every_flag() -> None:
    # Without --date the run is keyed by today (UTC); the hint must name it, or
    # the "same command" after midnight UTC would open a new, empty run directory.
    args = run.parse_args(["--stage", "rankers"])
    args.rankers = list(config.LOCAL_RANKERS)  # as main resolves an omitted --rankers
    assert run.resume_command(args) == (
        f"python -m eval.retrieval.run --split test --stage rankers --date {run.today()} "
        f"--rankers {','.join(config.LOCAL_RANKERS)} --store pg --jev-runs {config.JEV_VARIANCE_RUNS}"
    )
    everything = run.resume_command(
        run.parse_args(
            ["--split", "dev", "--date", "2026-11-07", "--rankers", "bge_m3,jev_full_pool", "--store", "memory",
             "--jev-runs", "2", "--limit", "5", "--workers", "3", "--notes", "notes.md", "--no-cache"]
        )
    )
    assert everything == (
        "python -m eval.retrieval.run --split dev --stage all --date 2026-11-07 --rankers bge_m3,jev_full_pool "
        "--store memory --jev-runs 2 --limit 5 --workers 3 --notes notes.md --no-cache"
    )
    assert run.parse_args(everything.split()[3:]).date == "2026-11-07"  # the hint parses back


# --------------------------------------------------------------------------- #
# The run directory: its dataset and its environment                           #
# --------------------------------------------------------------------------- #
class RecordedStore(InMemoryCorpusStore):
    """The fixture's memory store posing as a pg store: it answers with a recorded fingerprint."""

    def __init__(self, recorded: str | None) -> None:
        super().__init__(PEOPLE, CONVERSATIONS, {r.episode_id: bag_of_words(r.summary) for r in CONVERSATIONS})
        self.recorded = recorded

    def corpus_fingerprint(self) -> str | None:
        return self.recorded


def test_a_pg_run_ranks_only_over_the_database_that_holds_the_committed_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Episode ids are stable across regenerations, so a stale load would serve old rows under
    the gold ids; the fingerprint load_db recorded must be the committed files' own."""

    live = make_runtime(tmp_path, monkeypatch)
    live.store_kind = "pg"
    dataset.save_people(PEOPLE, tmp_path / "data" / "people.jsonl")
    dataset.save_conversations(CONVERSATIONS, tmp_path / "data" / "conversations.jsonl")
    monkeypatch.setattr(config, "PEOPLE_PATH", tmp_path / "data" / "people.jsonl")
    monkeypatch.setattr(config, "CONVERSATIONS_PATH", tmp_path / "data" / "conversations.jsonl")
    expected = db.dataset_fingerprint(PEOPLE, CONVERSATIONS)

    for recorded in ("0" * 64, None):  # another dataset's fingerprint; a load that recorded none
        live.store = RecordedStore(recorded)
        for stage in ("rankers", "system"):
            with pytest.raises(SystemExit) as stopped:
                main(live, "--stage", stage, "--rankers", "bge_m3")
            message = str(stopped.value)
            assert message.startswith("error: the eval database does not hold the committed dataset")
            assert expected in message and "load_db --reset" in message
            assert ("no fingerprint" in message) == (recorded is None)
        assert not (run.run_dir(DATE, SPLIT) / run.RANKER_RECORDS).exists()  # nothing was ranked
        assert not (run.run_dir(DATE, SPLIT) / run.RUN_META).exists()

    live.store = RecordedStore(expected)
    assert main(live, "--rankers", "bge_m3") == 0
    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["store"] == "pg" and results["meta"]["corpus_fingerprint"] == expected
    recorded_env = json.loads((run.run_dir(DATE, SPLIT) / run.RUN_META).read_text(encoding="utf-8"))
    assert recorded_env["corpus_fingerprint"] == expected
    markdown = (tmp_path / "results" / f"{DATE}.md").read_text(encoding="utf-8")
    assert f"- **corpus_fingerprint:** {expected}" in markdown
    # The memory store is built from the files, so it states the same fingerprint unchecked.
    assert make_store().corpus_fingerprint() == expected


def test_a_query_flagged_by_label_pooling_stops_the_run_with_the_regenerate_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flagged unanswerable query has a conversation that answers it, so a decline is not
    the correct outcome: it must be regenerated before it can enter the decline rates."""

    live = make_runtime(tmp_path, monkeypatch)
    unanswerable = next(query for query in QUERIES if query.type == "unanswerable")
    pooling = {"depth": 30, "rankers": list(config.RANKER_NAMES), "judged": 3, "flagged": ["e-budget"]}
    flagged = replace(unanswerable, meta={**unanswerable.meta, "pooling": pooling})
    dataset.save_queries([flagged if query is unanswerable else query for query in QUERIES], config.QUERIES_PATH)

    for stage in ("rankers", "system", "report"):
        with pytest.raises(SystemExit) as stopped:
            main(live, "--stage", stage, "--rankers", "bge_m3")
        message = str(stopped.value)
        assert message.startswith(f"error: 1 {SPLIT} query is flagged by label pooling as answerable after all")
        assert f"({unanswerable.query_id})" in message
        assert message.endswith(f"regenerate them first: {regenerate_command([unanswerable.query_id], config.QUERIES_PATH)}")
    assert not run.run_dir(DATE, SPLIT).exists()  # nothing was started

    # A --limit that leaves the flagged query out does not trip on it; the regenerated file runs.
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3", "--limit", "1") == 0
    dataset.save_queries(QUERIES, config.QUERIES_PATH)
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3", "--date", "2026-11-07") == 0


def test_run_dir_refuses_records_made_against_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    calls = count_rank_calls(live)
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3") == 0
    before = ranker_records()

    # Label pooling grows a gold set in place, under the same query id.
    queries = dataset.load_queries()
    grown = [
        replace(query, gold_episode_ids=[*query.gold_episode_ids, "e-added-by-pooling"])
        if query.query_id == ANSWERABLE[0].query_id
        else query
        for query in queries
    ]
    dataset.save_queries(grown)

    for stage in ("rankers", "report", "all"):
        with pytest.raises(SystemExit, match="recorded against a different dataset") as stopped:
            main(live, "--stage", stage, "--rankers", "bge_m3")
        assert "pass a new --date" in str(stopped.value) and str(run.run_dir(DATE, SPLIT)) in str(stopped.value)
    assert ranker_records() == before and calls["bge_m3"] == len(ANSWERABLE) + 1  # nothing touched
    assert not (tmp_path / "results" / f"{DATE}.json").exists()
    # A new date starts afresh against the grown dataset, and its records carry the new gold ids.
    assert run.main(["--split", SPLIT, "--date", "2026-11-07", "--stage", "rankers", "--rankers", "bge_m3"], runtime=live) == 0
    fresh = read_records(run.run_dir("2026-11-07", SPLIT) / run.RANKER_RECORDS, RankerRecord.from_json)
    assert next(r for r in fresh if r.query_id == ANSWERABLE[0].query_id).gold_ids[-1] == "e-added-by-pooling"
    # Putting the dataset back makes the first directory usable again, with no new calls.
    dataset.save_queries(queries)
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3") == 0
    assert ranker_records() == before and calls["bge_m3"] == 2 * (len(ANSWERABLE) + 1)
    # A directory with records but no marker is refused too (one made before the marker existed).
    (run.run_dir(DATE, SPLIT) / run.DATASET_MARKER).unlink()
    with pytest.raises(SystemExit, match="has records but no dataset marker"):
        main(live, "--stage", "report")


def test_limit_rerun_reports_the_limited_queries_only_and_says_what_was_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    assert main(live, "--rankers", "bge_m3") == 0
    full_ranker, full_system = ranker_records(), system_records()
    first = QUERIES[:3]

    assert main(live, "--rankers", "bge_m3", "--limit", "3") == 0

    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    wanted = {query.query_id for query in first}
    n_ranker = sum(1 for r in full_ranker if r.query_id in wanted)
    n_system = sum(1 for r in full_system if r.query_id in wanted)
    assert results["meta"]["n_ranker_records"] == n_ranker and results["meta"]["n_system_records"] == n_system
    assert results["meta"]["counts"]["queries"] == 3
    assert set(results["ranker_comparison"]["bge_m3"]["by_type"]) == {q.type for q in first if q.answerable}
    assert (
        f"also holds {len(full_ranker) - n_ranker} ranker records and {len(full_system) - n_system} "
        "system records of queries outside this selection; they were left out"
    ) in results["notes"]
    assert "Smoke run: only the first 3 queries" in results["notes"]
    # The full report says nothing of the kind.
    assert main(live, "--stage", "report", "--rankers", "bge_m3") == 0
    results = json.loads((tmp_path / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["n_ranker_records"] == len(full_ranker) and "notes" not in results


def test_report_states_the_environment_recorded_when_the_records_were_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3") == 0
    meta_path = run.run_dir(DATE, SPLIT) / run.RUN_META
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert recorded["models"]["jev"] == live.jev_model and recorded["store"] == "memory"
    assert {"machine", "cpu_count"} <= set(recorded["hardware"]) and recorded["jev_pin"] == {}

    # The report is written later, under a different Jev pin: it states what ran.
    (tmp_path / "jev_scoring.json").write_text(json.dumps({"model": "typesafe/jev-1.99", "mode": "pair"}), encoding="utf-8")
    later = make_runtime(tmp_path / "later", monkeypatch)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runtime_module, "JEV_SCORING_PATH", tmp_path / "jev_scoring.json")
    assert later.jev_model == "typesafe/jev-1.99"
    assert main(later, "--stage", "report", "--rankers", "bge_m3") == 0
    results = json.loads((tmp_path / "later" / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["models"].pop("jev_snapshots") == []  # no Jev ranker ran: nothing served
    assert results["meta"]["models"] == recorded["models"] and results["meta"]["hardware"] == recorded["hardware"]
    assert "notes" not in results
    assert meta_path.exists()  # a report never writes it
    # Without the file the report can only use the live values, and says so.
    meta_path.unlink()
    assert main(later, "--stage", "report", "--rankers", "bge_m3") == 0
    results = json.loads((tmp_path / "later" / "results" / f"{DATE}.json").read_text(encoding="utf-8"))
    assert results["meta"]["models"]["jev"] == "typesafe/jev-1.99"
    assert "read when this report was written" in results["notes"]
    assert not meta_path.exists()


def test_system_rank_latency_is_the_rankers_own_whether_ranked_now_or_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch, llm=MisnamingLLM())
    pin_rank_latency(live, 0.125)
    assert main(live, "--stage", "rankers", "--rankers", "bge_m3") == 0
    assert {record.latency_s for record in ranker_records()} == {0.125}

    assert main(live, "--stage", "system", "--rankers", "bge_m3", "--workers", "2") == 0

    ranked = [row for row in system_records() if row.query_id not in EARLY_EXIT]
    assert ranked and {row.latency_by_stage["rank"] for row in ranked} == {0.125}  # reused and fresh alike
    assert {row.latency_by_stage["rank"] for row in system_records() if row.query_id in EARLY_EXIT} == {0.0}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def test_cli_flags_parse() -> None:
    args = run.parse_args(
        [
            "--split", "dev", "--stage", "system", "--rankers", "jev_full_pool, bge_m3",
            "--store", "memory", "--date", "2026-11-07", "--jev-runs", "2", "--limit", "5",
            "--workers", "3", "--notes", "notes.md", "--no-cache",
        ]
    )
    assert (args.split, args.stage, args.rankers) == ("dev", "system", ["jev_full_pool", "bge_m3"])
    assert (args.store, args.date, args.jev_runs) == ("memory", "2026-11-07", 2)
    assert (args.limit, args.workers, args.notes, args.no_cache) == (5, 3, Path("notes.md"), True)

    defaults = run.parse_args([])
    assert (defaults.split, defaults.stage, defaults.store) == ("test", "all", "pg")
    assert defaults.rankers is None  # main fills in the rankers the gateway serves
    assert defaults.jev_runs == config.JEV_VARIANCE_RUNS
    assert (defaults.limit, defaults.workers, defaults.notes, defaults.no_cache) == (None, None, None, False)
    assert defaults.date == run.today() and len(defaults.date) == 10

    for bad in (["--rankers", "bm25"], ["--date", "6/11/2026"], ["--jev-runs", "0"], ["--limit", "x"],
                ["--stage", "judge"], ["--store", "sqlite"], ["--workers", "-1"]):
        with pytest.raises(SystemExit):
            run.parse_args(bad)


def test_main_builds_a_runtime_only_when_none_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = make_runtime(tmp_path, monkeypatch)
    built: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> Runtime:
        built.append(kwargs)
        return live

    monkeypatch.setattr(run, "Runtime", factory)
    assert run.main(["--stage", "report", "--date", DATE, "--store", "memory", "--no-cache"]) == 0
    assert built == [{"store_kind": "memory", "use_cache": False}]
    assert run.main(["--stage", "report", "--date", DATE], runtime=live) == 0
    assert len(built) == 1
    with pytest.raises(ValueError, match="unknown rankers"):
        run.Benchmark(live, split=SPLIT, date=DATE, rankers=["bm25"])
    with pytest.raises(ValueError, match="jev_runs"):
        run.Benchmark(live, split=SPLIT, date=DATE, jev_runs=0)
