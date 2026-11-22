"""The Jev study, offline: the grid, its numbers, the JSON, the pin and the CLI.

The study runs over the integration fixture's corpus and queries through an
injected ``Runtime``, with ``jev.build_jev`` patched to hand back a ``FakeJev``
for whatever model id is asked. The ordering rules are checked on constructed
cells, where the numbers are chosen, not measured.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from test_eval_integration import ANSWERABLE, QUERIES, FakeEmbedder, FakeJev, FakeScorer, make_store
from test_eval_integration import settings as fixture_settings

from eval.retrieval import config, jev, jev_study, metrics, runs, runtime
from eval.retrieval.interfaces import Usage
from eval.retrieval.jev_study import Cell, Study, aggregate, choose_pin, study_variants, table_rows
from eval.retrieval.pipelines.jev_scoring import MAX_CHOICE_OPTIONS, JevScoringConfig

OPENROUTER_MODELS = ["typesafe/jev-1.13", "typesafe/jev-1.14"]
VERCEL_MODELS = ["typesafe-ai/jev-1.13.0", "typesafe-ai/jev-1.14.0"]
SNAPSHOT_SUFFIX = "-20261014"  # what the fake gateway appends to a requested id


# --------------------------------------------------------------------------- #
# Harness: the fixture corpus behind a Runtime, fake Jev clients per model     #
# --------------------------------------------------------------------------- #
@dataclass
class Harness:
    runtime: runtime.Runtime
    built: list[str] = field(default_factory=list)  # model ids build_jev was asked for, in order
    clients: dict[str, FakeJev] = field(default_factory=dict)


@pytest.fixture
def harness(monkeypatch) -> Harness:
    live = runtime.Runtime(fixture_settings(), store_kind="memory", use_cache=False)
    live.store, live.embedder, live.scorer = make_store(), FakeEmbedder(), FakeScorer()
    built = Harness(live)

    def fake_build(settings: config.Settings, cache: Any = None, spend: Any = None) -> FakeJev:
        built.built.append(settings.jev_model)
        client = FakeJev(model=f"{settings.jev_model}{SNAPSHOT_SUFFIX}")
        built.clients[settings.jev_model] = client
        return client

    monkeypatch.setattr(jev, "build_jev", fake_build)
    monkeypatch.setattr(jev_study, "load_queries", lambda split: list(QUERIES))
    return built


def run_main(harness: Harness, out: Path, *args: str) -> dict[str, Any]:
    assert jev_study.main(["--out", str(out), *args], runtime=harness.runtime) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def labels(variants: Sequence[JevScoringConfig]) -> list[str]:
    return [variant.label for variant in variants]


# --------------------------------------------------------------------------- #
# Constructed cells, for the rules                                             #
# --------------------------------------------------------------------------- #
def record(
    query_id: str,
    hit: bool,
    *,
    cost: float,
    latency: float,
    n_cached: int = 0,
    ranker: str = "bge_m3_jev",
    run: int = 0,
    variant: str = "x",
) -> metrics.RankerRecord:
    return metrics.RankerRecord(
        query_id=query_id,
        type="lookup",
        pool_size=10,
        gold_ids=["gold"],
        ranker=ranker,
        run=run,
        ranked_ids=["gold" if hit else "other"],
        scores=[1.0],
        stage1_ids=None,
        latency_s=latency,
        stage_latency_s={},
        usage=Usage(input_tokens=100, cost_usd=cost, latency_s=latency, calls=1),
        extra={
            "n_requests": 1,
            "n_missing": 0,
            "n_cached": n_cached,
            "jev_model": "m-2026",
            "scoring": variant,
        },
    )


def cell(
    variant: JevScoringConfig,
    hits: Sequence[bool],
    *,
    cost: float,
    latency: float,
    model: str = "m",
    ranker: str = "bge_m3_jev",
    run: int = 0,
    n_cached: int = 0,
) -> Cell:
    records = [
        record(
            f"q{index}",
            hit,
            cost=cost,
            latency=latency,
            n_cached=n_cached,
            ranker=ranker,
            run=run,
            variant=variant.label,
        )
        for index, hit in enumerate(hits)
    ]
    scoring = replace(variant, salt=jev_study.study_salt(model, run))
    return Cell(ranker, model, scoring, run, records)


PAIR_V1 = JevScoringConfig(mode="pair", primitive="noul", wording="v1")
PAIR_SCORE_V2 = JevScoringConfig(mode="pair", primitive="score", wording="v2")
BATCHED_V1 = JevScoringConfig(mode="batched", primitive="noul", wording="v1", batch_size=16)
BATCHED_V2 = JevScoringConfig(mode="batched", primitive="noul", wording="v2", batch_size=16)
LISTWISE_V1 = JevScoringConfig(mode="listwise", wording="v1")


# --------------------------------------------------------------------------- #
# The grid                                                                     #
# --------------------------------------------------------------------------- #
def test_the_grid_is_pair_and_batched_over_primitive_and_wording_plus_listwise() -> None:
    grid = labels(study_variants())

    assert len(grid) == len(set(grid)) == 14
    assert grid[:4] == ["pair-noul-v1", "pair-noul-v2", "pair-score-v1", "pair-score-v2"]
    assert grid[4:8] == [
        "batched-noul-v1-b16",
        "batched-noul-v1-b64",
        "batched-noul-v2-b16",
        "batched-noul-v2-b64",
    ]
    assert grid[-2:] == ["listwise-v1", "listwise-v2"]
    batched_b8 = [
        f"batched-{primitive}-{wording}-b8"
        for primitive in ("noul", "score")
        for wording in ("v1", "v2")
    ]
    assert labels(study_variants([8])) == grid[:4] + batched_b8 + grid[-2:]
    assert labels(study_variants([64, 16, 16])) == grid  # sizes are sorted and deduplicated
    with pytest.raises(ValueError, match="positive"):
        study_variants([0, 16])

    chosen = jev_study.select_variants(study_variants(), ["listwise-v2", "pair-noul-v1"])
    assert labels(chosen) == ["pair-noul-v1", "listwise-v2"]  # grid order, not the order asked
    with pytest.raises(ValueError, match="unknown variant label.*batched-noul-v1-b8"):
        jev_study.select_variants(study_variants(), ["batched-noul-v1-b8"])


def test_listwise_fits_the_stage1_candidates_but_not_a_whole_pool() -> None:
    assert jev_study.supports_listwise("bge_m3_jev") is (config.STAGE1_TOP_K <= MAX_CHOICE_OPTIONS)
    assert not jev_study.supports_listwise("jev_full_pool")
    assert str(MAX_CHOICE_OPTIONS) in jev_study.listwise_skip_reason("jev_full_pool")
    assert jev_study.default_models("openrouter") == OPENROUTER_MODELS
    assert jev_study.default_models("vercel") == VERCEL_MODELS
    with pytest.raises(ValueError, match="--models"):
        jev_study.default_models("azure")


def test_the_study_runs_every_cell_and_skips_listwise_for_the_full_pool(
    harness: Harness, tmp_path: Path, capsys
) -> None:
    data = run_main(harness, tmp_path / "study.json")
    grid = labels(study_variants())

    expected = {
        (ranker, model, variant, run)
        for ranker in config.JEV_RANKERS
        for model in OPENROUTER_MODELS
        for variant in grid
        for run in (0, 1)
        if not (variant.startswith("listwise") and ranker == "jev_full_pool")
    }
    cells = {(c["ranker"], c["model"], c["variant"], c["run"]) for c in data["cells"]}
    assert cells == expected and len(data["cells"]) == len(expected) == 104
    assert data["skipped"] == [
        {
            "ranker": "jev_full_pool",
            "variant": label,
            "reason": jev_study.listwise_skip_reason("jev_full_pool"),
        }
        for label in ("listwise-v1", "listwise-v2")
    ]
    out = capsys.readouterr().out
    assert "skipped jev_full_pool x listwise-v1" in out and f"{MAX_CHOICE_OPTIONS}" in out
    assert f"Wrote {tmp_path / 'study.json'}" in out

    # One client per model, each asked with that model's per-run salts only.
    assert harness.built == OPENROUTER_MODELS
    for model, client in harness.clients.items():
        assert {salt for _, _, salt in client.requests} == {
            jev_study.study_salt(model, run) for run in (0, 1)
        }
    for one in data["cells"]:
        assert one["salt"] == jev_study.study_salt(one["model"], one["run"])
        assert one["scoring"] == {
            **JevScoringConfig.from_json(one["scoring"]).to_json(),
            "concurrency": 3,  # the fixture settings' jev_concurrency
            "salt": one["salt"],
        }
        assert JevScoringConfig.from_json(one["scoring"]).label == one["variant"]
        assert [r["query_id"] for r in one["records"]] == [q.query_id for q in ANSWERABLE]
        assert {r["ranker"] for r in one["records"]} == {one["ranker"]}
        assert {r["extra"]["scoring"] for r in one["records"]} == {one["variant"]}
        assert one["summary"]["jev_models"] == [f"{one['model']}{SNAPSHOT_SUFFIX}"]

    meta = data["meta"]
    assert meta["split"] == "dev" and meta["store"] == "memory" and meta["runs"] == 2
    assert meta["models"] == OPENROUTER_MODELS and meta["variants"] == grid
    assert meta["rankers"] == list(config.JEV_RANKERS)
    assert meta["n_queries"] == len(ANSWERABLE) and meta["seed"] == config.SEED
    assert meta["query_ids"] == [q.query_id for q in ANSWERABLE]
    assert {"machine", "cpu_count"} <= set(meta["hardware"])

    rows = data["table"]
    assert [row["rank"] for row in rows] == list(range(1, 53))
    groups = {(row["ranker"], row["model"], row["variant"]) for row in rows}
    assert groups == {key[:3] for key in expected}
    assert all(row["n_runs"] == 2 and row["n_cells"] == 2 for row in rows)
    keys = [jev_study._ordering_key(row) for row in rows]
    assert keys == sorted(keys)


def test_cells_rank_through_the_runtime_and_the_benchmark_loop(harness: Harness, tmp_path: Path) -> None:
    builds: list[tuple[str, dict[str, Any]]] = []
    build = harness.runtime.ranker

    def ranker(name: str, **kwargs: Any) -> Any:
        builds.append((name, kwargs))
        return build(name, **kwargs)

    harness.runtime.ranker = ranker  # type: ignore[method-assign]

    data = run_main(
        harness, tmp_path / "study.json", "--variants", "pair-noul-v1", "--models", "m", "--runs", "1"
    )

    # One ranker per cell, from the runtime, asking that cell's client with that cell's scoring.
    assert [name for name, _ in builds] == list(config.JEV_RANKERS)
    for (name, kwargs), one in zip(builds, data["cells"]):
        assert kwargs["jev"] is harness.clients["m"] and kwargs["k"] == jev_study.RANKER_K
        assert kwargs["jev_scoring"] == JevScoringConfig.from_json(one["scoring"])
        assert kwargs["jev_scoring"].salt == jev_study.study_salt("m", 0)
    # No warm-up: every Jev request belongs to a recorded query, so the first record is timed too.
    [client] = harness.clients.values()
    records = [record for one in data["cells"] for record in one["records"]]
    assert len(client.requests) == sum(record["extra"]["n_requests"] for record in records)
    assert all(not record["extra"].get("n_cached") for record in records)

    # The benchmark's stage-1 guard holds in the study: a store that hands on too
    # few candidates stops the run instead of quietly lowering every cell's recall.
    class ShortStore:
        def __init__(self, store: Any) -> None:
            self._store = store

        def __getattr__(self, name: str) -> Any:
            return getattr(self._store, name)

        def vector_search(self, *args: Any, **kwargs: Any) -> list[tuple[str, float]]:
            return self._store.vector_search(*args, **kwargs)[:1]

    harness.runtime.store = ShortStore(make_store())
    with pytest.raises(runs.Stage1Shortfall, match="stage 1 returned 1 candidates"):
        jev_study.run_study(
            harness.runtime,
            list(QUERIES),
            models=["m"],
            variants=[JevScoringConfig(mode="pair")],
            runs=1,
            rankers=["bge_m3_jev"],
            log=lambda message: None,
        )


def test_every_cell_carries_the_metrics_with_their_cis(harness: Harness, tmp_path: Path) -> None:
    data = run_main(
        harness,
        tmp_path / "study.json",
        "--models",
        "m",
        "--variants",
        "pair-noul-v1,batched-noul-v1-b16",
        "--runs",
        "2",
    )
    assert len(data["cells"]) == 2 * 2 * 2  # rankers x variants x runs
    multi_gold = [q for q in ANSWERABLE if len(q.gold_episode_ids) >= 2]

    for one in data["cells"]:
        summary = one["summary"]
        assert summary["n_queries"] == len(ANSWERABLE) and summary["n_cells"] == 1
        assert set(summary["recall"]) == {"5", "10", "20"}
        for estimate in summary["recall"].values():
            assert set(estimate) == {"mean", "lo", "hi", "n"} and estimate["n"] == len(ANSWERABLE)
            assert 0.0 <= estimate["lo"] <= estimate["mean"] <= estimate["hi"] <= 1.0
        all_found = summary["all_found"]["10"]
        assert all_found["n"] == len(multi_gold) and all_found["mean"] in (0.0, 1.0)
        assert summary["recall"]["10"]["mean"] == 1.0  # the fakes find every gold conversation
        latency = summary["latency"]
        assert latency["n"] == len(ANSWERABLE) and 0.0 <= latency["p50"] <= latency["p95"]
        assert summary["n_cached_records"] == 0 and summary["n_missing"] == 0
        assert summary["cost_per_query"] > 0.0
        if one["variant"] == "pair-noul-v1":
            assert summary["requests_per_query"] > 1.0  # one request per candidate
        else:
            assert summary["requests_per_query"] == 1.0  # every fixture pool fits one batch
        assert summary["recall_by_run"] == {str(one["run"]): 1.0} and summary["run_spread"] is None
        assert summary["jev_models"] == [f"m{SNAPSHOT_SUFFIX}"]

    for row in data["table"]:
        assert row["recall_by_run"] == {"0": 1.0, "1": 1.0} and row["run_spread"] == 0.0
        assert row["latency"]["n"] == 2 * len(ANSWERABLE)
    # Per-query rankings are in the file: a record reads back as a RankerRecord.
    first = metrics.RankerRecord.from_json(data["cells"][0]["records"][0])
    assert first.ranked_ids and first.query_id == ANSWERABLE[0].query_id


def test_the_study_json_round_trips(harness: Harness, tmp_path: Path) -> None:
    out = tmp_path / "study.json"
    variants = "batched-score-v2-b16,listwise-v1"
    data = run_main(harness, out, "--models", "m", "--variants", variants, "--runs", "1")

    text = out.read_text(encoding="utf-8")
    assert text == json.dumps(data, indent=2, sort_keys=True) + "\n"  # sorted keys, no NaN
    study = Study.from_json(data)
    assert study.to_json() == data
    assert [c.variant for c in study.cells] == [
        "batched-score-v2-b16",
        "listwise-v1",
        "batched-score-v2-b16",
    ]
    assert study.table() == data["table"]
    assert data["pin"] is None and study.skipped == data["skipped"]


# --------------------------------------------------------------------------- #
# The pin                                                                      #
# --------------------------------------------------------------------------- #
def without_model(pinned: dict[str, Any]) -> dict[str, Any]:
    """The scoring part of a pin: what is left once the fields naming the model are dropped."""

    return {key: value for key, value in pinned.items() if key not in runtime.JEV_PIN_KEYS}


def test_pin_writes_what_runtime_loads(
    harness: Harness, tmp_path: Path, monkeypatch, capsys
) -> None:
    pin_path = tmp_path / "jev_scoring.json"
    monkeypatch.setattr(runtime, "JEV_SCORING_PATH", pin_path)
    monkeypatch.delenv("EVAL_JEV_MODEL", raising=False)
    data = run_main(harness, tmp_path / "study.json", "--pin", "--batch-sizes", "16")

    pinned = json.loads(pin_path.read_text(encoding="utf-8"))
    assert pinned["model"] in OPENROUTER_MODELS and pinned["gateway"] == "openrouter"
    assert not {"concurrency", "salt"} & set(pinned)  # per-run and per-machine settings stay out
    scoring = JevScoringConfig.from_json(without_model(pinned))
    assert scoring.salt == "" and scoring.mode != "listwise"
    assert scoring.label == data["pin"]["variant"] and pinned["model"] == data["pin"]["model"]
    assert data["pin"]["rankers"] == list(config.JEV_RANKERS)
    assert data["pin"]["n_cells"] == 4 and data["pin"]["n_runs"] == 2

    fresh = runtime.Runtime(fixture_settings(), store_kind="memory", use_cache=False)
    assert fresh.jev_model == pinned["model"]
    assert fresh.jev_scoring("run-3") == replace(scoring, concurrency=3, salt="run-3")

    # Best mean Recall@10 over both rankers and both runs, then the cheapest.
    cells = Study.from_json(data).cells
    groups: dict[tuple[str, str], list[Cell]] = {}
    for one in cells:
        if one.scoring.mode != "listwise":
            groups.setdefault((one.model, one.variant), []).append(one)
    costs = {key: aggregate(group)["cost_per_query"] for key, group in groups.items()}
    assert all(aggregate(group)["recall"]["10"]["mean"] == 1.0 for group in groups.values())
    assert data["pin"]["recall"]["10"]["mean"] == 1.0
    assert costs[(data["pin"]["model"], data["pin"]["variant"])] == min(costs.values())

    out = capsys.readouterr().out
    assert data["pin"]["reason"] in out and "decided by" in data["pin"]["reason"]
    assert f"Pinned {scoring.label} on {pinned['model']} to {pin_path}" in out


def test_pin_refuses_listwise_unless_allowed(harness: Harness, tmp_path: Path, monkeypatch) -> None:
    pin_path = tmp_path / "jev_scoring.json"
    monkeypatch.setattr(runtime, "JEV_SCORING_PATH", pin_path)
    with pytest.raises(SystemExit, match="--allow-listwise"):
        jev_study.main(
            ["--out", str(tmp_path / "s.json"), "--pin", "--variants", "listwise-v1,listwise-v2"],
            runtime=harness.runtime,
        )
    assert not pin_path.exists() and harness.built == []  # refused before any client was built

    listwise_only = ["--pin", "--allow-listwise", "--variants", "listwise-v2", "--models", "m"]
    data = run_main(harness, tmp_path / "s.json", *listwise_only)
    pinned = json.loads(pin_path.read_text(encoding="utf-8"))
    assert pinned["model"] == "m" and data["pin"]["rankers"] == ["bge_m3_jev"]
    assert JevScoringConfig.from_json(without_model(pinned)).label == "listwise-v2"


def test_pin_tie_breaks_on_a_constructed_table() -> None:
    two_of_three = [True, True, False]
    pair = cell(PAIR_V1, two_of_three, cost=0.001, latency=1.0)
    batched_v1 = cell(BATCHED_V1, two_of_three, cost=0.0005, latency=2.0)  # same recall, cheaper
    batched_v2 = cell(BATCHED_V2, two_of_three, cost=0.0005, latency=0.5)  # same cost, faster
    best = cell(PAIR_SCORE_V2, [True, True, True], cost=0.01, latency=9.0)  # best recall, dearest
    listwise = cell(LISTWISE_V1, [True, True, True], cost=0.0001, latency=0.1)  # best and cheapest

    choice = choose_pin([pair, batched_v1, batched_v2, best, listwise])
    assert (choice["model"], choice["variant"]) == ("m", "pair-score-v2")
    assert choice["scoring"] == PAIR_SCORE_V2.to_json() and choice["scoring"]["salt"] == ""
    assert choice["runner_up"]["variant"] == "batched-noul-v2-b16"
    assert "decided by Recall@10" in choice["reason"]

    choice = choose_pin([pair, batched_v1, batched_v2])
    assert choice["variant"] == "batched-noul-v2-b16"
    assert "decided by p50 latency" in choice["reason"]
    choice = choose_pin([pair, batched_v1])
    assert choice["variant"] == "batched-noul-v1-b16"
    assert "decided by $ per query" in choice["reason"]
    assert choose_pin([pair])["runner_up"] is None

    allowed = choose_pin([pair, batched_v1, batched_v2, best, listwise], allow_listwise=True)
    assert allowed["variant"] == "listwise-v1" and "decided by $ per query" in allowed["reason"]
    with pytest.raises(ValueError, match="allow-listwise"):
        choose_pin([listwise])
    with pytest.raises(ValueError, match="nothing to pin"):
        choose_pin([])

    # The table keeps listwise and ranks every (ranker, model, variant) by the same rules.
    rows = table_rows([pair, batched_v1, batched_v2, best, listwise])
    assert [row["variant"] for row in rows] == [
        "listwise-v1",
        "pair-score-v2",
        "batched-noul-v2-b16",
        "batched-noul-v1-b16",
        "pair-noul-v1",
    ]
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]


def test_pin_averages_over_both_rankers_and_every_run() -> None:
    # Perfect on one ranker and useless on the other loses to decent on both.
    def on(ranker: str, variant: JevScoringConfig, hits: list[bool], run: int = 0) -> Cell:
        return cell(variant, hits, cost=0.001, latency=1.0, ranker=ranker, run=run)

    lopsided = [
        on("bge_m3_jev", PAIR_V1, [True, True, True]),
        on("jev_full_pool", PAIR_V1, [False, False, False]),
    ]
    steady = [
        on("bge_m3_jev", BATCHED_V1, [True, True, False], run=0),
        on("bge_m3_jev", BATCHED_V1, [True, True, False], run=1),
        on("jev_full_pool", BATCHED_V1, [True, False, True], run=0),
        on("jev_full_pool", BATCHED_V1, [True, True, True], run=1),
    ]
    choice = choose_pin(lopsided + steady)

    assert choice["variant"] == "batched-noul-v1-b16"
    assert choice["recall"]["10"]["mean"] == pytest.approx(9 / 12)
    assert choice["recall"]["10"]["n"] == 3  # the CI is over queries, not over cells
    assert choice["rankers"] == ["bge_m3_jev", "jev_full_pool"] and choice["n_runs"] == 2
    assert choice["recall_by_run"] == {"0": pytest.approx(4 / 6), "1": pytest.approx(5 / 6)}
    assert choice["run_spread"] == pytest.approx(1 / 6)
    assert choice["runner_up"]["recall"]["10"]["mean"] == pytest.approx(0.5)

    # A second model is its own candidate, even for the same variant.
    other = cell(BATCHED_V1, [True, True, True], cost=0.001, latency=1.0, model="m2")
    assert choose_pin(lopsided + steady + [other])["model"] == "m2"


def test_latency_leaves_out_cache_replays_and_unknown_p50_ranks_last() -> None:
    fresh = cell(PAIR_V1, [True, True], cost=0.001, latency=4.0)
    replayed = cell(PAIR_V1, [True, True], cost=0.001, latency=0.01, run=1, n_cached=3)
    summary = aggregate([fresh, replayed])

    assert summary["latency"] == {"p50": 4.0, "p95": 4.0, "mean": 4.0, "n": 2}
    assert summary["n_cached_records"] == 2 and summary["n_cells"] == 2
    assert aggregate([replayed])["latency"] == {"p50": None, "p95": None, "mean": None, "n": 0}

    all_replayed = cell(BATCHED_V1, [True, True], cost=0.001, latency=0.01, n_cached=1)
    rows = table_rows([all_replayed, fresh])
    assert [row["variant"] for row in rows] == ["pair-noul-v1", "batched-noul-v1-b16"]
    assert rows[1]["latency"]["p50"] is None


def test_write_pin_is_what_runtime_reads(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runtime, "JEV_SCORING_PATH", tmp_path / "nested" / "jev_scoring.json")
    monkeypatch.delenv("EVAL_JEV_MODEL", raising=False)
    only = cell(BATCHED_V2, [True], cost=0.001, latency=1.0, model="typesafe/jev-1.13")

    path = jev_study.write_pin(choose_pin([only]), "openrouter")
    assert path == runtime.JEV_SCORING_PATH
    scoring = {k: v for k, v in BATCHED_V2.to_json().items() if k not in ("concurrency", "salt")}
    assert runtime.load_jev_pin() == {
        "model": "typesafe/jev-1.13",
        "gateway": "openrouter",
        "served_models": ["m-2026"],  # the snapshot the API served the study (the cell's trace)
        **scoring,
    }
    live = runtime.Runtime(fixture_settings(), store_kind="memory", use_cache=False)
    assert live.jev_model == "typesafe/jev-1.13"
    assert live.jev_scoring("r") == replace(BATCHED_V2, concurrency=3, salt="r")
    with pytest.raises(ValueError, match="unknown gateway"):
        jev_study.write_pin(choose_pin([only]), "bedrock")

    # The pinned id is OpenRouter's spelling: on another gateway the pin's scoring still
    # applies, but the model falls back to that gateway's default, with a warning.
    vercel_default = config.GATEWAYS["vercel"]["jev_model"]
    elsewhere = runtime.Runtime(
        fixture_settings(gateway="vercel", jev_model=vercel_default), store_kind="memory", use_cache=False
    )
    assert elsewhere.jev_model == vercel_default
    assert elsewhere.jev_scoring("r") == replace(BATCHED_V2, concurrency=3, salt="r")
    warning = capsys.readouterr().err
    assert "pins Jev 'typesafe/jev-1.13' on the openrouter gateway" in warning and "EVAL_JEV_MODEL" in warning
    # Settings that name a model other than the gateway's default are an explicit choice.
    chosen = runtime.Runtime(fixture_settings(jev_model="typesafe/jev-1.12"), store_kind="memory", use_cache=False)
    assert chosen.jev_model == "typesafe/jev-1.12"
    # EVAL_JEV_MODEL is read by get_settings alone: settings built elsewhere keep their own model.
    monkeypatch.setenv("EVAL_JEV_MODEL", "typesafe/jev-1.15")
    assert runtime.Runtime(fixture_settings(), store_kind="memory", use_cache=False).jev_model == "typesafe/jev-1.13"


# --------------------------------------------------------------------------- #
# The CLI                                                                      #
# --------------------------------------------------------------------------- #
def test_variants_and_models_filters(harness: Harness, tmp_path: Path) -> None:
    data = run_main(
        harness,
        tmp_path / "study.json",
        "--variants",
        "pair-noul-v1,listwise-v2",
        "--models",
        "only/one",
    )

    cells = {(c["ranker"], c["variant"], c["run"]) for c in data["cells"]}
    assert cells == {
        ("bge_m3_jev", "pair-noul-v1", 0),
        ("bge_m3_jev", "pair-noul-v1", 1),
        ("bge_m3_jev", "listwise-v2", 0),
        ("bge_m3_jev", "listwise-v2", 1),
        ("jev_full_pool", "pair-noul-v1", 0),
        ("jev_full_pool", "pair-noul-v1", 1),
    }
    assert {c["model"] for c in data["cells"]} == {"only/one"} and harness.built == ["only/one"]
    assert [skip["variant"] for skip in data["skipped"]] == ["listwise-v2"]
    assert data["meta"]["variants"] == ["pair-noul-v1", "listwise-v2"]

    unknown = ["--variants", "batched-noul-v1-b8", "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match="unknown variant label"):
        jev_study.main(unknown, runtime=harness.runtime)
    b8 = ["--batch-sizes", "8", "--variants", "batched-noul-v1-b8", "--models", "m", "--runs", "1"]
    data = run_main(harness, tmp_path / "b8.json", *b8)
    assert {c["variant"] for c in data["cells"]} == {"batched-noul-v1-b8"}
    assert {c["scoring"]["batch_size"] for c in data["cells"]} == {8}


def test_cli_parses_and_rejects_bad_input(harness: Harness, tmp_path: Path, monkeypatch) -> None:
    parser = jev_study.build_parser()
    args = parser.parse_args([])
    assert (args.split, args.store, args.runs) == ("dev", "pg", 2)
    assert (args.pin, args.allow_listwise) == (False, False)
    assert args.batch_sizes == (16, 64) and args.models is None and args.variants is None
    assert args.out == config.DATA_DIR / "jev_study.json"

    args = parser.parse_args(
        [
            "--split", "test", "--store", "memory", "--models", "a, b,a", "--variants", "x,y",
            "--runs", "3", "--batch-sizes", "64,16,16", "--out", "o.json", "--pin", "--allow-listwise",
        ]
    )
    assert (args.split, args.store, args.runs) == ("test", "memory", 3)
    assert (args.pin, args.allow_listwise) == (True, True)
    assert args.models == ["a", "b"] and args.variants == ["x", "y"]
    assert args.batch_sizes == (16, 64) and args.out == Path("o.json")
    bad_inputs = (
        ["--runs", "0"],
        ["--batch-sizes", "0,16"],
        ["--batch-sizes", "x"],
        ["--store", "sqlite"],
        ["--models", " , "],
    )
    for bad in bad_inputs:
        with pytest.raises(SystemExit):
            parser.parse_args(bad)

    unanswerable = [q for q in QUERIES if not q.answerable]
    monkeypatch.setattr(jev_study, "load_queries", lambda split: unanswerable)
    with pytest.raises(SystemExit, match="no answerable queries in the dev split"):
        jev_study.main(["--out", str(tmp_path / "x.json")], runtime=harness.runtime)
    assert harness.built == []


def test_main_refuses_a_gateway_without_jev_before_any_call(harness: Harness, tmp_path: Path) -> None:
    harness.runtime.settings = replace(
        harness.runtime.settings, gateway="openai", jev_url="", jev_api_key=None, jev_model=""
    )

    with pytest.raises(SystemExit, match="Jev is not served on the openai gateway"):
        jev_study.main(["--out", str(tmp_path / "x.json")], runtime=harness.runtime)
    assert harness.built == [] and not (tmp_path / "x.json").exists()


def test_main_builds_a_runtime_only_when_none_is_given(
    harness: Harness, tmp_path: Path, monkeypatch
) -> None:
    constructed: list[str] = []

    class StubRuntime(runtime.Runtime):
        def __init__(self, settings: Any = None, *, store_kind: str = "pg", **_: Any) -> None:
            super().__init__(fixture_settings(gateway="vercel"), store_kind=store_kind, use_cache=False)
            constructed.append(store_kind)
            self.store, self.embedder, self.scorer = make_store(), FakeEmbedder(), FakeScorer()

    monkeypatch.setattr(jev_study, "Runtime", StubRuntime)
    small = ["--store", "memory", "--variants", "pair-noul-v1", "--runs", "1"]
    data = run_main(harness, tmp_path / "injected.json", *small)
    assert constructed == [] and data["meta"]["gateway"] == "openrouter"

    out = tmp_path / "built.json"
    assert jev_study.main(["--out", str(out), *small]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert constructed == ["memory"] and data["meta"]["gateway"] == "vercel"
    assert data["meta"]["models"] == VERCEL_MODELS  # the gateway's defaults
