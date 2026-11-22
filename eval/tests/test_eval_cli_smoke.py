"""Every CLI under ``eval/retrieval`` run in sequence with fakes, the way a person runs them.

Two legs, each in its own temporary workspace with ``config``'s directories,
the query file and the Jev pin pointed there, so nothing touches the repo:

* The generators: ``data.generate --smoke`` writes a six-person corpus with
  the corpus fake of ``test_eval_corpus``, ``data.queries`` writes queries over
  it with the query fake of ``test_eval_querygen``, ``data.show_queries``
  renders them from the files, and ``run --stage all`` benchmarks them with a
  system model that declines every question. That proves the generated files
  are what the sheet and the runner expect, which the fixture cannot.
* The fixture corpus of ``test_eval_integration`` with its scripted fakes:
  ``jev_study --pin`` writes the pin a fresh ``Runtime`` reads, ``pool_labels``
  adds the label the query file misses and flags an unanswerable query whose
  answer sits in a dialogue (exit 2), ``run --stage all --split dev`` writes a
  report with every section filled and resumes with no new hosted call, and
  ``judge_validation`` samples and scores the run's answers, which the report
  stage then folds in. Two queries are checked by hand against the records.

Each leg runs once, inside a ``MonkeyPatch`` context that is undone before
the tests read what the leg left on disk. A CLI is driven through
``main(argv, runtime=...)`` with a ``Runtime`` whose models and clients are
fakes and whose store the runtime builds from the files, as ``--store memory``
does; the Jev study builds one client per model through ``jev.build_jev``,
which is patched to hand back fakes.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

import test_eval_corpus
import test_eval_querygen
from test_eval_integration import (
    ANSWERABLE,
    OFFSITE_ANSWER,
    PEOPLE,
    CONVERSATIONS,
    QUERIES,
    FakeEmbedder,
    FakeJev,
    FakeLLM,
    FakeScorer,
    settings,
)
from test_eval_run import count_rank_calls

from eval.retrieval import config, dataset, jev, jev_study, judge_validation, report, run
from eval.retrieval import runtime as runtime_module
from eval.retrieval.data import generate, pool_labels, show_queries
from eval.retrieval.data import queries as query_generator
from eval.retrieval.interfaces import LLMResult, Usage
from eval.retrieval.metrics import RankerRecord, SystemRecord
from eval.retrieval.pipelines import JevScoringConfig
from eval.retrieval.runtime import Runtime
from eval.retrieval.schema import VERDICTS, Conversation, Person, Query

DATE = "2026-11-06"
GENERATED_DATE = "2026-11-05"
N_GENERATED_QUERIES = 24
N_CELLS = len(config.SYSTEM_KS) * len(config.FALLBACK_KS)
VENUE_ANSWER = "The unapproved venue deposit."
MULTI = next(record for record in QUERIES if record.query_id == "q-multi")
FLAGGED_ID = "q-offsite-none"
SNAPSHOT_SUFFIX = "-20261014"
SECTIONS = (
    "## Summary",
    "## Setup",
    "## Ranker comparison",
    "## Jev run-to-run variance",
    "## System eval",
    "## Recommendation",
)


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Outcome:
    """What one CLI call left behind: its exit code and what it printed."""

    code: int
    out: str
    err: str


def call(main: Callable[..., int], argv: Sequence[str], **kwargs: Any) -> Outcome:
    """Run one CLI's ``main`` with its output captured, as a terminal would show it."""

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv), **kwargs)
    return Outcome(code=code, out=out.getvalue(), err=err.getvalue())


def point_at(patch: pytest.MonkeyPatch, root: Path) -> None:
    """Every path the CLIs read or write, under ``root``, and no network.

    The legs run as module fixtures, before the function-scoped socket block
    of ``conftest`` is in place, so the same block is applied here.
    """

    def blocked(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"the CLI smoke must not use the network (tried {address!r})")

    patch.setattr(socket.socket, "connect", blocked)
    patch.setattr(socket.socket, "connect_ex", blocked)
    data_dir = root / "data"
    patch.setattr(config, "DATA_DIR", data_dir)
    patch.setattr(config, "PEOPLE_PATH", data_dir / "people.jsonl")
    patch.setattr(config, "CONVERSATIONS_PATH", data_dir / "conversations.jsonl")
    patch.setattr(config, "QUERIES_PATH", data_dir / "queries.jsonl")
    patch.setattr(config, "JUDGE_VALIDATION_PATH", data_dir / "judge_validation.json")
    patch.setattr(config, "CACHE_DIR", root / "cache")
    patch.setattr(config, "RESULTS_DIR", root / "results")
    patch.setattr(runtime_module, "JEV_SCORING_PATH", root / "jev_scoring.json")
    patch.delenv("EVAL_JEV_MODEL", raising=False)


def fake_runtime(llm: Any = None, embedder: Any = None) -> Runtime:
    """A memory-store ``Runtime`` with fake models and clients; the store comes from the files."""

    live = Runtime(settings(), store_kind="memory", use_cache=False)
    live.llm = llm if llm is not None else FakeLLM()
    live.jev = FakeJev()
    live.embedder = embedder if embedder is not None else FakeEmbedder()
    live.scorer = FakeScorer()
    return live


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path, from_json: Callable[[dict[str, Any]], Any]) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        return [from_json(json.loads(line)) for line in handle if line.strip()]


def results_of(root: Path, date: str) -> tuple[dict[str, Any], str]:
    """The report a run wrote under ``root``: the JSON and the Markdown."""

    results_dir = root / "results"
    return read_json(results_dir / f"{date}.json"), (results_dir / f"{date}.md").read_text(encoding="utf-8")


def run_dir(root: Path, date: str, split: str) -> Path:
    return root / "cache" / "runs" / f"{date}-{split}"


# --------------------------------------------------------------------------- #
# Fakes for the generated corpus: generic, so they know nothing of its text    #
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Bag of hashed words, L2-normalised: a bi-encoder that works on any text."""

    model_name = "fake/hashing-bag-of-words"
    dim = 64

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass
class DecliningLLM:
    """A system model that extracts nothing and never answers, and a judge that grades that."""

    calls: list[str] = field(default_factory=list)  # namespaces, in call order

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        namespace: str = "default",
        salt: str = "",
    ) -> LLMResult:
        self.calls.append(namespace)
        if system.startswith("You extract search filters"):
            data: dict[str, Any] = {"name": None, "time_range": None}
        elif system.startswith("You answer a question"):
            data = {"can_answer": False, "answer": ""}
        elif system.startswith("You grade answers"):
            data = {"verdict": "incorrect", "reason": "no answer was given"}
        else:
            raise AssertionError(f"unexpected prompt: {system[:60]!r}")
        usage = Usage(input_tokens=len(user) // 4, output_tokens=8, cost_usd=len(user) * 1e-7, latency_s=0.01, calls=1)
        return LLMResult(data=data, usage=usage, model=model, raw_text=json.dumps(data))


# --------------------------------------------------------------------------- #
# Leg 1: the generators, then the sheet and the runner over what they wrote    #
# --------------------------------------------------------------------------- #
@dataclass
class GeneratedLeg:
    root: Path
    smoke_dir: Path
    generated: Outcome
    written: Outcome
    sheet: Outcome
    benchmark: Outcome
    people: list[Person]
    conversations: list[Conversation]
    queries: list[Query]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> GeneratedLeg:
    root = tmp_path_factory.mktemp("generated")
    with pytest.MonkeyPatch.context() as patch:
        point_at(patch, root)
        smoke_dir = root / "cache" / "smoke"
        corpus_llm = test_eval_corpus.FakeLLM()
        made = call(generate.main, ["--smoke", "--workers", "2"], runtime=fake_runtime(llm=corpus_llm))
        queries_path = smoke_dir / "queries.jsonl"
        written = call(
            query_generator.main,
            [
                "--corpus-dir", str(smoke_dir),
                "--out", str(queries_path),
                "--n-queries", str(N_GENERATED_QUERIES),
                "--workers", "2",
            ],
            runtime=fake_runtime(llm=test_eval_querygen.FakeLLM()),
        )
        # From here on the generated files are the dataset every CLI reads.
        patch.setattr(config, "PEOPLE_PATH", smoke_dir / "people.jsonl")
        patch.setattr(config, "CONVERSATIONS_PATH", smoke_dir / "conversations.jsonl")
        patch.setattr(config, "QUERIES_PATH", queries_path)
        sheet = call(show_queries.main, ["--split", "test"])  # no runtime: the store comes from the files
        live = fake_runtime(llm=DecliningLLM(), embedder=HashingEmbedder())
        benchmark = call(
            run.main,
            ["--split", "test", "--stage", "all", "--date", GENERATED_DATE, "--jev-runs", "1", "--workers", "2"],
            runtime=live,
        )
        people = dataset.load_people() if made.code == 0 else []
        conversations = dataset.load_conversations() if made.code == 0 else []
        queries = dataset.load_queries() if written.code == 0 else []
    return GeneratedLeg(root, smoke_dir, made, written, sheet, benchmark, people, conversations, queries)


def test_generate_smoke_writes_a_corpus_the_dataset_loads(generated: GeneratedLeg) -> None:
    assert generated.generated.code == 0, generated.generated.err
    assert (generated.smoke_dir / "people.jsonl").exists() and (generated.smoke_dir / "conversations.jsonl").exists()
    assert len(generated.people) == generate.SMOKE_PEOPLE
    assert sum(person.is_heavy for person in generated.people) == len(generate.SMOKE_HEAVY_COUNTS)
    assert len(generated.conversations) == generate.SMOKE_TOTAL
    assert all(record.meta.get("person_name") for record in generated.conversations)
    out = generated.generated.out
    assert f"written to {generated.smoke_dir}" in out and "validation: ok" in out
    assert "hosted spend this run: $0.00" in out


def test_queries_cli_writes_queries_over_the_generated_corpus(generated: GeneratedLeg) -> None:
    assert generated.written.code == 0, generated.written.err
    queries = generated.queries
    assert len(queries) == N_GENERATED_QUERIES
    assert {query.split for query in queries} == {"dev", "test"}
    assert {query.type for query in queries} == set(query_generator.scaled_counts(N_GENERATED_QUERIES))
    by_id = {record.episode_id: record for record in generated.conversations}
    for query in queries:
        assert all(episode_id in by_id for episode_id in query.gold_episode_ids), query.query_id
        if query.gold_person_id is not None:
            assert all(by_id[episode_id].person_id == query.gold_person_id for episode_id in query.gold_episode_ids)
        assert query.answerable == bool(query.gold_episode_ids) == (query.gold_answer is not None)
    assert f"wrote {N_GENERATED_QUERIES} queries to {generated.smoke_dir / 'queries.jsonl'}" in generated.written.out


def test_show_queries_renders_the_generated_sheet_from_the_files(generated: GeneratedLeg) -> None:
    assert generated.sheet.code == 0, generated.sheet.err
    shown = [query for query in generated.queries if query.split == "test"]
    assert shown and all(query.query_id in generated.sheet.out for query in shown)
    assert f"({len(shown)} of {len(generated.queries)} queries shown)" in generated.sheet.out


def test_runner_benchmarks_the_generated_files_end_to_end(generated: GeneratedLeg) -> None:
    assert generated.benchmark.code == 0, generated.benchmark.err
    queries = [query for query in generated.queries if query.split == "test"]
    answerable = [query for query in queries if query.answerable]
    logs = run_dir(generated.root, GENERATED_DATE, "test")
    ranker_records = read_records(logs / run.RANKER_RECORDS, RankerRecord.from_json)
    system_records = read_records(logs / run.SYSTEM_RECORDS, SystemRecord.from_json)
    assert {(r.query_id, r.ranker, r.run) for r in ranker_records} == {
        (query.query_id, name, 0) for query in answerable for name in config.RANKER_NAMES
    }
    assert len(system_records) == len(queries) * len(config.RANKER_NAMES) * N_CELLS
    # The model extracts no name and declines, so every query ends on the declined path,
    # ranked over the whole corpus, and every answer to a query with a gold answer is graded.
    assert {record.path for record in system_records} == {"declined"}
    assert {record.pool_size for record in ranker_records} <= set(range(1, generate.SMOKE_TOTAL + 1))
    assert all(record.verdict == "incorrect" for record in system_records if record.gold_ids)
    results, markdown = results_of(generated.root, GENERATED_DATE)
    assert results["meta"]["n_ranker_records"] == len(ranker_records)
    assert results["meta"]["n_system_records"] == len(system_records)
    assert set(results["ranker_comparison"]) == set(config.RANKER_NAMES)
    assert all(section in markdown for section in SECTIONS)


# --------------------------------------------------------------------------- #
# Leg 2: the fixture corpus through the study, pooling, the runner, validation #
# --------------------------------------------------------------------------- #
def fixture_queries() -> list[Query]:
    """The fixture queries as the dev split, with one label missing and one query to flag.

    q-multi is labelled with the offsite conversation only, and q-offsite-none
    files the same question as unanswerable although the budget conversation's
    dialogue answers it.
    """

    queries = [
        replace(record, split="dev", gold_episode_ids=["e-offsite"], gold_answer=VENUE_ANSWER)
        if record is MULTI
        else replace(record, split="dev")
        for record in QUERIES
    ]
    queries.append(
        replace(MULTI, query_id=FLAGGED_ID, split="dev", type="unanswerable", gold_episode_ids=[], gold_answer=None)
    )
    return queries


@dataclass
class FixtureLeg:
    root: Path
    study: Outcome
    built_models: list[str]
    pooled: Outcome
    file_after_pooling: str
    pooled_again: Outcome
    file_after_repooling: str
    sheet: Outcome
    first_run: Outcome
    first_results: dict[str, Any]
    first_markdown: str
    resumed: Outcome
    new_calls: dict[str, int]
    sampled: Outcome
    scored: Outcome
    reported: Outcome

    @property
    def pin(self) -> dict[str, Any]:
        return read_json(self.root / "jev_scoring.json")

    @property
    def queries(self) -> dict[str, Query]:
        return {record.query_id: record for record in dataset.load_queries(path=self.root / "data" / "queries.jsonl")}

    def ranker_records(self) -> list[RankerRecord]:
        return read_records(run_dir(self.root, DATE, "dev") / run.RANKER_RECORDS, RankerRecord.from_json)

    def system_records(self) -> list[SystemRecord]:
        return read_records(run_dir(self.root, DATE, "dev") / run.SYSTEM_RECORDS, SystemRecord.from_json)


@pytest.fixture(scope="module")
def fixture_leg(tmp_path_factory: pytest.TempPathFactory) -> FixtureLeg:
    root = tmp_path_factory.mktemp("fixture")
    built_models: list[str] = []

    def fake_build(settings: config.Settings, cache: Any = None, spend: Any = None) -> FakeJev:
        built_models.append(settings.jev_model)
        return FakeJev(model=f"{settings.jev_model}{SNAPSHOT_SUFFIX}")

    with pytest.MonkeyPatch.context() as patch:
        point_at(patch, root)
        patch.setattr(jev, "build_jev", fake_build)
        dataset.save_people(PEOPLE)
        dataset.save_conversations(CONVERSATIONS)
        dataset.save_queries(fixture_queries())
        queries_path = root / "data" / "queries.jsonl"

        study = call(
            jev_study.main,
            ["--split", "dev", "--pin", "--batch-sizes", "16", "--out", str(root / "cache" / "jev_study.json")],
            runtime=fake_runtime(),
        )
        pooled = call(pool_labels.main, ["--split", "dev", "--workers", "2"], runtime=fake_runtime())
        file_after_pooling = queries_path.read_text(encoding="utf-8")
        # The printed `--regenerate` replaces a flagged query the generator wrote (its id is a
        # slot of the per-type counts; test_eval_querygen's CLI test runs it over a pooled file).
        # These queries are written by hand, so the generator cannot regenerate them: the
        # flagged one is dropped instead, and pooling then finds nothing new.
        dataset.save_queries(
            [record for record in dataset.load_queries() if record.query_id != FLAGGED_ID]
        )
        pooled_again = call(pool_labels.main, ["--split", "dev", "--workers", "2"], runtime=fake_runtime())
        file_after_repooling = queries_path.read_text(encoding="utf-8")
        sheet = call(show_queries.main, ["--split", "dev", "--out", str(root / "cache" / "review.txt")])

        live = fake_runtime()  # built after the pin, so it reads it, as the real command would
        live.jev = FakeJev(model=f"{live.jev_model}{SNAPSHOT_SUFFIX}")  # serves the pinned model, as the client would
        rank_calls = count_rank_calls(live)
        argv = ["--split", "dev", "--stage", "all", "--date", DATE, "--jev-runs", "2", "--workers", "2"]
        first_run = call(run.main, argv, runtime=live)
        first_results, first_markdown = results_of(root, DATE) if first_run.code == 0 else ({}, "")
        before = (len(live.llm.calls), len(live.jev.requests), sum(rank_calls.values()))
        resumed = call(run.main, argv, runtime=live)
        after = (len(live.llm.calls), len(live.jev.requests), sum(rank_calls.values()))
        new_calls = dict(zip(("llm", "jev", "rank"), (b - a for a, b in zip(before, after))))

        records_path = run_dir(root, DATE, "dev") / run.SYSTEM_RECORDS
        sample_path, labels_path = root / "cache" / "sample.json", root / "cache" / "labels.json"
        sampled = call(judge_validation.main, ["sample", "--records", str(records_path), "--out", str(sample_path)])
        labels = {item["key"]: item["judge_verdict"] for item in read_json(sample_path)} if sampled.code == 0 else {}
        if labels:  # one human label disagrees with the judge
            first = min(labels)
            labels[first] = next(verdict for verdict in VERDICTS if verdict != labels[first])
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        scored = call(
            judge_validation.main,
            ["score", "--sample", str(sample_path), "--labels", str(labels_path)],  # --out defaults to the report's path
        )
        reported = call(run.main, ["--split", "dev", "--stage", "report", "--date", DATE, "--jev-runs", "2"], runtime=live)
    return FixtureLeg(
        root, study, built_models, pooled, file_after_pooling, pooled_again, file_after_repooling,
        sheet, first_run, first_results, first_markdown, resumed, new_calls, sampled, scored, reported,
    )


def test_jev_study_pins_what_a_fresh_runtime_reads(fixture_leg: FixtureLeg, monkeypatch: pytest.MonkeyPatch) -> None:
    assert fixture_leg.study.code == 0, fixture_leg.study.err
    assert sorted(set(fixture_leg.built_models)) == sorted(jev_study.default_models("openrouter"))
    study = read_json(fixture_leg.root / "cache" / "jev_study.json")
    assert study["cells"] and study["pin"]["rankers"] == list(config.JEV_RANKERS)
    pin = fixture_leg.pin
    scoring = JevScoringConfig.from_json({k: v for k, v in pin.items() if k not in runtime_module.JEV_PIN_KEYS})
    assert pin["model"] == study["pin"]["model"] and scoring.label == study["pin"]["variant"]
    assert pin["gateway"] == "openrouter" and not {"concurrency", "salt"} & set(pin)
    assert scoring.mode != "listwise"
    assert f"Pinned {scoring.label} on {pin['model']} to {fixture_leg.root / 'jev_scoring.json'}" in fixture_leg.study.out

    monkeypatch.setattr(runtime_module, "JEV_SCORING_PATH", fixture_leg.root / "jev_scoring.json")
    monkeypatch.delenv("EVAL_JEV_MODEL", raising=False)
    fresh = Runtime(settings(), store_kind="memory", use_cache=False)
    assert fresh.jev_model == pin["model"]
    assert fresh.jev_scoring("run0") == replace(scoring, concurrency=fresh.settings.jev_concurrency, salt="run0")


def test_pool_labels_adds_the_missed_label_and_flags_the_answerable_unanswerable(fixture_leg: FixtureLeg) -> None:
    assert fixture_leg.pooled.code == pool_labels.EXIT_FLAGGED, fixture_leg.pooled.err
    lines = [json.loads(line) for line in fixture_leg.file_after_pooling.splitlines() if line.strip()]
    after = {row["query_id"]: Query.from_json(row) for row in lines}
    multi = after["q-multi"]
    assert multi.gold_episode_ids == ["e-offsite", "e-budget"]  # oldest first
    assert multi.gold_answer == OFFSITE_ANSWER
    assert multi.meta["pooling"]["added"] == ["e-budget"]
    assert multi.meta["pooling"]["previous_gold_answer"] == VENUE_ANSWER
    flagged = after[FLAGGED_ID]
    assert flagged.type == "unanswerable" and flagged.gold_episode_ids == [] and flagged.gold_answer is None
    assert flagged.meta["pooling"]["flagged"] == ["e-budget"]
    out = fixture_leg.pooled.out
    assert "added e-budget; gold answer rewritten" in out and "flagged e-budget" in out
    assert f"--regenerate {FLAGGED_ID}" in out


def test_pool_labels_changes_nothing_once_the_flagged_query_is_gone(fixture_leg: FixtureLeg) -> None:
    assert fixture_leg.pooled_again.code == 0, fixture_leg.pooled_again.err
    before = [line for line in fixture_leg.file_after_pooling.splitlines() if FLAGGED_ID not in line]
    assert fixture_leg.file_after_repooling.splitlines() == before
    assert "labels added            0" in fixture_leg.pooled_again.out
    assert "flagged unanswerable    0" in fixture_leg.pooled_again.out
    assert set(fixture_leg.queries) == {record.query_id for record in QUERIES}


def test_show_queries_reads_the_dataset_files_without_a_runtime(fixture_leg: FixtureLeg) -> None:
    assert fixture_leg.sheet.code == 0, fixture_leg.sheet.err
    sheet = fixture_leg.sheet.out
    assert all(record.query_id in sheet for record in QUERIES)
    assert f"({len(QUERIES)} of {len(QUERIES)} queries shown)" in sheet
    written = (fixture_leg.root / "cache" / "review.txt").read_text(encoding="utf-8")
    assert written and sheet.startswith(written)


def test_run_all_stages_writes_a_report_with_every_section(fixture_leg: FixtureLeg) -> None:
    assert fixture_leg.first_run.code == 0, fixture_leg.first_run.err
    results, markdown = fixture_leg.first_results, fixture_leg.first_markdown
    meta = results["meta"]
    n_ranker = len(ANSWERABLE) * (len(config.RANKER_NAMES) - len(config.JEV_RANKERS) + 2 * len(config.JEV_RANKERS))
    assert (meta["date"], meta["split"], meta["store"], meta["jev_runs"]) == (DATE, "dev", "memory", 2)
    assert meta["n_ranker_records"] == n_ranker == len(fixture_leg.ranker_records())
    assert meta["n_system_records"] == len(QUERIES) * len(config.RANKER_NAMES) * N_CELLS == len(fixture_leg.system_records())
    pin = fixture_leg.pin
    assert meta["models"]["jev"] == pin["model"]
    scoring = JevScoringConfig.from_json({k: v for k, v in pin.items() if k not in runtime_module.JEV_PIN_KEYS})
    assert meta["models"]["jev_scoring"] == scoring.label
    # The run directory is bound to the dataset and keeps the environment the records were made under.
    logs = run_dir(fixture_leg.root, DATE, "dev")
    assert (logs / run.DATASET_MARKER).exists()
    recorded = read_json(logs / run.RUN_META)
    # The served snapshot comes from the ranker records, not from meta.json: the pinned
    # model's, which is also what the study's pin recorded, so the report has no note on it.
    assert meta["models"].pop("jev_snapshots") == [f"{pin['model']}{SNAPSHOT_SUFFIX}"]
    assert pin["served_models"] == [f"{pin['model']}{SNAPSHOT_SUFFIX}"]
    assert recorded["models"] == meta["models"] and recorded["hardware"] == meta["hardware"]
    assert recorded["jev_pin"] == pin and recorded["store"] == "memory"
    assert "notes" not in results
    assert set(results["ranker_comparison"]) == set(config.RANKER_NAMES)
    assert set(results["jev_variance"]) == set(config.JEV_RANKERS)
    assert set(results["system_eval"]) == set(config.RANKER_NAMES)
    assert results["recommendation"]["ranker"]["choice"] in config.RANKER_NAMES
    assert "judge_validation" not in results  # scored only after this run
    assert markdown.startswith("# Retrieval ranker benchmark")
    assert all(section in markdown for section in SECTIONS)
    assert report.NOT_RUN not in markdown
    assert str(fixture_leg.root / "results" / f"{DATE}.md") in fixture_leg.first_run.out


def test_second_invocation_resumes_with_no_new_hosted_calls(fixture_leg: FixtureLeg) -> None:
    assert fixture_leg.resumed.code == 0, fixture_leg.resumed.err
    assert fixture_leg.new_calls == {"llm": 0, "jev": 0, "rank": 0}
    results, _ = results_of(fixture_leg.root, DATE)
    assert results["ranker_comparison"] == fixture_leg.first_results["ranker_comparison"]
    assert results["system_eval"] == fixture_leg.first_results["system_eval"]
    assert "hosted spend this command: $0.0000" in fixture_leg.resumed.out


def test_judge_validation_round_trip_lands_in_the_report(fixture_leg: FixtureLeg) -> None:
    assert fixture_leg.sampled.code == 0, fixture_leg.sampled.err
    items = read_json(fixture_leg.root / "cache" / "sample.json")
    with_gold = {record.query_id for record in QUERIES if record.gold_answer}
    assert 0 < len(items) <= min(10, len(with_gold))
    assert len({item["query_id"] for item in items}) == len(items)
    assert all(item["query_id"] in with_gold and item["judge_verdict"] in VERDICTS for item in items)
    assert fixture_leg.scored.code == 0, fixture_leg.scored.err
    validation = read_json(fixture_leg.root / "data" / "judge_validation.json")  # config.JUDGE_VALIDATION_PATH
    assert validation["n"] == len(items)
    assert validation["agreement_rate"] == pytest.approx((len(items) - 1) / len(items))
    assert [item["key"] for item in validation["disagreements"]] == [min(item["key"] for item in items)]
    assert f"agree on {len(items) - 1} of {len(items)} items" in fixture_leg.scored.out
    assert fixture_leg.reported.code == 0, fixture_leg.reported.err
    results, markdown = results_of(fixture_leg.root, DATE)
    assert results["judge_validation"] == validation
    assert "## Judge validation" in markdown
    assert results["ranker_comparison"] == fixture_leg.first_results["ranker_comparison"]


def test_records_agree_with_a_hand_check_of_two_queries(fixture_leg: FixtureLeg) -> None:
    """q-pottery (detail only: the fallback path) and q-dana-week (last week: the time filter)."""

    by_cell = {(r.query_id, r.ranker, r.k, r.k_prime): r for r in fixture_leg.system_records()}
    for name in config.RANKER_NAMES:
        for k in config.SYSTEM_KS:
            for k_prime in config.FALLBACK_KS:
                # The pottery evidence is in the dialogue only: the summary attempt fails, the
                # fallback answers, and the cost is one call per stage.
                pottery = by_cell[("q-pottery", name, k, k_prime)]
                assert pottery.path == "fallback" and pottery.needed_dialogue is True
                assert pottery.verdict == "correct" and pottery.time_filter_ok is None
                calls = {stage: usage.calls for stage, usage in pottery.usage_by_stage.items()}
                jev_requests = 1 if name in config.JEV_RANKERS else 0  # one Jev request for a pool of one
                assert calls == {"extraction": 1, "rank": jev_requests, "answer_summary": 1, "answer_dialogue": 1}
                assert pottery.retrieved_ids == ["e-pottery"]  # Lena's only conversation
                # "Last week", padded, keeps the vendor conversation; the summary answers it.
                week = by_cell[("q-dana-week", name, k, k_prime)]
                assert week.path == "summary" and week.needed_dialogue is False
                assert week.time_filter_ok is True and week.name_correct and week.person_correct
                assert week.usage_by_stage["answer_dialogue"].calls == 0
                assert week.usage_by_stage["answer_dialogue"].cost_usd == 0.0
                assert "e-vendor" in week.retrieved_ids[:2] and len(week.retrieved_ids) == 2
    for record in fixture_leg.ranker_records():
        if record.query_id in ("q-pottery", "q-dana-week"):
            assert set(record.gold_ids) <= set(record.ranked_ids[:5])  # Recall@5 = 1 for every ranker
            assert record.pool_size == {"q-pottery": 1, "q-dana-week": 2}[record.query_id]
    # The report's tables show the same: Recall@k by type, the fallback and time-filter rates,
    # and the per-path cost that sums the stages of the one detail-only query.
    results, markdown = results_of(fixture_leg.root, DATE)
    for name in config.RANKER_NAMES:
        pottery_cost = sum(u.cost_usd for u in by_cell[("q-pottery", name, 5, 3)].usage_by_stage.values())
        by_type = results["ranker_comparison"][name]["by_type"]
        for k in config.RECALL_KS:
            assert by_type["detail_only"]["recall"][str(k)]["mean"] == 1.0
            assert by_type["temporal"]["recall"][str(k)]["mean"] == 1.0
        cell = results["system_eval"][name]["5"]["3"]
        assert cell["by_type"]["detail_only"]["fallback_rate"]["mean"] == 1.0
        assert cell["by_type"]["temporal"]["fallback_rate"]["mean"] == 0.0
        assert cell["by_type"]["temporal"]["time_filter_recall"]["mean"] == 1.0
        assert cell["overall"]["time_filter_recall"] == {"mean": 1.0, "lo": 1.0, "hi": 1.0, "n": 1}
        assert cell["by_type"]["detail_only"]["cost"]["fallback"]["cost_usd"] == pytest.approx(pottery_cost)
        assert cell["by_type"]["detail_only"]["cost"]["summary"]["n"] == 0
    assert "| bge_m3 | 5 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |" in markdown  # lookup, detail_only
    assert "| Time-filter recall (temporal queries) | 100.0 [100.0, 100.0] | 1 |" in markdown
