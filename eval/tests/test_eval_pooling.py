"""Gold-label pooling and the query review sheet, over the integration fixture with fakes.

The corpus, queries and fake models come from ``test_eval_integration``; the
fake relevance judge calls a conversation relevant when the query's evidence
phrase is in its dialogue, and the fake gold-answer writer returns the query's
scripted answer. Pooling runs through ``main`` with an injected ``Runtime``
over a temporary copy of the query file.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval.retrieval import config, dataset, judge, runtime
from eval.retrieval.data import pool_labels, show_queries
from eval.retrieval.interfaces import LLMResult, Usage
from eval.retrieval.records import gold_filters
from eval.retrieval.schema import QUERY_TYPES, Query, TimeRange
from eval.retrieval.store import InMemoryCorpusStore
from eval.retrieval.timeutil import describe_moment, pad_time_range
from test_eval_integration import (
    BY_ID,
    CONVERSATIONS,
    DANA,
    LAST_WEEK,
    OFFSITE_ANSWER,
    OWNER,
    PEOPLE,
    QUERIES,
    FakeEmbedder,
    FakeJev,
    FakeLLM,
    FakeScorer,
    make_store,
    settings,
    utc,
)

RANKERS = list(config.RANKER_NAMES)
VENUE_ANSWER = "The unapproved venue deposit."
MULTI = next(record for record in QUERIES if record.query_id == "q-multi")
STORYLINES = [{"storyline_id": "s-1", "title": "The Q4 budget review"}, {"storyline_id": "s-2", "title": "Hiring plan"}]


def fake_runtime() -> runtime.Runtime:
    """A Runtime whose every component is the fixture's fake, assigned before first use."""

    live = runtime.Runtime(settings(), store_kind="memory", use_cache=False)
    live.store, live.embedder, live.scorer, live.jev, live.llm = (
        make_store(),
        FakeEmbedder(),
        FakeScorer(),
        FakeJev(),
        FakeLLM(),
    )
    return live


def write(path: Path, queries: list[Query]) -> str:
    dataset.save_queries(queries, path)
    return path.read_text(encoding="utf-8")


def reload(path: Path) -> dict[str, Query]:
    return {record.query_id: record for record in dataset.load_queries(path=path)}


def missing_label() -> list[Query]:
    """The fixture queries, with q-multi labelled with the offsite conversation only."""

    return [
        replace(record, gold_episode_ids=["e-offsite"], gold_answer=VENUE_ANSWER) if record is MULTI else record
        for record in QUERIES
    ]


def not_unanswerable() -> Query:
    """q-multi's question filed as unanswerable: its evidence is in the budget conversation's dialogue."""

    return replace(MULTI, query_id="q-offsite-none", type="unanswerable", gold_episode_ids=[], gold_answer=None)


def expected_candidates(record: Query) -> int:
    """Every ranker returns its whole pool at depth 30 here, so the pool minus the gold set is judged."""

    pool = make_store().pool_ids(OWNER, *gold_filters(record))
    return len(set(pool) - set(record.gold_episode_ids))


# --------------------------------------------------------------------------- #
# pool_labels                                                                  #
# --------------------------------------------------------------------------- #
def test_pooling_adds_the_missed_label_and_rewrites_that_gold_answer(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    before = write(path, missing_label()).splitlines()
    live = fake_runtime()

    code = pool_labels.main(["--queries", str(path), "--workers", "2"], runtime=live)

    assert code == 0
    after = reload(path)
    multi = after["q-multi"]
    assert multi.gold_episode_ids == ["e-offsite", "e-budget"]  # oldest first
    assert multi.gold_answer == OFFSITE_ANSWER
    assert multi.meta["pooling"] == {
        "depth": config.LABEL_POOL_DEPTH,
        "rankers": RANKERS,
        "judged": 6,
        "added": ["e-budget"],
        "previous_gold_answer": VENUE_ANSWER,
    }
    assert list(after) == [record.query_id for record in QUERIES]  # original order
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:-1] == before[:-1] and lines[-1] != before[-1]  # only q-multi's line changed
    # The writer saw the old answer as its starting point and the full gold set, oldest first.
    (rewrite,) = live.llm.prompts("gold_answer")
    assert f"<gold_answer>\n{VENUE_ANSWER}\n</gold_answer>" in rewrite
    assert rewrite.index("venue deposit is due") < rewrite.index("Q4 budget review")
    in_scope = [record for record in missing_label() if record.type != "unknown_name"]
    judged = sum(expected_candidates(record) for record in in_scope)
    assert len(live.llm.prompts("judge_relevance")) == judged - 1  # e-offsite for q-dana-week: settled in code
    assert {salt for _, _, salt in live.jev.requests} == {pool_labels.POOLING_SALT}
    out = capsys.readouterr().out
    assert "q-multi (multi_conversation, pool 7, judged 6): added e-budget; gold answer rewritten" in out
    assert f"queries in scope        {len(in_scope)}" in out
    assert f"candidates judged       {judged}" in out
    assert "labels added            1 (multi_conversation 1)" in out
    assert "gold answers rewritten  1" in out
    assert "flagged unanswerable    0" in out
    assert "hosted spend            $0.0000 charged this run" in out
    assert f"wrote {len(QUERIES)} queries to {path} (1 changed)" in out


class UnjudgableRelevanceLLM(FakeLLM):
    """The fixture model, except that the relevance judge never gives a usable verdict on
    the (q-multi, e-budget) pair: the one the pooling would otherwise add."""

    @staticmethod
    def is_multi_budget(system: str, user: str) -> bool:
        return (
            system.startswith("You label conversations")
            and f"<question>\n{MULTI.query}\n</question>" in user
            and "Travel is over by twelve percent this quarter." in user  # e-budget's dialogue
        )

    def complete_json(self, *, model: str, system: str, user: str, **kwargs):
        if self.is_multi_budget(system, user):
            self.calls.append((kwargs.get("namespace", "default"), system, user))
            data = {"relevant": "maybe", "reason": "not a verdict the judge accepts"}
            return LLMResult(data=data, usage=Usage(calls=1), model=model, raw_text=json.dumps(data))
        return super().complete_json(model=model, system=system, user=user, **kwargs)


def test_a_candidate_the_judge_cannot_grade_stays_out_with_a_warning_and_the_pass_goes_on(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "queries.jsonl"
    before = write(path, missing_label())
    live = fake_runtime()
    live.llm = UnjudgableRelevanceLLM()

    code = pool_labels.main(["--queries", str(path), "--workers", "2"], runtime=live)

    assert code == 0
    after = reload(path)
    assert after["q-multi"].gold_episode_ids == ["e-offsite"]  # e-budget was not added
    assert "pooling" not in after["q-multi"].meta
    assert path.read_text(encoding="utf-8") == before  # nothing changed, so nothing was written
    captured = capsys.readouterr()
    assert "warning: q-multi: the relevance of e-budget was not judged (" in captured.err
    assert "treated as not relevant" in captured.err
    assert "labels added            0" in captured.out
    assert f"queries in scope        {len([r for r in missing_label() if r.type != 'unknown_name'])}" in captured.out
    # The judge made its three attempts (the last one past the cache) before giving up on that pair.
    attempts = [user for _, system, user in live.llm.calls if UnjudgableRelevanceLLM.is_multi_budget(system, user)]
    assert len(attempts) == judge.ATTEMPTS
    # Every other pair was judged once, as in a run where the judge never fails.
    clean = fake_runtime()
    assert pool_labels.main(["--queries", str(path), "--workers", "2"], runtime=clean) == 0
    assert len(live.llm.prompts("judge_relevance")) == len(clean.llm.prompts("judge_relevance")) + judge.ATTEMPTS - 1


def test_a_gateway_without_jev_pools_the_local_rankers_only(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    write(path, missing_label())
    live = fake_runtime()
    live.settings = replace(live.settings, gateway="openai", jev_url="", jev_api_key=None, jev_model="")

    assert pool_labels.main(["--queries", str(path)], runtime=live) == 0

    pooling = reload(path)["q-multi"].meta["pooling"]
    assert pooling["rankers"] == ["bge_m3", "bge_m3_reranker"] and pooling["added"] == ["e-budget"]
    assert live.jev.requests == []
    assert "over bge_m3, bge_m3_reranker:" in capsys.readouterr().out


def test_the_second_run_adds_nothing_and_leaves_the_file_byte_identical(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    write(path, missing_label())
    assert pool_labels.main(["--queries", str(path)], runtime=fake_runtime()) == 0
    first = path.read_bytes()
    live = fake_runtime()

    code = pool_labels.main(["--queries", str(path)], runtime=live)

    assert code == 0 and path.read_bytes() == first
    assert live.llm.prompts("gold_answer") == []
    assert "no changes" in capsys.readouterr().out
    assert reload(path)["q-multi"].meta["pooling"]["added"] == ["e-budget"]


def test_a_relevant_hit_on_an_unanswerable_query_is_flagged_with_exit_code_2(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    before = write(path, [*QUERIES, not_unanswerable()]).splitlines()
    live = fake_runtime()

    code = pool_labels.main(["--queries", str(path)], runtime=live)

    assert code == pool_labels.EXIT_FLAGGED == 2
    flagged = reload(path)["q-offsite-none"]
    assert flagged.gold_episode_ids == [] and flagged.gold_answer is None and flagged.type == "unanswerable"
    assert flagged.meta["pooling"] == {
        "depth": config.LABEL_POOL_DEPTH,
        "rankers": RANKERS,
        "judged": 7,
        "flagged": ["e-budget"],
    }
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:-1] == before[:-1]  # every other query was written back unchanged
    assert live.llm.prompts("gold_answer") == []
    out = capsys.readouterr().out
    assert "q-offsite-none (unanswerable, pool 7, judged 7): flagged e-budget" in out
    assert "flagged unanswerable    1" in out
    assert f"python -m eval.retrieval.data.queries --regenerate q-offsite-none --out {path}" in out
    # Until it is regenerated, every run flags it again, with the file left as it is.
    written = path.read_bytes()
    assert pool_labels.main(["--queries", str(path)], runtime=fake_runtime()) == 2
    assert path.read_bytes() == written and "no changes" in capsys.readouterr().out


def test_unknown_name_queries_are_skipped_and_a_clean_file_round_trips(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    write(path, QUERIES)
    before = path.read_bytes()
    live = fake_runtime()

    code = pool_labels.main(["--queries", str(path)], runtime=live)

    assert code == 0 and path.read_bytes() == before
    assert not any("Priya" in state["query"] for state, _, _ in live.jev.requests)
    assert not any("Priya" in prompt for prompt in live.llm.prompts("judge_relevance"))
    out = capsys.readouterr().out
    assert f"queries in scope        {len(QUERIES) - 1}" in out
    assert "labels added            0" in out and "no changes" in out


def test_conversations_outside_the_gold_time_range_are_never_added(tmp_path: Path) -> None:
    # The budget conversation (Tuesday 2026-11-03) carries the query's evidence, sits
    # inside the padded filter the rankers use, but outside the gold window itself.
    window = TimeRange(utc(2026, 10, 30), utc(2026, 11, 2, 23, 59))
    budget_start = BY_ID["e-budget"].start_time
    assert pad_time_range(window).contains(budget_start) and not window.contains(budget_start)
    temporal = replace(
        MULTI, query_id="q-offsite-window", type="temporal", gold_time_range=window,
        gold_episode_ids=["e-offsite"], gold_answer=VENUE_ANSWER,
    )
    path = tmp_path / "queries.jsonl"
    write(path, [temporal])
    before = path.read_bytes()
    live = fake_runtime()

    code = pool_labels.main(["--queries", str(path)], runtime=live)

    assert code == 0 and path.read_bytes() == before
    assert reload(path)["q-offsite-window"].gold_episode_ids == ["e-offsite"]
    assert live.llm.calls == []  # the window settled the only candidate without a call


def test_dry_run_judges_and_reports_but_writes_nothing(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    write(path, missing_label())
    before = path.read_bytes()
    live = fake_runtime()

    code = pool_labels.main(["--queries", str(path), "--dry-run"], runtime=live)

    assert code == 0 and path.read_bytes() == before
    assert len(live.llm.prompts("gold_answer")) == 1  # the would-be answer is written (and cached) too
    out = capsys.readouterr().out
    assert "q-multi (multi_conversation, pool 7, judged 6): would add e-budget; gold answer to be rewritten" in out
    assert f"dry run: {path} was not written (1 queries would change)" in out

    write(path, [not_unanswerable()])
    before = path.read_bytes()
    assert pool_labels.main(["--queries", str(path), "--dry-run"], runtime=fake_runtime()) == 2
    assert path.read_bytes() == before
    assert "would flag e-budget" in capsys.readouterr().out


def test_split_and_depth_select_what_is_pooled(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    queries = [replace(record, split="dev") if record.query_id == "q-multi" else record for record in missing_label()]
    before = write(path, queries).splitlines()

    assert pool_labels.main(["--queries", str(path), "--split", "test"], runtime=fake_runtime()) == 0
    assert path.read_text(encoding="utf-8").splitlines() == before  # q-multi is dev: not pooled
    assert "queries in scope        7" in capsys.readouterr().out

    assert pool_labels.main(["--queries", str(path), "--split", "dev", "--depth", "3"], runtime=fake_runtime()) == 0
    multi = reload(path)["q-multi"]
    assert multi.gold_episode_ids == ["e-offsite", "e-budget"] and multi.meta["pooling"]["depth"] == 3
    assert multi.meta["pooling"]["judged"] <= 4 * 3  # at most the depth from each ranker

    with pytest.raises(SystemExit):
        pool_labels.main(["--queries", str(path), "--depth", "0"], runtime=fake_runtime())
    with pytest.raises(SystemExit):
        pool_labels.main(["--queries", str(path), "--workers", "0"], runtime=fake_runtime())


def test_additions_accumulate_over_runs_and_keep_the_original_answer() -> None:
    conversations = make_store().get_conversations(["e-budget", "e-offsite", "e-vendor"])
    once = replace(
        MULTI,
        gold_episode_ids=["e-offsite", "e-vendor"],
        gold_answer="rewritten once",
        meta={
            "pooling": {
                "depth": 30,
                "rankers": RANKERS,
                "judged": 5,
                "added": ["e-vendor"],
                "previous_gold_answer": VENUE_ANSWER,
            }
        },
    )
    pool = pool_labels.Pool(pool_size=7, candidates=("e-budget", "e-hiring"))

    outcome = pool_labels._outcome(once, pool, ["e-budget"], Usage(), 50, RANKERS, conversations)

    assert outcome.added == ["e-budget"] and outcome.query.gold_episode_ids == ["e-vendor", "e-offsite", "e-budget"]
    assert outcome.query.meta["pooling"] == {
        "depth": 50,
        "rankers": RANKERS,
        "judged": 2,
        "added": ["e-vendor", "e-budget"],  # e-offsite was the generator's label, never pooling's
        "previous_gold_answer": VENUE_ANSWER,
    }
    assert once.gold_episode_ids == ["e-offsite", "e-vendor"] and "added" in once.meta["pooling"]  # untouched


# --------------------------------------------------------------------------- #
# show_queries                                                                 #
# --------------------------------------------------------------------------- #
def review_queries() -> list[Query]:
    """Every query type: the fixture plus a paraphrase, with a dev query and generator meta."""

    paraphrase = replace(
        QUERIES[0], query_id="q-budget-para", type="paraphrase", meta={"subkind": "single", "pool_size": 7}
    )
    dev = replace(QUERIES[1], split="dev")  # q-pottery, the detail-only query
    return [paraphrase, dev, *QUERIES[2:]]


def review_runtime() -> runtime.Runtime:
    live = fake_runtime()
    people = [replace(person, meta={"storylines": STORYLINES}) if person is DANA else person for person in PEOPLE]
    live.store = InMemoryCorpusStore(people, CONVERSATIONS)
    return live


def blocks(text: str) -> dict[str, str]:
    return {block.split()[1]: block for block in text.strip().split("\n\n")}


def test_show_queries_renders_every_type(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    queries = review_queries()
    assert {record.type for record in queries} == set(QUERY_TYPES)
    write(path, queries)

    code = show_queries.main(["--queries", str(path)], runtime=review_runtime())

    assert code == 0
    out = capsys.readouterr().out
    shown = blocks(out.rsplit("\n(", 1)[0])
    assert list(shown) == [record.query_id for record in queries]
    assert all(len(block.splitlines()) <= show_queries.MAX_LINES for block in shown.values())
    assert f"({len(queries)} of {len(queries)} queries shown)" in out

    para = shown["q-budget-para"]
    assert "== q-budget-para  [paraphrase/single, test]" in para
    assert "gold name:    Dana  (person: Dana Whitfield)" in para
    assert "pool size:    7" in para and "time range:   -" in para
    assert f"gold answer:  {QUERIES[0].gold_answer}" in para
    assert "gold conversations (1):" in para
    assert f"  1. {describe_moment(BY_ID['e-budget'].start_time)}, Dana Whitfield [e-budget]" in para
    assert "Dana walked through the Q4 budget review" in para and "matching turns" not in para

    detail = shown["q-pottery"]
    assert "[detail_only/-, dev]" in detail
    assert "matching turns (1):" in detail
    assert 'Lena: "The pottery class is on Thursdays at the Kiln Room."' in detail
    assert 'Lena: "Come to the open evening on the 3rd."' not in detail  # shares no word of the gold answer

    unanswerable = shown["q-car"]
    assert "storylines of Dana Whitfield (2):" in unanswerable
    assert "  - The Q4 budget review\n  - Hiring plan" in unanswerable
    assert "gold conversations" not in unanswerable and "gold answer:  -" in unanswerable
    assert "storylines of Lena Fischer (0):" in shown["q-yesterday"]

    assert "gold name:    Priya  (person: -)" in shown["q-priya"] and "gold conversations (0):" in shown["q-priya"]
    assert "gold name:    Sam  (person: -)" in shown["q-sam"]  # ambiguous: resolves to nobody
    assert "gold name:    -  (person: -)" in shown["q-gym"]
    week = shown["q-dana-week"]
    assert f"time range:   {describe_moment(LAST_WEEK.start)} to {describe_moment(LAST_WEEK.end)}" in week
    assert "gold conversations (2):" in shown["q-multi"] and "[e-offsite]" in shown["q-multi"]


def test_show_queries_respects_the_filters_and_writes_the_sheet(tmp_path: Path, capsys) -> None:
    path = tmp_path / "queries.jsonl"
    write(path, review_queries())
    live = review_runtime()

    assert show_queries.main(["--queries", str(path), "--split", "dev"], runtime=live) == 0
    assert list(blocks(capsys.readouterr().out.rsplit("\n(", 1)[0])) == ["q-pottery"]

    assert show_queries.main(["--queries", str(path), "--type", "unanswerable"], runtime=live) == 0
    assert list(blocks(capsys.readouterr().out.rsplit("\n(", 1)[0])) == ["q-car", "q-yesterday"]

    out_path = tmp_path / "sheets" / "review.txt"
    argv = ["--queries", str(path), "--ids", "q-gym", "q-sam", "--out", str(out_path)]
    assert show_queries.main(argv, runtime=live) == 0
    out = capsys.readouterr().out
    assert list(blocks(out.rsplit("\n(", 1)[0])) == ["q-sam", "q-gym"]  # file order, not argument order
    assert out.startswith(out_path.read_text(encoding="utf-8"))
    assert f"wrote the sheet to {out_path}" in out

    assert show_queries.main(["--queries", str(path), "--split", "dev", "--type", "lookup"], runtime=live) == 0
    assert capsys.readouterr().out == "(0 of 9 queries shown)\n"

    assert show_queries.main(["--queries", str(path), "--ids", "q-nope"], runtime=live) == 1
    assert "unknown query ids: q-nope" in capsys.readouterr().err


def test_show_queries_caps_a_block_at_sixty_lines() -> None:
    everything = replace(
        QUERIES[1],
        type="detail_only",
        gold_episode_ids=[record.episode_id for record in CONVERSATIONS],
        gold_answer=" ".join(turn.text for record in CONVERSATIONS for turn in record.dialogue),
    )
    store = make_store()
    people = {person.person_id: person for person in store.list_people(OWNER)}

    block = show_queries.render_query(everything, store, people)

    lines = block.splitlines()
    assert len(lines) == show_queries.MAX_LINES
    assert lines[-1].startswith("  ... ") and lines[-1].endswith("more lines; see the dataset files")
