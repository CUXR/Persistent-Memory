"""The LLM judges (answer, relevance, gold-answer rewrite) and the human check of the answer judge."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from eval.retrieval import config, dataset, judge, judge_validation
from eval.retrieval.interfaces import ChatLLM, LLMResult, Usage
from eval.retrieval.schema import ANSWERABLE_TYPES, Conversation, Query, TimeRange, Turn

MODEL = "test/judge-model"
CALL_USAGE = Usage(input_tokens=120, output_tokens=15, cost_usd=0.002, latency_s=0.25, calls=1)
BLOCK = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.DOTALL)
RANKERS = ("bge_m3", "bge_m3_jev")


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


LAST_WEEK = TimeRange(utc(2026, 10, 26), utc(2026, 11, 1, 23, 59, 59))  # Monday through Sunday


class ScriptedLLM:
    """A ChatLLM that replays scripted JSON objects in order and records every call."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "namespace": namespace,
                "salt": salt,
            }
        )
        if not self.replies:
            raise AssertionError("the judge made more calls than the test scripted")
        return LLMResult(data=self.replies.pop(0), usage=CALL_USAGE, model=model)


def fields(prompt: str) -> list[tuple[str, str]]:
    """The ``<tag>`` fields of a prompt in order, after checking that it holds nothing else."""

    assert BLOCK.sub("", prompt).strip() == ""
    return BLOCK.findall(prompt)


def call_settings(call: dict[str, Any]) -> tuple[Any, ...]:
    return call["model"], call["namespace"], call["temperature"], call["salt"]


def make_conversation(
    episode_id: str = "ep-move",
    start: datetime = utc(2026, 10, 27, 9, 30),  # a Tuesday inside LAST_WEEK
    summary: str = "Sam is moving to Denver in March for a job at Acme.",
    turns: tuple[tuple[str, str], ...] = (
        ("Sam", "The movers are booked for the 14th."),
        ("Me", "Do you need a hand packing?"),
    ),
) -> Conversation:
    return Conversation(
        episode_id=episode_id,
        person_id="person-sam",
        start_time=start,
        end_time=start,
        dialogue=[Turn(speaker, text) for speaker, text in turns],
        summary=summary,
        meta={"person_name": "Sam Rivera"},
    )


def make_query(
    query_id: str = "q-move",
    text: str = "What did Sam say about his move?",
    type: str = "lookup",
    gold_answer: str | None = "Sam is moving to Denver in March.",
    gold_time_range: TimeRange | None = None,
) -> Query:
    return Query(
        query_id=query_id,
        query=text,
        type=type,
        gold_name="Sam",
        gold_time_range=gold_time_range,
        gold_episode_ids=["ep-move"],
        gold_answer=gold_answer,
    )


TEMPORAL_QUERY = make_query(
    text="What did Sam say about his move last week?", type="temporal", gold_time_range=LAST_WEEK
)


# --------------------------------------------------------------------------- #
# Answer judge                                                                 #
# --------------------------------------------------------------------------- #
def test_scripted_llm_is_a_chat_llm() -> None:
    assert isinstance(ScriptedLLM(), ChatLLM)


def test_answer_judge_sees_only_the_question_and_the_two_answers() -> None:
    llm = ScriptedLLM({"reason": "Same city and month.", "verdict": "correct"})

    verdict = judge.judge_answer(
        "Where is Sam moving?",
        "Sam is moving to Denver in March.",
        "He is relocating to Denver this March.",
        llm,
        MODEL,
    )

    assert verdict == judge.AnswerVerdict("correct", "Same city and month.", CALL_USAGE)
    [call] = llm.calls
    assert fields(call["user"]) == [
        ("question", "Where is Sam moving?"),
        ("gold_answer", "Sam is moving to Denver in March."),
        ("system_answer", "He is relocating to Denver this March."),
    ]
    assert call_settings(call) == (MODEL, "judge_answer", 0.0, "")
    prompt = (call["system"] + call["user"]).lower()
    assert "summary" not in prompt and "dialogue" not in prompt


def test_answer_judge_prompt_states_the_rubric() -> None:
    llm = ScriptedLLM({"verdict": "incorrect"})

    judge.judge_answer("Where is Sam moving?", "Denver.", config.DECLINE_ANSWER, llm, MODEL)

    system = llm.calls[0]["system"]
    assert judge.ANSWER_RUBRIC in system
    for label, rule in (
        ("correct", "contradicts none of them"),
        ("partial", "some but not all of the key facts"),
        ("incorrect", 'a refusal such as "I don\'t know"'),
    ):
        [line] = [line for line in judge.ANSWER_RUBRIC.splitlines() if line.startswith(f'- "{label}"')]
        assert rule in line
    assert '"verdict": "correct" | "partial" | "incorrect"' in system


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("correct", "correct"),
        ("partial", "partial"),
        ("incorrect", "incorrect"),
        (" Partial\n", "partial"),
        ("INCORRECT", "incorrect"),
    ],
)
def test_answer_verdict_is_read_from_the_reply(raw: str, expected: str) -> None:
    llm = ScriptedLLM({"verdict": raw, "reason": "  One fact is missing.  "})

    verdict = judge.judge_answer("q?", "gold", "answer", llm, MODEL)

    assert (verdict.verdict, verdict.reason, verdict.usage) == (expected, "One fact is missing.", CALL_USAGE)
    assert len(llm.calls) == 1


def test_answer_verdict_without_a_reason_is_still_valid() -> None:
    verdict = judge.judge_answer("q?", "gold", "answer", ScriptedLLM({"verdict": "correct"}), MODEL)

    assert (verdict.verdict, verdict.reason) == ("correct", "")


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"verdict": None},
        {"verdict": True},
        {"verdict": ["correct"]},
        {"verdict": "mostly correct"},
        {"grade": "correct", "reason": "Matches."},
    ],
)
def test_invalid_answer_verdict_is_retried_once_with_a_different_salt(invalid: dict[str, Any]) -> None:
    llm = ScriptedLLM(invalid, {"verdict": "partial", "reason": "One fact is missing."})

    verdict = judge.judge_answer("q?", "gold", "answer", llm, MODEL)

    assert (verdict.verdict, verdict.reason) == ("partial", "One fact is missing.")
    assert verdict.usage == CALL_USAGE + CALL_USAGE
    first, second = llm.calls
    assert first["salt"] != second["salt"]
    assert {**first, "salt": None} == {**second, "salt": None}


def test_answer_judge_raises_when_every_attempt_is_invalid() -> None:
    llm = ScriptedLLM({"verdict": "maybe"}, {"verdict": "unsure"}, {"verdict": "partially correct"})

    with pytest.raises(judge.JudgeOutputError, match="judge_answer.*maybe.*unsure.*partially"):
        judge.judge_answer("q?", "gold", "answer", llm, MODEL)

    assert len(llm.calls) == judge.ATTEMPTS == 3
    first, second, third = (call["salt"] for call in llm.calls)
    assert (first, second) == ("", judge.RETRY_SALT)
    assert third.startswith(f"{judge.RETRY_SALT}:") and third != judge.RETRY_SALT


def test_the_last_attempt_is_never_a_cache_hit() -> None:
    # The salt of the third attempt is new on every call, so a rerun does not
    # replay two cached invalid replies and fail the same query for good.
    runs = []
    for _ in range(2):
        llm = ScriptedLLM({"verdict": "maybe"}, {"verdict": "unsure"}, {"verdict": "partially correct"})
        with pytest.raises(judge.JudgeOutputError):
            judge.judge_answer("q?", "gold", "answer", llm, MODEL)
        runs.append([call["salt"] for call in llm.calls])

    first_run, second_run = runs
    assert first_run[:2] == second_run[:2] == ["", judge.RETRY_SALT]  # replayable on a resume
    assert first_run[2] != second_run[2]


class CachingLLM(ScriptedLLM):
    """A ScriptedLLM behind a response cache keyed the way the real client keys it."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        super().__init__(*replies)
        self.store: dict[tuple[str, str, str, str], LLMResult] = {}

    def complete_json(self, **request: Any) -> LLMResult:
        key = (request["namespace"], request["system"], request["user"], request["salt"])
        if key not in self.store:
            self.store[key] = super().complete_json(**request)
        return self.store[key]


def test_a_rerun_gets_past_two_cached_invalid_replies() -> None:
    llm = CachingLLM({"verdict": "maybe"}, {"verdict": "unsure"}, {"verdict": "partially correct"})
    with pytest.raises(judge.JudgeOutputError):
        judge.judge_answer("q?", "gold", "answer", llm, MODEL)
    assert len(llm.calls) == 3

    llm.replies.append({"verdict": "correct", "reason": "Same facts."})
    verdict = judge.judge_answer("q?", "gold", "answer", llm, MODEL)

    assert (verdict.verdict, verdict.reason) == ("correct", "Same facts.")
    assert len(llm.calls) == 4  # the two fixed-salt attempts replayed; one fresh call
    assert verdict.usage == CALL_USAGE + CALL_USAGE + CALL_USAGE


@pytest.mark.parametrize("gold_answer", [None, "", "   "])
def test_answer_judge_refuses_to_grade_without_a_gold_answer(gold_answer: str | None) -> None:
    llm = ScriptedLLM()

    with pytest.raises(ValueError, match="gold answer"):
        judge.judge_answer("q?", gold_answer, "answer", llm, MODEL)

    assert llm.calls == []


# --------------------------------------------------------------------------- #
# Relevance judge                                                              #
# --------------------------------------------------------------------------- #
def test_relevance_judge_sees_the_question_the_gold_answer_and_the_whole_conversation() -> None:
    llm = ScriptedLLM({"reason": "Gives the moving date.", "relevant": True})
    conversation = make_conversation()

    verdict = judge.judge_relevance(make_query(), conversation, llm, MODEL)

    assert verdict == judge.RelevanceVerdict(True, "Gives the moving date.", CALL_USAGE)
    [call] = llm.calls
    question, gold_answer, (tag, body) = fields(call["user"])
    assert question == ("question", "What did Sam say about his move?")
    assert gold_answer == ("gold_answer", "Sam is moving to Denver in March.")
    assert tag == "conversation"
    assert fields(body) == [
        ("date", "Tuesday 2026-10-27 09:30 UTC"),
        ("person", "Sam Rivera"),
        ("summary", "Sam is moving to Denver in March for a job at Acme."),
        ("dialogue", 'Sam: "The movers are booked for the 14th."\nMe: "Do you need a hand packing?"'),
    ]
    assert call_settings(call) == (MODEL, "judge_relevance", 0.0, "")


def test_relevance_judge_sees_the_time_window_only_when_the_query_has_one() -> None:
    llm = ScriptedLLM({"relevant": True}, {"relevant": True})
    conversation = make_conversation()

    judge.judge_relevance(TEMPORAL_QUERY, conversation, llm, MODEL)
    judge.judge_relevance(make_query(), conversation, llm, MODEL)

    with_window, without_window = (dict(fields(call["user"])) for call in llm.calls)
    assert list(with_window) == ["question", "gold_answer", "time_window", "conversation"]
    assert with_window["time_window"] == "Monday 2026-10-26 00:00 UTC to Sunday 2026-11-01 23:59 UTC"
    assert list(without_window) == ["question", "gold_answer", "conversation"]
    assert "2026-10-26" not in llm.calls[1]["user"] and "2026-11-01" not in llm.calls[1]["user"]


def test_relevance_judge_gets_no_gold_answer_field_for_a_query_without_one() -> None:
    llm = ScriptedLLM({"relevant": False, "reason": "Small talk only."})

    verdict = judge.judge_relevance(
        make_query(type="unanswerable", gold_answer=None), make_conversation(), llm, MODEL
    )

    assert (verdict.relevant, verdict.reason) == (False, "Small talk only.")
    assert [tag for tag, _ in fields(llm.calls[0]["user"])] == ["question", "conversation"]


@pytest.mark.parametrize("start", [utc(2026, 10, 25, 23, 59, 59), utc(2026, 11, 2), utc(2026, 3, 1, 12)])
def test_conversation_outside_the_time_window_is_not_relevant_and_costs_no_call(start: datetime) -> None:
    llm = ScriptedLLM()

    verdict = judge.judge_relevance(TEMPORAL_QUERY, make_conversation(start=start), llm, MODEL)

    assert verdict == judge.RelevanceVerdict(False, judge.OUTSIDE_WINDOW_REASON, Usage())
    assert llm.calls == []


@pytest.mark.parametrize("start", [LAST_WEEK.start, LAST_WEEK.end])
def test_conversation_on_the_edge_of_the_time_window_goes_to_the_judge(start: datetime) -> None:
    llm = ScriptedLLM({"relevant": True, "reason": "Gives the moving date."})

    verdict = judge.judge_relevance(TEMPORAL_QUERY, make_conversation(start=start), llm, MODEL)

    assert verdict.relevant is True
    assert len(llm.calls) == 1


@pytest.mark.parametrize("invalid", [{}, {"relevant": None}, {"relevant": "true"}, {"relevant": 1}])
def test_invalid_relevance_output_is_retried_once_then_raises(invalid: dict[str, Any]) -> None:
    llm = ScriptedLLM(invalid, {"relevant": False, "reason": "A different trip."})

    verdict = judge.judge_relevance(make_query(), make_conversation(), llm, MODEL)

    assert (verdict.relevant, verdict.reason) == (False, "A different trip.")
    assert verdict.usage == CALL_USAGE + CALL_USAGE
    assert [call["salt"] for call in llm.calls] == ["", judge.RETRY_SALT]
    assert llm.calls[0]["user"] == llm.calls[1]["user"]

    stubborn = ScriptedLLM(invalid, invalid, invalid)
    with pytest.raises(judge.JudgeOutputError, match="judge_relevance"):
        judge.judge_relevance(make_query(), make_conversation(), stubborn, MODEL)
    assert len(stubborn.calls) == judge.ATTEMPTS


def test_relevance_judge_needs_the_person_name() -> None:
    conversation = make_conversation()
    conversation.meta.clear()
    llm = ScriptedLLM()

    with pytest.raises(ValueError, match=r"ep-move has no meta\['person_name'\]"):
        judge.judge_relevance(make_query(), conversation, llm, MODEL)

    assert llm.calls == []


# --------------------------------------------------------------------------- #
# Gold-answer rewrite                                                          #
# --------------------------------------------------------------------------- #
def test_gold_answer_is_rewritten_from_the_whole_gold_set_oldest_first() -> None:
    older = make_conversation(
        episode_id="ep-plan",
        start=utc(2026, 9, 1, 8, 0),
        summary="Sam is thinking about leaving Boston.",
        turns=(("Sam", "I might take the Denver offer."),),
    )
    newer = make_conversation()
    llm = ScriptedLLM({"answer": "  Sam took the Denver offer and moves in March.  "})

    answer, usage = judge.write_gold_answer(make_query(), [newer, older], llm, MODEL)

    assert (answer, usage) == ("Sam took the Denver offer and moves in March.", CALL_USAGE)
    [call] = llm.calls
    question, gold_answer, note, first, second = fields(call["user"])
    assert question == ("question", "What did Sam say about his move?")
    assert gold_answer == ("gold_answer", "Sam is moving to Denver in March.")
    assert note == ("note", judge.SUMMARIES_ONLY_NOTE)
    assert (first[0], second[0]) == ("conversation", "conversation")
    # A lookup's gold answer was written from the summary alone (the writer prompt
    # forbids dialogue-only facts), so the rewrite sees the summaries alone too.
    assert fields(first[1]) == [
        ("date", "Tuesday 2026-09-01 08:00 UTC"),
        ("person", "Sam Rivera"),
        ("summary", "Sam is thinking about leaving Boston."),
    ]
    assert dict(fields(second[1]))["summary"] == "Sam is moving to Denver in March for a job at Acme."
    assert "Denver offer" not in call["user"]  # said in the dialogue only
    assert call_settings(call) == (MODEL, "gold_answer", 0.0, "")
    for requirement in ("complete", "grounded", "short"):
        assert f"- {requirement}:" in call["system"]

    reordered = ScriptedLLM({"answer": "Sam moves in March."})
    judge.write_gold_answer(make_query(), [older, newer], reordered, MODEL)
    assert reordered.calls[0]["user"] == call["user"]


@pytest.mark.parametrize("type", [type for type in ANSWERABLE_TYPES if type != judge.DIALOGUE_TYPE])
def test_gold_answer_of_a_summary_type_is_rewritten_from_the_summaries_only(type: str) -> None:
    llm = ScriptedLLM({"answer": "Sam moves in March."})

    judge.write_gold_answer(make_query(type=type), [make_conversation()], llm, MODEL)

    [call] = llm.calls
    tags = fields(call["user"])
    assert [tag for tag, _ in tags] == ["question", "gold_answer", "note", "conversation"]
    assert dict(tags)["note"] == judge.SUMMARIES_ONLY_NOTE
    assert [tag for tag, _ in fields(dict(tags)["conversation"])] == ["date", "person", "summary"]


def test_gold_answer_of_a_detail_query_is_rewritten_with_the_dialogues() -> None:
    llm = ScriptedLLM({"answer": "The movers are booked for the 14th."})

    judge.write_gold_answer(make_query(type="detail_only"), [make_conversation()], llm, MODEL)

    [call] = llm.calls
    tags = fields(call["user"])
    assert [tag for tag, _ in tags] == ["question", "gold_answer", "conversation"]  # no note
    assert fields(dict(tags)["conversation"]) == [
        ("date", "Tuesday 2026-10-27 09:30 UTC"),
        ("person", "Sam Rivera"),
        ("summary", "Sam is moving to Denver in March for a job at Acme."),
        ("dialogue", 'Sam: "The movers are booked for the 14th."\nMe: "Do you need a hand packing?"'),
    ]


def test_gold_answer_prompt_carries_the_time_window_and_no_missing_gold_answer() -> None:
    llm = ScriptedLLM({"answer": "Sam moves in March."}, {"answer": "Sam moves in March."})

    judge.write_gold_answer(TEMPORAL_QUERY, [make_conversation()], llm, MODEL)
    judge.write_gold_answer(make_query(gold_answer=None), [make_conversation()], llm, MODEL)

    temporal, without_gold = ([tag for tag, _ in fields(call["user"])] for call in llm.calls)
    assert temporal == ["question", "gold_answer", "time_window", "note", "conversation"]
    assert without_gold == ["question", "note", "conversation"]


def test_gold_answer_needs_conversations_and_a_non_empty_answer() -> None:
    unused = ScriptedLLM()
    with pytest.raises(ValueError, match="q-move: a gold answer needs at least one conversation"):
        judge.write_gold_answer(make_query(), [], unused, MODEL)
    assert unused.calls == []

    llm = ScriptedLLM({"answer": "   "}, {"answer": "Sam moves in March."})
    answer, usage = judge.write_gold_answer(make_query(), [make_conversation()], llm, MODEL)
    assert (answer, usage) == ("Sam moves in March.", CALL_USAGE + CALL_USAGE)
    assert [call["salt"] for call in llm.calls] == ["", judge.RETRY_SALT]

    stubborn = ScriptedLLM({}, {"answer": 3}, {"answer": ""})
    with pytest.raises(judge.JudgeOutputError, match="gold_answer"):
        judge.write_gold_answer(make_query(), [make_conversation()], stubborn, MODEL)
    assert len(stubborn.calls) == judge.ATTEMPTS


# --------------------------------------------------------------------------- #
# Validation sample                                                            #
# --------------------------------------------------------------------------- #
def system_record(
    query_id: str,
    verdict: str | None,
    path: str = "summary",
    ranker: str = "bge_m3",
    k: int = 5,
    k_prime: int = 3,
) -> dict[str, Any]:
    """A system record with the fields the sampler reads (the runner's records have more)."""

    declined = path in ("declined", "empty_pool")
    return {
        "query_id": query_id,
        "type": "lookup",
        "ranker": ranker,
        "k": k,
        "k_prime": k_prime,
        "path": path,
        "final_answer": config.DECLINE_ANSWER if declined else f"{ranker} answer to {query_id}",
        "verdict": verdict,
    }


def answerable(query_id: str) -> Query:
    return make_query(
        query_id=query_id, text=f"Question {query_id}?", gold_answer=f"Gold answer to {query_id}."
    )


def population() -> tuple[list[dict[str, Any]], list[Query]]:
    """26 queries with judged answers under two rankers, plus records that must never be drawn."""

    records: list[dict[str, Any]] = []
    for index in range(12):  # correct with both rankers: two records in one stratum
        records += [system_record(f"q{index:02d}", "correct", ranker=ranker) for ranker in RANKERS]
    for index in range(12, 20):  # partial with one ranker, incorrect with the other
        records.append(system_record(f"q{index:02d}", "partial", path="fallback"))
        records.append(system_record(f"q{index:02d}", "incorrect", ranker="bge_m3_jev"))
    for index in range(20, 26):  # declined with one ranker, correct with the other: two strata
        records.append(system_record(f"q{index:02d}", "incorrect", path="declined"))
        records.append(system_record(f"q{index:02d}", "correct", ranker="bge_m3_jev"))
    records += [
        system_record("q26", None, path="declined"),  # answerable, but the judge did not grade it
        system_record("q27", None, path="unknown_name"),
        system_record("u00", None, path="declined"),  # unanswerable: no gold answer to grade against
        system_record("u01", "incorrect"),
        system_record("ghost", "correct"),  # not among the queries
    ]
    queries = [answerable(f"q{index:02d}") for index in range(28)]
    queries += [
        make_query(query_id=query_id, type="unanswerable", gold_answer=None) for query_id in ("u00", "u01")
    ]
    return records, queries


def stratum_of(record: dict[str, Any]) -> str:
    if record["path"] in ("declined", "empty_pool"):
        return "declined"
    return "correct" if record["verdict"] == "correct" else "partial_incorrect"


def strata_counts(items: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, int]:
    by_key = {judge_validation.item_key(record): record for record in records}
    return dict(Counter(stratum_of(by_key[item["key"]]) for item in items))


def other_label(verdict: str) -> str:
    return judge.VERDICTS[(judge.VERDICTS.index(verdict) + 1) % len(judge.VERDICTS)]


def test_sample_is_stratified_with_at_most_one_item_per_query() -> None:
    records, queries = population()

    items = judge_validation.sample_items(records, queries)

    assert len(items) == 10
    assert strata_counts(items, records) == {"correct": 4, "partial_incorrect": 4, "declined": 2}
    assert len({item["query_id"] for item in items}) == 10
    assert {item["query_id"] for item in items}.isdisjoint({"q26", "q27", "u00", "u01", "ghost"})
    assert [item["key"] for item in items] == sorted(item["key"] for item in items)


def test_sample_items_carry_the_answer_the_gold_answer_and_the_setting() -> None:
    records, queries = population()
    by_key = {judge_validation.item_key(record): record for record in records}
    queries_by_id = {query.query_id: query for query in queries}

    for item in judge_validation.sample_items(records, queries):
        record = by_key[item["key"]]
        query = queries_by_id[record["query_id"]]
        assert item == {
            "key": f"{record['query_id']}/{record['ranker']}/k5/kp3",
            "query_id": record["query_id"],
            "query": query.query,
            "gold_answer": query.gold_answer,
            "system_answer": record["final_answer"],
            "ranker": record["ranker"],
            "k": 5,
            "k_prime": 3,
            "judge_verdict": record["verdict"],
        }
        assert list(item)[-1] == "judge_verdict"  # below everything the labeller reads first
        if stratum_of(record) == "declined":
            assert (item["system_answer"], item["judge_verdict"]) == (config.DECLINE_ANSWER, "incorrect")


def test_sample_depends_on_the_seed_and_not_on_the_order_of_the_input() -> None:
    records, queries = population()
    items = judge_validation.sample_items(records, queries)
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)

    assert judge_validation.sample_items(records, queries) == items
    assert judge_validation.sample_items(shuffled, list(reversed(queries))) == items
    assert judge_validation.sample_items(records, queries, seed=config.SEED) == items
    assert judge_validation.sample_items(records, queries, seed=config.SEED + 1) != items


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, {"correct": 1}),
        (2, {"correct": 1, "partial_incorrect": 1}),
        (3, {"correct": 1, "partial_incorrect": 1, "declined": 1}),
        (5, {"correct": 2, "partial_incorrect": 2, "declined": 1}),
        (20, {"correct": 8, "partial_incorrect": 8, "declined": 4}),
    ],
)
def test_slots_are_split_two_two_one(n: int, expected: dict[str, int]) -> None:
    records, queries = population()

    items = judge_validation.sample_items(records, queries, n=n)

    assert strata_counts(items, records) == expected
    assert len({item["query_id"] for item in items}) == n


def test_every_stratum_appears_even_when_the_strata_share_their_queries() -> None:
    # Each stratum has two of the three queries, so a stratum-by-stratum draw can
    # leave the last stratum with no unused query.
    records = [
        system_record("a", "correct"),
        system_record("a", "incorrect", path="declined", ranker="bge_m3_jev"),
        system_record("b", "correct"),
        system_record("b", "partial", ranker="bge_m3_jev"),
        system_record("c", "incorrect"),
        system_record("c", "incorrect", path="empty_pool", ranker="bge_m3_jev"),
    ]
    queries = [answerable(query_id) for query_id in "abc"]

    for seed in range(40):
        items = judge_validation.sample_items(records, queries, n=3, seed=seed)
        assert strata_counts(items, records) == {"correct": 1, "partial_incorrect": 1, "declined": 1}
        assert [item["query_id"] for item in items] == ["a", "b", "c"]


def test_a_small_stratum_keeps_its_quota_when_a_large_one_shares_its_queries() -> None:
    # Both declined queries are also correct with the other ranker, so the correct
    # stratum must leave them alone although it could draw them.
    records = [system_record(f"c{index:02d}", "correct") for index in range(10)]
    records += [system_record(f"p{index:02d}", "partial") for index in range(10)]
    for query_id in ("d00", "d01"):
        records.append(system_record(query_id, "incorrect", path="declined"))
        records.append(system_record(query_id, "correct", ranker="bge_m3_jev"))
    queries = [answerable(query_id) for query_id in sorted({record["query_id"] for record in records})]

    for seed in range(40):
        items = judge_validation.sample_items(records, queries, seed=seed)
        assert strata_counts(items, records) == {"correct": 4, "partial_incorrect": 4, "declined": 2}


def test_a_short_stratum_passes_its_slots_on() -> None:
    records = [system_record("c0", "correct")]
    records += [system_record(f"p{index:02d}", "incorrect") for index in range(20)]
    queries = [answerable(record["query_id"]) for record in records]

    items = judge_validation.sample_items(records, queries)
    assert strata_counts(items, records) == {"correct": 1, "partial_incorrect": 9}

    few = judge_validation.sample_items(records[:4], queries)
    assert sorted(item["query_id"] for item in few) == ["c0", "p00", "p01", "p02"]


def test_only_judged_answers_to_queries_with_a_gold_answer_qualify() -> None:
    records, queries = population()
    unjudged = [record for record in records if record["verdict"] is None]
    no_gold_answer = [system_record("u01", "incorrect"), system_record("ghost", "correct")]

    assert [record["query_id"] for record in unjudged] == ["q26", "q27", "u00"]
    assert judge_validation.sample_items(unjudged, queries) == []
    assert judge_validation.sample_items(no_gold_answer, queries) == []
    with pytest.raises(ValueError, match="n must be at least 1"):
        judge_validation.sample_items(records, queries, n=0)


def test_vocabulary_matches_the_metrics_module() -> None:
    # Imported here so that a problem in metrics.py fails this test alone.
    from eval.retrieval import metrics

    assert judge.VERDICTS == metrics.VERDICTS
    assert judge_validation.DECLINE_PATHS == metrics.DECLINE_PATHS


# --------------------------------------------------------------------------- #
# Agreement                                                                    #
# --------------------------------------------------------------------------- #
def test_agreement_counts_matching_labels_and_lists_the_rest() -> None:
    records, queries = population()
    items = judge_validation.sample_items(records, queries)
    labels = {item["key"]: item["judge_verdict"] for item in items}

    assert judge_validation.agreement(items, labels) == {"n": 10, "agreement_rate": 1.0, "disagreements": []}

    disputed = [items[1], items[6]]
    for item in disputed:
        labels[item["key"]] = other_label(item["judge_verdict"])
    labels["q99/bge_m3/k5/kp3"] = "correct"  # not in the sample: ignored
    summary = judge_validation.agreement(items, labels)

    assert (summary["n"], summary["agreement_rate"]) == (10, 0.8)
    assert summary["disagreements"] == [
        {
            "key": item["key"],
            "query": item["query"],
            "gold_answer": item["gold_answer"],
            "system_answer": item["system_answer"],
            "judge_verdict": item["judge_verdict"],
            "human_label": other_label(item["judge_verdict"]),
        }
        for item in disputed
    ]


def test_agreement_needs_a_valid_label_for_every_item() -> None:
    records, queries = population()
    items = judge_validation.sample_items(records, queries, n=3)
    labels = {item["key"]: item["judge_verdict"] for item in items}
    key = items[0]["key"]

    with pytest.raises(ValueError, match=f"no human label for 1 item.*{key}"):
        judge_validation.agreement(items, {k: label for k, label in labels.items() if k != key})
    with pytest.raises(ValueError, match=f"the label for {key} must be one of .*got 'Correct'"):
        judge_validation.agreement(items, {**labels, key: "Correct"})
    assert judge_validation.agreement([], {}) == {"n": 0, "agreement_rate": None, "disagreements": []}


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #
def test_cli_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records, queries = population()
    queries_path = tmp_path / "data" / "queries.jsonl"
    dataset.save_queries(queries, queries_path)
    monkeypatch.setattr(config, "QUERIES_PATH", queries_path)
    monkeypatch.setattr(config, "JUDGE_VALIDATION_PATH", tmp_path / "data" / "judge_validation.json")
    records_path = tmp_path / "records.jsonl"
    dataset.write_jsonl(records_path, records)
    sample_path = tmp_path / "check" / "sample.json"
    labels_path = tmp_path / "check" / "labels.json"
    summary_path = tmp_path / "check" / "agreement.json"

    assert judge_validation.main(["sample", "--records", str(records_path), "--out", str(sample_path)]) == 0

    items = json.loads(sample_path.read_text(encoding="utf-8"))
    assert items == judge_validation.sample_items(records, queries)
    printed = capsys.readouterr().out
    assert f"Wrote 10 items to {sample_path}: 4 correct, 4 partial/incorrect, 2 declined." in printed
    assert judge.ANSWER_RUBRIC in printed
    assert "The questions were asked at Wednesday 2026-11-04 18:00 UTC." in printed

    labels = {item["key"]: item["judge_verdict"] for item in items}
    labels[items[0]["key"]] = other_label(items[0]["judge_verdict"])
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    score = ["score", "--sample", str(sample_path), "--labels", str(labels_path)]

    assert judge_validation.main([*score, "--out", str(summary_path)]) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == judge_validation.agreement(items, labels)
    assert (summary["n"], summary["agreement_rate"], len(summary["disagreements"])) == (10, 0.9, 1)
    printed = capsys.readouterr().out
    assert "Judge and human agree on 9 of 10 items (90%)." in printed
    disputed = items[0]
    assert f"- {disputed['key']}: judge {disputed['judge_verdict']}, human {labels[disputed['key']]}" in printed
    assert f"    system answer: {disputed['system_answer']}" in printed

    # Without --out the summary goes where the benchmark report reads it.
    summary_path.unlink()
    assert judge_validation.main(score) == 0
    assert not summary_path.exists()
    assert json.loads(config.JUDGE_VALIDATION_PATH.read_text(encoding="utf-8")) == summary
    assert f"Wrote the summary to {config.JUDGE_VALIDATION_PATH}; the report includes it from there." in capsys.readouterr().out


def test_cli_stops_on_input_it_cannot_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records, queries = population()
    queries_path = tmp_path / "queries.jsonl"
    dataset.save_queries(queries, queries_path)
    monkeypatch.setattr(config, "QUERIES_PATH", queries_path)
    records_path = tmp_path / "records.jsonl"
    dataset.write_jsonl(records_path, [record for record in records if record["verdict"] is None])
    sample_path = tmp_path / "sample.json"
    labels_path = tmp_path / "labels.json"

    with pytest.raises(SystemExit, match="error: .*records.jsonl has no judged answers"):
        judge_validation.main(["sample", "--records", str(records_path), "--out", str(sample_path)])
    assert not sample_path.exists()

    items = judge_validation.sample_items(records, queries, n=3)
    sample_path.write_text(json.dumps(items), encoding="utf-8")
    score = ["score", "--sample", str(sample_path), "--labels", str(labels_path)]
    with pytest.raises(SystemExit, match=r"error: .*labels\.json"):  # no labels file yet
        judge_validation.main(score)
    for content, message in (
        ("{}", "error: no human label for 3 item"),
        ("[]", "error: .*--labels a JSON object"),
        ('{"key": "correct",}', "error: .*labels.json is not valid JSON"),
    ):
        labels_path.write_text(content, encoding="utf-8")
        with pytest.raises(SystemExit, match=message):
            judge_validation.main(score)


def test_module_runs_as_a_script(tmp_path: Path) -> None:
    records, queries = population()
    items = judge_validation.sample_items(records, queries, n=2)
    sample_path = tmp_path / "sample.json"
    labels_path = tmp_path / "labels.json"
    sample_path.write_text(json.dumps(items), encoding="utf-8")
    labels = {item["key"]: item["judge_verdict"] for item in items}
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.retrieval.judge_validation",
            "score",
            "--sample",
            str(sample_path),
            "--labels",
            str(labels_path),
            "--out",
            str(tmp_path / "summary.json"),  # the default is the report's path under the repo
        ],
        cwd=config.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "Judge and human agree on 2 of 2 items (100%).",
        f"Wrote the summary to {tmp_path / 'summary.json'}; the report includes it from there.",
    ]
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["n"] == 2
    assert not config.JUDGE_VALIDATION_PATH.exists()  # nothing was written under the repo
