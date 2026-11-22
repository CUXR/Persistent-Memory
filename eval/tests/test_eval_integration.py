"""The benchmark's modules joined up, with fakes only for the models and the gateways.

Every module has unit tests against fakes of its neighbours. This file runs the
real modules against each other: a small corpus in an ``InMemoryCorpusStore``,
the four rankers from ``build_ranker``, the system pipeline through every path,
the judges and the label-pooling helpers, then the records, aggregates and
report built from those results, and ``Runtime`` wiring it all from settings.

The corpus: four people, two of whom share a first name, one with most of the
conversations, spread over the ten months before ``config.NOW``. The fake
models are deterministic word-overlap scorers, so the rankers agree on the
obvious cases and the scripted system model answers only when the evidence
phrase of a query is in the material it was shown.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from eval.retrieval import config, judge, judge_validation, metrics, records, report, runtime
from eval.retrieval.interfaces import (
    ChatLLM,
    CorpusStore,
    DecisionClient,
    DecisionResult,
    Embedder,
    LLMResult,
    PairScorer,
    Ranker,
    Usage,
)
from eval.retrieval.jev import JevClient
from eval.retrieval.llm import OpenAICompatLLM, fallback_prices
from eval.retrieval.local_models import BgeM3Embedder, BgeReranker
from eval.retrieval.pipelines import JevScoringConfig, build_ranker
from eval.retrieval.schema import (
    SYSTEM_PATHS,
    SYSTEM_STAGES,
    VERDICTS,
    Conversation,
    Person,
    Query,
    TimeRange,
    Turn,
)
from eval.retrieval.store import InMemoryCorpusStore, PgCorpusStore
from eval.retrieval.system import SystemPipeline, SystemResult
from eval.retrieval.timeutil import pad_time_range, time_expressions

OWNER = config.OWNER_ID
SYSTEM_MODEL = "fake/system-model"
JUDGE_MODEL = "fake/judge-model"
JEV_SNAPSHOT = "typesafe/jev-1.14-20261014"
BATCHED_SUMMARY_SEPARATOR = "\n\nconversation_summary: "  # what jev_scoring puts before a batched summary
K_GRID = (5, 10)
K_PRIME_GRID = (5, 10)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Corpus: four people, twelve conversations                                   #
# --------------------------------------------------------------------------- #
def person(person_id: str, first: str, last: str, heavy: bool = False) -> Person:
    return Person(
        person_id=person_id,
        first_name=first,
        last_name=last,
        display_name=f"{first} {last}",
        is_heavy=heavy,
    )


SAM_CARTER = person("p-sam-carter", "Sam", "Carter")
SAM_OKAFOR = person("p-sam-okafor", "Sam", "Okafor")
DANA = person("p-dana", "Dana", "Whitfield", heavy=True)
LENA = person("p-lena", "Lena", "Fischer")
PEOPLE = [SAM_OKAFOR, DANA, LENA, SAM_CARTER]  # deliberately not in name order


def conversation(
    episode_id: str,
    who: Person,
    start: datetime,
    summary: str,
    *turns: tuple[str, str],
) -> Conversation:
    return Conversation(
        episode_id=episode_id,
        person_id=who.person_id,
        start_time=start,
        end_time=start + timedelta(minutes=20),
        dialogue=[Turn(speaker=speaker, text=text) for speaker, text in turns],
        summary=summary,
        meta={"topic": episode_id[2:]},
    )


CONVERSATIONS = [
    conversation(
        "e-budget",
        DANA,
        utc(2026, 11, 3, 14),  # Tuesday of this week: after last week even once padded
        "Dana walked through the Q4 budget review and flagged the travel line as overspent, "
        "which puts the offsite in question.",
        ("Dana", "Travel is over by twelve percent this quarter."),
        ("Me", "Do we cut the offsite?"),
        ("Dana", "No, we trim conference tickets instead."),
    ),
    conversation(
        "e-offsite",
        DANA,
        utc(2026, 11, 2, 10),  # Monday of this week: outside last week, inside its padded filter
        "Dana flagged that the offsite venue deposit is due Friday and the budget line for it "
        "was never approved.",
        ("Dana", "The venue wants the deposit by Friday."),
        ("Me", "Was that line ever approved?"),
        ("Dana", "No, and finance will ask."),
    ),
    conversation(
        "e-vendor",
        DANA,
        utc(2026, 10, 28, 9, 30),  # inside last week
        "Dana said the vendor contract renews in January and the price goes up eight percent.",
        ("Dana", "The vendor contract renews in January, eight percent higher."),
        ("Me", "Can we negotiate?"),
        ("Dana", "I will try before the holidays."),
    ),
    conversation(
        "e-hiring",
        DANA,
        utc(2026, 9, 15, 16),
        "Dana shared the hiring plan: two engineers in the first quarter and a designer later.",
        ("Dana", "Two engineers first, then a designer."),
        ("Me", "When do the reqs open?"),
        ("Dana", "January, if the budget holds."),
    ),
    conversation(
        "e-roadmap",
        DANA,
        utc(2026, 7, 8, 11),
        "Dana presented the roadmap for the second half of the year, with analytics first.",
        ("Dana", "Analytics ships first, then the mobile work."),
        ("Me", "Is the team on board?"),
        ("Dana", "Mostly."),
    ),
    conversation(
        "e-launch",
        DANA,
        utc(2026, 4, 21, 15, 30),
        "Dana debriefed the spring launch: the sign-up flow worked but the pricing page "
        "confused people.",
        ("Dana", "Sign-ups were fine; pricing confused everyone."),
        ("Me", "Do we rewrite it?"),
        ("Dana", "Yes, next sprint."),
    ),
    conversation(
        "e-kickoff",
        DANA,
        utc(2026, 1, 13, 9),
        "Dana kicked off the year with team goals and a plan for the quarterly reviews.",
        ("Dana", "Three goals this year, reviewed every quarter."),
        ("Me", "Sounds good."),
        ("Dana", "I will send the doc."),
    ),
    conversation(
        "e-gym",
        SAM_CARTER,
        utc(2026, 10, 20, 18, 30),
        "Sam recommended the Boulder Barn, a new climbing gym near the station, and offered "
        "to go together.",
        ("Sam", "The Boulder Barn opened near the station, a proper climbing gym."),
        ("Me", "Shall we go on Saturday?"),
        ("Sam", "Yes, I will book two passes."),
    ),
    conversation(
        "e-lease",
        SAM_CARTER,
        utc(2026, 6, 30, 10),
        "Sam said he is renewing his lease for another year despite the rent increase.",
        ("Sam", "I am renewing the lease, even with the rent going up."),
        ("Me", "By how much?"),
        ("Sam", "Eighty a month."),
    ),
    conversation(
        "e-marathon",
        SAM_OKAFOR,
        utc(2026, 10, 30, 8, 15),
        "Sam talked about training for the Valencia marathon and his long Sunday runs.",
        ("Sam", "I am up to thirty kilometres on Sundays for Valencia."),
        ("Me", "When is the race?"),
        ("Sam", "The first weekend of December."),
    ),
    conversation(
        "e-wedding",
        SAM_OKAFOR,
        utc(2026, 3, 5, 19),
        "Sam asked Riley to give a toast at his wedding in June and worried about the "
        "seating plan.",
        ("Sam", "Would you give a toast at the wedding?"),
        ("Me", "Of course."),
        ("Sam", "The seating plan is the hard part."),
    ),
    conversation(
        "e-pottery",
        LENA,
        utc(2026, 8, 14, 19),
        "Lena raved about her pottery class and invited Riley to the studio's open evening.",
        ("Lena", "The pottery class is on Thursdays at the Kiln Room."),
        ("Me", "Could I come along?"),
        ("Lena", "Come to the open evening on the 3rd."),
    ),
]
BY_ID = {record.episode_id: record for record in CONVERSATIONS}
EXPRESSIONS = {expression.key: expression.time_range for expression in time_expressions()}
LAST_WEEK = EXPRESSIONS["last_week"]
YESTERDAY = EXPRESSIONS["yesterday"]


# --------------------------------------------------------------------------- #
# Queries, and the script the fake system model follows for each              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Script:
    """How the fake system model behaves on one query.

    ``written_name`` is what its extraction returns (``"Lena's"`` to exercise
    the possessive tidy-up). It answers only when ``evidence`` is in the
    material it is shown; ``evidence`` is in the gold summary for summary-path
    queries and only in the gold dialogue for the detail-only query.
    """

    written_name: str | None
    time_range: TimeRange | None = None
    evidence: str | None = None
    answer: str = ""


def query(
    query_id: str,
    text: str,
    query_type: str,
    gold_name: str | None,
    gold_person_id: str | None,
    gold_ids: Sequence[str],
    gold_answer: str | None,
    script: Script,
    gold_time_range: TimeRange | None = None,
) -> tuple[Query, Script]:
    record = Query(
        query_id=query_id,
        query=text,
        type=query_type,
        gold_name=gold_name,
        gold_time_range=gold_time_range,
        gold_episode_ids=list(gold_ids),
        gold_answer=gold_answer,
        split="test",
        gold_person_id=gold_person_id,
    )
    return record, script


BUDGET_ANSWER = "Dana flagged the travel line as overspent in the Q4 budget review."
POTTERY_ANSWER = "At the Kiln Room, on Thursdays."
MARATHON_ANSWER = "Sam is training for the Valencia marathon with long Sunday runs."
GYM_ANSWER = "Sam Carter told you about the Boulder Barn climbing gym near the station."
VENDOR_ANSWER = "Dana said the vendor contract renews in January, eight percent higher."
OFFSITE_ANSWER = (
    "The travel overspend that puts the offsite in question, and the unapproved venue deposit."
)

SCRIPTED: list[tuple[Query, Script]] = [
    query(  # summary path
        "q-budget",
        "What did Dana say about the Q4 budget?",
        "lookup",
        "Dana",
        DANA.person_id,
        ["e-budget"],
        BUDGET_ANSWER,
        Script("Dana", evidence="travel line", answer=BUDGET_ANSWER),
    ),
    query(  # fallback path: the evidence is in the dialogue only
        "q-pottery",
        "Where is Lena's pottery class held?",
        "detail_only",
        "Lena",
        LENA.person_id,
        ["e-pottery"],
        POTTERY_ANSWER,
        Script("Lena's", evidence="Kiln Room", answer=POTTERY_ANSWER),
    ),
    query(  # declined: neither attempt finds anything
        "q-car",
        "What did Dana say about her new car?",
        "unanswerable",
        "Dana",
        DANA.person_id,
        [],
        None,
        Script("Dana"),
    ),
    query(  # empty pool: Lena has nothing from yesterday
        "q-yesterday",
        "What did Lena tell me yesterday?",
        "unanswerable",
        "Lena",
        LENA.person_id,
        [],
        None,
        Script("Lena", time_range=YESTERDAY),
        gold_time_range=YESTERDAY,
    ),
    query(  # unknown name: the exact message, no ranking
        "q-priya",
        "What did Priya say about the hackathon?",
        "unknown_name",
        "Priya",
        None,
        [],
        None,
        Script("Priya"),
    ),
    query(  # ambiguous: two people are called Sam
        "q-sam",
        "What did Sam say about the marathon?",
        "lookup",
        "Sam",
        None,
        ["e-marathon"],
        MARATHON_ANSWER,
        Script("Sam", evidence="Valencia", answer=MARATHON_ANSWER),
    ),
    query(  # no name: the whole corpus is the pool
        "q-gym",
        "Who told me about a climbing gym?",
        "cross_person",
        None,
        None,
        ["e-gym"],
        GYM_ANSWER,
        Script(None, evidence="Boulder Barn", answer=GYM_ANSWER),
    ),
    query(  # time filter with padding
        "q-dana-week",
        "What did Dana say last week?",
        "temporal",
        "Dana",
        DANA.person_id,
        ["e-vendor"],
        VENDOR_ANSWER,
        Script("Dana", time_range=LAST_WEEK, evidence="vendor contract", answer=VENDOR_ANSWER),
        gold_time_range=LAST_WEEK,
    ),
    query(  # two gold conversations
        "q-multi",
        "What has Dana flagged about the offsite?",
        "multi_conversation",
        "Dana",
        DANA.person_id,
        ["e-budget", "e-offsite"],
        OFFSITE_ANSWER,
        Script("Dana", evidence="offsite", answer=OFFSITE_ANSWER),
    ),
]
QUERIES = [record for record, _ in SCRIPTED]
SCRIPT_BY_TEXT = {record.query: script for record, script in SCRIPTED}
ANSWERABLE = [record for record in QUERIES if record.answerable]
EXPECTED_PATH = {
    "q-budget": "summary",
    "q-pottery": "fallback",
    "q-car": "declined",
    "q-yesterday": "empty_pool",
    "q-priya": "unknown_name",
    "q-sam": "ambiguous",
    "q-gym": "summary",
    "q-dana-week": "summary",
    "q-multi": "summary",
}


# --------------------------------------------------------------------------- #
# Fake local models and the fake Jev: deterministic word overlap              #
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "the a an of to and i me my did what who has about say said tell told in on at for is was "
    "her his he she it that this with you do we our where which from".split()
)


def words(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


VOCABULARY = sorted(
    {word for record in CONVERSATIONS for word in words(record.summary)}
    | {word for record in QUERIES for word in words(record.query)}
)
INDEX = {word: position for position, word in enumerate(VOCABULARY)}


def bag_of_words(text: str) -> list[float]:
    vector = [0.0] * len(VOCABULARY)
    for word in words(text):
        if word in INDEX:
            vector[INDEX[word]] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def overlap(query_text: str, text: str) -> float:
    """Share of the query's words that the text contains, in [0, 1]."""

    asked = set(words(query_text))
    return len(asked & set(words(text))) / len(asked) if asked else 0.0


class FakeEmbedder:
    model_name = "fake/bag-of-words"

    def embed_query(self, text: str) -> list[float]:
        return bag_of_words(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [bag_of_words(text) for text in texts]


class FakeScorer:
    model_name = "fake/overlap-cross-encoder"

    def score(self, query_text: str, texts: Sequence[str]) -> list[float]:
        return [overlap(query_text, text) for text in texts]


def jev_answers(state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]) -> dict:
    """Answer a Jev request the way ``jev_scoring`` builds them, in any of its three modes."""

    answers: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        if question["type"] == "choice":  # listwise: the options are the summaries
            raw = {option: overlap(state["query"], text) for option, text in question["criteria"].items()}
            total = sum(raw.values()) or 1.0
            probabilities = {option: value / total for option, value in raw.items()}
            best = max(probabilities, key=probabilities.get)
            answers[name] = {
                "type": "choice",
                "choice": best,
                "probabilities": probabilities,
                "confidence": probabilities[best],
            }
            continue
        if "conversation_summary" in state:  # pair: the summary is in the state
            summary = state["conversation_summary"]
        else:  # batched: the summary follows the label at the end of the instruction text
            summary = question["instructions"].rsplit(BATCHED_SUMMARY_SEPARATOR, 1)[1]
        relevance = overlap(state["query"], summary)
        if question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": relevance}
        else:
            level = round(relevance * (len(question["criteria"]) - 1))
            answers[name] = {
                "type": "score",
                "score": level,
                "probabilities": {str(level): 1.0},
                "confidence": 1.0,
            }
    return answers


@dataclass
class FakeJev:
    model: str = JEV_SNAPSHOT
    requests: list[tuple[Any, dict[str, Any], str]] = field(default_factory=list)
    replay: bool = False  # answer as the response cache would: the same answers, flagged cached

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        salt: str = "",
    ) -> DecisionResult:
        self.requests.append((state, dict(questions), salt))
        tokens = len(json.dumps(state)) // 4 + sum(len(json.dumps(q)) // 4 for q in questions.values())
        return DecisionResult(
            answers=jev_answers(state, questions),
            usage=Usage(input_tokens=tokens, cost_usd=tokens * 1e-7, latency_s=0.05, calls=1),
            model=self.model,
            cached=self.replay,
        )


# --------------------------------------------------------------------------- #
# Fake chat model: scripted per query, told apart by the system prompt        #
# --------------------------------------------------------------------------- #
def _tagged(user: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", user, re.DOTALL)
    return match.group(1) if match else ""


def _script_for(user: str) -> Script:
    """The script of the query a system-model prompt is about (``Question: ...`` closes it)."""

    return SCRIPT_BY_TEXT[user.rsplit("Question: ", 1)[1].strip()]


def scripted_reply(system: str, user: str) -> dict[str, Any]:
    """What the fake model replies, given only what a real gateway would see."""

    if system.startswith("You extract search filters"):
        script = _script_for(user)
        time_range = None if script.time_range is None else script.time_range.to_json()
        return {"name": script.written_name, "time_range": time_range}
    if system.startswith("You answer a question"):
        script = _script_for(user)
        if script.evidence is not None and script.evidence in user:
            return {"can_answer": True, "answer": script.answer}
        return {"can_answer": False, "answer": ""}
    if system.startswith("You grade answers"):  # the answer judge
        gold, given = _tagged(user, "gold_answer"), _tagged(user, "system_answer")
        if given == gold:
            verdict = "correct"
        elif given == config.DECLINE_ANSWER or given.startswith(("There is no", "More than one")):
            verdict = "incorrect"
        else:
            verdict = "partial"
        return {"verdict": verdict, "reason": "scripted"}
    if system.startswith("You label conversations"):  # the relevance judge
        script = SCRIPT_BY_TEXT[_tagged(user, "question")]
        relevant = script.evidence is not None and script.evidence in _tagged(user, "dialogue")
        return {"relevant": relevant, "reason": "scripted"}
    if system.startswith("You write the reference answer"):  # the gold-answer writer
        return {"answer": SCRIPT_BY_TEXT[_tagged(user, "question")].answer or "nothing"}
    raise AssertionError(f"unexpected prompt: {system[:60]!r}")


@dataclass
class FakeLLM:
    calls: list[tuple[str, str, str]] = field(default_factory=list)  # (namespace, system, user)

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
        self.calls.append((namespace, system, user))
        data = scripted_reply(system, user)
        usage = Usage(
            input_tokens=len(user) // 4,
            output_tokens=20,
            cost_usd=len(user) * 1e-7,
            latency_s=0.02,
            calls=1,
        )
        return LLMResult(data=data, usage=usage, model=f"{model}-served", raw_text=json.dumps(data))

    def prompts(self, namespace: str) -> list[str]:
        return [user for called, _, user in self.calls if called == namespace]


# --------------------------------------------------------------------------- #
# Wiring                                                                      #
# --------------------------------------------------------------------------- #
def make_store() -> InMemoryCorpusStore:
    embeddings = {record.episode_id: bag_of_words(record.summary) for record in CONVERSATIONS}
    return InMemoryCorpusStore(PEOPLE, CONVERSATIONS, embeddings)


def make_rankers(
    store: CorpusStore, jev: DecisionClient, k: int = max(K_GRID), salt: str = ""
) -> dict[str, Ranker]:
    scoring = JevScoringConfig(salt=salt)
    return {
        name: build_ranker(
            name,
            store=store,
            embedder=FakeEmbedder(),
            scorer=FakeScorer(),
            jev=jev,
            jev_scoring=scoring,
            k=k,
        )
        for name in config.RANKER_NAMES
    }


def gold_filters(record: Query) -> tuple[str | None, TimeRange | None]:
    """The filters a correct extraction leads to: the gold person and the padded gold range."""

    padded = None if record.gold_time_range is None else pad_time_range(record.gold_time_range)
    return record.gold_person_id, padded


@pytest.fixture(scope="module")
def store() -> InMemoryCorpusStore:
    return make_store()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(store: InMemoryCorpusStore, llm: FakeLLM) -> SystemPipeline:
    return SystemPipeline(store, llm, SYSTEM_MODEL)


# --------------------------------------------------------------------------- #
# Records: what the runner will build from the results                        #
# --------------------------------------------------------------------------- #
def ranker_record(record: Query, trace_result: Any, run: int) -> metrics.RankerRecord:
    trace = trace_result.trace
    return metrics.RankerRecord(
        query_id=record.query_id,
        type=record.type,
        pool_size=trace.pool_size,
        gold_ids=list(record.gold_episode_ids),
        ranker=trace.ranker,
        run=run,
        ranked_ids=[episode_id for episode_id, _ in trace_result.ranked],
        scores=[score for _, score in trace_result.ranked],
        stage1_ids=trace.stage1_ids,
        latency_s=trace.latency_s,
        stage_latency_s=dict(trace.stage_latency_s),
        usage=trace.usage,
        extra=dict(trace.extra),
    )


START_TIMES = {episode_id: record.start_time for episode_id, record in BY_ID.items()}


def system_record(
    record: Query,
    ranker_name: str,
    k: int,
    k_prime: int,
    result: SystemResult,
    pool_size: int | None,
    verdict: str | None,
) -> metrics.SystemRecord:
    """The runner's own record builder, given this corpus's start times for the time-filter check."""

    return records.system_record(
        record, ranker_name, k, k_prime, result, pool_size, verdict, START_TIMES
    )


@dataclass
class Run:
    """Everything one benchmark run produces, for the aggregation tests."""

    ranker_records: list[metrics.RankerRecord]
    system_records: list[metrics.SystemRecord]
    results: dict[tuple[str, str, int, int], SystemResult]  # (query_id, ranker, k, k')
    llm: FakeLLM
    jev: FakeJev


@pytest.fixture(scope="module")
def run(store: InMemoryCorpusStore) -> Run:
    llm = FakeLLM()
    jev = FakeJev()
    pipeline = SystemPipeline(store, llm, SYSTEM_MODEL)
    rankers = make_rankers(store, jev)
    ranker_records: list[metrics.RankerRecord] = []
    pool_sizes: dict[str, int | None] = {}
    for record in QUERIES:
        person_id, padded = gold_filters(record)
        pool_sizes[record.query_id] = (
            None if record.type == "unknown_name" else len(store.pool_ids(OWNER, person_id, padded))
        )
        if not record.answerable:
            continue
        for name, ranker in rankers.items():
            traced = ranker.rank_with_trace(OWNER, record.query, person_id, padded)
            ranker_records.append(ranker_record(record, traced, run=0))
    # A second run of the Jev rankers under another salt, for the variance table.
    for name, ranker in make_rankers(store, jev, salt="variance-run-1").items():
        if name in config.JEV_RANKERS:
            for record in ANSWERABLE:
                traced = ranker.rank_with_trace(OWNER, record.query, *gold_filters(record))
                ranker_records.append(ranker_record(record, traced, run=1))
    system_records: list[metrics.SystemRecord] = []
    results: dict[tuple[str, str, int, int], SystemResult] = {}
    for name, ranker in rankers.items():
        for record in QUERIES:
            for k in K_GRID:
                for k_prime in K_PRIME_GRID:
                    result = pipeline.run(record.query, ranker, k, k_prime)
                    results[(record.query_id, name, k, k_prime)] = result
                    verdict = None
                    if record.gold_answer is not None:  # declined and misrouted answers included
                        verdict = judge.judge_answer(
                            record.query, record.gold_answer, result.final_answer, llm, JUDGE_MODEL
                        ).verdict
                    system_records.append(
                        system_record(
                            record, name, k, k_prime, result, pool_sizes[record.query_id], verdict
                        )
                    )
    return Run(ranker_records, system_records, results, llm, jev)


# --------------------------------------------------------------------------- #
# Protocols and the store                                                     #
# --------------------------------------------------------------------------- #
def test_fakes_and_store_satisfy_the_shared_protocols(store: InMemoryCorpusStore) -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeScorer(), PairScorer)
    assert isinstance(FakeJev(), DecisionClient)
    assert isinstance(FakeLLM(), ChatLLM)
    assert isinstance(store, CorpusStore)
    for ranker in make_rankers(store, FakeJev()).values():
        assert isinstance(ranker, Ranker)


def test_store_lists_people_by_name_and_fills_in_person_names(store: InMemoryCorpusStore) -> None:
    assert store.list_people(OWNER) == [DANA, LENA, SAM_CARTER, SAM_OKAFOR]
    assert store.list_people("someone-else") == []
    assert len(store.pool_ids(OWNER)) == len(CONVERSATIONS) == 12
    assert store.pool_ids(OWNER, DANA.person_id)[:3] == ["e-budget", "e-offsite", "e-vendor"]
    headers = store.get_headers(["e-gym", "e-pottery", "nope"])
    assert {h.person_name for h in headers.values()} == {"Sam Carter", "Lena Fischer"}
    full = store.get_conversations(["e-marathon"])["e-marathon"]
    assert full.meta["person_name"] == "Sam Okafor"
    assert full.dialogue_text().startswith('Sam: "I am up to thirty kilometres')


def test_padding_widens_the_gold_time_filter_by_a_whole_day(store: InMemoryCorpusStore) -> None:
    padded = pad_time_range(LAST_WEEK)

    assert padded == TimeRange(utc(2026, 10, 25), LAST_WEEK.end + timedelta(days=1))
    assert padded.end == utc(2026, 11, 2, 23, 59) + timedelta(seconds=59)
    # The Monday conversation is outside "last week" but inside the padded filter;
    # Tuesday's is outside both.
    assert store.pool_ids(OWNER, DANA.person_id, LAST_WEEK) == ["e-vendor"]
    assert store.pool_ids(OWNER, DANA.person_id, padded) == ["e-offsite", "e-vendor"]
    assert store.pool_ids(OWNER, LENA.person_id, pad_time_range(YESTERDAY)) == []


# --------------------------------------------------------------------------- #
# Rankers                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_every_ranker_finds_the_gold_under_the_gold_filters(
    store: InMemoryCorpusStore, name: str
) -> None:
    ranker = make_rankers(store, FakeJev())[name]
    for record in ANSWERABLE:
        person_id, padded = gold_filters(record)
        result = ranker.rank_with_trace(OWNER, record.query, person_id, padded)
        pool = store.pool_ids(OWNER, person_id, padded)
        ranked_ids = [episode_id for episode_id, _ in result.ranked]

        assert len(ranked_ids) == min(len(pool), max(K_GRID))
        assert set(ranked_ids) <= set(pool)
        assert len(set(ranked_ids)) == len(ranked_ids)
        assert metrics.recall_at_k(ranked_ids, record.gold_episode_ids, 5) == 1.0, record.query_id
        assert result.trace.pool_size == len(pool)
        assert result.trace.ranker == name
        assert all(isinstance(score, float) for _, score in result.ranked)
    assert ranker.rank(OWNER, "What did Dana say about the Q4 budget?", DANA.person_id)[0][0] == "e-budget"
    assert ranker.rank(OWNER, "Who told me about a climbing gym?")[0][0] == "e-gym"


def test_ranker_traces_carry_what_the_records_need(store: InMemoryCorpusStore) -> None:
    jev = FakeJev()
    rankers = make_rankers(store, jev)
    traces = {
        name: ranker.rank_with_trace(OWNER, "What did Dana say about the Q4 budget?", DANA.person_id).trace
        for name, ranker in rankers.items()
    }

    assert set(traces["bge_m3"].stage_latency_s) == {"embed_query", "vector_search"}
    assert set(traces["bge_m3_reranker"].stage_latency_s) == {
        "embed_query",
        "vector_search",
        "fetch_summaries",
        "rerank",
    }
    assert set(traces["bge_m3_jev"].stage_latency_s) == set(traces["bge_m3_reranker"].stage_latency_s)
    assert set(traces["jev_full_pool"].stage_latency_s) == {"fetch_pool", "fetch_summaries", "rerank"}
    assert traces["bge_m3"].stage1_ids is None and traces["jev_full_pool"].stage1_ids is None
    assert set(traces["bge_m3_reranker"].stage1_ids) == set(store.pool_ids(OWNER, DANA.person_id))
    for name in ("bge_m3", "bge_m3_reranker"):
        assert traces[name].usage == Usage() and traces[name].extra == {}
    for name in config.JEV_RANKERS:
        assert traces[name].usage.calls == 1  # seven candidates fit in one batch of 16
        assert traces[name].usage.cost_usd > 0
        assert traces[name].extra == {
            "n_requests": 1,
            "n_missing": 0,
            "n_cached": 0,
            "jev_model": JEV_SNAPSHOT,
            "scoring": "batched-noul-v1-b16",
        }
    assert all(salt == "" for _, _, salt in jev.requests)


@pytest.mark.parametrize("mode", ("pair", "batched", "listwise"))
@pytest.mark.parametrize("primitive", ("noul", "score"))
def test_jev_rankers_rank_under_every_scoring_mode(
    store: InMemoryCorpusStore, mode: str, primitive: str
) -> None:
    scoring = JevScoringConfig(mode=mode, primitive=primitive, salt="study")
    jev = FakeJev()
    for name in config.JEV_RANKERS:
        ranker = build_ranker(name, store=store, embedder=FakeEmbedder(), jev=jev, jev_scoring=scoring)

        result = ranker.rank_with_trace(OWNER, "Who told me about a climbing gym?")

        assert result.ranked[0][0] == "e-gym"
        assert result.trace.extra["scoring"] == scoring.label
        assert result.trace.extra["n_missing"] == 0
    assert {salt for _, _, salt in jev.requests} == {"study"}


# --------------------------------------------------------------------------- #
# The system pipeline, path by path                                           #
# --------------------------------------------------------------------------- #
def test_the_fixture_drives_every_system_path(run: Run) -> None:
    observed = {result.path for result in run.results.values()}

    assert observed == set(SYSTEM_PATHS)
    for (query_id, _, _, _), result in run.results.items():
        assert result.path == EXPECTED_PATH[query_id], query_id
        assert list(result.usage_by_stage) == list(result.latency_by_stage) == list(SYSTEM_STAGES)
        assert result.usage == sum(result.usage_by_stage.values(), Usage())
        assert result.latency_s == pytest.approx(sum(result.latency_by_stage.values()))


def test_summary_path_answers_from_the_top_k_summaries(pipeline: SystemPipeline, llm: FakeLLM) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3"]

    result = pipeline.run("What did Dana say about the Q4 budget?", ranker, k=5, k_prime=10)

    assert (result.path, result.final_answer) == ("summary", BUDGET_ANSWER)
    assert result.prepared.person_id == DANA.person_id
    assert result.prepared.time_range is None
    assert result.rank is not None and result.rank.trace.pool_size == 7
    assert result.answered is not None and result.answered.fallback_attempt is None
    assert [namespace for namespace, _, _ in llm.calls] == ["extraction", "answer_summary"]
    assert result.usage_by_stage["answer_dialogue"] == Usage()
    assert result.usage_by_stage["rank"] == Usage()  # a local ranker
    assert result.usage_by_stage["extraction"].calls == 1
    summaries_prompt = llm.prompts("answer_summary")[0]
    assert "[1] " in summaries_prompt and "with Dana Whitfield" in summaries_prompt
    assert summaries_prompt.count("| with ") == 5  # k summaries shown
    assert "Full dialogues" not in summaries_prompt


def test_fallback_path_adds_the_dialogues_and_cuts_the_possessive(
    pipeline: SystemPipeline, llm: FakeLLM
) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3_reranker"]

    result = pipeline.run("Where is Lena's pottery class held?", ranker, k=5, k_prime=3)

    assert (result.path, result.final_answer) == ("fallback", POTTERY_ANSWER)
    assert result.prepared.extraction.name == "Lena"  # "Lena's" tidied before resolving
    assert result.prepared.person_id == LENA.person_id
    assert result.answered.summary_attempt.can_answer is False
    assert result.answered.fallback_attempt.can_answer is True
    assert [namespace for namespace, _, _ in llm.calls] == [
        "extraction",
        "answer_summary",
        "answer_dialogue",
    ]
    dialogue_prompt = llm.prompts("answer_dialogue")[0]
    assert "Kiln Room" in dialogue_prompt and "Kiln Room" not in llm.prompts("answer_summary")[0]
    assert result.usage_by_stage["answer_dialogue"].calls == 1
    assert result.latency_by_stage["answer_dialogue"] >= 0.02  # at least the recorded call latency


def test_declined_path_tries_both_attempts_then_gives_the_decline_answer(
    pipeline: SystemPipeline, llm: FakeLLM
) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3_jev"]

    result = pipeline.run("What did Dana say about her new car?", ranker, k=10, k_prime=10)

    assert (result.path, result.final_answer) == ("declined", config.DECLINE_ANSWER)
    assert result.answered.fallback_attempt.can_answer is False
    assert result.usage_by_stage["rank"].calls == 1  # the Jev call is charged to the rank stage
    assert len(llm.prompts("answer_dialogue")) == 1
    assert llm.prompts("answer_dialogue")[0].count("Full dialogues") == 1


def test_empty_pool_declines_without_an_llm_call(pipeline: SystemPipeline, llm: FakeLLM) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["jev_full_pool"]

    result = pipeline.run("What did Lena tell me yesterday?", ranker, k=5, k_prime=5)

    assert (result.path, result.final_answer) == ("empty_pool", config.DECLINE_ANSWER)
    assert result.prepared.time_range == pad_time_range(YESTERDAY)
    assert result.rank.trace.pool_size == 0 and result.rank.ranked == []
    assert result.answered.summary_attempt is None
    assert [namespace for namespace, _, _ in llm.calls] == ["extraction"]
    assert result.usage.calls == 1


def test_unknown_name_stops_with_the_exact_message(pipeline: SystemPipeline, llm: FakeLLM) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3"]

    result = pipeline.run("What did Priya say about the hackathon?", ranker, k=5, k_prime=5)

    assert result.path == "unknown_name"
    assert result.final_answer == "There is no one in your memory named Priya."
    assert result.final_answer == config.UNKNOWN_NAME_TEMPLATE.format(name="Priya")
    assert result.prepared.resolution.status == "unknown"
    assert result.rank is None and result.answered is None
    assert [namespace for namespace, _, _ in llm.calls] == ["extraction"]
    assert result.usage_by_stage["rank"] == Usage()


def test_ambiguous_name_lists_both_people(pipeline: SystemPipeline) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3"]

    result = pipeline.run("What did Sam say about the marathon?", ranker, k=5, k_prime=5)

    assert result.path == "ambiguous"
    assert result.final_answer == "More than one person in your memory matches Sam: Sam Carter, Sam Okafor."
    assert result.final_answer == config.AMBIGUOUS_NAME_TEMPLATE.format(
        name="Sam", candidates="Sam Carter, Sam Okafor"
    )
    assert result.prepared.resolution.candidates == [SAM_CARTER, SAM_OKAFOR]
    assert result.rank is None


def test_no_name_ranks_the_whole_corpus(pipeline: SystemPipeline, llm: FakeLLM) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3_reranker"]

    result = pipeline.run("Who told me about a climbing gym?", ranker, k=5, k_prime=5)

    assert (result.path, result.final_answer) == ("summary", GYM_ANSWER)
    assert result.prepared.extraction.name is None and result.prepared.resolution is None
    assert result.prepared.person_id is None
    assert result.rank.trace.pool_size == 12
    assert result.rank.ranked[0][0] == "e-gym"
    assert "with Sam Carter" in llm.prompts("answer_summary")[0]


def test_time_filtered_query_ranks_the_padded_pool(pipeline: SystemPipeline, llm: FakeLLM) -> None:
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3"]

    result = pipeline.run("What did Dana say last week?", ranker, k=5, k_prime=5)

    assert (result.path, result.final_answer) == ("summary", VENDOR_ANSWER)
    assert result.prepared.extraction.time_range == LAST_WEEK  # unpadded, as extracted
    assert result.prepared.time_range == pad_time_range(LAST_WEEK)
    assert result.rank.trace.pool_size == 2
    assert {episode_id for episode_id, _ in result.rank.ranked} == {"e-offsite", "e-vendor"}
    assert llm.prompts("answer_summary")[0].count("| with ") == 2


def test_a_missed_time_expression_is_a_time_filter_miss_charged_to_extraction(
    pipeline: SystemPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extraction returns no time range for a temporal query: no filter is applied, the
    unfiltered pool still ranks the gold first, and the record says the time filter failed."""

    text = "What did Dana tell me the week before this one?"
    script = Script("Dana", evidence="vendor contract", answer=VENDOR_ANSWER)  # no time_range
    monkeypatch.setitem(SCRIPT_BY_TEXT, text, script)
    record, _ = query(
        "q-dana-missed", text, "temporal", "Dana", DANA.person_id, ["e-vendor"], VENDOR_ANSWER, script,
        gold_time_range=LAST_WEEK,
    )
    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3"]

    result = pipeline.run(text, ranker, k=5, k_prime=5)
    assert result.prepared.extraction.time_range is None and result.prepared.time_range is None
    unfiltered = len(pipeline.store.pool_ids(OWNER, DANA.person_id, None))
    assert result.rank.trace.pool_size == unfiltered > 2  # every Dana conversation: nothing was filtered

    cell = system_record(record, "bge_m3", 5, 5, result, pool_size=2, verdict="correct")
    assert cell.extracted_time_range is None and cell.time_filter_ok is False
    assert set(record.gold_episode_ids) <= set(cell.retrieved_ids)

    table = metrics.system_eval([cell])["bge_m3"]["5"]["5"]
    assert table["overall"]["time_range_extracted_rate"].mean == 0.0
    assert table["overall"]["time_filter_recall"].mean == 0.0
    failures = table["failures"]
    assert failures["missed_time_ranges"] == 1 and failures["time_filter_misses"] == 0
    assert failures["gold_not_in_top_k"] == 0 and failures["empty_pools"] == 0


def test_a_ranking_replayed_from_the_cache_is_flagged_and_left_out_of_the_latency(
    pipeline: SystemPipeline,
) -> None:
    """A system record says when its Jev ranking was a cache replay, and the cost aggregate
    then leaves that query's latency out, as the ranker comparison leaves out cached records."""

    record = next(query for query in QUERIES if query.query_id == "q-pottery")
    fresh, replayed = FakeJev(), FakeJev(replay=True)
    cells = []
    for jev in (fresh, replayed):
        ranker = make_rankers(pipeline.store, jev)["jev_full_pool"]
        result = pipeline.run(record.query, ranker, k=5, k_prime=5)
        cells.append(system_record(record, "jev_full_pool", 5, 5, result, pool_size=3, verdict="correct"))
    timed, cached = cells
    assert timed.rank_cached is False and cached.rank_cached is True
    assert timed.latency_by_stage["rank"] > 0 and cached.latency_by_stage["rank"] > 0

    cost = metrics.system_eval([timed])["jev_full_pool"]["5"]["5"]["overall"]["cost"]["overall"]
    assert cost["n"] == cost["n_timed"] == cost["latency"]["n"] == 1
    cost = metrics.system_eval([cached])["jev_full_pool"]["5"]["5"]["overall"]["cost"]["overall"]
    assert cost["n"] == 1 and cost["n_timed"] == 0 and cost["latency"]["n"] == 0
    assert math.isnan(cost["latency"]["p50"])
    assert cost["cost_usd"] > 0  # the replay still counts the tokens and dollars it reports


def test_prepare_once_then_answer_per_cell_matches_run(pipeline: SystemPipeline, run: Run) -> None:
    """The runner's shape: prepare a query once, rank once, answer at every (k, k')."""

    ranker = make_rankers(pipeline.store, FakeJev())["bge_m3"]
    prepared = pipeline.prepare("What has Dana flagged about the offsite?")
    ranked = ranker.rank(OWNER, "What has Dana flagged about the offsite?", prepared.person_id, prepared.time_range)
    ranked_ids = [episode_id for episode_id, _ in ranked]

    for k in K_GRID:
        for k_prime in K_PRIME_GRID:
            answered = pipeline.answer("What has Dana flagged about the offsite?", ranked_ids, k, k_prime)
            whole = run.results[("q-multi", "bge_m3", k, k_prime)]
            assert (answered.path, answered.final_answer) == (whole.path, whole.final_answer)
            assert set(answered.usage_by_stage) == {"answer_summary", "answer_dialogue"}


# --------------------------------------------------------------------------- #
# Judges and label pooling                                                    #
# --------------------------------------------------------------------------- #
def test_relevance_judge_runs_over_the_pooled_candidates(
    store: InMemoryCorpusStore, llm: FakeLLM
) -> None:
    record = QUERIES[6]  # q-gym
    assert record.query_id == "q-gym"
    pooled: dict[str, None] = {}
    for ranker in make_rankers(store, FakeJev(), k=config.LABEL_POOL_DEPTH).values():
        for episode_id, _ in ranker.rank(OWNER, record.query, *gold_filters(record))[:3]:
            pooled.setdefault(episode_id, None)
    conversations = store.get_conversations(list(pooled))

    verdicts = {
        episode_id: judge.judge_relevance(record, conversation, llm, JUDGE_MODEL)
        for episode_id, conversation in conversations.items()
    }

    assert "e-gym" in verdicts and verdicts["e-gym"].relevant is True
    assert {episode_id for episode_id, verdict in verdicts.items() if verdict.relevant} == {"e-gym"}
    assert all(verdict.usage.calls == 1 for verdict in verdicts.values())
    assert all("<person>" in prompt for prompt in llm.prompts("judge_relevance"))


def test_relevance_judge_settles_conversations_outside_the_window_in_code(
    store: InMemoryCorpusStore, llm: FakeLLM
) -> None:
    record = QUERIES[7]  # q-dana-week: gold range is last week
    outside = store.get_conversations(["e-budget"])["e-budget"]  # Tuesday of this week

    verdict = judge.judge_relevance(record, outside, llm, JUDGE_MODEL)

    assert verdict.relevant is False and verdict.usage == Usage()
    assert verdict.reason == judge.OUTSIDE_WINDOW_REASON
    assert llm.calls == []


def test_gold_answer_writer_takes_the_stores_conversations(store: InMemoryCorpusStore, llm: FakeLLM) -> None:
    record = QUERIES[8]  # q-multi
    conversations = list(store.get_conversations(record.gold_episode_ids).values())

    answer, usage = judge.write_gold_answer(record, conversations, llm, JUDGE_MODEL)

    assert answer == OFFSITE_ANSWER and usage.calls == 1


def test_answer_judge_grades_every_kind_of_system_answer(run: Run) -> None:
    by_path = {}
    for record in run.system_records:
        by_path.setdefault(record.path, set()).add(record.verdict)

    assert by_path["summary"] == {"correct"}
    assert by_path["fallback"] == {"correct"}
    assert by_path["ambiguous"] == {"incorrect"}  # an answerable query that was misrouted
    assert by_path["unknown_name"] == {None} and by_path["empty_pool"] == {None}
    assert by_path["declined"] == {None}  # q-car has no gold answer to grade against
    assert set(VERDICTS) >= {verdict for verdicts in by_path.values() for verdict in verdicts if verdict}


# --------------------------------------------------------------------------- #
# Records, aggregates and the report                                          #
# --------------------------------------------------------------------------- #
def test_records_round_trip_through_json(run: Run) -> None:
    for record in run.ranker_records[:8] + run.ranker_records[-2:]:
        assert metrics.RankerRecord.from_json(json.loads(json.dumps(record.to_json()))) == record
    for record in run.system_records[::7]:
        assert metrics.SystemRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_ranker_comparison_over_the_run(run: Run) -> None:
    table = metrics.ranker_comparison(run.ranker_records)

    assert list(table) == list(config.RANKER_NAMES)
    for name, entry in table.items():
        overall = entry["overall"]
        assert overall["n"] == len(ANSWERABLE) == 6
        assert overall["recall"]["5"].mean == 1.0
        assert overall["recall"]["10"].mean == 1.0
        assert set(entry["by_type"]) == {
            "lookup",
            "detail_only",
            "cross_person",
            "temporal",
            "multi_conversation",
        }
        assert set(entry["by_pool"]) == {"<=20"}
        assert entry["failures"]["gold_not_in_top_k"]["5"] == 0
        assert overall["latency"]["n"] == 6  # no record came from a cache
        assert entry["recall_gain"]["5"]["10"].mean == 0.0
    assert table["bge_m3"]["n_missing"] is None
    assert table["bge_m3_jev"]["n_missing"] == 0
    assert table["bge_m3"]["overall"]["stage1_recall"] is None
    assert table["bge_m3_reranker"]["overall"]["stage1_recall_all_pools"].mean == 1.0
    assert table["bge_m3"]["overall"]["cost_per_1k_usd"] == 0.0
    assert table["jev_full_pool"]["overall"]["cost_per_1k_usd"] > 0.0
    assert table["bge_m3_jev"]["recall_diff_vs"]["bge_m3"]["overall"]["10"].mean == 0.0


def test_jev_variance_over_the_repeated_runs(run: Run) -> None:
    table = metrics.jev_variance(run.ranker_records)

    assert list(table) == list(config.JEV_RANKERS)
    for entry in table.values():
        assert entry["runs"] == [0, 1] and entry["n_queries"] == 6
        assert entry["recall"]["10"]["std"] == 0.0
        assert entry["mean_abs_score_diff"] == 0.0 and entry["top10_jaccard"] == 1.0


def test_system_eval_over_the_run(run: Run) -> None:
    table = metrics.system_eval(run.system_records)

    assert list(table) == list(config.RANKER_NAMES)
    for name in config.RANKER_NAMES:
        assert list(table[name]) == ["5", "10"] and list(table[name]["5"]) == ["5", "10"]
        cell = table[name]["10"]["10"]
        overall = cell["overall"]
        assert overall["n"] == len(QUERIES) == 9
        assert overall["answer_accuracy"].mean == pytest.approx(5 / 6)  # q-sam is misrouted
        assert overall["misrouted_rate"].mean == pytest.approx(1 / 6)
        assert overall["fallback_rate"].mean == pytest.approx(1 / 6)  # q-pottery
        assert overall["false_decline_rate"].mean == 0.0
        assert overall["correct_decline_rate"].mean == 1.0  # q-car, q-yesterday
        assert overall["unknown_name_exact_rate"].mean == 1.0
        assert overall["name_accuracy"].mean == 1.0
        assert overall["person_accuracy"].mean == 1.0
        assert overall["time_range_extracted_rate"].mean == 1.0  # q-dana-week
        assert overall["time_filter_recall"].mean == 1.0
        assert overall["spurious_time_range_rate"].mean == 0.0
        assert overall["cost"]["fallback"]["n"] == 2  # q-pottery (fallback) and q-car (declined)
        assert overall["cost"]["summary"]["n"] == 4
        # q-sam stopped at the ambiguous name: misrouted, so not a ranking failure.
        assert cell["failures"]["misrouted"] == 1 and cell["failures"]["gold_not_in_top_k"] == 0
        assert cell["failures"]["empty_pools"] == 0  # q-yesterday's empty pool is an unanswerable query's
        assert cell["failures"]["missed_time_ranges"] == 0
        assert cell["failures"]["wrong_answers_gold_retrieved"] == 0
        assert cell["next_k"] is None and cell["next_k_prime"] is None
        assert table[name]["5"]["5"]["next_k"]["k"] == "10"
        assert table[name]["5"]["5"]["next_k_prime"]["k_prime"] == "10"
        assert set(cell["by_pool"]) == {"<=20"}
        assert set(cell["by_type"]) == {record.type for record in QUERIES}
    assert table["bge_m3_jev"]["10"]["10"]["accuracy_diff_vs"]["bge_m3"].mean == 0.0


def test_judge_validation_sample_reads_the_system_records(run: Run) -> None:
    records = [record.to_json() for record in run.system_records]

    items = judge_validation.sample_items(records, QUERIES, n=4, seed=config.SEED)

    assert len(items) == 4
    assert {item["key"].split("/")[0] for item in items} <= {record.query_id for record in ANSWERABLE}
    by_key = {judge_validation.item_key(record): record for record in records}
    for item in items:
        record = by_key[item["key"]]
        assert item["judge_verdict"] == record["verdict"] in VERDICTS
        assert item["system_answer"] == record["final_answer"]
        assert (item["ranker"], item["k"], item["k_prime"]) == (record["ranker"], record["k"], record["k_prime"])
        assert item["query"] == next(q.query for q in QUERIES if q.query_id == item["query_id"])
    labels = {item["key"]: item["judge_verdict"] for item in items}
    summary = judge_validation.agreement(items, labels)
    assert summary == {"n": 4, "agreement_rate": 1.0, "disagreements": []}


def test_recommendation_and_report_render_from_the_run(run: Run, tmp_path) -> None:
    ranker_table = metrics.ranker_comparison(run.ranker_records)
    system_table = metrics.system_eval(run.system_records)
    recommendation = metrics.recommend(ranker_table, system_table)
    records = [record.to_json() for record in run.system_records]
    items = judge_validation.sample_items(records, QUERIES, n=4)
    labels = {item["key"]: item["judge_verdict"] for item in items}
    labels[items[0]["key"]] = "partial" if items[0]["judge_verdict"] != "partial" else "incorrect"
    results = {
        "meta": {
            "date": "2026-11-06",
            "seed": config.SEED,
            "split": "test",
            "store": "memory",
            "counts": {"people": len(PEOPLE), "conversations": len(CONVERSATIONS), "queries": 9},
            "models": {"system": SYSTEM_MODEL, "judge": JUDGE_MODEL},
            "jev": {"model": JEV_SNAPSHOT, "scoring": JevScoringConfig().label},
            "hardware": {"machine": "integration test"},
        },
        "ranker_comparison": ranker_table,
        "jev_variance": metrics.jev_variance(run.ranker_records),
        "system_eval": system_table,
        "judge_validation": judge_validation.agreement(items, labels),
        "recommendation": recommendation,
        "notes": "Produced by eval/tests/test_eval_integration.py with fake models.",
    }

    assert recommendation["ranker"]["choice"] in config.RANKER_NAMES
    assert recommendation["k"]["choice"] in set(K_GRID)  # values are ints; table keys are strings
    assert recommendation["k_prime"]["choice"] in set(K_PRIME_GRID)
    assert set(recommendation["failure_modes"]) == set(metrics.FAILURE_MODES)

    markdown = report.render_markdown(results)
    for heading in ("Judge validation", "Disagreements", "Notes"):
        assert heading in markdown
    for name in config.RANKER_NAMES:
        assert name in markdown
    assert "batched-noul-v1-b16" in markdown
    md_path, json_path = report.write_report(results, "2026-11-06", out_dir=tmp_path)
    assert md_path.read_text(encoding="utf-8") == markdown
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert report.render_markdown(reloaded) == markdown  # the JSON file re-renders identically


# --------------------------------------------------------------------------- #
# Runtime: the lead's wiring against the real constructors                   #
# --------------------------------------------------------------------------- #
def settings(**overrides: Any) -> config.Settings:
    values: dict[str, Any] = dict(
        database_url=None,
        gateway="openrouter",
        llm_base_url="https://gateway.test/v1",
        llm_api_key="test-key",
        gen_model="fake/gen",
        summary_model="fake/summary",
        system_model=SYSTEM_MODEL,
        judge_model=JUDGE_MODEL,
        jev_url="https://gateway.test/decisions",
        jev_api_key="test-key",
        jev_model="typesafe/jev-1.14",
        llm_concurrency=2,
        jev_concurrency=3,
        device=None,
        max_spend_usd=1.5,
    )
    values.update(overrides)
    return config.Settings(**values)


def test_runtime_builds_the_live_components_without_loading_anything(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EVAL_JEV_MODEL", raising=False)
    monkeypatch.setattr(runtime, "JEV_SCORING_PATH", tmp_path / "jev_scoring.json")
    heavy = {"torch", "sentence_transformers", "transformers"}
    live = runtime.Runtime(settings(), store_kind="memory", use_cache=False)

    assert live.cache is None
    assert live.spend.limit_usd == 1.5
    assert isinstance(live.llm, OpenAICompatLLM)
    assert isinstance(live.jev, JevClient) and live.jev.model == "typesafe/jev-1.14"
    assert live.jev.url == "https://gateway.test/decisions"
    assert isinstance(live.embedder, BgeM3Embedder) and isinstance(live.scorer, BgeReranker)
    assert live.embedder.model_name == config.EMBEDDING_MODEL
    assert not heavy & set(sys.modules)  # nothing loads a model until the first call
    assert live.jev_scoring("run-1") == JevScoringConfig(concurrency=3, salt="run-1")
    assert live.jev_scoring().label == JevScoringConfig().label

    pg = runtime.Runtime(
        settings(database_url="postgresql://eval:pw@db.example.test:5432/eval?sslmode=require"),
        store_kind="pg",
    )
    assert isinstance(pg.store, PgCorpusStore)  # built, not connected (the conftest blocks sockets)
    with pytest.raises(RuntimeError, match="EVAL_DATABASE_URL"):
        monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
        runtime.Runtime(settings(), store_kind="pg").store
    with pytest.raises(ValueError, match="store_kind"):
        runtime.Runtime(settings(), store_kind="sqlite")


def test_runtime_applies_the_pinned_jev_study_and_the_env_override(monkeypatch, tmp_path) -> None:
    pin = tmp_path / "jev_scoring.json"
    pinned = JevScoringConfig(mode="pair", primitive="score", wording="v2", concurrency=1, salt="study")
    pin.write_text(json.dumps({"model": JEV_SNAPSHOT, **pinned.to_json()}), encoding="utf-8")
    monkeypatch.setattr(runtime, "JEV_SCORING_PATH", pin)
    monkeypatch.delenv("EVAL_JEV_MODEL", raising=False)
    live = runtime.Runtime(settings(), store_kind="memory", use_cache=False)

    assert live.jev_model == JEV_SNAPSHOT  # the study's snapshot beats the settings default
    assert live.jev.model == JEV_SNAPSHOT
    # The pinned application, with this machine's concurrency and this run's salt.
    assert live.jev_scoring("run-2") == JevScoringConfig(
        mode="pair", primitive="score", wording="v2", concurrency=3, salt="run-2"
    )
    assert live.jev_scoring("run-2").label == "pair-score-v2"

    # Settings that name a model other than the gateway's default are the explicit choice,
    # which is how EVAL_JEV_MODEL arrives: through get_settings, never read by the runtime itself.
    assert runtime.Runtime(settings(jev_model="typesafe/jev-1.15"), use_cache=False).jev_model == "typesafe/jev-1.15"
    monkeypatch.setenv("EVAL_GATEWAY", "openrouter")
    monkeypatch.setenv("EVAL_JEV_MODEL", "typesafe/jev-1.16")
    assert runtime.Runtime(settings(), use_cache=False).jev_model == JEV_SNAPSHOT  # the pin, not the environment
    config.get_settings.cache_clear()
    try:
        assert runtime.Runtime(config.get_settings(), use_cache=False).jev_model == "typesafe/jev-1.16"
    finally:
        config.get_settings.cache_clear()


def test_runtime_rankers_take_injected_components(store: InMemoryCorpusStore) -> None:
    live = runtime.Runtime(settings(), store_kind="memory", use_cache=False)
    live.store, live.embedder, live.scorer, live.jev = store, FakeEmbedder(), FakeScorer(), FakeJev()

    for name in config.RANKER_NAMES:
        ranker = live.ranker(name, k=5, salt="run-7")

        assert ranker.name == name and ranker.k == 5
        result = ranker.rank_with_trace(OWNER, "What did Dana say about the Q4 budget?", DANA.person_id)
        assert result.ranked[0][0] == "e-budget" and len(result.ranked) == 5
        if name in config.JEV_RANKERS:
            assert ranker.scoring == JevScoringConfig(concurrency=3, salt="run-7")
        else:
            assert not hasattr(ranker, "scoring")
    assert {salt for _, _, salt in live.jev.requests} == {"run-7"}
    with pytest.raises(ValueError, match="unknown ranker"):
        live.ranker("bm25")


# --------------------------------------------------------------------------- #
# The real hosted clients, over a mock transport                              #
# --------------------------------------------------------------------------- #
def chat_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    system, user = (message["content"] for message in body["messages"])
    reply = scripted_reply(system, user)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "model": f"{body['model']}-2026-10-01",
            "choices": [
                {"message": {"role": "assistant", "content": json.dumps(reply)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 20},
        },
    )


def jev_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "model": JEV_SNAPSHOT,
            "answers": jev_answers(body["state"], body["questions"]),
            "usage": {"input_tokens": 300, "output_tokens": 0},
        },
    )


def test_real_llm_client_drives_the_pipeline_and_the_judge(store: InMemoryCorpusStore) -> None:
    llm = OpenAICompatLLM("test-key", "https://gateway.test/v1", transport=httpx.MockTransport(chat_handler))
    pipeline = SystemPipeline(store, llm, SYSTEM_MODEL)
    ranker = make_rankers(store, FakeJev())["bge_m3"]
    try:
        result = pipeline.run("Where is Lena's pottery class held?", ranker, k=5, k_prime=5)
        verdict = judge.judge_answer(
            "Where is Lena's pottery class held?", POTTERY_ANSWER, result.final_answer, llm, JUDGE_MODEL
        )
    finally:
        llm.close()

    assert (result.path, result.final_answer) == ("fallback", POTTERY_ANSWER)
    assert result.usage.calls == 3  # extraction, summaries, dialogues
    assert result.usage_by_stage["extraction"].input_tokens == 120
    # No price is listed for the fake model and the gateway reported no cost, so every call is
    # charged at the highest listed prices (an upper bound) rather than at nothing.
    price_in, price_out = fallback_prices()
    expected = (result.usage.input_tokens * price_in + result.usage.output_tokens * price_out) / 1e6
    assert result.usage.cost_usd == pytest.approx(expected) and expected > 0.0
    assert verdict.verdict == "correct" and verdict.usage.calls == 1


def test_real_jev_client_drives_the_jev_rankers(store: InMemoryCorpusStore) -> None:
    client = JevClient(
        "test-key",
        provider="openrouter",
        url="https://gateway.test/decisions",
        model="typesafe/jev-1.14",
        transport=httpx.MockTransport(jev_handler),
    )
    try:
        for name in config.JEV_RANKERS:
            ranker = build_ranker(name, store=store, embedder=FakeEmbedder(), jev=client)

            result = ranker.rank_with_trace(OWNER, "Who told me about a climbing gym?")

            assert result.ranked[0][0] == "e-gym"
            assert result.trace.usage.calls == 1  # the twelve candidates fit in one batch
            assert result.trace.usage.input_tokens == 300
            assert result.trace.usage.cost_usd == pytest.approx(300 * config.JEV_PRICE_USD_PER_MTOK_INPUT / 1e6)
            assert result.trace.extra["jev_model"] == JEV_SNAPSHOT
    finally:
        client.close()
