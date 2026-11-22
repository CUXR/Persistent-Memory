"""The query flow under test: extraction, name resolution, answering, and the pipeline that chains them.

Everything runs offline against a scripted chat model, a small in-memory store
and a newest-first ranker. Name resolution is not faked: it goes through the
backend's ``PersonResolver``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from eval.retrieval import config
from eval.retrieval.interfaces import ChatLLM, LLMResult, Ranker, RankResult, RankTrace, Usage
from eval.retrieval.schema import Conversation, ConversationHeader, Person, TimeRange, Turn
from eval.retrieval.system import (
    AMBIGUOUS_NAME_TEMPLATE,
    STAGES,
    Answered,
    Resolution,
    SystemPipeline,
    answer_from_summaries,
    answer_with_dialogues,
    extract,
    resolve_name,
)
from eval.retrieval.system import pipeline as pipeline_module
from eval.retrieval.system.resolver import load_person_resolver
from eval.retrieval.timeutil import EXTRACTION_TIME_RULES, pad_time_range, time_expressions

MODEL = "test/system-model"
NO = {"can_answer": False, "answer": ""}


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def yes(answer: str) -> dict[str, Any]:
    return {"can_answer": True, "answer": answer}


def extraction_reply(name: str | None = None, time_range: TimeRange | None = None) -> dict[str, Any]:
    return {"name": name, "time_range": time_range.to_json() if time_range else None}


# --------------------------------------------------------------------------- #
# A small memory: four people, two of them called Sam, and six conversations   #
# --------------------------------------------------------------------------- #
SAM_CARTER = Person("p-sam-carter", "Sam", "Carter", "Sam Carter")
SAM_OKAFOR = Person("p-sam-okafor", "Sam", "Okafor", "Sam Okafor")
DANA = Person("p-dana", "Dana", "Whitfield", "Dana Whitfield")
LENA = Person("p-lena", "Lena", "Fischer", "Lena Fischer")
PEOPLE = [DANA, LENA, SAM_OKAFOR, SAM_CARTER]  # not in name order, so ordering has to come from the resolver
NAMES = {person.person_id: person.display_name for person in PEOPLE}


def conversation(
    episode_id: str, person: Person, start: datetime, summary: str, *turns: tuple[str, str]
) -> Conversation:
    return Conversation(
        episode_id=episode_id,
        person_id=person.person_id,
        start_time=start,
        end_time=start + timedelta(minutes=20),
        dialogue=[Turn(speaker, text) for speaker, text in turns],
        summary=summary,
    )


GYM = conversation(
    "e-gym",
    SAM_CARTER,
    utc(2026, 10, 28, 17, 30),  # Wednesday of last week
    "Sam recommended the Vertical Edge climbing gym on Mill Road and offered a guest pass.",
    ("Sam", "You should try Vertical Edge on Mill Road. I have a guest pass."),
    ("Me", "Is it any good for beginners?"),
    ("Sam", "Yes. The auto-belay wall is free on Saturdays."),
)
WEDDING = conversation(
    "e-wedding",
    SAM_CARTER,
    utc(2026, 10, 12, 9),
    "Sam described his sister's wedding in Lisbon and the speech he has to give.",
    ("Sam", "My sister is getting married in Lisbon in May."),
    ("Me", "Are you giving a speech?"),
    ("Sam", "Yes, and I am dreading it."),
)
LEASE = conversation(
    "e-lease",
    SAM_CARTER,
    utc(2026, 11, 2, 10),  # Monday of this week: outside "last week", inside its padded filter
    "Sam said he is renewing his lease for another year despite the rent increase.",
    ("Sam", "I am renewing the lease, even with the rent going up."),
    ("Me", "By how much?"),
    ("Sam", "Eighty a month."),
)
MARATHON = conversation(
    "e-marathon",
    SAM_OKAFOR,
    utc(2026, 10, 30, 8, 15),
    "Sam talked about training for the Valencia marathon and his long Sunday runs.",
    ("Sam", "I am up to thirty kilometres on Sundays for Valencia."),
    ("Me", "When is the race?"),
    ("Sam", "The first weekend of December."),
)
BUDGET = conversation(
    "e-budget",
    DANA,
    utc(2026, 11, 3, 14),
    "Dana walked through the Q4 budget review and flagged the travel line as overspent.",
    ("Dana", "Travel is over by twelve percent this quarter."),
    ("Me", "Do we cut the offsite?"),
    ("Dana", "No, we trim conference tickets instead."),
)
POTTERY = conversation(
    "e-pottery",
    LENA,
    utc(2026, 8, 14, 19),
    "Lena raved about her pottery class and invited Riley to the studio's open evening.",
    ("Lena", "The pottery class is on Thursdays at the Kiln Room."),
    ("Me", "Could I come along?"),
    ("Lena", "Come to the open evening on the 3rd."),
)
CONVERSATIONS = [GYM, WEDDING, LEASE, MARATHON, BUDGET, POTTERY]
NEWEST_FIRST = ["e-budget", "e-lease", "e-marathon", "e-gym", "e-wedding", "e-pottery"]

LAST_WEEK = {expression.key: expression.time_range for expression in time_expressions()}["last_week"]
LAST_WEEK_PADDED = TimeRange(utc(2026, 10, 25), utc(2026, 11, 2, 23, 59, 59))  # one day each side


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeClock:
    """Stands in for ``perf_counter``: time passes only when a fake says so."""

    def __init__(self) -> None:
        self.seconds = 1000.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class ScriptedLLM:
    """A ``ChatLLM`` that replays scripted JSON per namespace and records every call.

    A call outside the script fails the test. With a clock, a call in a
    namespace takes ``wall_s[namespace]`` seconds of fake time.
    """

    def __init__(
        self,
        script: dict[str, list[dict[str, Any]]],
        usage: dict[str, Usage] | None = None,
        clock: FakeClock | None = None,
        wall_s: dict[str, float] | None = None,
    ) -> None:
        self.script = {namespace: list(replies) for namespace, replies in script.items()}
        self.usage = usage or {}
        self.clock = clock
        self.wall_s = wall_s or {}
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
        replies = self.script.get(namespace)
        assert replies, f"unscripted LLM call in namespace {namespace!r}"
        if self.clock is not None:
            self.clock.advance(self.wall_s.get(namespace, 0.0))
        return LLMResult(data=replies.pop(0), usage=self.usage.get(namespace, Usage()), model=model)

    def namespaces(self) -> list[str]:
        return [call["namespace"] for call in self.calls]

    def prompts(self, namespace: str) -> list[str]:
        """The user messages sent in one namespace, in call order."""

        return [call["user"] for call in self.calls if call["namespace"] == namespace]


class FakeStore:
    """The parts of ``CorpusStore`` the flow reads, over the six conversations, with a call log."""

    def __init__(self, owner_id: str = config.OWNER_ID) -> None:
        self.owner_id = owner_id
        self.conversations = {conversation.episode_id: conversation for conversation in CONVERSATIONS}
        self.calls: list[tuple[str, Any]] = []

    def list_people(self, owner_id: str) -> list[Person]:
        assert owner_id == self.owner_id
        self.calls.append(("list_people", owner_id))
        return list(PEOPLE)

    def pool_ids(self, owner_id: str, person_id: str | None = None, time_range: TimeRange | None = None) -> list[str]:
        assert owner_id == self.owner_id
        pool = [
            conversation
            for conversation in self.conversations.values()
            if person_id in (None, conversation.person_id)
            and (time_range is None or time_range.contains(conversation.start_time))
        ]
        pool.sort(key=lambda conversation: (-conversation.start_time.timestamp(), conversation.episode_id))
        return [conversation.episode_id for conversation in pool]

    def get_headers(self, episode_ids: Sequence[str]) -> dict[str, ConversationHeader]:
        self.calls.append(("get_headers", list(episode_ids)))
        # Reversed: a store owes its callers a mapping, not the order they asked in.
        return {episode_id: self.header(episode_id) for episode_id in reversed(episode_ids)}

    def get_conversations(self, episode_ids: Sequence[str]) -> dict[str, Conversation]:
        self.calls.append(("get_conversations", list(episode_ids)))
        return {episode_id: self.conversation(episode_id) for episode_id in reversed(episode_ids)}

    def conversation(self, episode_id: str) -> Conversation:
        """A copy that carries the person's name in ``meta``, as both real stores return it."""

        conversation = self.conversations[episode_id]
        return replace(conversation, meta={"person_name": NAMES[conversation.person_id]})

    def header(self, episode_id: str) -> ConversationHeader:
        conversation = self.conversations[episode_id]
        return ConversationHeader(
            episode_id=episode_id,
            person_id=conversation.person_id,
            person_name=NAMES[conversation.person_id],
            start_time=conversation.start_time,
            summary=conversation.summary,
        )


class FakeRanker:
    """Ranks the filtered pool newest first and records the filters it was given.

    A call advances the clock by ``wall_s``; its trace reports ``trace_s`` when
    given (a real ranker reads its clock before the pool-size diagnostic, so
    the two differ), else ``wall_s``.
    """

    name = "newest_first"

    def __init__(
        self,
        store: FakeStore,
        usage: Usage | None = None,
        clock: FakeClock | None = None,
        wall_s: float = 0.0,
        trace_s: float | None = None,
    ) -> None:
        self.store = store
        self.usage = usage or Usage()
        self.clock = clock
        self.wall_s = wall_s
        self.trace_s = wall_s if trace_s is None else trace_s
        self.calls: list[dict[str, Any]] = []

    def rank(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[tuple[str, float]]:
        return self.rank_with_trace(owner_id, query, person_id, time_range).ranked

    def rank_with_trace(
        self,
        owner_id: str,
        query: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> RankResult:
        self.calls.append({"owner_id": owner_id, "query": query, "person_id": person_id, "time_range": time_range})
        if self.clock is not None:
            self.clock.advance(self.wall_s)
        pool = self.store.pool_ids(owner_id, person_id, time_range)
        ranked = [(episode_id, 1.0 - position / 10) for position, episode_id in enumerate(pool)]
        trace = RankTrace(ranker=self.name, pool_size=len(pool), latency_s=self.trace_s, usage=self.usage)
        return RankResult(ranked=ranked, trace=trace)


def test_fakes_match_the_shared_protocols():
    assert isinstance(ScriptedLLM({}), ChatLLM)
    assert isinstance(FakeRanker(FakeStore()), Ranker)


# --------------------------------------------------------------------------- #
# Extraction                                                                   #
# --------------------------------------------------------------------------- #
def test_extract_makes_one_call_and_returns_the_name_and_the_unpadded_range():
    usage = Usage(input_tokens=420, output_tokens=35, cost_usd=0.00004, latency_s=0.8, calls=1)
    reply = {"name": "Sam Carter", "time_range": {"start": "2026-10-26T00:00:00Z", "end": "2026-11-01T23:59:59Z"}}
    llm = ScriptedLLM({"extraction": [reply]}, usage={"extraction": usage})

    extraction = extract("What did Sam Carter say about the climbing gym last week?", llm, MODEL)

    assert extraction.name == "Sam Carter"
    assert extraction.time_range == LAST_WEEK
    assert extraction.usage == usage
    assert extraction.raw == {"response": reply, "rejected": {}}
    [call] = llm.calls
    assert (call["namespace"], call["model"], call["reasoning_effort"]) == ("extraction", MODEL, "low")
    assert call["user"] == "Question: What did Sam Carter say about the climbing gym last week?"


def test_extraction_prompt_states_the_clock_the_time_rules_and_the_name_rules():
    llm = ScriptedLLM({"extraction": [extraction_reply()]})

    extract("Who told me about the pottery class?", llm, MODEL)

    system = llm.calls[0]["system"]
    assert "The current time is Wednesday 2026-11-04 18:00 UTC." in system
    assert EXTRACTION_TIME_RULES in system
    assert "exactly as it is written in the question" in system
    assert "the full name when a surname is given" in system
    assert "Use null when the question does not name that person" in system
    assert "writes in the first person" in system
    assert "is never the name" in system
    assert '{"name": string or null, "time_range": {"start": "YYYY-MM-DDTHH:MM:SSZ"' in system


def test_extract_uses_the_now_and_the_reasoning_effort_it_is_given():
    llm = ScriptedLLM({"extraction": [extraction_reply()]})

    query = "Who told me about the pottery class?"

    extraction = extract(query, llm, MODEL, now=utc(2026, 1, 1, 9, 30), reasoning_effort="medium")

    assert (extraction.name, extraction.time_range) == (None, None)
    assert extraction.raw["rejected"] == {}
    assert "The current time is Thursday 2026-01-01 09:30 UTC." in llm.calls[0]["system"]
    assert llm.calls[0]["reasoning_effort"] == "medium"


@pytest.mark.parametrize(
    ("returned", "name", "rejected"),
    [
        ("Sam", "Sam", False),
        ("  Sam \n Carter ", "Sam Carter", False),  # tidied, not rejected
        ("Sam's", "Sam", False),  # the resolver would not find "Sam" in "Sam's"
        ("Sam Carter\u2019S", "Sam Carter", False),  # typographic apostrophe
        ("'s", None, True),
        (None, None, False),  # a real null: the query names nobody
        ("", None, True),
        ("   ", None, True),
        ("null", None, True),
        ("None", None, True),
        (" N/A ", None, True),
        ("unknown", None, True),
        (42, None, True),
        (["Sam"], None, True),
    ],
)
def test_extract_keeps_usable_names_and_drops_blank_or_null_like_ones(returned, name, rejected):
    llm = ScriptedLLM({"extraction": [{"name": returned, "time_range": None}]})

    extraction = extract("What did Sam say?", llm, MODEL)

    assert extraction.name == name
    assert ("name" in extraction.raw["rejected"]) is rejected
    assert extraction.raw["response"]["name"] == returned  # what the model wrote is kept as it was


@pytest.mark.parametrize(
    "returned",
    [
        {"start": "2026-10-26T00:00:00Z", "end": "2026-11-01T23:59:59Z"},
        {"start": "2026-10-26T00:00:00", "end": "2026-11-01T23:59:59"},  # no offset means UTC
        {"start": "2026-10-26T02:00:00+02:00", "end": "2026-11-01T18:59:59-05:00"},  # offsets are converted
        {"start": " 2026-10-26T00:00:00.000Z ", "end": "2026-11-01 23:59:59+00:00"},
    ],
)
def test_extract_reads_iso_timestamps_as_utc(returned):
    llm = ScriptedLLM({"extraction": [{"name": None, "time_range": returned}]})

    extraction = extract("What happened last week?", llm, MODEL)

    assert extraction.time_range == LAST_WEEK
    assert extraction.raw["rejected"] == {}


def test_extract_accepts_a_range_that_is_one_instant():
    instant = {"start": "2026-11-04T06:00:00Z", "end": "2026-11-04T06:00:00Z"}
    llm = ScriptedLLM({"extraction": [{"name": None, "time_range": instant}]})

    assert extract("q", llm, MODEL).time_range == TimeRange(utc(2026, 11, 4, 6), utc(2026, 11, 4, 6))


@pytest.mark.parametrize(
    ("returned", "reason"),
    [
        ("last week", "not an object"),
        (["2026-10-26T00:00:00Z", "2026-11-01T23:59:59Z"], "not an object"),
        ({}, "string start and a string end"),
        ({"start": "2026-10-26T00:00:00Z"}, "string start and a string end"),
        ({"start": "2026-10-26T00:00:00Z", "end": None}, "string start and a string end"),
        ({"start": 1792972800, "end": 1793577599}, "string start and a string end"),
        ({"start": "last Monday", "end": "2026-11-01T23:59:59Z"}, "unparseable"),
        ({"start": "2026-13-45T00:00:00Z", "end": "2026-11-01T23:59:59Z"}, "unparseable"),
        ({"start": "0001-01-01T00:00:00+05:00", "end": "2026-11-01T23:59:59Z"}, "unparseable"),  # before year 1 in UTC
        ({"start": "2026-11-01T23:59:59Z", "end": "2026-10-26T00:00:00Z"}, "ends before it starts"),
    ],
)
def test_extract_drops_an_unusable_range_and_records_why(returned, reason):
    llm = ScriptedLLM({"extraction": [{"name": "Sam", "time_range": returned}]})

    extraction = extract("What did Sam say last week?", llm, MODEL)

    assert extraction.time_range is None
    assert reason in extraction.raw["rejected"]["time_range"]
    assert extraction.raw["response"]["time_range"] == returned
    assert extraction.name == "Sam"  # a bad range does not cost the name


def test_extract_treats_missing_fields_as_null():
    llm = ScriptedLLM({"extraction": [{}]})

    extraction = extract("Who told me about the pottery class?", llm, MODEL)

    assert (extraction.name, extraction.time_range) == (None, None)
    assert extraction.raw == {"response": {}, "rejected": {}}


# --------------------------------------------------------------------------- #
# Name resolution (the backend's PersonResolver)                               #
# --------------------------------------------------------------------------- #
def test_full_display_name_resolves_among_people_who_share_a_first_name():
    assert resolve_name("Sam Carter", PEOPLE) == Resolution("resolved", "p-sam-carter", [])
    assert resolve_name("Sam Okafor", PEOPLE) == Resolution("resolved", "p-sam-okafor", [])


def test_shared_first_name_alone_is_ambiguous_with_both_candidates():
    resolution = resolve_name("Sam", PEOPLE)

    assert resolution == Resolution("ambiguous", None, [SAM_CARTER, SAM_OKAFOR])  # ordered by display name
    assert resolve_name("Sam", list(reversed(PEOPLE))) == resolution


def test_first_name_that_only_one_person_has_resolves():
    assert resolve_name("Dana", PEOPLE) == Resolution("resolved", "p-dana", [])


def test_name_nobody_has_is_unknown():
    assert resolve_name("Priya", PEOPLE) == Resolution("unknown", None, [])
    assert resolve_name("Priya Raman", PEOPLE) == Resolution("unknown", None, [])
    assert resolve_name("Sam Carter", []) == Resolution("unknown", None, [])  # an empty memory


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sam carter", Resolution("resolved", "p-sam-carter", [])),
        ("SAM CARTER", Resolution("resolved", "p-sam-carter", [])),
        ("  Sam   Carter ", Resolution("resolved", "p-sam-carter", [])),
        ("dana", Resolution("resolved", "p-dana", [])),
        ("LENA FISCHER", Resolution("resolved", "p-lena", [])),
        ("sAM", Resolution("ambiguous", None, [SAM_CARTER, SAM_OKAFOR])),
    ],
)
def test_matching_ignores_case_and_spacing(name, expected):
    assert resolve_name(name, PEOPLE) == expected


def test_dataset_style_uuid_person_ids_come_back_unchanged():
    people = [
        Person(config.stable_id("person", index), person.first_name, person.last_name, person.display_name)
        for index, person in enumerate(PEOPLE)
    ]

    assert resolve_name("Sam Okafor", people) == Resolution("resolved", people[2].person_id, [])
    assert resolve_name("Sam", people).candidates == [people[3], people[2]]


def test_matching_is_done_by_the_backend_person_resolver(monkeypatch):
    backend_resolver = load_person_resolver()
    backend_file = Path(sys.modules[backend_resolver.__module__].__file__)
    real_method = backend_resolver.resolve_person_from_query
    seen: list[str] = []

    def spy(self, query_text):
        rows = self.store.list_people()
        assert sorted(row.display_name for row in rows) == sorted(NAMES.values())
        assert all(isinstance(row.id, uuid.UUID) for row in rows)
        assert self.store.get_disambiguation_hints(rows[0].id) == {}
        seen.append(query_text)
        return real_method(self, query_text)

    monkeypatch.setattr(backend_resolver, "resolve_person_from_query", spy)

    assert (backend_resolver.__module__, backend_resolver.__name__) == ("app.crud.person_resolver", "PersonResolver")
    assert backend_file.is_relative_to(config.BACKEND_ROOT)
    assert resolve_name("Sam", PEOPLE).status == "ambiguous"
    assert seen == ["Sam"]


def test_loading_the_backend_keeps_settings_that_are_already_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///somewhere-else.db")
    monkeypatch.setenv("OPENAI_API_KEY", "a-real-key")
    backend_entries = sys.path.count(str(config.BACKEND_ROOT))  # conftest.py has already added it
    load_person_resolver.cache_clear()

    load_person_resolver()

    assert os.environ["DATABASE_URL"] == "sqlite+pysqlite:///somewhere-else.db"
    assert os.environ["OPENAI_API_KEY"] == "a-real-key"
    assert sys.path.count(str(config.BACKEND_ROOT)) == backend_entries >= 1  # no second entry


def test_resolver_brings_in_the_backend_by_itself_and_only_on_first_use(tmp_path):
    """Outside pytest, nothing has put ``backend/`` on the path or set the backend's required settings."""

    script = """
import json, os, sys
import eval.retrieval.system
from eval.retrieval.schema import Person
from eval.retrieval.system import resolve_name

def backend_loaded():
    return any(name == "app" or name.startswith("app.") for name in sys.modules)

loaded_on_import = backend_loaded()
people = [Person("a", "Sam", "Carter", "Sam Carter"), Person("b", "Sam", "Okafor", "Sam Okafor")]
print(json.dumps({
    "loaded_on_import": loaded_on_import,
    "resolved": resolve_name("Sam Okafor", people).person_id,
    "ambiguous": [person.display_name for person in resolve_name("sam", people).candidates],
    "unknown": resolve_name("Priya", people).status,
    "loaded_on_use": backend_loaded(),
    "database_url": os.environ.get("DATABASE_URL"),
    "openai_api_key": os.environ.get("OPENAI_API_KEY"),
}))
"""
    env = {key: value for key, value in os.environ.items() if key not in ("DATABASE_URL", "OPENAI_API_KEY")}
    env["PYTHONPATH"] = str(config.REPO_ROOT)

    # cwd is an empty directory: no .env for the backend settings to pick values up from.
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "loaded_on_import": False,
        "resolved": "b",
        "ambiguous": ["Sam Carter", "Sam Okafor"],
        "unknown": "unknown",
        "loaded_on_use": True,
        "database_url": "sqlite+pysqlite:///:memory:",
        "openai_api_key": "unused",
    }


# --------------------------------------------------------------------------- #
# The two answer attempts                                                      #
# --------------------------------------------------------------------------- #
SUMMARIES_OF_GYM_AND_BUDGET = (
    "Conversation summaries, most relevant first:\n\n"
    "[1] Wednesday 2026-10-28 17:30 UTC | with Sam Carter\n"
    "Sam recommended the Vertical Edge climbing gym on Mill Road and offered a guest pass.\n\n"
    "[2] Tuesday 2026-11-03 14:00 UTC | with Dana Whitfield\n"
    "Dana walked through the Q4 budget review and flagged the travel line as overspent."
)


def test_summary_attempt_shows_date_person_and_summary_in_rank_order():
    store = FakeStore()
    usage = Usage(input_tokens=1500, output_tokens=40, cost_usd=0.004, latency_s=1.5, calls=1)
    llm = ScriptedLLM({"answer_summary": [yes(" Vertical Edge, on Mill Road. ")]}, usage={"answer_summary": usage})
    headers = [store.header("e-gym"), store.header("e-budget")]

    attempt = answer_from_summaries("Which climbing gym did Sam recommend?", headers, llm, MODEL)

    assert (attempt.can_answer, attempt.answer) == (True, "Vertical Edge, on Mill Road.")
    assert attempt.usage == usage
    assert attempt.raw == yes(" Vertical Edge, on Mill Road. ")
    [call] = llm.calls
    assert (call["namespace"], call["model"]) == ("answer_summary", MODEL)
    assert call["user"] == f"{SUMMARIES_OF_GYM_AND_BUDGET}\n\nQuestion: Which climbing gym did Sam recommend?"
    assert "Me:" not in call["user"]  # summaries only


def test_dialogue_attempt_repeats_the_summaries_and_adds_the_dialogues():
    store = FakeStore()
    llm = ScriptedLLM({"answer_dialogue": [yes("On Saturdays.")]})
    headers = [store.header("e-gym"), store.header("e-budget")]

    attempt = answer_with_dialogues("When is the auto-belay wall free?", headers, [GYM], llm, MODEL)

    assert (attempt.can_answer, attempt.answer) == (True, "On Saturdays.")
    [call] = llm.calls
    assert (call["namespace"], call["model"]) == ("answer_dialogue", MODEL)
    assert call["user"] == (
        f"{SUMMARIES_OF_GYM_AND_BUDGET}\n\n"
        "Full dialogues of the top conversations, numbered like the summaries:\n\n"
        "[1] Wednesday 2026-10-28 17:30 UTC | with Sam Carter\n"
        'Sam: "You should try Vertical Edge on Mill Road. I have a guest pass."\n'
        'Me: "Is it any good for beginners?"\n'
        'Sam: "Yes. The auto-belay wall is free on Saturdays."\n\n'
        "Question: When is the auto-belay wall free?"
    )


def test_answer_prompt_sets_the_ground_rules():
    store = FakeStore()
    llm = ScriptedLLM({"answer_summary": [NO], "answer_dialogue": [NO]})
    headers = [store.header("e-gym")]

    answer_from_summaries("q", headers, llm, MODEL)
    answer_with_dialogues("q", headers, [GYM], llm, MODEL, now=utc(2026, 1, 1, 9, 30))

    summary_system, dialogue_system = (call["system"] for call in llm.calls)
    assert "The current time is Wednesday 2026-11-04 18:00 UTC." in summary_system
    assert "The current time is Thursday 2026-01-01 09:30 UTC." in dialogue_system
    assert dialogue_system.replace("Thursday 2026-01-01 09:30", "Wednesday 2026-11-04 18:00") == summary_system
    assert "a question that Riley Morgan asks about their own past conversations" in summary_system
    assert "Use only the material in the user message" in summary_system
    assert 'the speaker "Me" is Riley Morgan' in summary_system
    assert 'If it does not, set "can_answer" to false' in summary_system
    assert "Never guess" in summary_system
    assert "short, specific" in summary_system
    assert '{"can_answer": true or false, "answer": string}' in summary_system


def test_dialogue_with_the_same_person_as_a_summary_takes_the_name_from_it():
    llm = ScriptedLLM({"answer_dialogue": [NO]})

    answer_with_dialogues("q", [FakeStore().header("e-gym")], [GYM, WEDDING], llm, MODEL)  # k = 1, k' = 2

    assert "[2] Monday 2026-10-12 09:00 UTC | with Sam Carter\n" in llm.calls[0]["user"]


def test_dialogue_with_someone_outside_the_summaries_takes_the_name_the_store_put_in_meta():
    store = FakeStore()
    headers = [store.header("e-gym")]  # k = 1: the one summary is with Sam Carter
    dialogues = [store.conversation("e-gym"), store.conversation("e-budget")]  # k' = 2
    llm = ScriptedLLM({"answer_dialogue": [NO]})

    answer_with_dialogues("q", headers, dialogues, llm, MODEL)

    assert "[2] Tuesday 2026-11-03 14:00 UTC | with Dana Whitfield\n" in llm.calls[0]["user"]


def test_dialogue_that_nothing_names_is_an_error_rather_than_an_unlabelled_prompt():
    headers = [FakeStore().header("e-gym")]
    llm = ScriptedLLM({"answer_dialogue": [NO]})

    with pytest.raises(ValueError, match=r"e-budget.*person_name"):
        answer_with_dialogues("q", headers, [GYM, BUDGET], llm, MODEL)  # records without the meta

    assert llm.calls == []


@pytest.mark.parametrize(
    ("reply", "can_answer", "answer"),
    [
        ({"can_answer": True, "answer": "In Lisbon."}, True, "In Lisbon."),
        ({"can_answer": "True", "answer": "In Lisbon."}, True, "In Lisbon."),  # a string for the boolean
        ({"can_answer": False, "answer": ""}, False, ""),
        ({"can_answer": False, "answer": "The summaries do not say."}, False, "The summaries do not say."),
        ({"can_answer": "false", "answer": "In Lisbon."}, False, "In Lisbon."),
        ({"can_answer": True, "answer": "   "}, False, ""),  # a yes without an answer is a miss
        ({"can_answer": True, "answer": None}, False, ""),
        ({"can_answer": True}, False, ""),
        ({"answer": "In Lisbon."}, False, "In Lisbon."),
        ({}, False, ""),
    ],
)
def test_only_a_yes_with_an_answer_counts_as_answered(reply, can_answer, answer):
    llm = ScriptedLLM({"answer_summary": [reply]})

    attempt = answer_from_summaries("Where is the wedding?", [FakeStore().header("e-wedding")], llm, MODEL)

    assert (attempt.can_answer, attempt.answer) == (can_answer, answer)
    assert attempt.raw == reply


# --------------------------------------------------------------------------- #
# Pipeline: prepare (extract -> resolve -> pad)                                #
# --------------------------------------------------------------------------- #
def test_unknown_name_stops_with_the_exact_message():
    usage = Usage(input_tokens=400, output_tokens=30, cost_usd=0.001, latency_s=0.5, calls=1)
    store = FakeStore()
    llm = ScriptedLLM({"extraction": [extraction_reply("Priya Raman")]}, usage={"extraction": usage})
    ranker = FakeRanker(store)
    query = "What did Priya Raman say about the offsite?"

    result = SystemPipeline(store, llm, MODEL).run(query, ranker, k=5, k_prime=3)

    assert result.final_answer == "There is no one in your memory named Priya Raman."
    assert result.path == "unknown_name"
    assert result.prepared.early_answer == result.final_answer
    assert result.prepared.resolution == Resolution("unknown", None, [])
    assert result.prepared.person_id is None
    assert (result.rank, result.answered) == (None, None)
    assert ranker.calls == []
    assert llm.namespaces() == ["extraction"]
    assert result.usage == usage
    assert result.usage_by_stage == {
        "extraction": usage,
        "rank": Usage(),
        "answer_summary": Usage(),
        "answer_dialogue": Usage(),
    }


def test_possessive_first_name_still_resolves():
    store = FakeStore()
    llm = ScriptedLLM({"extraction": [extraction_reply("Dana's")]})

    prepared = SystemPipeline(store, llm, MODEL).prepare("What was Dana's advice on the travel budget?")

    assert prepared.extraction.name == "Dana"
    assert (prepared.person_id, prepared.early_answer) == ("p-dana", None)


def test_unknown_name_message_quotes_the_name_as_extracted():
    llm = ScriptedLLM({"extraction": [extraction_reply("  priya ")]})

    prepared = SystemPipeline(FakeStore(), llm, MODEL).prepare("what did priya say?")

    assert prepared.early_answer == config.UNKNOWN_NAME_TEMPLATE.format(name="priya")
    assert prepared.path == "unknown_name"


def test_ambiguous_name_stops_and_lists_the_candidates_on_one_line():
    store = FakeStore()
    llm = ScriptedLLM({"extraction": [extraction_reply("Sam", LAST_WEEK)]})
    ranker = FakeRanker(store)

    result = SystemPipeline(store, llm, MODEL).run("What did Sam say last week?", ranker, k=5, k_prime=3)

    assert result.final_answer == "More than one person in your memory matches Sam: Sam Carter, Sam Okafor."
    assert result.final_answer == AMBIGUOUS_NAME_TEMPLATE.format(name="Sam", candidates="Sam Carter, Sam Okafor")
    assert "\n" not in result.final_answer
    assert result.path == "ambiguous"
    assert result.prepared.resolution == Resolution("ambiguous", None, [SAM_CARTER, SAM_OKAFOR])
    assert result.prepared.person_id is None
    assert result.prepared.time_range == LAST_WEEK_PADDED  # still padded, for the time-filter metrics
    assert (result.rank, result.answered) == (None, None)
    assert ranker.calls == []
    assert llm.namespaces() == ["extraction"]


def test_query_without_a_name_searches_the_whole_owner_pool():
    store = FakeStore()
    llm = ScriptedLLM({"extraction": [extraction_reply()], "answer_summary": [yes("Lena Fischer.")]})
    ranker = FakeRanker(store)
    query = "Who told me about the pottery class?"

    result = SystemPipeline(store, llm, MODEL).run(query, ranker, k=10, k_prime=3)

    assert result.prepared.resolution is None
    assert (result.prepared.person_id, result.prepared.time_range) == (None, None)
    assert (result.prepared.early_answer, result.prepared.path) == (None, None)
    assert ranker.calls == [{"owner_id": config.OWNER_ID, "query": query, "person_id": None, "time_range": None}]
    assert [episode_id for episode_id, _ in result.rank.ranked] == NEWEST_FIRST  # every conversation the owner has
    assert "list_people" not in [name for name, _ in store.calls]  # nothing to resolve
    assert (result.path, result.final_answer) == ("summary", "Lena Fischer.")


def test_resolved_person_and_padded_range_reach_the_ranker():
    store = FakeStore()
    llm = ScriptedLLM(
        {"extraction": [extraction_reply("Sam Carter", LAST_WEEK)], "answer_summary": [yes("He is renewing it.")]}
    )
    ranker = FakeRanker(store)
    query = "What did Sam Carter decide about his lease last week?"

    result = SystemPipeline(store, llm, MODEL).run(query, ranker, k=5, k_prime=3)

    assert result.prepared.extraction.time_range == LAST_WEEK  # as extracted
    assert result.prepared.time_range == LAST_WEEK_PADDED == pad_time_range(LAST_WEEK)
    assert result.prepared.resolution == Resolution("resolved", "p-sam-carter", [])
    assert result.prepared.person_id == "p-sam-carter"
    assert ranker.calls == [
        {"owner_id": config.OWNER_ID, "query": query, "person_id": "p-sam-carter", "time_range": LAST_WEEK_PADDED}
    ]
    # Monday's conversation is a day past "last week": only the padding lets it into the pool.
    assert not LAST_WEEK.contains(LEASE.start_time)
    assert [episode_id for episode_id, _ in result.rank.ranked] == ["e-lease", "e-gym"]
    assert store.calls[0] == ("list_people", config.OWNER_ID)


def test_time_range_alone_filters_the_whole_owner_pool():
    store = FakeStore()
    llm = ScriptedLLM({"extraction": [extraction_reply(None, LAST_WEEK)], "answer_summary": [yes("Sam Okafor.")]})
    ranker = FakeRanker(store)

    result = SystemPipeline(store, llm, MODEL).run("Who was training for a marathon last week?", ranker, k=5, k_prime=3)

    assert ranker.calls[0]["person_id"] is None
    assert ranker.calls[0]["time_range"] == LAST_WEEK_PADDED
    assert [episode_id for episode_id, _ in result.rank.ranked] == ["e-lease", "e-marathon", "e-gym"]


def test_pipeline_clock_drives_the_prompts_and_caps_the_padded_range():
    now = utc(2026, 11, 3, 15)  # a Tuesday afternoon, a day before the benchmark clock
    today = TimeRange(utc(2026, 11, 3), now)
    store = FakeStore()
    llm = ScriptedLLM(
        {"extraction": [extraction_reply("Dana", today)], "answer_summary": [NO], "answer_dialogue": [NO]}
    )
    pipeline = SystemPipeline(store, llm, MODEL, now=now)

    result = pipeline.run("What did Dana say today?", FakeRanker(store), k=5, k_prime=3)

    # Four hours back, and no further forward than now.
    assert result.prepared.time_range == TimeRange(utc(2026, 11, 2, 20), now)
    assert result.prepared.person_id == "p-dana"
    assert [episode_id for episode_id, _ in result.rank.ranked] == ["e-budget"]
    assert llm.namespaces() == ["extraction", "answer_summary", "answer_dialogue"]
    assert all("The current time is Tuesday 2026-11-03 15:00 UTC." in call["system"] for call in llm.calls)
    assert {call["model"] for call in llm.calls} == {MODEL}


def test_pipeline_defaults_to_the_benchmark_owner_and_clock():
    pipeline = SystemPipeline(FakeStore(), ScriptedLLM({}), MODEL)

    assert (pipeline.owner_id, pipeline.now) == (config.OWNER_ID, config.NOW)


def test_another_owner_id_is_the_one_the_store_and_the_ranker_see():
    store = FakeStore(owner_id="another-owner")  # the fake refuses any other owner id
    llm = ScriptedLLM({"extraction": [extraction_reply("Lena")], "answer_summary": [yes("Thursdays.")]})
    ranker = FakeRanker(store)

    result = SystemPipeline(store, llm, MODEL, owner_id="another-owner").run("When is Lena's class?", ranker, 5, 3)

    assert store.calls[0] == ("list_people", "another-owner")
    assert ranker.calls[0]["owner_id"] == "another-owner"
    assert result.path == "summary"


def test_building_a_pipeline_loads_the_backend_resolver_up_front(monkeypatch):
    loads: list[str] = []
    monkeypatch.setattr(pipeline_module, "load_person_resolver", lambda: loads.append("loaded"))

    SystemPipeline(FakeStore(), ScriptedLLM({}), MODEL)

    assert loads == ["loaded"]  # so the import is not timed as part of the first query


# --------------------------------------------------------------------------- #
# Pipeline: answer (summaries first, dialogues as the fallback)                #
# --------------------------------------------------------------------------- #
RANKED = ["e-gym", "e-budget", "e-wedding", "e-pottery", "e-marathon", "e-lease"]


def test_summary_path_answers_from_the_first_k_summaries():
    usage = Usage(input_tokens=1500, output_tokens=40, cost_usd=0.004, latency_s=1.5, calls=1)
    store = FakeStore()
    llm = ScriptedLLM({"answer_summary": [yes("Vertical Edge, on Mill Road.")]}, usage={"answer_summary": usage})
    query = "Which climbing gym did Sam recommend?"

    answered = SystemPipeline(store, llm, MODEL).answer(query, RANKED, k=2, k_prime=3)

    assert (answered.path, answered.final_answer) == ("summary", "Vertical Edge, on Mill Road.")
    assert answered.summary_attempt.can_answer is True
    assert answered.fallback_attempt is None
    assert llm.prompts("answer_summary") == [f"{SUMMARIES_OF_GYM_AND_BUDGET}\n\nQuestion: {query}"]  # ranks 1-2 only
    assert llm.namespaces() == ["answer_summary"]
    assert store.calls == [("get_headers", ["e-gym", "e-budget"])]  # no dialogue was read
    assert answered.usage_by_stage == {"answer_summary": usage, "answer_dialogue": Usage()}
    assert answered.latency_by_stage["answer_dialogue"] == 0.0


def test_fallback_path_reads_the_first_k_prime_dialogues_next_to_the_same_k_summaries():
    store = FakeStore()
    llm = ScriptedLLM({"answer_summary": [NO], "answer_dialogue": [yes("They trim conference tickets instead.")]})
    query = "What gets cut instead of the offsite?"

    answered = SystemPipeline(store, llm, MODEL).answer(query, RANKED, k=1, k_prime=3)  # k' > k

    assert (answered.path, answered.final_answer) == ("fallback", "They trim conference tickets instead.")
    assert answered.summary_attempt.can_answer is False
    assert answered.fallback_attempt.can_answer is True
    assert llm.namespaces() == ["answer_summary", "answer_dialogue"]
    [summary_prompt], [dialogue_prompt] = llm.prompts("answer_summary"), llm.prompts("answer_dialogue")
    summaries = summary_prompt.removesuffix(f"\n\nQuestion: {query}")
    assert summaries.count(" | with ") == 1  # k = 1
    assert dialogue_prompt.startswith(f"{summaries}\n\nFull dialogues")  # the same summaries, word for word
    assert dialogue_prompt.endswith(f"\n\nQuestion: {query}")
    dialogues = dialogue_prompt.removeprefix(summaries)
    assert f"[1] Wednesday 2026-10-28 17:30 UTC | with Sam Carter\n{GYM.dialogue_text()}\n\n" in dialogues
    assert f"[2] Tuesday 2026-11-03 14:00 UTC | with Dana Whitfield\n{BUDGET.dialogue_text()}\n\n" in dialogues
    assert f"[3] Monday 2026-10-12 09:00 UTC | with Sam Carter\n{WEDDING.dialogue_text()}\n\n" in dialogues
    assert dialogues.count(" | with ") == 3  # k' = 3: rank 4 stays out
    assert BUDGET.summary not in dialogue_prompt  # rank 2 is shown as a dialogue only
    assert store.calls == [("get_headers", ["e-gym"]), ("get_conversations", ["e-gym", "e-budget", "e-wedding"])]


def test_fallback_with_k_prime_below_k_reads_fewer_dialogues_than_summaries():
    store = FakeStore()
    llm = ScriptedLLM({"answer_summary": [NO], "answer_dialogue": [yes("Eighty a month.")]})

    answered = SystemPipeline(store, llm, MODEL).answer("How much is the rent going up?", RANKED, k=3, k_prime=2)

    assert answered.path == "fallback"
    [dialogue_prompt] = llm.prompts("answer_dialogue")
    assert dialogue_prompt.count(" | with ") == 3 + 2  # three summaries, then two dialogues
    assert store.calls == [
        ("get_headers", ["e-gym", "e-budget", "e-wedding"]),
        ("get_conversations", ["e-gym", "e-budget"]),
    ]


def test_decline_path_when_neither_attempt_can_answer():
    store = FakeStore()
    still_no = {"can_answer": False, "answer": "Nobody mentions a dentist."}
    llm = ScriptedLLM({"answer_summary": [NO], "answer_dialogue": [still_no]})

    answered = SystemPipeline(store, llm, MODEL).answer("Which dentist did Sam recommend?", RANKED, k=3, k_prime=2)

    assert (answered.path, answered.final_answer) == ("declined", "I don't know.")
    assert answered.final_answer == config.DECLINE_ANSWER
    assert answered.summary_attempt.can_answer is False
    assert answered.fallback_attempt.can_answer is False
    assert answered.fallback_attempt.answer == "Nobody mentions a dentist."  # kept, though not used
    assert llm.namespaces() == ["answer_summary", "answer_dialogue"]


def test_empty_pool_declines_without_an_llm_call_or_a_store_read():
    store = FakeStore()
    llm = ScriptedLLM({})

    answered = SystemPipeline(store, llm, MODEL).answer("What did Sam say this morning?", [], k=5, k_prime=3)

    assert answered == Answered(
        final_answer=config.DECLINE_ANSWER,
        path="empty_pool",
        summary_attempt=None,
        fallback_attempt=None,
        usage_by_stage={"answer_summary": Usage(), "answer_dialogue": Usage()},
        latency_by_stage={"answer_summary": 0.0, "answer_dialogue": 0.0},
    )
    assert (llm.calls, store.calls) == ([], [])


def test_time_filter_that_matches_nothing_ends_in_empty_pool():
    store = FakeStore()
    quiet_morning = TimeRange(utc(2026, 9, 1, 8), utc(2026, 9, 1, 12))
    llm = ScriptedLLM({"extraction": [extraction_reply("Sam Carter", quiet_morning)]})

    result = SystemPipeline(store, llm, MODEL).run("What did Sam Carter say that morning?", FakeRanker(store), 5, 3)

    assert (result.path, result.final_answer) == ("empty_pool", config.DECLINE_ANSWER)
    assert (result.rank.ranked, result.rank.trace.pool_size) == ([], 0)
    assert (result.answered.summary_attempt, result.answered.fallback_attempt) == (None, None)
    assert llm.namespaces() == ["extraction"]


@pytest.mark.parametrize(("k", "k_prime"), [(0, 3), (5, 0), (-1, 3)])
def test_answer_rejects_cutoffs_below_one(k, k_prime):
    with pytest.raises(ValueError, match="at least 1"):
        SystemPipeline(FakeStore(), ScriptedLLM({}), MODEL).answer("q", RANKED, k, k_prime)


def test_one_prepare_and_one_ranking_serve_every_cell_of_the_grid():
    store = FakeStore()
    llm = ScriptedLLM(
        {
            "extraction": [extraction_reply("Sam Carter")],
            "answer_summary": [NO] * 4,
            "answer_dialogue": [yes("a"), yes("b"), yes("c"), yes("d")],
        }
    )
    pipeline = SystemPipeline(store, llm, MODEL)
    ranker = FakeRanker(store)
    query = "How much is Sam Carter's rent going up?"

    prepared = pipeline.prepare(query)
    ranked = ranker.rank(config.OWNER_ID, query, prepared.person_id, prepared.time_range)
    ranked_ids = [episode_id for episode_id, _ in ranked]
    cells = {(k, k_prime): pipeline.answer(query, ranked_ids, k, k_prime) for k in (1, 2) for k_prime in (1, 3)}

    assert ranked_ids == ["e-lease", "e-gym", "e-wedding"]
    assert {cell: answered.final_answer for cell, answered in cells.items()} == {
        (1, 1): "a",
        (1, 3): "b",
        (2, 1): "c",
        (2, 3): "d",
    }
    # Cells that differ only in k' repeat the summary attempt word for word, so the
    # response cache answers the repeats. Every dialogue attempt is its own call.
    first, second, third, fourth = llm.prompts("answer_summary")
    assert first == second
    assert third == fourth
    assert first != third
    assert len(set(llm.prompts("answer_dialogue"))) == 4
    assert llm.namespaces().count("extraction") == 1
    assert [name for name, _ in store.calls] == ["list_people"] + ["get_headers", "get_conversations"] * 4


# --------------------------------------------------------------------------- #
# Pipeline: usage and latency                                                  #
# --------------------------------------------------------------------------- #
def test_run_adds_up_usage_and_latency_stage_by_stage(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(pipeline_module, "perf_counter", clock)
    usage = {
        "extraction": Usage(input_tokens=400, output_tokens=30, cost_usd=0.001, latency_s=0.5, calls=1),
        "answer_summary": Usage(input_tokens=1500, output_tokens=40, cost_usd=0.004, latency_s=1.5, calls=1),
        "answer_dialogue": Usage(input_tokens=5000, output_tokens=60, cost_usd=0.012, latency_s=3.0, calls=1),
    }
    # Eight hosted scoring calls of half a second each, run side by side within one second.
    rank_usage = Usage(input_tokens=900, output_tokens=0, cost_usd=0.0005, latency_s=4.0, calls=8)
    store = FakeStore()
    llm = ScriptedLLM(
        {
            "extraction": [extraction_reply("Sam Carter")],
            "answer_summary": [NO],
            "answer_dialogue": [yes("Eighty a month.")],
        },
        usage=usage,
        clock=clock,
        # The first two calls are made now and take a little longer than their own
        # latency. The dialogue call comes back from the response cache at once.
        wall_s={"extraction": 0.75, "answer_summary": 2.0, "answer_dialogue": 0.0},
    )
    # The ranking took 0.9s by the ranker's own clock; the call, with the pool-size
    # diagnostic after it, 1.0s of wall clock.
    ranker = FakeRanker(store, usage=rank_usage, clock=clock, wall_s=1.0, trace_s=0.9)

    result = SystemPipeline(store, llm, MODEL).run("How much is Sam Carter's rent going up?", ranker, k=2, k_prime=1)

    assert (result.path, result.final_answer) == ("fallback", "Eighty a month.")
    assert list(result.usage_by_stage) == list(result.latency_by_stage) == list(STAGES)
    assert result.usage_by_stage == {**usage, "rank": rank_usage}
    assert result.usage.to_json() == pytest.approx(
        {"input_tokens": 7800, "output_tokens": 130, "cost_usd": 0.0175, "latency_s": 9.0, "calls": 11}
    )
    assert result.latency_by_stage == {
        "extraction": 0.75,  # wall clock
        "rank": 0.9,  # the ranker's trace: not the wall clock, nor the 4.0s its parallel calls add up to
        "answer_summary": 2.0,  # wall clock
        "answer_dialogue": 3.0,  # replayed from the cache: what the call took when it was made
    }
    assert result.latency_s == pytest.approx(6.65)
    assert (result.prepared.usage, result.prepared.latency_s) == (usage["extraction"], 0.75)
    assert result.answered.usage_by_stage == {
        "answer_summary": usage["answer_summary"],
        "answer_dialogue": usage["answer_dialogue"],
    }
    assert result.answered.latency_by_stage == {"answer_summary": 2.0, "answer_dialogue": 3.0}
    assert result.rank.trace.usage == rank_usage
    # The system under test leaves the temperature to the model's default (a reasoning
    # model takes none), on every one of its calls.
    assert [call["namespace"] for call in llm.calls] == ["extraction", "answer_summary", "answer_dialogue"]
    assert all(call["temperature"] is None for call in llm.calls)


def test_stage_that_did_not_run_costs_nothing(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(pipeline_module, "perf_counter", clock)
    usage = {
        "extraction": Usage(input_tokens=400, output_tokens=30, cost_usd=0.001, latency_s=0.5, calls=1),
        "answer_summary": Usage(input_tokens=1500, output_tokens=40, cost_usd=0.004, latency_s=1.5, calls=1),
    }
    store = FakeStore()
    llm = ScriptedLLM(
        {"extraction": [extraction_reply("Dana")], "answer_summary": [yes("Travel, by twelve percent.")]},
        usage=usage,
        clock=clock,
        wall_s={"extraction": 0.75, "answer_summary": 2.0},
    )

    result = SystemPipeline(store, llm, MODEL).run("Which budget line is overspent?", FakeRanker(store), k=5, k_prime=3)

    assert result.path == "summary"
    assert result.usage_by_stage == {**usage, "rank": Usage(), "answer_dialogue": Usage()}
    assert result.latency_by_stage == {"extraction": 0.75, "rank": 0.0, "answer_summary": 2.0, "answer_dialogue": 0.0}
    assert result.usage.to_json() == pytest.approx(
        {"input_tokens": 1900, "output_tokens": 70, "cost_usd": 0.005, "latency_s": 2.0, "calls": 2}
    )
    assert result.latency_s == 2.75
