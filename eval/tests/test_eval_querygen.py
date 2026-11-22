"""Offline tests for the query generator.

A synthetic corpus built in code and a scripted ``ChatLLM`` stand in for the
committed dataset and the hosted models. The scripted writer obeys the name and
time lines of the prompt, and the scripted checker accepts, unless a test
scripts something else for one call.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import threading
import types
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import timedelta
from functools import lru_cache
from typing import Any

import pytest

from eval.retrieval import config, dataset
from eval.retrieval.data import queries as qg
from eval.retrieval.data import query_prompts as prompts
from eval.retrieval.interfaces import LLMResult, Usage
from eval.retrieval.schema import ANSWERABLE_TYPES, QUERY_TYPES, Conversation, Person, Query, TimeRange, Turn, pool_bucket
from eval.retrieval.timeutil import pad_time_range

SETTINGS = config.Settings(
    database_url=None,
    gateway="openrouter",
    llm_base_url="",
    llm_api_key=None,
    gen_model="fake/writer",
    summary_model="fake/summary",
    system_model="fake/system",
    judge_model="fake/judge",
    jev_url="",
    jev_api_key=None,
    jev_model="fake/jev",
    llm_concurrency=4,
    jev_concurrency=1,
    device=None,
    max_spend_usd=1.0,
)
COUNTS = {
    "lookup": 4,
    "paraphrase": 3,
    "detail_only": 3,
    "multi_conversation": 3,
    "cross_person": 5,
    "temporal": 10,
    "unanswerable": 4,
    "unknown_name": 3,
}
UNKNOWN_NAMES = ["Priya Raman", "Tobias Engel", "Mireille Laurent", "Kenji Watanabe", "Odalys Ferrer", "Bram Vos"]
TOPICS = ("climbing", "cooking", "travel")
HEAVY = (("Sam", "Okafor"), ("Hannah", "Lindqvist"))
LIGHT = (
    ("Sam", "Whitfield"),
    ("Diego", "Alvarez"),
    ("Mei", "Tanaka"),
    ("Jonas", "Becker"),
    ("Aisha", "Karim"),
    ("Lucia", "Romano"),
    ("Tomasz", "Nowak"),
    ("Greta", "Holloway"),
)
LIGHT_SIZES = (1, 2, 3, 4, 5, 6, 3, 2)


# --------------------------------------------------------------------------- #
# Synthetic corpus                                                             #
# --------------------------------------------------------------------------- #
def _person(first: str, last: str, heavy: bool) -> Person:
    return Person(
        person_id=config.stable_id("test-person", f"{first} {last}"),
        first_name=first,
        last_name=last,
        display_name=f"{first} {last}",
        relationship="friend",
        is_heavy=heavy,
        meta={"persona": f"{first} is a test persona."},
    )


def _conversation(person: Person, storyline: int, beat: int, hours_back: float, topic: str | None) -> Conversation:
    tag = f"{person.last_name.lower()}{storyline}x{beat}"  # a word unique to this conversation
    title = f"{topic or 'project'} plan {storyline}"
    start = config.NOW - timedelta(hours=hours_back)
    return Conversation(
        episode_id=config.stable_id("test-episode", f"{person.person_id}:{storyline}:{beat}"),
        person_id=person.person_id,
        start_time=start,
        end_time=start + timedelta(minutes=10),
        dialogue=[
            Turn(person.first_name, f"The {tag} booking for the {title} is confirmed."),
            Turn("Me", "Good to hear."),
            Turn(person.first_name, f"Also, the locker code is {tag}99."),
        ],
        summary=f"{person.first_name} confirmed the {tag} booking for the {title}.",
        meta={
            "person_name": person.display_name,
            "storyline_id": f"s{storyline}",
            "storyline_title": title,
            "beat_index": beat,
            "shared_topic": topic,
            "key_facts": [f"the {tag} booking is confirmed"],
            "minor_details": [f"the locker code is {tag}99"],
        },
    )


@lru_cache(maxsize=None)
def corpus(dense: bool = True) -> tuple[tuple[Person, ...], tuple[Conversation, ...]]:
    """Two heavy people with 40 conversations each and eight light ones; two people are called Sam.

    Each heavy person has eight storylines of five beats; the last 24 of their
    conversations fall in the weeks before ``config.NOW`` (40 hours apart when
    ``dense``, 80 otherwise), the latest ones today and yesterday.
    """

    spacing = 40 if dense else 80
    people = [_person(first, last, True) for first, last in HEAVY]
    people += [_person(first, last, False) for first, last in LIGHT]
    conversations = []
    for h, person in enumerate(people[:2]):
        for j in range(40):
            storyline, beat = divmod(j, 5)
            recent = (39 - j) * spacing + 5 + 7 * h
            hours_back = recent if j >= 16 else (16 - j) * 450 + 23 * spacing + 100
            topic = TOPICS[storyline] if storyline < len(TOPICS) else None
            conversations.append(_conversation(person, storyline, beat, hours_back, topic))
    for i, person in enumerate(people[2:]):
        size = LIGHT_SIZES[i]
        topic = TOPICS[i % 3] if i % 2 == 0 else None
        for k in range(size):
            conversations.append(_conversation(person, 0, k, 20 + 31 * i + (size - 1 - k) * 500, topic))
    return tuple(people), tuple(conversations)


def person_named(display_name: str) -> Person:
    return next(person for person in corpus()[0] if person.display_name == display_name)


# --------------------------------------------------------------------------- #
# Scripted LLM                                                                 #
# --------------------------------------------------------------------------- #
Script = Callable[[str, str], "dict[str, Any] | None"]


def token(salt: str) -> str:
    """A made-up word, unique to the call, that no summary contains."""

    return "".join(chr(ord("a") + int(digit, 16)) for digit in hashlib.sha1(salt.encode()).hexdigest()[:10])


def written(salt: str, user: str) -> dict[str, Any]:
    """A question that obeys the prompt's name and time lines, and an answer that adds something to it."""

    name = re.search(r'^Name to use in the question: "(.+)"$', user, re.MULTILINE)
    phrase = re.search(r'^Time phrase to use in the question: "(.+)"$', user, re.MULTILINE)
    opening = f"What did {name.group(1)} mention regarding {token(salt)}" if name else f"Who brought up {token(salt)} with me"
    closing = f" {phrase.group(1)}?" if phrase else "?"
    return {"question": opening + closing, "answer": f"It was {token(salt + ':answer')}."}


def verdict(salt: str, user: str) -> dict[str, Any]:
    """The checker's confirmation of a candidate of the salt's type."""

    qtype = salt.split(":")[0]
    if qtype == "unanswerable":
        return {"answerable": False, "evidence": ""}
    if qtype == "detail_only":
        return {
            "answerable_from_summary": False,
            "answerable_from_dialogue": True,
            "answer_matches_dialogue": True,
            "stated_elsewhere": False,
        }
    shown = len(re.findall(r"^Conversation \d+ \(with ", user, re.MULTILINE))
    confirmed = {"answerable": True, "answer_agrees": True, "contributing": list(range(1, shown + 1))}
    if qtype == "temporal":
        confirmed["time_phrase_dates_the_conversation"] = True
    return confirmed


class FakeLLM:
    """Scripted ``ChatLLM``. ``write`` and ``check`` may return a response for a call, or None for the default."""

    def __init__(self, write: Script | None = None, check: Script | None = None) -> None:
        self.calls: list[tuple[str, str, str, float | None, str, str]] = []
        self._lock = threading.Lock()
        self._write = write or (lambda salt, user: None)
        self._check = check or (lambda salt, user: None)

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
        with self._lock:
            self.calls.append((namespace, salt, model, temperature, system, user))
        if namespace == qg.WRITER_NAMESPACE:
            data = self._write(salt, user) or written(salt, user)
        else:
            data = self._check(salt, user) or verdict(salt, user)
        return LLMResult(data=data, usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, calls=1), model=model)

    def prompt(self, namespace: str, salt: str) -> str:
        return next(call[5] for call in self.calls if call[:2] == (namespace, salt))


def generator(llm: FakeLLM | None = None, counts: dict[str, int] | None = None, workers: int = 4, dense: bool = True):
    people, conversations = corpus(dense)
    return qg.QueryGenerator(llm or FakeLLM(), SETTINGS, people, conversations, UNKNOWN_NAMES, counts or COUNTS, workers)


@lru_cache(maxsize=None)
def baseline() -> tuple[Query, ...]:
    """The query set of an unscripted run (do not mutate: copy with ``copies``)."""

    return tuple(generator().generate())


def copies(queries) -> list[Query]:
    return [Query.from_json(json.loads(json.dumps(query.to_json()))) for query in queries]


def dumps(queries) -> list[str]:
    return [json.dumps(query.to_json(), sort_keys=True) for query in queries]


def slot(queries, qtype: str, index: int) -> Query:
    return next(query for query in queries if query.query_id == config.stable_id("query", f"{qtype}:{index}"))


def of_type(queries, qtype: str) -> list[Query]:
    return [query for query in queries if query.type == qtype]


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #
def test_counts_name_forms_and_gold_labels():
    people, conversations = corpus()
    llm = FakeLLM()
    queries = generator(llm).generate()

    assert Counter(query.type for query in queries) == COUNTS
    assert [query.type for query in queries] == [t for t in QUERY_TYPES for _ in range(COUNTS[t])]
    assert qg.validate_queries(queries, people, conversations, COUNTS) == []

    forms = qg.name_forms(people)
    assert forms[person_named("Sam Okafor").person_id] == "Sam Okafor"
    assert forms[person_named("Sam Whitfield").person_id] == "Sam Whitfield"
    assert forms[person_named("Hannah Lindqvist").person_id] == "Hannah"
    named = [query for query in queries if query.gold_person_id is not None]
    assert {query.gold_name for query in named} >= {"Sam Okafor", "Hannah"}
    for query in named:
        assert query.gold_name == forms[query.gold_person_id]
        assert qg.contains_phrase(query.query, query.gold_name)
    assert all(query.gold_name is None and query.gold_person_id is None for query in of_type(queries, "cross_person"))

    for query in queries:
        assert set(query.meta) >= {"source_episode_ids", "subkind", "writer_model", "checker_model", "attempt"}
        assert query.meta["writer_model"] == "fake/writer"
        if query.type in ANSWERABLE_TYPES:
            assert query.gold_episode_ids and query.gold_episode_ids == query.meta["source_episode_ids"]
            assert query.gold_answer and query.meta["checker_model"] == "fake/judge"
        else:
            assert query.gold_episode_ids == [] and query.gold_answer is None and query.meta["source_episode_ids"] == []
    assert all(len(query.gold_episode_ids) in (2, 3) for query in of_type(queries, "multi_conversation"))
    assert all("storyline_id" in query.meta for query in of_type(queries, "multi_conversation"))
    assert Counter(q.meta["subkind"] for q in of_type(queries, "cross_person")) == {"single": 3, "group": 2}
    for query in of_type(queries, "cross_person"):
        owners = [next(c for c in conversations if c.episode_id == e).person_id for e in query.gold_episode_ids]
        assert len(set(owners)) == len(owners) == (1 if query.meta["subkind"] == "single" else len(owners)) >= 1
        assert (query.meta["subkind"] == "group") == ("shared_topic" in query.meta) == (len(owners) > 1)
    assert Counter(q.meta["subkind"] for q in of_type(queries, "unanswerable")) == {"absent_topic": 2, "missing_detail": 2}
    heavy_ids = {person.person_id for person in people if person.is_heavy}
    assert sum(query.gold_person_id in heavy_ids for query in of_type(queries, "unanswerable")) == 2
    unknown = of_type(queries, "unknown_name")
    assert Counter(query.meta["subkind"] for query in unknown) == {"first_name": 2, "full_name": 1}
    for query in unknown:
        full = next(name for name in UNKNOWN_NAMES if name.startswith(query.gold_name))
        assert query.gold_name == (full if query.meta["subkind"] == "full_name" else full.split()[0])
        assert query.gold_person_id is None and query.meta["checker_model"] is None

    writer_calls = [call for call in llm.calls if call[0] == qg.WRITER_NAMESPACE]
    checker_calls = [call for call in llm.calls if call[0] == qg.CHECKER_NAMESPACE]
    assert len(writer_calls) == len(queries) and len(checker_calls) == len(queries) - COUNTS["unknown_name"]
    assert {(call[2], call[3]) for call in writer_calls} == {("fake/writer", 0.7)}
    assert {(call[2], call[3]) for call in checker_calls} == {("fake/judge", 0.0)}
    assert len({call[1] for call in writer_calls}) == len(writer_calls)  # one salt per attempt


def test_prompts_carry_what_each_type_needs():
    llm = FakeLLM()
    queries = generator(llm).generate()

    def writer_prompt(query: Query) -> str:
        index = of_type(queries, query.type).index(query)
        return llm.prompt(qg.WRITER_NAMESPACE, f"{query.type}:{index}:{query.meta['attempt']}")

    for query in queries:
        text = writer_prompt(query)
        name_line = prompts.NAME_LINE.format(name=query.gold_name) if query.gold_name else prompts.NO_NAME_LINE
        assert name_line in text
        assert (prompts.NO_TIME_LINE in text) == (query.type != "temporal")
    paraphrase = of_type(queries, "paraphrase")[0]
    assert "Words to avoid: " in writer_prompt(paraphrase) and "booking" in writer_prompt(paraphrase)
    assert "the locker code is" in writer_prompt(of_type(queries, "detail_only")[0])
    # A lookup shows the other beats of its source's storyline, and no other storyline.
    lookup = of_type(queries, "lookup")[1]
    conversations = corpus()[1]
    source = next(c for c in conversations if c.episode_id == lookup.gold_episode_ids[0])
    own = [c for c in conversations if c.person_id == source.person_id]
    thread = [c for c in own if c.meta["storyline_id"] == source.meta["storyline_id"]]
    assert len(thread) == 5 and "must not be answerable from any of them" in writer_prompt(lookup)
    assert all((c.summary in writer_prompt(lookup)) == (c in thread) for c in own)
    missing = next(q for q in of_type(queries, "unanswerable") if q.meta["subkind"] == "missing_detail")
    assert "a detail that was never stated" in writer_prompt(missing) and "Dialogue:" in writer_prompt(missing)
    absent = next(q for q in of_type(queries, "unanswerable") if q.meta["subkind"] == "absent_topic")
    assert "Dialogue:" not in writer_prompt(absent) and "Every thread of conversation" in writer_prompt(absent)
    assert "Model conversation" in writer_prompt(of_type(queries, "unknown_name")[0])

    # Every writer is told not to ask a yes/no question; the summary-path writers are told to
    # answer from the summaries alone, and their checker accepts only answers the summaries state.
    assert all("it is not a yes/no question" in system for system in (prompts.WRITER_SYSTEM, prompts.WRITER_SYSTEM_NO_ANSWER))
    for query in queries:
        if query.type in ("lookup", "paraphrase", "multi_conversation", "cross_person", "temporal"):
            assert "The answer states only what the summar" in writer_prompt(query)
            checker = llm.prompt(qg.CHECKER_NAMESPACE, f"{query.type}:{of_type(queries, query.type).index(query)}:{query.meta['attempt']}")
            assert "every claim in it is stated by the summaries" in checker
        elif query.type == "detail_only":
            assert "The answer states only what the summar" not in writer_prompt(query)


def test_temporal_specific_questions_are_written_against_the_period():
    """A "specific" temporal question is checked against what the time filter leaves in the pool."""

    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    llm = FakeLLM()
    queries = generator(llm).generate()
    specific = [q for q in of_type(queries, "temporal") if q.meta["subkind"] == "specific"]
    assert specific
    for query in specific:
        expression = index.expressions[query.meta["time_expression"]]
        scope = [c for c in conversations if query.gold_person_id in (None, c.person_id)]
        inside = {c.episode_id for c in scope if expression.time_range.contains(c.start_time)}
        salt = f"temporal:{of_type(queries, 'temporal').index(query)}:{query.meta['attempt']}"
        prompt = llm.prompt(qg.WRITER_NAMESPACE, salt)
        assert prompts._PLACE_TIME_PHRASE in prompt
        assert "time_phrase_dates_the_conversation" in llm.prompt(qg.CHECKER_NAMESPACE, salt)
        period = prompt.split(prompts._PERIOD_OTHERS)[1].split("\n\n")[0]
        thread = prompt.split(prompts._OTHERS)[1].split("\n\n")[0] if prompts._OTHERS in prompt else ""
        in_period = [c for c in conversations if c.summary in period]
        in_thread = [c for c in conversations if c.summary in thread]
        assert len(in_period) == min(qg.MAX_OTHERS, len(inside) - len(query.gold_episode_ids))
        assert all(c.episode_id in inside and c.episode_id not in query.gold_episode_ids for c in in_period)
        assert len(in_period) + len(in_thread) <= qg.MAX_OTHERS
        assert all(c.episode_id not in inside for c in in_thread)  # the thread fills room the period leaves
        if in_thread:
            keys = {qg.storyline_key(index.conversation[e]) for e in query.gold_episode_ids}
            assert all(qg.storyline_key(c) in keys for c in in_thread)

    # In a person's window the nearest conversations in time are the rivals.
    hannah = person_named("Hannah Lindqvist")
    window = next(w for w in index.windows if w.person == hannah and w.expression.key == "past_30_days")
    source = window.inside[len(window.inside) // 2]
    rivals = index.period_rivals(window, (source,), qg.MAX_OTHERS)
    by_distance = sorted((c for c in window.inside if c is not source), key=lambda c: abs(c.start_time - source.start_time))
    assert list(rivals) == sorted(by_distance[: qg.MAX_OTHERS], key=lambda c: c.start_time)
    # In everyone's window other people's conversations come first, those on the source's topic before the rest.
    window = next(w for w in index.windows if w.person is None and w.expression.key == "past_30_days")
    whitfield = person_named("Sam Whitfield")
    source = next(c for c in window.inside if c.person_id == whitfield.person_id)
    assert source.meta["shared_topic"] == "climbing"
    rivals = index.period_rivals(window, (source,), qg.MAX_OTHERS)
    assert len(rivals) == qg.MAX_OTHERS and all(c.person_id != whitfield.person_id for c in rivals)
    on_topic = [c for c in window.inside if c.person_id != whitfield.person_id and c.meta["shared_topic"] == "climbing"]
    assert on_topic and set(c.episode_id for c in on_topic) <= {c.episode_id for c in rivals}
    assert index.period_rivals(window, (source,), 0) == ()


def test_unanswerable_checker_sees_what_every_dialogue_states():
    """A detail that only a dialogue holds counts as held, even for a heavy person whose dialogues are not shown."""

    people, conversations = corpus()
    llm = FakeLLM()
    queries = generator(llm).generate()
    heavy_ids = {person.person_id for person in people if person.is_heavy}
    for index, query in enumerate(of_type(queries, "unanswerable")):
        prompt = llm.prompt(qg.CHECKER_NAMESPACE, f"unanswerable:{index}:{query.meta['attempt']}")
        own = [c for c in conversations if c.person_id == query.gold_person_id]
        assert all(c.meta["minor_details"][0] in prompt and c.meta["key_facts"][0] in prompt for c in own)
        assert prompt.count(prompts._FACTS_LINE) == len(own)
        shown = prompt.count("Dialogue of conversation")
        if query.gold_person_id not in heavy_ids:
            assert shown == len(own) and "the full dialogue of every one of them" in prompt
        elif query.meta["subkind"] == "missing_detail":
            assert 0 < shown < len(own)
        else:
            assert shown == 0 and "no full dialogues" in prompt
    assert prompts.dialogue_facts(conversations[0]) == [*conversations[0].meta["key_facts"], *conversations[0].meta["minor_details"]]
    bare = Conversation.from_json({**conversations[0].to_json(), "meta": {}})
    assert prompts.dialogue_facts(bare) == []


def test_detail_only_is_checked_against_the_other_beats():
    """The writer sees what the other beats' dialogues state; the checker reads those dialogues in full."""

    people, conversations = corpus()
    llm = FakeLLM()
    queries = generator(llm).generate()
    for index, query in enumerate(of_type(queries, "detail_only")):
        salt = f"detail_only:{index}:{query.meta['attempt']}"
        source = next(c for c in conversations if c.episode_id == query.gold_episode_ids[0])
        others = [c for c in conversations if c.person_id == source.person_id
                  and c.meta["storyline_id"] == source.meta["storyline_id"] and c is not source]
        writer = llm.prompt(qg.WRITER_NAMESPACE, salt)
        assert all(c.summary in writer and c.meta["minor_details"][0] in writer for c in others)
        assert "No other conversation listed states the detail" in writer
        checker = llm.prompt(qg.CHECKER_NAMESPACE, salt)
        assert checker.count("Other conversation ") == len(others) and "stated_elsewhere" in checker
        assert all(c.dialogue_text() in checker for c in others)
    alone = prompts.check_detail("Q?", "A.", conversations[0], (), qg.Corpus(people, conversations).person).user
    assert "stated_elsewhere" not in alone and "Other conversation" not in alone


def _with_meta(conversation: Conversation, **meta: Any) -> Conversation:
    return Conversation.from_json({**conversation.to_json(), "meta": {**conversation.meta, **meta}})


def test_facts_the_dialogue_never_states_are_not_held_memory():
    """The generator's unstated and in-summary records decide what counts as stated and as left out."""

    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    source = conversations[0]
    [fact], [detail] = source.meta["key_facts"], source.meta["minor_details"]
    details = [detail, "the bus leaves at 7:15", "the café is called Luna"]
    recorded = _with_meta(
        source,
        minor_details=details,
        unstated_facts=[fact],
        unstated_details=["the bus leaves at 7:15"],
        details_in_summary=[detail],
    )

    assert prompts.dialogue_facts(recorded) == [detail, "the café is called Luna"]
    assert prompts.dialogue_facts(_with_meta(source, unstated_facts=[], unstated_details=[])) == [fact, detail]
    checker = prompts.check_unanswerable("Q?", index.person[source.person_id], [recorded], set()).user
    assert "the café is called Luna" in checker and fact not in checker and "7:15" not in checker
    assert qg.details_left_out(recorded) == ["the café is called Luna"]
    assert qg.details_left_out(_with_meta(source, minor_details=details, unstated_details=[], details_in_summary=[])) == details
    # A corpus without the record falls back to comparing the wording with the summary.
    assert qg.details_left_out(source) == qg.details_missing_from_summary(source) == [detail]
    assert qg.details_left_out(_with_meta(source, minor_details=[f"the {source.summary}"])) == []

    # The detail-only writer's hints are the details left out, and the thread list states only what was said.
    recorded_corpus = [recorded, *conversations[1:]]
    worker = qg.QueryGenerator(FakeLLM(), SETTINGS, people, recorded_corpus, UNKNOWN_NAMES, COUNTS, 1)
    assert worker.corpus.left_out[source.episode_id] == ("the café is called Luna",)
    others = worker.corpus.neighbours((recorded,), qg.MAX_OTHERS)
    person = worker.corpus.person[source.person_id]
    target = qg.Target("detail_only", 0, 0, person=person, name=worker.corpus.names[person.person_id], sources=(recorded,), others=others)
    hints = worker._writer_prompt(target).user.split("Details the dialogue states that the summary seems to leave out")[1].split("\n\n")[0]
    assert "- the café is called Luna" in hints and detail not in hints and "7:15" not in hints
    thread = worker._writer_prompt(qg.Target("detail_only", 0, 0, person=person, name=target.name, sources=(others[0],), others=(recorded,))).user
    assert "the café is called Luna" in thread and "7:15" not in thread and fact not in thread


def _with_recaps(conversations: tuple[Conversation, ...]) -> list[Conversation]:
    """The fixture corpus with the last beat of every heavy storyline padded in code as a recap of the beat before."""

    index = qg.Corpus(corpus()[0], conversations)
    recapped = []
    for conversation in conversations:
        beats = index.storylines[qg.storyline_key(conversation)]
        if index.person[conversation.person_id].is_heavy and conversation is beats[-1]:
            previous = beats[-2]
            conversation = _with_meta(conversation, recap=True, key_facts=list(previous.meta["key_facts"]), minor_details=[])
        recapped.append(conversation)
    return recapped


def test_a_recap_and_the_beat_it_restates_are_never_a_lone_gold():
    people, conversations = corpus()
    recapped = _with_recaps(conversations)
    index = qg.Corpus(people, recapped)
    recaps = [c for c in recapped if c.meta.get("recap")]
    assert len(recaps) == 16
    for recap in recaps:
        previous = index.storylines[qg.storyline_key(recap)][-2]
        assert index.twins[recap.episode_id] == (previous,) and index.twins[previous.episode_id] == (recap,)
        assert not index.stands_alone(recap) and not index.stands_alone(previous)
    twinned = {c.episode_id for c in recapped if not index.stands_alone(c)}
    assert len(twinned) == 32 and all(index.stands_alone(c) for c in recapped if c.episode_id not in twinned)

    queries = qg.QueryGenerator(FakeLLM(), SETTINGS, people, recapped, UNKNOWN_NAMES, COUNTS, 4).generate()
    assert qg.validate_queries(queries, people, recapped, COUNTS) == []

    # A recap of a period is answered by whatever the time filter leaves, twin or not; every
    # other single-source query is drawn on a conversation that answers it alone.
    def lone_gold(queries) -> list[str]:
        return [
            q.gold_episode_ids[0]
            for q in queries
            if q.answerable and len(q.gold_episode_ids) == 1 and q.meta["subkind"] != "recap"
        ]

    assert len(lone_gold(queries)) >= COUNTS["lookup"] + COUNTS["paraphrase"] + COUNTS["detail_only"] + 1
    assert not twinned & set(lone_gold(queries))
    assert any(q.meta["subkind"] == "specific" for q in of_type(queries, "temporal"))
    # Without the recaps the same draws do land on those conversations: it is the twins that keep them out.
    assert twinned & set(lone_gold(baseline()))

    # A time window whose free conversations are all twins still carries a specific query, as the pair.
    worker = qg.QueryGenerator(FakeLLM(), SETTINGS, people, recapped, UNKNOWN_NAMES, COUNTS, 1)
    recap = recaps[0]
    previous = index.storylines[qg.storyline_key(recap)][-2]
    assert worker._beat_pairs([recap, previous]) == [(previous, recap)] and worker._beat_pairs([recap]) == []
    for seed in range(20):
        assert worker._specific_sources([recap, previous], random.Random(seed)) == (previous, recap)
    free = [previous, recap, conversations[-1]]
    drawn = {tuple(c.episode_id for c in worker._specific_sources(free, random.Random(seed))) for seed in range(20)}
    assert drawn == {(previous.episode_id, recap.episode_id), (conversations[-1].episode_id,)}


def test_group_questions_are_written_against_the_members_other_beats():
    """A group's others are rivals on the topic and the sources' own storylines, as for a single who-question."""

    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    llm = FakeLLM()
    queries = generator(llm).generate()
    groups = [q for q in of_type(queries, "cross_person") if q.meta["subkind"] == "group"]
    assert groups
    heading = "and these people's other conversations (summaries only). The question must not fit any of them:"
    own_shown = []
    for query in groups:
        salt = f"cross_person:{of_type(queries, 'cross_person').index(query)}:{query.meta['attempt']}"
        prompt = llm.prompt(qg.WRITER_NAMESPACE, salt)
        assert heading in prompt and "including the same people's other conversations" in prompt
        block = prompt.split(heading)[1].split("\n\n")[0]
        shown = [c for c in conversations if c.summary in block]
        sources = [index.conversation[e] for e in query.gold_episode_ids]
        members = {c.person_id for c in sources}
        keys = {qg.storyline_key(c) for c in sources}
        rivals = [c for c in shown if c.person_id not in members]
        own = [c for c in shown if c.person_id in members]
        on_topic = [c for c in index.by_topic[query.meta["shared_topic"]] if c.person_id not in members]
        assert len(rivals) == min(qg.CROSS_RIVALS, len(on_topic)) and all(c in on_topic for c in rivals)
        assert len(shown) <= qg.MAX_OTHERS
        assert all(qg.storyline_key(c) in keys and c.episode_id not in query.gold_episode_ids for c in own)
        assert len(own) == min(qg.MAX_OTHERS - len(rivals), sum(len(index.storylines[key]) - 1 for key in keys))
        own_shown.append(len(own))
    assert any(own_shown)


def test_query_expressions_leave_out_the_current_month():
    from eval.retrieval.timeutil import time_expressions

    assert config.NOW.month == 11
    keys = {expression.key for expression in qg.query_expressions()}
    assert "in_november" not in keys and "in_october" not in keys and "in_december" in keys and "in_september" in keys
    assert len(keys) == len(time_expressions()) - 1 and "in_november" not in qg.Corpus(*corpus()).expressions
    march = config.NOW.replace(month=3, day=10)
    assert "in_march" not in {e.key for e in qg.query_expressions(march)} and "in_march" in {e.key for e in time_expressions(march)}
    assert not any(q.meta["time_expression"] == "in_november" for q in of_type(baseline(), "temporal"))


# --------------------------------------------------------------------------- #
# Validators                                                                   #
# --------------------------------------------------------------------------- #
def test_question_validators_reject_what_they_should():
    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    hannah, okafor = person_named("Hannah Lindqvist"), person_named("Sam Okafor")
    source = index.by_person[hannah.person_id][0]
    named = qg.Target("lookup", 0, 0, person=hannah, name="Hannah", sources=(source,))
    nameless = qg.Target("cross_person", 0, 0, sources=(source,))
    last_week = index.expressions["last_week"]
    temporal = qg.Target("temporal", 0, 0, person=hannah, name="Hannah", sources=(source,), expression=last_week)

    def problem(target: qg.Target, question: object) -> str | None:
        return qg.question_problem(index, target, question)

    assert problem(named, "What did Hannah say about the permit?") is None
    assert problem(named, "What did she say about the permit?") == "name_missing"
    assert problem(named, "What did Hannahs friend say?") == "name_missing"
    assert problem(named, "What did Hannah say about Diego's permit?") == "other_person_named"
    assert problem(named, "What did Hannah tell Riley about the permit?") == "owner_named"
    assert problem(named, "According to the summary, what did Hannah say?") == "mentions_source_material"
    assert problem(named, "What did Hannah say about the permit last week?") == "time_expression"
    assert problem(named, "What did Hannah say about the permit in March?") == "time_expression"
    assert problem(named, "Hannah mentioned the permit.") == "not_one_question"
    assert problem(named, None) == "writer_declined"
    assert problem(named, "  ") == "question_missing"
    assert problem(named, "What did Hannah say " + "and say " * 20 + "?") == "question_too_long"

    sam = qg.Target("lookup", 0, 0, person=okafor, name="Sam Okafor", sources=(source,))
    assert problem(sam, "What did Sam Okafor's manager decide?") is None
    assert problem(sam, "What did Sam decide?") == "name_missing"
    assert problem(sam, "What did Sam Okafor say when Sam called?") == "other_person_named"

    assert problem(nameless, "Who told me about the permit?") is None
    assert problem(nameless, "Who told me about Hannah's permit?") == "person_named"
    assert problem(nameless, "Did Sam Whitfield tell me about the permit?") == "person_named"

    assert problem(temporal, "What did Hannah say about the permit last week?") is None
    assert problem(temporal, "Last week, what did Hannah say about the permit?") is None
    assert problem(temporal, "What did Hannah say about the permit?") == "time_phrase_missing"
    assert problem(temporal, "What did Hannah say about the permit two weeks ago?") == "time_phrase_missing"
    assert problem(temporal, "What did Hannah say last week about Friday?") == "extra_time_expression"
    assert problem(temporal, "What did Hannah say about the party a few weeks back last week?") == "extra_time_expression"
    # The phrase must date the conversation, not a thing talked about.
    assert problem(temporal, "What did Hannah think of last week's climbing plan?") == "time_phrase_dates_a_thing"
    assert problem(temporal, "What did Hannah think of last week’s climbing plan?") == "time_phrase_dates_a_thing"
    today = qg.Target("temporal", 0, 0, person=hannah, name="Hannah", sources=(source,), expression=index.expressions["today"])
    assert problem(today, "What did Hannah say about today's booking?") == "time_phrase_dates_a_thing"
    assert problem(today, "What did Hannah say today about the booking?") is None

    # A question that asks for confirmation, or that carries its own answer, tests nothing.
    assert problem(named, "Did Hannah confirm the permit?") == "yes_no_question"
    assert problem(named, "Is Hannah's permit approved?") == "yes_no_question"
    assert problem(named, "Has Hannah decided on the permit?") == "yes_no_question"
    assert problem(named, "Which permit did Hannah apply for?") is None
    assert qg.question_problem(index, named, "What did Hannah say about the permit?", "The permit.") == "question_contains_answer"
    assert qg.question_problem(index, named, "What did Hannah say about the permit?", "She said it was approved.") is None
    assert qg.question_problem(index, named, "When does Hannah's flight leave?", "It leaves at 9.") is None
    assert qg.question_problem(index, named, "What did Hannah say about the permit?", None) is None
    assert qg.question_carries_answer("Is the code 4471?", "Yes, it is 4471.")
    assert not qg.question_carries_answer("What is the code?", "It is 4471.")
    assert qg.question_carries_answer("What did Hannah Lindqvist plan?", "Hannah planned it.", qg.person_words(hannah))

    # Paraphrase: at most one content word of the question may occur in the summary.
    paraphrase = qg.Target("paraphrase", 0, 0, person=hannah, name="Hannah", sources=(source,))
    assert "confirmed" in source.summary and "booking" in source.summary and "plan" in source.summary
    assert problem(paraphrase, "Which reservation did Hannah lock in?") is None
    assert problem(paraphrase, "Which booking did Hannah lock in?") is None
    assert problem(paraphrase, "Which bookings did Hannah confirm?") == "reuses_summary_wording"
    assert problem(paraphrase, "What is Hannah planning to reserve?") is None
    assert problem(paraphrase, "What is Hannah planning for her bookings?") == "reuses_summary_wording"
    assert qg.shared_content_words("What did Hannah Lindqvist confirm?", source.summary, qg.person_words(hannah)) == {"confirm"}


def test_target_answer_and_checker_validators():
    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    hannah = person_named("Hannah Lindqvist")
    own = index.by_person[hannah.person_id]
    last_week = index.expressions["last_week"]
    inside = [c for c in own if last_week.time_range.contains(c.start_time)]
    assert inside and not last_week.time_range.contains(own[0].start_time)

    def temporal(*sources: Conversation) -> qg.Target:
        return qg.Target("temporal", 0, 0, person=hannah, name="Hannah", sources=sources, expression=last_week)

    assert qg.target_problem(index, temporal(*inside)) is None
    assert qg.target_problem(index, temporal(inside[0], own[0])) == "gold_outside_time_range"
    assert qg.target_problem(index, qg.Target("lookup", 0, 0, person=hannah, name="Sam", sources=(own[0],))) == (
        "name_does_not_resolve"
    )
    for name, expected in (("Priya", None), ("Priya Raman", None), ("Hannah", "unknown_name_resolves"),
                           ("Sam", "unknown_name_resolves"), ("Diego Alvarez", "unknown_name_resolves")):
        assert qg.target_problem(index, qg.Target("unknown_name", 0, 0, name=name)) == expected

    assert qg.answer_problem("It is on the fourth floor. Bring the badge.") is None
    assert qg.answer_problem("") == "answer_missing"
    assert qg.answer_problem(None) == "answer_missing"
    assert qg.answer_problem("One. Two. Three. Four.") == "answer_too_long"

    detail = qg.Target("detail_only", 0, 0, sources=(own[0],))
    confirmed = {"answerable_from_summary": False, "answerable_from_dialogue": True, "answer_matches_dialogue": True}
    assert qg.check_problem(detail, confirmed) is None
    assert qg.check_problem(detail, {**confirmed, "answerable_from_summary": True}) == "answerable_from_summary"
    assert qg.check_problem(detail, {**confirmed, "answerable_from_dialogue": False}) == "not_answerable_from_dialogue"
    assert qg.check_problem(detail, {**confirmed, "answer_matches_dialogue": False}) == "answer_disagrees_with_dialogue"
    assert qg.check_problem(detail, {"answerable_from_summary": "no"}) == "checker_malformed"
    # With other beats shown, the checker must also say whether one of them states the detail.
    detail = qg.Target("detail_only", 0, 0, sources=(own[0],), others=(own[1],))
    assert qg.check_problem(detail, confirmed) == "checker_malformed"
    assert qg.check_problem(detail, {**confirmed, "stated_elsewhere": False}) is None
    assert qg.check_problem(detail, {**confirmed, "stated_elsewhere": True}) == "detail_stated_elsewhere"

    multi = qg.Target("multi_conversation", 0, 0, sources=(own[0], own[1]))
    agreed = {"answerable": True, "answer_agrees": True, "contributing": [1, 2]}
    assert qg.check_problem(multi, agreed) is None
    assert qg.check_problem(multi, {**agreed, "contributing": [2]}) == "source_does_not_contribute"
    assert qg.check_problem(multi, {**agreed, "answerable": False}) == "not_answerable_from_summaries"
    assert qg.check_problem(multi, {**agreed, "answer_agrees": False}) == "answer_disagrees_with_summaries"
    assert qg.check_problem(multi, {"answerable": True, "answer_agrees": True}) == "checker_malformed"
    # A temporal candidate also needs the verdict on what the time phrase dates.
    dated = {**agreed, "contributing": [1], "time_phrase_dates_the_conversation": True}
    assert qg.check_problem(temporal(inside[0]), dated) is None
    assert qg.check_problem(temporal(inside[0]), {**dated, "time_phrase_dates_the_conversation": False}) == "time_phrase_dates_a_thing"
    assert qg.check_problem(temporal(inside[0]), {**agreed, "contributing": [1]}) == "checker_malformed"

    unanswerable = qg.Target("unanswerable", 0, 0, person=hannah, name="Hannah")
    assert qg.check_problem(unanswerable, {"answerable": False}) is None
    assert qg.check_problem(unanswerable, {"answerable": True}) == "answerable"
    assert qg.check_problem(unanswerable, {}) == "checker_malformed"


def test_text_helpers():
    assert {qg.light_stem(word) for word in ("planned", "plans", "planning", "plan")} == {"plan"}
    assert qg.light_stem("moved") == qg.light_stem("move") == qg.light_stem("moving")
    assert qg.light_stem("parties") == qg.light_stem("party")
    assert qg.content_words("What did Sam's sister say about the two new kittens?", {"sam"}) == [
        "sister", "two", "new", "kittens",
    ]
    assert qg.time_words("Where does Hannah keep the spare key?") == []
    assert qg.time_words("What did she say on Monday and again 3 days ago, in 2025?") == ["Monday", "ago", "2025"]
    assert qg.time_words("What happened in the past 10 days?") == ["past 10 days"]
    for phrase in ("a few weeks back", "a couple of days earlier", "3 weeks before", "over the weekend", "during the holidays",
                   "at the end of the month", "the start of last year", "the last time we spoke", "earlier", "last summer",
                   "this spring", "the past couple of months"):
        assert qg.time_words(f"What did Sam say about the party {phrase}?"), phrase
    assert qg.time_words("What did Sam say about the weekend plan and the summer house?") == []
    conversation = corpus()[1][0]
    assert qg.details_missing_from_summary(conversation) == conversation.meta["minor_details"]
    conversation = Conversation.from_json({**conversation.to_json(), "summary": "They shared the locker code."})
    assert qg.details_missing_from_summary(conversation) == []


def test_the_query_type_vocabulary_is_spelled_once():
    assert tuple(config.QUERY_TYPE_COUNTS) == QUERY_TYPES
    assert tuple(qg.scaled_counts(config.N_QUERIES)) == QUERY_TYPES


def test_metered_llm_counts_per_namespace_and_reports_the_hosted_spend():
    class CachingLLM:
        def complete_json(self, **kwargs: Any) -> LLMResult:
            cached = kwargs["salt"].endswith(":cached")
            usage = Usage(input_tokens=100, output_tokens=20, cost_usd=0.01, calls=1)
            return LLMResult(data={}, usage=usage, model=kwargs["model"], cached=cached)

    metered = qg.MeteredLLM(CachingLLM())
    for namespace, salt in (("beats", "a"), ("beats", "b:cached"), ("summary", "c"), ("summary", "d"), ("summary", "e:cached")):
        metered.complete_json(model="m", system="s", user="u", namespace=namespace, salt=salt)
    metered.complete_json(model="m", system="s", user="u", salt="f")

    assert list(metered.stats) == ["beats", "summary", "default"]
    assert (metered.stats["beats"].calls, metered.stats["beats"].cached) == (2, 1)
    assert (metered.stats["summary"].calls, metered.stats["summary"].cached) == (3, 1)
    assert metered.stats["beats"].cost_usd == pytest.approx(0.02)  # what the replies cost when first made
    assert metered.stats["beats"].hosted == Usage(input_tokens=100, output_tokens=20, cost_usd=0.01, calls=1)
    assert (metered.hosted_calls, metered.cached_calls) == (4, 2)
    hosted = metered.hosted
    assert (hosted.input_tokens, hosted.output_tokens, hosted.calls) == (400, 80, 4)
    assert hosted.cost_usd == pytest.approx(0.04)
    assert metered.report() == "LLM calls: 4 hosted, 2 from the cache; hosted spend: $0.0400 (400 input tokens, 80 output tokens)"
    assert qg.MeteredLLM(CachingLLM()).report() == "LLM calls: 0 hosted, 0 from the cache; hosted spend: $0.0000 (0 input tokens, 0 output tokens)"


# --------------------------------------------------------------------------- #
# Rejections, reuse, temporal targeting                                        #
# --------------------------------------------------------------------------- #
def test_a_rejection_draws_the_next_target_deterministically():
    def reject_lookup(salt: str, user: str) -> dict[str, Any] | None:
        return {"answerable": False, "answer_agrees": True, "contributing": [1]} if salt == "lookup:1:0" else None

    runs = [generator(FakeLLM(check=reject_lookup)) for _ in range(2)]
    first, second = (run.generate() for run in runs)
    assert dumps(first) == dumps(second)
    assert runs[0].rejections["lookup"] == {"not_answerable_from_summaries": 1}

    # Lookups are drawn after every source-consuming type, so the baseline shows the rejected target.
    rejected_source = slot(baseline(), "lookup", 1).meta["source_episode_ids"]
    redrawn = slot(first, "lookup", 1)
    assert redrawn.meta["attempt"] == 1 and redrawn.meta["source_episode_ids"] != rejected_source
    lookup_sources = [e for query in of_type(first, "lookup") for e in query.meta["source_episode_ids"]]
    assert rejected_source[0] not in lookup_sources  # never drawn twice within one type
    others = [query for query in first if query.query_id != redrawn.query_id]
    assert dumps(others) == dumps([query for query in baseline() if query.query_id != redrawn.query_id])


def test_validator_rejections_are_counted_and_redrawn():
    def write(salt: str, user: str) -> dict[str, Any] | None:
        if salt == "temporal:0:0":
            return {"question": "What did we talk about?", "answer": "Plans."}
        if salt == "cross_person:0:0":
            return {"question": "What did Hannah tell me about the route?", "answer": "Plans."}
        if salt == "paraphrase:0:0":
            name = re.search(r'^Name to use in the question: "(.+)"$', user, re.MULTILINE).group(1)
            return {"question": f"Which booking did {name} confirm for the plan?", "answer": "Yes."}
        if salt == "lookup:0:0":
            return {"question": None}
        if salt == "lookup:2:0":
            return {"answer": "No question at all."}
        if salt == "unknown_name:0:0":
            return {"question": "What did they say about the trip today?"}
        if salt == "cross_person:1:0":  # the same text as the next slot's question
            return written("cross_person:2:0", user)
        return None

    def check(salt: str, user: str) -> dict[str, Any] | None:
        if salt == "detail_only:0:0":
            return {**verdict(salt, user), "answerable_from_summary": True}
        if salt == "unanswerable:1:0":
            return {"answerable": True, "evidence": "It is in the summary."}
        return None

    run = generator(FakeLLM(write=write, check=check))
    queries = run.generate()
    people, conversations = corpus()
    assert qg.validate_queries(queries, people, conversations, COUNTS) == []
    assert run.rejections["temporal"] == {"time_phrase_missing": 1}
    assert run.rejections["cross_person"] == {"person_named": 1, "duplicate_question": 1}
    assert run.rejections["paraphrase"] == {"reuses_summary_wording": 1}
    assert run.rejections["lookup"] == {"writer_declined": 1, "writer_malformed": 1}
    assert run.rejections["unknown_name"] == {"name_missing": 1}
    assert run.rejections["detail_only"] == {"answerable_from_summary": 1}
    assert run.rejections["unanswerable"] == {"answerable": 1}
    for qtype, index in (("temporal", 0), ("cross_person", 0), ("cross_person", 2), ("paraphrase", 0), ("lookup", 0),
                         ("lookup", 2), ("unknown_name", 0), ("detail_only", 0), ("unanswerable", 1)):
        assert slot(queries, qtype, index).meta["attempt"] == 1
    assert sum(query.meta["attempt"] for query in queries) == 9


def test_a_type_that_cannot_be_filled_fails_loudly():
    def reject(salt: str, user: str) -> dict[str, Any] | None:
        return {"answerable": False, "answer_agrees": False, "contributing": []}

    with pytest.raises(qg.QueryGenerationError) as error:
        generator(FakeLLM(check=reject), counts={"lookup": 2}).generate()
    assert "lookup: filled 0 of 2" in str(error.value) and "'not_answerable_from_summaries': 10" in str(error.value)

    # An "unknown" name that the roster resolves is rejected before any model call.
    llm = FakeLLM()
    with pytest.raises(qg.QueryGenerationError) as error:
        qg.QueryGenerator(llm, SETTINGS, *corpus(), ["Hannah Nobody"], {"unknown_name": 2}).generate()
    assert "the budget of 10 attempts is used up" in str(error.value) and "'unknown_name_resolves': 10" in str(error.value)
    assert llm.calls == []
    with pytest.raises(qg.QueryGenerationError, match="unknown_name: filled 0 of 2 queries, then no target is left to draw"):
        qg.QueryGenerator(FakeLLM(), SETTINGS, *corpus(), [], {"unknown_name": 2}).generate()
    with pytest.raises(ValueError, match="unknown query types"):
        generator(counts={"riddle": 1})


def test_source_reuse_limit():
    people, conversations = corpus()
    # Enough single-conversation queries that the light people's 26 conversations must be reused.
    counts = {"lookup": 24, "paraphrase": 24, "detail_only": 24, "cross_person": 10}
    queries = generator(counts=counts).generate()
    uses = Counter(e for query in queries for e in query.meta["source_episode_ids"])
    assert max(uses.values()) == qg.MAX_SOURCE_USES
    for qtype in counts:
        within = [e for query in of_type(queries, qtype) for e in query.meta["source_episode_ids"]]
        assert len(within) == len(set(within))
    heavy_ids = {person.person_id for person in people if person.is_heavy}
    by_id = {conversation.episode_id: conversation for conversation in conversations}
    for qtype in ("lookup", "paraphrase", "detail_only"):
        heavy = sum(by_id[query.gold_episode_ids[0]].person_id in heavy_ids for query in of_type(queries, qtype))
        assert heavy == 12  # half from heavy people

    state = qg._State()
    target = qg.Target("lookup", 0, 0, sources=(conversations[0],))
    assert state.free(conversations[0], "lookup")
    state.reserve(target)
    assert not state.free(conversations[0], "lookup") and state.free(conversations[0], "paraphrase")
    state.reserve(qg.Target("paraphrase", 0, 0, sources=(conversations[0],)))
    assert not state.free(conversations[0], "detail_only")
    state.settle(target, None, accepted=False)  # a rejected candidate frees the source for other types only
    assert state.free(conversations[0], "detail_only") and not state.free(conversations[0], "lookup")


def test_temporal_subkinds_and_pool_sizes():
    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    temporal = of_type(baseline(), "temporal")
    assert sum(query.gold_name is not None for query in temporal) == 7
    kinds = Counter((query.gold_name is not None, query.meta["subkind"]) for query in temporal)
    assert kinds == {(True, "recap"): 3, (True, "specific"): 4, (False, "recap"): 1, (False, "specific"): 2}

    for query in temporal:
        expression = index.expressions[query.meta["time_expression"]]
        assert query.gold_time_range == expression.time_range
        assert expression.phrase.lower() in query.query.lower()
        scope = [c for c in conversations if query.gold_person_id in (None, c.person_id)]
        ordered = sorted(scope, key=lambda c: (c.start_time, c.episode_id))
        inside = [c.episode_id for c in ordered if expression.time_range.contains(c.start_time)]
        if query.meta["subkind"] == "recap":
            assert 1 <= len(inside) <= 3 and query.gold_episode_ids == inside
        else:
            assert len(inside) > 3 and 1 <= len(query.gold_episode_ids) <= 2 and set(query.gold_episode_ids) <= set(inside)
        padded = pad_time_range(expression.time_range)
        assert query.meta["pool_size"] == sum(padded.contains(c.start_time) for c in scope)
    buckets = Counter(pool_bucket(query.meta["pool_size"]) for query in temporal)
    assert buckets["21-100"] >= 3 and buckets["<=20"] >= 2  # 12 and 8 of 40, scaled to 10
    mid = [query for query in temporal if pool_bucket(query.meta["pool_size"]) == "21-100"]
    assert sum(query.gold_name is None for query in mid) == 2 and all(q.meta["subkind"] == "specific" for q in mid)

    full = Counter((p.named, p.subkind, p.pool_bucket) for p in qg.build_plans(index, config.QUERY_TYPE_COUNTS)["temporal"])
    assert full == {
        (True, "recap", "<=20"): 8, (True, "recap", None): 2, (True, "specific", "21-100"): 2,
        (True, "specific", None): 13, (False, "recap", None): 5, (False, "specific", "21-100"): 10,
    }


def test_temporal_plan_adapts_to_the_corpus():
    # Without a person-scoped window of 21-100 conversations, no-name recap slots turn specific.
    sparse = qg.Corpus(*corpus(dense=False))
    plans = Counter((p.named, p.subkind, p.pool_bucket) for p in qg.build_plans(sparse, COUNTS)["temporal"])
    assert plans == {
        (True, "recap", "<=20"): 2, (True, "recap", None): 1, (True, "specific", None): 4,
        (False, "specific", "21-100"): 3,
    }
    queries = generator(counts={"temporal": 10}, dense=False).generate()
    assert sum(pool_bucket(query.meta["pool_size"]) == "21-100" for query in queries) == 3

    people, conversations = corpus()
    light = [person for person in people if not person.is_heavy]
    theirs = [c for c in conversations if c.person_id in {person.person_id for person in light}]
    with pytest.raises(qg.QueryGenerationError, match="pool of 21-100"):
        qg.build_plans(qg.Corpus(light, theirs), COUNTS)


def test_draws_leave_the_plan_only_when_the_corpus_runs_out():
    run = generator()
    run.generate()
    assert sum(run.off_plan.values()) == 0

    # Light people only and no shared topics: heavy slots, cross-person groups and "specific"
    # time windows have nothing to draw, so those slots take what the corpus does have.
    people, conversations = corpus()
    light = [person for person in people if not person.is_heavy]
    theirs = [
        Conversation.from_json({**conversation.to_json(), "meta": {**conversation.meta, "shared_topic": None}})
        for conversation in conversations
        if conversation.person_id in {person.person_id for person in light}
    ]
    counts = {"lookup": 4, "multi_conversation": 3, "cross_person": 5, "temporal": 3, "unanswerable": 2}
    run = qg.QueryGenerator(FakeLLM(), SETTINGS, light, theirs, UNKNOWN_NAMES, counts)
    queries = run.generate()
    assert qg.validate_queries(queries, light, theirs, counts) == []
    assert run.off_plan == {"lookup": 2, "multi_conversation": 2, "cross_person": 2, "temporal": 1, "unanswerable": 1}
    assert {query.meta["subkind"] for query in of_type(queries, "cross_person")} == {"single"}
    assert {query.meta["subkind"] for query in of_type(queries, "temporal")} == {"recap"}


def test_unanswerable_questions_are_written_against_the_earlier_ones():
    llm = FakeLLM()
    queries = generator(llm, counts={"unanswerable": 12}).generate()
    heavy_ids = {person.person_id for person in corpus()[0] if person.is_heavy}
    per_person = Counter(query.gold_person_id for query in queries)
    assert sorted(per_person[person_id] for person_id in heavy_ids) == [3, 3]
    assert all(count == 1 for person_id, count in per_person.items() if person_id not in heavy_ids)

    # Two questions about one person are never written in the same round: each prompt
    # lists every question about that person that exists by then.
    listed: dict[str, list[int]] = defaultdict(list)
    for index, query in enumerate(queries):
        prompt = llm.prompt(qg.WRITER_NAMESPACE, f"unanswerable:{index}:0")
        asked = set(re.findall(r"^- (.+\?)$", prompt, re.MULTILINE))
        assert asked <= {other.query for other in queries if other.gold_person_id == query.gold_person_id} - {query.query}
        listed[query.gold_person_id].append(len(asked))
    assert all(sorted(sizes) == list(range(len(sizes))) for sizes in listed.values())

    threads = [(q.gold_person_id, q.meta["storyline_id"]) for q in queries if q.meta["subkind"] == "missing_detail"]
    assert len(threads) == len(set(threads)) == 6
    assert all("storyline_id" not in q.meta for q in queries if q.meta["subkind"] == "absent_topic")


# --------------------------------------------------------------------------- #
# Split, determinism, regeneration                                             #
# --------------------------------------------------------------------------- #
def test_split_is_seeded_and_stratified():
    allocation = qg.dev_allocation(config.QUERY_TYPE_COUNTS)
    assert sum(allocation.values()) == 30 == round(config.DEV_FRACTION * config.N_QUERIES)
    for qtype, count in config.QUERY_TYPE_COUNTS.items():
        assert allocation[qtype] >= 1 and abs(allocation[qtype] - config.DEV_FRACTION * count) < 1
    slots = qg.dev_slots(config.QUERY_TYPE_COUNTS)
    assert slots == qg.dev_slots(config.QUERY_TYPE_COUNTS)
    assert {qtype: len(indexes) for qtype, indexes in slots.items()} == allocation
    assert all(max(indexes) < config.QUERY_TYPE_COUNTS[qtype] for qtype, indexes in slots.items())
    assert qg.dev_allocation({"lookup": 100, "unknown_name": 4, "temporal": 6}) == {
        **dict.fromkeys(QUERY_TYPES, 0), "lookup": 9, "unknown_name": 1, "temporal": 1,
    }

    queries = baseline()
    small = qg.dev_allocation(COUNTS)
    assert sum(small.values()) == 4  # 3.5 rounds half up
    assert Counter(query.type for query in queries if query.split == "dev") == {t: n for t, n in small.items() if n}
    assert {query.split for query in queries} == {"dev", "test"}
    for qtype, indexes in qg.dev_slots(COUNTS).items():
        assert {i for i, query in enumerate(of_type(queries, qtype)) if query.split == "dev"} == indexes

    assert qg.scaled_counts(config.N_QUERIES) == config.QUERY_TYPE_COUNTS
    assert sum(qg.scaled_counts(35).values()) == 35 and min(qg.scaled_counts(35).values()) >= 2


def test_runs_are_deterministic():
    serial = [FakeLLM(), FakeLLM()]
    first, second = (generator(llm, workers=1).generate() for llm in serial)
    assert dumps(first) == dumps(second) == dumps(baseline())
    assert serial[0].calls == serial[1].calls  # the same prompts, in the same order

    threaded = FakeLLM()
    assert dumps(generator(threaded, workers=8).generate()) == dumps(first)
    assert sorted(threaded.calls) == sorted(serial[0].calls)


def test_regenerate_changes_only_the_listed_queries():
    people, conversations = corpus()
    original = copies(baseline())
    targets = [of_type(original, "unanswerable")[0], of_type(original, "lookup")[1], of_type(original, "temporal")[2],
               of_type(original, "unknown_name")[0]]
    listed = [query.query_id for query in targets]
    llm = FakeLLM()
    updated = generator(llm).regenerate(copies(original), listed)

    assert [query.query_id for query in updated] == [query.query_id for query in original]
    assert qg.validate_queries(updated, people, conversations, COUNTS) == []
    for old, new in zip(original, updated):
        if old.query_id not in listed:
            assert json.dumps(new.to_json(), sort_keys=True) == json.dumps(old.to_json(), sort_keys=True)
            continue
        assert (new.type, new.split, new.gold_person_id is None) == (old.type, old.split, old.gold_person_id is None)
        assert new.query != old.query and new.meta["attempt"] == old.meta["attempt"] + 1
        assert new.meta["subkind"] == old.meta["subkind"]
        if old.type in ANSWERABLE_TYPES:
            assert not set(new.meta["source_episode_ids"]) & set(old.meta["source_episode_ids"])
    # Only the listed slots cost model calls, and the unanswerable writer is told what was asked before.
    assert sorted({call[1] for call in llm.calls}) == sorted(
        f"{query.type}:{of_type(original, query.type).index(query)}:1" for query in targets
    )
    person_id = updated[original.index(targets[0])].gold_person_id
    earlier = [query.query for query in of_type(original, "unanswerable") if query.gold_person_id == person_id]
    assert earlier and all(f"- {question}" in llm.prompt(qg.WRITER_NAMESPACE, "unanswerable:0:1") for question in earlier)
    uses = Counter(e for query in updated for e in query.meta["source_episode_ids"])
    assert max(uses.values()) <= qg.MAX_SOURCE_USES

    assert dumps(generator().regenerate(copies(original), listed)) == dumps(updated)
    with pytest.raises(ValueError, match="not in the query file"):
        generator().regenerate(copies(original), ["no-such-id"])
    with pytest.raises(ValueError, match="not a slot"):
        generator(counts={"lookup": 2}).regenerate(copies(original), [])


# --------------------------------------------------------------------------- #
# validate_queries                                                             #
# --------------------------------------------------------------------------- #
def _other_person_episode(query: Query) -> str:
    return next(c.episode_id for c in corpus()[1] if c.person_id != query.gold_person_id)


def _drop_one(queries: list[Query]) -> None:
    queries.remove(of_type(queries, "lookup")[0])


def _move_to_dev(queries: list[Query]) -> None:
    next(query for query in queries if query.split == "test").split = "dev"


def _repeat_id(queries: list[Query]) -> None:
    queries[1].query_id = queries[0].query_id


def _repeat_text(queries: list[Query]) -> None:
    queries[1].query = queries[0].query.upper()


def _bogus_gold(queries: list[Query]) -> None:
    of_type(queries, "lookup")[0].gold_episode_ids.append("no-such-episode")


def _gold_of_another_person(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.gold_episode_ids = [_other_person_episode(query)]


def _name_of_another_person(queries: list[Query]) -> None:
    query = next(query for query in of_type(queries, "lookup") if query.gold_name != "Diego")
    query.query = query.query.replace(query.gold_name, "Diego")
    query.gold_name = "Diego"


def _ambiguous_name(queries: list[Query]) -> None:
    query = next(query for query in queries if query.gold_name == "Sam Okafor")
    query.query = query.query.replace("Sam Okafor", "Sam")
    query.gold_name = "Sam"


def _name_not_in_text(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.query = query.query.replace(query.gold_name, "they")


def _named_cross_person(queries: list[Query]) -> None:
    of_type(queries, "cross_person")[0].gold_name = "Hannah"


def _lookup_without_name(queries: list[Query]) -> None:
    of_type(queries, "lookup")[0].gold_name = None


def _gold_outside_range(queries: list[Query]) -> None:
    query = of_type(queries, "temporal")[0]
    query.gold_time_range = TimeRange(query.gold_time_range.start - timedelta(days=900), query.gold_time_range.start - timedelta(days=800))


def _no_time_phrase(queries: list[Query]) -> None:
    query = of_type(queries, "temporal")[0]
    query.query = "What did we talk about?" if query.gold_name is None else f"What did {query.gold_name} say?"


def _temporal_without_range(queries: list[Query]) -> None:
    of_type(queries, "temporal")[0].gold_time_range = None


def _range_on_a_lookup(queries: list[Query]) -> None:
    of_type(queries, "lookup")[0].gold_time_range = of_type(queries, "temporal")[0].gold_time_range


def _unanswerable_with_gold(queries: list[Query]) -> None:
    query = of_type(queries, "unanswerable")[0]
    query.gold_episode_ids = [next(c.episode_id for c in corpus()[1] if c.person_id == query.gold_person_id)]


def _unanswerable_with_answer(queries: list[Query]) -> None:
    of_type(queries, "unanswerable")[0].gold_answer = "Yes."


def _unanswerable_nobody(queries: list[Query]) -> None:
    query = of_type(queries, "unanswerable")[0]
    query.query = query.query.replace(query.gold_name, "Priya")
    query.gold_name = "Priya"


def _unknown_name_that_resolves(queries: list[Query]) -> None:
    query = of_type(queries, "unknown_name")[0]
    query.query = query.query.replace(query.gold_name, "Hannah")
    query.gold_name = "Hannah"


def _answerable_without_gold(queries: list[Query]) -> None:
    of_type(queries, "paraphrase")[0].gold_episode_ids = []


def _answerable_without_answer(queries: list[Query]) -> None:
    of_type(queries, "multi_conversation")[0].gold_answer = None


def _stray_time_word(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.query = query.query[:-1] + " yesterday?"


def _owner_named(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.query = query.query[:-1] + " to Riley?"


def _paraphrase_in_the_summarys_words(queries: list[Query]) -> None:
    query = of_type(queries, "paraphrase")[0]
    query.query = f"Which booking did {query.gold_name} confirm for the plan?"


def _yes_no_question(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.query = f"Did {query.gold_name} confirm the booking?"


def _question_carries_its_answer(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.gold_answer = f"It was {query.query.split('regarding ')[1].rstrip('?')}."


def _possessive_time_phrase(queries: list[Query]) -> None:
    query = of_type(queries, "temporal")[0]
    phrase = qg.Corpus(*corpus()).expressions[query.meta["time_expression"]].phrase
    query.query = query.query.replace(phrase, f"{phrase}'s plan")


def _relabel(query: Query, episode_ids: list[str]) -> None:
    """Edit both label sets by hand, as a file edited outside the generator and pooling would be."""

    query.gold_episode_ids = list(episode_ids)
    query.meta["source_episode_ids"] = list(episode_ids)


def _recap_missing_one(queries: list[Query]) -> None:
    query = next(q for q in of_type(queries, "temporal") if q.meta["subkind"] == "recap" and len(q.gold_episode_ids) > 1)
    _relabel(query, query.gold_episode_ids[:-1])


def _multi_cut_to_one(queries: list[Query]) -> None:
    query = of_type(queries, "multi_conversation")[0]
    _relabel(query, query.gold_episode_ids[:1])


def _multi_skips_a_beat(queries: list[Query]) -> None:
    by_id = {c.episode_id: c for c in corpus()[1]}
    for query in of_type(queries, "multi_conversation"):
        beats = qg.Corpus(*corpus()).storylines[qg.storyline_key(by_id[query.gold_episode_ids[0]])]
        if len(beats) >= 3:
            _relabel(query, [beats[0].episode_id, beats[2].episode_id])
            return
    raise AssertionError("no multi-conversation query on a storyline of three beats or more")


def _group_with_one_person_twice(queries: list[Query]) -> None:
    query = next(q for q in of_type(queries, "cross_person") if q.meta["subkind"] == "group")
    first = next(c for c in corpus()[1] if c.episode_id == query.gold_episode_ids[0])
    twin = next(c for c in corpus()[1] if c.person_id == first.person_id and c is not first)
    _relabel(query, [first.episode_id, twin.episode_id, *query.gold_episode_ids[2:]])


def _single_with_two_sources(queries: list[Query]) -> None:
    query = next(q for q in of_type(queries, "cross_person") if q.meta["subkind"] == "single")
    _relabel(query, [*query.gold_episode_ids, _other_person_episode(query)])


def _specific_with_two_sources_apart(queries: list[Query]) -> None:
    index = qg.Corpus(*corpus())
    for query in of_type(queries, "temporal"):
        if query.meta["subkind"] != "specific" or len(query.gold_episode_ids) != 1:
            continue
        source = index.conversation[query.gold_episode_ids[0]]
        beats = index.storylines[qg.storyline_key(source)]
        apart = [b for b in beats if abs(beats.index(b) - beats.index(source)) >= 2 and query.gold_time_range.contains(b.start_time)]
        if apart:
            _relabel(query, [source.episode_id, apart[0].episode_id])
            return
    raise AssertionError("no specific temporal query with a non-adjacent beat of its storyline inside the range")


def _source_dropped_from_gold(queries: list[Query]) -> None:
    query = of_type(queries, "multi_conversation")[0]
    query.gold_episode_ids = query.gold_episode_ids[1:]


def _gold_grown_without_a_pooling_record(queries: list[Query]) -> None:
    query = next(q for q in of_type(queries, "cross_person") if q.meta["subkind"] == "single")
    query.gold_episode_ids.append(_other_person_episode(query))


def _pooled_label_dropped_from_gold(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.meta["pooling"] = {"depth": 30, "added": [_other_person_episode(query)]}


def _source_not_in_corpus(queries: list[Query]) -> None:
    query = of_type(queries, "lookup")[0]
    query.meta["source_episode_ids"] = ["no-such-episode"]
    query.meta["pooling"] = {"depth": 30, "added": list(query.gold_episode_ids)}


def _wrong_pool_size(queries: list[Query]) -> None:
    of_type(queries, "temporal")[0].meta["pool_size"] = 999


def _swapped_splits(queries: list[Query]) -> None:
    dev = next(q for q in queries if q.split == "dev")
    test = next(q for q in queries if q.split == "test" and q.type == dev.type)
    dev.split, test.split = "test", "dev"


def _foreign_id(queries: list[Query]) -> None:
    of_type(queries, "lookup")[0].query_id = config.stable_id("query", "lookup:99")


DEFECTS: list[tuple[Callable[[list[Query]], None], str]] = [
    (_drop_one, "lookup: 3 queries, expected 4"),
    (_move_to_dev, "dev queries, expected"),
    (_repeat_id, "duplicate query id"),
    (_repeat_text, "duplicate query text"),
    (_bogus_gold, "gold episode no-such-episode is not in the corpus"),
    (_gold_of_another_person, "a gold conversation belongs to another person"),
    (_name_of_another_person, "does not resolve to gold_person_id"),
    (_ambiguous_name, "does not resolve to gold_person_id"),
    (_name_not_in_text, "the question fails the 'name_missing' rule"),
    (_named_cross_person, "a no-name query must have gold_name and gold_person_id of None"),
    (_lookup_without_name, "a named query needs gold_name and gold_person_id"),
    (_gold_outside_range, "a gold conversation starts outside the gold_time_range"),
    (_no_time_phrase, "the question fails the 'time_phrase_missing' rule"),
    (_temporal_without_range, "a temporal query needs a gold_time_range"),
    (_range_on_a_lookup, "only temporal queries may have a gold_time_range"),
    (_unanswerable_with_gold, "must have no gold episode ids"),
    (_unanswerable_with_answer, "must have no gold answer"),
    (_unanswerable_nobody, "does not resolve to gold_person_id"),
    (_unknown_name_that_resolves, "resolves to someone in the roster"),
    (_answerable_without_gold, "answerable but has no gold episode ids"),
    (_answerable_without_answer, "answerable but has no gold answer"),
    (_stray_time_word, "the question fails the 'time_expression' rule"),
    (_owner_named, "the question fails the 'owner_named' rule"),
    (_paraphrase_in_the_summarys_words, "the question fails the 'reuses_summary_wording' rule"),
    (_yes_no_question, "the question fails the 'yes_no_question' rule"),
    (_question_carries_its_answer, "the question fails the 'question_contains_answer' rule"),
    (_possessive_time_phrase, "the question fails the 'time_phrase_dates_a_thing' rule"),
    (_recap_missing_one, "a recap's gold episode ids are not every conversation of its scope inside the gold_time_range"),
    (_multi_cut_to_one, "the sources are not 2 or 3 consecutive beats of one storyline"),
    (_multi_skips_a_beat, "the sources are not 2 or 3 consecutive beats of one storyline"),
    (_group_with_one_person_twice, "a group query's sources must be with pairwise different people"),
    (_single_with_two_sources, "a single who-question is written from exactly one source conversation"),
    (_specific_with_two_sources_apart, "a specific temporal query's sources are one conversation or two consecutive beats"),
    (_source_dropped_from_gold, "a source conversation is not gold"),
    (_gold_grown_without_a_pooling_record, "the gold conversations beyond the sources are not the ones pooling added"),
    (_pooled_label_dropped_from_gold, "the gold conversations beyond the sources are not the ones pooling added"),
    (_source_not_in_corpus, "source episode no-such-episode is not in the corpus"),
    (_wrong_pool_size, "meta.pool_size is 999, the padded range holds"),
    (_swapped_splits, "the seeded split puts this slot in"),
    (_foreign_id, "the query id is not a slot of these per-type counts"),
]


@pytest.mark.parametrize("defect, expected", DEFECTS, ids=[defect.__name__.lstrip("_") for defect, _ in DEFECTS])
def test_validate_queries_catches_each_defect(defect, expected):
    people, conversations = corpus()
    queries = copies(baseline())
    assert qg.validate_queries(queries, people, conversations, COUNTS) == []
    defect(queries)
    problems = qg.validate_queries(queries, people, conversations, COUNTS)
    assert any(expected in problem for problem in problems), problems


def test_validate_queries_defaults_to_the_benchmark_counts():
    people, conversations = corpus()
    problems = qg.validate_queries(copies(baseline()), people, conversations)
    assert "lookup: 4 queries, expected 50" in problems and "split: 4 dev queries, expected 30" in problems


def pooled(query: Query, extra_ids: list[str]) -> Query:
    """The record ``pool_labels`` writes once the judge confirms ``extra_ids`` for the query."""

    from eval.retrieval.data import pool_labels

    by_id = {c.episode_id: c for c in corpus()[1]}
    pool = pool_labels.Pool(pool_size=50, candidates=tuple(extra_ids))
    grown = pool_labels._outcome(
        query, pool, list(extra_ids), pool_labels.Usage(), 30, config.RANKER_NAMES, by_id
    ).query
    grown.gold_answer = f"{grown.gold_answer} Rewritten over every gold conversation."
    return grown


def test_a_pooled_query_file_is_valid_and_keeps_the_rules_that_still_hold():
    """Pooling grows gold sets beyond the draw's shape; the draw's rules hold for the sources, the rest for every gold id."""

    people, conversations = corpus()
    index = qg.Corpus(people, conversations)
    queries = copies(baseline())
    grown: dict[str, list[str]] = {}

    def grow(query: Query, extra: list[str]) -> None:
        queries[queries.index(query)] = pooled(query, extra)
        grown[query.query_id] = extra

    single = next(q for q in of_type(queries, "cross_person") if q.meta["subkind"] == "single")
    source = index.conversation[single.gold_episode_ids[0]]
    grow(single, [next(c.episode_id for c in index.by_person[source.person_id] if c is not source)])
    multi = next(q for q in of_type(queries, "multi_conversation") if len(q.gold_episode_ids) == 3)
    first = index.conversation[multi.gold_episode_ids[0]]
    grow(multi, [next(c.episode_id for c in index.by_person[first.person_id] if qg.storyline_key(c) != qg.storyline_key(first))])
    specific = next(q for q in of_type(queries, "temporal") if q.meta["subkind"] == "specific" and len(q.gold_episode_ids) == 1)
    scope = index.conversations if specific.gold_person_id is None else index.by_person[specific.gold_person_id]
    inside = [c for c in scope if specific.gold_time_range.contains(c.start_time) and c.episode_id not in specific.gold_episode_ids]
    grow(specific, [inside[-1].episode_id])
    paraphrase = of_type(queries, "paraphrase")[0]
    first = index.conversation[paraphrase.gold_episode_ids[0]]
    earlier = [c for c in index.by_person[first.person_id] if c.start_time < first.start_time]
    grow(paraphrase, [earlier[0].episode_id])  # oldest first: the source is no longer gold[0]
    group = next(q for q in of_type(queries, "cross_person") if q.meta["subkind"] == "group")
    member = index.conversation[group.gold_episode_ids[0]]
    grow(group, [next(c.episode_id for c in index.by_person[member.person_id] if c is not member)])

    assert len(grown) == 5
    for query in queries:
        if query.query_id in grown:
            assert set(query.gold_episode_ids) == set(query.meta["source_episode_ids"]) | set(grown[query.query_id])
            assert query.meta["pooling"]["added"] == grown[query.query_id]
    assert qg.validate_queries(queries, people, conversations, COUNTS) == []

    # The rules over every gold conversation still apply to a pooled label.
    other = copies(queries)
    pooled_single = next(q for q in other if q.query_id == single.query_id)
    [added] = pooled_single.meta["pooling"]["added"]
    pooled_single.gold_episode_ids[pooled_single.gold_episode_ids.index(added)] = "no-such-episode"
    pooled_single.meta["pooling"]["added"] = ["no-such-episode"]
    problems = qg.validate_queries(other, people, conversations, COUNTS)
    assert problems == [f"{single.query_id} (cross_person): gold episode no-such-episode is not in the corpus"]
    other = copies(queries)
    pooled_specific = next(q for q in other if q.query_id == specific.query_id)
    outside = next(c for c in scope if not specific.gold_time_range.contains(c.start_time))
    pooled_specific.gold_episode_ids.append(outside.episode_id)
    pooled_specific.meta["pooling"]["added"].append(outside.episode_id)
    problems = qg.validate_queries(other, people, conversations, COUNTS)
    assert problems == [f"{specific.query_id} (temporal): a gold conversation starts outside the gold_time_range"]

    # Regenerating other queries of a pooled file keeps the pooled ones as they are.
    replaced = [of_type(queries, "unanswerable")[0].query_id, of_type(queries, "lookup")[1].query_id]
    updated = generator().regenerate(copies(queries), replaced)
    assert qg.validate_queries(updated, people, conversations, COUNTS) == []
    assert dumps(q for q in updated if q.query_id in grown) == dumps(q for q in queries if q.query_id in grown)
    assert all(q.meta["attempt"] == 1 and "pooling" not in q.meta for q in updated if q.query_id in replaced)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def test_cli_generates_regenerates_and_reports(tmp_path, monkeypatch, capsys):
    people, conversations = corpus()
    dataset.save_people(people, tmp_path / "people.jsonl")
    dataset.save_conversations(conversations, tmp_path / "conversations.jsonl")
    out = tmp_path / "out" / "queries.jsonl"
    llm = FakeLLM()
    monkeypatch.setattr("eval.retrieval.runtime.Runtime", lambda: types.SimpleNamespace(llm=llm, settings=SETTINGS))
    roster = types.ModuleType("eval.retrieval.data.roster")
    roster.UNKNOWN_NAMES = tuple(UNKNOWN_NAMES)
    monkeypatch.setitem(sys.modules, "eval.retrieval.data.roster", roster)
    base = ["--corpus-dir", str(tmp_path), "--out", str(out), "--n-queries", "35"]

    assert qg.main([*base, "--workers", "2"]) == 0
    printed = capsys.readouterr().out
    queries = dataset.load_queries(path=out)
    counts = qg.scaled_counts(35)
    assert Counter(query.type for query in queries) == counts
    assert qg.validate_queries(queries, people, conversations, counts) == []
    assert len(dataset.load_queries("dev", path=out)) == 4
    hosted = len(llm.calls)
    assert f"LLM calls: {hosted} hosted, 0 from the cache; hosted spend: ${hosted * 0.001:.4f}" in printed
    assert re.search(r"^lookup\s+6\s+\d\s+\d$", printed, re.MULTILINE) and re.search(r"^total\s+35\s+4\s+31$", printed, re.MULTILINE)

    # Pooling has grown one who-question's gold set in the meantime (the command it prints
    # for a flagged unanswerable query must still replace that query and keep this one).
    single = next(q for q in queries if q.type == "cross_person" and q.meta["subkind"] == "single")
    source = next(c for c in conversations if c.episode_id == single.gold_episode_ids[0])
    extra = next(c for c in conversations if c.person_id == source.person_id and c is not source)
    dataset.save_queries([pooled(q, [extra.episode_id]) if q is single else q for q in queries], out)
    before = out.read_text(encoding="utf-8").splitlines()
    assert len(json.loads(before[queries.index(single)])["gold_episode_ids"]) == 2
    replaced = of_type(queries, "unanswerable")[1].query_id
    assert qg.main([*base, "--regenerate", replaced]) == 0
    after = out.read_text(encoding="utf-8").splitlines()
    changed = [index for index, (old, new) in enumerate(zip(before, after, strict=True)) if old != new]
    assert [json.loads(after[index])["query_id"] for index in changed] == [replaced]

    # The small corpus cannot carry the full benchmark: the command fails and leaves the file alone.
    capsys.readouterr()
    assert qg.main(["--corpus-dir", str(tmp_path), "--out", str(out)]) == 1
    assert "error: " in capsys.readouterr().err
    assert out.read_text(encoding="utf-8").splitlines() == after
