"""The corpus generator: roster, schedule, prompts, and an offline end-to-end run."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone

import pytest

from eval.retrieval import config, dataset
from eval.retrieval.data import corpus_prompts as prompts
from eval.retrieval.data import generate, plan
from eval.retrieval.data.roster import (
    OWNER_BIO,
    SETTINGS,
    SHARED_NAME_RANGE,
    SHARED_TOPICS,
    UNKNOWN_NAMES,
    RosterEntry,
    build_roster,
    seeded,
)
from eval.retrieval.interfaces import ChatLLM, LLMResult, Usage
from eval.retrieval.schema import OWNER_SPEAKER, Conversation, Turn

GEN_MODEL = "fake/gen"
SUMMARY_MODEL = "fake/summary"
SETTINGS_FOR_TESTS = config.Settings(
    database_url=None,
    gateway="openrouter",
    llm_base_url="http://llm.invalid/v1",
    llm_api_key=None,
    gen_model=GEN_MODEL,
    summary_model=SUMMARY_MODEL,
    system_model="fake/system",
    judge_model="fake/judge",
    jev_url="http://jev.invalid",
    jev_api_key=None,
    jev_model="fake/jev",
    llm_concurrency=4,
    jev_concurrency=1,
    device=None,
    max_spend_usd=1.0,
)
KEY_FACTS = re.compile(r"Key facts:\n(.*?)\nMinor details:", re.S)
MINOR_DETAILS = re.compile(r"Minor details:\n(.*?)\n\nWrite the conversation", re.S)
MINOR_TOKEN = re.compile(r"MINOR\[[^\]]+\]")
JSONL_NAMES = ["conversations.jsonl", "people.jsonl"]
FACTS_PER_BEAT = 3  # what the fake returns; a short conversation keeps two of them


# --------------------------------------------------------------------------- #
# A deterministic model                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Call:
    namespace: str
    model: str
    system: str
    user: str
    salt: str
    temperature: float | None
    reasoning_effort: str | None


class FakeLLM:
    """Fabricates a valid reply for each kind of call from the prompt alone; records every call.

    ``faults`` makes the first attempts of a kind go wrong, e.g. ``summary_long=1``
    returns a 150-word summary until the first re-ask. A reply depends only on
    the prompt and the salt, never on call order, so threads cannot change it.
    """

    def __init__(self, **faults: int) -> None:
        self.faults = faults
        self.calls: list[Call] = []
        self._lock = threading.Lock()

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
        call = Call(namespace, model, system, user, salt, temperature, reasoning_effort)
        with self._lock:
            self.calls.append(call)
        attempt = int(salt.rsplit("-", 1)[1]) if salt else 0
        data = getattr(self, f"_{namespace}")(user, attempt)
        usage = Usage(input_tokens=len(user) // 4, output_tokens=40, cost_usd=0.001, calls=1)
        return LLMResult(data=data, usage=usage, model=model, raw_text=json.dumps(data))

    def of(self, namespace: str) -> list[Call]:
        return [call for call in self.calls if call.namespace == namespace]

    def _persona(self, user: str, attempt: int) -> dict:
        """A title and premise for every storyline row the prompt lists, in reverse order."""

        name = re.search(r"Person: (.+)", user).group(1)
        rows = re.findall(r"^Storyline (\d+): \d+ conversations?", user, re.M)
        reserved = re.search(r'recurring element: "([^"]+)"', user).group(1)
        persona = {field: f"{field} of {name}" for field in prompts.PERSONA_FIELDS}
        if attempt < self.faults.get("outline_bad", 0):
            del persona["age"]
        storylines = []
        for number in map(int, reversed(rows)):
            premise = f"PREMISE[{name}#{number}] unfolds."
            if number == 1 and attempt < self.faults.get("outline_reserved", 0):
                premise += f" It is really about {reserved}."
            storylines.append(
                {"storyline": number, "title": f"{name} thread {number}", "premise": premise}
            )
        return {"persona": persona, "storylines": storylines}

    def _beats(self, user: str, attempt: int) -> dict:
        title = re.search(r'Storyline to write: "(.+)"', user).group(1)
        # A marker that is unique to the storyline without containing its title.
        tag = hashlib.sha1(title.encode()).hexdigest()[:8]
        n = int(re.search(r"Write exactly (\d+) beats?,", user).group(1))
        if attempt < self.faults.get("beats_wrong", 0):
            n = max(1, n + self.faults.get("beats_delta", 1))
        beats = [
            {
                "what_happens": f"WHAT[{tag}#{beat}] happens.",
                "key_facts": [f"FACT[{tag}#{beat}{k}]" for k in "abc"[:FACTS_PER_BEAT]],
                "minor_details": [f"MINOR[{tag}#{beat}{k}]" for k in "abc"[:FACTS_PER_BEAT]],
            }
            for beat in range(n)
        ]
        if n > 1 and attempt < self.faults.get("beats_repeat", 0):
            beats[-1]["key_facts"][0] = beats[0]["key_facts"][0]
        return {"beats": beats}

    def _dialogue(self, user: str, attempt: int) -> dict:
        """Turns that state every key fact and, unless muted, every minor detail."""

        name = re.search(r'\{"turns": \[\{"speaker": "([^"]+)"', user).group(1)
        facts = KEY_FACTS.search(user).group(1).splitlines()
        details = MINOR_DETAILS.search(user).group(1).splitlines()
        if attempt < self.faults.get("dialogue_mute", 0):
            details = []
        owner = "Riley" if attempt < self.faults.get("dialogue_bad", 0) else OWNER_SPEAKER
        lines = [f"Did you hear? {fact[2:]}" for fact in facts]
        lines += [f"By the way, {detail[2:]}" for detail in details if detail != "- None."]
        lines += ["Go on.", "That is all.", "Good to know.", "Anything else?", "No.", "Talk soon."]
        return {
            "turns": [
                {"speaker": owner if number % 2 else name, "text": text}
                for number, text in enumerate(lines[:12])
            ]
        }

    def _summary(self, user: str, attempt: int) -> dict:
        """Thirty words that repeat the first minor detail the transcript happens to contain."""

        words = 150 if attempt < self.faults.get("summary_long", 0) else 30
        summary = ["word"] * words
        if repeated := MINOR_TOKEN.search(user):
            summary[1] = repeated.group(0)
        return {"summary": " ".join(summary)}


def run(out_dir, roster=None, workers: int = 4, **faults: int):
    """Generate the smoke roster (or ``roster``) offline: the fake, the corpus and the roster."""

    roster = generate.smoke_roster() if roster is None else roster
    llm = FakeLLM(**faults)
    corpus = generate.generate_corpus(llm, SETTINGS_FOR_TESTS, roster, out_dir, workers)
    return llm, corpus, roster


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("smoke")


@pytest.fixture(scope="module")
def smoke(smoke_dir):
    return run(smoke_dir)


@pytest.fixture(scope="module")
def full():
    roster = build_roster()
    return roster, plan.build_schedule(roster)


def seconds_into_day(moment: datetime) -> int:
    return moment.hour * 3600 + moment.minute * 60 + moment.second


# --------------------------------------------------------------------------- #
# Roster                                                                       #
# --------------------------------------------------------------------------- #
def test_roster_has_the_configured_shape():
    roster = build_roster()
    counts = [entry.n_conversations for entry in roster]
    low, high = config.LIGHT_PERSON_RANGE

    assert len(roster) == config.N_PEOPLE == 40
    assert sum(counts) == config.N_CONVERSATIONS_TARGET == 800
    assert tuple(counts[:4]) == config.HEAVY_PERSON_COUNTS
    assert [entry.person.display_name for entry in roster[:4]] == [
        "Maya Chen",
        "Sam Okafor",
        "Daniel Reyes",
        "Hannah Blake",
    ]
    assert [entry.person.is_heavy for entry in roster] == [True] * 4 + [False] * 36
    assert all(low <= count <= high for count in counts[4:])
    ids = [entry.person.person_id for entry in roster]
    assert ids == [config.stable_id("person", index) for index in range(40)]
    for entry in roster:
        person = entry.person
        assert person.display_name == f"{person.first_name} {person.last_name}"
        assert person.first_name.isalpha() and person.last_name.isalpha()
        assert person.relationship and entry.setting in SETTINGS and person.meta == {}
    assert build_roster() == roster


def test_exactly_one_first_name_is_shared():
    roster = build_roster()
    first_names = Counter(entry.person.first_name for entry in roster)

    assert {name: count for name, count in first_names.items() if count > 1} == {"Sam": 2}
    assert config.OWNER_FIRST_NAME not in first_names
    other_sam = next(e for e in roster if e.person.first_name == "Sam" and not e.person.is_heavy)
    assert other_sam.person.display_name == "Sam Whitfield"
    assert SHARED_NAME_RANGE[0] <= other_sam.n_conversations <= SHARED_NAME_RANGE[1]
    assert SHARED_NAME_RANGE == (4, 8)


def test_unknown_names_match_nobody():
    roster = build_roster()
    taken_first = {entry.person.first_name for entry in roster} | {config.OWNER_FIRST_NAME}
    taken_last = {entry.person.last_name for entry in roster} | {config.OWNER_LAST_NAME}

    assert len(set(UNKNOWN_NAMES)) == len(UNKNOWN_NAMES) == 20
    firsts = [name.split()[0] for name in UNKNOWN_NAMES]
    assert all(len(name.split()) == 2 for name in UNKNOWN_NAMES)
    assert len(set(firsts)) == 20
    assert not taken_first & set(firsts)
    assert not taken_last & {name.split()[1] for name in UNKNOWN_NAMES}


def test_shared_topics_are_spread_over_several_people():
    roster = build_roster()
    holders = {topic: [e for e in roster if topic in e.shared_topics] for topic in SHARED_TOPICS}

    assert len(set(SHARED_TOPICS)) == 12
    assert all(3 <= len(people) <= 4 for people in holders.values())
    for entry in roster:
        assert set(entry.shared_topics) <= set(SHARED_TOPICS)
        assert len(set(entry.shared_topics)) == len(entry.shared_topics)
        if entry.person.is_heavy:
            assert len(entry.shared_topics) == 3
        else:
            assert len(entry.shared_topics) <= min(2, entry.n_conversations)
    sams = [entry for entry in roster if entry.person.first_name == "Sam"]
    assert set(sams[0].shared_topics) & set(sams[1].shared_topics)
    by_name = {entry.person.display_name: entry for entry in roster}
    assert "rock climbing" in by_name["Andre Baptiste"].shared_topics  # the climbing-gym friend


def test_reduced_roster_keeps_the_rules():
    roster = generate.smoke_roster()

    assert [entry.n_conversations for entry in roster[:2]] == [10, 10]
    assert len(roster) == 6 and sum(entry.n_conversations for entry in roster) == 36
    assert [entry.person.is_heavy for entry in roster] == [True, True, False, False, False, False]
    assert Counter(entry.person.first_name for entry in roster)["Sam"] == 2
    assert roster[0].person == build_roster()[0].person


@pytest.mark.parametrize(
    "arguments",
    [
        {"total": 1000},  # more than 36 light people can hold
        {"total": 600},  # fewer than one conversation each
        {"n_people": 41},
        {"heavy_counts": (150, 148, 146, 142, 100)},
        {"n_people": 3, "heavy_counts": (3, 10), "total": 20},
    ],
)
def test_impossible_rosters_are_rejected(arguments):
    with pytest.raises(ValueError):
        build_roster(**arguments)


# --------------------------------------------------------------------------- #
# Schedule                                                                     #
# --------------------------------------------------------------------------- #
def test_schedule_stays_inside_the_span_and_the_day(full):
    roster, schedule = full
    conversations = schedule.conversations

    assert len(conversations) == 800
    assert [c.start_time for c in conversations] == sorted(c.start_time for c in conversations)
    for planned in conversations:
        length = (planned.end_time - planned.start_time).total_seconds()
        assert config.DATASET_START <= planned.start_time < planned.end_time <= config.NOW
        assert planned.start_time.date() == planned.end_time.date()
        assert time(7, 0) <= planned.start_time.time() and planned.end_time.time() <= time(22, 0)
        assert 3 * 60 <= length <= 25 * 60
    for index, entry in enumerate(roster):
        assert sum(c.person_index == index for c in conversations) == entry.n_conversations


def test_owner_is_never_in_two_conversations(full):
    _, schedule = full
    conversations = schedule.conversations

    for earlier, later in zip(conversations, conversations[1:]):
        assert (later.start_time - earlier.end_time).total_seconds() >= plan.MIN_GAP_S


def test_people_are_met_where_and_when_they_would_be(full):
    roster, schedule = full

    for planned in schedule.conversations:
        entry = roster[planned.person_index]
        weekend = planned.start_time.weekday() >= 5
        first, last = seconds_into_day(planned.start_time), seconds_into_day(planned.end_time)
        if entry.setting == "work":
            assert not weekend and 8 * 3600 <= first and last <= 18.5 * 3600
        if entry.setting == "home" and not weekend:
            assert last <= 9 * 3600 or first >= 17.5 * 3600


def test_recent_coverage_is_guaranteed(full):
    roster, schedule = full
    starts = [planned.start_time for planned in schedule.conversations]
    today = config.NOW.date()

    assert sum(s.date() == today and s.time() < time(12, 0) for s in starts) >= 2
    assert sum(s.date() == today - timedelta(days=1) for s in starts) >= 3
    assert sum(config.NOW - timedelta(days=7) <= s < config.NOW for s in starts) >= 14
    assert plan.coverage_targets(800) == plan.CoverageTargets(2, 3, 14)
    assert plan.coverage_targets(36) == plan.CoverageTargets(1, 1, 1)

    by_person: list[list[datetime]] = [[] for _ in roster]
    for planned in schedule.conversations:
        by_person[planned.person_index].append(planned.start_time)
    assert plan.coverage_problems(roster, by_person) == []
    without_today = [[start for start in own if start.date() != today] for own in by_person]
    problems = plan.coverage_problems(roster, without_today)
    assert any("before 12:00" in problem for problem in problems)


def test_every_heavy_person_appears_in_every_calendar_month(full):
    roster, schedule = full
    months = plan.calendar_months()

    assert months[0] == (2025, 11) and months[-1] == (2026, 11) and len(months) == 13
    for index in range(4):
        starts = [c.start_time for c in schedule.conversations if c.person_index == index]
        assert {(start.year, start.month) for start in starts} == set(months)
        assert plan.guarantees_monthly_coverage(roster[index].n_conversations)
    assert not plan.guarantees_monthly_coverage(10)


def test_beats_are_chronological_within_each_storyline(full):
    roster, schedule = full
    conversations = schedule.conversations

    assert len({planned.episode_id for planned in conversations}) == 800
    for index in range(len(roster)):
        own = [planned for planned in conversations if planned.person_index == index]
        assert [planned.position for planned in own] == list(range(len(own)))
        for planned in own:
            assert planned.episode_id == config.stable_id("episode", f"{index}:{planned.position}")
        for which, shape in enumerate(schedule.storylines[index]):
            beats = [planned.beat_index for planned in own if planned.storyline_index == which]
            assert beats == list(range(shape.n_beats))


def test_storyline_shapes_fit_the_budget_and_the_timeline(full):
    roster, schedule = full

    for index, entry in enumerate(roster):
        shapes = schedule.storylines[index]
        budget = plan.storyline_budget(entry)
        own = [c for c in schedule.conversations if c.person_index == index]
        assert budget.min_storylines <= len(shapes) <= budget.max_storylines
        assert sum(shape.n_beats for shape in shapes) == entry.n_conversations
        assert all(budget.min_beats <= shape.n_beats <= budget.max_beats for shape in shapes)
        # Each shared subject is the subject of exactly one storyline.
        held = [shape.shared_topic for shape in shapes if shape.shared_topic]
        assert sorted(held) == sorted(entry.shared_topics)
        # Numbered in the order they start, and each stretch is where its beats really are.
        assert [shape.first for shape in shapes] == sorted(shape.first for shape in shapes)
        for which, shape in enumerate(shapes):
            starts = [c.start_time for c in own if c.storyline_index == which]
            assert (shape.first, shape.last) == (starts[0], starts[-1])
    # A heavy person gets a spread: short recurring subjects and long-running threads.
    for shapes in schedule.storylines[:4]:
        lengths = [shape.n_beats for shape in shapes]
        assert min(lengths) <= 6 and max(lengths) >= 11


def test_beat_counts_are_drawn_within_the_budget():
    budget = plan.StorylineBudget(2, 5, 2, 9, 0)

    for label in ("a", "b", "c"):
        counts = plan.draw_beat_counts(seeded(label), 20, budget)
        assert sum(counts) == 20 and 2 <= len(counts) <= 5
        assert all(2 <= count <= 9 for count in counts)
    assert plan.draw_beat_counts(seeded("a"), 20, budget) == plan.draw_beat_counts(
        seeded("a"), 20, budget
    )
    with pytest.raises(ValueError):
        plan.draw_beat_counts(seeded("a"), 50, budget)


def test_storylines_interleave_over_their_own_stretches(full):
    _, schedule = full

    for index in range(4):
        order = [c.storyline_index for c in schedule.conversations if c.person_index == index]
        stretches = [
            (order.index(which), len(order) - 1 - order[::-1].index(which))
            for which in range(len(schedule.storylines[index]))
        ]
        switches = sum(a != b for a, b in zip(order, order[1:]))
        active = [sum(first <= at <= last for first, last in stretches) for at in range(len(order))]
        assert switches > len(order) // 2
        assert max(active) <= plan.PARALLEL_STORYLINES
        # The longest thread (14 beats in a lane of about 30) runs for under half the year.
        assert all(last - first <= 0.55 * len(order) for first, last in stretches)


def test_schedule_is_deterministic(full):
    roster, schedule = full

    assert plan.build_schedule(build_roster()) == schedule
    assert plan.build_schedule(roster, plan.build_timeline(roster)) == schedule


def test_storyline_budgets():
    roster, smoke_roster = build_roster(), generate.smoke_roster()
    single = next(entry for entry in roster if entry.n_conversations == 1)

    assert plan.storyline_budget(roster[0]) == plan.StorylineBudget(14, 20, 3, 14, 2)
    assert plan.storyline_budget(single) == plan.StorylineBudget(1, 1, 1, 1, 0)
    for entry in roster + smoke_roster:
        budget = plan.storyline_budget(entry)
        n = entry.n_conversations
        # Whatever storyline count is drawn, the beats can be made to add up.
        assert budget.min_storylines >= max(1, len(entry.shared_topics))
        assert budget.max_storylines * budget.min_beats <= n
        assert n <= budget.min_storylines * budget.max_beats
        if not entry.person.is_heavy:
            assert budget.max_storylines <= 3
    # Three shared topics and one thread of their own.
    assert plan.storyline_budget(smoke_roster[0]).min_storylines == 4


# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #
def test_outline_prompt_lists_the_storyline_shapes_with_their_dates(full):
    roster, schedule = full
    heavy, single = roster[0], next(e for e in roster if e.n_conversations == 1)

    def outline(entry: RosterEntry) -> str:
        index = roster.index(entry)
        budget = plan.storyline_budget(entry)
        return prompts.outline_prompt(roster, entry, budget, schedule.storylines[index]).user

    text = outline(heavy)
    shapes = schedule.storylines[0]
    assert OWNER_BIO in text and "Person: Maya Chen" in text
    assert "Conversations: 150, between Wednesday 5 November 2025 and Wednesday 4 November 2026"
    row = r"^Storyline (\d+): (\d+) conversations? between (.+?) and (.+?)(,|$)"
    rows = re.findall(row, text, re.M)
    assert [int(number) for number, *_ in rows] == list(range(1, len(shapes) + 1))
    assert [int(n_beats) for _, n_beats, *_ in rows] == [shape.n_beats for shape in shapes]
    assert rows[0][2:4] == (prompts._date(shapes[0].first), prompts._date(shapes[0].last))
    for shape in shapes:
        marker = f', shared subject: "{shape.shared_topic}"'
        assert (marker in text) == bool(shape.shared_topic)
    assert "at least 2 pairs of storylines that are deliberately close in subject" in text
    assert "Fit each storyline's own dates and the time of year they fall in." in text
    assert "n_beats" not in text and "Fit the dates above" not in text
    _, reserved = text.split("Reserved subjects.")
    for topic in SHARED_TOPICS:
        assert (f'"{topic}"' in reserved.split("\n")[0]) == (topic not in heavy.shared_topics)
    assert "- Sam Whitfield: neighbour" in text and "- Maya Chen:" not in text
    assert all(name.split()[0] in text for name in UNKNOWN_NAMES)
    assert '{"storyline": 1, "title": "...", "premise": "..."}' in text

    text = outline(single)
    assert "Conversations: 1, on " in text
    assert re.search(r"^Storyline 1: 1 conversation on ", text, re.M)
    assert "Storyline 2" not in text and "pairs of storylines" not in text


def test_prompt_dates_use_the_benchmarks_calendar_names():
    """One spelling of the weekday and month names, shared with the time expressions the queries carry."""

    from eval.retrieval import timeutil
    from eval.retrieval.data import query_prompts

    assert prompts._WEEKDAYS is timeutil.WEEKDAY_NAMES and prompts._MONTHS is timeutil.MONTH_NAMES
    assert prompts._date(datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)) == "Wednesday 4 November 2026"
    assert prompts._date(datetime(2025, 12, 28, 9, 0, tzinfo=timezone.utc)) == "Sunday 28 December 2025"
    assert query_prompts.Prompt is prompts.Prompt  # and one Prompt type for both generators


def test_heavy_outline_examples_are_seeded_per_person_and_avoid_reserved_subjects(full):
    roster, schedule = full

    def examples(index: int) -> str:
        entry = roster[index]
        budget = plan.storyline_budget(entry)
        text = prompts.outline_prompt(roster, entry, budget, schedule.storylines[index]).user
        return re.search(r"long-running thread that develops \((.+?)\);", text).group(1)

    seen = [examples(index) for index in range(4)]
    assert len(set(seen)) == 4 and all(len(line.split(", ")) == 3 for line in seen)
    assert examples(0) == examples(0)
    for line in seen:
        assert "renovation" not in line and "trip" not in line
        for topic in SHARED_TOPICS:
            assert prompts.reserved_subject(line, [topic]) is None


def test_beats_prompt_marks_short_conversations():
    roster = build_roster()
    entry = roster[1]
    persona = {field: f"{field} text" for field in prompts.PERSONA_FIELDS}
    storylines = [
        plan.Storyline("a", "The migration", "Sam moves the billing service.", None, 2),
        plan.Storyline("b", "The offsite", "Sam plans the offsite.", None, 1),
    ]
    first = datetime(2025, 12, 11, 18, 0, tzinfo=timezone.utc)
    second = datetime(2025, 12, 19, 9, 0, tzinfo=timezone.utc)
    slots = [(first, first + timedelta(minutes=3)), (second, second + timedelta(minutes=25))]

    text = prompts.beats_prompt(roster, entry, persona, storylines, 0, slots).user
    assert "1. Thursday 11 December 2025, evening, short, about 8 turns" in text
    assert "2. Friday 19 December 2025, morning, about 22 turns" in text
    assert "exactly 2 for a conversation marked short" in text
    assert "1 to 2 for a conversation marked short" in text
    assert "implies a season or a month that does not match them, follow the dates" in text
    assert '- "The offsite": Sam plans the offsite.' in text
    assert prompts.is_short(*slots[0]) and not prompts.is_short(*slots[1])


def test_dialogue_prompt_carries_the_beat_and_the_story_so_far():
    entry = build_roster()[1]
    persona = {field: f"{field} text" for field in prompts.PERSONA_FIELDS}
    storyline = plan.Storyline("s", "The migration", "Sam moves the billing service.", None, 3)
    facts = ("New date is 12 May", "Dana signs off")
    beat = prompts.Beat("The cutover slips.", facts, ("a $40 lunch",))
    kickoff = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    earlier = [(kickoff, prompts.Beat("Kickoff.", ("Plan A",), ()))]
    start = datetime(2026, 4, 20, 14, 5, tzinfo=timezone.utc)

    def dialogue(episode_id: str, minutes: int = 3) -> str:
        end = start + timedelta(minutes=minutes)
        return prompts.dialogue_prompt(
            entry, persona, storyline, beat, earlier, episode_id, start, end
        ).user

    text = dialogue("episode-1")
    shape = '{"turns": [{"speaker": "Sam", "text": "..."}, {"speaker": "Me", "text": "..."}]}'
    assert text == dialogue("episode-1")
    assert OWNER_BIO in text and "The other speaker: Sam Okafor" in text
    assert "When: Monday 20 April 2026, 14:05 (afternoon)" in text
    assert "Speech style: speech_style text" in text
    assert 'Storyline: "The migration"' in text
    assert "Premise: Sam moves the billing service." in text
    assert "- Thursday 2 April 2026: Kickoff. Facts stated: Plan A" in text
    assert "What happens: The cutover slips." in text
    assert "- New date is 12 May" in text and "- Dana signs off" in text and "- a $40 lunch" in text
    assert "8 to 22 turns of natural spoken conversation, about 8 this time" in text
    assert "about 22 this time" in dialogue("episode-1", minutes=25)
    assert "Nothing in it that comes after this conversation has happened yet" in text
    assert "no narration, stage directions" in text and "glasses" in text
    assert shape in text
    openings = {re.search(r"- Opening: (.+)", dialogue(f"episode-{n}")).group(1) for n in range(40)}
    assert len(openings) >= 4  # conversations do not all start the same way


def test_summary_prompt_adapts_the_product_prompt_and_states_the_range():
    transcript = 'Sam: "Hi."\nMe: "Hello."'
    prompt = prompts.summary_prompt("Sam Okafor", transcript)

    assert prompt.user == f"Conversation between Riley Morgan and Sam Okafor:\n\n{transcript}"
    assert "memory system for a pair of smart glasses worn by Riley Morgan" in prompt.system
    assert "key topics, decisions, and any action items" in prompt.system
    assert "third person" in prompt.system and "between 20 and 100 words" in prompt.system
    assert '{"summary": "<the summary>"}' in prompt.system
    assert OWNER_BIO not in prompt.system
    again = prompts.summary_prompt("Sam Okafor", transcript, previous_words=131)
    assert again.system == prompt.system
    assert again.user.startswith(prompt.user)
    assert "was 131 words long" in again.user and "between 20 and 100 words" in again.user


def test_outline_replies_are_checked():
    roster = build_roster()
    entry = roster[3]  # Hannah Blake: three shared topics
    shapes = plan.build_schedule(roster).storylines[3]
    persona = {field: "x" for field in prompts.PERSONA_FIELDS}
    reserved = next(topic for topic in SHARED_TOPICS if topic not in entry.shared_topics)

    def outline(numbers=None, premise: str = "p", **changes) -> dict:
        numbers = list(range(1, len(shapes) + 1)) if numbers is None else numbers
        storylines = [
            {"storyline": number, "title": f"t{number}", "premise": premise} for number in numbers
        ]
        storylines[0].update(changes)
        return {"persona": persona, "storylines": storylines}

    _, drafts = prompts.parse_outline(outline(), entry, shapes)
    assert [draft.title for draft in drafts] == [f"t{n}" for n in range(1, len(shapes) + 1)]
    # Any order is fine: the drafts come back in shape order.
    _, drafts = prompts.parse_outline(outline(list(range(len(shapes), 0, -1))), entry, shapes)
    assert drafts[0] == prompts.StorylineDraft("t1", "p")
    unusable = {
        "one storyline too few": outline(list(range(1, len(shapes)))),
        "a number twice": outline([1, *range(1, len(shapes))]),
        "no number": outline(storyline=None),
        "a repeated title": outline(title="t2"),
        "a reserved subject": outline(premise=f"A thread about {reserved}, all year."),
        "a reserved subject in the title": outline(title=reserved.title()),
        "no persona": {"storylines": []},
        "no storylines": {"persona": persona},
    }
    for label, reply in unusable.items():
        with pytest.raises(prompts.ResponseError):
            prompts.parse_outline(reply, entry, shapes)
    # Their own shared subjects are allowed anywhere in their premises.
    own = outline(premise=f"About {entry.shared_topics[0]} and {entry.shared_topics[1]}.")
    prompts.parse_outline(own, entry, shapes)
    with pytest.raises(prompts.ResponseError, match="reserved subject"):
        prompts.parse_outline(outline(premise=f"A thread about {reserved}."), entry, shapes)


def test_reserved_subjects_are_found_by_phrase():
    found = prompts.reserved_subject

    assert found("Planning the Whistler ski trips", ["a ski trip"]) == "a ski trip"
    assert found("their tabletop game nights", ["a tabletop-game group"]) is None
    assert found("The Tabletop-Game Group grows", ["a tabletop-game group"])
    assert found("Learning Italian slowly", ["learning Italian"])
    assert found("A dog rescue", ["a rescue dog"]) is None
    assert found("a kitchen renovation", ["a ski trip", "a kitchen renovation"]) == (
        "a kitchen renovation"
    )


def test_beat_and_dialogue_replies_are_checked():
    beat = {"what_happens": " a\nb ", "key_facts": list("abcdef"), "minor_details": list("wxyz")}
    expected = prompts.Beat("a b", tuple("abcd"), tuple("wxy"))
    assert prompts.parse_beats({"beats": [beat]}) == [expected]
    assert prompts.parse_beats({"beats": [beat]}, short=[True]) == [
        prompts.Beat("a b", tuple("ab"), tuple("wx"))
    ]
    other = {"what_happens": "c", "key_facts": list("ghijkl"), "minor_details": list("mnop")}
    assert prompts.parse_beats({"beats": [beat, other]}, short=[True, False])[1] == prompts.Beat(
        "c", tuple("ghij"), tuple("mno")
    )
    repeated = {"what_happens": "x", "key_facts": ["Dana signs off"], "minor_details": ["q"]}
    twice = {"what_happens": "y", "key_facts": ["dana signs off"], "minor_details": ["r"]}
    with pytest.raises(prompts.ResponseError, match="beats 1 and 2 share 'dana signs off'"):
        prompts.parse_beats({"beats": [repeated, twice]})
    for reply in ({}, {"beats": []}, {"beats": [{"what_happens": "x", "key_facts": []}]}):
        with pytest.raises(prompts.ResponseError):
            prompts.parse_beats(reply)

    last = prompts.Beat("The vet calls.", ("Biscuit weighs 9 kg",), ("a chew toy",))
    recap = prompts.recap_beat(last)
    assert recap.recap and not last.recap
    assert recap.key_facts == last.key_facts and recap.minor_details == ()
    assert recap.what_happens.endswith("Last time: The vet calls.")

    def turns(*speakers: str, text: str = "words") -> dict:
        return {"turns": [{"speaker": speaker, "text": text} for speaker in speakers]}

    good = turns("Hannah", "Me", "Hannah", "Me", "Hannah", "Me")
    assert prompts.parse_dialogue(good, "Hannah")[1] == Turn(speaker="Me", text="words")
    unusable = (
        turns("Hannah", "Me", "Hannah", "Me", "Hannah"),
        turns("Hannah", "Riley", "Hannah", "Riley", "Hannah", "Riley"),
        turns("Hannah", "Hannah", "Hannah", "Hannah", "Hannah", "Hannah"),
        turns("Hannah", "Me", "Hannah", "Me", "Hannah", "Me", text="  "),
    )
    for reply in unusable:
        with pytest.raises(prompts.ResponseError):
            prompts.parse_dialogue(reply, "Hannah")

    summary = prompts.parse_summary({"summary": "  Riley and Hannah talked.\n"})
    assert summary == "Riley and Hannah talked."
    assert prompts.word_count("Riley and  Hannah\ntalked.") == 4
    with pytest.raises(prompts.ResponseError):
        prompts.parse_summary({"summary": ""})


def test_specifics_of_a_fact_are_looked_for_in_the_spoken_text():
    fact = "Dana's vet appointment is on 14 March at 9:30, costing $18,400"
    assert prompts.specifics(fact) == ["dana", "14", "march", "930", "18400"]
    assert prompts.specifics("a flat white") == []

    spoken = "Yeah, Dana's signing off May 12th. They quoted 18,400 dollars, 9:30 sharp."
    assert prompts.unstated(["Dana signs off on 12 May", "The quote was $18,400"], spoken) == []
    assert prompts.unstated(["a flat white"], spoken) == []  # nothing to check counts as stated
    assert prompts.unstated(["Dana signs off on 13 May"], spoken) == ["Dana signs off on 13 May"]
    assert prompts.unstated(["Marcus quoted $18,400"], spoken) == ["Marcus quoted $18,400"]
    assert prompts.unstated(["Room 2012"], "we met in room 2012") == []
    assert prompts.unstated(["Room 12"], "we met in room 2012") == ["Room 12"]


def test_speaker_names_and_sentence_openers_are_not_looked_for():
    speakers = prompts.speaker_names("Maya Chen")
    assert speakers == {"riley", "morgan", "maya", "chen"}

    # Maya says "I", and nobody addresses Riley by name: the names are not in the speech.
    spoken = "I've booked it at Ballard Cycle Works, November 12. Is that toast? Yeah, $480."
    booked = "Maya booked a replacement at Ballard Cycle Works for November 12."
    assert prompts.specifics(booked) == ["maya", "ballard", "cycle", "works", "november", "12"]
    assert prompts.specifics(booked, speakers) == ["ballard", "cycle", "works", "november", "12"]
    assert prompts.unstated([booked], spoken) == [booked]
    assert prompts.unstated([booked, "Riley is making toast.", "Maya's quote was $480."], spoken, speakers) == []
    assert prompts.unstated(["Maya's quote was $520."], spoken, speakers) == ["Maya's quote was $520."]

    # A first word capitalised for opening the sentence is not a name to find...
    assert prompts.specifics("Their last game was Ticket to Ride.") == ["ticket", "ride"]
    assert prompts.specifics("She plans to run around Green Lake.") == ["green", "lake"]
    assert prompts.specifics("Comparing footholds, they pick a green smear.") == []
    assert prompts.specifics("During their April practice, the scraping stopped.") == ["april"]
    # ...but a name or a number in first place still is, and openers elsewhere never were.
    assert prompts.specifics("Dana and The Shed agreed.") == ["dana", "the", "shed"]
    assert prompts.specifics("3 of them left.") == ["3"]
    assert prompts.unstated(["Marcus quoted $480"], spoken, speakers) == ["Marcus quoted $480"]


# --------------------------------------------------------------------------- #
# End to end                                                                   #
# --------------------------------------------------------------------------- #
def test_generated_corpus_is_valid_and_round_trips(smoke, smoke_dir):
    llm, corpus, roster = smoke
    people = dataset.load_people(smoke_dir / "people.jsonl")
    conversations = dataset.load_conversations(smoke_dir / "conversations.jsonl")

    assert isinstance(llm, ChatLLM)
    assert sorted(path.name for path in smoke_dir.iterdir()) == JSONL_NAMES
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []
    assert generate.validate_corpus(people, conversations, roster) == []
    assert [p.to_json() for p in people] == [p.to_json() for p in corpus.people]
    assert [c.to_json() for c in conversations] == [c.to_json() for c in corpus.conversations]
    assert len(people) == 6 and len(conversations) == 36
    assert [c.start_time for c in conversations] == sorted(c.start_time for c in conversations)


def test_generated_records_carry_their_meta(smoke):
    llm, corpus, roster = smoke
    by_id = {person.person_id: person for person in corpus.people}
    schedule = plan.build_schedule(roster)

    for index, (person, entry) in enumerate(zip(corpus.people, roster, strict=True)):
        own = person.meta["storylines"]
        assert set(person.meta) == set(generate.PERSON_META_KEYS)
        assert set(person.meta["persona"]) == set(prompts.PERSONA_FIELDS)
        assert person.meta["n_conversations"] == entry.n_conversations
        assert person.meta["shared_topics"] == list(entry.shared_topics)
        assert all(set(s) == {"storyline_id", "title", "shared_topic", "n_beats"} for s in own)
        # The storylines are the schedule's shapes, in order, with the model's titles.
        shapes = schedule.storylines[index]
        assert [(s["n_beats"], s["shared_topic"]) for s in own] == [
            (shape.n_beats, shape.shared_topic) for shape in shapes
        ]
        assert [s["title"] for s in own] == [
            f"{person.display_name} thread {n}" for n in range(1, len(shapes) + 1)
        ]
    for conversation in corpus.conversations:
        meta, person = conversation.meta, by_id[conversation.person_id]
        storylines = {s["storyline_id"]: s for s in person.meta["storylines"]}
        storyline = storylines[meta["storyline_id"]]
        speakers = {turn.speaker for turn in conversation.dialogue}
        spoken = conversation.dialogue_text()
        kept = 2 if prompts.is_short(conversation.start_time, conversation.end_time) else 3
        assert set(meta) == set(generate.CONVERSATION_META_KEYS)
        assert meta["person_name"] == person.display_name
        assert meta["storyline_title"] == storyline["title"]
        assert meta["shared_topic"] == storyline["shared_topic"]
        assert 0 <= meta["beat_index"] < storyline["n_beats"]
        assert len(meta["key_facts"]) == kept and len(meta["minor_details"]) == kept
        assert meta["recap"] is False
        assert meta["unstated_facts"] == [] and meta["unstated_details"] == []
        assert meta["details_in_summary"] == meta["minor_details"][:1]
        assert (meta["dialogue_model"], meta["summary_model"]) == (GEN_MODEL, SUMMARY_MODEL)
        assert speakers == {person.first_name, OWNER_SPEAKER}
        assert all(fact in spoken for fact in meta["key_facts"] + meta["minor_details"])
    assert any(prompts.is_short(c.start_time, c.end_time) for c in corpus.conversations)
    n_storylines = sum(len(person.meta["storylines"]) for person in corpus.people)
    assert {name: stats.calls for name, stats in corpus.stats.items()} == {
        "persona": 6,
        "beats": n_storylines,
        "dialogue": 36,
        "summary": 36,
    }
    assert all(stats.cached == 0 for stats in corpus.stats.values())
    total_cost = sum(stats.cost_usd for stats in corpus.stats.values())
    assert total_cost == pytest.approx(0.001 * len(llm.calls))


def test_each_kind_of_call_uses_its_model_and_settings(smoke):
    llm, _, _ = smoke

    assert {call.namespace for call in llm.calls} == {"persona", "beats", "dialogue", "summary"}
    for call in llm.calls:
        how = (call.model, call.temperature, call.reasoning_effort)
        if call.namespace == "summary":
            assert how == (SUMMARY_MODEL, None, "low")  # a reasoning model takes no temperature
        else:
            assert how == (GEN_MODEL, 0.8, None)
        assert call.salt == ""
    assert all(OWNER_BIO in call.user for call in llm.calls if call.namespace != "summary")


def test_summary_call_sees_only_the_names_and_the_dialogue(smoke):
    llm, corpus, _ = smoke
    seen = {call.user: call for call in llm.of("summary")}

    assert len(seen) == len(corpus.conversations)
    for conversation in corpus.conversations:
        name, transcript = conversation.meta["person_name"], conversation.dialogue_text()
        call = seen[f"Conversation between Riley Morgan and {name}:\n\n{transcript}"]
        everything = call.system + call.user
        # The facts and details are there only because the dialogue says them; nothing else is.
        assert all(fact in transcript for fact in conversation.meta["key_facts"])
        assert conversation.meta["storyline_title"] not in everything
        assert "WHAT[" not in everything and "PREMISE[" not in everything
        assert OWNER_BIO not in everything and "persona" not in everything.lower()
    assert len({call.system for call in llm.of("summary")}) == 1


def test_dialogue_calls_follow_the_storyline(smoke):
    llm, corpus, _ = smoke
    prompts_by_fact = {
        fact: call.user
        for call in llm.of("dialogue")
        for fact in KEY_FACTS.search(call.user).group(1).splitlines()
    }

    for conversation in corpus.conversations:
        meta, start = conversation.meta, conversation.start_time
        text = prompts_by_fact[f"- {meta['key_facts'][0]}"]
        earlier = text.split("in this storyline, oldest first:\n")[1].split("\n\n")[0]
        assert f"When: {prompts._date(start)}, {start:%H:%M}" in text
        assert all(f"- {detail}" in text for detail in meta["minor_details"])
        assert len(earlier.splitlines()) == max(1, meta["beat_index"])
        assert ("- None:" in earlier) == (meta["beat_index"] == 0)


def test_summary_outside_the_word_range_is_asked_again(tmp_path):
    llm, corpus, roster = run(tmp_path, summary_long=2)
    by_salt = Counter(call.salt for call in llm.of("summary"))

    assert by_salt == {"": 36, "reask-1": 36, "reask-2": 36}
    assert all("was 150 words long" in call.user for call in llm.of("summary") if call.salt)
    assert all(prompts.word_count(c.summary) == 30 for c in corpus.conversations)
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []
    assert corpus.stats["summary"].calls == 108
    with pytest.raises(generate.GenerationError, match="150 words, outside 20-100"):
        run(tmp_path, summary_long=generate.SUMMARY_REASKS + 1)


def test_wrong_beat_count_is_asked_again_once_then_padded_or_truncated(tmp_path):
    llm, corpus, roster = run(tmp_path / "reask", beats_wrong=1)
    n_storylines = sum(len(person.meta["storylines"]) for person in corpus.people)
    salts = Counter(call.salt for call in llm.of("beats"))
    assert salts == {"": n_storylines, "reask-1": n_storylines}
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []

    llm, corpus, roster = run(tmp_path / "truncate", beats_wrong=2, beats_delta=2)
    assert len(llm.of("beats")) == 2 * n_storylines
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []
    assert not any(c.meta["recap"] for c in corpus.conversations)

    llm, corpus, roster = run(tmp_path / "pad", beats_wrong=2, beats_delta=-1)
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []
    lengths = {s["storyline_id"]: s["n_beats"] for p in corpus.people for s in p.meta["storylines"]}
    by_beat = {(c.meta["storyline_id"], c.meta["beat_index"]): c for c in corpus.conversations}
    padded = [c for c in corpus.conversations if c.meta["recap"]]
    assert {c.meta["storyline_id"] for c in padded} == {key for key, n in lengths.items() if n > 1}
    asked = {call.user for call in llm.of("dialogue")}
    for conversation in padded:  # the catch-up conversation restates the beat before it
        storyline_id, beat = conversation.meta["storyline_id"], conversation.meta["beat_index"]
        previous = by_beat[(storyline_id, beat - 1)]
        assert beat == lengths[storyline_id] - 1
        assert conversation.meta["key_facts"] == previous.meta["key_facts"]
        assert conversation.meta["minor_details"] == [] and not previous.meta["recap"]
        stated = f"Facts stated: {'; '.join(previous.meta['key_facts'])}"
        prompt = next(text for text in asked if stated in text)
        what_happens = re.search(r"What happens: (.+)", prompt).group(1)
        assert conversation.meta["storyline_title"] not in what_happens


def test_duplicate_facts_in_a_beats_reply_are_asked_again(tmp_path):
    llm, corpus, roster = run(tmp_path, beats_repeat=1)
    n_storylines = sum(len(person.meta["storylines"]) for person in corpus.people)
    longer = sum(s["n_beats"] > 1 for p in corpus.people for s in p.meta["storylines"])

    assert Counter(call.salt for call in llm.of("beats")) == {"": n_storylines, "reask-1": longer}
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []
    with pytest.raises(generate.GenerationError, match="unusable after 2 attempts"):
        run(tmp_path / "again", beats_repeat=2)


def test_dialogue_that_leaves_a_detail_unspoken_is_asked_again_then_recorded(tmp_path):
    llm, corpus, roster = run(tmp_path / "once", dialogue_mute=1)
    assert Counter(call.salt for call in llm.of("dialogue")) == {"": 36, "reask-1": 36}
    assert all(not c.meta["unstated_details"] for c in corpus.conversations)
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []

    llm, corpus, roster = run(tmp_path / "kept", dialogue_mute=generate.DIALOGUE_REASKS + 1)
    salts = Counter(call.salt for call in llm.of("dialogue"))
    assert salts == {"": 36, "reask-1": 36, "reask-2": 36}
    for conversation in corpus.conversations:
        meta = conversation.meta
        assert meta["unstated_facts"] == []
        assert meta["unstated_details"] == meta["minor_details"]
        assert meta["details_in_summary"] == []
        assert not any(detail in conversation.dialogue_text() for detail in meta["minor_details"])
    # Recorded for the query writer, not a reason to reject the corpus.
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []


def test_unusable_replies_are_asked_again_and_then_fail_loudly(tmp_path):
    llm, corpus, roster = run(tmp_path / "ok", outline_bad=1, outline_reserved=2, dialogue_bad=2)
    for namespace, n in (("persona", 6), ("dialogue", 36)):
        salts = Counter(call.salt for call in llm.of(namespace))
        assert salts == {"": n, "reask-1": n, "reask-2": n}
    assert generate.validate_corpus(corpus.people, corpus.conversations, roster) == []

    with pytest.raises(generate.GenerationError, match="outline for .* persona lacks age"):
        run(tmp_path / "outline", workers=1, outline_bad=3)
    with pytest.raises(generate.GenerationError, match="is about the reserved subject"):
        run(tmp_path / "reserved", workers=1, outline_reserved=3)
    with pytest.raises(generate.GenerationError, match="unexpected speaker label 'Riley'"):
        run(tmp_path / "dialogue", dialogue_bad=3)
    assert not list((tmp_path / "dialogue").glob("*.jsonl"))  # a failed run writes nothing


def test_runs_are_reproducible(tmp_path):
    first, _, _ = run(tmp_path / "a", workers=1)
    second, _, _ = run(tmp_path / "b", workers=1)
    threaded, _, _ = run(tmp_path / "c", workers=8)

    assert first.calls == second.calls
    assert sorted(threaded.calls, key=repr) == sorted(first.calls, key=repr)
    for name in JSONL_NAMES:
        expected = (tmp_path / "a" / name).read_bytes()
        assert (tmp_path / "b" / name).read_bytes() == expected
        assert (tmp_path / "c" / name).read_bytes() == expected


def test_full_roster_generates_a_valid_corpus(tmp_path):
    llm, corpus, _ = run(tmp_path, roster=build_roster(), workers=8)
    sizes = Counter(conversation.person_id for conversation in corpus.conversations)

    assert generate.validate_corpus(corpus.people, corpus.conversations) == []
    assert len(corpus.people) == 40 and len(corpus.conversations) == 800
    assert sorted(sizes.values())[-4:] == sorted(config.HEAVY_PERSON_COUNTS)
    assert all(14 <= len(person.meta["storylines"]) <= 20 for person in corpus.people[:4])
    assert all(1 <= len(person.meta["storylines"]) <= 3 for person in corpus.people[4:])
    for topic in SHARED_TOPICS:
        talkers = {c.person_id for c in corpus.conversations if c.meta["shared_topic"] == topic}
        assert 3 <= len(talkers) <= 4
    assert len(llm.of("dialogue")) == len(llm.of("summary")) == 800


# --------------------------------------------------------------------------- #
# Validation and the command line                                              #
# --------------------------------------------------------------------------- #
Damage = Callable[[list[Conversation]], list[Conversation]]


def _first(change: Callable[[Conversation], Conversation]) -> Damage:
    """A corpus edit that rewrites the earliest conversation."""

    return lambda conversations: [change(conversations[0]), *conversations[1:]]


def _shifted(conversation: Conversation, **delta: int) -> Conversation:
    shift = timedelta(**delta)
    start, end = conversation.start_time + shift, conversation.end_time + shift
    return replace(conversation, start_time=start, end_time=end)


def _relabelled(conversation: Conversation) -> Conversation:
    turns = [Turn("Riley", turn.text) for turn in conversation.dialogue]
    return replace(conversation, dialogue=turns)


def _with_meta(conversation: Conversation, **changes) -> Conversation:
    return replace(conversation, meta={**conversation.meta, **changes})


def _overlapping(conversations: list[Conversation]) -> list[Conversation]:
    first, second = conversations[:2]
    late = second.start_time + timedelta(minutes=1)
    return [replace(first, end_time=late), *conversations[1:]]


def _beats_swapped(conversations: list[Conversation]) -> list[Conversation]:
    second = next(c for c in conversations if c.meta["beat_index"] == 1)
    thread = [c for c in conversations if c.meta["storyline_id"] == second.meta["storyline_id"]]
    first = next(c for c in thread if c.meta["beat_index"] == 0)
    swapped = {
        first.episode_id: replace(first, start_time=second.start_time, end_time=second.end_time),
        second.episode_id: replace(second, start_time=first.start_time, end_time=first.end_time),
    }
    return [swapped.get(c.episode_id, c) for c in conversations]


def _fact_copied(conversations: list[Conversation], recap: bool = False) -> list[Conversation]:
    """A later conversation of the same person carries the first one's first key fact."""

    first = conversations[0]
    later = next(c for c in conversations[1:] if c.person_id == first.person_id)
    facts = [first.meta["key_facts"][0], *later.meta["key_facts"][1:]]
    turns = [*later.dialogue, Turn(OWNER_SPEAKER, f"Did you hear? {facts[0]}")]
    copied = replace(_with_meta(later, key_facts=facts, recap=recap), dialogue=turns)
    return [copied if c.episode_id == later.episode_id else c for c in conversations]


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        (lambda cs: cs[1:], "35 conversations, expected 36"),
        (lambda cs: cs[1:], "has 9 conversations, expected 10"),
        (_first(lambda c: replace(c, summary="too short")), "has 2 words, not 20-100"),
        (_first(lambda c: replace(c, dialogue=c.dialogue[:3])), "has only 3 turns"),
        (_first(_relabelled), "is not between"),
        (_first(lambda c: replace(c, meta={"person_name": "x"})), "lacks meta storyline_id"),
        (_overlapping, "overlap"),
        (_first(lambda c: replace(c, end_time=c.start_time + timedelta(hours=1))), "3-25 minutes"),
        (_first(lambda c: _shifted(c, days=-400)), "outside the dataset span"),
        (_first(lambda c: _shifted(c, hours=-12)), "outside 07:00-22:00"),
        (lambda cs: [*cs[:-1], _shifted(cs[-1], days=-30)], "before 12:00, need at least 1"),
        (_beats_swapped, "out of order: [1, 0"),
        (lambda cs: [*cs[1:], replace(cs[1], person_id="nobody")], "belong to nobody"),
        (lambda cs: [*cs[1:], cs[1]], "episode ids are not unique"),
        (_fact_copied, "repeats the key fact"),
        (_first(lambda c: _with_meta(c, unstated_facts=["x"])), "meta unstated_facts is stale"),
        (
            _first(lambda c: _with_meta(c, unstated_details=c.meta["minor_details"])),
            "meta unstated_details is stale",
        ),
        (_first(lambda c: replace(c, summary="word " * 30)), "meta details_in_summary is stale"),
        (_first(lambda c: replace(c, dialogue=c.dialogue[2:])), "meta unstated_facts is stale"),
    ],
)
def test_validation_names_what_is_wrong(smoke, damage: Damage, expected: str):
    _, corpus, roster = smoke
    problems = generate.validate_corpus(corpus.people, damage(list(corpus.conversations)), roster)

    assert any(expected in problem for problem in problems), problems


def test_a_recap_may_repeat_the_key_facts_before_it(smoke):
    _, corpus, roster = smoke
    conversations = _fact_copied(list(corpus.conversations), recap=True)

    problems = generate.validate_corpus(corpus.people, conversations, roster)
    assert not any("repeats the key fact" in problem for problem in problems)


def test_validation_checks_the_people(smoke):
    _, corpus, roster = smoke
    people = corpus.people

    def problems(changed) -> str:
        return " | ".join(generate.validate_corpus(changed, corpus.conversations, roster))

    assert "5 people, expected 6" in problems(people[:-1])
    renamed = [*people[:-1], replace(people[-1], first_name="Maya", display_name="Maya Morgan")]
    assert "expected one first name shared by two people" in problems(renamed)
    assert "is not the roster's Walter Morgan" in problems(renamed)
    assert "lacks meta persona" in problems([replace(people[0], meta={}), *people[1:]])
    assert "is not 'First Last'" in problems([replace(people[0], display_name="Maya"), *people[1:]])
    # Without a roster the corpus is held against the full, committed shape.
    unscoped = generate.validate_corpus(people, corpus.conversations)
    assert "36 conversations, expected 800" in unscoped


def test_a_corpus_that_fails_validation_is_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "validate_corpus", lambda *args: ["something is off"])

    with pytest.raises(generate.ValidationError, match="something is off") as caught:
        run(tmp_path)
    assert caught.value.problems == ["something is off"]
    assert isinstance(caught.value, generate.GenerationError)
    assert not list(tmp_path.glob("*.jsonl"))


class _FakeSpend:
    total = 1.25


class _FakeRuntime:
    def __init__(self) -> None:
        self.llm = FakeLLM()
        self.settings = SETTINGS_FOR_TESTS
        self.spend = _FakeSpend()


def test_command_line_smoke_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(generate, "Runtime", _FakeRuntime)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "good")

    assert generate.main(["--smoke", "--workers", "2"]) == 0
    output = capsys.readouterr().out
    assert sorted(path.name for path in (tmp_path / "good" / "smoke").iterdir()) == JSONL_NAMES
    assert "people: 6 (2 heavy)" in output and "conversations: 36" in output
    assert "leave a key fact or minor detail unspoken: 0" in output
    assert "hosted spend this run: $1.25" in output and "validation: ok" in output
    assert re.search(r"dialogue\s+36 calls,\s+0 from cache", output)

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "bad")
    monkeypatch.setattr(generate, "validate_corpus", lambda *args: ["something is off"])
    assert generate.main(["--smoke"]) == 1
    output = capsys.readouterr().out
    assert "validation: 1 problems; nothing written" in output and "something is off" in output
    assert "hosted spend this run: $1.25" in output
    assert not (tmp_path / "bad").exists()
