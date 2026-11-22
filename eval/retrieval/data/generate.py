"""Generate the benchmark corpus: ``people.jsonl`` and ``conversations.jsonl``.

    python -m eval.retrieval.data.generate [--smoke] [--workers N]

Who talks to the owner, how often, when, how their storylines are shaped and
which storyline beat a conversation carries is pure code seeded from
``config.SEED``. The text comes from the LLM in four kinds of calls: an
outline per person (persona and a title and premise per storyline), the beats
of each storyline, a dialogue per conversation, and a summary per conversation
written by a separate model that sees only the dialogue. The client caches
every reply by prompt, so an interrupted run resumes where it stopped and a
re-run reproduces the same corpus. Nothing is written until the whole corpus
has passed ``validate_corpus``.
"""

from __future__ import annotations

import argparse
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

from .. import config, dataset
from ..interfaces import ChatLLM, LLMResult
from ..runtime import Runtime
from ..schema import OWNER_SPEAKER, Conversation, Person, Turn
from . import corpus_prompts as prompts
from . import plan
from .corpus_prompts import Beat, Prompt
from .plan import PlannedConversation, Slot, Storyline, StorylineShape
from .queries import CallStats, MeteredLLM, content_words, light_stem
from .roster import RosterEntry, build_roster

logger = logging.getLogger(__name__)

# How often a call is asked again (with a new cache salt) before the run stops.
OUTLINE_REASKS = 2
BEATS_REASKS = 1  # after that a wrong beat count is padded or truncated in code
DIALOGUE_REASKS = 2  # after that the attempt that states the most is kept, misses recorded
SUMMARY_REASKS = 3

SMOKE_PEOPLE = 6
SMOKE_HEAVY_COUNTS = (10, 10)
SMOKE_TOTAL = 36

PERSON_META_KEYS = ("persona", "n_conversations", "shared_topics", "storylines")
CONVERSATION_META_KEYS = (
    "person_name",
    "storyline_id",
    "storyline_title",
    "beat_index",
    "shared_topic",
    "key_facts",
    "minor_details",
    "recap",  # padding written in code: repeats the key facts of the beat before it
    "unstated_facts",  # key facts whose names and numbers the dialogue does not contain
    "unstated_details",  # the same for minor details
    "details_in_summary",  # minor details the summary repeats (or that have nothing to check)
    "dialogue_model",
    "summary_model",
)

T = TypeVar("T")
R = TypeVar("R")


class GenerationError(RuntimeError):
    """The model kept returning something unusable; the run stops rather than patch the text."""


class ValidationError(GenerationError):
    """The generated corpus failed ``validate_corpus``, so nothing was written."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__(f"the corpus failed validation: {'; '.join(problems)}")
        self.problems = list(problems)


@dataclass(frozen=True)
class _CallKind:
    namespace: str
    max_tokens: int
    temperature: float | None  # None: the model's default, the only value reasoning models take
    reasoning_effort: str | None = None


# max_tokens leaves room for the longest reply of each kind (a heavy person's
# outline, a 14-beat storyline, a 22-turn dialogue, a summary plus its reasoning).
_OUTLINE = _CallKind("persona", 6000, 0.8)
_BEATS = _CallKind("beats", 8000, 0.8)
_DIALOGUE = _CallKind("dialogue", 4000, 0.8)
_SUMMARY = _CallKind("summary", 2000, None, "low")


@dataclass
class GeneratedCorpus:
    people: list[Person]
    conversations: list[Conversation]  # in chronological order
    stats: dict[str, CallStats] = field(default_factory=dict)  # by call namespace


class _Client:
    """The LLM with the generator's settings for each kind of call; the metered LLM counts what it returns."""

    def __init__(self, llm: ChatLLM, settings: config.Settings) -> None:
        self._llm = MeteredLLM(llm)
        self._settings = settings
        for kind in (_OUTLINE, _BEATS, _DIALOGUE, _SUMMARY):  # every kind is reported, used or not
            self._llm.stats.setdefault(kind.namespace, CallStats())

    @property
    def stats(self) -> dict[str, CallStats]:
        return self._llm.stats

    def ask(self, kind: _CallKind, prompt: Prompt, attempt: int) -> LLMResult:
        """One reply. ``attempt`` 0 is the plain call; a re-ask gets its own cache entry."""

        settings = self._settings
        return self._llm.complete_json(
            model=settings.summary_model if kind is _SUMMARY else settings.gen_model,
            system=prompt.system,
            user=prompt.user,
            max_tokens=kind.max_tokens,
            temperature=kind.temperature,
            reasoning_effort=kind.reasoning_effort,
            namespace=kind.namespace,
            salt="" if attempt == 0 else f"reask-{attempt}",
        )


@dataclass(frozen=True)
class _Thread:
    """One storyline of one person, with what is needed to write its conversations."""

    entry: RosterEntry
    persona: dict[str, str]
    storyline: Storyline
    planned: list[PlannedConversation]  # by beat index
    beats: list[Beat]


@dataclass(frozen=True)
class _Dialogue:
    """A usable dialogue reply, and what the beat asked for that it does not say."""

    turns: list[Turn]
    model: str
    unstated_facts: list[str]
    unstated_details: list[str]

    @property
    def n_unstated(self) -> int:
        return len(self.unstated_facts) + len(self.unstated_details)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def generate_corpus(
    llm: ChatLLM,
    settings: config.Settings,
    roster: Sequence[RosterEntry] | None = None,
    out_dir: Path = config.DATA_DIR,
    workers: int | None = None,
) -> GeneratedCorpus:
    """Write ``people.jsonl`` and ``conversations.jsonl`` for ``roster`` into ``out_dir``.

    Calls that do not depend on each other run on ``workers`` threads
    (``settings.llm_concurrency`` by default); results are assembled in roster
    and chronological order, so the output does not depend on thread timing.
    The files are written only once the corpus has passed ``validate_corpus``;
    otherwise ``ValidationError`` carries the problems and nothing on disk changes.
    """

    roster = build_roster() if roster is None else list(roster)
    workers = settings.llm_concurrency if workers is None else workers
    client = _Client(llm, settings)

    # The whole schedule is code, so every outline knows when each of its storylines runs.
    schedule = plan.build_schedule(roster)
    outlines = _parallel(
        "outlines",
        lambda index: _write_outline(client, roster, index, schedule.storylines[index]),
        range(len(roster)),
        workers,
    )
    personas = [persona for persona, _ in outlines]
    storylines = [own for _, own in outlines]

    planned_by_thread: dict[tuple[int, int], list[PlannedConversation]] = {}
    for planned in schedule.conversations:  # chronological, so each list ends up in beat order
        key = (planned.person_index, planned.storyline_index)
        planned_by_thread.setdefault(key, []).append(planned)
    keys = sorted(planned_by_thread)
    beats = _parallel(
        "beats",
        lambda key: _write_beats(
            client,
            roster,
            key[0],
            personas[key[0]],
            storylines[key[0]],
            key[1],
            [(planned.start_time, planned.end_time) for planned in planned_by_thread[key]],
        ),
        keys,
        workers,
    )
    threads = {
        key: _Thread(
            entry=roster[key[0]],
            persona=personas[key[0]],
            storyline=storylines[key[0]][key[1]],
            planned=planned_by_thread[key],
            beats=own,
        )
        for key, own in zip(keys, beats, strict=True)
    }

    def thread_of(planned: PlannedConversation) -> _Thread:
        return threads[(planned.person_index, planned.storyline_index)]

    dialogues = _parallel(
        "dialogues",
        lambda planned: _write_dialogue(client, thread_of(planned), planned),
        schedule.conversations,
        workers,
    )
    conversations = [
        _conversation(thread_of(planned), planned, dialogue)
        for planned, dialogue in zip(schedule.conversations, dialogues, strict=True)
    ]
    summaries = _parallel(
        "summaries",
        lambda conversation: _write_summary(client, conversation),
        conversations,
        workers,
    )
    for conversation, (summary, model) in zip(conversations, summaries, strict=True):
        conversation.summary = summary
        conversation.meta["summary_model"] = model
        conversation.meta["details_in_summary"] = _in_summary(
            conversation.meta["minor_details"], summary, conversation.meta["person_name"]
        )

    people = [
        replace(
            entry.person,
            meta={
                "persona": persona,
                "n_conversations": entry.n_conversations,
                "shared_topics": list(entry.shared_topics),
                "storylines": [
                    {
                        "storyline_id": storyline.storyline_id,
                        "title": storyline.title,
                        "shared_topic": storyline.shared_topic,
                        "n_beats": storyline.n_beats,
                    }
                    for storyline in own
                ],
            },
        )
        for entry, persona, own in zip(roster, personas, storylines, strict=True)
    ]
    problems = validate_corpus(people, conversations, roster)
    if problems:
        raise ValidationError(problems)
    out_dir = Path(out_dir)
    dataset.save_people(people, out_dir / config.PEOPLE_PATH.name)
    dataset.save_conversations(conversations, out_dir / config.CONVERSATIONS_PATH.name)
    logger.info("wrote %d people, %d conversations to %s", len(people), len(conversations), out_dir)
    return GeneratedCorpus(people=people, conversations=conversations, stats=client.stats)


def _parallel(label: str, task: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """Run ``task`` over ``items`` on a thread pool; results come back in the order of ``items``."""

    logger.info("%s: %d to write", label, len(items))
    lock = threading.Lock()
    done = 0
    step = max(1, len(items) // 10)

    def tracked(item: T) -> R:
        nonlocal done
        result = task(item)
        with lock:
            done += 1
            if done % step == 0 or done == len(items):
                logger.info("%s: %d/%d", label, done, len(items))
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(tracked, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # One failure stops the run: do not start (and pay for) the calls still queued.
            for future in futures:
                future.cancel()
            raise


# --------------------------------------------------------------------------- #
# One call each, with its re-ask rule                                          #
# --------------------------------------------------------------------------- #
def _write_outline(
    client: _Client, roster: Sequence[RosterEntry], index: int, shapes: Sequence[StorylineShape]
) -> tuple[dict[str, str], list[Storyline]]:
    """A person's persona and their storyline shapes filled in with a title and a premise."""

    entry = roster[index]
    name = entry.person.display_name
    prompt = prompts.outline_prompt(roster, entry, plan.storyline_budget(entry), shapes)
    problem = ""
    for attempt in range(OUTLINE_REASKS + 1):
        reply = client.ask(_OUTLINE, prompt, attempt)
        try:
            persona, drafts = prompts.parse_outline(reply.data, entry, shapes)
        except prompts.ResponseError as error:
            problem = str(error)
            logger.warning("outline for %s: %s (attempt %d)", name, problem, attempt + 1)
            continue
        return persona, [
            Storyline(
                storyline_id=config.stable_id("storyline", f"{index}:{number}"),
                title=draft.title,
                premise=draft.premise,
                shared_topic=shape.shared_topic,
                n_beats=shape.n_beats,
            )
            for number, (draft, shape) in enumerate(zip(drafts, shapes, strict=True))
        ]
    raise GenerationError(
        f"outline for {name} unusable after {OUTLINE_REASKS + 1} attempts: {problem}"
    )


def _write_beats(
    client: _Client,
    roster: Sequence[RosterEntry],
    index: int,
    persona: dict[str, str],
    storylines: Sequence[Storyline],
    which: int,
    slots: Sequence[Slot],
) -> list[Beat]:
    """The beats of one storyline, exactly one per conversation in ``slots``."""

    storyline, name = storylines[which], roster[index].person.display_name
    prompt = prompts.beats_prompt(roster, roster[index], persona, storylines, which, slots)
    short = [prompts.is_short(start, end) for start, end in slots]
    beats: list[Beat] | None = None
    for attempt in range(BEATS_REASKS + 1):
        reply = client.ask(_BEATS, prompt, attempt)
        try:
            beats = prompts.parse_beats(reply.data, short)
        except prompts.ResponseError as error:
            logger.warning(
                "beats for %s / %r: %s (attempt %d)", name, storyline.title, error, attempt + 1
            )
            continue
        if len(beats) == len(slots):
            return beats
        logger.warning(
            "beats for %s / %r: %d returned for %d conversations (attempt %d)",
            name,
            storyline.title,
            len(beats),
            len(slots),
            attempt + 1,
        )
    if beats is None:
        raise GenerationError(
            f"beats for {name} / {storyline.title!r} unusable after {BEATS_REASKS + 1} attempts"
        )
    # Still the wrong number: drop the surplus, or fill up with catch-up conversations.
    logger.warning("beats for %s / %r: fitted to %d in code", name, storyline.title, len(slots))
    fitted = beats[: len(slots)]
    fitted += [prompts.recap_beat(beats[-1])] * (len(slots) - len(fitted))
    return fitted


def _write_dialogue(client: _Client, thread: _Thread, planned: PlannedConversation) -> _Dialogue:
    """The turns of one conversation, the model that wrote them, and what they fail to state.

    A reply that leaves a key fact or a minor detail unspoken is asked for
    again, like one with the wrong shape; after the re-asks the attempt that
    states the most is kept and its misses go into the record, where the query
    writer can see them.
    """

    beat = thread.beats[planned.beat_index]
    earlier = [
        (thread.planned[past].start_time, thread.beats[past]) for past in range(planned.beat_index)
    ]
    prompt = prompts.dialogue_prompt(
        thread.entry,
        thread.persona,
        thread.storyline,
        beat,
        earlier,
        planned.episode_id,
        planned.start_time,
        planned.end_time,
    )
    problem = ""
    best: _Dialogue | None = None
    speakers = prompts.speaker_names(thread.entry.person.display_name)
    for attempt in range(DIALOGUE_REASKS + 1):
        reply = client.ask(_DIALOGUE, prompt, attempt)
        try:
            turns = prompts.parse_dialogue(reply.data, thread.entry.person.first_name)
        except prompts.ResponseError as error:
            problem = str(error)
            logger.warning("dialogue %s: %s (attempt %d)", planned.episode_id, problem, attempt + 1)
            continue
        spoken = " ".join(turn.text for turn in turns)
        written = _Dialogue(
            turns=turns,
            model=reply.model,
            unstated_facts=prompts.unstated(beat.key_facts, spoken, speakers),
            unstated_details=prompts.unstated(beat.minor_details, spoken, speakers),
        )
        if not written.n_unstated:
            return written
        problem = "does not state " + "; ".join(written.unstated_facts + written.unstated_details)
        logger.warning("dialogue %s: %s (attempt %d)", planned.episode_id, problem, attempt + 1)
        if best is None or written.n_unstated < best.n_unstated:
            best = written
    if best is not None:
        logger.warning("dialogue %s: kept with %d unstated", planned.episode_id, best.n_unstated)
        return best
    name, title = thread.entry.person.display_name, thread.storyline.title
    raise GenerationError(
        f"dialogue {planned.episode_id} ({name}, {title!r}, beat {planned.beat_index}) "
        f"unusable after {DIALOGUE_REASKS + 1} attempts: {problem}"
    )


def _write_summary(client: _Client, conversation: Conversation) -> tuple[str, str]:
    """A summary inside the word range and the model that wrote it.

    The prompt carries the two names and the rendered dialogue, nothing else. A
    summary outside the range is asked for again, never trimmed or padded here.
    """

    low, high = config.SUMMARY_WORD_RANGE
    name, transcript = conversation.meta["person_name"], conversation.dialogue_text()
    previous_words: int | None = None
    problem = ""
    for attempt in range(SUMMARY_REASKS + 1):
        prompt = prompts.summary_prompt(name, transcript, previous_words)
        reply = client.ask(_SUMMARY, prompt, attempt)
        try:
            summary = prompts.parse_summary(reply.data)
        except prompts.ResponseError as error:
            problem = str(error)
        else:
            words = prompts.word_count(summary)
            if low <= words <= high:
                return summary, reply.model
            previous_words, problem = words, f"{words} words, outside {low}-{high}"
        logger.warning("summary %s: %s (attempt %d)", conversation.episode_id, problem, attempt + 1)
    raise GenerationError(
        f"summary {conversation.episode_id} unusable after {SUMMARY_REASKS + 1} attempts: {problem}"
    )


def _conversation(
    thread: _Thread, planned: PlannedConversation, dialogue: _Dialogue
) -> Conversation:
    """The record for one conversation, complete except for its summary and what that repeats."""

    beat = thread.beats[planned.beat_index]
    return Conversation(
        episode_id=planned.episode_id,
        person_id=thread.entry.person.person_id,
        start_time=planned.start_time,
        end_time=planned.end_time,
        dialogue=dialogue.turns,
        summary="",
        meta={
            "person_name": thread.entry.person.display_name,
            "storyline_id": thread.storyline.storyline_id,
            "storyline_title": thread.storyline.title,
            "beat_index": planned.beat_index,
            "shared_topic": thread.storyline.shared_topic,
            "key_facts": list(beat.key_facts),
            "minor_details": list(beat.minor_details),
            "recap": beat.recap,
            "unstated_facts": list(dialogue.unstated_facts),
            "unstated_details": list(dialogue.unstated_details),
            "details_in_summary": [],
            "dialogue_model": dialogue.model,
            "summary_model": "",
        },
    )


def _in_summary(details: Sequence[str], summary: str, person_name: str) -> list[str]:
    """The minor details the summary gives away.

    Detail-only questions are written from the other details: the ones a
    summary demonstrably leaves out. A detail with a name or a number is given
    away when the summary repeats all of them. Most details have neither
    ("Sam's thermos has a dent in its lid"); such a detail is given away when
    the summary shares at least half of its content words. The speakers' own
    names are not evidence either way (every summary names them), so they are
    never looked for.
    """

    speakers = prompts.speaker_names(person_name)
    in_summary = {light_stem(word) for word in content_words(summary)}
    given_away = []
    for detail in details:
        if prompts.specifics(detail, speakers):
            repeated = not prompts.unstated([detail], summary, speakers)
        else:
            # Possessives go first, so that "Sam's" is dropped as the speaker's name.
            words = content_words(re.sub(r"['’]s\b", "", detail), drop=speakers)
            stems = {light_stem(word) for word in words}
            repeated = not stems or len(stems & in_summary) * 2 >= len(stems)
        if repeated:
            given_away.append(detail)
    return given_away


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate_corpus(
    people: Sequence[Person],
    conversations: Sequence[Conversation],
    roster: Sequence[RosterEntry] | None = None,
) -> list[str]:
    """Everything wrong with a corpus, as readable problems; empty when it is sound.

    ``roster`` is the cast the corpus was generated for (the committed, full
    one by default): it fixes who exists, how many conversations each person
    has, and how much recent coverage is owed.
    """

    roster = build_roster() if roster is None else list(roster)
    ordered = sorted(conversations, key=lambda conversation: conversation.start_time)
    by_person: dict[str, list[Conversation]] = {entry.person.person_id: [] for entry in roster}
    strangers = 0
    for conversation in ordered:
        if conversation.person_id in by_person:
            by_person[conversation.person_id].append(conversation)
        else:
            strangers += 1

    problems = _people_problems(people, roster)
    total = sum(entry.n_conversations for entry in roster)
    if len(conversations) != total:
        problems.append(f"{len(conversations)} conversations, expected {total}")
    if strangers:
        problems.append(f"{strangers} conversations belong to nobody in the roster")
    if len({conversation.episode_id for conversation in conversations}) != len(conversations):
        problems.append("episode ids are not unique")
    for entry in roster:
        found = len(by_person[entry.person.person_id])
        if found != entry.n_conversations:
            kind = "heavy" if entry.person.is_heavy else "light"
            problems.append(
                f"{entry.person.display_name} ({kind}) has {found} conversations, "
                f"expected {entry.n_conversations}"
            )
    problems += _time_problems(ordered)
    for person in people:
        own = by_person.get(person.person_id, [])
        problems += _storyline_problems(person, own)
        problems += _text_problems(person, own)
        problems += _fact_problems(own)
    starts = [[c.start_time for c in by_person[entry.person.person_id]] for entry in roster]
    return problems + plan.coverage_problems(roster, starts)


def _people_problems(people: Sequence[Person], roster: Sequence[RosterEntry]) -> list[str]:
    problems = []
    if len(people) != len(roster):
        problems.append(f"{len(people)} people, expected {len(roster)}")
    for person, entry in zip(people, roster):
        expected = entry.person
        if (person.person_id, person.display_name, person.is_heavy) != (
            expected.person_id,
            expected.display_name,
            expected.is_heavy,
        ):
            problems.append(f"{person.display_name} is not the roster's {expected.display_name}")
    for person in people:
        if person.display_name != f"{person.first_name} {person.last_name}":
            problems.append(f"display name {person.display_name!r} is not 'First Last'")
        missing = [key for key in PERSON_META_KEYS if key not in person.meta]
        if missing:
            problems.append(f"{person.display_name} lacks meta {', '.join(missing)}")
    first_names = Counter(person.first_name for person in people)
    shared = {name: count for name, count in first_names.items() if count > 1}
    if len(shared) != 1 or list(shared.values()) != [2]:
        problems.append(f"expected one first name shared by two people, found {shared or 'none'}")
    if config.OWNER_FIRST_NAME in first_names:
        problems.append(f"someone shares the owner's first name {config.OWNER_FIRST_NAME}")
    return problems


def _time_problems(ordered: Sequence[Conversation]) -> list[str]:
    """Span, hours, length and overlap problems of conversations sorted by start."""

    problems = []
    for conversation in ordered:
        start, end = conversation.start_time, conversation.end_time
        label = f"conversation {conversation.episode_id} ({start:%Y-%m-%d %H:%M})"
        if not config.DATASET_START <= start < config.NOW or end > config.NOW:
            problems.append(f"{label} is outside the dataset span")
        first_second = start.hour * 3600 + start.minute * 60 + start.second
        last_second = first_second + (end - start).total_seconds()
        if first_second < plan.DAY_START_S or last_second > plan.DAY_END_S:
            problems.append(f"{label} is outside 07:00-22:00")
        if not plan.MIN_DURATION_S <= (end - start).total_seconds() <= plan.MAX_DURATION_S:
            problems.append(f"{label} does not last 3-25 minutes")
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start_time < earlier.end_time:
            problems.append(f"conversations {earlier.episode_id} and {later.episode_id} overlap")
    return problems


def _storyline_problems(person: Person, own: Sequence[Conversation]) -> list[str]:
    """Meta and beat-order problems of one person's conversations (sorted by start)."""

    name = person.display_name
    problems = []
    beats: dict[str, list[int]] = {}
    for conversation in own:
        meta = conversation.meta
        missing = [key for key in CONVERSATION_META_KEYS if key not in meta]
        if missing:
            problems.append(f"{conversation.episode_id} lacks meta {', '.join(missing)}")
            continue
        beats.setdefault(meta["storyline_id"], []).append(meta["beat_index"])
    for storyline_id, indexes in beats.items():
        if indexes != list(range(len(indexes))):
            problems.append(f"{name}: beats of storyline {storyline_id} out of order: {indexes}")

    declared = {s.get("storyline_id"): s for s in person.meta.get("storylines", [])}
    if {key: len(indexes) for key, indexes in beats.items()} != {
        key: storyline.get("n_beats") for key, storyline in declared.items()
    }:
        problems.append(f"{name}: conversations do not match the storylines in the person's meta")
    covered = [storyline.get("shared_topic") for storyline in declared.values()]
    for topic in person.meta.get("shared_topics", []):
        if topic not in covered:
            problems.append(f"{name}: shared topic {topic!r} has no storyline")
    for conversation in own:
        storyline = declared.get(conversation.meta.get("storyline_id"))
        if storyline and conversation.meta.get("shared_topic") != storyline.get("shared_topic"):
            problems.append(f"conversation {conversation.episode_id} has the wrong shared_topic")
    return problems


def _text_problems(person: Person, own: Sequence[Conversation]) -> list[str]:
    """Summary-length and speaker-label problems of one person's conversations."""

    low, high = config.SUMMARY_WORD_RANGE
    speakers = {person.first_name, OWNER_SPEAKER}
    problems = []
    for conversation in own:
        label, turns = conversation.episode_id, conversation.dialogue
        words = prompts.word_count(conversation.summary)
        if not low <= words <= high:
            problems.append(f"summary of {label} has {words} words, not {low}-{high}")
        if len(turns) < prompts.MIN_TURNS:
            problems.append(f"dialogue of {label} has only {len(turns)} turns")
        if {turn.speaker for turn in turns} != speakers:
            problems.append(
                f"dialogue of {label} is not between {person.first_name} and {OWNER_SPEAKER}"
            )
        if any(not turn.text.strip() for turn in turns):
            problems.append(f"dialogue of {label} has an empty turn")
    return problems


def _fact_problems(own: Sequence[Conversation]) -> list[str]:
    """Key facts repeated across one person's conversations, and text-derived meta gone stale.

    A key fact that two conversations share makes a lookup question about
    either ambiguous, unless the later one is a recap written to restate it.
    The unstated and in-summary lists must be what the dialogue and summary
    actually give, since the query writer trusts them without re-reading.
    """

    problems = []
    seen: dict[str, str] = {}
    complete = [c for c in own if all(key in c.meta for key in CONVERSATION_META_KEYS)]
    for conversation in complete:
        if conversation.meta["recap"]:
            continue
        for fact in conversation.meta["key_facts"]:
            earlier = seen.setdefault(fact.casefold(), conversation.episode_id)
            if earlier != conversation.episode_id:
                problems.append(
                    f"conversation {conversation.episode_id} repeats the key fact {fact!r} "
                    f"of {earlier}"
                )
    for conversation in complete:
        meta = conversation.meta
        spoken = " ".join(turn.text for turn in conversation.dialogue)
        speakers = prompts.speaker_names(meta["person_name"])
        derived = {
            "unstated_facts": prompts.unstated(meta["key_facts"], spoken, speakers),
            "unstated_details": prompts.unstated(meta["minor_details"], spoken, speakers),
            "details_in_summary": _in_summary(
                meta["minor_details"], conversation.summary, meta["person_name"]
            ),
        }
        for key, value in derived.items():
            if meta[key] != value:
                problems.append(f"conversation {conversation.episode_id}: meta {key} is stale")
    return problems


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #
def smoke_roster() -> list[RosterEntry]:
    """Two heavy people with ten conversations each and four light people, for checking prompts."""

    return build_roster(n_people=SMOKE_PEOPLE, heavy_counts=SMOKE_HEAVY_COUNTS, total=SMOKE_TOTAL)


def main(argv: Sequence[str] | None = None, runtime: Runtime | None = None) -> int:
    """Generate the corpus; a ``runtime`` is built from the settings unless one is injected."""

    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.data.generate",
        description="Generate the retrieval benchmark's people and conversations.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="generate a six-person roster into the cache directory instead of the committed files",
    )
    parser.add_argument(
        "--workers", type=int, help="parallel LLM calls (default: EVAL_LLM_CONCURRENCY)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too many

    if runtime is None:
        runtime = Runtime()
    roster = smoke_roster() if args.smoke else build_roster()
    out_dir = config.CACHE_DIR / "smoke" if args.smoke else config.DATA_DIR
    try:
        corpus = generate_corpus(runtime.llm, runtime.settings, roster, out_dir, args.workers)
    except ValidationError as error:
        print(f"validation: {len(error.problems)} problems; nothing written to {out_dir}")
        for problem in error.problems[:25]:
            print(f"  - {problem}")
        print(f"hosted spend this run: ${runtime.spend.total:.2f}")
        return 1

    heavy = sum(person.is_heavy for person in corpus.people)
    unstated = sum(
        bool(c.meta["unstated_facts"] or c.meta["unstated_details"]) for c in corpus.conversations
    )
    print(f"people: {len(corpus.people)} ({heavy} heavy)")
    print(f"conversations: {len(corpus.conversations)}")
    print(f"conversations that leave a key fact or minor detail unspoken: {unstated}")
    print(f"written to {out_dir}")
    for namespace, stats in corpus.stats.items():
        print(
            f"  {namespace:<9} {stats.calls:>5} calls, {stats.cached:>5} from cache, "
            f"${stats.cost_usd:.2f} when first made"
        )
    print(f"hosted spend this run: ${runtime.spend.total:.2f}")
    print("validation: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
