"""Prompts for the corpus generator and the checks on what the model returns.

Every builder is a pure function of its arguments. The LLM client caches by
prompt text, so the same roster and the same earlier answers must give
byte-identical prompts on every run: nothing here reads the clock, the locale
or an unseeded random generator.

The outline, beats and dialogue prompts share the owner's bio. The summary
prompt deliberately does not: like the product's summariser, it sees two names
and a transcript and nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from . import Prompt
from .. import config
from ..schema import OWNER_SPEAKER, Turn
from ..timeutil import MONTH_NAMES as _MONTHS
from ..timeutil import WEEKDAY_NAMES as _WEEKDAYS
from .plan import MAX_DURATION_S, MIN_DURATION_S, Slot, Storyline, StorylineBudget, StorylineShape
from .roster import OWNER_BIO, SHARED_TOPICS, UNKNOWN_NAMES, RosterEntry, seeded

MIN_TURNS = 6  # fewer than this is not a usable conversation
TURN_RANGE = (8, 22)  # what the dialogue prompt asks for
SHORT_TURNS = 12  # a conversation asked for fewer turns than this carries fewer specifics
MAX_KEY_FACTS = 4
MAX_MINOR_DETAILS = 3
SHORT_MAX_KEY_FACTS = 2
SHORT_MAX_MINOR_DETAILS = 2
N_THREAD_EXAMPLES = 3  # long-running thread examples shown to each heavy person's outline

# Examples of long-running threads for a heavy person's outline. None of them is
# adjacent to a shared topic, so none nudges the model towards a subject that is
# reserved for other people; each person sees a seeded draw of them.
_THREAD_EXAMPLES: tuple[str, ...] = (
    "a job search",
    "a health scare",
    "a conflict at work",
    "a wedding to plan",
    "a parent moving house",
    "a visa application",
    "a friend's business getting off the ground",
    "a book being written",
    "a tax mess",
    "a dispute with a neighbour",
)

_OWNER = config.OWNER_FIRST_NAME
_PERSONA_LABELS: tuple[tuple[str, str], ...] = (
    ("age", "Age"),
    ("occupation", "Occupation"),
    ("personality", "Personality"),
    ("speech_style", "Speech style"),
    ("background", "Background"),
    ("how_they_know_riley", f"How they know {_OWNER}"),
)
PERSONA_FIELDS: tuple[str, ...] = tuple(name for name, _ in _PERSONA_LABELS)

_SETTING_NOTES = {
    "home": "at home: mornings and evenings on weekdays, any time at weekends",
    "work": "at work, on weekdays",
    "social": (
        "in person, or on a call if they live away from Seattle: "
        "at lunchtime or in the evening on weekdays, during the day at weekends"
    ),
}

_OPENINGS = (
    "Start in the middle of things, already on the subject, with no greeting.",
    "{name} opens by raising the subject directly.",
    f"{_OWNER} opens with a question about how things stand.",
    "Start with a quick hello in {name}'s own style and a line of small talk before the subject.",
    "Start with a remark about what one of them is doing or where they are right now.",
    "{name} opens with a reaction, as if the news has only just arrived.",
)

_STORY_SYSTEM = (
    "You are a writer building a synthetic dataset that tests a personal memory assistant. "
    f"The dataset follows one person, {config.OWNER_DISPLAY_NAME}, through a year of spoken "
    "conversations with the people in their life. Everything reads like ordinary real life: "
    "specific, mundane, internally consistent, never melodramatic. "
    "Reply with one JSON object and nothing else."
)

_DIALOGUE_SYSTEM = (
    "You write realistic spoken conversations for a synthetic dataset that tests a personal "
    f"memory assistant. Each conversation is between {config.OWNER_DISPLAY_NAME} and one person "
    f"in {_OWNER}'s life, written down exactly as it was said. "
    "Reply with one JSON object and nothing else."
)

# Adapted from the product's summariser (backend/app/services/llm_client.py):
# the same role, voice and content rules, with the benchmark's word range in
# place of the sentence count and without the importance score.
_SUMMARY_SYSTEM = (
    f"You are the memory system for a pair of smart glasses worn by {config.OWNER_DISPLAY_NAME}. "
    "You receive raw conversation transcripts and produce concise, factual episode summaries.\n\n"
    "Guidelines:\n"
    "- Summarise the key topics, decisions, and any action items.\n"
    f"- Write between {config.SUMMARY_WORD_RANGE[0]} and {config.SUMMARY_WORD_RANGE[1]} words.\n"
    f"- Write in third person (e.g. '{_OWNER} discussed …'). "
    f"Lines labelled {OWNER_SPEAKER} are spoken by {_OWNER}.\n"
    "- Be objective; do not infer emotions unless explicitly stated.\n"
    '- Reply with one JSON object: {"summary": "<the summary>"}'
)

_NAME_RULE = (
    "Anyone you invent (relatives, friends, colleagues, pets) must not share a first name with "
    f"{_OWNER}, with anyone listed above, or with any of: "
    + ", ".join(name.split()[0] for name in UNKNOWN_NAMES)
    + "."
)


@dataclass(frozen=True)
class StorylineDraft:
    """The title and premise the outline call wrote for one storyline shape."""

    title: str
    premise: str


@dataclass(frozen=True)
class Beat:
    """What one conversation adds to its storyline."""

    what_happens: str
    key_facts: tuple[str, ...]  # the dialogue states these; lookup questions ask about them
    minor_details: tuple[str, ...]  # said in passing; a short summary leaves them out
    recap: bool = False  # padding written in code: restates the beat before it, adds nothing


class ResponseError(ValueError):
    """The model's JSON does not have the shape or content the prompt asked for."""


# --------------------------------------------------------------------------- #
# Outline: persona and storylines                                              #
# --------------------------------------------------------------------------- #
def outline_prompt(
    roster: Sequence[RosterEntry],
    entry: RosterEntry,
    budget: StorylineBudget,
    shapes: Sequence[StorylineShape],
) -> Prompt:
    """Ask for a persona and a title and premise for each of one person's storyline shapes.

    The shapes come from the schedule, so the prompt can state how many
    conversations each storyline takes and the dates it runs between, and a
    premise fits its own stretch of the year rather than the whole of it.
    """

    person, n = entry.person, entry.n_conversations
    first, last = min(shape.first for shape in shapes), max(shape.last for shape in shapes)
    rows = [f"Storyline {number}: {_row(shape)}" for number, shape in enumerate(shapes, start=1)]
    rules = []
    if person.is_heavy:
        examples = seeded(f"outline:examples:{person.person_id}").sample(
            _THREAD_EXAMPLES, N_THREAD_EXAMPLES
        )
        rules.append(
            "- A storyline with many conversations over months is a long-running thread that "
            f"develops ({', '.join(examples)}); one with a few conversations is a small, "
            "concrete, low-stakes subject (weekend plans, a show they both watch)."
        )
    else:
        rules.append("- With this few conversations, keep each storyline small and concrete.")
    if budget.close_pairs:
        rules.append(
            f"- Include at least {_plural(budget.close_pairs, 'pair')} of "
            "storylines that are deliberately close in subject but different in substance (two "
            "different work projects, two different trips), so that a short summary of a "
            "conversation from one could be mistaken for the other."
        )
    if entry.shared_topics:
        rules.append(
            "- Shared subjects. A storyline marked with a shared subject is about exactly that "
            f"subject. Other people talk to {_OWNER} about the same subjects, so make it "
            f"particular to {person.first_name}: their own experience of the subject, or "
            "something the two of them do together."
        )
    rules += [
        f"- Reserved subjects. These belong to {_OWNER}'s conversations with other people. "
        "Do not build a storyline around any of them or make one a recurring element: "
        + _quoted([topic for topic in SHARED_TOPICS if topic not in entry.shared_topics])
        + ".",
        "- Give each storyline a short, distinctive title and a premise of 2-3 sentences: who and "
        "what is involved, what is at stake, and how it develops from its first conversation to "
        "its last. Fit each storyline's own dates and the time of year they fall in.",
        f"- {_NAME_RULE}",
    ]
    user = "\n".join(
        [
            _background(),
            "",
            f"Plan {_OWNER}'s conversations with one person.",
            "",
            f"Person: {person.display_name}",
            f"Relationship to {_OWNER}: {person.relationship}",
            f"Conversations: {n}, {_between(first, last)}",
            f"They talk {_SETTING_NOTES[entry.setting]}.",
            "",
            _circle(roster, entry),
            "",
            "1. Persona",
            "Invent a believable, specific person: age, occupation, personality, speech_style "
            "(how they actually talk: pace, humour, verbal habits), background (where they live "
            "and the recurring people, places and things in their life, with names, so that "
            "later conversations stay consistent) and how_they_know_riley. Stay consistent with "
            f"{_OWNER}'s background and the relationship.",
            "",
            "2. Storylines",
            "A storyline is one subject that runs through several conversations; each "
            "conversation advances exactly one storyline by one beat. How many storylines there "
            "are, how many conversations each takes and when those happen is already fixed:",
            *rows,
            "",
            "Write a title and a premise for every storyline listed, by its number.",
            *rules,
            "",
            "Return JSON in exactly this shape:",
            '{"persona": {"age": 34, "occupation": "...", "personality": "...", '
            '"speech_style": "...", "background": "...", "how_they_know_riley": "..."}, '
            '"storylines": [{"storyline": 1, "title": "...", "premise": "..."}]}',
        ]
    )
    return Prompt(_STORY_SYSTEM, user)


def parse_outline(
    data: Mapping[str, Any], entry: RosterEntry, shapes: Sequence[StorylineShape]
) -> tuple[dict[str, str], list[StorylineDraft]]:
    """The persona and one storyline draft per shape, in shape order.

    Rejects a missing persona field, a storyline list that does not fill the
    shapes exactly once each, a repeated title, and a storyline that names a
    subject reserved for other people.
    """

    raw_persona = data.get("persona")
    if not isinstance(raw_persona, Mapping):
        raise ResponseError("the reply has no persona object")
    persona = {name: _clean(raw_persona.get(name)) for name in PERSONA_FIELDS}
    missing = [name for name, value in persona.items() if not value]
    if missing:
        raise ResponseError(f"the persona lacks {', '.join(missing)}")

    raw_storylines = data.get("storylines")
    if not isinstance(raw_storylines, list):
        raise ResponseError("the reply has no storylines list")
    numbered: dict[int, StorylineDraft] = {}
    for item in raw_storylines:
        if not isinstance(item, Mapping):
            raise ResponseError("a storyline is not an object")
        title, premise = _clean(item.get("title")), _clean(item.get("premise"))
        if not title or not premise:
            raise ResponseError("a storyline lacks a title or a premise")
        try:
            number = int(item.get("storyline"))
        except (TypeError, ValueError):
            raise ResponseError(f"storyline {title!r} has no number") from None
        if number in numbered:
            raise ResponseError(f"two storylines are numbered {number}")
        numbered[number] = StorylineDraft(title, premise)

    expected = set(range(1, len(shapes) + 1))
    if set(numbered) != expected:
        raise ResponseError(
            f"storylines numbered {sorted(numbered)}, expected 1 to {len(shapes)}"
        )
    drafts = [numbered[number] for number in sorted(numbered)]
    if len({draft.title.casefold() for draft in drafts}) < len(drafts):
        raise ResponseError("two storylines have the same title")
    reserved = [topic for topic in SHARED_TOPICS if topic not in entry.shared_topics]
    for draft in drafts:
        topic = reserved_subject(f"{draft.title} {draft.premise}", reserved)
        if topic:
            raise ResponseError(
                f"storyline {draft.title!r} is about the reserved subject {topic!r}"
            )
    return persona, drafts


def reserved_subject(text: str, reserved: Sequence[str]) -> str | None:
    """The first of ``reserved`` that ``text`` names (its words in order, article dropped)."""

    words = re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()
    flat = " ".join(words)
    for topic in reserved:
        phrase = _bare(topic).replace("-", " ")
        if re.search(r"(?<![a-z0-9])" + re.escape(phrase), flat):
            return topic
    return None


# --------------------------------------------------------------------------- #
# Beats                                                                        #
# --------------------------------------------------------------------------- #
def beats_prompt(
    roster: Sequence[RosterEntry],
    entry: RosterEntry,
    persona: Mapping[str, str],
    storylines: Sequence[Storyline],
    index: int,
    slots: Sequence[Slot],
) -> Prompt:
    """Ask for the beats of ``storylines[index]``, one per conversation in ``slots``.

    Each conversation is listed with its date and how long it will be, so a
    short conversation is not asked to carry as many specifics as a long one.
    """

    storyline, name = storylines[index], entry.person.first_name
    n = len(slots)
    others = [
        f'- "{other.title}": {other.premise}'
        for position, other in enumerate(storylines)
        if position != index
    ]
    context = []
    if others:
        context = [
            f"{_OWNER}'s other storylines with {name} (context only: stay consistent with them, "
            "keep this one distinct, and do not develop them here):",
            *others,
            "",
        ]
    user = "\n".join(
        [
            _background(),
            "",
            f"Person: {entry.person.display_name}",
            f"Relationship to {_OWNER}: {entry.person.relationship}",
            _persona(persona),
            "",
            _circle(roster, entry),
            "",
            f'Storyline to write: "{storyline.title}"',
            f"Premise: {storyline.premise}",
            *([f"Shared subject: {storyline.shared_topic}"] if storyline.shared_topic else []),
            "",
            *context,
            f"This storyline comes up in {_plural(n, 'conversation')}:",
            *[
                f"{number}. {_date(start)}, {_part_of_day(start)}, "
                f"{'short, ' if is_short(start, end) else ''}about "
                f"{_plural(_turn_target(start, end), 'turn')}"
                for number, (start, end) in enumerate(slots, start=1)
            ],
            "",
            f"Write exactly {_plural(n, 'beat')}, one per conversation, in the same order. "
            "A beat is what is new in that conversation:",
            f"- what_happens: 1-2 sentences on the new development {name} and {_OWNER} discuss.",
            f"- key_facts: 2 to {MAX_KEY_FACTS} concrete facts (names, numbers, places, dates) "
            f"that the conversation must state; exactly {SHORT_MAX_KEY_FACTS} for a conversation "
            "marked short.",
            f"- minor_details: 2 to {MAX_MINOR_DETAILS} small specifics mentioned only in passing "
            "(a price, a street, a dish, a time of day), the kind a short summary would leave "
            f"out; 1 to {SHORT_MAX_MINOR_DETAILS} for a conversation marked short.",
            "",
            "Rules:",
            "- Facts are specific, consistent with each other along the storyline, and consistent "
            "with the conversation dates: something planned for a date is discussed before that "
            "date, and its outcome after it. The dates listed are fixed; where the premise "
            "implies a season or a month that does not match them, follow the dates.",
            "- Every beat moves the storyline forward. No two beats share a key fact or a minor "
            "detail, so that a question about one conversation cannot be answered from another.",
            "- A minor detail never repeats or paraphrases a key fact.",
            "- Stay inside this storyline and its premise.",
            f"- {_NAME_RULE}",
            "",
            "Return JSON in exactly this shape:",
            '{"beats": [{"what_happens": "...", "key_facts": ["...", "..."], '
            '"minor_details": ["...", "..."]}]}',
        ]
    )
    return Prompt(_STORY_SYSTEM, user)


def parse_beats(data: Mapping[str, Any], short: Sequence[bool] = ()) -> list[Beat]:
    """The beats in a reply, in order. How many there should be is the caller's check.

    ``short[i]`` caps beat ``i`` at the short conversation's number of facts and
    details. Two beats that share a key fact or a minor detail are rejected,
    because a question about one conversation could then be answered from the other.
    """

    raw = data.get("beats")
    if not isinstance(raw, list) or not raw:
        raise ResponseError("the reply has no beats list")
    beats = []
    seen: dict[str, int] = {}  # a fact or detail -> the 1-based beat that first states it
    for number, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ResponseError("a beat is not an object")
        brief = number < len(short) and short[number]
        max_facts = SHORT_MAX_KEY_FACTS if brief else MAX_KEY_FACTS
        max_details = SHORT_MAX_MINOR_DETAILS if brief else MAX_MINOR_DETAILS
        what_happens = _clean(item.get("what_happens"))
        key_facts = _strings(item.get("key_facts"))[:max_facts]
        if not what_happens or not key_facts:
            raise ResponseError("a beat lacks what_happens or key_facts")
        minor_details = _strings(item.get("minor_details"))[:max_details]
        for text in key_facts + minor_details:
            earlier = seen.setdefault(text.casefold(), number + 1)
            if earlier != number + 1:
                raise ResponseError(f"beats {earlier} and {number + 1} share {text!r}")
        beats.append(Beat(what_happens, tuple(key_facts), tuple(minor_details)))
    return beats


def recap_beat(last: Beat) -> Beat:
    """A beat without a new development, for padding a storyline the model left short.

    The two of them go back over where things stand, so the conversation
    restates the previous beat's facts instead of inventing new ones. The
    record is marked as a recap so the query writer can tell it from a real beat.
    """

    return Beat(
        what_happens=(
            "They have nothing new to report and go back over where things stand. "
            f"Last time: {last.what_happens}"
        ),
        key_facts=last.key_facts,
        minor_details=(),
        recap=True,
    )


# --------------------------------------------------------------------------- #
# Dialogue                                                                     #
# --------------------------------------------------------------------------- #
def dialogue_prompt(
    entry: RosterEntry,
    persona: Mapping[str, str],
    storyline: Storyline,
    beat: Beat,
    earlier: Sequence[tuple[datetime, Beat]],
    episode_id: str,
    start: datetime,
    end: datetime,
) -> Prompt:
    """Ask for one conversation: ``beat``, after the ``earlier`` beats of its storyline."""

    name = entry.person.first_name
    opening = seeded(f"dialogue:opening:{episode_id}").choice(_OPENINGS).format(name=name)
    recap = [
        f"- {_date(moment)}: {past.what_happens} Facts stated: {'; '.join(past.key_facts)}"
        for moment, past in earlier
    ] or ["- None: this is the first conversation of the storyline."]
    low, high = TURN_RANGE
    user = "\n".join(
        [
            f"{_OWNER}'s background (for consistency only; do not bring it up unless the notes "
            "below call for it):",
            OWNER_BIO,
            "",
            f"The other speaker: {entry.person.display_name}",
            f"Relationship to {_OWNER}: {entry.person.relationship}",
            _persona(persona),
            "",
            f"When: {_date(start)}, {start:%H:%M} ({_part_of_day(start)})",
            f"They talk {_SETTING_NOTES[entry.setting]}.",
            "",
            f'Storyline: "{storyline.title}"',
            f"Premise: {storyline.premise}",
            "",
            "Earlier conversations in this storyline, oldest first:",
            *recap,
            "",
            "This conversation",
            f"What happens: {beat.what_happens}",
            "Key facts:",
            *[f"- {fact}" for fact in beat.key_facts],
            "Minor details:",
            *([f"- {detail}" for detail in beat.minor_details] or ["- None."]),
            "",
            "Write the conversation:",
            f"- {low} to {high} turns of natural spoken conversation, about "
            f"{_turn_target(start, end)} this time: contractions, interruptions, half-finished "
            "thoughts, each person in their own voice.",
            "- Every key fact and every minor detail above is said out loud by one of them, with "
            "its exact names, numbers, places and dates. The key facts are what the conversation "
            "is about; the minor details come up in passing.",
            "- Write numbers the way a transcript does, in digits: $480, 2 hours and 34 minutes, "
            "November 12, 7:30.",
            "- The premise describes the whole storyline. Nothing in it that comes after this "
            "conversation has happened yet: do not mention, anticipate or hint at it.",
            "- They refer back to earlier developments the way real people do, using only what "
            "the earlier conversations above establish. Do not invent new facts about earlier "
            "events.",
            "- Stay with this development. A little small talk is fine, but bring in no other "
            "news, plans or named people beyond the notes above.",
            f"- Opening: {opening}",
            "- Spoken words only: no narration, stage directions, sound effects or descriptions "
            "of actions. Nobody mentions glasses, a recording or a transcript.",
            "- Fit the date, the weekday and the time of day.",
            f'- {_OWNER}\'s lines are labelled "{OWNER_SPEAKER}" and {name}\'s lines are labelled '
            f'"{name}". When someone quotes another person, use single quotation marks.',
            "",
            "Return JSON in exactly this shape:",
            f'{{"turns": [{{"speaker": "{name}", "text": "..."}}, '
            f'{{"speaker": "{OWNER_SPEAKER}", "text": "..."}}]}}',
        ]
    )
    return Prompt(_DIALOGUE_SYSTEM, user)


def parse_dialogue(data: Mapping[str, Any], first_name: str) -> list[Turn]:
    """The turns in a dialogue reply: both speakers present, correctly labelled, nothing empty."""

    raw = data.get("turns")
    if not isinstance(raw, list) or len(raw) < MIN_TURNS:
        raise ResponseError(f"fewer than {MIN_TURNS} turns")
    turns = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ResponseError("a turn is not an object")
        speaker, text = _clean(item.get("speaker")), _clean(item.get("text"))
        if speaker not in (first_name, OWNER_SPEAKER):
            raise ResponseError(f"unexpected speaker label {speaker!r}")
        if not text:
            raise ResponseError("a turn has no text")
        turns.append(Turn(speaker=speaker, text=text))
    if {turn.speaker for turn in turns} != {first_name, OWNER_SPEAKER}:
        raise ResponseError("only one of the two speakers talks")
    return turns


# Words that open a fact with a capital only because they open its sentence
# ("Their last game was ...", "During their April practice ..."). A fact's
# first word is not a specific when it is one of these, or a gerund.
_SENTENCE_OPENERS = frozenset(
    """
    a about after all also although an and another any around as at back because before
    between both but by despite during each earlier eight either every few five for four
    from he her here his how i if in instead it its last later many most my neither next
    nine no none now on once one other our over seven several she since six so some ten
    that the their then there these they this those though three through to two under
    until we what when where which while who why with without you your
    """.split()
)


def speaker_names(person_name: str) -> frozenset[str]:
    """The name tokens of a conversation's two speakers, for the ``ignore`` of the checks below.

    The facts are written in the third person ("Maya booked ...", "Riley is
    making toast"), but nobody says their own name or the listener's out loud:
    Maya says "I booked ...". A speaker's name in a fact is therefore not
    something the dialogue can be expected to contain.
    """

    words = f"{config.OWNER_DISPLAY_NAME} {person_name}".split()
    return frozenset(filter(None, (_normalise(word) for word in words)))


def specifics(text: str, ignore: Collection[str] = ()) -> list[str]:
    """The checkable tokens of a fact or detail: its words that carry a digit or a capital.

    Normalised to lowercase letters and digits, possessive dropped, so that
    "Dana's", "$18,400" and "12th" become "dana", "18400" and "12th". Tokens in
    ``ignore`` (see :func:`speaker_names`) are left out, and so is a first word
    that is capitalised only for opening the sentence.
    """

    found = []
    for position, word in enumerate(text.split()):
        bare = re.sub(r"^[^A-Za-z0-9]+", "", word)
        has_digit = any(ch.isdigit() for ch in bare)
        if not (has_digit or bare[:1].isupper()):
            continue
        token = _normalise(word)
        if not token or token in ignore:
            continue
        opener = token in _SENTENCE_OPENERS or token.endswith("ing")
        if position == 0 and opener and not has_digit:
            continue
        found.append(token)
    return found


def unstated(facts: Sequence[str], spoken: str, ignore: Collection[str] = ()) -> list[str]:
    """The facts whose specifics are not all found in ``spoken``, word-initial and case-blind.

    A cheap check that the dialogue states what the beat asked for (and that
    a summary gives away a detail). A fact without a name or a number has
    nothing to check and counts as stated; ``ignore`` holds the speakers' name
    tokens, which are never looked for.
    """

    text = " ".join(filter(None, (_normalise(word) for word in spoken.split())))
    missing = []
    for fact in facts:
        tokens = specifics(fact, ignore)
        if not all(re.search(r"(?<![a-z0-9])" + re.escape(token), text) for token in tokens):
            missing.append(fact)
    return missing


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #
def summary_prompt(person_name: str, transcript: str, previous_words: int | None = None) -> Prompt:
    """Ask for the 20-100 word summary of one transcript.

    ``previous_words`` is the length of an earlier attempt that missed the
    range; stating it changes the prompt, which is what makes the re-ask come
    out different rather than a change of sampling.
    """

    user = f"Conversation between {config.OWNER_DISPLAY_NAME} and {person_name}:\n\n{transcript}"
    if previous_words is not None:
        low, high = config.SUMMARY_WORD_RANGE
        user += (
            f"\n\nYour previous summary of this conversation was {previous_words} words long. "
            f"Write one that is between {low} and {high} words."
        )
    return Prompt(_SUMMARY_SYSTEM, user)


def parse_summary(data: Mapping[str, Any]) -> str:
    """The summary text in a reply, untouched apart from surrounding whitespace."""

    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ResponseError("the reply has no summary text")
    return summary.strip()


def word_count(text: str) -> int:
    """Words as the summary range counts them: whitespace-separated tokens."""

    return len(text.split())


# --------------------------------------------------------------------------- #
# Shared pieces                                                                #
# --------------------------------------------------------------------------- #
def _background() -> str:
    return f"{_OWNER}'s background (fixed; the same in every part of the dataset):\n{OWNER_BIO}"


def _circle(roster: Sequence[RosterEntry], entry: RosterEntry) -> str:
    """The rest of the cast, so that invented details do not contradict people who exist."""

    lines = [
        f"- {other.person.display_name}: {other.person.relationship}"
        for other in roster
        if other.person.person_id != entry.person.person_id
    ]
    return "\n".join(
        [
            f"Other people in {_OWNER}'s life. They exist, so stay consistent with them, but keep "
            "them in the background: do not make them central to a storyline or invent major "
            "news about them.",
            *lines,
        ]
    )


def _persona(persona: Mapping[str, str]) -> str:
    return "\n".join(f"{label}: {persona[name]}" for name, label in _PERSONA_LABELS)


def _date(moment: datetime) -> str:
    """'Wednesday 4 November 2026', spelled out by hand so the locale cannot change a prompt."""

    return f"{_WEEKDAYS[moment.weekday()]} {moment.day} {_MONTHS[moment.month - 1]} {moment.year}"


def _between(first: datetime, last: datetime) -> str:
    """'on Monday 4 May 2026' when both fall on one day, else 'between ... and ...'."""

    if first.date() == last.date():
        return f"on {_date(first)}"
    return f"between {_date(first)} and {_date(last)}"


def _row(shape: StorylineShape) -> str:
    """One storyline shape as the outline lists it: its size, dates and shared subject."""

    row = f"{_plural(shape.n_beats, 'conversation')} {_between(shape.first, shape.last)}"
    if shape.shared_topic:
        row += f', shared subject: "{shape.shared_topic}"'
    return row


def _part_of_day(moment: datetime) -> str:
    return "morning" if moment.hour < 12 else "afternoon" if moment.hour < 17 else "evening"


def _turn_target(start: datetime, end: datetime) -> int:
    """A turn count in ``TURN_RANGE`` that grows with the conversation's planned length."""

    length = (end - start).total_seconds()
    share = (length - MIN_DURATION_S) / (MAX_DURATION_S - MIN_DURATION_S)
    low, high = TURN_RANGE
    return low + round(min(max(share, 0.0), 1.0) * (high - low))


def is_short(start: datetime, end: datetime) -> bool:
    """Whether a conversation is asked for too few turns to carry the full number of specifics."""

    return _turn_target(start, end) < SHORT_TURNS


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _count(low: int, high: int, noun: str) -> str:
    """'exactly 1 storyline' or '14 to 20 storylines'."""

    return f"exactly {_plural(low, noun)}" if low == high else f"{low} to {high} {noun}s"


def _quoted(items: Sequence[str]) -> str:
    return ", ".join(f'"{item}"' for item in items)


def _clean(value: Any) -> str:
    """Model text on one line: surrounding and repeated whitespace removed."""

    return "" if value is None else " ".join(str(value).split())


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for text in (_clean(item) for item in value) if text]


def _normalise(word: str) -> str:
    """Lowercase letters and digits of one word, with a possessive 's dropped."""

    return re.sub(r"[^a-z0-9]", "", re.sub(r"['’]s(?![a-z0-9])", "", word.casefold()))


def _bare(topic: str) -> str:
    """A shared topic in lowercase without its leading article: 'a ski trip' -> 'ski trip'."""

    words = topic.casefold().split()
    return " ".join(words[1:] if words and words[0] in ("a", "an", "the") else words)
