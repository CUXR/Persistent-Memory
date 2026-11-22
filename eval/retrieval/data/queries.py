"""Generate the benchmark's queries: ``python -m eval.retrieval.data.queries``.

For every query a seeded draw fixes in code what the query is about: its type
and sub-kind, the source conversations, the name it must contain, the time
phrase. A writer model then phrases the question and the gold answer, and
validators in code plus a checker model decide whether the candidate stands. A
rejected candidate is dropped and its slot draws the next target, so the output
depends only on the corpus, the seed and the (cached) model responses.

The gold labels are what the draw fixed, never what a model said: gold episode
ids are the sources, the gold name is the name the question was required to
contain, and the gold time range is the range of the required time phrase.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from .. import config, dataset
from ..interfaces import ChatLLM, LLMResult, Usage
from ..schema import ANSWERABLE_TYPES, POOL_BUCKETS, QUERY_TYPES, Conversation, Person, Query, parse_dt, pool_bucket
from ..system.resolver import Resolution, resolve_name
from ..timeutil import MONTH_NAMES, WEEKDAY_NAMES, TimeExpression, pad_time_range, time_expressions
from . import query_prompts as prompts

if TYPE_CHECKING:
    from ..runtime import Runtime

logger = logging.getLogger(__name__)

T = TypeVar("T")

WRITER_NAMESPACE = "query_write"
CHECKER_NAMESPACE = "query_check"
WRITER_TEMPERATURE = 0.7
CHECKER_TEMPERATURE = 0.0
MAX_TOKENS = 1024

MAX_SOURCE_USES = 2  # queries one conversation may be a source for, over all types
ATTEMPTS_PER_QUERY = 5  # a type's attempt budget is this times the number of slots it fills
MAX_OTHERS = 6  # other summaries shown as "must not be answerable from these"
CROSS_RIVALS = 3  # of those, conversations with other people on the same shared topic
MAX_THREAD_DIALOGUES = 6  # dialogues of one storyline shown to the missing-detail writer
WRITER_SUMMARY_CAP = 60  # summaries of one person shown to the unanswerable writer
MULTI_SIZES = (2, 3)  # consecutive beats behind a multi-conversation query
GROUP_SIZES = (2, 3)  # people behind a cross-person group query
RECAP_MAX_CONVERSATIONS = 3  # a time window holding more than this gets a "specific" query
PAIR_SHARE = 1 / 3  # share of "specific" temporal queries built on two conversations, where possible
MAX_SHARED_WORDS = 1  # content words a paraphrase query may share with the summary
MAX_QUESTION_WORDS = 35
MAX_ANSWER_SENTENCES = 3
MAX_ANSWER_WORDS = 90

SMALL_POOL, MID_POOL = POOL_BUCKETS[0], POOL_BUCKETS[1]
EVERYONE = "everyone"  # scope id of a time window that is not tied to a person

# Most constrained first: a temporal query needs specific recent conversations
# and a multi-conversation query needs consecutive beats, so they draw before
# the single-conversation types use sources up.
GENERATION_ORDER: tuple[str, ...] = (
    "temporal",
    "multi_conversation",
    "cross_person",
    "detail_only",
    "paraphrase",
    "lookup",
    "unanswerable",
    "unknown_name",
)


class QueryGenerationError(RuntimeError):
    """A query type could not be filled from this corpus."""


# --------------------------------------------------------------------------- #
# Text checks                                                                  #
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are aren as at be because been before being below
    between both but by can could couldn did didn do does doesn doing don down during each else ever few
    for from further had hadn has hasn have haven having he her here hers herself him himself his how i if
    in into is isn it its itself just ll me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own re same she should shouldn so some such than that the their
    theirs them themselves then there these they this those through to too under until up ve very was
    wasn we were weren what when where which while who whom whose why will with won would wouldn you
    your yours yourself
    say says said saying tell tells told telling talk talks talked talking mention mentions mentioned
    mentioning discuss discusses discussed discussing ask asks asked asking speak speaks spoke
    thing things something anything someone anyone one ones much many get gets got
    """.split()
)
_CONFIRMATION_WORDS = frozenset("yes yeah yep nope sure indeed okay ok right correct true false".split())
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_SOURCE_WORDS_RE = re.compile(r"(?<!\w)(?:according to|summar(?:y|ies)|transcripts?|glasses)(?!\w)", re.IGNORECASE)
# Anything a rule-following extractor could turn into a time filter. Over-matching
# ("the earlier plan") only costs a redraw; under-matching leaves a filter in a
# query whose gold labels do not expect one.
_RELATIVE_TIME_RE = re.compile(
    r"(?<!\w)(?:today|tonight|yesterday|tomorrow|ago|recently|lately|earlier|nowadays"
    r"|(?:this|last|next|past|previous|coming)\s+"
    r"(?:morning|afternoon|evening|night|week|weekend|month|year|spring|summer|autumn|fall|winter"
    r"|few\s+\w+|couple\s+of\s+\w+|\d+\s+\w+)"
    r"|last\s+time"
    r"|the\s+other\s+(?:day|night|week)"
    r"|(?:a\s+few|a\s+couple\s+of|several|\d+)\s+\w+\s+(?:back|earlier|before)"
    r"|(?:over|on|at|during)\s+the\s+(?:weekend|holidays|break|summer|winter|spring|autumn|fall)"
    r"|(?:end|start|beginning|middle)\s+of\s+(?:the|this|last)\s+(?:week|weekend|month|year)"
    r"|" + "|".join(name.lower() for name in WEEKDAY_NAMES) + r")(?!\w)",
    re.IGNORECASE,
)
# Month names only when capitalized ("may" and "march" are also plain words).
_CALENDAR_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(MONTH_NAMES) + r")(?!\w)|(?<!\d)(?:19|20)\d{2}(?!\d)|(?<!\d)\d{1,2}/\d{1,2}(?!\d)"
)
# A question that opens with an auxiliary asks for confirmation, not for information.
_AUXILIARY_RE = re.compile(
    r"^(?:did|do|does|is|was|are|were|am|has|have|had|can|could|will|would|should|shall|may|might|must)(?!\w)",
    re.IGNORECASE,
)


def light_stem(word: str) -> str:
    """Reduce a lowercase word to a crude stem, so that inflections of one word compare equal.

    Plural and verb endings go, then a doubled final consonant and a final
    "e" or "y" are normalized: "planned", "plans" and "plan" all give "plan",
    "moved" and "move" give "mov". Deliberately light, so that it does not join
    words with different roots.
    """

    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    if len(word) > 3 and word[-1] in "ey":
        word = word[:-1] + ("i" if word[-1] == "y" else "")
    return word


def content_words(text: str, drop: Collection[str] = ()) -> list[str]:
    """The lowercase words of ``text`` that carry content: no stopwords, no single letters, none of ``drop``."""

    return [
        word
        for word in _WORD_RE.findall(text.lower())
        if len(word) > 1 and word not in _STOPWORDS and word not in drop
    ]


def person_words(person: Person | None) -> frozenset[str]:
    """The lowercase words of a person's names, which never count as shared wording."""

    if person is None:
        return frozenset()
    return frozenset(_WORD_RE.findall(f"{person.first_name} {person.last_name} {person.display_name}".lower()))


def shared_content_words(question: str, summary: str, drop: Collection[str] = ()) -> set[str]:
    """Stems of the question's content words that also occur in the summary."""

    in_summary = {light_stem(word) for word in _WORD_RE.findall(summary.lower())}
    return {light_stem(word) for word in content_words(question, drop)} & in_summary


def details_missing_from_summary(conversation: Conversation) -> list[str]:
    """The conversation's minor details whose wording is mostly absent from its summary."""

    in_summary = {light_stem(word) for word in _WORD_RE.findall(conversation.summary.lower())}
    missing = []
    for detail in conversation.meta.get("minor_details") or ():
        stems = {light_stem(word) for word in content_words(str(detail))}
        if stems and len(stems & in_summary) * 2 < len(stems):
            missing.append(str(detail))
    return missing


def details_left_out(conversation: Conversation) -> list[str]:
    """The minor details a detail-only question may ask for: said in the dialogue, left out of the summary.

    The corpus generator records which details the summary repeats
    (``details_in_summary``) and which the dialogue never states after all
    (``unstated_details``), checked against the texts when the corpus is
    validated; a conversation without that record falls back to comparing
    each detail's wording with the summary.
    """

    meta = conversation.meta
    if "details_in_summary" not in meta:
        return details_missing_from_summary(conversation)
    skip = {str(detail) for key in ("details_in_summary", "unstated_details") for detail in meta.get(key) or ()}
    return [str(detail) for detail in meta.get("minor_details") or () if str(detail) not in skip]


def time_words(text: str) -> list[str]:
    """Dates and relative-time words in ``text`` (what an extractor could read as a time filter)."""

    return [match.group(0) for pattern in (_RELATIVE_TIME_RE, _CALENDAR_RE) for match in pattern.finditer(text)]


def question_carries_answer(question: str, answer: str, drop: Collection[str] = ()) -> bool:
    """True when every content word of the answer already occurs in the question.

    Such a pair is a yes/no question in disguise ("Is the locker code 4471?" /
    "Yes, it is 4471."): the retrieval query carries the answer and the answer
    judge has nothing to grade. Numbers count as content whatever their
    length; words of confirmation do not.
    """

    asked = {light_stem(word) for word in content_words(question, drop)}
    stated = [
        word
        for word in _WORD_RE.findall(answer.lower())
        if word not in _STOPWORDS
        and word not in _CONFIRMATION_WORDS
        and word not in drop
        and (len(word) > 1 or word.isdigit())
    ]
    return all(light_stem(word) in asked for word in stated)


def _phrase_re(phrase: str, ignore_case: bool = False) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE if ignore_case else 0)


def contains_phrase(text: str, phrase: str, *, ignore_case: bool = False) -> bool:
    """True when ``phrase`` occurs in ``text`` as whole words."""

    return _phrase_re(phrase, ignore_case).search(text) is not None


def is_possessive(text: str, phrase: str) -> bool:
    """True when ``phrase`` occurs in ``text`` as a possessive ("today's booking"), which dates a thing, not the talk."""

    return re.search(rf"(?<!\w){re.escape(phrase)}['’]s(?!\w)", text, re.IGNORECASE) is not None


def _without_phrase(text: str, phrase: str, *, ignore_case: bool = False) -> str:
    return _phrase_re(phrase, ignore_case).sub(" ", text)


def _names_re(names: Iterable[str]) -> re.Pattern[str]:
    """Match any of the names as whole words, case-sensitively, the longest name first."""

    ordered = sorted({name.strip() for name in names if name.strip()}, key=lambda name: (-len(name), name))
    if not ordered:
        return re.compile(r"(?!)")
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(name) for name in ordered) + r")(?!\w)")


def _normal(text: str) -> str:
    """A question reduced to what makes two questions the same question."""

    return " ".join(text.lower().split())


# --------------------------------------------------------------------------- #
# Corpus index                                                                 #
# --------------------------------------------------------------------------- #
def _chronological(conversation: Conversation) -> tuple[datetime, str]:
    return (conversation.start_time, conversation.episode_id)


def _beat_order(conversation: Conversation) -> tuple[int, datetime, str]:
    return (int(conversation.meta.get("beat_index") or 0), conversation.start_time, conversation.episode_id)


def storyline_key(conversation: Conversation) -> tuple[str, str]:
    """(person id, storyline id); a conversation without a storyline is a storyline of its own."""

    return (conversation.person_id, str(conversation.meta.get("storyline_id") or conversation.episode_id))


def name_forms(people: Sequence[Person]) -> dict[str, str]:
    """How a query names each person: the first name when nobody shares it, else the display name."""

    holders = Counter(person.first_name.strip().casefold() for person in people)
    return {
        person.person_id: person.first_name if holders[person.first_name.strip().casefold()] == 1 else person.display_name
        for person in people
    }


def query_expressions(now: datetime = config.NOW) -> list[TimeExpression]:
    """The time expressions a query may carry: ``time_expressions`` less the bare name of the current month.

    Asked on 4 November, "in November" reads as this month to most people and
    models, while the shared rule resolves it to last year's November; the
    weekday rule already leaves out today's weekday for the same reason.
    """

    current = f"in_{MONTH_NAMES[parse_dt(now).month - 1].lower()}"
    return [expression for expression in time_expressions(now) if expression.key != current]


@dataclass(frozen=True)
class TimeWindow:
    """One time expression applied to one scope: a person, or everyone when ``person`` is None."""

    expression: TimeExpression
    person: Person | None
    inside: tuple[Conversation, ...]  # the scope's conversations that start inside the range
    pool_size: int  # the scope's conversations that start inside the padded range

    @property
    def scope_id(self) -> str:
        return self.person.person_id if self.person is not None else EVERYONE

    @property
    def subkind(self) -> str:
        return "recap" if len(self.inside) <= RECAP_MAX_CONVERSATIONS else "specific"


class Corpus:
    """People and conversations indexed the ways the target draws need them.

    Every collection has a fixed order (people by display name, conversations
    by start time), so a seeded choice from one of them is reproducible.
    """

    def __init__(self, people: Sequence[Person], conversations: Sequence[Conversation]) -> None:
        self.people: tuple[Person, ...] = tuple(sorted(people, key=lambda p: (p.display_name, p.person_id)))
        self.person: dict[str, Person] = {person.person_id: person for person in self.people}
        self.conversations: tuple[Conversation, ...] = tuple(sorted(conversations, key=_chronological))
        self.conversation: dict[str, Conversation] = {c.episode_id: c for c in self.conversations}
        by_person: dict[str, list[Conversation]] = defaultdict(list)
        storylines: dict[tuple[str, str], list[Conversation]] = defaultdict(list)
        by_topic: dict[str, list[Conversation]] = defaultdict(list)
        for conversation in self.conversations:
            if conversation.person_id not in self.person:
                raise ValueError(
                    f"conversation {conversation.episode_id} belongs to unknown person {conversation.person_id}"
                )
            by_person[conversation.person_id].append(conversation)
            storylines[storyline_key(conversation)].append(conversation)
            topic = conversation.meta.get("shared_topic")
            if topic:
                by_topic[str(topic)].append(conversation)
        self.by_person: dict[str, tuple[Conversation, ...]] = {key: tuple(value) for key, value in by_person.items()}
        self.storylines: dict[tuple[str, str], tuple[Conversation, ...]] = {
            key: tuple(sorted(beats, key=_beat_order)) for key, beats in storylines.items()
        }
        self.by_topic: dict[str, tuple[Conversation, ...]] = {key: tuple(value) for key, value in by_topic.items()}
        self.names: dict[str, str] = name_forms(self.people)
        self.expressions: dict[str, TimeExpression] = {e.key: e for e in query_expressions(config.NOW)}
        self.left_out: dict[str, tuple[str, ...]] = {c.episode_id: tuple(details_left_out(c)) for c in self.conversations}
        # A recap (padding the corpus generator wrote in code) restates the key facts of
        # the beat before it, so the two answer the same questions: each is the other's twin.
        twins: dict[str, list[Conversation]] = defaultdict(list)
        for beats in self.storylines.values():
            for previous, beat in zip(beats, beats[1:]):
                if beat.meta.get("recap"):
                    twins[previous.episode_id].append(beat)
                    twins[beat.episode_id].append(previous)
        self.twins: dict[str, tuple[Conversation, ...]] = {key: tuple(value) for key, value in twins.items()}
        self.roster_re = _names_re(name for p in self.people for name in (p.first_name, p.display_name))
        self.owner_re = _names_re((config.OWNER_FIRST_NAME, config.OWNER_DISPLAY_NAME))
        self._resolutions: dict[str, Resolution] = {}
        self._lock = threading.Lock()

    def resolve(self, name: str) -> Resolution:
        """``resolve_name`` against this roster, remembered per name (safe to call from worker threads)."""

        with self._lock:
            if name not in self._resolutions:
                self._resolutions[name] = resolve_name(name, self.people)
            return self._resolutions[name]

    def stands_alone(self, conversation: Conversation) -> bool:
        """May the conversation be a query's only gold: it is no recap, and no recap restates it.

        A question written from one twin is answered by the other as well, so
        a single-source query drawn on either would carry an incomplete label.
        """

        return not conversation.meta.get("recap") and conversation.episode_id not in self.twins

    def threads(self, person_id: str) -> list[tuple[str, str]]:
        """The storyline keys of one person, oldest storyline first."""

        keys = [key for key in self.storylines if key[0] == person_id]
        return sorted(keys, key=lambda key: (_chronological(self.storylines[key][0]), key))

    def thread_title(self, key: tuple[str, str]) -> str:
        return str(self.storylines[key][0].meta.get("storyline_title") or key[1])

    def neighbours(
        self, sources: Sequence[Conversation], limit: int, skip: Collection[str] = ()
    ) -> tuple[Conversation, ...]:
        """Up to ``limit`` other beats of the sources' storylines: the nearest beats, oldest first.

        Beats whose episode id is in ``skip`` are left out without counting
        against ``limit``.
        """

        chosen = {conversation.episode_id for conversation in sources}
        ranked: list[tuple[int, Conversation]] = []
        for key in sorted({storyline_key(conversation) for conversation in sources}):
            beats = self.storylines[key]
            positions = [index for index, beat in enumerate(beats) if beat.episode_id in chosen]
            ranked += [
                (min(abs(index - position) for position in positions), beat)
                for index, beat in enumerate(beats)
                if beat.episode_id not in chosen and beat.episode_id not in skip
            ]
        ranked.sort(key=lambda item: (item[0], _chronological(item[1])))
        return tuple(sorted((beat for _, beat in ranked[: max(limit, 0)]), key=_chronological))

    def period_rivals(
        self, window: TimeWindow, sources: Sequence[Conversation], limit: int
    ) -> tuple[Conversation, ...]:
        """Up to ``limit`` other conversations inside the window, the likeliest rivals of the sources first.

        These are what the time filter leaves in the pool next to the sources,
        so a "specific" temporal question is written against them. In a
        person's window the nearest in time come first. In everyone's window
        other people's conversations come first, those on a source's shared
        topic before the rest, because a no-name question must not fit
        another person. The result is in chronological order.
        """

        chosen = {conversation.episode_id for conversation in sources}
        people = {conversation.person_id for conversation in sources}
        topics = {str(conversation.meta.get("shared_topic")) for conversation in sources if conversation.meta.get("shared_topic")}

        def rank(conversation: Conversation) -> tuple[Any, ...]:
            distance = min(abs(conversation.start_time - source.start_time) for source in sources)
            if window.person is not None:
                return (distance, _chronological(conversation))
            other_person = conversation.person_id not in people
            on_topic = str(conversation.meta.get("shared_topic")) in topics
            return (not other_person, not on_topic, distance, _chronological(conversation))

        ranked = sorted((c for c in window.inside if c.episode_id not in chosen), key=rank)
        return tuple(sorted(ranked[: max(limit, 0)], key=_chronological))

    @cached_property
    def windows(self) -> tuple[TimeWindow, ...]:
        """Every (time expression, scope) pair with at least one conversation inside the range."""

        found = []
        for expression in self.expressions.values():
            padded = pad_time_range(expression.time_range, config.NOW)
            for person in (None, *self.people):
                scope = self.conversations if person is None else self.by_person.get(person.person_id, ())
                inside = tuple(c for c in scope if expression.time_range.contains(c.start_time))
                if inside:
                    pool_size = sum(1 for c in scope if padded.contains(c.start_time))
                    found.append(TimeWindow(expression, person, inside, pool_size))
        return tuple(found)


# --------------------------------------------------------------------------- #
# Slots, targets, bookkeeping                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlotPlan:
    """What the query in one slot should be, fixed before any target is drawn.

    ``subkind`` and ``heavy`` are the ticket's "about half" and "roughly"
    mixes: a draw takes the other sub-kind or weight class once the corpus has
    no target of the planned one left. ``named`` and ``pool_bucket`` are
    conditions and are never relaxed.
    """

    subkind: str | None = None
    heavy: bool | None = None  # the person's weight class, where the type balances it
    named: bool = True  # False: the query names nobody
    pool_bucket: str | None = None  # temporal: the pool-size bucket the query must fall in


@dataclass(frozen=True)
class Target:
    """Everything one attempt fixes in code before the writer is called."""

    qtype: str
    index: int
    attempt: int
    subkind: str | None = None
    person: Person | None = None  # who a named query is about
    name: str | None = None  # the name the question must contain, verbatim
    sources: tuple[Conversation, ...] = ()  # the gold conversations of an answerable query
    others: tuple[Conversation, ...] = ()  # summaries the question must not be answerable from
    expression: TimeExpression | None = None
    pool_size: int | None = None
    storyline_id: str | None = None
    shared_topic: str | None = None
    shown: tuple[Conversation, ...] = ()  # missing_detail: the thread's dialogues; unknown_name: the model
    asked: tuple[str, ...] = ()  # unanswerable: earlier questions about the person

    @property
    def salt(self) -> str:
        return f"{self.qtype}:{self.index}:{self.attempt}"

    def tried_keys(self) -> list[tuple[str, ...]]:
        """What this target uses up for the least-tried-first choices of later draws."""

        person_id = self.person.person_id if self.person is not None else EVERYONE
        if self.qtype == "temporal" and self.expression is not None:
            scope = "person" if self.person is not None else EVERYONE
            return [("expression", scope, self.expression.key), ("window", self.expression.key, person_id)]
        if self.qtype == "multi_conversation":
            return [("storyline", person_id, str(self.storyline_id))]
        if self.qtype == "cross_person":
            if self.subkind == "group":
                return [("topic", str(self.shared_topic))]
            return [("answer", conversation.person_id) for conversation in self.sources]
        if self.qtype == "unanswerable":
            keys = [("unanswerable", person_id)]
            if self.storyline_id is not None:
                keys.append(("unanswerable_thread", person_id, self.storyline_id))
            return keys
        if self.qtype == "unknown_name" and self.name:
            return [("unknown_name", _first_name(self.name)), *(("model", c.episode_id) for c in self.shown)]
        return []

    def busy_keys(self) -> list[tuple[str, ...]]:
        """What no other target of the same batch may share.

        An unanswerable question is written against the questions already
        asked about its person, so two of them for one person cannot be
        written at the same time.
        """

        if self.qtype == "unanswerable" and self.person is not None:
            return [("unanswerable", self.person.person_id)]
        return []


@dataclass(frozen=True)
class _Outcome:
    """What one attempt produced: a question (and answer), or the reason it was rejected."""

    writer_model: str
    checker_model: str | None = None
    question: str = ""
    answer: str | None = None
    problem: str | None = None


def _first_name(name: str) -> str:
    return name.split()[0].casefold()


class _State:
    """What earlier queries have used up, so that later draws avoid it."""

    def __init__(self) -> None:
        self.uses: Counter[str] = Counter()  # episode id -> queries it is a source for
        self.seen: dict[str, set[str]] = defaultdict(set)  # type -> episode ids already drawn for it
        self.tried: Counter[tuple[str, ...]] = Counter()  # draws per storyline, window, name, ...
        self.busy: set[tuple[str, ...]] = set()  # keys held by the batch that is running
        self.texts: set[str] = set()  # questions that exist, normalized
        self.asked: dict[str, list[str]] = defaultdict(list)  # person id -> unanswerable questions

    def free(self, conversation: Conversation, qtype: str) -> bool:
        """May the conversation be a source for one more query of this type?"""

        return self.uses[conversation.episode_id] < MAX_SOURCE_USES and conversation.episode_id not in self.seen[qtype]

    def reserve(self, target: Target) -> None:
        """Count a drawn target as used; within its type its sources are never drawn again."""

        for conversation in target.sources:
            self.uses[conversation.episode_id] += 1
            self.seen[target.qtype].add(conversation.episode_id)
        self.tried.update(target.tried_keys())
        self.busy.update(target.busy_keys())

    def settle(self, target: Target, question: str | None, *, accepted: bool) -> None:
        """Close a reserved target: a rejected one gives its sources back to the other types."""

        self.busy.difference_update(target.busy_keys())
        if not accepted:
            for conversation in target.sources:
                self.uses[conversation.episode_id] -= 1
        if question is not None:
            self.texts.add(_normal(question))
            if target.qtype == "unanswerable" and target.person is not None:
                self.asked[target.person.person_id].append(question)


# --------------------------------------------------------------------------- #
# Slot plans and the dev/test split                                            #
# --------------------------------------------------------------------------- #
def _spread(n: int, numerator: int, denominator: int) -> list[bool]:
    """``n`` flags, ``n * numerator // denominator`` of them True, spaced evenly."""

    return [(i + 1) * numerator // denominator > i * numerator // denominator for i in range(n)]


def _temporal_plans(corpus: Corpus, n: int) -> list[SlotPlan]:
    """Scope, sub-kind and required pool bucket of each temporal slot.

    The ticket's mix out of 40 is 25 person-scoped (10 recap, 15 specific) and
    15 with no name (5 recap, 10 specific), with at least 12 pools of 21-100
    and at least 8 of at most 20. Mid-size pools come from no-name specific
    queries first and from person-scoped ones second; where the corpus has too
    few person-scoped windows that large, no-name recap slots turn specific.
    """

    everyone = n * 15 // 40
    person = n - everyone
    specific = {True: person * 15 // 25, False: everyone * 10 // 15}
    recap = {True: person - specific[True], False: everyone - specific[False]}
    need_mid, need_small = n * 12 // 40, n * 8 // 40
    mid_windows = {
        named: sum(
            1
            for window in corpus.windows
            if window.subkind == "specific"
            and (window.person is not None) == named
            and pool_bucket(window.pool_size) == MID_POOL
        )
        for named in (True, False)
    }
    mid = {False: min(specific[False], need_mid) if mid_windows[False] else 0}
    mid[True] = min(need_mid - mid[False], specific[True], mid_windows[True])
    short = need_mid - mid[True] - mid[False]
    if short and mid_windows[False]:
        moved = min(short, recap[False])
        recap[False] -= moved
        specific[False] += moved
        mid[False] += moved
        short -= moved
    if short:
        raise QueryGenerationError(
            f"temporal: the corpus supports {need_mid - short} queries with a pool of {MID_POOL}, "
            f"{need_mid} are needed"
        )
    # No-name slots come first: they draw first, and the few days that hold one
    # to three conversations in all are the only windows a no-name recap can use.
    plans: list[SlotPlan] = []
    for named in (False, True):
        plans += [SlotPlan("recap", named=named)] * recap[named]
        plans += [
            SlotPlan("specific", named=named, pool_bucket=MID_POOL if i < mid[named] else None)
            for i in range(specific[named])
        ]
    unassigned = [i for i, plan in enumerate(plans) if plan.pool_bucket is None]
    unassigned.sort(key=lambda i: (plans[i].subkind != "recap", not plans[i].named))  # the smallest windows first
    for i in unassigned[:need_small]:
        plans[i] = replace(plans[i], pool_bucket=SMALL_POOL)
    return plans


def build_plans(corpus: Corpus, counts: Mapping[str, int]) -> dict[str, list[SlotPlan]]:
    """The plan of every slot: the balance each type must strike, decided before any draw."""

    def n(qtype: str) -> int:
        return counts.get(qtype, 0)

    plans: dict[str, list[SlotPlan]] = {
        qtype: [SlotPlan(heavy=heavy) for heavy in _spread(n(qtype), 1, 2)]
        for qtype in ("lookup", "paraphrase", "detail_only")
    }
    plans["multi_conversation"] = [SlotPlan(heavy=heavy) for heavy in _spread(n("multi_conversation"), 7, 10)]
    plans["cross_person"] = [
        SlotPlan("group" if group else "single", named=False) for group in _spread(n("cross_person"), 2, 5)
    ]
    plans["temporal"] = _temporal_plans(corpus, n("temporal"))
    plans["unanswerable"] = [
        SlotPlan(("absent_topic", "missing_detail")[i // 2 % 2], heavy=i % 2 == 0) for i in range(n("unanswerable"))
    ]
    plans["unknown_name"] = [
        SlotPlan("full_name" if full else "first_name") for full in _spread(n("unknown_name"), 1, 3)
    ]
    return plans


def scaled_counts(total: int) -> dict[str, int]:
    """Per-type counts for ``total`` queries, in the proportions of ``config.QUERY_TYPE_COUNTS``."""

    quotas = {qtype: Fraction(count * total, config.N_QUERIES) for qtype, count in config.QUERY_TYPE_COUNTS.items()}
    counts = {qtype: int(quota) for qtype, quota in quotas.items()}
    by_remainder = sorted(quotas, key=lambda qtype: (counts[qtype] - quotas[qtype], QUERY_TYPES.index(qtype)))
    for qtype in by_remainder[: total - sum(counts.values())]:
        counts[qtype] += 1
    return counts


def dev_allocation(counts: Mapping[str, int]) -> dict[str, int]:
    """How many queries of each type go to dev.

    ``DEV_FRACTION`` of each type, rounded by largest remainder so that the
    total is exactly the rounded fraction of all queries; a tie goes to the
    smaller type. When there are at least as many dev queries as types, every
    type gets one.
    """

    fraction = Fraction(str(config.DEV_FRACTION))
    sizes = {qtype: counts.get(qtype, 0) for qtype in QUERY_TYPES}
    total = int(sum(sizes.values()) * fraction + Fraction(1, 2))
    allocation = {qtype: int(size * fraction) for qtype, size in sizes.items()}
    by_remainder = sorted(
        QUERY_TYPES,
        key=lambda qtype: (allocation[qtype] - sizes[qtype] * fraction, sizes[qtype], QUERY_TYPES.index(qtype)),
    )
    for qtype in by_remainder[: total - sum(allocation.values())]:
        allocation[qtype] += 1
    present = [qtype for qtype in QUERY_TYPES if sizes[qtype]]
    if total >= len(present):
        for qtype in present:
            if allocation[qtype] == 0:
                donor = max(present, key=lambda other: allocation[other])
                allocation[donor] -= 1
                allocation[qtype] = 1
    return allocation


def dev_slots(counts: Mapping[str, int]) -> dict[str, set[int]]:
    """The slot indexes of each type that form the dev split (a seeded draw per type)."""

    allocation = dev_allocation(counts)
    return {
        qtype: set(
            random.Random(f"{config.SEED}:query-split:{qtype}").sample(range(counts.get(qtype, 0)), allocation[qtype])
        )
        for qtype in QUERY_TYPES
    }


# --------------------------------------------------------------------------- #
# Validators                                                                   #
# --------------------------------------------------------------------------- #
def target_problem(corpus: Corpus, target: Target) -> str | None:
    """Why a drawn target cannot carry a query, or None. Runs before any model call."""

    if target.qtype == "unknown_name":
        return None if corpus.resolve(target.name or "").status == "unknown" else "unknown_name_resolves"
    if target.person is not None and corpus.resolve(target.name or "").person_id != target.person.person_id:
        return "name_does_not_resolve"
    if target.expression is not None:
        if not all(target.expression.time_range.contains(c.start_time) for c in target.sources):
            return "gold_outside_time_range"
    return None


def question_problem(corpus: Corpus, target: Target, question: object, answer: object = None) -> str | None:
    """Why a written question is unusable, or None when it passes every check in code.

    ``answer`` is the proposed gold answer of an answerable type; when it is a
    string, a question that already contains it is rejected as well.
    """

    if question is None:
        return "writer_declined"
    if not isinstance(question, str) or not question.strip():
        return "question_missing"
    text = question.strip()
    if "\n" in text or not text.endswith("?"):
        return "not_one_question"
    if len(text.split()) > MAX_QUESTION_WORDS:
        return "question_too_long"
    if _SOURCE_WORDS_RE.search(text):
        return "mentions_source_material"
    if corpus.owner_re.search(text):
        return "owner_named"
    rest = text
    if target.name is not None:
        if not contains_phrase(text, target.name):
            return "name_missing"
        rest = _without_phrase(text, target.name)
    if corpus.roster_re.search(rest):
        return "other_person_named" if target.name is not None else "person_named"
    if target.expression is not None:
        phrase = target.expression.phrase
        if not contains_phrase(rest, phrase, ignore_case=True):
            return "time_phrase_missing"
        if is_possessive(rest, phrase):
            return "time_phrase_dates_a_thing"
        if time_words(_without_phrase(rest, phrase, ignore_case=True)):
            return "extra_time_expression"
    elif time_words(rest):
        return "time_expression"
    if _AUXILIARY_RE.match(text):
        return "yes_no_question"
    if target.qtype == "paraphrase":
        shared = shared_content_words(text, target.sources[0].summary, person_words(target.person))
        if len(shared) > MAX_SHARED_WORDS:
            return "reuses_summary_wording"
    if isinstance(answer, str) and question_carries_answer(text, answer, person_words(target.person)):
        return "question_contains_answer"
    return None


def answer_problem(answer: object) -> str | None:
    """Why a gold answer is unusable: it must be one to three sentences."""

    if not isinstance(answer, str) or not answer.strip():
        return "answer_missing"
    text = answer.strip()
    if len(_SENTENCE_BREAK_RE.split(text)) > MAX_ANSWER_SENTENCES or len(text.split()) > MAX_ANSWER_WORDS:
        return "answer_too_long"
    return None


def check_problem(target: Target, data: Mapping[str, Any]) -> str | None:
    """Why the checker's verdict rejects a candidate, or None when it confirms it."""

    def verdicts(*keys: str) -> list[bool] | None:
        values = [data.get(key) for key in keys]
        return values if all(isinstance(value, bool) for value in values) else None

    if target.qtype == "unanswerable":
        found = verdicts("answerable")
        if found is None:
            return "checker_malformed"
        return "answerable" if found[0] else None
    if target.qtype == "detail_only":
        found = verdicts("answerable_from_summary", "answerable_from_dialogue", "answer_matches_dialogue")
        elsewhere = verdicts("stated_elsewhere") if target.others else [False]
        if found is None or elsewhere is None:
            return "checker_malformed"
        if found[0]:
            return "answerable_from_summary"
        if not found[1]:
            return "not_answerable_from_dialogue"
        if not found[2]:
            return "answer_disagrees_with_dialogue"
        return "detail_stated_elsewhere" if elsewhere[0] else None
    found = verdicts("answerable", "answer_agrees")
    contributing = data.get("contributing")
    dated = verdicts("time_phrase_dates_the_conversation") if target.expression is not None else [True]
    if found is None or dated is None or not isinstance(contributing, list):
        return "checker_malformed"
    if not found[0]:
        return "not_answerable_from_summaries"
    if not found[1]:
        return "answer_disagrees_with_summaries"
    numbers = {number for number in contributing if type(number) is int}
    if not set(range(1, len(target.sources) + 1)) <= numbers:
        return "source_does_not_contribute"
    return None if dated[0] else "time_phrase_dates_a_thing"


# --------------------------------------------------------------------------- #
# Generator                                                                    #
# --------------------------------------------------------------------------- #
def _evenly_spaced(items: Sequence[T], cap: int) -> list[T]:
    """All items, or ``cap`` of them spread over the whole sequence."""

    if len(items) <= cap:
        return list(items)
    return [items[i * (len(items) - 1) // (cap - 1)] for i in range(cap)]


class QueryGenerator:
    """Fills the query slots of each type from seeded target draws.

    A slot is one (type, index) pair; it owns a stream of targets, one per
    attempt, each drawn from ``random.Random`` seeded with the slot and the
    attempt number. Model calls of one round run in parallel, but targets are
    drawn and results are applied in slot order, so the outcome does not depend
    on thread timing. An instance keeps what its queries used up, so it serves
    one ``generate`` or ``regenerate`` call.
    """

    def __init__(
        self,
        llm: ChatLLM,
        settings: config.Settings,
        people: Sequence[Person],
        conversations: Sequence[Conversation],
        unknown_names: Sequence[str],
        counts: Mapping[str, int] = config.QUERY_TYPE_COUNTS,
        workers: int | None = None,
    ) -> None:
        unknown_types = sorted(set(counts) - set(QUERY_TYPES))
        if unknown_types:
            raise ValueError(f"unknown query types in counts: {unknown_types}")
        self.llm = llm
        self.settings = settings
        self.corpus = Corpus(people, conversations)
        self.unknown_names: list[str] = sorted({name.strip() for name in unknown_names if name.strip()})
        self.counts: dict[str, int] = {qtype: int(counts.get(qtype, 0)) for qtype in QUERY_TYPES}
        self.workers = max(1, workers or settings.llm_concurrency)
        self.plans = build_plans(self.corpus, self.counts)
        self.state = _State()
        self.rejections: dict[str, Counter[str]] = {qtype: Counter() for qtype in QUERY_TYPES}
        self.off_plan: Counter[str] = Counter()  # type -> queries whose sub-kind or weight class is not the planned one

    # ---- entry points ------------------------------------------------------ #
    def generate(self) -> list[Query]:
        """Fill every slot and return the queries in type order, split assigned."""

        dev = dev_slots(self.counts)
        filled: dict[tuple[str, int], Query] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for qtype in GENERATION_ORDER:
                wanted = {index: 0 for index in range(self.counts[qtype])}
                for index, query in self._fill(pool, qtype, wanted).items():
                    query.split = "dev" if index in dev[qtype] else "test"
                    filled[(qtype, index)] = query
        return [filled[(qtype, index)] for qtype in QUERY_TYPES for index in range(self.counts[qtype])]

    def regenerate(self, queries: Sequence[Query], query_ids: Collection[str]) -> list[Query]:
        """Replace the listed queries with fresh ones in the same slots; the others are returned as they are.

        A replacement keeps its query id, type and split and continues its
        slot's target stream after the attempt that produced the old query.
        The old query's sources are not drawn again for its type, and its text
        counts as taken.
        """

        slots = {
            config.stable_id("query", f"{qtype}:{index}"): (qtype, index)
            for qtype in QUERY_TYPES
            for index in range(self.counts[qtype])
        }
        listed = set(query_ids)
        absent = sorted(listed - {query.query_id for query in queries})
        if absent:
            raise ValueError(f"cannot regenerate query ids that are not in the query file: {absent}")
        wanted: dict[str, dict[int, int]] = defaultdict(dict)
        for query in queries:
            slot = slots.get(query.query_id)
            if slot is None or slot[0] != query.type:
                raise ValueError(f"query {query.query_id} ({query.type}) is not a slot of these per-type counts")
            target = self._target_of(query, slot[1])
            self.state.reserve(target)
            self.state.settle(target, query.query, accepted=query.query_id not in listed)
            if query.query_id in listed:
                wanted[query.type][slot[1]] = target.attempt + 1
        fresh: dict[str, Query] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for qtype in GENERATION_ORDER:
                if wanted[qtype]:
                    for query in self._fill(pool, qtype, wanted[qtype]).values():
                        fresh[query.query_id] = query
        result = []
        for query in queries:
            replacement = fresh.get(query.query_id)
            if replacement is not None:
                replacement.split = query.split
            result.append(replacement or query)
        return result

    # ---- the fill loop ----------------------------------------------------- #
    def _fill(self, pool: ThreadPoolExecutor, qtype: str, wanted: Mapping[int, int]) -> dict[int, Query]:
        """Fill the given slots of one type; ``wanted`` maps a slot index to its first attempt number."""

        attempts = dict(wanted)
        pending = sorted(wanted)
        filled: dict[int, Query] = {}
        rejected = self.rejections[qtype]
        budget = ATTEMPTS_PER_QUERY * len(wanted)
        spent = 0
        while pending:
            batch: list[Target] = []
            for index in pending:
                while spent < budget:
                    target = self._draw(qtype, index, attempts[index])
                    if target is None:  # nothing this slot can draw while the batch holds its reservations
                        break
                    spent += 1
                    self.state.reserve(target)
                    problem = target_problem(self.corpus, target)
                    if problem is None:
                        batch.append(target)
                        break
                    self.state.settle(target, None, accepted=False)
                    rejected[problem] += 1
                    attempts[index] += 1
            if not batch:
                why = f"the budget of {budget} attempts is used up" if spent >= budget else "no target is left to draw"
                raise QueryGenerationError(
                    f"{qtype}: filled {len(filled)} of {len(wanted)} queries, then {why}; "
                    f"rejections: {dict(sorted(rejected.items()))}"
                )
            for target, outcome in zip(batch, list(pool.map(self._attempt, batch))):
                problem = outcome.problem
                if problem is None and _normal(outcome.question) in self.state.texts:
                    problem = "duplicate_question"
                if problem is None:
                    self.state.settle(target, outcome.question, accepted=True)
                    filled[target.index] = self._query(target, outcome)
                    if self._is_off_plan(target):
                        self.off_plan[qtype] += 1
                else:
                    self.state.settle(target, None, accepted=False)
                    rejected[problem] += 1
                    attempts[target.index] += 1
            pending = [index for index in pending if index not in filled]
        logger.info(
            "%s: %d queries from %d attempts, %d off plan; rejections: %s",
            qtype,
            len(filled),
            spent,
            self.off_plan[qtype],
            dict(sorted(rejected.items())),
        )
        return filled

    def _is_off_plan(self, target: Target) -> bool:
        """Did the draw have to leave the slot's planned sub-kind or weight class?"""

        plan = self.plans[target.qtype][target.index]
        other_weight = plan.heavy is not None and target.person is not None and target.person.is_heavy != plan.heavy
        return target.subkind != plan.subkind or other_weight

    def _attempt(self, target: Target) -> _Outcome:
        """Write one candidate and check it. Runs on a worker thread and touches no shared state."""

        prompt = self._writer_prompt(target)
        written = self.llm.complete_json(
            model=self.settings.gen_model,
            system=prompt.system,
            user=prompt.user,
            max_tokens=MAX_TOKENS,
            temperature=WRITER_TEMPERATURE,
            namespace=WRITER_NAMESPACE,
            salt=target.salt,
        )
        outcome = _Outcome(writer_model=written.model or self.settings.gen_model)
        data = written.data
        if not isinstance(data, dict) or "question" not in data:
            return replace(outcome, problem="writer_malformed")
        answerable = target.qtype in ANSWERABLE_TYPES
        problem = question_problem(self.corpus, target, data["question"], data.get("answer") if answerable else None)
        if problem is None and answerable:
            problem = answer_problem(data.get("answer"))
        if problem is not None:
            return replace(outcome, problem=problem)
        outcome = replace(
            outcome, question=data["question"].strip(), answer=data["answer"].strip() if answerable else None
        )
        if target.qtype == "unknown_name":  # nothing to check against: the name matches nobody
            return outcome
        prompt = self._checker_prompt(target, outcome.question, outcome.answer or "")
        checked = self.llm.complete_json(
            model=self.settings.judge_model,
            system=prompt.system,
            user=prompt.user,
            max_tokens=MAX_TOKENS,
            temperature=CHECKER_TEMPERATURE,
            namespace=CHECKER_NAMESPACE,
            salt=target.salt,
        )
        verdict = checked.data if isinstance(checked.data, dict) else {}
        return replace(
            outcome,
            checker_model=checked.model or self.settings.judge_model,
            problem=check_problem(target, verdict),
        )

    def _query(self, target: Target, outcome: _Outcome) -> Query:
        """The dataset record of an accepted candidate. Its gold labels come from the target."""

        meta: dict[str, Any] = {
            "source_episode_ids": [conversation.episode_id for conversation in target.sources],
            "subkind": target.subkind,
            "writer_model": outcome.writer_model,
            "checker_model": outcome.checker_model,
            "attempt": target.attempt,
        }
        if target.storyline_id is not None:
            meta["storyline_id"] = target.storyline_id
        if target.shared_topic is not None:
            meta["shared_topic"] = target.shared_topic
        if target.expression is not None:
            meta["time_expression"] = target.expression.key
            meta["pool_size"] = target.pool_size
        if target.qtype == "unknown_name":
            meta["model_episode_id"] = target.shown[0].episode_id
        answerable = target.qtype in ANSWERABLE_TYPES
        return Query(
            query_id=config.stable_id("query", f"{target.qtype}:{target.index}"),
            query=outcome.question,
            type=target.qtype,
            gold_name=target.name,
            gold_time_range=target.expression.time_range if target.expression is not None else None,
            gold_episode_ids=[conversation.episode_id for conversation in target.sources] if answerable else [],
            gold_answer=outcome.answer if answerable else None,
            gold_person_id=target.person.person_id if target.person is not None else None,
            meta=meta,
        )

    def _target_of(self, query: Query, index: int) -> Target:
        """Rebuild, from an existing query, the target that the bookkeeping needs."""

        meta = query.meta
        missing = [e for e in meta.get("source_episode_ids", ()) if e not in self.corpus.conversation]
        if missing:
            raise ValueError(f"query {query.query_id} was written from conversations that are not in the corpus")
        model = self.corpus.conversation.get(str(meta.get("model_episode_id")))
        return Target(
            qtype=query.type,
            index=index,
            attempt=int(meta.get("attempt", 0)),
            subkind=meta.get("subkind"),
            person=self.corpus.person.get(query.gold_person_id or ""),
            name=query.gold_name,
            sources=tuple(self.corpus.conversation[e] for e in meta.get("source_episode_ids", ())),
            expression=self.corpus.expressions.get(str(meta.get("time_expression"))),
            storyline_id=meta.get("storyline_id"),
            shared_topic=meta.get("shared_topic"),
            shown=(model,) if model is not None else (),
        )

    # ---- target draws ------------------------------------------------------ #
    def _draw(self, qtype: str, index: int, attempt: int) -> Target | None:
        """The slot's target for this attempt, or None when nothing can be drawn right now."""

        plan = self.plans[qtype][index]
        rng = random.Random(f"{config.SEED}:query:{qtype}:{index}:{attempt}")
        fields = self._DRAWS[qtype](self, qtype, plan, rng)
        if fields is None:
            return None
        return Target(qtype=qtype, index=index, attempt=attempt, **{"subkind": plan.subkind, **fields})

    def _least_tried(self, items: Sequence[T], key: Callable[[T], tuple[str, ...]]) -> list[T]:
        """The items drawn least often so far, in their given order."""

        tries = [self.state.tried[key(item)] for item in items]
        fewest = min(tries)
        return [item for item, count in zip(items, tries) if count == fewest]

    def _weighs(self, person_id: str, heavy: bool | None) -> bool:
        return heavy is None or self.corpus.person[person_id].is_heavy == heavy

    @staticmethod
    def _weights(plan: SlotPlan) -> tuple[bool | None, ...]:
        """The weight classes a draw tries: the planned one, then the other."""

        return (None,) if plan.heavy is None else (plan.heavy, not plan.heavy)

    def _named(self, person_id: str) -> dict[str, Any]:
        return {"person": self.corpus.person[person_id], "name": self.corpus.names[person_id]}

    def _draw_single(self, qtype: str, plan: SlotPlan, rng: random.Random) -> dict[str, Any] | None:
        """Lookup, paraphrase, detail-only: one conversation of a person of the planned weight."""

        for heavy in self._weights(plan):
            free = [
                c
                for c in self.corpus.conversations
                if self._weighs(c.person_id, heavy) and self.state.free(c, qtype) and self.corpus.stands_alone(c)
            ]
            if free:
                break
        else:
            return None
        if qtype == "detail_only":  # a summary that leaves a detail out makes the better source
            free = [c for c in free if self.corpus.left_out[c.episode_id]] or free
        source = rng.choice(free)
        return {
            **self._named(source.person_id),
            "sources": (source,),
            "others": self.corpus.neighbours((source,), MAX_OTHERS),
        }

    def _draw_multi(self, qtype: str, plan: SlotPlan, rng: random.Random) -> dict[str, Any] | None:
        """Two or three consecutive beats of one storyline."""

        for heavy in self._weights(plan):
            runs: dict[tuple[str, str], list[tuple[Conversation, ...]]] = {}
            for key, beats in self.corpus.storylines.items():
                if not self._weighs(key[0], heavy):
                    continue
                for size in MULTI_SIZES:
                    for start in range(len(beats) - size + 1):
                        run = beats[start : start + size]
                        if all(self.state.free(c, qtype) for c in run):
                            runs.setdefault(key, []).append(run)
            if runs:
                break
        else:
            return None
        key = rng.choice(self._least_tried(sorted(runs), lambda k: ("storyline", *k)))
        size = rng.choice(sorted({len(run) for run in runs[key]}))
        sources = rng.choice([run for run in runs[key] if len(run) == size])
        return {
            **self._named(key[0]),
            "sources": sources,
            "others": self.corpus.neighbours(sources, MAX_OTHERS),
            "storyline_id": key[1],
        }

    def _draw_cross(self, qtype: str, plan: SlotPlan, rng: random.Random) -> dict[str, Any] | None:
        """One conversation behind a who-question, or a shared-topic group of people."""

        if plan.subkind == "group":
            group = self._draw_group(qtype, rng)
            if group is not None:
                return group
        free = {
            person_id: [c for c in conversations if self.state.free(c, qtype) and self.corpus.stands_alone(c)]
            for person_id, conversations in self.corpus.by_person.items()
        }
        candidates = sorted(person_id for person_id, conversations in free.items() if conversations)
        if not candidates:
            return None
        person_id = rng.choice(self._least_tried(candidates, lambda p: ("answer", p)))
        source = rng.choice(free[person_id])
        # The conversations most likely to answer the same who-question: other
        # people on the same shared topic, then the rest of the storyline.
        rivals = [c for c in self.corpus.by_topic.get(str(source.meta.get("shared_topic")), ()) if c.person_id != person_id]
        others = sorted(rng.sample(rivals, min(len(rivals), CROSS_RIVALS)), key=_chronological)
        others += self.corpus.neighbours((source,), MAX_OTHERS - len(others))
        return {"subkind": "single", "sources": (source,), "others": tuple(others)}

    def _draw_group(self, qtype: str, rng: random.Random) -> dict[str, Any] | None:
        groups: dict[str, dict[str, list[Conversation]]] = {}
        for topic, conversations in self.corpus.by_topic.items():
            members: dict[str, list[Conversation]] = defaultdict(list)
            for conversation in conversations:
                if self.state.free(conversation, qtype):
                    members[conversation.person_id].append(conversation)
            if len(members) >= GROUP_SIZES[0]:
                groups[topic] = members
        if not groups:
            return None
        topic = rng.choice(self._least_tried(sorted(groups), lambda t: ("topic", t)))
        members = groups[topic]
        size = rng.choice([size for size in GROUP_SIZES if size <= len(members)])
        chosen = rng.sample(sorted(members), size)
        sources = sorted((rng.choice(members[person_id]) for person_id in chosen), key=_chronological)
        # As for a single who-question: other people on the topic, then the rest of the
        # sources' own storylines, so the aspect asked about is not one a chosen person's
        # whole thread on the topic answers.
        rivals = [c for c in self.corpus.by_topic[topic] if c.person_id not in chosen]
        others = sorted(rng.sample(rivals, min(len(rivals), CROSS_RIVALS)), key=_chronological)
        others += self.corpus.neighbours(sources, MAX_OTHERS - len(others))
        return {"sources": tuple(sources), "others": tuple(others), "shared_topic": topic}

    def _draw_temporal(self, qtype: str, plan: SlotPlan, rng: random.Random) -> dict[str, Any] | None:
        """A time window of the planned scope, sub-kind and pool bucket, and the sources inside it."""

        scope = "person" if plan.named else EVERYONE
        for subkind in (plan.subkind, "specific" if plan.subkind == "recap" else "recap"):
            open_windows: dict[str, list[tuple[TimeWindow, list[Conversation]]]] = {}
            for window in self.corpus.windows:
                if (window.person is not None) != plan.named or window.subkind != subkind:
                    continue
                if plan.pool_bucket is not None and pool_bucket(window.pool_size) != plan.pool_bucket:
                    continue
                free = [c for c in window.inside if self.state.free(c, qtype)]
                # A recap's gold is every conversation in the window, so all of them must be free;
                # a specific query needs a conversation that stands alone, or two beats in a row.
                if subkind == "recap":
                    usable = bool(free) and len(free) == len(window.inside)
                else:
                    usable = any(self.corpus.stands_alone(c) for c in free) or bool(self._beat_pairs(free))
                if usable:
                    open_windows.setdefault(window.expression.key, []).append((window, free))
            if open_windows:
                break
        else:
            return None
        key = rng.choice(self._least_tried(sorted(open_windows), lambda k: ("expression", scope, k)))
        window, free = rng.choice(self._least_tried(open_windows[key], lambda item: ("window", key, item[0].scope_id)))
        if subkind == "recap":
            sources: tuple[Conversation, ...] = tuple(free)
            others: tuple[Conversation, ...] = ()
        else:
            # The question is checked against what the time filter leaves in the pool
            # first; further beats of the sources' storylines fill any room left.
            sources = self._specific_sources(free, rng)
            others = self.corpus.period_rivals(window, sources, MAX_OTHERS)
            others += self.corpus.neighbours(sources, MAX_OTHERS - len(others), skip={c.episode_id for c in others})
        fields: dict[str, Any] = {
            "subkind": subkind,
            "sources": sources,
            "others": others,
            "expression": window.expression,
            "pool_size": window.pool_size,
        }
        if window.person is not None:
            fields.update(self._named(window.person.person_id))
        return fields

    def _beat_pairs(self, free: Sequence[Conversation]) -> list[tuple[Conversation, Conversation]]:
        """Consecutive beats of one storyline that are both in ``free``, in storyline order."""

        inside = {conversation.episode_id for conversation in free}
        return [
            (first, second)
            for key in sorted({storyline_key(conversation) for conversation in free})
            for first, second in zip(self.corpus.storylines[key], self.corpus.storylines[key][1:])
            if first.episode_id in inside and second.episode_id in inside
        ]

    def _specific_sources(self, free: Sequence[Conversation], rng: random.Random) -> tuple[Conversation, ...]:
        """One conversation of the window that stands alone, or sometimes two consecutive beats of one storyline.

        The window was opened with at least one of the two on offer.
        """

        pairs = self._beat_pairs(free)
        alone = [conversation for conversation in free if self.corpus.stands_alone(conversation)]
        wants_pair = rng.random() < PAIR_SHARE
        if pairs and (wants_pair or not alone):
            return rng.choice(pairs)
        return (rng.choice(alone),)

    def _draw_unanswerable(self, qtype: str, plan: SlotPlan, rng: random.Random) -> dict[str, Any] | None:
        """A person of the planned weight and, for a missing detail, one of their storylines."""

        for heavy in self._weights(plan):
            known = [p for p in self.corpus.people if self._weighs(p.person_id, heavy) and p.person_id in self.corpus.by_person]
            if known:
                break
        else:
            return None
        # A person whose question is being written right now has to wait for it: the next
        # question about them is written against it.
        people = [person for person in known if ("unanswerable", person.person_id) not in self.state.busy]
        if not people:
            return None
        person = rng.choice(self._least_tried(people, lambda p: ("unanswerable", p.person_id)))
        fields: dict[str, Any] = {
            **self._named(person.person_id),
            "asked": tuple(self.state.asked.get(person.person_id, ())),
        }
        if plan.subkind == "missing_detail":
            threads = self.corpus.threads(person.person_id)
            key = rng.choice(self._least_tried(threads, lambda k: ("unanswerable_thread", *k)))
            beats = self.corpus.storylines[key]
            start = rng.randrange(max(1, len(beats) - MAX_THREAD_DIALOGUES + 1))
            fields.update(storyline_id=key[1], shown=beats[start : start + MAX_THREAD_DIALOGUES])
        return fields

    def _draw_unknown(self, qtype: str, plan: SlotPlan, rng: random.Random) -> dict[str, Any] | None:
        """An unknown name and a real conversation to model the question on."""

        if not self.unknown_names or not self.corpus.conversations:
            return None
        full_name = rng.choice(self._least_tried(self.unknown_names, lambda name: ("unknown_name", _first_name(name))))
        model = rng.choice(self._least_tried(self.corpus.conversations, lambda c: ("model", c.episode_id)))
        return {"name": full_name if plan.subkind == "full_name" else full_name.split()[0], "shown": (model,)}

    _DRAWS: dict[str, Callable[..., dict[str, Any] | None]] = {
        "lookup": _draw_single,
        "paraphrase": _draw_single,
        "detail_only": _draw_single,
        "multi_conversation": _draw_multi,
        "cross_person": _draw_cross,
        "temporal": _draw_temporal,
        "unanswerable": _draw_unanswerable,
        "unknown_name": _draw_unknown,
    }

    # ---- prompts ----------------------------------------------------------- #
    def _writer_prompt(self, target: Target) -> prompts.Prompt:
        corpus, people = self.corpus, self.corpus.person
        name = target.name or ""
        if target.qtype == "lookup":
            return prompts.write_lookup(name, target.sources[0], target.others, people)
        if target.qtype == "paraphrase":
            source = target.sources[0]
            avoid = sorted(set(content_words(source.summary, person_words(target.person))))
            return prompts.write_paraphrase(name, source, target.others, people, avoid)
        if target.qtype == "detail_only":
            details = corpus.left_out[target.sources[0].episode_id]
            return prompts.write_detail_only(name, target.sources[0], target.others, people, details)
        if target.qtype == "multi_conversation":
            title = corpus.thread_title(storyline_key(target.sources[0]))
            return prompts.write_multi(name, target.sources, target.others, people, title)
        if target.qtype == "cross_person":
            if target.subkind == "group":
                return prompts.write_cross_group(target.sources, target.others, people, str(target.shared_topic))
            return prompts.write_cross_single(target.sources[0], target.others, people)
        if target.qtype == "temporal" and target.expression is not None:
            time_range = target.expression.time_range
            period = tuple(c for c in target.others if time_range.contains(c.start_time))
            thread = tuple(c for c in target.others if not time_range.contains(c.start_time))
            return prompts.write_temporal(
                target.name, target.expression.phrase, str(target.subkind), target.sources, period, thread, people
            )
        if target.qtype == "unanswerable" and target.person is not None:
            person = target.person
            own = corpus.by_person[person.person_id]
            summaries = _evenly_spaced(own, WRITER_SUMMARY_CAP)
            if target.subkind == "missing_detail":
                key = (person.person_id, str(target.storyline_id))
                return prompts.write_missing_detail(
                    name,
                    person,
                    corpus.thread_title(key),
                    target.shown,
                    len(corpus.storylines[key]),
                    summaries,
                    len(own),
                    target.asked,
                )
            threads = [(corpus.thread_title(key), len(corpus.storylines[key])) for key in corpus.threads(person.person_id)]
            return prompts.write_absent_topic(name, person, threads, summaries, len(own), target.asked)
        return prompts.write_unknown_name(name, target.shown[0])

    def _checker_prompt(self, target: Target, question: str, answer: str) -> prompts.Prompt:
        people = self.corpus.person
        if target.qtype == "detail_only":
            return prompts.check_detail(question, answer, target.sources[0], target.others, people)
        if target.qtype == "unanswerable" and target.person is not None:
            person = target.person
            own = self.corpus.by_person[person.person_id]
            # Every dialogue in full for a light person; for a heavy one, the storyline the
            # question is about. Every conversation also carries what its dialogue states.
            if not person.is_heavy:
                shown = own
            elif target.storyline_id is not None:
                shown = self.corpus.storylines[(person.person_id, target.storyline_id)]
            else:
                shown = ()
            return prompts.check_unanswerable(question, person, own, {c.episode_id for c in shown})
        phrase = target.expression.phrase if target.expression is not None else None
        return prompts.check_summaries(question, answer, target.sources, people, phrase)


def generate_queries(
    llm: ChatLLM,
    settings: config.Settings,
    people: Sequence[Person],
    conversations: Sequence[Conversation],
    unknown_names: Sequence[str],
    counts: Mapping[str, int] = config.QUERY_TYPE_COUNTS,
    workers: int | None = None,
) -> list[Query]:
    """Generate ``counts`` queries per type with a seeded, type-stratified dev/test split.

    ``workers`` defaults to ``settings.llm_concurrency``. Raises
    ``QueryGenerationError`` when a type cannot be filled.
    """

    return QueryGenerator(llm, settings, people, conversations, unknown_names, counts, workers).generate()


def regenerate_queries(
    llm: ChatLLM,
    settings: config.Settings,
    people: Sequence[Person],
    conversations: Sequence[Conversation],
    unknown_names: Sequence[str],
    queries: Sequence[Query],
    query_ids: Collection[str],
    counts: Mapping[str, int] = config.QUERY_TYPE_COUNTS,
    workers: int | None = None,
) -> list[Query]:
    """Replace the queries in ``query_ids`` with fresh ones of the same type and split."""

    generator = QueryGenerator(llm, settings, people, conversations, unknown_names, counts, workers)
    return generator.regenerate(queries, query_ids)


# --------------------------------------------------------------------------- #
# Validation of a finished query set                                           #
# --------------------------------------------------------------------------- #
def validate_queries(
    queries: Sequence[Query],
    people: Sequence[Person],
    conversations: Sequence[Conversation],
    counts: Mapping[str, int] = config.QUERY_TYPE_COUNTS,
) -> list[str]:
    """Everything that is wrong with a query set, as readable problems (empty when it is sound)."""

    problems: list[str] = []
    per_type = Counter(query.type for query in queries)
    for qtype in QUERY_TYPES:
        if per_type[qtype] != counts.get(qtype, 0):
            problems.append(f"{qtype}: {per_type[qtype]} queries, expected {counts.get(qtype, 0)}")

    allocation = dev_allocation(counts)
    dev = Counter(query.type for query in queries if query.split == "dev")
    if sum(dev.values()) != sum(allocation.values()):
        problems.append(f"split: {sum(dev.values())} dev queries, expected {sum(allocation.values())}")
    for qtype in QUERY_TYPES:
        if dev[qtype] != allocation[qtype]:
            found = "no dev queries" if dev[qtype] == 0 else f"{dev[qtype]} dev queries"
            problems.append(f"split: {qtype} has {found}, expected {allocation[qtype]}")

    for label, values in (
        ("query id", [query.query_id for query in queries]),
        ("query text", [_normal(query.query) for query in queries]),
    ):
        for value, times in sorted(Counter(values).items()):
            if times > 1:
                problems.append(f"duplicate {label} ({times} times): {value}")

    dev = dev_slots(counts)
    slots = {
        config.stable_id("query", f"{qtype}:{index}"): (qtype, "dev" if index in dev[qtype] else "test")
        for qtype in QUERY_TYPES
        for index in range(counts.get(qtype, 0))
    }
    corpus = Corpus(people, conversations)
    for query in queries:
        found = _slot_problems(query, slots) + _query_problems(query, corpus)
        problems += [f"{query.query_id} ({query.type}): {problem}" for problem in found]
    return problems


def _slot_problems(query: Query, slots: Mapping[str, tuple[str, str]]) -> list[str]:
    """Is the query in a slot of these counts, with the split the seeded draw gives that slot?"""

    slot = slots.get(query.query_id)
    if slot is None:
        return ["the query id is not a slot of these per-type counts"]
    qtype, split = slot
    if qtype != query.type:
        return [f"the query id is a {qtype} slot"]
    if query.split != split:
        return [f"split is {query.split!r}, the seeded split puts this slot in {split!r}"]
    return []


def _consecutive_beats(corpus: Corpus, gold: Sequence[Conversation]) -> bool:
    """Are these conversations consecutive beats of one storyline?"""

    keys = {storyline_key(conversation) for conversation in gold}
    if len(keys) != 1:
        return False
    beats = corpus.storylines[keys.pop()]
    positions = sorted(beats.index(conversation) for conversation in gold)
    return positions == list(range(positions[0], positions[0] + len(positions)))


def _query_problems(query: Query, corpus: Corpus) -> list[str]:
    """What is wrong with one query's text and labels.

    The text is re-checked with the same ``question_problem`` rules the
    generator enforced, so a file edited or produced elsewhere passes the
    same gate as a generated one.

    A query has two label sets. The sources (``meta.source_episode_ids``) are
    what the draw fixed and the question was written from; the rules on the
    draw's shape (consecutive beats, one person per group member, one gold for
    a who-question, a specific temporal query's size, the paraphrase wording)
    hold for them. The gold ids start as the sources and grow when
    ``pool_labels`` finds other conversations that answer the question, so the
    rules on every gold conversation (in the corpus, the named person's, inside
    the time range, a recap's whole window) hold for all of them, and every
    gold id beyond the sources must be one pooling added.
    """

    found: list[str] = []
    if query.split not in ("dev", "test"):
        found.append(f"split is {query.split!r}")
    found += [f"gold episode {e} is not in the corpus" for e in query.gold_episode_ids if e not in corpus.conversation]
    gold = [corpus.conversation[e] for e in query.gold_episode_ids if e in corpus.conversation]
    source_ids = [str(e) for e in query.meta.get("source_episode_ids", query.gold_episode_ids)]
    found += [f"source episode {e} is not in the corpus" for e in source_ids if e not in corpus.conversation]
    sources = [corpus.conversation[e] for e in source_ids if e in corpus.conversation]
    subkind = query.meta.get("subkind")

    if query.answerable:
        if not query.gold_episode_ids:
            found.append("answerable but has no gold episode ids")
        if not (query.gold_answer or "").strip():
            found.append("answerable but has no gold answer")
        if not set(source_ids) <= set(query.gold_episode_ids):
            found.append("a source conversation is not gold")
        added = {str(e) for e in (query.meta.get("pooling") or {}).get("added", ())}
        if set(query.gold_episode_ids) - set(source_ids) != added:
            found.append("the gold conversations beyond the sources are not the ones pooling added")
    else:
        if query.gold_episode_ids:
            found.append("must have no gold episode ids")
        if query.gold_answer is not None:
            found.append("must have no gold answer")

    name, person_id = query.gold_name, query.gold_person_id
    if query.type == "unknown_name":
        if not name:
            found.append("an unknown-name query needs a gold_name")
        if person_id is not None or (name and corpus.resolve(name).status != "unknown"):
            found.append(f"the name {name!r} resolves to someone in the roster")
    elif query.type == "cross_person" or (query.type == "temporal" and person_id is None):
        if name is not None or person_id is not None:
            found.append("a no-name query must have gold_name and gold_person_id of None")
    else:
        if not name or person_id is None:
            found.append("a named query needs gold_name and gold_person_id")
        else:
            if corpus.resolve(name).person_id != person_id:
                found.append(f"the name {name!r} does not resolve to gold_person_id")
            if any(conversation.person_id != person_id for conversation in gold):
                found.append("a gold conversation belongs to another person")

    if query.type == "multi_conversation" and sources and not (len(sources) in MULTI_SIZES and _consecutive_beats(corpus, sources)):
        found.append(f"the sources are not {' or '.join(map(str, MULTI_SIZES))} consecutive beats of one storyline")
    if query.type == "cross_person" and sources:
        owners = [conversation.person_id for conversation in sources]
        if subkind == "group" and not (len(owners) in GROUP_SIZES and len(set(owners)) == len(owners)):
            found.append("a group query's sources must be with pairwise different people")
        if subkind == "single" and len(owners) != 1:
            found.append("a single who-question is written from exactly one source conversation")

    time_range = query.gold_time_range
    expression = None
    if query.type != "temporal":
        if time_range is not None:
            found.append("only temporal queries may have a gold_time_range")
    elif time_range is None:
        found.append("a temporal query needs a gold_time_range")
    else:
        if not all(time_range.contains(conversation.start_time) for conversation in gold):
            found.append("a gold conversation starts outside the gold_time_range")
        expression = corpus.expressions.get(str(query.meta.get("time_expression")))
        if expression is None:
            found.append(f"unknown time expression {query.meta.get('time_expression')!r}")
        elif expression.time_range != time_range:
            found.append(f"gold_time_range is not the range of {expression.key!r}")
        scope = corpus.conversations if person_id is None else corpus.by_person.get(person_id, ())
        inside = sorted(c.episode_id for c in scope if time_range.contains(c.start_time))
        if subkind == "recap" and sorted(query.gold_episode_ids) != inside:
            found.append("a recap's gold episode ids are not every conversation of its scope inside the gold_time_range")
        if subkind == "specific" and sources and not (len(sources) == 1 or (len(sources) == 2 and _consecutive_beats(corpus, sources))):
            found.append("a specific temporal query's sources are one conversation or two consecutive beats")
        padded = pad_time_range(time_range, config.NOW)
        pool_size = sum(1 for c in scope if padded.contains(c.start_time))
        if query.meta.get("pool_size") != pool_size:
            found.append(f"meta.pool_size is {query.meta.get('pool_size')!r}, the padded range holds {pool_size}")

    if not query.answerable or sources:
        target = Target(
            qtype=query.type,
            index=0,
            attempt=0,
            subkind=subkind,
            person=corpus.person.get(person_id or ""),
            name=name,
            sources=tuple(sources),
            expression=expression,
        )
        problem = question_problem(corpus, target, query.query, query.gold_answer)
        if problem is not None:
            found.append(f"the question fails the {problem!r} rule")
    return found


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class CallStats:
    """The replies one namespace of calls used."""

    calls: int = 0
    cached: int = 0  # of ``calls``, answered from the response cache at no charge
    cost_usd: float = 0.0  # what every reply cost when it was first made, cached ones included
    hosted: Usage = field(default_factory=Usage)  # the replies the gateway answered during this run

    def add(self, result: LLMResult) -> None:
        self.calls += 1
        self.cost_usd += result.usage.cost_usd
        if result.cached:
            self.cached += 1
        else:
            self.hosted = self.hosted + result.usage


class MeteredLLM:
    """Passes calls through to the run's LLM and counts what they used, per namespace, for the closing report.

    Shared by the corpus and the query generators, which read ``stats`` by the
    namespace of each kind of call. Safe to call from worker threads.
    """

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm
        self._lock = threading.Lock()
        self.stats: dict[str, CallStats] = {}  # by call namespace, in the order first used

    def complete_json(self, **kwargs: Any) -> LLMResult:
        result = self._llm.complete_json(**kwargs)
        with self._lock:
            self.stats.setdefault(str(kwargs.get("namespace", "default")), CallStats()).add(result)
        return result

    @property
    def hosted_calls(self) -> int:
        """Calls answered by the gateway during this run."""

        return sum(stats.calls - stats.cached for stats in self.stats.values())

    @property
    def cached_calls(self) -> int:
        """Calls answered from the response cache."""

        return sum(stats.cached for stats in self.stats.values())

    @property
    def hosted(self) -> Usage:
        """What the hosted calls used, over every namespace."""

        return sum((stats.hosted for stats in self.stats.values()), Usage())

    def report(self) -> str:
        hosted = self.hosted
        return (
            f"LLM calls: {self.hosted_calls} hosted, {self.cached_calls} from the cache; "
            f"hosted spend: ${hosted.cost_usd:.4f} "
            f"({hosted.input_tokens} input tokens, {hosted.output_tokens} output tokens)"
        )


def _counts_table(queries: Sequence[Query]) -> str:
    rows = [f"{'type':<20}{'queries':>8}{'dev':>6}{'test':>6}"]
    for qtype in (*QUERY_TYPES, "total"):
        of_type = [query for query in queries if qtype in (query.type, "total")]
        dev = sum(1 for query in of_type if query.split == "dev")
        rows.append(f"{qtype:<20}{len(of_type):>8}{dev:>6}{len(of_type) - dev:>6}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None, runtime: Runtime | None = None) -> int:
    """Generate (or partly regenerate) the query file; return the process exit code.

    A ``runtime`` is built from the settings unless one is injected.
    """

    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.data.queries",
        description="Generate the benchmark's queries from the conversation corpus.",
    )
    parser.add_argument("--workers", type=int, default=None, help="parallel LLM calls (default: EVAL_LLM_CONCURRENCY)")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=config.DATA_DIR,
        help="directory holding people.jsonl and conversations.jsonl",
    )
    parser.add_argument("--out", type=Path, default=config.QUERIES_PATH, help="query file to write")
    parser.add_argument(
        "--n-queries",
        type=int,
        default=config.N_QUERIES,
        help="total number of queries, split over the types in the benchmark's proportions "
        "(lower it for a corpus too small to carry the full set)",
    )
    parser.add_argument(
        "--regenerate",
        nargs="+",
        default=[],
        metavar="QUERY_ID",
        help="replace these queries of the existing --out file and keep every other query as it is",
    )
    args = parser.parse_args(argv)
    if args.n_queries < 1:
        parser.error("--n-queries must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from .roster import UNKNOWN_NAMES

    people = dataset.load_people(args.corpus_dir / config.PEOPLE_PATH.name)
    conversations = dataset.load_conversations(args.corpus_dir / config.CONVERSATIONS_PATH.name)
    counts = scaled_counts(args.n_queries)
    if runtime is None:
        from ..runtime import Runtime

        runtime = Runtime()
    llm = MeteredLLM(runtime.llm)
    try:
        generator = QueryGenerator(
            llm, runtime.settings, people, conversations, list(UNKNOWN_NAMES), counts, args.workers
        )
        if args.regenerate:
            queries = generator.regenerate(dataset.load_queries(path=args.out), args.regenerate)
        else:
            queries = generator.generate()
    except (QueryGenerationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        print(llm.report())
        return 1

    print(_counts_table(queries))
    print(llm.report())
    problems = validate_queries(queries, people, conversations, counts)
    if problems:
        print(f"{len(problems)} problems; {args.out} was not written:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    dataset.save_queries(queries, args.out)
    print(f"wrote {len(queries)} queries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
