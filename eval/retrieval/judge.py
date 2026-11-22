"""LLM judges for the benchmark: grade answers, label relevance, rewrite gold answers.

``judge_answer`` grades a system answer against the gold answer; the answer
accuracy metric is built on its verdicts. ``judge_relevance`` and
``write_gold_answer`` serve label pooling: the first finds conversations that
the dataset generator did not label as gold, the second rewrites the gold
answer once such conversations have joined the gold set.

Each of the three is one JSON call at temperature 0. Output that does not
parse into a valid result is asked for again under a different salt (with the
same salt the response cache would replay the invalid output): once under a
fixed retry salt, then once under a salt that is new every time, so a rerun
makes a fresh call instead of replaying two cached failures forever. A third
invalid output raises ``JudgeOutputError``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

from . import config
from .interfaces import ChatLLM, Usage
from .schema import OWNER_SPEAKER, VERDICTS, Conversation, Query
from .timeutil import describe_moment

NAMESPACE_ANSWER = "judge_answer"
NAMESPACE_RELEVANCE = "judge_relevance"
NAMESPACE_GOLD_ANSWER = "gold_answer"
RETRY_SALT = "retry"
ATTEMPTS = 3  # the call, the cached retry, and the uncached one
# Replies are short; the room is for the hidden reasoning tokens of a reasoning
# model, which count against the same limit.
MAX_TOKENS = 4096
OUTSIDE_WINDOW_REASON = "The conversation took place outside the question's time window."

T = TypeVar("T")

_OWNER = config.OWNER_DISPLAY_NAME
# Relative dates in a question or an answer ("last week") only mean something
# against the moment the question was asked, which is the benchmark's fixed clock.
_ASKED_AT = describe_moment(config.NOW)

# Also printed for the person who labels the validation sample (see
# judge_validation), so the judge and the human grade by one rubric.
ANSWER_RUBRIC = """\
A key fact is a fact in the gold answer that the question asks for.
- "correct": the system answer states the gold answer's key facts and contradicts none of them. \
Different wording is fine, and so is extra detail that does not contradict the gold answer.
- "partial": the system answer states some but not all of the key facts, or it mixes right facts \
with a wrong one (one that contradicts the gold answer).
- "incorrect": the system answer states none of the key facts: it is wrong, unrelated to the \
question, or a refusal such as "I don't know"."""

ANSWER_SYSTEM_PROMPT = f"""\
You grade answers from a personal-memory assistant. {_OWNER} asked the assistant a question about \
their own past conversations. You are given the question, the gold answer and the system answer \
(the assistant's reply), and you decide how well the system answer matches the gold answer.

The gold answer is the reference: treat it as true, and as complete for what the question \
asks, and grade only against it. Do not use outside knowledge, and give no credit for an answer \
that merely sounds plausible. "I", "you" and "{config.OWNER_FIRST_NAME}" can all refer to \
{_OWNER}. The question was asked at {_ASKED_AT}; use that only to compare relative dates such as \
"last Tuesday" with calendar dates.

{ANSWER_RUBRIC}

Reply with one JSON object and nothing else:
{{"reason": one short sentence, "verdict": "correct" | "partial" | "incorrect"}}"""

RELEVANCE_SYSTEM_PROMPT = f"""\
You label conversations for a retrieval benchmark. {_OWNER} asks a question about their own past \
conversations, and you decide whether one conversation is relevant to that question.

You are given the question, its gold answer when there is one, the time window it refers to when \
it has one, and one conversation between {_OWNER} and the person it names: its date, its summary \
and its full dialogue. In the dialogue, the speaker "{OWNER_SPEAKER}" is {_OWNER}. The question \
was asked at {_ASKED_AT}.

The conversation is relevant if it contains information that answers the question, or \
information that a complete answer must include. That information can be in the summary, in the \
dialogue, or in both. The conversation is not relevant if it merely shares the question's topic \
or person, or if its date falls outside the question's time window.

The gold answer shows what the question is after. It was written from the conversations already \
labelled relevant and may be incomplete, so do not require this conversation to match it: a \
relevant conversation may repeat the gold answer's facts or hold answer information that the \
gold answer lacks.

Reply with one JSON object and nothing else:
{{"reason": one short sentence, "relevant": true or false}}"""

GOLD_ANSWER_SYSTEM_PROMPT = f"""\
You write the reference answer for a retrieval benchmark. {_OWNER} asks a question about their \
own past conversations, and every conversation in the user message has been labelled as holding \
part of the answer.

Each conversation is between {_OWNER} and the person it names, and comes with its date and its \
summary; its full dialogue is shown as well only when the question asks for a detail that the \
summaries leave out. In a dialogue, the speaker "{OWNER_SPEAKER}" is {_OWNER}. The question was \
asked at {_ASKED_AT}. When a time window is given, it is the period the question refers to.

Write the answer the question asks for. It must be:
- complete: it includes every fact the question asks for, from all of these conversations;
- grounded: it states nothing that these conversations, as shown, do not say, and uses no \
outside knowledge;
- short: one or two plain sentences, a third only when the facts do not fit, with no preamble \
and no mention of the conversations or summaries as sources.

When a gold answer is given, it is the current one, written from only some of these \
conversations. Keep its facts where the conversations support them, drop what they do not \
support, and add what the other conversations contribute.

Reply with one JSON object and nothing else:
{{"answer": string}}"""

# Shown to the gold-answer writer for every query type but detail_only. The
# generator's gold answers for those types state only what the summaries state
# (query_prompts._ANSWER_FROM_SUMMARY), so the rewrite is held to the same rule:
# a rewritten answer that picked up a dialogue-only fact would grade every
# summary-path system answer as partial.
SUMMARIES_ONLY_NOTE = (
    "Only the summaries of these conversations are shown. The answer states only what the "
    "summaries state, as the gold answers for this kind of question always do; a fact that "
    "only a dialogue would hold does not go into it."
)
DIALOGUE_TYPE = "detail_only"  # the one query type whose gold answer comes from the dialogue


class JudgeOutputError(ValueError):
    """A judge returned invalid output on every attempt, the uncached one included."""


@dataclass
class AnswerVerdict:
    """The answer judge's grade for one system answer."""

    verdict: str  # one of VERDICTS
    reason: str  # the judge's one-sentence justification; empty if it gave none
    usage: Usage  # every attempt made, when the first output was invalid


@dataclass
class RelevanceVerdict:
    """Whether one conversation belongs in a query's gold set."""

    relevant: bool
    reason: str
    usage: Usage  # zero when the time window settled it without a call


def judge_answer(
    query: str,
    gold_answer: str,
    system_answer: str,
    llm: ChatLLM,
    model: str,
) -> AnswerVerdict:
    """Grade ``system_answer`` against ``gold_answer``.

    The judge sees the question and the two answers and nothing else: no
    summary and no dialogue, so the gold answer is its only reference.
    """

    if not gold_answer or not gold_answer.strip():
        raise ValueError("judge_answer needs a gold answer: only answerable queries can be graded")
    user = "\n\n".join(
        (
            _block("question", query),
            _block("gold_answer", gold_answer),
            _block("system_answer", system_answer),
        )
    )
    (verdict, reason), usage = _complete(
        llm, model, NAMESPACE_ANSWER, ANSWER_SYSTEM_PROMPT, user, _parse_answer_verdict
    )
    return AnswerVerdict(verdict, reason, usage)


def judge_relevance(
    query: Query,
    conversation: Conversation,
    llm: ChatLLM,
    model: str,
) -> RelevanceVerdict:
    """Decide whether ``conversation`` holds part of the answer to ``query``.

    Used to find gold labels the generator missed. The judge sees the question,
    its gold answer and time window when it has them, and the conversation's
    date, person name, summary and full dialogue.

    A conversation that starts outside the query's gold time range is not
    relevant by definition. That is a comparison of two timestamps, so it is
    settled here rather than left to the model: no call, zero usage.
    """

    window = query.gold_time_range
    if window is not None and not window.contains(conversation.start_time):
        return RelevanceVerdict(False, OUTSIDE_WINDOW_REASON, Usage())
    user = "\n\n".join((*_query_blocks(query), _conversation_block(conversation)))
    (relevant, reason), usage = _complete(
        llm, model, NAMESPACE_RELEVANCE, RELEVANCE_SYSTEM_PROMPT, user, _parse_relevance
    )
    return RelevanceVerdict(relevant, reason, usage)


def write_gold_answer(
    query: Query,
    conversations: Sequence[Conversation],
    llm: ChatLLM,
    model: str,
) -> tuple[str, Usage]:
    """Rewrite the gold answer of ``query`` from its full gold set.

    Called after pooling has added conversations to the gold set: the answer
    must then cover what they contribute. ``conversations`` is the whole gold
    set in any order (the prompt lists it oldest first, so one gold set always
    yields one prompt). The writer also sees the query's time window when it
    has one, and its current gold answer, written from the smaller set, as a
    starting point.

    The writer sees the dialogues only for a ``detail_only`` query. Every other
    type's gold answer was written from the summaries alone, and the rewrite
    keeps to them (:data:`SUMMARIES_ONLY_NOTE`), so a summary-path system
    answer is not later graded against a fact only a dialogue holds.
    """

    if not conversations:
        raise ValueError(f"query {query.query_id}: a gold answer needs at least one conversation")
    oldest_first = sorted(conversations, key=lambda item: (item.start_time, item.episode_id))
    with_dialogue = query.type == DIALOGUE_TYPE
    blocks = _query_blocks(query)
    if not with_dialogue:
        blocks.append(_block("note", SUMMARIES_ONLY_NOTE))
    blocks.extend(_conversation_block(item, with_dialogue=with_dialogue) for item in oldest_first)
    user = "\n\n".join(blocks)
    return _complete(
        llm, model, NAMESPACE_GOLD_ANSWER, GOLD_ANSWER_SYSTEM_PROMPT, user, _parse_gold_answer
    )


# --------------------------------------------------------------------------- #
# Prompt fields                                                                #
# --------------------------------------------------------------------------- #
def _block(tag: str, text: str) -> str:
    """One prompt field between ``<tag>`` lines, so the model sees where it starts and ends."""

    return f"<{tag}>\n{text.strip()}\n</{tag}>"


def _query_blocks(query: Query) -> list[str]:
    """The question, then its gold answer and its time window when it has them."""

    blocks = [_block("question", query.query)]
    if query.gold_answer:
        blocks.append(_block("gold_answer", query.gold_answer))
    window = query.gold_time_range
    if window is not None:
        span = f"{describe_moment(window.start)} to {describe_moment(window.end)}"
        blocks.append(_block("time_window", span))
    return blocks


def _conversation_block(conversation: Conversation, *, with_dialogue: bool = True) -> str:
    """One conversation's date, person and summary, and its dialogue unless told otherwise."""

    person_name = conversation.meta.get("person_name")
    if not person_name:
        raise ValueError(f"conversation {conversation.episode_id} has no meta['person_name']")
    fields = [
        _block("date", describe_moment(conversation.start_time)),
        _block("person", person_name),
        _block("summary", conversation.summary),
    ]
    if with_dialogue:
        fields.append(_block("dialogue", conversation.dialogue_text()))
    return _block("conversation", "\n".join(fields))


# --------------------------------------------------------------------------- #
# Calling and parsing                                                          #
# --------------------------------------------------------------------------- #
def _complete(
    llm: ChatLLM,
    model: str,
    namespace: str,
    system: str,
    user: str,
    parse: Callable[[dict[str, Any]], T | None],
) -> tuple[T, Usage]:
    """Call the model until ``parse`` accepts its JSON, at most ``ATTEMPTS`` times.

    The first two attempts run under fixed salts, so a resumed run replays
    them from the response cache. The third runs under a salt that is new
    every time, so it is never a cache hit: two invalid replies would
    otherwise be replayed on every rerun and fail the same query for good,
    and the cache has no way to evict them. The usage covers every attempt made.
    """

    usage = Usage()
    rejected: list[dict[str, Any]] = []
    for salt in ("", RETRY_SALT, f"{RETRY_SALT}:{uuid4().hex}"):
        result = llm.complete_json(
            model=model,
            system=system,
            user=user,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            namespace=namespace,
            salt=salt,
        )
        usage = usage + result.usage
        parsed = parse(result.data)
        if parsed is not None:
            return parsed, usage
        rejected.append(result.data)
    raise JudgeOutputError(f"{namespace}: invalid output on all {ATTEMPTS} attempts: {rejected!r}")


def _reason(data: dict[str, Any]) -> str:
    reason = data.get("reason")
    return reason.strip() if isinstance(reason, str) else ""


def _parse_answer_verdict(data: dict[str, Any]) -> tuple[str, str] | None:
    """``(verdict, reason)``, or None unless "verdict" is one of the three labels."""

    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().lower() not in VERDICTS:
        return None
    return verdict.strip().lower(), _reason(data)


def _parse_relevance(data: dict[str, Any]) -> tuple[bool, str] | None:
    """``(relevant, reason)``, or None unless "relevant" is a JSON boolean."""

    relevant = data.get("relevant")
    if not isinstance(relevant, bool):
        return None
    return relevant, _reason(data)


def _parse_gold_answer(data: dict[str, Any]) -> str | None:
    """The answer text, or None unless "answer" is a non-empty string."""

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    return answer.strip()
