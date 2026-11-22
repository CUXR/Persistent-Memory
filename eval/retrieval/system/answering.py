"""Step 5 of the query flow: answer from the top-k summaries, then from full dialogues.

Both attempts are one LLM call each and share a system prompt. The dialogue
attempt repeats the summaries block of the summary attempt verbatim before the
dialogues, so the model keeps what it already saw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .. import config
from ..interfaces import ChatLLM, LLMResult, Usage
from ..schema import OWNER_SPEAKER, Conversation, ConversationHeader
from ..timeutil import describe_moment

NAMESPACE_SUMMARY = "answer_summary"
NAMESPACE_DIALOGUE = "answer_dialogue"
# Answers are short; the room is for the hidden reasoning tokens of a reasoning
# model, which count against the same limit.
MAX_TOKENS = 4096

_SYSTEM_TEMPLATE = """\
You answer a question that {owner} asks about their own past conversations.

The current time is {now}. Use it to resolve relative dates such as "last week" against the \
dates of the conversations.

Use only the material in the user message: summaries of {owner}'s conversations and, when \
included, their full dialogues. Each conversation is between {owner} and the person it names. \
In a dialogue, the speaker "{owner_speaker}" is {owner}.

If the material contains the answer, set "can_answer" to true and give the answer in one or two \
short, specific sentences. If it does not, set "can_answer" to false and leave "answer" empty. \
Never guess, and never use knowledge from outside the material.

Reply with one JSON object and nothing else:
{{"can_answer": true or false, "answer": string}}"""


@dataclass
class AnswerAttempt:
    """One answering call."""

    can_answer: bool  # the model said it can answer and gave a non-empty answer
    answer: str  # the model's answer text, stripped (whatever it wrote, even when declining)
    usage: Usage
    raw: dict[str, Any]  # the model's JSON


def answer_from_summaries(
    query: str,
    headers: Sequence[ConversationHeader],
    llm: ChatLLM,
    model: str,
    now: datetime = config.NOW,
) -> AnswerAttempt:
    """First attempt: answer from the top-k summaries (``headers`` in rank order)."""

    result = llm.complete_json(
        model=model,
        system=_system_prompt(now),
        user=f"{_summaries_block(headers)}\n\nQuestion: {query}",
        max_tokens=MAX_TOKENS,
        temperature=None,  # the model's default: a reasoning model takes no temperature
        namespace=NAMESPACE_SUMMARY,
    )
    return _attempt(result)


def answer_with_dialogues(
    query: str,
    headers: Sequence[ConversationHeader],
    conversations: Sequence[Conversation],
    llm: ChatLLM,
    model: str,
    now: datetime = config.NOW,
) -> AnswerAttempt:
    """Second attempt: the same top-k summaries plus the full dialogues of the top k'.

    ``conversations`` are in rank order and may outnumber ``headers`` (k' > k).
    A dialogue with someone who has no summary among ``headers`` must carry
    their name in ``meta["person_name"]``, as the records a store returns do.
    """

    names = {header.person_id: header.person_name for header in headers}
    dialogues = "\n\n".join(
        f"{_label(position, conversation.start_time, _person_name(conversation, names))}\n"
        f"{conversation.dialogue_text()}"
        for position, conversation in enumerate(conversations, start=1)
    )
    result = llm.complete_json(
        model=model,
        system=_system_prompt(now),
        user=(
            f"{_summaries_block(headers)}\n\n"
            "Full dialogues of the top conversations, numbered like the summaries:\n\n"
            f"{dialogues}\n\n"
            f"Question: {query}"
        ),
        max_tokens=MAX_TOKENS,
        temperature=None,  # the model's default: a reasoning model takes no temperature
        namespace=NAMESPACE_DIALOGUE,
    )
    return _attempt(result)


def _system_prompt(now: datetime) -> str:
    return _SYSTEM_TEMPLATE.format(
        owner=config.OWNER_DISPLAY_NAME,
        owner_speaker=OWNER_SPEAKER,
        now=describe_moment(now),
    )


def _summaries_block(headers: Sequence[ConversationHeader]) -> str:
    summaries = "\n\n".join(
        f"{_label(position, header.start_time, header.person_name)}\n{header.summary}"
        for position, header in enumerate(headers, start=1)
    )
    return f"Conversation summaries, most relevant first:\n\n{summaries}"


def _label(position: int, start_time: datetime, person_name: str) -> str:
    """The line above a summary or a dialogue: its rank, its date, and who it was with."""

    return f"[{position}] {describe_moment(start_time)} | with {person_name}"


def _person_name(conversation: Conversation, names: Mapping[str, str]) -> str:
    """The name to show above a dialogue; ``names`` maps person id to name for the summaries.

    A ``Conversation`` holds its person's id, not their name. A summary with the
    same person supplies it. Otherwise (a dialogue ranked below every summary,
    in a pool of more than one person) it is the ``meta["person_name"]`` that
    the stores add in ``get_conversations``. With neither, sending the dialogue
    unlabelled would quietly change what the model is asked, so that is an error.
    """

    name = names.get(conversation.person_id) or conversation.meta.get("person_name")
    if not name:
        raise ValueError(
            f"no name for the person in conversation {conversation.episode_id}: none of the "
            'summaries is with them and the record has no meta["person_name"]'
        )
    return name


def _attempt(result: LLMResult) -> AnswerAttempt:
    """Read ``{"can_answer", "answer"}``; only a yes that comes with an answer counts."""

    answer = result.data.get("answer")
    text = answer.strip() if isinstance(answer, str) else ""
    return AnswerAttempt(
        can_answer=_is_yes(result.data.get("can_answer")) and bool(text),
        answer=text,
        usage=result.usage,
        raw=result.data,
    )


def _is_yes(value: Any) -> bool:
    """JSON ``true``, or the string ``"true"`` that a model sometimes writes instead."""

    return value is True or (isinstance(value, str) and value.strip().lower() == "true")
