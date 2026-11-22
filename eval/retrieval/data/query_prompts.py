"""Prompts for the query generator: a writer prompt per query type and three checker prompts.

Every function here is a pure function of its arguments, so one target always
renders to the same bytes and a re-run is answered from the response cache.
The "... to use in the question" lines state, in a fixed form, the two
constraints that the generator's validators enforce in code.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence

from .. import config
from ..schema import Conversation, Person
from . import Prompt

NAME_LINE = 'Name to use in the question: "{name}"'
NO_NAME_LINE = "Name to use in the question: none"
TIME_LINE = 'Time phrase to use in the question: "{phrase}"'
NO_TIME_LINE = "Time phrase to use in the question: none"

_OWNER = config.OWNER_FIRST_NAME

_ROLE = (
    f"You write test questions for a personal memory assistant. The assistant belongs to "
    f"{config.OWNER_DISPLAY_NAME}, who talks with friends, family and colleagues and later asks the "
    f"assistant what was said. You are given material from those conversations and write one question "
    f'{_OWNER} could ask the assistant. In dialogues {_OWNER} speaks as "Me"; in the question {_OWNER} is "I".'
)
_QUESTION_RULES = "\n".join(
    (
        "How the question must read:",
        "- It is one short, natural sentence in the first person that ends with a question mark, the way "
        'someone asks their own assistant: "What did Sam Okafor say about ...?", "When is ...?", '
        '"Which restaurant did Hannah recommend ...?".',
        '- It never refers to the material: no "according to the conversation", and no mention of a '
        f"summary, a transcript, a recording or glasses. It never contains the name {_OWNER}.",
        "- It is specific: together with the name and time phrase it contains, it points at the source "
        "material and at nothing else.",
        '- It asks for information, never for confirmation: it is not a yes/no question ("Did Sam confirm '
        'the booking?", "Is her flight on Friday?") and it does not contain its own answer.',
        "- It follows the naming and time requirements of the task to the letter. Anyone other than the "
        'person you are told to name is referred to by role ("her sister", "his manager"), never by name.',
    )
)
_ANSWER_RULES = "\n".join(
    (
        "How the answer must read:",
        "- One to three complete sentences that give everything the question asks for.",
        "- It states only what the source material says: no outside knowledge and no guesses.",
    )
)
_DECLINE = (
    "If the material does not allow a question that meets every requirement, return "
    '{"question": null} rather than bend a requirement.'
)

WRITER_SYSTEM = "\n\n".join(
    (
        _ROLE,
        _QUESTION_RULES,
        _ANSWER_RULES,
        _DECLINE,
        'Return one JSON object and nothing else: {"question": "...", "answer": "..."}',
    )
)
WRITER_SYSTEM_NO_ANSWER = "\n\n".join(
    (_ROLE, _QUESTION_RULES, _DECLINE, 'Return one JSON object and nothing else: {"question": "..."}')
)
CHECKER_SYSTEM = (
    "You check test questions for a personal memory assistant. Its owner asks it about conversations "
    'the owner had with people they know; in dialogues the owner speaks as "Me". Judge strictly, and '
    "only from the material you are shown. Return one JSON object and nothing else."
)

_OTHERS = (
    "Other conversations from the same thread (summaries only). "
    "The question must not be answerable from any of them:"
)
_PERIOD_OTHERS = (
    "Other conversations from the same period (summaries only). These are what the assistant will "
    "look through next to the source. The question must not be answerable from any of them:"
)
_NOT_FROM_OTHERS = "Nothing in the other conversations listed answers the question."
_SUMMARY_IS_ENOUGH = (
    "The summary alone is enough to answer the question. The dialogue is shown so that you can check "
    "the summary against what was said; do not ask about anything the summary leaves out."
)
_ANSWER_FROM_SUMMARY = (
    "The answer states only what the summary states. Nothing that only the dialogue says (a price, a "
    "name, a number the summary leaves out) goes into the answer."
)
_ANSWER_FROM_SUMMARIES = (
    "The answer states only what the summaries state. Nothing that only the dialogues say (a price, a "
    "name, a number the summaries leave out) goes into the answer."
)
_LIKE_A_REAL_ONE = (
    "The question reads exactly like a question about a real conversation: specific, plausible for "
    "this person, and taking for granted that the conversation happened. It does not ask whether "
    "something was ever mentioned."
)
_PLACE_TIME_PHRASE = (
    "Place the time phrase so that it can only mean when we spoke, as in \"What did Sam tell me last week "
    'about the trip?", never next to a thing it could date ("her trip in March", "last week\'s booking").'
)
_FACTS_LINE = "Its dialogue also states"


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def _with(conversation: Conversation, people: Mapping[str, Person]) -> str:
    return people[conversation.person_id].display_name


def _numbered(texts: Sequence[str]) -> str:
    return "\n".join(f"{number}. {text}" for number, text in enumerate(texts, 1))


def _full(label: str, conversation: Conversation, people: Mapping[str, Person]) -> str:
    """One conversation with its summary and its dialogue."""

    return "\n".join(
        (
            f"{label} (with {_with(conversation, people)})",
            f"Summary: {conversation.summary.strip()}",
            "Dialogue:",
            conversation.dialogue_text(),
        )
    )


def _sources(conversations: Sequence[Conversation], people: Mapping[str, Person]) -> str:
    if len(conversations) == 1:
        return _full("Source conversation", conversations[0], people)
    total = len(conversations)
    return "\n\n".join(
        _full(f"Source conversation {number} of {total}", conversation, people)
        for number, conversation in enumerate(conversations, 1)
    )


def dialogue_facts(conversation: Conversation) -> list[str]:
    """What the dialogue states of what the corpus generator planted: its key facts, then its minor details.

    A 20-100 word summary leaves most of these out by construction, so they
    stand in for the dialogue wherever showing every dialogue is too much. A
    planted fact the dialogue never got round to saying (the generator records
    those as ``unstated_facts`` and ``unstated_details``) is left out: nothing
    in memory states it.
    """

    unstated = {
        str(item).strip()
        for key in ("unstated_facts", "unstated_details")
        for item in conversation.meta.get(key) or ()
    }
    facts: list[str] = []
    for key in ("key_facts", "minor_details"):
        for item in conversation.meta.get(key) or ():
            text = str(item).strip()
            if text and text not in facts and text not in unstated:
                facts.append(text)
    return facts


def _summary_line(conversation: Conversation, people: Mapping[str, Person], facts: bool) -> str:
    line = f"(with {_with(conversation, people)}) {conversation.summary.strip()}"
    stated = dialogue_facts(conversation) if facts else []
    return f"{line}\n   {_FACTS_LINE}: {'; '.join(stated)}" if stated else line


def _summaries(
    heading: str,
    conversations: Sequence[Conversation],
    people: Mapping[str, Person],
    facts: bool = False,
) -> str | None:
    """A numbered list of summaries (each with what its dialogue adds when ``facts``), or None when empty."""

    if not conversations:
        return None
    return f"{heading}\n{_numbered([_summary_line(conversation, people, facts) for conversation in conversations])}"


def _person_card(person: Person) -> str:
    relationship = f" ({person.relationship})" if person.relationship else ""
    lines = [f"Person: {person.display_name}{relationship}"]
    persona = person.meta.get("persona")
    if persona:
        text = persona if isinstance(persona, str) else json.dumps(persona, ensure_ascii=False, sort_keys=True)
        lines.append(f"About them: {text}")
    return "\n".join(lines)


def _own_summaries(person: Person, conversations: Sequence[Conversation], total: int) -> str:
    """Summaries of one person's conversations, saying whether the list is complete."""

    share = f"all {total}" if len(conversations) == total else f"{len(conversations)} of the {total}"
    heading = f"Summaries of {share} conversations I have had with {person.display_name}, oldest first:"
    return f"{heading}\n{_numbered([conversation.summary.strip() for conversation in conversations])}"


def _asked(name: str, questions: Sequence[str]) -> str | None:
    if not questions:
        return None
    listed = "\n".join(f"- {question}" for question in questions)
    return f"Questions already written about {name}. Ask about something clearly different:\n{listed}"


def _name_rule(name: str | None) -> str:
    if name is None:
        return "The question contains no person's name at all, neither a first name nor a full name."
    return f'The question contains "{name}" exactly as written, and no other person\'s name.'


def _time_rule(phrase: str | None) -> str:
    if phrase is None:
        return (
            "The question contains no date and no time word: no weekday, month or year, and nothing like "
            '"today", "yesterday", "last week", "recently" or "ago".'
        )
    return f'The question contains the time phrase "{phrase}" exactly as written, and no other date or time word.'


def _writer_user(
    kind: str,
    name: str | None,
    phrase: str | None,
    sections: Sequence[str | None],
    requirements: Sequence[str | None],
) -> str:
    """Lay out a writer task: the fixed header lines, the material, then the requirements."""

    header = "\n".join(
        (
            f"Question type: {kind}",
            NAME_LINE.format(name=name) if name is not None else NO_NAME_LINE,
            TIME_LINE.format(phrase=phrase) if phrase is not None else NO_TIME_LINE,
        )
    )
    rules = [rule for rule in requirements if rule] + [_name_rule(name), _time_rule(phrase)]
    listed = "\n".join(f"- {rule}" for rule in rules)
    return "\n\n".join([header, *(section for section in sections if section), f"Requirements:\n{listed}"])


# --------------------------------------------------------------------------- #
# Writer prompts                                                               #
# --------------------------------------------------------------------------- #
def write_lookup(
    name: str,
    source: Conversation,
    others: Sequence[Conversation],
    people: Mapping[str, Person],
) -> Prompt:
    """A question about one specific thing one conversation established."""

    user = _writer_user(
        "single-conversation lookup",
        name,
        None,
        [_sources([source], people), _summaries(_OTHERS, others, people)],
        [
            "The question asks about one specific thing this conversation established.",
            _SUMMARY_IS_ENOUGH,
            _ANSWER_FROM_SUMMARY,
            _NOT_FROM_OTHERS if others else None,
        ],
    )
    return Prompt(WRITER_SYSTEM, user)


def write_paraphrase(
    name: str,
    source: Conversation,
    others: Sequence[Conversation],
    people: Mapping[str, Person],
    avoid: Sequence[str],
) -> Prompt:
    """A lookup whose wording stays away from the summary's (``avoid`` lists the summary's content words)."""

    user = _writer_user(
        "lookup in different words",
        name,
        None,
        [_sources([source], people), _summaries(_OTHERS, others, people)],
        [
            "The question asks about one specific thing this conversation established.",
            _SUMMARY_IS_ENOUGH,
            "The question does not reuse the summary's wording. Apart from the name, at most one meaningful "
            "word of the question may also occur in the summary, in any inflected form. Say things in other "
            "words (a synonym, or a description of what the thing is) and keep the question natural and "
            f"unambiguous. Words to avoid: {', '.join(avoid)}.",
            _ANSWER_FROM_SUMMARY,
            _NOT_FROM_OTHERS if others else None,
        ],
    )
    return Prompt(WRITER_SYSTEM, user)


def write_detail_only(
    name: str,
    source: Conversation,
    others: Sequence[Conversation],
    people: Mapping[str, Person],
    details: Sequence[str],
) -> Prompt:
    """A question whose answer the dialogue states and the summary leaves out.

    The other beats of the thread are shown with what their dialogues state,
    so that the detail asked for is not one the thread repeats.
    """

    hints = None
    if details:
        listed = "\n".join(f"- {detail}" for detail in details)
        hints = (
            "Details the dialogue states that the summary seems to leave out "
            f"(prefer one of these, after checking it against both texts):\n{listed}"
        )
    heading = (
        "Other conversations from the same thread (summaries, each with what its dialogue states beyond "
        "the summary). The question must not be answerable from any of them:"
    )
    user = _writer_user(
        "detail that the summary leaves out",
        name,
        None,
        [_sources([source], people), hints, _summaries(heading, others, people, facts=True)],
        [
            "The question asks for one specific detail that the dialogue states and the summary does not. "
            "Someone who has read only the summary must be unable to answer it, even in part.",
            "The question may mention the subject of the conversation so that it is clear which one is "
            "meant, as long as the summary still does not give the answer.",
            "The answer gives the detail exactly as the dialogue states it.",
            "No other conversation listed states the detail, neither in its summary nor in what its "
            "dialogue states: this conversation is the only one that answers the question."
            if others
            else None,
        ],
    )
    return Prompt(WRITER_SYSTEM, user)


def write_multi(
    name: str,
    sources: Sequence[Conversation],
    others: Sequence[Conversation],
    people: Mapping[str, Person],
    title: str,
) -> Prompt:
    """A question that needs every one of several consecutive conversations of one thread."""

    user = _writer_user(
        "question that needs several conversations",
        name,
        None,
        [
            f'The {len(sources)} source conversations are consecutive conversations from one thread ("{title}"), '
            "oldest first.",
            _sources(sources, people),
            _summaries(_OTHERS, others, people),
        ],
        [
            "The question needs information from every source conversation: how something developed, what "
            "changed from one conversation to the next, or a list that spans them. No single one of them is "
            "enough to answer it.",
            "The summaries together are enough to answer the question. The dialogues are shown so that you "
            "can check the summaries against what was said.",
            "The answer draws on each source conversation.",
            _ANSWER_FROM_SUMMARIES,
            _NOT_FROM_OTHERS if others else None,
        ],
    )
    return Prompt(WRITER_SYSTEM, user)


def write_cross_single(
    source: Conversation,
    others: Sequence[Conversation],
    people: Mapping[str, Person],
) -> Prompt:
    """A who-question with no name, answered by one conversation's person."""

    heading = (
        "Other conversations, some with other people (summaries only). "
        "The question must not be answerable from any of them:"
    )
    user = _writer_user(
        "who-question without a name",
        None,
        None,
        [_sources([source], people), _summaries(heading, others, people)],
        [
            "The question asks who, or which person, did or said something this conversation established, "
            'for example "Who told me about ...?" or "Which friend is training for ...?".',
            _SUMMARY_IS_ENOUGH,
            f"The answer names {_with(source, people)} and states the fact.",
            _ANSWER_FROM_SUMMARY,
            "The question fits this person only: nothing in the other conversations listed answers it."
            if others
            else None,
        ],
    )
    return Prompt(WRITER_SYSTEM, user)


def write_cross_group(
    sources: Sequence[Conversation],
    others: Sequence[Conversation],
    people: Mapping[str, Person],
    topic: str,
) -> Prompt:
    """A who-question with no name whose answer is every person of a shared-topic group."""

    names = ", ".join(_with(conversation, people) for conversation in sources)
    heading = (
        "Conversations with other people on the same general topic, and these people's other "
        "conversations (summaries only). The question must not fit any of them:"
    )
    user = _writer_user(
        "who-question across several people",
        None,
        None,
        [
            f'Each of the {len(sources)} source conversations is with a different person; all of them touch on "{topic}".',
            _sources(sources, people),
            _summaries(heading, others, people),
        ],
        [
            "The question asks who I talked to about one specific aspect that all the source conversations "
            'have in common, for example "Who have I talked to about ...?".',
            "The summaries are enough to see that each source conversation fits the question.",
            f"The answer names every one of these people ({names}) and says in a few words what each said.",
            _ANSWER_FROM_SUMMARIES,
            "The aspect is specific to the source conversations: none of the other conversations listed fits "
            "it, including the same people's other conversations."
            if others
            else None,
        ],
    )
    return Prompt(WRITER_SYSTEM, user)


def write_temporal(
    name: str | None,
    phrase: str,
    subkind: str,
    sources: Sequence[Conversation],
    period: Sequence[Conversation],
    thread: Sequence[Conversation],
    people: Mapping[str, Person],
) -> Prompt:
    """A question carrying a time phrase: a recap of the period, or a lookup inside it.

    ``period`` lists the other conversations the time filter leaves next to the
    sources, which the question must not fit; ``thread`` lists further beats of
    the sources' storylines from outside the period.
    """

    count = len(sources)
    plural = "s" if count > 1 else ""
    others = [*period, *thread]
    if subkind == "recap":
        if name is None:
            setting = f"The source material is every conversation I had {phrase}, with anyone ({count})."
            rules = [
                f"The question asks who I talked to {phrase}. It may also ask what about, but it does not "
                "single out a topic.",
                "The answer names each person and says in a few words what that conversation was about.",
                _ANSWER_FROM_SUMMARIES,
            ]
        else:
            person = _with(sources[0], people)
            setting = f"The source material is every conversation I had with {person} {phrase} ({count})."
            rules = [
                f"The question asks in general terms what {name} and I talked about, or what {name} told me, "
                f"{phrase}. It does not single out a topic.",
                "The answer covers every source conversation.",
                _ANSWER_FROM_SUMMARIES,
            ]
    else:
        setting = f"The source conversation{plural} took place {phrase}."
        rules = [
            f"The question asks about one specific thing the source conversation{plural} established, as an "
            "ordinary lookup would, and carries the time phrase.",
            "The summaries are enough to answer the question. The dialogues are shown so that you can check "
            "the summaries against what was said.",
            _ANSWER_FROM_SUMMARIES if count > 1 else _ANSWER_FROM_SUMMARY,
            "The question needs both source conversations." if count > 1 else None,
            "The question asks who did or said something, or asks for a fact, in a way that fits only "
            f"the source conversation{plural}; the answer names the person and states the fact."
            if name is None
            else None,
            _NOT_FROM_OTHERS if others else None,
        ]
    rules.append(_PLACE_TIME_PHRASE)
    user = _writer_user(
        "question with a time phrase",
        name,
        phrase,
        [
            setting,
            _sources(sources, people),
            _summaries(_PERIOD_OTHERS, period, people),
            _summaries(_OTHERS, thread, people),
        ],
        rules,
    )
    return Prompt(WRITER_SYSTEM, user)


def write_absent_topic(
    name: str,
    person: Person,
    threads: Sequence[tuple[str, int]],
    summaries: Sequence[Conversation],
    total: int,
    asked: Sequence[str],
) -> Prompt:
    """An unanswerable question: about something the person never discussed with the owner."""

    listed = "\n".join(f"- {title} ({size} conversation{'s' if size != 1 else ''})" for title, size in threads)
    user = _writer_user(
        "question that cannot be answered: a topic that never came up",
        name,
        None,
        [
            _person_card(person),
            f"Every thread of conversation I have had with {person.display_name}:\n{listed}",
            _own_summaries(person, summaries, total),
            _asked(name, asked),
        ],
        [
            f"The question asks about something {person.display_name} and I never discussed. Its subject "
            "belongs to none of the threads above, appears in none of the summaries, and is not a detail of "
            "any of them.",
            _LIKE_A_REAL_ONE,
        ],
    )
    return Prompt(WRITER_SYSTEM_NO_ANSWER, user)


def write_missing_detail(
    name: str,
    person: Person,
    title: str,
    shown: Sequence[Conversation],
    thread_size: int,
    summaries: Sequence[Conversation],
    total: int,
    asked: Sequence[str],
) -> Prompt:
    """An unanswerable question: a detail of a real thread that no conversation states."""

    share = "all of its conversations" if len(shown) == thread_size else f"{len(shown)} of its {thread_size} conversations"
    dialogues = "\n\n".join(
        f"Conversation {number} of the thread\nSummary: {conversation.summary.strip()}\n"
        f"Dialogue:\n{conversation.dialogue_text()}"
        for number, conversation in enumerate(shown, 1)
    )
    user = _writer_user(
        "question that cannot be answered: a detail that was never stated",
        name,
        None,
        [
            _person_card(person),
            f'One thread of conversation with {person.display_name}: "{title}". Below are {share}, oldest first.',
            dialogues,
            _own_summaries(person, summaries, total),
            _asked(name, asked),
        ],
        [
            f'The question asks for one specific detail about the thread "{title}" that no conversation '
            "states: a name, a date, a price, a place, a number or a reason that never came up and cannot be "
            "worked out from what was said.",
            "It is the kind of detail that could well have been mentioned.",
            _LIKE_A_REAL_ONE,
        ],
    )
    return Prompt(WRITER_SYSTEM_NO_ANSWER, user)


def write_unknown_name(name: str, model: Conversation) -> Prompt:
    """A lookup-style question about a person who is not in the owner's memory."""

    user = _writer_user(
        "question about a person",
        name,
        None,
        [
            "Model conversation (summary). Use it only as a model for the subject and the style of the "
            f"question:\n{model.summary.strip()}"
        ],
        [
            f'The question is about a person called "{name}". It asks about something the model conversation '
            f"covers, as if I had had that conversation with {name}.",
            "The question reads like an ordinary lookup about a real conversation.",
            "The question does not contain the name of the person in the model conversation.",
        ],
    )
    return Prompt(WRITER_SYSTEM_NO_ANSWER, user)


# --------------------------------------------------------------------------- #
# Checker prompts                                                              #
# --------------------------------------------------------------------------- #
def check_summaries(
    question: str,
    answer: str,
    sources: Sequence[Conversation],
    people: Mapping[str, Person],
    phrase: str | None = None,
) -> Prompt:
    """Is the question answerable from the sources' summaries, and does each source contribute?

    With ``phrase`` (a temporal query's time phrase) the checker also judges
    whether the phrase dates the conversation rather than something discussed.
    """

    shown = "\n\n".join(
        f"Conversation {number} (with {_with(conversation, people)})\nSummary: {conversation.summary.strip()}"
        for number, conversation in enumerate(sources, 1)
    )
    if phrase is None:
        framing = "A name in the question refers to the person shown."
        phrase_verdict = ""
        phrase_key = ""
    else:
        framing = (
            f'The time phrase "{phrase}" is satisfied as far as dates go: every conversation shown took place '
            f"{phrase}. Judge only how the phrase reads. A name in the question refers to the person shown."
        )
        phrase_verdict = (
            f'\n- "time_phrase_dates_the_conversation": true only if "{phrase}" can be read solely as when the '
            'conversation took place ("What did Sam tell me last week about the trip?"); false if it can be '
            'read as dating something that was discussed ("her trip last week", "last week\'s booking").'
        )
        phrase_key = '"time_phrase_dates_the_conversation": true or false, '
    user = "\n\n".join(
        (
            f"Question: {question}\nProposed answer: {answer}",
            shown,
            framing,
            "Decide:\n"
            '- "answerable": true if these summaries alone are enough to answer the question fully and '
            "correctly; false if the answer needs something the summaries do not state.\n"
            '- "answer_agrees": true only if the proposed answer answers the question and every claim in it '
            "is stated by the summaries; false if any part of it goes beyond or against them.\n"
            '- "contributing": the numbers of the conversations whose summary supplies something the proposed '
            f"answer needs. Leave out a conversation the answer could do without.{phrase_verdict}",
            'Return: {"answerable": true or false, "answer_agrees": true or false, '
            f'"contributing": [numbers], {phrase_key}"reason": "one short sentence"}}',
        )
    )
    return Prompt(CHECKER_SYSTEM, user)


def check_detail(
    question: str,
    answer: str,
    source: Conversation,
    others: Sequence[Conversation],
    people: Mapping[str, Person],
) -> Prompt:
    """Does the answer sit in the dialogue, not in the summary, and in no other conversation of the thread?

    The other beats are shown in full: on the dialogue fallback path the
    system reads their dialogues too, so a detail they repeat would make them
    answer the question as well.
    """

    listed = [
        _full(f"Other conversation {number} of {len(others)} from the same thread", conversation, people)
        for number, conversation in enumerate(others, 1)
    ]
    elsewhere = (
        '\n- "stated_elsewhere": true if the summary or dialogue of any other conversation shown also states '
        "the answer, fully or in its essentials."
        if others
        else ""
    )
    elsewhere_key = '"stated_elsewhere": true or false, ' if others else ""
    user = "\n\n".join(
        (
            f"Question: {question}\nProposed answer: {answer}",
            _full("Conversation", source, people),
            *listed,
            "Decide:\n"
            '- "answerable_from_summary": true if someone who has read only the summary of the conversation '
            "could answer the question, fully or in its essentials; false only if the summary does not "
            "contain the answer.\n"
            '- "answerable_from_dialogue": true if the dialogue of the conversation states the answer.\n'
            '- "answer_matches_dialogue": true if the proposed answer is correct and complete according to '
            f"that dialogue.{elsewhere}",
            'Return: {"answerable_from_summary": true or false, "answerable_from_dialogue": true or false, '
            f'"answer_matches_dialogue": true or false, {elsewhere_key}"reason": "one short sentence"}}',
        )
    )
    return Prompt(CHECKER_SYSTEM, user)


def check_unanswerable(
    question: str,
    person: Person,
    conversations: Sequence[Conversation],
    with_dialogue: Collection[str],
) -> Prompt:
    """Can the question be answered from anything the owner's memory holds about the person?

    Every conversation is shown as its summary plus what its dialogue states
    beyond the summary, so that a detail only a dialogue holds counts as held
    even when that dialogue is not shown in full.
    """

    dialogues = [
        f"Dialogue of conversation {number}:\n{conversation.dialogue_text()}"
        for number, conversation in enumerate(conversations, 1)
        if conversation.episode_id in with_dialogue
    ]
    if len(dialogues) == len(conversations):
        coverage = "the full dialogue of every one of them"
    elif dialogues:
        coverage = "the full dialogues of the conversations the question is closest to"
    else:
        coverage = "no full dialogues"
    entries = []
    for conversation in conversations:
        facts = dialogue_facts(conversation)
        summary = conversation.summary.strip()
        entries.append(f"{summary}\n   {_FACTS_LINE}: {'; '.join(facts)}" if facts else summary)
    user = "\n\n".join(
        (
            f"Question: {question}",
            f"The question is about {person.display_name}. Below is what the assistant holds on the owner's "
            f"conversations with {person.display_name}: every conversation, oldest first, as its summary "
            f"and the facts its dialogue states beyond the summary, and {coverage}.",
            _numbered(entries),
            *dialogues,
            "Decide:\n"
            '- "answerable": true if anything shown states the answer or makes it clear enough to answer the '
            "question, even in part; false if the material does not contain the answer.",
            'Return: {"answerable": true or false, "evidence": "the sentence that answers it, or an empty string"}',
        )
    )
    return Prompt(CHECKER_SYSTEM, user)
