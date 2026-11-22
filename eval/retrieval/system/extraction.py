"""Step 1 of the query flow: one LLM call that pulls a name and a time range out of the query."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .. import config
from ..interfaces import ChatLLM, Usage
from ..schema import TimeRange, parse_dt
from ..timeutil import EXTRACTION_TIME_RULES, describe_moment

NAMESPACE = "extraction"
# The JSON is tiny; the room is for the hidden reasoning tokens of a reasoning
# model, which count against the same limit.
MAX_TOKENS = 2048
# What a model writes in the name field when it means "nobody" but does not use a JSON null.
_NULL_LIKE_NAMES = frozenset({"null", "none", "nil", "n/a", "unknown", "undefined"})
_POSSESSIVE_ENDINGS = ("'s", "’s")  # with a straight or a typographic apostrophe

_SYSTEM_TEMPLATE = """\
You extract search filters from a question that a person asks about their own past conversations.

The current time is {now}.

Reply with one JSON object and nothing else:
{{"name": string or null, "time_range": {{"start": "YYYY-MM-DDTHH:MM:SSZ", \
"end": "YYYY-MM-DDTHH:MM:SSZ"}} or null}}

"name" is the person the asker had the conversation with.
- Copy the name exactly as it is written in the question: the full name when a surname is given, \
otherwise the first name alone. Leave out a possessive 's.
- Use null when the question does not name that person (for example "who told me about the \
climbing gym?").
- The asker writes in the first person ("I", "me", "my") and is never the name.

"time_range" is the period, in UTC, in which the conversation took place.
- Use null when the question does not limit when the conversation happened. A time that belongs \
to the topic (a trip planned for "next month") is not a limit.
- Otherwise convert the time expression with the time rules below, exactly as they define it. \
Do not add a margin: the range is widened later.

Time rules:
{time_rules}"""


@dataclass
class Extraction:
    """What the extraction call found in a query."""

    name: str | None  # as written in the query; None when it names nobody
    time_range: TimeRange | None  # unpadded; None when the query has no usable time expression
    usage: Usage
    raw: dict[str, Any]  # {"response": the model's JSON, "rejected": {field: why it was dropped}}


def extract(
    query: str,
    llm: ChatLLM,
    model: str,
    now: datetime = config.NOW,
    reasoning_effort: str | None = "low",
) -> Extraction:
    """Extract ``{name, time_range}`` from ``query`` with exactly one LLM call.

    A value the model returned but that cannot be used (a blank or null-like
    name, an unparseable or inverted range) becomes None, and the reason is
    kept under ``raw["rejected"]`` so the eval can tell it from a real null.
    """

    result = llm.complete_json(
        model=model,
        system=_SYSTEM_TEMPLATE.format(now=describe_moment(now), time_rules=EXTRACTION_TIME_RULES),
        user=f"Question: {query}",
        max_tokens=MAX_TOKENS,
        temperature=None,  # the model's default: a reasoning model takes no temperature
        reasoning_effort=reasoning_effort,
        namespace=NAMESPACE,
    )
    name, name_problem = _parse_name(result.data.get("name"))
    time_range, range_problem = _parse_time_range(result.data.get("time_range"))
    rejected = {
        field: problem
        for field, problem in (("name", name_problem), ("time_range", range_problem))
        if problem is not None
    }
    return Extraction(
        name=name,
        time_range=time_range,
        usage=result.usage,
        raw={"response": result.data, "rejected": rejected},
    )


def _parse_name(value: Any) -> tuple[str | None, str | None]:
    """Return ``(name, why the model's value was dropped)``.

    A usable name is tidied first: runs of whitespace collapse, and a possessive
    copied along with it ("Sam's") is cut, because the resolver matches whole
    words and "sam's" is not the word "sam".
    """

    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"name is not a string: {value!r}"
    name = " ".join(value.split())
    if name.lower().endswith(_POSSESSIVE_ENDINGS):
        name = name[:-2].rstrip()
    if not name:
        return None, "name is blank"
    if name.lower() in _NULL_LIKE_NAMES:
        return None, f"name is null-like: {name!r}"
    return name, None


def _parse_time_range(value: Any) -> tuple[TimeRange | None, str | None]:
    """Return ``(time_range, why the model's value was dropped)``."""

    if value is None:
        return None, None
    if not isinstance(value, dict):
        return None, f"time_range is not an object: {value!r}"
    start, end = value.get("start"), value.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        return None, "time_range needs a string start and a string end"
    try:
        parsed_start, parsed_end = parse_dt(start), parse_dt(end)
    except (ValueError, OverflowError) as error:
        return None, f"time_range has an unparseable timestamp: {error}"
    if parsed_end < parsed_start:
        return None, f"time_range ends before it starts: {start} > {end}"
    return TimeRange(start=parsed_start, end=parsed_end), None
