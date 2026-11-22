"""How Jev is applied to score candidate conversations for a query.

Jev is a hosted decision model: a request carries one ``state`` and a set of
typed questions, and each question is answered on its own against that state.
There is more than one reasonable way to turn that into a relevance score, and
which one works best is part of what the benchmark studies. So the application
is an explicit ``JevScoringConfig`` rather than something buried in the rankers:

* mode ``pair`` (the vendor's rerank recipe): one request per candidate, with
  the query and that candidate's summary in the state and a single question.
* mode ``batched``: the query alone is the state, billed once per request, and
  each request carries up to ``batch_size`` questions, one per candidate, with
  the candidate's summary appended to the question's instructions.
* mode ``listwise``: one ``choice`` question whose options are the candidates;
  a candidate's score is the probability of its option.

In ``pair`` and ``batched`` mode every candidate is judged on its own with one
of two primitives: ``noul`` (a yes/no probability) or ``score`` (a four-level
rubric, rescaled to [0, 1]).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
from typing import Any

from ..interfaces import DecisionClient, DecisionResult, Usage

MODES: tuple[str, ...] = ("pair", "batched", "listwise")
PRIMITIVES: tuple[str, ...] = ("noul", "score")
MAX_CHOICE_OPTIONS = 255  # Jev's limit on the options of one choice question
MISSING_SCORE = -1.0  # below every real score (all lie in [0, 1]), so it sorts last

# Stated in every question: what the job is, in plain words.
TASK = (
    "The owner of a personal memory has asked a question about their past conversations, each "
    "of which is stored as a short summary. The task is to find which of the owner's past "
    "conversations answer the owner's question."
)

# Where the question and the summary are, which depends on the mode.
_LAYOUT: dict[str, str] = {
    "pair": (
        'The state holds the owner\'s question ("query") and the summary of one past '
        'conversation ("conversation_summary").'
    ),
    "batched": (
        'The state holds the owner\'s question ("query"). The summary of one past conversation '
        'is given alongside this instruction ("conversation_summary").'
    ),
    "listwise": (
        'The state holds the owner\'s question ("query"). Each option is the summary of one '
        "past conversation."
    ),
}

# The ``score`` rubric: four situations, least relevant first (level 0).
SCORE_LEVELS: tuple[str, ...] = (
    "The conversation is unrelated to the question.",
    "The conversation involves the same person or topic as the question, but does not help "
    "answer it.",
    "The conversation contains part of what is needed to answer the question.",
    "The conversation directly contains the answer to the question.",
)

_LISTWISE_QUESTION = "pick"
# Batched mode: the summary follows the instruction text after a blank line, under
# the name the layout wording gives it. Instructions stay a plain string, which
# both gateways accept and send identically; a structured object would reach
# OpenRouter as JSON and Vercel as JSON text, two different questions.
_BATCHED_SUMMARY_LABEL = "conversation_summary: "


@dataclass(frozen=True)
class Wording:
    """One phrasing of the relevance judgement, for each question type."""

    noul: str  # the yes/no question
    noul_true: str
    noul_false: str
    score: str  # asked with the SCORE_LEVELS rubric
    choice: str  # asked over all candidates at once


WORDINGS: dict[str, Wording] = {
    # v1: does the summary hold information that answers the question or is needed for it?
    "v1": Wording(
        noul=(
            "Does this conversation's summary contain information that answers the owner's "
            "question, or that is needed to answer it?"
        ),
        noul_true=(
            "The summary contains information that answers the question or is needed to "
            "answer it."
        ),
        noul_false=(
            "The summary contains nothing that answers the question or is needed to answer it."
        ),
        score="How far does this conversation's summary go towards answering the owner's question?",
        choice=(
            "Which conversation's summary contains the information that answers the owner's "
            "question?"
        ),
    ),
    # v2, stricter: is the question about this conversation, not merely near it?
    "v2": Wording(
        noul=(
            "Is this a conversation the owner's question is actually about, rather than one that "
            "merely shares a topic or a person with it?"
        ),
        noul_true=(
            "The question is about this conversation: what was said in it is what the owner is "
            "asking for."
        ),
        noul_false=(
            "The conversation is unrelated to the question, or it merely shares a topic or a "
            "person with it."
        ),
        score=(
            "How far does this conversation's summary go towards answering the owner's question? "
            "Judge strictly: a conversation that merely shares a topic or a person with the "
            "question does not help answer it."
        ),
        choice=(
            "Which conversation is the owner's question actually about? Do not choose one that "
            "merely shares a topic or a person with the question."
        ),
    ),
}


@dataclass(frozen=True)
class JevScoringConfig:
    """One way of applying Jev to a candidate list (see the module docstring).

    ``concurrency`` is the number of requests in flight at once: a client
    setting that changes how fast the scores arrive, never what they are.
    ``salt`` is passed to the client untouched, so a benchmark run can force
    fresh calls instead of reading the response cache; it does not change what
    is asked either.
    """

    mode: str = "batched"
    primitive: str = "noul"
    wording: str = "v1"
    batch_size: int = 16
    concurrency: int = 8
    salt: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"primitive must be one of {PRIMITIVES}, got {self.primitive!r}")
        if self.wording not in WORDINGS:
            raise ValueError(f"wording must be one of {tuple(WORDINGS)}, got {self.wording!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")

    @property
    def label(self) -> str:
        """Short stable name of this variant for traces and reports.

        It names what is asked of Jev and nothing else. So it leaves out
        ``salt`` (runs that differ only in salt are repeats of one variant),
        ``concurrency`` (the same requests, sent faster or slower) and any
        field the mode does not use: two configs that ask Jev the same thing
        share a label.
        """

        if self.mode == "listwise":
            return f"listwise-{self.wording}"
        parts = [self.mode, self.primitive, self.wording]
        if self.mode == "batched":
            parts.append(f"b{self.batch_size}")
        return "-".join(parts)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "JevScoringConfig":
        """Inverse of ``to_json``; absent fields take their defaults."""

        unknown = sorted(set(data) - {spec.name for spec in fields(cls)})
        if unknown:
            raise ValueError(f"unknown JevScoringConfig fields: {unknown}")
        return cls(**data)


DEFAULT_SCORING = JevScoringConfig()


@dataclass
class JevScores:
    """Jev's scores for one candidate list, and what getting them took.

    ``model`` is the snapshot the API reported: comma-joined if the requests
    were served by more than one, empty if nothing was sent. ``n_cached``
    counts requests answered from the client's response cache; when it is not
    zero, the wall clock of the call says nothing about Jev's latency.
    """

    scores: dict[str, float]  # episode_id -> score in [0, 1], or MISSING_SCORE
    usage: Usage
    n_requests: int
    n_missing: int
    model: str
    n_cached: int = 0


_Request = tuple[dict[str, Any], dict[str, dict[str, Any]]]  # (state, questions by name)


def score_candidates(
    client: DecisionClient,
    query: str,
    candidates: Sequence[tuple[str, str]],
    config: JevScoringConfig,
) -> JevScores:
    """Score ``(episode_id, summary)`` candidates for ``query`` with Jev.

    Requests run concurrently on ``config.concurrency`` threads (so the client
    must be thread-safe) and are read back in request order: the result does
    not depend on which request finishes first. A candidate whose answer is a
    refusal, is absent, or carries no usable number scores ``MISSING_SCORE``
    and is counted in ``n_missing``.
    """

    episode_ids = [episode_id for episode_id, _ in candidates]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("candidates must have distinct episode ids")
    if config.mode == "listwise" and len(candidates) > MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"listwise scoring takes at most {MAX_CHOICE_OPTIONS} candidates "
            f"(one choice question), got {len(candidates)}"
        )
    if not candidates:
        return JevScores(scores={}, usage=Usage(), n_requests=0, n_missing=0, model="")

    requests = _build_requests(query, [summary for _, summary in candidates], config)
    results = _send(client, requests, config)
    values = _read_scores(requests, results, config)
    return JevScores(
        scores={
            episode_id: MISSING_SCORE if value is None else value
            for episode_id, value in zip(episode_ids, values)
        },
        usage=sum((result.usage for result in results), Usage()),
        n_requests=len(requests),
        n_missing=sum(value is None for value in values),
        model=",".join(sorted({result.model for result in results if result.model})),
        n_cached=sum(result.cached for result in results),
    )


def trace_extra(scores: JevScores, config: JevScoringConfig) -> dict[str, Any]:
    """What a Jev ranker records in ``RankTrace.extra`` about its scoring call."""

    return {
        "n_requests": scores.n_requests,
        "n_missing": scores.n_missing,
        "n_cached": scores.n_cached,
        "jev_model": scores.model,
        "scoring": config.label,
    }


def _instruction_text(config: JevScoringConfig) -> str:
    """The task, where its inputs are in this mode, and the judgement asked for."""

    wording = WORDINGS[config.wording]
    if config.mode == "listwise":
        ask = wording.choice
    elif config.primitive == "noul":
        ask = wording.noul
    else:
        ask = wording.score
    return f"{TASK} {_LAYOUT[config.mode]} {ask}"


def _relevance_question(config: JevScoringConfig, instructions: str) -> dict[str, Any]:
    """The question asked about one candidate, for the configured primitive."""

    if config.primitive == "noul":
        wording = WORDINGS[config.wording]
        criteria: Any = {"true": wording.noul_true, "false": wording.noul_false}
    else:
        criteria = list(SCORE_LEVELS)
    return {"type": config.primitive, "instructions": instructions, "criteria": criteria}


def _build_requests(
    query: str,
    summaries: Sequence[str],
    config: JevScoringConfig,
) -> list[_Request]:
    """The requests that cover ``summaries``, in candidate order.

    Questions (and listwise options) are named by position within their
    request, ``c0``, ``c1``, ..., never by episode id: an id means nothing to
    the model, and positional names keep identical requests identical.
    """

    text = _instruction_text(config)
    if config.mode == "listwise":
        question = {
            "type": "choice",
            "instructions": text,
            "criteria": {f"c{index}": summary for index, summary in enumerate(summaries)},
        }
        return [({"query": query}, {_LISTWISE_QUESTION: question})]
    if config.mode == "pair":
        return [
            (
                {"query": query, "conversation_summary": summary},
                {"c0": _relevance_question(config, text)},
            )
            for summary in summaries
        ]
    requests: list[_Request] = []
    for start in range(0, len(summaries), config.batch_size):
        batch = summaries[start : start + config.batch_size]
        questions = {
            f"c{index}": _relevance_question(config, f"{text}\n\n{_BATCHED_SUMMARY_LABEL}{summary}")
            for index, summary in enumerate(batch)
        }
        requests.append(({"query": query}, questions))
    return requests


def _send(
    client: DecisionClient,
    requests: Sequence[_Request],
    config: JevScoringConfig,
) -> list[DecisionResult]:
    """Run every request on the thread pool; results come back in request order.

    When a request fails, the ones still queued are cancelled (there is no
    point paying for them) and the failure propagates. If several fail, the
    one that comes first in request order is raised.
    """

    executor = ThreadPoolExecutor(max_workers=config.concurrency)
    try:
        futures = [
            executor.submit(client.decide, state, questions, salt=config.salt)
            for state, questions in requests
        ]
        wait(futures, return_when=FIRST_EXCEPTION)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    # Workers take requests in order, so a cancelled request always comes after
    # the failed one: reading in order raises the real error, never the cancellation.
    return [future.result() for future in futures]


def _read_scores(
    requests: Sequence[_Request],
    results: Sequence[DecisionResult],
    config: JevScoringConfig,
) -> list[float | None]:
    """One score per candidate, in candidate order; None where Jev gave no usable answer."""

    if config.mode == "listwise":
        options = requests[0][1][_LISTWISE_QUESTION]["criteria"]
        answer = results[0].answers.get(_LISTWISE_QUESTION)
        probabilities = _answer_field(answer, "choice", "probabilities")
        if not isinstance(probabilities, Mapping):
            return [None] * len(options)
        return [_number(probabilities.get(option)) for option in options]

    top_level = len(SCORE_LEVELS) - 1
    scores: list[float | None] = []
    for (_, questions), result in zip(requests, results):
        for name in questions:
            answer = result.answers.get(name)
            value = _number(_answer_field(answer, config.primitive, config.primitive))
            if value is not None and config.primitive == "score":
                value /= top_level  # position on the rubric, rescaled to [0, 1]
            scores.append(value)
    return scores


def _answer_field(answer: Any, answer_type: str, key: str) -> Any:
    """``answer[key]`` if ``answer`` has the expected type; None for a refusal or no answer."""

    if isinstance(answer, Mapping) and answer.get("type") == answer_type:
        return answer.get(key)
    return None


def _number(value: Any) -> float | None:
    """``value`` as a finite float, or None if it is not a usable number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
