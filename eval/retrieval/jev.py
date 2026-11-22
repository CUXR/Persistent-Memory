"""Client for Jev, TypeSafe's hosted decision model, through either gateway.

Callers always speak OpenRouter's shapes (typed questions in, typed answers
out). When the Vercel AI Gateway is configured the request and the response
are translated here, so nothing above this module depends on the gateway.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from . import config
from .cache import DiskCache, make_key
from .interfaces import DecisionResult, Usage
from .metrics import dig
from .llm import (
    MAX_RETRIES,
    GatewaySession,
    HostedAPIError,
    as_number,
    missing_key_message,
    token_count,
    warn_once,
)
from .spend import SpendTracker

QUESTION_TYPES: tuple[str, ...] = ("noul", "score", "choice")
Questions = dict[str, dict[str, Any]]  # question name -> OpenRouter-shaped question

# The gateway cookbook retries a 402 "whose body marks an in-flight budget
# error" (credit held by requests still running) without giving the exact
# field, so any spelling of "in-flight" anywhere in the body counts.
_IN_FLIGHT = re.compile(r"in[\s_-]?flight", re.IGNORECASE)


class JevError(HostedAPIError):
    """A Jev request failed for good; ``status`` and ``message`` come from the gateway."""


def _in_flight_budget_error(status: int, body: str) -> bool:
    return status == 402 and _IN_FLIGHT.search(body) is not None


def _plain(value: Any) -> Any:
    """Copy JSON-like data into plain dicts and lists, so any mapping serializes and hashes."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _openrouter_body(model: str, state: Any, questions: Questions) -> dict[str, Any]:
    return {"model": model, "state": state, "questions": questions}


def _openrouter_answers(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    answers = payload.get("answers")
    return answers if isinstance(answers, Mapping) else None


# Vercel documents its predicate shape in full but its choice and score shapes
# only in outline, so the three functions below are a best-effort adapter:
# they follow the outline, and anything it leaves open is noted where it is decided.


def _vercel_body(model: str, state: Any, questions: Questions) -> dict[str, Any]:
    if isinstance(state, str):
        state_input: Any = state
    else:
        state_input = [{"role": "user", "content": [{"type": "input_json", "data": state}]}]
    translated = []
    for name, question in questions.items():
        kind = question["type"]
        item: dict[str, Any] = {
            "type": "predicate" if kind == "noul" else kind,
            "name": name,
            "instructions": _vercel_instructions(question),
        }
        if kind == "choice":
            item["choices"] = [
                {"value": value, "description": description}
                for value, description in question["criteria"].items()
            ]
        elif kind == "score":
            item["levels"] = [{"label": label} for label in question["criteria"]]
        translated.append(item)
    return {"model": model, "input": state_input, "questions": translated}


def _vercel_instructions(question: Mapping[str, Any]) -> str:
    """Instructions as the single string Vercel documents.

    Structured instructions go as JSON text. A noul's true/false criteria are
    appended to the text: a predicate has no criteria field, and dropping them
    would change the question between gateways.
    """

    instructions = question["instructions"]
    if isinstance(instructions, str):
        text = instructions
    else:
        text = json.dumps(instructions, ensure_ascii=False)
    criteria = question.get("criteria")
    if question["type"] == "noul" and isinstance(criteria, Mapping):
        lines = [f"{label}: {criteria[label]}" for label in ("true", "false") if label in criteria]
        if lines:
            text = "\n".join([text, "", *lines])
    return text


def _vercel_answers(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Vercel's list of named answers as OpenRouter's mapping of name to answer."""

    answers = payload.get("answers")
    if not isinstance(answers, list):
        return None
    translated: dict[str, Any] = {}
    for answer in answers:
        if not isinstance(answer, Mapping) or not isinstance(answer.get("name"), str):
            continue
        fields = {key: value for key, value in answer.items() if key != "name"}
        if fields.get("type") == "predicate":
            fields = {"type": "noul", "noul": fields.get("probability")}
        elif fields.get("type") in ("score", "choice"):
            fields.setdefault("confidence", None)  # not in Vercel's documented response
        translated[answer["name"]] = fields
    return translated


@dataclass(frozen=True)
class _Wire:
    """How one gateway spells a decisions request and its response."""

    body: Callable[[str, Any, Questions], dict[str, Any]]
    answers: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
    cost_path: tuple[str, ...]  # where the response reports what the call cost


_WIRES: dict[str, _Wire] = {
    "openrouter": _Wire(_openrouter_body, _openrouter_answers, ("usage", "cost")),
    "vercel": _Wire(_vercel_body, _vercel_answers, ("provider_metadata", "gateway", "cost")),
}


def _request_size(body: Mapping[str, Any]) -> int:
    """The request's JSON text in ASCII-escaped characters: an upper bound on its tokens.

    Every token covers at least one byte, and the escaped text has at least
    as many characters as the request has UTF-8 bytes.
    """

    return len(json.dumps(body))


def _normalized(answer: Any) -> dict[str, Any]:
    """One answer as a plain dict; a refusal, or no answer at all, is ``{"type": "refusal"}``."""

    if not isinstance(answer, Mapping) or answer.get("type") == "refusal":
        return {"type": "refusal"}
    return dict(answer)


class JevClient:
    """:class:`~.interfaces.DecisionClient` for Jev on OpenRouter or the Vercel AI Gateway.

    Thread-safe: rankers call :meth:`decide` from a thread pool. With a cache,
    a repeated request returns the stored answers and their original usage
    without a call or a charge. Jev has no seed, so benchmark runs pass a
    per-run ``salt``: repeated runs are then fresh calls (which is what
    measures run-to-run variance) while an interrupted run still resumes.
    """

    def __init__(
        self,
        api_key: str,
        *,
        provider: str,
        url: str,
        model: str,
        cache: DiskCache | None = None,
        spend: SpendTracker | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if provider not in _WIRES:
            raise ValueError(f"provider must be one of {sorted(_WIRES)}, got {provider!r}")
        self.provider = provider
        self.url = url
        self.model = model
        self._wire = _WIRES[provider]
        self._session = GatewaySession(
            url,
            api_key,
            error=JevError,
            max_retries=max_retries,
            timeout=timeout,
            sleep=sleep,
            spend=spend,
            transport=transport,
            also_transient=_in_flight_budget_error,
        )
        self._cache = cache
        self._spend = spend

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        salt: str = "",
    ) -> DecisionResult:
        """Answer every question against ``state`` in one request.

        ``questions`` maps a name to an OpenRouter-shaped question
        (``{"type": "noul" | "score" | "choice", "instructions": ..., "criteria": ...}``).
        The result has one answer per question, in the shapes listed on
        :class:`~.interfaces.DecisionResult`; a question Jev refused or left
        out comes back as ``{"type": "refusal"}``.
        """

        state = _plain(state)
        asked: Questions = _plain(questions)
        for name, question in asked.items():
            if question.get("type") not in QUESTION_TYPES:
                raise ValueError(
                    f"question {name!r} has type {question.get('type')!r}; "
                    f"expected one of {QUESTION_TYPES}"
                )

        if self._cache is None:
            result = self._call(state, asked)
        else:
            key = make_key("jev", self.provider, self.model, state, asked, salt)
            with self._cache.lock(key):
                stored = self._cache.get(key)
                if stored is not None:
                    return DecisionResult(
                        answers=stored["answers"],
                        usage=Usage.from_json(stored["usage"]),
                        model=stored["model"],
                        cached=True,
                    )
                result = self._call(state, asked)
                self._cache.set(
                    key,
                    {
                        "answers": result.answers,
                        "usage": result.usage.to_json(),
                        "model": result.model,
                    },
                )
        # Charged after the answers are stored: if this call is the one that
        # trips the cap, what it paid for is still there for the next run.
        self._charge(result.usage.cost_usd)
        return result

    def close(self) -> None:
        """Close the connection pool."""

        self._session.close()

    def _call(self, state: Any, questions: Questions) -> DecisionResult:
        """One uncached request, with the response normalized to OpenRouter's shapes."""

        body = self._wire.body(self.model, state, questions)
        payload, latency_s = self._session.post_json(body)
        input_tokens = token_count(dig(payload, "usage", "input_tokens"))
        cost = as_number(dig(payload, *self._wire.cost_path))
        if cost is None or cost <= 0:
            billed_tokens = input_tokens
            if billed_tokens == 0:
                # Neither a cost nor a token count: charge the request's own size,
                # an upper bound on its tokens, so the spend cap still sees the call.
                billed_tokens = _request_size(body)
                warn_once(
                    f"jev-usage:{self.url}",
                    "%s reported neither a cost nor input tokens: charging each such request's "
                    "size in characters at the Jev input price, an upper bound, so the spend cap "
                    "still counts it",
                    self.url,
                )
            cost = billed_tokens * config.JEV_PRICE_USD_PER_MTOK_INPUT / 1_000_000
        answers = self._wire.answers(payload)
        if answers is None:
            self._charge(cost)
            raise JevError(f"the response carries no answers: {json.dumps(payload)[:300]}")
        served = payload.get("model")
        return DecisionResult(
            answers={name: _normalized(answers.get(name)) for name in questions},
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=token_count(dig(payload, "usage", "output_tokens")),
                cost_usd=cost,
                latency_s=latency_s,
                calls=1,
            ),
            model=served if isinstance(served, str) and served else self.model,
        )

    def _charge(self, cost_usd: float) -> None:
        if self._spend is not None:
            self._spend.add(cost_usd)


def build_jev(
    settings: config.Settings | None = None,
    cache: DiskCache | None = None,
    spend: SpendTracker | None = None,
) -> JevClient:
    """The Jev client for the configured gateway (``config.get_settings()`` by default)."""

    settings = settings or config.get_settings()
    if not settings.jev_url:
        raise JevError(no_jev_message(settings.gateway))
    if not settings.jev_api_key:
        raise JevError(missing_key_message("Jev", settings.gateway, "EVAL_JEV_API_KEY"))
    return JevClient(
        settings.jev_api_key,
        provider=settings.gateway,
        url=settings.jev_url,
        model=settings.jev_model,
        cache=cache,
        spend=spend,
    )


def no_jev_message(gateway: str) -> str:
    """What to tell the operator who asks for Jev on a gateway that does not serve it."""

    served = ", ".join(name for name, spec in config.GATEWAYS.items() if spec["jev_url"])
    return (
        f"Jev is not served on the {gateway} gateway (EVAL_GATEWAY): it needs one of "
        f"{served}, each with its own paid key. Without it, run the local rankers "
        f"({', '.join(config.LOCAL_RANKERS)})"
    )
