"""JevClient against mock gateways: both wire formats, normalization, retries, caching, spend."""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Any

import httpx
import pytest

from eval.retrieval import config
from eval.retrieval import llm as llm_module
from eval.retrieval.cache import DiskCache
from eval.retrieval.interfaces import DecisionClient
from eval.retrieval.jev import JevClient, JevError, build_jev
from eval.retrieval.llm import GATEWAY_KEY_ENV
from eval.retrieval.spend import SpendLimitExceeded, SpendTracker

ENDPOINTS = {  # provider -> (url, model), as in the gateway docs
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.14"),
    "vercel": ("https://ai-gateway.vercel.sh/v1/decisions", "typesafe-ai/jev-1.14.0"),
}
SNAPSHOT = "typesafe/jev-1.14-20261014"
PRICE = config.JEV_PRICE_USD_PER_MTOK_INPUT / 1e6  # dollars per input token
WAIT_S = 10  # upper bound for anything that would otherwise hang a broken test

STATE = {"query": "Where does Sam climb?"}
STRUCTURED = {"instruction": "Grade the summary.", "conversation_summary": "Sam joined a gym."}
QUESTIONS: dict[str, dict[str, Any]] = {
    "relevant": {
        "type": "noul",
        "instructions": "Does the summary answer the query?",
        "criteria": {"true": "It answers the query.", "false": "It does not."},
    },
    "plain": {"type": "noul", "instructions": "Is a place named?"},
    "grade": {
        "type": "score",
        "instructions": STRUCTURED,
        "criteria": ["Irrelevant", "Related", "Answers the query"],
    },
    "best": {
        "type": "choice",
        "instructions": "Pick the summary that answers the query.",
        "criteria": {"c0": "Sam joined a climbing gym.", "c1": "Alex moved to Lisbon."},
    },
}
SCORE_ANSWER = {
    "type": "score",
    "score": 1.99,
    "confidence": 0.99,
    "probabilities": {"0": 0, "1": 0.01, "2": 0.99},
    "legend": {"0": "Irrelevant", "1": "Related", "2": "Answers the query"},
}
CHOICE_ANSWER = {
    "type": "choice",
    "choice": "c0",
    "confidence": 0.67,
    "probabilities": {"c0": 0.78, "c1": 0.22},
}
ANSWERS: dict[str, dict[str, Any]] = {
    "relevant": {"type": "noul", "noul": 0.96},
    "plain": {"type": "noul", "noul": 0.12},
    "grade": SCORE_ANSWER,
    "best": CHOICE_ANSWER,
}
ONE_QUESTION = {"c0": QUESTIONS["relevant"]}
ONE_ANSWER = {"c0": {"type": "noul", "noul": 0.96}}


def openrouter_body(
    answers: dict[str, Any] | None = None, *, usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    """An OpenRouter decisions response (476 input tokens, a reported cost unless overridden)."""

    return {
        "id": "gen-dec-1",
        "model": SNAPSHOT,
        "provider": "TypeSafe",
        "answers": ONE_ANSWER if answers is None else answers,
        "usage": (
            {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002} if usage is None else usage
        ),
    }


def vercel_body(answers: list[dict[str, Any]], *, cost: Any = "0.000012") -> dict[str, Any]:
    """A Vercel decisions response (282 input tokens; ``cost=None`` leaves the cost out)."""

    body: dict[str, Any] = {
        "answers": answers,
        "model": "typesafe-ai/jev-1.14.0",
        "usage": {"input_tokens": 282, "output_tokens": 20, "total_tokens": 302},
    }
    if cost is not None:
        body["provider_metadata"] = {"gateway": {"cost": cost}}
    return body


def ok(answers: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    return 200, openrouter_body(answers)


def error(status: int, message: str, **headers: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    return status, {"error": {"code": status, "message": message}}, headers


class Gateway:
    """``httpx.MockTransport`` handler: replays scripted replies and records every request.

    A reply is ``(status, body)`` or ``(status, body, headers)``, an exception
    to raise, or a callable that takes the request. The last reply repeats.
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(request)
        status, body, *rest = reply
        return httpx.Response(status, json=body, headers=rest[0] if rest else None)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def make_jev(
    gateway: Gateway, provider: str = "openrouter", **kwargs: Any
) -> tuple[JevClient, list[float]]:
    """A client wired to ``gateway``, plus the list its (never waiting) sleeps are recorded in."""

    sleeps: list[float] = []
    url, model = ENDPOINTS[provider]
    client = JevClient(
        "sk-test",
        provider=provider,
        url=url,
        model=kwargs.pop("model", model),
        transport=httpx.MockTransport(gateway),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, sleeps


@pytest.fixture
def cache(tmp_path):
    store = DiskCache(tmp_path / "cache.sqlite")
    yield store
    store.close()


@pytest.fixture
def load_settings(monkeypatch):
    """Build ``config.Settings`` from a scrubbed environment plus the given variables."""

    def load(**env: str) -> config.Settings:
        for name in list(os.environ):
            is_eval = name.startswith("EVAL_") and name != "EVAL_DISABLE_DOTENV"
            if is_eval or name in GATEWAY_KEY_ENV.values():
                monkeypatch.delenv(name)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        config.get_settings.cache_clear()
        return config.get_settings()

    yield load
    config.get_settings.cache_clear()  # never leak test settings into other tests


@pytest.fixture
def route_http(monkeypatch):
    """Send every ``httpx.Client`` built during the test through a mock gateway."""

    def route(gateway: Gateway) -> None:
        real_client = httpx.Client

        def client(**kwargs: Any) -> httpx.Client:
            return real_client(**{**kwargs, "transport": httpx.MockTransport(gateway)})

        monkeypatch.setattr(httpx, "Client", client)

    return route


# --------------------------------------------------------------------------- #
# Requests                                                                     #
# --------------------------------------------------------------------------- #
def test_openrouter_request_passes_every_question_type_through():
    gateway = Gateway(ok(ANSWERS))
    jev, _ = make_jev(gateway)

    jev.decide(STATE, QUESTIONS)

    (request,) = gateway.requests
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"] == "application/json"
    assert gateway.bodies[0] == {
        "model": "typesafe/jev-1.14",
        "state": STATE,
        "questions": QUESTIONS,
    }


def test_vercel_request_translates_every_question_type():
    gateway = Gateway((200, vercel_body([])))
    jev, _ = make_jev(gateway, "vercel")

    jev.decide(STATE, QUESTIONS)

    (request,) = gateway.requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/v1/decisions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert gateway.bodies[0] == {
        "model": "typesafe-ai/jev-1.14.0",
        "input": [{"role": "user", "content": [{"type": "input_json", "data": STATE}]}],
        "questions": [
            {
                "type": "predicate",
                "name": "relevant",
                # A predicate has no criteria field, so the criteria travel in the text.
                "instructions": (
                    "Does the summary answer the query?\n\n"
                    "true: It answers the query.\nfalse: It does not."
                ),
            },
            {"type": "predicate", "name": "plain", "instructions": "Is a place named?"},
            {
                "type": "score",
                "name": "grade",
                "instructions": json.dumps(STRUCTURED),
                "levels": [
                    {"label": "Irrelevant"},
                    {"label": "Related"},
                    {"label": "Answers the query"},
                ],
            },
            {
                "type": "choice",
                "name": "best",
                "instructions": "Pick the summary that answers the query.",
                "choices": [
                    {"value": "c0", "description": "Sam joined a climbing gym."},
                    {"value": "c1", "description": "Alex moved to Lisbon."},
                ],
            },
        ],
    }


@pytest.mark.parametrize(
    ("provider", "field"), [("openrouter", "state"), ("vercel", "input")]
)
def test_a_text_state_is_sent_as_text(provider, field):
    reply = ok() if provider == "openrouter" else (200, vercel_body([]))
    gateway = Gateway(reply)
    jev, _ = make_jev(gateway, provider)

    jev.decide("Query: where does Sam climb?", ONE_QUESTION)

    assert gateway.bodies[0][field] == "Query: where does Sam climb?"


def test_any_mapping_and_sequence_is_accepted():
    gateway = Gateway(ok())
    jev, _ = make_jev(gateway)
    question = MappingProxyType(
        {"type": "score", "instructions": "Grade it.", "criteria": ("Irrelevant", "Relevant")}
    )

    jev.decide(MappingProxyType(STATE), MappingProxyType({"c0": question}))

    assert gateway.bodies[0]["state"] == STATE
    assert gateway.bodies[0]["questions"] == {
        "c0": {"type": "score", "instructions": "Grade it.", "criteria": ["Irrelevant", "Relevant"]}
    }


def test_an_unknown_question_type_is_refused_before_anything_is_sent():
    gateway = Gateway(ok())
    jev, _ = make_jev(gateway)

    with pytest.raises(ValueError, match="'c1'.*'predicate'"):
        jev.decide(STATE, {**ONE_QUESTION, "c1": {"type": "predicate", "instructions": "?"}})

    assert gateway.requests == []


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError, match="bedrock"):
        JevClient("sk-test", provider="bedrock", url="https://example.test/d", model="jev")


# --------------------------------------------------------------------------- #
# Answers                                                                      #
# --------------------------------------------------------------------------- #
def test_openrouter_answers_come_back_as_sent_in_question_order():
    reply = openrouter_body({name: ANSWERS[name] for name in reversed(ANSWERS)})
    jev, _ = make_jev(Gateway((200, reply)))

    result = jev.decide(STATE, QUESTIONS)

    assert isinstance(jev, DecisionClient)
    assert result.answers == ANSWERS
    assert list(result.answers) == list(QUESTIONS)
    assert result.model == SNAPSHOT  # the dated snapshot, not the alias that was asked for
    assert result.cached is False


def test_openrouter_refused_and_missing_answers_become_refusals():
    answers = {
        "relevant": ANSWERS["relevant"],
        "plain": {"type": "refusal", "reason": "policy"},
        "grade": None,
        "stray": {"type": "noul", "noul": 0.5},  # never asked: dropped
    }
    jev, _ = make_jev(Gateway(ok(answers)))

    result = jev.decide(STATE, QUESTIONS)

    assert result.answers == {
        "relevant": {"type": "noul", "noul": 0.96},
        "plain": {"type": "refusal"},
        "grade": {"type": "refusal"},
        "best": {"type": "refusal"},
    }


def test_vercel_answers_are_normalized_to_the_openrouter_shape():
    answers = [
        {"type": "choice", "name": "best", "choice": "c1", "probabilities": {"c0": 0.3, "c1": 0.7}},
        {"type": "predicate", "name": "relevant", "probability": 0.98},
        {
            "type": "score",
            "name": "grade",
            "score": 1.4,
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        },
        {"type": "predicate", "name": "plain", "probability": 0.02},
    ]
    jev, _ = make_jev(Gateway((200, vercel_body(answers))), "vercel")

    result = jev.decide(STATE, QUESTIONS)

    assert result.answers == {
        "relevant": {"type": "noul", "noul": 0.98},
        "plain": {"type": "noul", "noul": 0.02},
        "grade": {
            "type": "score",
            "score": 1.4,
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
            "confidence": None,
        },
        "best": {
            "type": "choice",
            "choice": "c1",
            "probabilities": {"c0": 0.3, "c1": 0.7},
            "confidence": None,
        },
    }
    assert list(result.answers) == list(QUESTIONS)
    assert result.model == "typesafe-ai/jev-1.14.0"


def test_vercel_refused_and_missing_answers_become_refusals():
    answers = [
        {"type": "refusal", "name": "relevant"},
        {"type": "predicate", "name": "plain", "probability": 0.5},
        {"type": "predicate", "probability": 0.9},  # no name: cannot belong to a question
    ]
    jev, _ = make_jev(Gateway((200, vercel_body(answers))), "vercel")

    result = jev.decide(STATE, QUESTIONS)

    assert result.answers == {
        "relevant": {"type": "refusal"},
        "plain": {"type": "noul", "noul": 0.5},
        "grade": {"type": "refusal"},
        "best": {"type": "refusal"},
    }


@pytest.mark.parametrize("provider", ["openrouter", "vercel"])
def test_a_response_without_answers_is_an_error_that_is_still_charged(provider):
    body = openrouter_body() if provider == "openrouter" else vercel_body([])
    del body["answers"]
    gateway = Gateway((200, body))
    spend = SpendTracker(1.0)
    jev, _ = make_jev(gateway, provider, spend=spend)

    with pytest.raises(JevError, match="no answers"):
        jev.decide(STATE, ONE_QUESTION)

    assert len(gateway.requests) == 1
    assert spend.total > 0


# --------------------------------------------------------------------------- #
# Usage and cost                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("usage", "expected_cost"),
    [
        ({"input_tokens": 476, "output_tokens": 70, "cost": 0.00002}, 0.00002),
        ({"input_tokens": 476, "output_tokens": 70, "cost": 0}, 476 * PRICE),
        ({"input_tokens": 476, "output_tokens": 70, "cost": None}, 476 * PRICE),
        ({"input_tokens": 476, "output_tokens": 70}, 476 * PRICE),
        ({"input_tokens": 1_000_000, "output_tokens": 500_000}, 0.042),  # output is free
    ],
    ids=["reported", "reported-zero", "reported-null", "not-reported", "price-per-million"],
)
def test_openrouter_usage_and_cost(usage, expected_cost):
    jev, _ = make_jev(Gateway((200, openrouter_body(usage=usage))))

    result = jev.decide(STATE, ONE_QUESTION)

    assert result.usage.input_tokens == usage["input_tokens"]
    assert result.usage.output_tokens == usage["output_tokens"]
    assert result.usage.cost_usd == pytest.approx(expected_cost, rel=1e-12)
    assert result.usage.calls == 1
    assert result.usage.latency_s >= 0


@pytest.mark.parametrize(
    ("cost", "expected_cost"),
    [("0.000012", 0.000012), (0.000012, 0.000012), ("0", 282 * PRICE), (None, 282 * PRICE)],
    ids=["reported-as-text", "reported-as-number", "reported-zero", "not-reported"],
)
def test_vercel_usage_and_cost(cost, expected_cost):
    answers = [{"type": "predicate", "name": "c0", "probability": 0.9}]
    jev, _ = make_jev(Gateway((200, vercel_body(answers, cost=cost))), "vercel")

    result = jev.decide(STATE, ONE_QUESTION)

    assert (result.usage.input_tokens, result.usage.output_tokens) == (282, 20)
    assert result.usage.cost_usd == pytest.approx(expected_cost, rel=1e-12)
    assert result.usage.calls == 1


@pytest.mark.parametrize(
    "usage", [{}, {"output_tokens": 70}, {"input_tokens": 0}], ids=["none", "output-only", "zero"]
)
def test_a_response_without_cost_or_input_tokens_is_charged_by_its_request_size(
    usage, monkeypatch, caplog
):
    # Nothing to price by would otherwise cost $0, and the spend cap would never
    # see such calls; the request's size bounds its tokens, so charge that.
    monkeypatch.setattr(llm_module, "_warned", set())
    gateway = Gateway((200, openrouter_body(usage=usage)))
    spend = SpendTracker(1.0)
    jev, _ = make_jev(gateway, spend=spend)

    with caplog.at_level(logging.WARNING, logger="eval.retrieval.llm"):
        first = jev.decide(STATE, ONE_QUESTION)
        second = jev.decide({"query": "Where does Dana climb?"}, ONE_QUESTION)

    sizes = [len(json.dumps(body)) for body in gateway.bodies]
    assert sizes[0] > 100 and sizes[0] != sizes[1]
    assert first.usage.input_tokens == 0  # as reported: the estimate is of the cost alone
    assert first.usage.cost_usd == pytest.approx(sizes[0] * PRICE, rel=1e-12)
    assert second.usage.cost_usd == pytest.approx(sizes[1] * PRICE, rel=1e-12)
    assert spend.total == pytest.approx(sum(sizes) * PRICE, rel=1e-12)
    (warning,) = caplog.messages  # once per endpoint, not per call
    assert ENDPOINTS["openrouter"][0] in warning and "upper bound" in warning


def test_latency_is_the_wall_time_of_the_successful_call_only(monkeypatch):
    # First attempt (a 429) runs from 5.0 to 9.0; the retry from 30.0 to 30.125.
    clock = iter([5.0, 9.0, 30.0, 30.125])
    monkeypatch.setattr(llm_module, "perf_counter", lambda: next(clock))
    jev, _ = make_jev(Gateway(error(429, "slow down"), ok()))

    assert jev.decide(STATE, ONE_QUESTION).usage.latency_s == 0.125


# --------------------------------------------------------------------------- #
# Retries                                                                      #
# --------------------------------------------------------------------------- #
def test_429_is_retried_after_the_gateways_retry_after():
    gateway = Gateway(error(429, "rate limited", **{"Retry-After": "3"}), ok())
    jev, sleeps = make_jev(gateway)

    result = jev.decide(STATE, ONE_QUESTION)

    assert result.answers == ONE_ANSWER
    assert len(gateway.requests) == 2
    assert sleeps == [3.0]


def test_5xx_is_retried_with_exponential_backoff_and_jitter():
    gateway = Gateway(error(500, "boom"), error(524, "timeout"), error(529, "overloaded"), ok())
    jev, sleeps = make_jev(gateway)

    result = jev.decide(STATE, ONE_QUESTION)

    assert result.answers == ONE_ANSWER
    assert len(gateway.requests) == 4
    assert [int(seconds) for seconds in sleeps] == [1, 2, 4]  # 2**attempt, plus under 1s jitter


def test_transport_errors_are_retried():
    gateway = Gateway(httpx.ReadTimeout("timed out"), ok())
    jev, sleeps = make_jev(gateway)

    assert jev.decide(STATE, ONE_QUESTION).answers == ONE_ANSWER
    assert len(gateway.requests) == 2 and len(sleeps) == 1


@pytest.mark.parametrize(
    "error_body",
    [
        {"code": 402, "message": "Budget is held by in-flight requests; retry shortly"},
        {"code": 402, "message": "No credits", "metadata": {"reason": "in_flight_budget"}},
        {"code": 402, "message": "Payment required", "metadata": {"inFlight": True}},
    ],
    ids=["message", "metadata-reason", "metadata-flag"],
)
def test_402_for_an_in_flight_budget_is_retried(error_body):
    gateway = Gateway((402, {"error": error_body}), ok())
    jev, sleeps = make_jev(gateway)

    assert jev.decide(STATE, ONE_QUESTION).answers == ONE_ANSWER
    assert len(gateway.requests) == 2 and len(sleeps) == 1


def test_402_for_exhausted_credits_raises_at_once():
    gateway = Gateway(error(402, "Insufficient credits"))
    jev, sleeps = make_jev(gateway)

    with pytest.raises(JevError, match="Insufficient credits") as raised:
        jev.decide(STATE, ONE_QUESTION)

    assert raised.value.status == 402
    assert len(gateway.requests) == 1 and sleeps == []


@pytest.mark.parametrize("status", [400, 401, 413])
def test_non_transient_errors_raise_at_once_with_status_and_message(status):
    gateway = Gateway(error(status, "state plus questions exceed 64000 tokens"))
    jev, sleeps = make_jev(gateway)

    with pytest.raises(JevError) as raised:
        jev.decide(STATE, ONE_QUESTION)

    assert raised.value.status == status
    assert raised.value.message == "state plus questions exceed 64000 tokens"
    assert str(raised.value) == f"HTTP {status}: state plus questions exceed 64000 tokens"
    assert len(gateway.requests) == 1 and sleeps == []


def test_gives_up_after_five_attempts():
    gateway = Gateway(error(503, "unavailable"))
    jev, sleeps = make_jev(gateway)

    with pytest.raises(JevError, match="unavailable") as raised:
        jev.decide(STATE, ONE_QUESTION)

    assert raised.value.status == 503
    assert "gave up after 5 attempts" in str(raised.value)
    assert len(gateway.requests) == 5
    assert [int(seconds) for seconds in sleeps] == [1, 2, 4, 8]


def test_the_default_timeout_is_sixty_seconds():
    seen: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json=openrouter_body())

    jev, _ = make_jev(Gateway(record))
    jev.decide(STATE, ONE_QUESTION)

    assert seen == [{"connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0}]


# --------------------------------------------------------------------------- #
# Cache                                                                        #
# --------------------------------------------------------------------------- #
def test_cache_hit_returns_the_stored_answers_without_a_call_or_a_charge(cache):
    gateway = Gateway(ok(ANSWERS))
    spend = SpendTracker(1.0)
    jev, _ = make_jev(gateway, cache=cache, spend=spend)

    fresh = jev.decide(STATE, QUESTIONS)
    hit = jev.decide(STATE, {name: QUESTIONS[name] for name in reversed(QUESTIONS)})

    assert len(gateway.requests) == 1  # question order is not part of the request's identity
    assert fresh.cached is False and hit.cached is True
    assert hit.answers == fresh.answers == ANSWERS
    assert hit.model == SNAPSHOT
    assert hit.usage == fresh.usage  # the original call's usage, cost included
    assert spend.total == pytest.approx(0.00002)  # charged once


def test_a_new_salt_is_a_fresh_call_and_the_same_salt_resumes(cache):
    gateway = Gateway(ok())
    jev, _ = make_jev(gateway, cache=cache)

    first_run = [jev.decide(STATE, ONE_QUESTION, salt="run-1") for _ in range(2)]
    second_run = [jev.decide(STATE, ONE_QUESTION, salt="run-2") for _ in range(2)]

    assert [result.cached for result in first_run] == [False, True]
    assert [result.cached for result in second_run] == [False, True]
    assert len(gateway.requests) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"state": {"query": "Where does Alex live?"}},
        {"questions": {"c0": QUESTIONS["plain"]}},
        {"questions": {"c1": QUESTIONS["relevant"]}},
    ],
    ids=["state", "question", "question-name"],
)
def test_state_and_questions_are_part_of_the_key(cache, change):
    gateway = Gateway(ok({"c0": ONE_ANSWER["c0"], "c1": ONE_ANSWER["c0"]}))
    jev, _ = make_jev(gateway, cache=cache)
    request = {"state": STATE, "questions": ONE_QUESTION}

    jev.decide(**request)
    changed = jev.decide(**{**request, **change})

    assert changed.cached is False
    assert len(gateway.requests) == 2


def test_provider_and_model_are_part_of_the_key(cache):
    predicate = [{"type": "predicate", "name": "c0", "probability": 0.5}]
    openrouter, _ = make_jev(Gateway(ok()), cache=cache)
    newer, _ = make_jev(Gateway(ok()), cache=cache, model="typesafe/jev-1.15")
    # Same model string as the first client, so only the provider tells them apart.
    vercel, _ = make_jev(
        Gateway((200, vercel_body(predicate))), "vercel", cache=cache, model=openrouter.model
    )

    results = [client.decide(STATE, ONE_QUESTION) for client in (openrouter, newer, vercel)]

    assert [result.cached for result in results] == [False, False, False]
    assert results[2].answers == {"c0": {"type": "noul", "noul": 0.5}}
    assert openrouter.decide(STATE, ONE_QUESTION).cached is True


def test_without_a_cache_every_call_is_fresh():
    gateway = Gateway(ok())
    spend = SpendTracker(1.0)
    jev, _ = make_jev(gateway, spend=spend)

    results = [jev.decide(STATE, ONE_QUESTION, salt="same") for _ in range(2)]

    assert [result.cached for result in results] == [False, False]
    assert len(gateway.requests) == 2
    assert spend.total == pytest.approx(2 * 0.00002)


def test_a_new_client_resumes_from_the_cache_file(tmp_path):
    path = tmp_path / "cache.sqlite"
    first_cache = DiskCache(path)
    first, _ = make_jev(Gateway(ok(ANSWERS)), cache=first_cache)
    first.decide(STATE, QUESTIONS, salt="run-1")
    first_cache.close()

    second_cache = DiskCache(path)
    try:
        gateway = Gateway(error(500, "must not be called"))
        second, _ = make_jev(gateway, cache=second_cache)
        hit = second.decide(STATE, QUESTIONS, salt="run-1")
    finally:
        second_cache.close()

    assert hit.cached is True and hit.answers == ANSWERS
    assert gateway.requests == []


# --------------------------------------------------------------------------- #
# Spend cap                                                                    #
# --------------------------------------------------------------------------- #
def test_cap_stops_requests_but_keeps_what_was_paid_for(cache):
    gateway = Gateway(ok())
    spend = SpendTracker(0.00003)  # one call at $0.00002 fits, the second passes the cap
    jev, _ = make_jev(gateway, cache=cache, spend=spend)

    jev.decide(STATE, ONE_QUESTION, salt="1")
    with pytest.raises(SpendLimitExceeded):
        jev.decide(STATE, ONE_QUESTION, salt="2")
    assert len(gateway.requests) == 2

    with pytest.raises(SpendLimitExceeded):
        jev.decide(STATE, ONE_QUESTION, salt="3")
    assert len(gateway.requests) == 2  # refused before anything was sent

    kept = jev.decide(STATE, ONE_QUESTION, salt="2")  # stored before the cap complained
    assert kept.cached is True and kept.answers == ONE_ANSWER
    assert spend.total == pytest.approx(0.00004)


# --------------------------------------------------------------------------- #
# Threads                                                                      #
# --------------------------------------------------------------------------- #
def test_identical_concurrent_requests_make_one_call(cache):
    workers = 8
    in_flight, release = threading.Event(), threading.Event()

    def held(request: httpx.Request) -> httpx.Response:
        in_flight.set()
        assert release.wait(timeout=WAIT_S)
        return httpx.Response(200, json=openrouter_body())

    gateway = Gateway(held)
    spend = SpendTracker(1.0)
    jev, _ = make_jev(gateway, cache=cache, spend=spend)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(jev.decide, STATE, ONE_QUESTION, salt="run") for _ in range(workers)]
        assert in_flight.wait(timeout=WAIT_S)
        release.set()
        results = [future.result(timeout=WAIT_S) for future in futures]

    assert len(gateway.requests) == 1
    assert sorted(result.cached for result in results) == [False] + [True] * (workers - 1)
    assert all(result.answers == ONE_ANSWER for result in results)
    assert spend.total == pytest.approx(0.00002)


def test_distinct_concurrent_requests_each_get_their_own_answers(cache):
    def echo(request: httpx.Request) -> httpx.Response:
        number = json.loads(request.content)["state"]["number"]
        answers = {"c0": {"type": "noul", "noul": number / 100}}
        return httpx.Response(200, json=openrouter_body(answers))

    gateway = Gateway(echo)
    spend = SpendTracker(1.0)
    jev, _ = make_jev(gateway, cache=cache, spend=spend)
    numbers = list(range(40))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: jev.decide({"number": n}, ONE_QUESTION), numbers))

    assert [result.answers["c0"]["noul"] for result in results] == [n / 100 for n in numbers]
    assert len(gateway.requests) == len(numbers)
    assert spend.total == pytest.approx(len(numbers) * 0.00002)
    assert all(jev.decide({"number": n}, ONE_QUESTION).cached for n in numbers)


# --------------------------------------------------------------------------- #
# build_jev                                                                    #
# --------------------------------------------------------------------------- #
def test_build_jev_speaks_openrouter(load_settings, route_http, cache):
    gateway = Gateway(ok())
    route_http(gateway)
    spend = SpendTracker(1.0)
    settings = load_settings(EVAL_GATEWAY="openrouter", OPENROUTER_API_KEY="sk-or-test")

    jev = build_jev(settings, cache=cache, spend=spend)
    first, second = jev.decide(STATE, ONE_QUESTION), jev.decide(STATE, ONE_QUESTION)

    assert (jev.provider, jev.model) == ("openrouter", "typesafe/jev-1.14")
    (request,) = gateway.requests
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["authorization"] == "Bearer sk-or-test"
    assert set(gateway.bodies[0]) == {"model", "state", "questions"}
    assert (first.cached, second.cached) == (False, True)
    assert spend.total == pytest.approx(0.00002)


def test_build_jev_follows_the_gateway_setting(load_settings, route_http):
    answers = [{"type": "predicate", "name": "c0", "probability": 0.9}]
    gateway = Gateway((200, vercel_body(answers)))
    route_http(gateway)
    load_settings(
        EVAL_GATEWAY="vercel", AI_GATEWAY_API_KEY="vck-test", EVAL_JEV_API_KEY="jev-key"
    )

    jev = build_jev()  # settings come from the environment
    result = jev.decide(STATE, ONE_QUESTION)

    assert (jev.provider, jev.model) == ("vercel", "typesafe-ai/jev-1.14.0")
    (request,) = gateway.requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/v1/decisions"
    assert request.headers["authorization"] == "Bearer jev-key"
    assert set(gateway.bodies[0]) == {"model", "input", "questions"}
    assert result.answers == {"c0": {"type": "noul", "noul": 0.9}}


def test_build_jev_refuses_the_default_gateway_which_serves_no_jev(load_settings, route_http):
    # OpenAI does not serve Jev. Its key must never reach a decisions endpoint,
    # whatever the Jev overrides say, and the refusal names the way forward.
    gateway = Gateway(ok())
    route_http(gateway)
    settings = load_settings(
        OPENAI_API_KEY="sk-test", EVAL_JEV_URL="https://jev.test/decisions", EVAL_JEV_API_KEY="jev-key"
    )

    assert (settings.gateway, settings.jev_url, settings.jev_api_key) == ("openai", "", None)
    assert settings.rankers == config.LOCAL_RANKERS == ("bge_m3", "bge_m3_reranker")
    with pytest.raises(JevError) as raised:
        build_jev(settings)
    message = str(raised.value)
    assert "openai gateway" in message and "openrouter, vercel" in message
    assert "bge_m3, bge_m3_reranker" in message
    assert gateway.requests == []
    assert load_settings(EVAL_GATEWAY="openrouter").rankers == config.RANKER_NAMES


@pytest.mark.parametrize("gateway_name", ["openrouter", "vercel"])
def test_build_jev_names_the_variable_a_missing_key_belongs_in(load_settings, gateway_name):
    with pytest.raises(JevError) as raised:
        build_jev(load_settings(EVAL_GATEWAY=gateway_name))

    key_env = GATEWAY_KEY_ENV[gateway_name]
    assert key_env in str(raised.value)
    assert "EVAL_JEV_API_KEY" in str(raised.value)
    # The variable the message names is the one the settings really read.
    settings = load_settings(EVAL_GATEWAY=gateway_name, **{key_env: "from-env"})
    assert settings.jev_api_key == "from-env"
