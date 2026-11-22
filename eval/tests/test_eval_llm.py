"""OpenAICompatLLM against a mock gateway: requests, parsing, retries, caching, the spend cap."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from eval.retrieval import config
from eval.retrieval import llm as llm_module
from eval.retrieval.cache import DiskCache
from eval.retrieval.interfaces import ChatLLM
from eval.retrieval.llm import (
    GATEWAY_KEY_ENV,
    LLMError,
    OpenAICompatLLM,
    build_llm,
    parse_json_object,
    retry_delay,
)
from eval.retrieval.spend import SpendLimitExceeded, SpendTracker
from eval.retrieval.system import answering, extraction

MODEL = "openai/gpt-5-nano"  # listed in config.PRICES_USD_PER_MTOK at (0.05, 0.40)
HIGHEST_PRICES = (4.00, 20.00)  # the most expensive listed model (claude-opus-5.5)
SERVED = "openai/gpt-5-nano-2026-08-07"
COST = 0.000125
WAIT_S = 10  # upper bound for anything that would otherwise hang a broken test


def completion(content: str | None, *, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A chat-completions response body whose assistant message is ``content``."""

    return {
        "id": "gen-1",
        "model": SERVED,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": (
            {"prompt_tokens": 120, "completion_tokens": 30, "cost": COST}
            if usage is None
            else usage
        ),
    }


def ok(content: str = '{"answer": "Lisbon"}') -> tuple[int, dict[str, Any]]:
    return 200, completion(content)


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


def make_llm(gateway: Gateway, **kwargs: Any) -> tuple[OpenAICompatLLM, list[float]]:
    """A client wired to ``gateway``, plus the list its (never waiting) sleeps are recorded in."""

    sleeps: list[float] = []
    llm = OpenAICompatLLM(
        "sk-test", transport=httpx.MockTransport(gateway), sleep=sleeps.append, **kwargs
    )
    return llm, sleeps


def ask(llm: OpenAICompatLLM, **overrides: Any):
    request = {"model": MODEL, "system": "Extract the name.", "user": "Question: where is Sam?"}
    return llm.complete_json(**{**request, **overrides})


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
# Request                                                                      #
# --------------------------------------------------------------------------- #
def test_posts_the_documented_body_to_chat_completions():
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway)

    ask(llm)

    (request,) = gateway.requests
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"] == "application/json"
    assert gateway.bodies[0] == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Extract the name."},
            {"role": "user", "content": "Question: where is Sam?"},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "temperature": 0.0,
        "usage": {"include": True},
    }


def test_optional_fields_follow_the_arguments():
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway)

    ask(llm, max_tokens=64, temperature=None, reasoning_effort="low")
    ask(llm, temperature=0.7)

    first, second = gateway.bodies
    assert first["max_tokens"] == 64
    assert "temperature" not in first
    assert first["reasoning"] == {"effort": "low"}
    assert second["temperature"] == 0.7
    assert "reasoning" not in second


def test_cost_accounting_is_requested_from_openrouter_only():
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway, base_url="https://ai-gateway.vercel.sh/v1/")

    ask(llm)

    assert str(gateway.requests[0].url) == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert "usage" not in gateway.bodies[0]


def test_openai_gets_its_own_spelling_of_the_cap_and_the_effort():
    # OpenAI's API takes max_completion_tokens and a top-level reasoning_effort,
    # and its reasoning models answer 400 to a temperature other than the default.
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway, base_url="https://api.openai.com/v1")

    ask(llm, model="gpt-5-nano", max_tokens=64, temperature=0.0, reasoning_effort="low")
    ask(llm, model="gpt-6.1-terra", temperature=0.7)

    assert str(gateway.requests[0].url) == "https://api.openai.com/v1/chat/completions"
    first, second = gateway.bodies
    assert first == {
        "model": "gpt-5-nano",
        "messages": [
            {"role": "system", "content": "Extract the name."},
            {"role": "user", "content": "Question: where is Sam?"},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 64,
        "reasoning_effort": "low",
    }
    assert set(second) == {"model", "messages", "response_format", "max_completion_tokens"}


def test_openai_gpt4_models_keep_the_temperature_and_take_no_effort():
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway, base_url="https://api.openai.com/v1")

    ask(llm, model="gpt-4o-mini", temperature=0.0, reasoning_effort="low")

    (body,) = gateway.bodies
    assert body["temperature"] == 0.0 and body["max_completion_tokens"] == 1024
    assert not {"reasoning_effort", "reasoning", "max_tokens", "usage"} & set(body)


def test_system_under_test_calls_send_no_temperature():
    # The system model defaults to a reasoning model, which takes no temperature
    # (the generator's summary call on the same model passes None for that
    # reason). A gateway that rejects the field answers 400, which is not
    # retried, so the system stage would die on its first extraction.
    gateway = Gateway(
        ok('{"name": null, "time_range": null}'), ok('{"can_answer": false, "answer": ""}')
    )
    llm, _ = make_llm(gateway)
    question = "Where does Sam climb?"

    extraction.extract(question, llm, MODEL)
    answering.answer_from_summaries(question, [], llm, MODEL)
    answering.answer_with_dialogues(question, [], [], llm, MODEL)

    extract, from_summaries, with_dialogues = gateway.bodies
    assert "temperature" not in extract and extract["reasoning"] == {"effort": "low"}
    assert "temperature" not in from_summaries and "reasoning" not in from_summaries
    assert "temperature" not in with_dialogues and "reasoning" not in with_dialogues


@pytest.mark.parametrize(
    "kwargs", [{"base_url": "openrouter.ai/api/v1"}, {"max_retries": -1}], ids=["url", "retries"]
)
def test_rejects_a_configuration_that_cannot_work(kwargs):
    with pytest.raises(ValueError):
        OpenAICompatLLM("sk-test", **kwargs)


# --------------------------------------------------------------------------- #
# Result and usage                                                             #
# --------------------------------------------------------------------------- #
def test_result_carries_the_object_the_text_the_served_model_and_usage():
    llm, _ = make_llm(Gateway(ok('{"name": "Sam", "time_range": null}')))

    result = ask(llm)

    assert isinstance(llm, ChatLLM)
    assert result.data == {"name": "Sam", "time_range": None}
    assert result.raw_text == '{"name": "Sam", "time_range": null}'
    assert result.model == SERVED
    assert result.cached is False
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert result.usage.cost_usd == COST
    assert result.usage.calls == 1
    assert result.usage.latency_s >= 0


def test_latency_is_the_wall_time_of_the_successful_call_only(monkeypatch):
    # First attempt (a 500) runs from 10.0 to 13.0; the retry from 20.0 to 20.25.
    clock = iter([10.0, 13.0, 20.0, 20.25])
    monkeypatch.setattr(llm_module, "perf_counter", lambda: next(clock))
    llm, _ = make_llm(Gateway(error(500, "upstream down"), ok()))

    assert ask(llm).usage.latency_s == 0.25


@pytest.mark.parametrize(
    ("model", "usage", "expected"),
    [
        (MODEL, {"prompt_tokens": 120, "completion_tokens": 30}, (120 * 0.05 + 30 * 0.40) / 1e6),
        (MODEL, {"prompt_tokens": 120, "completion_tokens": 30, "cost": None}, 1.8e-05),
        (MODEL, {"prompt_tokens": 120, "completion_tokens": 30, "cost": 0}, 0.0),
        (
            "vendor/unlisted",
            {"prompt_tokens": 120, "completion_tokens": 30},
            (120 * HIGHEST_PRICES[0] + 30 * HIGHEST_PRICES[1]) / 1e6,
        ),
        (MODEL, None, 0.0),
    ],
    ids=["list-price", "null-cost", "reported-zero", "unlisted-model", "no-usage-block"],
)
def test_cost_is_the_gateways_number_else_list_prices_else_an_upper_bound(model, usage, expected):
    body = completion('{"a": 1}', usage=usage or {})
    if usage is None:
        del body["usage"]
    llm, _ = make_llm(Gateway((200, body)))

    result = ask(llm, model=model)

    assert result.usage.cost_usd == pytest.approx(expected, rel=1e-12, abs=0)
    assert result.usage.calls == 1


def test_fallback_prices_are_the_highest_listed():
    assert llm_module.fallback_prices() == HIGHEST_PRICES
    assert HIGHEST_PRICES == (
        max(price[0] for price in config.PRICES_USD_PER_MTOK.values()),
        max(price[1] for price in config.PRICES_USD_PER_MTOK.values()),
    )


def test_an_unlisted_model_counts_toward_the_cap_and_is_warned_about_once(monkeypatch, caplog):
    # Charging nothing would let a model outside the price table run uncapped on
    # a gateway that reports no cost; the upper bound keeps EVAL_MAX_SPEND_USD a bound.
    monkeypatch.setattr(llm_module, "_warned", set())
    body = completion('{"a": 1}', usage={"prompt_tokens": 100_000, "completion_tokens": 5_000})
    spend = SpendTracker(0.75)
    gateway = Gateway((200, body))
    llm, _ = make_llm(gateway, spend=spend)
    per_call = (100_000 * HIGHEST_PRICES[0] + 5_000 * HIGHEST_PRICES[1]) / 1e6  # $0.50

    with caplog.at_level(logging.WARNING, logger="eval.retrieval.llm"):
        ask(llm, model="vendor/unlisted")
        with pytest.raises(SpendLimitExceeded):
            ask(llm, model="vendor/unlisted", user="over the cap")

    assert len(gateway.requests) == 2
    assert spend.total == pytest.approx(2 * per_call)
    (warning,) = caplog.messages
    assert "vendor/unlisted" in warning and "PRICES_USD_PER_MTOK" in warning
    assert "4.0 in, 20.0 out" in warning


# --------------------------------------------------------------------------- #
# Parsing the reply                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        '{"a": {"b": [1, 2]}}',
        '\n  {"a": {"b": [1, 2]}}  \n',
        '```json\n{"a": {"b": [1, 2]}}\n```',
        '```\n{"a": {"b": [1, 2]}}\n```',
        'Here is the JSON you asked for:\n{"a": {"b": [1, 2]}}\nAnything else?',
        'Result:\n```json\n{"a": {"b": [1, 2]}}\n```\nDone {for now}.',
    ],
    ids=["bare", "padded", "json-fence", "plain-fence", "prose-around", "fence-in-prose"],
)
def test_reply_is_parsed_through_fences_and_prose(text):
    assert parse_json_object(text) == {"a": {"b": [1, 2]}}

    llm, _ = make_llm(Gateway(ok(text)))
    result = ask(llm)
    assert result.data == {"a": {"b": [1, 2]}}
    assert result.raw_text == text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I could not find that conversation.",
        '[{"a": 1}]',
        '"{\\"a\\": 1}"',
        '{"name": "Sam", "time_range": {"start": "2026-10-01", "end": "2026-10-08"}',
        'Format {name}: {"name": "Sam"}',
    ],
    ids=["empty", "prose-only", "array", "string", "cut-off", "brace-before-object"],
)
def test_reply_without_one_complete_object_is_rejected(text):
    # "cut-off" matters most: the nested time range must not pass for the whole answer.
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_unparseable_reply_is_asked_for_again(cache):
    gateway = Gateway(ok("Sure, one moment."), ok('{"answer": "Lisbon"}'))
    spend = SpendTracker(1.0)
    llm, sleeps = make_llm(gateway, cache=cache, spend=spend)

    result = ask(llm)

    assert len(gateway.requests) == 2
    assert sleeps == []  # a bad reply is not a transport failure: no backoff
    assert result.data == {"answer": "Lisbon"}
    assert result.usage.calls == 1 and result.usage.cost_usd == COST  # the good call alone
    assert spend.total == pytest.approx(2 * COST)  # but both calls were paid for
    assert ask(llm).cached is True


def test_three_unparseable_replies_raise_with_a_snippet(cache):
    gateway = Gateway(ok("Sorry, I can only answer in prose. " * 20))
    spend = SpendTracker(1.0)
    llm, _ = make_llm(gateway, cache=cache, spend=spend)

    with pytest.raises(LLMError, match="Sorry, I can only answer in prose") as raised:
        ask(llm)

    assert len(gateway.requests) == 3
    assert MODEL in str(raised.value) and "3 attempts" in str(raised.value)
    assert len(str(raised.value)) < 400  # a snippet, not the whole reply
    assert spend.total == pytest.approx(3 * COST)

    gateway.replies = (ok(),)  # nothing bad was cached: the next call goes out again
    assert ask(llm).cached is False
    assert len(gateway.requests) == 4


def test_empty_reply_is_reported_with_the_response_it_came_in():
    body = completion(None)
    body["choices"][0]["finish_reason"] = "length"
    llm, _ = make_llm(Gateway((200, body)))

    # The usual cause: a reasoning model spent all of max_tokens before answering.
    with pytest.raises(LLMError, match="finish_reason='length'") as raised:
        ask(llm)

    assert "gen-1" in str(raised.value)  # no text to quote, so the response itself is


# --------------------------------------------------------------------------- #
# Retries                                                                      #
# --------------------------------------------------------------------------- #
def test_429_is_retried_after_the_gateways_retry_after(caplog):
    gateway = Gateway(error(429, "slow down", **{"Retry-After": "7"}), ok())
    llm, sleeps = make_llm(gateway)

    with caplog.at_level(logging.WARNING, logger="eval.retrieval.llm"):
        result = ask(llm)

    assert result.data == {"answer": "Lisbon"}
    assert len(gateway.requests) == 2
    assert sleeps == [7.0]
    (warning,) = caplog.messages  # a client that waits says why and for how long
    assert "HTTP 429: slow down" in warning and "in 7.0s" in warning
    assert "sk-test" not in warning


def test_5xx_is_retried_with_exponential_backoff_and_jitter():
    gateway = Gateway(error(500, "boom"), error(503, "busy"), error(529, "overloaded"), ok())
    llm, sleeps = make_llm(gateway)

    result = ask(llm)

    assert result.data == {"answer": "Lisbon"}
    assert len(gateway.requests) == 4
    assert [int(seconds) for seconds in sleeps] == [1, 2, 4]  # 2**attempt, plus under 1s jitter
    assert all(seconds != int(seconds) for seconds in sleeps)


def test_transport_errors_and_timeouts_are_retried():
    failures = [httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out")]
    gateway = Gateway(*failures, ok())
    llm, sleeps = make_llm(gateway)

    assert ask(llm).data == {"answer": "Lisbon"}
    assert len(gateway.requests) == 3
    assert len(sleeps) == 2


def test_gives_up_when_the_retries_run_out():
    gateway = Gateway(error(503, "busy"))
    llm, sleeps = make_llm(gateway, max_retries=2)

    with pytest.raises(LLMError, match="busy") as raised:
        ask(llm)

    assert raised.value.status == 503
    assert "gave up after 3 attempts" in str(raised.value)
    assert len(gateway.requests) == 3
    assert len(sleeps) == 2


def test_a_dead_connection_surfaces_as_llm_error_with_its_cause():
    failure = httpx.ConnectError("connection refused")
    llm, _ = make_llm(Gateway(failure), max_retries=1)

    with pytest.raises(LLMError, match="ConnectError: connection refused") as raised:
        ask(llm)

    assert raised.value.status is None
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("status", [400, 401, 402, 404, 413, 422])
def test_other_4xx_raise_at_once_with_the_gateways_message(status):
    gateway = Gateway(error(status, "max_tokens is too large for this model"))
    llm, sleeps = make_llm(gateway)

    with pytest.raises(LLMError, match="max_tokens is too large for this model") as raised:
        ask(llm)

    assert raised.value.status == status
    assert raised.value.message == "max_tokens is too large for this model"
    assert len(gateway.requests) == 1
    assert sleeps == []


def test_error_detail_from_the_upstream_provider_is_kept():
    detail = {"provider_name": "OpenAI", "raw": "Unsupported value: temperature"}
    body = {"error": {"code": 400, "message": "Provider returned error", "metadata": detail}}
    llm, _ = make_llm(Gateway((400, body)))

    with pytest.raises(LLMError, match="Provider returned error.*Unsupported value: temperature"):
        ask(llm)


def test_an_error_that_is_not_json_is_quoted():
    reply = httpx.Response(502, text="<html>Bad Gateway</html>")
    llm, _ = make_llm(Gateway(lambda request: reply), max_retries=0)

    with pytest.raises(LLMError, match="Bad Gateway") as raised:
        ask(llm)

    assert raised.value.status == 502


def test_an_error_sent_under_http_200_is_handled_by_its_code():
    transient = (200, {"error": {"code": 502, "message": "Provider disconnected"}})
    llm, sleeps = make_llm(Gateway(transient, ok()))
    assert ask(llm).data == {"answer": "Lisbon"}
    assert len(sleeps) == 1

    permanent = Gateway((200, {"error": {"code": 400, "message": "Invalid model"}}))
    llm, sleeps = make_llm(permanent)
    with pytest.raises(LLMError, match="Invalid model") as raised:
        ask(llm)
    assert raised.value.status == 400
    assert len(permanent.requests) == 1 and sleeps == []


def test_a_2xx_that_is_not_json_is_retried():
    garbage = httpx.Response(200, text="<html>checking your browser</html>")
    gateway = Gateway(lambda request: garbage, ok())
    llm, sleeps = make_llm(gateway)

    assert ask(llm).data == {"answer": "Lisbon"}
    assert len(gateway.requests) == 2 and len(sleeps) == 1


@pytest.mark.parametrize(
    ("kwargs", "seconds"), [({}, 600.0), ({"timeout": 5.0}, 5.0)], ids=["default", "given"]
)
def test_timeout_is_a_constructor_argument(kwargs, seconds):
    seen: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json=completion('{"a": 1}'))

    llm, _ = make_llm(Gateway(record), **kwargs)
    ask(llm)

    assert seen == [{"connect": seconds, "read": seconds, "write": seconds, "pool": seconds}]


def test_retry_delay_policy():
    jitter = random.Random(0)
    for attempt, base in [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16), (5, 30), (12, 30)]:
        assert base <= retry_delay(attempt, None, jitter) < base + 1  # min(30, 2**attempt) + jitter

    assert retry_delay(3, "0", jitter) == 0.0
    assert retry_delay(3, "2.5", jitter) == 2.5
    for not_a_wait in ["Wed, 04 Nov 2026 18:00:00 GMT", "-5", "nan", "inf", ""]:
        assert 8 <= retry_delay(3, not_a_wait, jitter) < 9  # only a usable number is honored


# --------------------------------------------------------------------------- #
# Cache                                                                        #
# --------------------------------------------------------------------------- #
def test_cache_hit_returns_the_stored_answer_without_a_call_or_a_charge(cache):
    gateway = Gateway(ok('```json\n{"answer": "Lisbon"}\n```'))
    spend = SpendTracker(1.0)
    llm, _ = make_llm(gateway, cache=cache, spend=spend)

    fresh = ask(llm)
    hit = ask(llm)

    assert len(gateway.requests) == 1
    assert fresh.cached is False and hit.cached is True
    assert hit.data == fresh.data == {"answer": "Lisbon"}
    assert hit.raw_text == fresh.raw_text
    assert hit.model == SERVED
    assert hit.usage == fresh.usage  # the original call's usage, cost included
    assert spend.total == pytest.approx(COST)  # charged once


@pytest.mark.parametrize(
    "change",
    [
        {"model": "openai/gpt-5-mini"},
        {"system": "Extract the date."},
        {"user": "Question: where is Alex?"},
        {"max_tokens": 64},
        {"temperature": 0.7},
        {"temperature": None},
        {"reasoning_effort": "low"},
        {"namespace": "answer"},
        {"salt": "run-2"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_request_field_namespace_and_salt_is_part_of_the_key(cache, change):
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway, cache=cache)

    ask(llm)
    changed = ask(llm, **change)
    repeated = ask(llm, **change)

    assert changed.cached is False and repeated.cached is True
    assert len(gateway.requests) == 2


def test_integer_and_float_temperature_are_one_request(cache):
    gateway = Gateway(ok())
    llm, _ = make_llm(gateway, cache=cache)

    ask(llm, temperature=0)

    assert ask(llm, temperature=0.0).cached is True
    assert isinstance(gateway.bodies[0]["temperature"], float)


def test_without_a_cache_every_call_is_fresh():
    gateway = Gateway(ok())
    spend = SpendTracker(1.0)
    llm, _ = make_llm(gateway, spend=spend)

    assert ask(llm).cached is False
    assert ask(llm).cached is False
    assert len(gateway.requests) == 2
    assert spend.total == pytest.approx(2 * COST)


def test_a_new_client_resumes_from_the_cache_file(tmp_path):
    path = tmp_path / "cache.sqlite"
    first_cache = DiskCache(path)
    first, _ = make_llm(Gateway(ok()), cache=first_cache)
    ask(first)
    first_cache.close()

    second_cache = DiskCache(path)
    try:
        gateway = Gateway(error(500, "must not be called"))
        second, _ = make_llm(gateway, cache=second_cache)
        hit = ask(second)
    finally:
        second_cache.close()

    assert hit.cached is True and hit.data == {"answer": "Lisbon"}
    assert gateway.requests == []


# --------------------------------------------------------------------------- #
# Spend cap                                                                    #
# --------------------------------------------------------------------------- #
def test_cap_stops_requests_but_keeps_what_was_paid_for(cache):
    gateway = Gateway(ok())
    spend = SpendTracker(COST * 1.5)
    llm, _ = make_llm(gateway, cache=cache, spend=spend)

    ask(llm, salt="1")
    with pytest.raises(SpendLimitExceeded):
        ask(llm, salt="2")  # this call takes the total past the cap
    assert len(gateway.requests) == 2

    with pytest.raises(SpendLimitExceeded):
        ask(llm, salt="3")
    assert len(gateway.requests) == 2  # refused before anything was sent

    kept = ask(llm, salt="2")  # the call that tripped the cap was stored first
    assert kept.cached is True and kept.data == {"answer": "Lisbon"}
    assert spend.total == pytest.approx(2 * COST)


def test_cap_is_checked_again_before_each_retry():
    spend = SpendTracker(1.0)

    def trip_the_cap(seconds: float) -> None:  # another thread spends while this one backs off
        with pytest.raises(SpendLimitExceeded):
            spend.add(2.0)

    gateway = Gateway(error(503, "busy"), ok())
    llm = OpenAICompatLLM(
        "sk-test", transport=httpx.MockTransport(gateway), sleep=trip_the_cap, spend=spend
    )

    with pytest.raises(SpendLimitExceeded):
        ask(llm)
    assert len(gateway.requests) == 1


# --------------------------------------------------------------------------- #
# Threads                                                                      #
# --------------------------------------------------------------------------- #
def test_identical_concurrent_requests_make_one_call(cache):
    workers = 8
    in_flight, release = threading.Event(), threading.Event()

    def held(request: httpx.Request) -> httpx.Response:
        in_flight.set()
        assert release.wait(timeout=WAIT_S)
        return httpx.Response(200, json=completion('{"answer": "Lisbon"}'))

    gateway = Gateway(held)
    spend = SpendTracker(1.0)
    llm, _ = make_llm(gateway, cache=cache, spend=spend)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, llm) for _ in range(workers)]
        assert in_flight.wait(timeout=WAIT_S)
        release.set()
        results = [future.result(timeout=WAIT_S) for future in futures]

    assert len(gateway.requests) == 1
    assert sorted(result.cached for result in results) == [False] + [True] * (workers - 1)
    assert all(result.data == {"answer": "Lisbon"} for result in results)
    assert spend.total == pytest.approx(COST)


def test_distinct_concurrent_requests_each_get_their_own_answer(cache):
    def echo(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(200, json=completion(json.dumps({"echo": user})))

    gateway = Gateway(echo)
    spend = SpendTracker(1.0)
    llm, _ = make_llm(gateway, cache=cache, spend=spend)
    users = [f"Question {number}" for number in range(40)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda user: ask(llm, user=user), users))

    assert [result.data for result in results] == [{"echo": user} for user in users]
    assert len(gateway.requests) == len(users)
    assert spend.total == pytest.approx(len(users) * COST)
    assert all(ask(llm, user=user).cached for user in users)


# --------------------------------------------------------------------------- #
# build_llm                                                                    #
# --------------------------------------------------------------------------- #
def test_build_llm_wires_settings_cache_and_spend(load_settings, route_http, cache):
    gateway = Gateway(ok())
    route_http(gateway)
    spend = SpendTracker(1.0)
    settings = load_settings(EVAL_GATEWAY="vercel", AI_GATEWAY_API_KEY="vck-test")

    llm = build_llm(settings, cache=cache, spend=spend)
    first, second = ask(llm), ask(llm)

    (request,) = gateway.requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer vck-test"
    assert (first.cached, second.cached) == (False, True)
    assert spend.total == pytest.approx(COST)


def test_build_llm_defaults_to_the_environment_settings(load_settings, route_http):
    gateway = Gateway(ok())
    route_http(gateway)
    load_settings(OPENAI_API_KEY="sk-test", EVAL_LLM_BASE_URL="https://llm.test/v1")

    ask(build_llm())

    (request,) = gateway.requests
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"


def test_the_default_gateway_is_openai_with_models_it_serves_and_prices(load_settings, route_http):
    # The project pays for OpenAI alone: with nothing but its key set, every
    # call goes to api.openai.com under a bare model id that has a list price
    # (OpenAI reports no cost per call, so the spend cap counts from the table).
    gateway = Gateway((200, completion('{"answer": "Lisbon"}', usage={"prompt_tokens": 120, "completion_tokens": 30})))
    route_http(gateway)
    settings = load_settings(OPENAI_API_KEY="sk-test")
    spend = SpendTracker(1.0)

    assert (settings.gateway, settings.llm_base_url) == ("openai", "https://api.openai.com/v1")
    assert (settings.gen_model, settings.judge_model) == ("gpt-6.1-terra", "gpt-6.1-terra")
    assert (settings.summary_model, settings.system_model) == ("gpt-5-nano", "gpt-5-nano")
    assert llm_module.unpriced_models(settings) == []
    result = ask(build_llm(settings, spend=spend), model=settings.system_model)

    (request,) = gateway.requests
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert result.usage.cost_usd == pytest.approx((120 * 0.05 + 30 * 0.40) / 1_000_000)
    assert spend.total == pytest.approx(result.usage.cost_usd)


def test_build_llm_refuses_an_unpriced_model_where_no_cost_is_reported(load_settings, route_http):
    # Costs are only requested from OpenRouter; elsewhere a model outside the
    # price table would be charged an estimate at best, so the run stops before
    # its first call and says which models to list.
    route_http(Gateway(ok()))
    models = {"EVAL_GEN_MODEL": "vendor/writer", "EVAL_JUDGE_MODEL": "vendor/grader"}

    with pytest.raises(LLMError) as raised:
        build_llm(load_settings(EVAL_GATEWAY="vercel", AI_GATEWAY_API_KEY="vck-test", **models))
    message = str(raised.value)
    assert "vendor/writer, vendor/grader" in message
    assert "PRICES_USD_PER_MTOK" in message and "EVAL_GATEWAY=openrouter" in message

    # The same models on OpenRouter: every call reports its cost, so nothing to refuse.
    llm = build_llm(load_settings(EVAL_GATEWAY="openrouter", OPENROUTER_API_KEY="sk-or", **models))
    assert ask(llm, model="vendor/writer").usage.cost_usd == COST


def test_unpriced_models_lists_each_configured_chat_model_once(load_settings):
    settings = load_settings(
        EVAL_SYSTEM_MODEL="vendor/same",
        EVAL_SUMMARY_MODEL="vendor/same",
        EVAL_GEN_MODEL="vendor/gen",
    )
    assert llm_module.unpriced_models(settings) == ["vendor/gen", "vendor/same"]
    assert llm_module.unpriced_models(load_settings()) == []  # the defaults are all listed


@pytest.mark.parametrize("gateway_name", ["openai", "openrouter", "vercel"])
def test_build_llm_names_the_variable_a_missing_key_belongs_in(load_settings, gateway_name):
    with pytest.raises(LLMError) as raised:
        build_llm(load_settings(EVAL_GATEWAY=gateway_name))

    key_env = GATEWAY_KEY_ENV[gateway_name]
    assert key_env in str(raised.value)
    assert "EVAL_LLM_API_KEY" in str(raised.value)
    # The variable the message names is the one the settings really read.
    settings = load_settings(EVAL_GATEWAY=gateway_name, **{key_env: "from-env"})
    assert settings.llm_api_key == "from-env"
