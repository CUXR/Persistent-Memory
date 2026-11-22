"""OpenAI-compatible chat client for the benchmark's LLM calls.

:class:`OpenAICompatLLM` implements :class:`~.interfaces.ChatLLM`: one JSON
object per call, cached on disk and counted against the spend cap.

:class:`GatewaySession`, the retrying HTTP layer underneath it, is shared with
the Jev client in ``jev.py``, so both hosted clients follow the one retry
policy the gateways document.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

import httpx

from . import config
from .cache import DiskCache, make_key
from .interfaces import LLMResult, Usage
from .spend import SpendTracker

# The environment variable that holds each gateway's key (as config.get_settings reads it).
GATEWAY_KEY_ENV: dict[str, str] = {name: spec["key_env"] for name, spec in config.GATEWAYS.items()}
MAX_RETRIES = 4  # five attempts in all, as in the gateway cookbook
# A completion arrives in one piece once the model has finished. A short read
# timeout would abandon a long generation after it was billed and then pay for
# it again on the retry, so wait as long as the vendors' own SDKs do.
LLM_TIMEOUT_S = 600.0
PARSE_ATTEMPTS = 3  # the call itself plus two repeats when the reply is not a JSON object
# OpenAI's own API spells three request fields differently from the routers
# (see ``OpenAICompatLLM.complete_json``); its GPT-4 families are the models
# that still take a temperature.
OPENAI_HOST = "api.openai.com"
OPENAI_SAMPLING_PREFIXES: tuple[str, ...] = ("gpt-4",)
SNIPPET_CHARS = 200

_DECODER = json.JSONDecoder()
logger = logging.getLogger(__name__)
_warned: set[str] = set()  # keys warn_once has already logged in this process
_warned_lock = threading.Lock()


def warn_once(key: str, message: str, *args: Any) -> None:
    """Log ``message`` the first time ``key`` is seen in this process, then stay quiet.

    For a condition that holds for every call of a run, such as a model
    without a list price: one line says it, one per call would bury the log.
    """

    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message, *args)


class HostedAPIError(RuntimeError):
    """A hosted call failed for good: a non-transient error, or the retries ran out.

    ``status`` is the HTTP status (None when no response arrived, or when the
    failure is not an HTTP one); ``message`` is the gateway's own wording.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(_describe(message, status))
        self.status = status
        self.message = message


def _describe(message: str, status: int | None) -> str:
    return message if status is None else f"HTTP {status}: {message}"


class LLMError(HostedAPIError):
    """The LLM client has no result: gateway error, reply without a JSON object, or no API key."""


def retry_delay(attempt: int, retry_after: str | None, jitter: random.Random) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (0-based).

    A numeric ``Retry-After`` header is honored as sent. Without one the wait
    is ``min(30, 2**attempt)`` seconds plus up to a second of jitter, which
    keeps parallel workers from retrying in step.
    """

    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = math.nan  # an HTTP-date, which the policy does not honor
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return min(30.0, 2.0**attempt) + jitter.uniform(0.0, 1.0)


class GatewaySession:
    """One endpoint, one authenticated ``httpx`` client, and the gateways' retry policy.

    Transient failures are retried up to ``max_retries`` times: HTTP 429 and
    5xx, transport errors and timeouts, a 2xx whose body is not a JSON object,
    and whatever ``also_transient(status, body)`` accepts. Waits follow
    :func:`retry_delay`. Every other failure raises ``error`` at once with the
    gateway's message. A gateway that reports a failure as an ``error`` object
    under HTTP 200 is treated as if it had sent that error's code as the status.

    The cap in ``spend`` is checked before every attempt, so no request is
    sent once it has been passed. The ``httpx`` client is thread-safe: one
    session serves all worker threads.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        error: type[HostedAPIError],
        max_retries: int,
        timeout: float,
        sleep: Callable[[float], None],
        spend: SpendTracker | None,
        transport: httpx.BaseTransport | None,
        also_transient: Callable[[int, str], bool] | None = None,
    ) -> None:
        if httpx.URL(url).scheme not in ("http", "https"):
            raise ValueError(f"endpoint must be an http(s) URL, got {url!r}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be at least 0, got {max_retries}")
        self.url = url
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._error = error
        self._max_retries = max_retries
        self._sleep = sleep
        self._spend = spend
        self._also_transient = also_transient
        self._jitter = random.Random(config.SEED)

    def post_json(self, body: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        """POST ``body``; return the response object and the seconds the successful call took."""

        attempt = 0
        while True:
            if self._spend is not None:
                self._spend.check()
            status: int | None = None
            cause: Exception | None = None
            retry_after: str | None = None
            started = perf_counter()
            try:
                response = self._http.post(self.url, json=body)
            except httpx.TransportError as exc:
                message = f"{type(exc).__name__}: {exc}"
                cause = exc
                transient = True
            else:
                latency_s = perf_counter() - started
                payload = _json_object(response)
                error = payload.get("error") if payload is not None else None
                if response.is_success and payload is not None and not error:
                    return payload, latency_s
                status = _error_status(response, error)
                message = _error_message(error, response.text)
                transient = self._is_transient(status, response.text) or (
                    response.is_success and payload is None
                )
                retry_after = response.headers.get("retry-after")
            if not transient:
                raise self._error(message, status=status) from cause
            if attempt == self._max_retries:
                gave_up = f"{message} (gave up after {attempt + 1} attempts)"
                raise self._error(gave_up, status=status) from cause
            delay = retry_delay(attempt, retry_after, self._jitter)
            logger.warning(
                "%s failed (%s); retry %d of %d in %.1fs",
                self.url,
                _describe(message, status),
                attempt + 1,
                self._max_retries,
                delay,
            )
            self._sleep(delay)
            attempt += 1

    def close(self) -> None:
        """Close the connection pool."""

        self._http.close()

    def _is_transient(self, status: int, body: str) -> bool:
        if status == 429 or status >= 500:
            return True
        return self._also_transient is not None and self._also_transient(status, body)


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    """The response body as a JSON object, or None when it is anything else."""

    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _error_status(response: httpx.Response, error: Any) -> int:
    """The HTTP status, or the body's error code when the failure came under a 2xx."""

    code = error.get("code") if isinstance(error, Mapping) else None
    if response.is_success and isinstance(code, int) and not isinstance(code, bool):
        return code
    return response.status_code


def _error_message(error: Any, body: str) -> str:
    """The gateway's words: the error's message and metadata, else the start of the body."""

    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        metadata = error.get("metadata")  # OpenRouter puts the upstream provider's reason here
        detail = f" {json.dumps(metadata)[:300]}" if metadata else ""
        return f"{error['message']}{detail}"
    if isinstance(error, str) and error:
        return error
    return body.strip()[:300] or "empty response body"


def as_number(value: Any) -> float | None:
    """``value`` as a finite float; None unless it is a number or a numeric string."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def token_count(value: Any) -> int:
    """A token count from a usage block; 0 when the gateway left it out."""

    number = as_number(value)
    return int(number) if number is not None and number > 0 else 0


def parse_json_object(text: str) -> dict[str, Any]:
    """The JSON object in an assistant message; ``ValueError`` when there is none.

    The object may stand alone, sit in a fenced code block, or have prose
    around it: decoding starts at the first ``{`` and ignores what follows the
    object. If that object does not decode (a reply cut off at ``max_tokens``,
    say) the reply is rejected rather than searched for a nested object, which
    would be returned as if it were the whole answer. A reply that is some
    other JSON value, such as an array, is rejected too.
    """

    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except ValueError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("no JSON object in the reply") from None
        value, _ = _DECODER.raw_decode(stripped, start)
    if not isinstance(value, dict):
        raise ValueError(f"the reply is a JSON {type(value).__name__}, not an object")
    return value


class OpenAICompatLLM:
    """:class:`~.interfaces.ChatLLM` over an OpenAI-compatible chat completions endpoint.

    Thread-safe: one instance, and so one connection pool, serves every worker
    thread. With a cache, a repeated request returns the stored answer and its
    original usage without a call or a charge; ``namespace`` and ``salt`` are
    part of the key, so changing either one forces a fresh call.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = config.GATEWAYS["openrouter"]["llm_base_url"],
        *,
        cache: DiskCache | None = None,
        spend: SpendTracker | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: float = LLM_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = GatewaySession(
            f"{self.base_url}/chat/completions",
            api_key,
            error=LLMError,
            max_retries=max_retries,
            timeout=timeout,
            sleep=sleep,
            spend=spend,
            transport=transport,
        )
        host = httpx.URL(self.base_url).host
        self._openrouter = host == "openrouter.ai" or host.endswith(".openrouter.ai")
        self._openai = host == OPENAI_HOST
        self._cache = cache
        self._spend = spend

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        namespace: str = "default",
        salt: str = "",
    ) -> LLMResult:
        """Return the JSON object ``model`` answers with.

        ``usage`` describes the one call that produced the object. A reply
        that is not a JSON object is asked for again (at most
        ``PARSE_ATTEMPTS`` calls, then :class:`LLMError`); those replies are
        charged to the spend cap but are neither cached nor counted in ``usage``.
        """

        if temperature is not None:
            temperature = float(temperature)  # 0 and 0.0 are one request and one cache key
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self._openai:
            # OpenAI's own spelling: the cap is max_completion_tokens (max_tokens
            # is deprecated there) and the effort a top-level string. A reasoning
            # model takes no temperature, and a sampling model no effort.
            body["max_completion_tokens"] = max_tokens
            if openai_takes_temperature(model):
                if temperature is not None:
                    body["temperature"] = temperature
            elif reasoning_effort is not None:
                body["reasoning_effort"] = reasoning_effort
        else:
            body["max_tokens"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature
            if reasoning_effort is not None:
                body["reasoning"] = {"effort": reasoning_effort}
        if self._openrouter:
            # OpenRouter's switch for per-call cost in the usage block. Other
            # OpenAI-compatible servers may reject a field they do not know.
            body["usage"] = {"include": True}

        if self._cache is None:
            result = self._call(model, body)
        else:
            key = make_key(
                "llm",
                model,
                system,
                user,
                max_tokens,
                temperature,
                reasoning_effort,
                namespace,
                salt,
            )
            with self._cache.lock(key):
                stored = self._cache.get(key)
                if stored is not None:
                    return LLMResult(
                        data=stored["data"],
                        usage=Usage.from_json(stored["usage"]),
                        model=stored["model"],
                        cached=True,
                        raw_text=stored["raw_text"],
                    )
                result = self._call(model, body)
                self._cache.set(
                    key,
                    {
                        "data": result.data,
                        "usage": result.usage.to_json(),
                        "model": result.model,
                        "raw_text": result.raw_text,
                    },
                )
        # Charged after the answer is stored: if this call is the one that
        # trips the cap, the answer it paid for is still there for the next run.
        self._charge(result.usage.cost_usd)
        return result

    def close(self) -> None:
        """Close the connection pool."""

        self._session.close()

    def _call(self, model: str, body: Mapping[str, Any]) -> LLMResult:
        """One uncached completion, repeated while the reply is not a JSON object."""

        snippet = ""
        finish_reason: Any = None
        for attempt in range(1, PARSE_ATTEMPTS + 1):
            payload, latency_s = self._session.post_json(body)
            choice = _first_choice(payload)
            text = _message_text(choice)
            usage = _usage(model, payload, latency_s)
            try:
                data = parse_json_object(text)
            except ValueError:
                self._charge(usage.cost_usd)
                snippet = (text or json.dumps(payload))[:SNIPPET_CHARS]
                finish_reason = choice.get("finish_reason")
                logger.warning(
                    "%s replied without a JSON object (attempt %d of %d, finish_reason=%r)",
                    model,
                    attempt,
                    PARSE_ATTEMPTS,
                    finish_reason,
                )
                continue
            served = payload.get("model")
            return LLMResult(
                data=data,
                usage=usage,
                model=served if isinstance(served, str) and served else model,
                raw_text=text,
            )
        raise LLMError(
            f"{model} did not return a JSON object in {PARSE_ATTEMPTS} attempts "
            f"(finish_reason={finish_reason!r}); last reply: {snippet!r}"
        )

    def _charge(self, cost_usd: float) -> None:
        if self._spend is not None:
            self._spend.add(cost_usd)


def openai_takes_temperature(model: str) -> bool:
    """Whether OpenAI's API accepts a temperature for ``model``.

    The GPT-4 families sample at the temperature given and have no reasoning
    effort. Everything later is a reasoning model: OpenAI's guide says to send
    those no temperature, so a caller's value is dropped and the reply is made
    reproducible by the response cache alone.
    """

    return model.startswith(OPENAI_SAMPLING_PREFIXES)


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """``choices[0]`` of a chat completions response; empty when there is none."""

    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    return first if isinstance(first, Mapping) else {}


def _message_text(choice: Mapping[str, Any]) -> str:
    """The assistant message's text; empty when the choice carries none."""

    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return content if isinstance(content, str) else ""


def fallback_prices() -> tuple[float, float]:
    """The highest listed input and output prices (USD per million tokens).

    What a call to a model outside ``config.PRICES_USD_PER_MTOK`` is charged
    when the gateway reports no cost for it: an upper bound, so the spend cap
    still sees the call. Charging nothing would let such a model run uncapped.
    """

    prices = config.PRICES_USD_PER_MTOK.values()
    return max(price[0] for price in prices), max(price[1] for price in prices)


def unpriced_models(settings: config.Settings) -> list[str]:
    """The configured chat models missing from ``config.PRICES_USD_PER_MTOK``, each once."""

    models = (
        settings.gen_model,
        settings.summary_model,
        settings.system_model,
        settings.judge_model,
    )
    return list(dict.fromkeys(model for model in models if model not in config.PRICES_USD_PER_MTOK))


def _usage(model: str, payload: Mapping[str, Any], latency_s: float) -> Usage:
    """Tokens and cost of one call: the gateway's cost if it sent one, else list prices.

    A model with no list price is charged at :func:`fallback_prices`, with a
    warning the first time, rather than at nothing.
    """

    block = payload.get("usage")
    reported = block if isinstance(block, Mapping) else {}
    input_tokens = token_count(reported.get("prompt_tokens"))
    output_tokens = token_count(reported.get("completion_tokens"))
    cost = as_number(reported.get("cost"))
    if cost is None or cost < 0:
        prices = config.PRICES_USD_PER_MTOK.get(model)
        if prices is None:
            prices = fallback_prices()
            warn_once(
                f"llm-price:{model}",
                "%s has no list price and the gateway reported no cost: charging the highest "
                "listed prices (%s in, %s out, USD per Mtok), an upper bound, so the spend cap "
                "still counts it; add the model to config.PRICES_USD_PER_MTOK for a true figure",
                model,
                *prices,
            )
        cost = (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_s=latency_s,
        calls=1,
    )


def build_llm(
    settings: config.Settings | None = None,
    cache: DiskCache | None = None,
    spend: SpendTracker | None = None,
) -> OpenAICompatLLM:
    """The LLM client for the configured gateway (``config.get_settings()`` by default).

    Per-call costs are only requested from OpenRouter, so on any other gateway
    every configured chat model must have a list price, or the spend figures
    would be upper bounds at best: the run is refused before its first call.
    """

    settings = settings or config.get_settings()
    if not settings.llm_api_key:
        raise LLMError(missing_key_message("LLM calls", settings.gateway, "EVAL_LLM_API_KEY"))
    unpriced = unpriced_models(settings)
    if unpriced and settings.gateway != "openrouter":
        raise LLMError(
            f"no list price for {', '.join(unpriced)}: per-call costs are only requested from "
            f"OpenRouter, so on the {settings.gateway} gateway the spend cap could not see these "
            "models' calls. Add them to config.PRICES_USD_PER_MTOK or set EVAL_GATEWAY=openrouter"
        )
    return OpenAICompatLLM(settings.llm_api_key, settings.llm_base_url, cache=cache, spend=spend)


def missing_key_message(purpose: str, gateway: str, override_env: str) -> str:
    """What to tell the operator when ``purpose`` has no API key configured."""

    key_env = GATEWAY_KEY_ENV.get(gateway, "the gateway's API key")
    return (
        f"no API key for {purpose}: set {key_env} (EVAL_GATEWAY={gateway}) "
        f"or {override_env} in the environment or in eval/.env"
    )
