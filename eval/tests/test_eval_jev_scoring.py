"""Unit tests for how Jev is applied to candidates (offline, against a scripted client)."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from eval.retrieval.interfaces import DecisionClient, DecisionResult, Usage
from eval.retrieval.pipelines.jev_scoring import (
    MAX_CHOICE_OPTIONS,
    MISSING_SCORE,
    MODES,
    SCORE_LEVELS,
    WORDINGS,
    JevScoringConfig,
    score_candidates,
)

QUERY = "What did Sam say about the climbing trip?"
SNAPSHOT = "typesafe/jev-1.14-20261014"
TASK_STATEMENT = "find which of the owner's past conversations answer the owner's question"
# Batched mode: how a question's instructions carry the candidate's summary (pinned:
# the wire format is part of the cache key and of what a dev-study pin was measured on).
BATCHED_SUMMARY_SEPARATOR = "\n\nconversation_summary: "

# (episode_id, summary); the summaries differ in length so no two requests cost the same.
CANDIDATES = [
    ("0b6f6c1e-51f0-5c0e-9d4f-000000000001", "Sam booked the climbing gym for Saturday morning."),
    ("0b6f6c1e-51f0-5c0e-9d4f-000000000002", "Sam and I compared notes on sourdough starters."),
    ("0b6f6c1e-51f0-5c0e-9d4f-000000000003", "Priya asked for the quarterly report."),
    ("0b6f6c1e-51f0-5c0e-9d4f-000000000004", "Sam said the climbing trip moves to early May."),
    ("0b6f6c1e-51f0-5c0e-9d4f-000000000005", "Jordan recommended a physio."),
]
IDS = [episode_id for episode_id, _ in CANDIDATES]
SUMMARIES = [summary for _, summary in CANDIDATES]
RELEVANCE = dict(zip(SUMMARIES, (0.62, 0.11, 0.02, 0.97, 0.05)))

Script = Callable[[Any, str, Mapping[str, Any]], "dict[str, Any] | None"]


def request_usage(state: Any, questions: Mapping[str, Any]) -> Usage:
    """Usage that differs per request, so a wrong count or summation order shows up."""

    tokens = len(json.dumps([state, questions], sort_keys=True))
    return Usage(
        input_tokens=tokens,
        output_tokens=len(questions),
        cost_usd=tokens * 0.042e-6,
        latency_s=tokens / 1e4,
        calls=1,
    )


class ScriptedJev:
    """DecisionClient fake: records every request and answers from a scripted function.

    The script gets ``(state, question_name, question)`` and returns that
    question's answer, or None to leave the answer out of the response.
    """

    model = "typesafe/jev-1.14"

    def __init__(
        self,
        script: Script,
        snapshots: tuple[str, ...] = (SNAPSHOT,),
        cached: bool = False,
    ) -> None:
        self.script = script
        self.snapshots = snapshots
        self.cached = cached
        self.requests: list[tuple[Any, dict[str, Any], str]] = []  # (state, questions, salt)
        self._lock = threading.Lock()

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        salt: str = "",
    ) -> DecisionResult:
        with self._lock:
            arrival = len(self.requests)
            self.requests.append((state, dict(questions), salt))
        answers = {}
        for name, question in questions.items():
            answer = self.script(state, name, question)
            if answer is not None:
                answers[name] = answer
        return DecisionResult(
            answers=answers,
            usage=request_usage(state, questions),
            model=self.snapshots[arrival % len(self.snapshots)],
            cached=self.cached,
        )


def split_batched(question: Mapping[str, Any]) -> tuple[str, str]:
    """``(instruction text, summary)`` of a batched question, whose instructions carry both."""

    text, _, summary = question["instructions"].partition(BATCHED_SUMMARY_SEPARATOR)
    return text, summary


def summary_of(state: Any, question: Mapping[str, Any]) -> str:
    """The candidate summary a pair or batched question is about."""

    if "conversation_summary" in state:
        return state["conversation_summary"]
    return split_batched(question)[1]


def instruction_text(question: Mapping[str, Any]) -> str:
    return split_batched(question)[0]


def noul_script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "noul", "noul": RELEVANCE[summary_of(state, question)]}


def choice_script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = {option: RELEVANCE[summary] for option, summary in question["criteria"].items()}
    best = max(probabilities, key=lambda option: probabilities[option])
    return {"type": "choice", "choice": best, "confidence": 0.5, "probabilities": probabilities}


def expected_usage(client: ScriptedJev) -> Usage:
    usages = (request_usage(state, questions) for state, questions, _ in client.requests)
    return sum(usages, Usage())


# --------------------------------------------------------------------------- #
# JevScoringConfig                                                             #
# --------------------------------------------------------------------------- #
def test_default_config_is_batched_noul_v1() -> None:
    config = JevScoringConfig()

    assert (config.mode, config.primitive, config.wording) == ("batched", "noul", "v1")
    assert (config.batch_size, config.concurrency, config.salt) == (16, 8, "")


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        JevScoringConfig().mode = "pair"  # type: ignore[misc]


def test_config_round_trips_through_json() -> None:
    config = JevScoringConfig(
        mode="pair", primitive="score", wording="v2", batch_size=4, concurrency=2, salt="run-2"
    )

    assert JevScoringConfig.from_json(json.loads(json.dumps(config.to_json()))) == config
    assert JevScoringConfig.from_json({"mode": "listwise"}) == JevScoringConfig(mode="listwise")


def test_from_json_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="batchsize"):
        JevScoringConfig.from_json({"mode": "batched", "batchsize": 4})


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "pointwise"},
        {"primitive": "choice"},
        {"wording": "v9"},
        {"batch_size": 0},
        {"concurrency": 0},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(overrides))):
        JevScoringConfig(**overrides)


def test_label_names_the_variant_and_ignores_salt() -> None:
    pair = JevScoringConfig(mode="pair", primitive="score", wording="v2")

    assert JevScoringConfig().label == "batched-noul-v1-b16"
    assert pair.label == "pair-score-v2"
    assert JevScoringConfig(mode="listwise", wording="v2").label == "listwise-v2"
    assert JevScoringConfig(salt="run-3").label == JevScoringConfig().label


@pytest.mark.parametrize("mode", MODES)
def test_label_ignores_concurrency(mode: str) -> None:
    # Concurrency is a client setting (the runtime overrides it from the
    # environment): it changes how fast the scores arrive, not what is asked.
    assert (
        JevScoringConfig(mode=mode, concurrency=1).label
        == JevScoringConfig(mode=mode, concurrency=32).label
    )
    assert "c1" not in JevScoringConfig(mode=mode, concurrency=1).label.split("-")


def test_label_separates_variants_that_behave_differently() -> None:
    variants = [
        JevScoringConfig(),
        JevScoringConfig(batch_size=8),
        JevScoringConfig(primitive="score"),
        JevScoringConfig(wording="v2"),
        JevScoringConfig(mode="pair"),
        JevScoringConfig(mode="listwise"),
    ]

    assert len({variant.label for variant in variants}) == len(variants)
    # Fields a mode does not use do not split it into several labels.
    assert JevScoringConfig(mode="pair", batch_size=4).label == JevScoringConfig(mode="pair").label
    assert (
        JevScoringConfig(mode="listwise", primitive="score", batch_size=4, concurrency=1).label
        == JevScoringConfig(mode="listwise").label
    )


# --------------------------------------------------------------------------- #
# Request shapes                                                               #
# --------------------------------------------------------------------------- #
def test_pair_mode_sends_one_request_per_candidate_with_the_summary_in_the_state() -> None:
    client = ScriptedJev(noul_script)
    config = JevScoringConfig(mode="pair", concurrency=1)

    result = score_candidates(client, QUERY, CANDIDATES, config)

    assert [state for state, _, _ in client.requests] == [
        {"query": QUERY, "conversation_summary": summary} for summary in SUMMARIES
    ]
    for _, questions, _ in client.requests:
        assert list(questions) == ["c0"]
        question = questions["c0"]
        assert question["type"] == "noul"
        assert isinstance(question["instructions"], str)
        assert set(question["criteria"]) == {"true", "false"}
    assert result.scores == {episode_id: RELEVANCE[summary] for episode_id, summary in CANDIDATES}
    assert (result.n_requests, result.n_missing) == (len(CANDIDATES), 0)


def test_batched_mode_chunks_candidates_and_keeps_the_summary_out_of_the_state() -> None:
    client = ScriptedJev(noul_script)
    config = JevScoringConfig(mode="batched", batch_size=2, concurrency=1)

    result = score_candidates(client, QUERY, CANDIDATES, config)

    assert [state for state, _, _ in client.requests] == [{"query": QUERY}] * 3
    names = [list(questions) for _, questions, _ in client.requests]
    assert names == [["c0", "c1"], ["c0", "c1"], ["c0"]]
    sent = [question for _, questions, _ in client.requests for question in questions.values()]
    # The instructions are one plain string, the same text for every candidate with
    # that candidate's summary after it: a structured object would reach the two
    # gateways in different shapes, and OpenRouter's schema promises only a string.
    assert all(isinstance(question["instructions"], str) for question in sent)
    texts = {split_batched(question)[0] for question in sent}
    assert len(texts) == 1 and TASK_STATEMENT in texts.pop()
    assert [split_batched(question)[1] for question in sent] == SUMMARIES
    assert sent[0]["instructions"].endswith(f"{BATCHED_SUMMARY_SEPARATOR}{SUMMARIES[0]}")
    assert all(
        question["type"] == "noul" and set(question["criteria"]) == {"true", "false"}
        for question in sent
    )
    assert result.scores == {episode_id: RELEVANCE[summary] for episode_id, summary in CANDIDATES}
    assert (result.n_requests, result.n_missing) == (3, 0)


@pytest.mark.parametrize(("batch_size", "sizes"), [(1, [1] * 5), (4, [4, 1]), (5, [5]), (16, [5])])
def test_batched_mode_fills_each_request_up_to_batch_size(
    batch_size: int, sizes: list[int]
) -> None:
    client = ScriptedJev(noul_script)
    config = JevScoringConfig(batch_size=batch_size, concurrency=1)

    score_candidates(client, QUERY, CANDIDATES, config)

    assert [len(questions) for _, questions, _ in client.requests] == sizes


def test_listwise_mode_asks_one_choice_question_over_all_candidates() -> None:
    client = ScriptedJev(choice_script)

    result = score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(mode="listwise"))

    assert len(client.requests) == 1
    state, questions, _ = client.requests[0]
    assert state == {"query": QUERY}
    (question,) = questions.values()
    assert question["type"] == "choice"
    assert isinstance(question["instructions"], str)
    assert question["criteria"] == {f"c{index}": summary for index, summary in enumerate(SUMMARIES)}
    assert result.scores == {episode_id: RELEVANCE[summary] for episode_id, summary in CANDIDATES}
    assert (result.n_requests, result.n_missing) == (1, 0)


@pytest.mark.parametrize("mode", ["pair", "batched"])
def test_score_primitive_asks_a_four_level_rubric_and_rescales_it(mode: str) -> None:
    levels = dict(zip(SUMMARIES, (2.25, 0.75, 0.0, 3.0, 1.5)))

    def script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        level = levels[summary_of(state, question)]
        return {"type": "score", "score": level, "confidence": 0.9, "probabilities": {"3": 1.0}}

    client = ScriptedJev(script)
    config = JevScoringConfig(mode=mode, primitive="score", batch_size=2, concurrency=1)

    result = score_candidates(client, QUERY, CANDIDATES, config)

    assert len(SCORE_LEVELS) == 4
    for _, questions, _ in client.requests:
        for question in questions.values():
            assert question["type"] == "score"
            assert question["criteria"] == list(SCORE_LEVELS)
    assert result.scores == pytest.approx(dict(zip(IDS, (0.75, 0.25, 0.0, 1.0, 0.5))))


@pytest.mark.parametrize("mode", ["pair", "batched", "listwise"])
def test_requests_name_candidates_by_position_and_never_by_id(mode: str) -> None:
    client = ScriptedJev(choice_script if mode == "listwise" else noul_script)

    score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(mode=mode, batch_size=2))

    payload = json.dumps([[state, questions] for state, questions, _ in client.requests])
    assert not any(episode_id in payload for episode_id in IDS)
    for _, questions, _ in client.requests:
        names = list(questions.values())[0]["criteria"] if mode == "listwise" else questions
        assert all(re.fullmatch(r"c\d+", name) for name in names)


@pytest.mark.parametrize("mode", ["pair", "batched", "listwise"])
@pytest.mark.parametrize("primitive", ["noul", "score"])
@pytest.mark.parametrize("wording", sorted(WORDINGS))
def test_every_variant_states_the_task_plainly(mode: str, primitive: str, wording: str) -> None:
    client = ScriptedJev(lambda state, name, question: {"type": "refusal"})
    config = JevScoringConfig(mode=mode, primitive=primitive, wording=wording)

    score_candidates(client, QUERY, CANDIDATES, config)

    for _, questions, _ in client.requests:
        for question in questions.values():
            assert TASK_STATEMENT in instruction_text(question)


@pytest.mark.parametrize(
    ("mode", "primitive"),
    [("pair", "noul"), ("batched", "score"), ("listwise", "noul")],
)
def test_v2_wording_asks_a_stricter_question_than_v1(mode: str, primitive: str) -> None:
    texts = {}
    for wording in ("v1", "v2"):
        client = ScriptedJev(lambda state, name, question: {"type": "refusal"})
        config = JevScoringConfig(mode=mode, primitive=primitive, wording=wording)
        score_candidates(client, QUERY, CANDIDATES[:1], config)
        (question,) = client.requests[0][1].values()
        texts[wording] = instruction_text(question)

    assert texts["v1"] != texts["v2"]
    assert "merely shares a topic or a person" in texts["v2"]
    assert "merely shares a topic or a person" not in texts["v1"]


# --------------------------------------------------------------------------- #
# Limits and missing answers                                                   #
# --------------------------------------------------------------------------- #
def test_listwise_rejects_more_than_255_candidates_before_any_request() -> None:
    client = ScriptedJev(choice_script)
    too_many = [(f"ep-{index}", f"summary {index}") for index in range(MAX_CHOICE_OPTIONS + 1)]

    with pytest.raises(ValueError, match="255"):
        score_candidates(client, QUERY, too_many, JevScoringConfig(mode="listwise"))

    assert client.requests == []


def test_listwise_accepts_exactly_255_candidates() -> None:
    def uniform(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        probabilities = {option: 1 / len(question["criteria"]) for option in question["criteria"]}
        return {"type": "choice", "choice": "c0", "confidence": 0.0, "probabilities": probabilities}

    client = ScriptedJev(uniform)
    candidates = [(f"ep-{index}", f"summary {index}") for index in range(MAX_CHOICE_OPTIONS)]

    result = score_candidates(client, QUERY, candidates, JevScoringConfig(mode="listwise"))

    assert MAX_CHOICE_OPTIONS == 255
    assert len(client.requests[0][1]["pick"]["criteria"]) == 255
    assert len(result.scores) == 255 and result.n_missing == 0


@pytest.mark.parametrize("mode", ["pair", "batched"])
def test_refused_absent_and_unusable_answers_score_minus_one_and_are_counted(mode: str) -> None:
    answers: dict[str, dict[str, Any] | None] = {
        SUMMARIES[0]: {"type": "noul", "noul": 0.4},
        SUMMARIES[1]: {"type": "refusal"},
        SUMMARIES[2]: None,  # no answer at all
        SUMMARIES[3]: {"type": "noul"},  # an answer without its number
        SUMMARIES[4]: {"type": "score", "score": 2.0},  # not the type that was asked for
    }
    client = ScriptedJev(lambda state, name, question: answers[summary_of(state, question)])

    result = score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(mode=mode, batch_size=3))

    assert MISSING_SCORE == -1.0
    assert result.scores == dict(zip(IDS, (0.4, -1.0, -1.0, -1.0, -1.0)))
    assert result.n_missing == 4


def test_listwise_refusal_leaves_every_candidate_missing() -> None:
    client = ScriptedJev(lambda state, name, question: {"type": "refusal"})

    result = score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(mode="listwise"))

    assert result.scores == dict.fromkeys(IDS, -1.0)
    assert result.n_missing == len(CANDIDATES)


def test_listwise_option_without_a_probability_is_missing() -> None:
    def script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        answer = choice_script(state, name, question)
        del answer["probabilities"]["c2"]
        return answer

    client = ScriptedJev(script)

    result = score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(mode="listwise"))

    assert result.scores[IDS[2]] == -1.0
    assert result.scores[IDS[3]] == RELEVANCE[SUMMARIES[3]]
    assert result.n_missing == 1


def test_no_candidates_means_no_requests() -> None:
    client = ScriptedJev(noul_script)

    for mode in ("pair", "batched", "listwise"):
        result = score_candidates(client, QUERY, [], JevScoringConfig(mode=mode))
        assert result.scores == {}
        assert (result.n_requests, result.n_missing, result.model) == (0, 0, "")
        assert result.usage == Usage()

    assert client.requests == []


def test_duplicate_episode_ids_are_rejected() -> None:
    client = ScriptedJev(noul_script)

    with pytest.raises(ValueError, match="distinct"):
        score_candidates(client, QUERY, [CANDIDATES[0], CANDIDATES[0]], JevScoringConfig())

    assert client.requests == []


# --------------------------------------------------------------------------- #
# Bookkeeping: usage, model, cache, salt                                       #
# --------------------------------------------------------------------------- #
def test_scripted_client_satisfies_the_decision_client_protocol() -> None:
    assert isinstance(ScriptedJev(noul_script), DecisionClient)


@pytest.mark.parametrize(("mode", "n_requests"), [("pair", 5), ("batched", 3), ("listwise", 1)])
def test_usage_is_summed_over_requests(mode: str, n_requests: int) -> None:
    client = ScriptedJev(choice_script if mode == "listwise" else noul_script)
    config = JevScoringConfig(mode=mode, batch_size=2, concurrency=1)

    result = score_candidates(client, QUERY, CANDIDATES, config)

    assert result.n_requests == n_requests == len(client.requests)
    assert result.usage == expected_usage(client)
    assert result.usage.calls == n_requests
    assert result.model == SNAPSHOT
    assert result.n_cached == 0


def test_model_lists_every_snapshot_that_served_the_call() -> None:
    newer = "typesafe/jev-1.14-20261101"
    client = ScriptedJev(noul_script, snapshots=(newer, SNAPSHOT))
    config = JevScoringConfig(mode="pair", concurrency=1)

    result = score_candidates(client, QUERY, CANDIDATES, config)

    assert result.model == f"{SNAPSHOT},{newer}"


def test_cached_responses_are_counted() -> None:
    client = ScriptedJev(noul_script, cached=True)

    result = score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(batch_size=2))

    assert result.n_cached == result.n_requests == 3


@pytest.mark.parametrize("mode", ["pair", "batched", "listwise"])
@pytest.mark.parametrize("salt", ["", "variance-run-2"])
def test_salt_reaches_every_request(mode: str, salt: str) -> None:
    client = ScriptedJev(choice_script if mode == "listwise" else noul_script)
    config = JevScoringConfig(mode=mode, batch_size=2, salt=salt)

    score_candidates(client, QUERY, CANDIDATES, config)

    assert client.requests
    assert all(sent_salt == salt for _, _, sent_salt in client.requests)


# --------------------------------------------------------------------------- #
# Concurrency                                                                  #
# --------------------------------------------------------------------------- #
def test_result_does_not_depend_on_completion_order() -> None:
    """Requests are forced to finish last-to-first; the outcome matches a sequential run."""

    done = [threading.Event() for _ in CANDIDATES]
    finish_order: list[int] = []

    def reversed_script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        index = SUMMARIES.index(state["conversation_summary"])
        if index + 1 < len(done):
            assert done[index + 1].wait(timeout=10), "the next request never finished"
        finish_order.append(index)
        done[index].set()
        return noul_script(state, name, question)

    concurrent = score_candidates(
        ScriptedJev(reversed_script),
        QUERY,
        CANDIDATES,
        JevScoringConfig(mode="pair", concurrency=len(CANDIDATES)),
    )
    sequential = score_candidates(
        ScriptedJev(noul_script), QUERY, CANDIDATES, JevScoringConfig(mode="pair", concurrency=1)
    )

    assert finish_order == [4, 3, 2, 1, 0]
    assert concurrent == sequential  # scores, exact usage sums, counts and model
    assert list(concurrent.scores) == IDS


def test_at_most_concurrency_requests_are_in_flight() -> None:
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        threading.Event().wait(0.002)  # hold the slot long enough for others to pile up
        with lock:
            in_flight -= 1
        return {"type": "noul", "noul": 0.5}

    candidates = [(f"ep-{index}", f"summary {index}") for index in range(24)]
    config = JevScoringConfig(mode="pair", concurrency=3)

    score_candidates(ScriptedJev(script), QUERY, candidates, config)

    assert 1 <= peak <= 3


def test_a_failed_request_propagates_and_cancels_the_queued_ones() -> None:
    def script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        if state["conversation_summary"] == SUMMARIES[0]:
            raise RuntimeError("402: out of credits")
        threading.Event().wait(0.2)  # slow enough that the queue is cancelled before it drains
        return noul_script(state, name, question)

    client = ScriptedJev(script)

    with pytest.raises(RuntimeError, match="out of credits"):
        score_candidates(client, QUERY, CANDIDATES, JevScoringConfig(mode="pair", concurrency=1))

    assert len(client.requests) < len(CANDIDATES)


def test_the_earliest_failed_request_is_the_one_raised() -> None:
    """Request 3 fails first in time, request 1 later: request order decides, not timing."""

    third_failed = threading.Event()

    def script(state: Any, name: str, question: Mapping[str, Any]) -> dict[str, Any]:
        index = SUMMARIES.index(state["conversation_summary"])
        if index == 3:
            third_failed.set()
            raise RuntimeError("request 3 failed")
        if index == 1:
            assert third_failed.wait(timeout=10)
            raise RuntimeError("request 1 failed")
        return noul_script(state, name, question)

    config = JevScoringConfig(mode="pair", concurrency=len(CANDIDATES))

    with pytest.raises(RuntimeError, match="request 1 failed"):
        score_candidates(ScriptedJev(script), QUERY, CANDIDATES, config)
