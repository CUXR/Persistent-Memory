"""Unit tests for the four rankers.

Everything runs offline against small in-test fakes: a dict-backed store that
hands out nothing but summaries, and scripted stand-ins for bge-m3, the
cross-encoder and Jev.
"""

from __future__ import annotations

import inspect
import json
import random
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest

from eval.retrieval import config
from eval.retrieval.interfaces import CorpusStore, DecisionResult, Ranker, Usage
from eval.retrieval.pipelines import (
    BgeJevRanker,
    BgeM3Ranker,
    BgeRerankerRanker,
    JevFullPoolRanker,
    JevScoringConfig,
    base,
    build_ranker,
)
from eval.retrieval.schema import TimeRange

OWNER = config.OWNER_ID
QUERY = "What did Sam say about the climbing trip?"
SNAPSHOT = "typesafe/jev-1.14-20261014"
TWO_STAGE = ("bge_m3_reranker", "bge_m3_jev")
BATCHED_SUMMARY_SEPARATOR = "\n\nconversation_summary: "  # what jev_scoring puts before a batched summary


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class Clock:
    """Stand-in for the ``time`` module whose clock moves only when a fake says so.

    Every operation costs a distinct power of two, so a measured duration says
    exactly which operations ran inside the timed region.
    """

    COST_S = {
        "embed_query": 1.0,
        "vector_search": 2.0,
        "get_summaries": 4.0,
        "score": 8.0,
        "decide": 16.0,
        "pool_ids": 32.0,
    }

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()

    def perf_counter(self) -> float:
        return self._now

    def spend(self, operation: str) -> None:
        with self._lock:
            self._now += self.COST_S[operation]


@dataclass(frozen=True)
class Row:
    episode_id: str
    person_id: str
    start_time: datetime
    summary: str
    cosine: float  # similarity of this summary's embedding to the query embedding


@dataclass(frozen=True)
class Call:
    method: str
    owner_id: str | None = None
    person_id: str | None = None
    time_range: TimeRange | None = None
    limit: int | None = None
    query_embedding: tuple[float, ...] = ()
    episode_ids: tuple[str, ...] = ()


class FakeStore:
    """Dict-backed CorpusStore that records its calls. Rankers may read summaries only."""

    def __init__(self, rows: Sequence[Row], clock: Clock) -> None:
        self.rows = {row.episode_id: row for row in rows}
        self.clock = clock
        self.calls: list[Call] = []

    def _pool(self, person_id: str | None, time_range: TimeRange | None) -> list[Row]:
        rows = [
            row
            for row in self.rows.values()
            if (person_id is None or row.person_id == person_id)
            and (time_range is None or time_range.contains(row.start_time))
        ]
        rows.sort(key=lambda row: row.episode_id)
        rows.sort(key=lambda row: row.start_time, reverse=True)  # newest first, ties by id
        return rows

    def calls_to(self, method: str) -> list[Call]:
        return [call for call in self.calls if call.method == method]

    def list_people(self, owner_id: str) -> list[Any]:
        raise AssertionError("a ranker has no business listing people")

    def pool_ids(
        self,
        owner_id: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[str]:
        self.clock.spend("pool_ids")
        self.calls.append(Call("pool_ids", owner_id, person_id, time_range))
        return [row.episode_id for row in self._pool(person_id, time_range)]

    def vector_search(
        self,
        owner_id: str,
        query_embedding: Sequence[float],
        person_id: str | None = None,
        time_range: TimeRange | None = None,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        self.clock.spend("vector_search")
        self.calls.append(
            Call("vector_search", owner_id, person_id, time_range, limit, tuple(query_embedding))
        )
        # Equal cosines come back newest first: deliberately not in episode-id order.
        hits = sorted(self._pool(person_id, time_range), key=lambda row: row.cosine, reverse=True)
        return [(row.episode_id, row.cosine) for row in hits[:limit]]

    def get_summaries(self, episode_ids: Sequence[str]) -> dict[str, str]:
        self.clock.spend("get_summaries")
        self.calls.append(Call("get_summaries", episode_ids=tuple(episode_ids)))
        return {episode_id: self.rows[episode_id].summary for episode_id in episode_ids}

    def get_headers(self, episode_ids: Sequence[str]) -> dict[str, Any]:
        raise AssertionError("rankers see summaries only, not headers")

    def get_conversations(self, episode_ids: Sequence[str]) -> dict[str, Any]:
        raise AssertionError("rankers see summaries only, not dialogues")

    def corpus_fingerprint(self) -> str | None:
        raise AssertionError("rankers never ask which corpus they rank over")


class FakeEmbedder:
    model_name = "fake-bge-m3"
    QUERY_VECTOR = (0.6, 0.8)

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.clock.spend("embed_query")
        self.queries.append(text)
        return list(self.QUERY_VECTOR)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("a ranker embeds the query, never documents")


class FakeScorer:
    """PairScorer fake: the score of a summary comes from a table."""

    model_name = "fake-bge-reranker"

    def __init__(self, relevance: Mapping[str, Any], clock: Clock) -> None:
        self.relevance = relevance
        self.clock = clock
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.clock.spend("score")
        self.calls.append((query, list(texts)))
        return [self.relevance[text] for text in texts]


class FakeJev:
    """DecisionClient fake: records every request, answers from a table of probabilities.

    A noul question gets its candidate's value (None makes Jev refuse it); a
    choice question gets every option's value as that option's probability.
    """

    model = "typesafe/jev-1.14"

    def __init__(self, relevance: Mapping[str, Any], clock: Clock) -> None:
        self.relevance = relevance
        self.clock = clock
        self.requests: list[tuple[Any, dict[str, Any], str]] = []  # (state, questions, salt)
        self._lock = threading.Lock()

    @staticmethod
    def _summary(state: Any, question: Mapping[str, Any]) -> str:
        if "conversation_summary" in state:  # pair: the summary is in the state
            return state["conversation_summary"]
        # batched: the summary follows the label at the end of the instruction text
        return question["instructions"].rsplit(BATCHED_SUMMARY_SEPARATOR, 1)[1]

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        salt: str = "",
    ) -> DecisionResult:
        self.clock.spend("decide")
        with self._lock:
            self.requests.append((state, dict(questions), salt))
        answers: dict[str, dict[str, Any]] = {}
        for name, question in questions.items():
            if question["type"] == "choice":
                probabilities = {
                    option: self.relevance[summary]
                    for option, summary in question["criteria"].items()
                }
                best = max(probabilities, key=lambda option: probabilities[option])
                answers[name] = {"type": "choice", "choice": best, "probabilities": probabilities}
                continue
            probability = self.relevance[self._summary(state, question)]
            if probability is None:
                answers[name] = {"type": "refusal"}
            else:
                answers[name] = {"type": "noul", "noul": probability}
        usage = Usage(
            input_tokens=100 + 10 * len(questions),
            output_tokens=len(questions),
            cost_usd=0.5,
            latency_s=0.25,
            calls=1,
        )
        return DecisionResult(answers=answers, usage=usage, model=SNAPSHOT)

    def scored_summaries(self) -> list[str]:
        """Every summary Jev was asked about, sorted (requests arrive in no fixed order)."""

        return sorted(
            self._summary(state, question)
            for state, questions, _ in self.requests
            for question in questions.values()
        )


def make_rows(n: int, person_id: str = "sam") -> list[Row]:
    """``n`` conversations with one person, the first the newest.

    Cosines are a fixed shuffle (seeded from config.SEED), so stage-1 order
    differs from pool order.
    """

    cosines = list(range(n))
    random.Random(config.SEED).shuffle(cosines)
    return [
        Row(
            episode_id=f"{person_id}-{index:03d}",
            person_id=person_id,
            start_time=config.NOW - timedelta(hours=index + 1),
            summary=f"summary of {person_id}-{index:03d}",
            cosine=cosines[index] / n,
        )
        for index in range(n)
    ]


def shuffled_relevance(rows: Sequence[Row]) -> dict[str, float]:
    """A second fixed shuffle, so second-stage order differs from stage-1 and pool order."""

    values = list(range(len(rows)))
    random.Random(config.SEED + 1).shuffle(values)
    return {row.summary: value / len(rows) for row, value in zip(rows, values)}


class Bench:
    """One corpus plus every dependency a ranker can ask for."""

    def __init__(self, rows: Sequence[Row], relevance: Mapping[str, Any] | None = None) -> None:
        self.rows = list(rows)
        self.relevance = dict(shuffled_relevance(rows) if relevance is None else relevance)
        self.clock = Clock()
        self.store = FakeStore(self.rows, self.clock)
        self.embedder = FakeEmbedder(self.clock)
        self.scorer = FakeScorer(self.relevance, self.clock)
        self.jev = FakeJev(self.relevance, self.clock)

    def ranker(self, name: str, k: int = 30, scoring: JevScoringConfig | None = None) -> Ranker:
        return build_ranker(
            name,
            store=self.store,
            embedder=self.embedder,
            scorer=self.scorer,
            jev=self.jev,
            jev_scoring=scoring,
            k=k,
        )

    def second_stage_summaries(self, name: str) -> list[str]:
        """The summaries ``name``'s second stage was given, sorted."""

        if name == "bge_m3_reranker":
            return sorted(text for _, texts in self.scorer.calls for text in texts)
        return self.jev.scored_summaries()


def by_cosine(rows: Sequence[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: (-row.cosine, row.episode_id))


def newest_first(rows: Sequence[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: (-row.start_time.timestamp(), row.episode_id))


def expected_ranking(name: str, bench: Bench, k: int) -> list[tuple[str, float]]:
    """What ``name`` should return over the unfiltered corpus, worked out from the rows."""

    if name == "bge_m3":
        return [(row.episode_id, row.cosine) for row in by_cosine(bench.rows)[:k]]
    if name == "jev_full_pool":
        candidates = newest_first(bench.rows)
    else:
        candidates = by_cosine(bench.rows)[:100]
    reranked = sorted(candidates, key=lambda row: bench.relevance[row.summary], reverse=True)
    return [(row.episode_id, bench.relevance[row.summary]) for row in reranked[:k]]


# --------------------------------------------------------------------------- #
# The Ranker contract                                                          #
# --------------------------------------------------------------------------- #
def test_fakes_satisfy_the_store_protocol() -> None:
    assert isinstance(Bench(make_rows(3)).store, CorpusStore)


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_every_ranker_satisfies_the_ranker_protocol(name: str) -> None:
    ranker = Bench(make_rows(5)).ranker(name)

    assert isinstance(ranker, Ranker)
    assert ranker.name == name


def test_rank_has_exactly_the_tickets_signature() -> None:
    for ranker_class in (BgeM3Ranker, BgeRerankerRanker, BgeJevRanker, JevFullPoolRanker):
        parameters = inspect.signature(ranker_class.rank).parameters
        assert list(parameters) == ["self", "owner_id", "query", "person_id", "time_range"]
        assert parameters["person_id"].default is None
        assert parameters["time_range"].default is None


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_rank_returns_the_ranked_list_of_rank_with_trace(name: str) -> None:
    ranker = Bench(make_rows(40) + make_rows(6, "alex")).ranker(name)
    window = TimeRange(config.NOW - timedelta(hours=20), config.NOW)

    assert ranker.rank(OWNER, QUERY) == ranker.rank_with_trace(OWNER, QUERY).ranked
    assert (
        ranker.rank(OWNER, QUERY, "sam", window)
        == ranker.rank_with_trace(OWNER, QUERY, person_id="sam", time_range=window).ranked
    )


@pytest.mark.parametrize("name", config.RANKER_NAMES)
@pytest.mark.parametrize("n_rows", [40, 130])
def test_each_ranker_returns_its_expected_ranking(name: str, n_rows: int) -> None:
    bench = Bench(make_rows(n_rows))

    ranked = bench.ranker(name).rank(OWNER, QUERY)

    assert ranked == expected_ranking(name, bench, 30)


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_results_are_truncated_to_k_and_a_smaller_k_is_a_prefix(name: str) -> None:
    bench = Bench(make_rows(40))

    deepest = bench.ranker(name, k=30).rank(OWNER, QUERY)

    assert len(deepest) == 30
    assert len({episode_id for episode_id, _ in deepest}) == 30
    scores = [score for _, score in deepest]
    assert scores == sorted(scores, reverse=True)
    for k in (1, 5, 10, 20):
        assert bench.ranker(name, k=k).rank(OWNER, QUERY) == deepest[:k]


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_a_pool_smaller_than_k_is_returned_whole(name: str) -> None:
    bench = Bench(make_rows(7))

    ranked = bench.ranker(name, k=30).rank(OWNER, QUERY)

    assert sorted(episode_id for episode_id, _ in ranked) == sorted(bench.store.rows)


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_scores_are_plain_floats_that_serialize(name: str) -> None:
    rows = make_rows(6)
    bench = Bench(rows, {row.summary: index % 2 for index, row in enumerate(rows)})  # int scores

    ranked = bench.ranker(name).rank(OWNER, QUERY)

    assert all(type(score) is float for _, score in ranked)
    assert json.loads(json.dumps(ranked)) == [list(pair) for pair in ranked]


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_rankers_read_candidate_text_only_through_get_summaries(name: str) -> None:
    bench = Bench(make_rows(12))

    result = bench.ranker(name).rank_with_trace(OWNER, QUERY)

    # The store raises on get_headers / get_conversations / list_people, so getting here
    # already rules those out; this pins down what was read instead.
    methods = {call.method for call in bench.store.calls}
    assert methods <= {"pool_ids", "vector_search", "get_summaries"}
    fetches = bench.store.calls_to("get_summaries")
    if name == "bge_m3":
        assert fetches == []  # cosine needs no text at all
    else:
        assert len(fetches) == 1
        assert len(fetches[0].episode_ids) == 12  # the second stage reads every candidate it scores


# --------------------------------------------------------------------------- #
# Stage 1                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", TWO_STAGE)
def test_stage1_is_the_top_100_of_a_larger_pool(name: str) -> None:
    bench = Bench(make_rows(130))
    top_100 = by_cosine(bench.rows)[:100]

    trace = bench.ranker(name).rank_with_trace(OWNER, QUERY).trace

    (search,) = bench.store.calls_to("vector_search")
    assert search.limit == config.STAGE1_TOP_K == 100
    assert search.query_embedding == FakeEmbedder.QUERY_VECTOR
    assert bench.embedder.queries == [QUERY]
    assert trace.stage1_ids == [row.episode_id for row in top_100]
    assert len(trace.stage1_ids) == 100  # what the second stage scored
    assert trace.pool_size == 130
    assert bench.second_stage_summaries(name) == sorted(row.summary for row in top_100)


@pytest.mark.parametrize("name", TWO_STAGE)
def test_stage1_is_the_whole_pool_when_it_is_smaller_than_100(name: str) -> None:
    bench = Bench(make_rows(7))

    trace = bench.ranker(name).rank_with_trace(OWNER, QUERY).trace

    assert trace.stage1_ids == [row.episode_id for row in by_cosine(bench.rows)]
    assert len(trace.stage1_ids) == trace.pool_size == 7  # the whole pool reached the second stage
    assert bench.second_stage_summaries(name) == sorted(row.summary for row in bench.rows)


def test_stage1_is_capped_even_if_the_store_returns_more_than_it_was_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = Bench(make_rows(130))
    search = bench.store.vector_search
    monkeypatch.setattr(
        bench.store,
        "vector_search",
        lambda *args, **kwargs: search(*args, **{**kwargs, "limit": 1000}),
    )

    trace = bench.ranker("bge_m3_reranker").rank_with_trace(OWNER, QUERY).trace

    assert trace.stage1_ids == [row.episode_id for row in by_cosine(bench.rows)[:100]]
    assert len(trace.stage1_ids) == 100


def test_reranker_scores_the_stage1_summaries_in_stage1_order() -> None:
    bench = Bench(make_rows(12))

    bench.ranker("bge_m3_reranker").rank(OWNER, QUERY)

    assert bench.scorer.calls == [(QUERY, [row.summary for row in by_cosine(bench.rows)])]


def test_second_stage_cannot_recover_what_stage1_dropped_but_the_full_pool_ranker_can() -> None:
    rows = make_rows(130)
    dropped = by_cosine(rows)[-1]
    relevance = {row.summary: 0.1 for row in rows} | {dropped.summary: 0.99}
    bench = Bench(rows, relevance)

    for name in TWO_STAGE:
        assert dropped.episode_id not in dict(bench.ranker(name).rank(OWNER, QUERY))
    assert bench.ranker("jev_full_pool").rank(OWNER, QUERY)[0] == (dropped.episode_id, 0.99)


@pytest.mark.parametrize("ranker_class", [BgeM3Ranker, BgeRerankerRanker, BgeJevRanker])
def test_stage1_k_is_configurable(ranker_class: type) -> None:
    bench = Bench(make_rows(40))
    extra_arguments = {BgeRerankerRanker: [bench.scorer], BgeJevRanker: [bench.jev]}
    second_stage = extra_arguments.get(ranker_class, [])

    ranker = ranker_class(bench.store, bench.embedder, *second_stage, k=30, stage1_k=10)
    result = ranker.rank_with_trace(OWNER, QUERY)

    assert bench.store.calls_to("vector_search")[0].limit == 10
    assert len(result.ranked) == 10
    assert {episode_id for episode_id, _ in result.ranked} == {
        row.episode_id for row in by_cosine(bench.rows)[:10]
    }


# --------------------------------------------------------------------------- #
# Filters                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_person_and_time_filters_reach_the_store_unchanged(name: str) -> None:
    bench = Bench(make_rows(12, "sam") + make_rows(12, "alex"))
    window = TimeRange(
        config.NOW - timedelta(hours=6, minutes=30),
        config.NOW - timedelta(hours=1, minutes=30),
    )
    in_window = {f"sam-{index:03d}" for index in range(1, 6)}  # 2 to 6 hours old

    result = bench.ranker(name).rank_with_trace(OWNER, QUERY, person_id="sam", time_range=window)

    filtered_calls = bench.store.calls_to("pool_ids") + bench.store.calls_to("vector_search")
    assert filtered_calls
    for call in filtered_calls:
        assert call.owner_id == OWNER
        assert call.person_id == "sam"
        assert call.time_range is window  # the padded range, exactly as it was handed in
    assert {episode_id for episode_id, _ in result.ranked} == in_window
    assert result.trace.pool_size == len(in_window)


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_no_filters_means_the_owners_whole_corpus(name: str) -> None:
    bench = Bench(make_rows(4, "sam") + make_rows(3, "alex"))

    result = bench.ranker(name).rank_with_trace(OWNER, QUERY)

    for call in bench.store.calls_to("pool_ids") + bench.store.calls_to("vector_search"):
        assert (call.owner_id, call.person_id, call.time_range) == (OWNER, None, None)
    assert len(result.ranked) == result.trace.pool_size == 7


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_an_empty_pool_ranks_nothing_and_scores_nothing(name: str) -> None:
    bench = Bench(make_rows(5))

    result = bench.ranker(name).rank_with_trace(OWNER, QUERY, person_id="nobody")

    assert result.ranked == []
    assert result.trace.pool_size == 0
    assert not result.trace.stage1_ids  # nothing reached a second stage
    assert result.trace.usage == Usage()
    assert bench.scorer.calls == []
    assert bench.jev.requests == []
    assert bench.store.calls_to("get_summaries") == []


# --------------------------------------------------------------------------- #
# Tie-breaks                                                                   #
# --------------------------------------------------------------------------- #
def tied_rows() -> list[Row]:
    """Four conversations whose id order, age order and cosine order all differ.

    Pool order (newest first) is d, a, c, b. ``a`` and ``d`` share a cosine.
    """

    def row(episode_id: str, hours_old: int, cosine: float) -> Row:
        start = config.NOW - timedelta(hours=hours_old)
        return Row(episode_id, "sam", start, f"summary of {episode_id}", cosine)

    return [row("a", 2, 0.5), row("b", 4, 0.9), row("c", 3, 0.7), row("d", 1, 0.5)]


def test_bge_m3_breaks_cosine_ties_by_episode_id() -> None:
    bench = Bench(tied_rows())

    # The store itself returns the tied pair newest first (d before a).
    store_order = [episode_id for episode_id, _ in bench.store.vector_search(OWNER, [0.0])]
    assert store_order == ["b", "c", "d", "a"]
    ranked = bench.ranker("bge_m3").rank(OWNER, QUERY)
    assert ranked == [("b", 0.9), ("c", 0.7), ("a", 0.5), ("d", 0.5)]


@pytest.mark.parametrize("name", TWO_STAGE)
def test_second_stage_ties_keep_stage1_order(name: str) -> None:
    rows = tied_rows()
    relevance = {row.summary: 0.3 for row in rows} | {"summary of c": 0.8}
    bench = Bench(rows, relevance)

    result = bench.ranker(name).rank_with_trace(OWNER, QUERY)

    assert result.trace.stage1_ids == ["b", "c", "a", "d"]
    assert result.ranked == [("c", 0.8), ("b", 0.3), ("a", 0.3), ("d", 0.3)]


def test_jev_full_pool_ties_keep_pool_order_newest_first() -> None:
    rows = tied_rows()
    relevance = {row.summary: 0.3 for row in rows} | {"summary of c": 0.8}
    bench = Bench(rows, relevance)

    ranked = bench.ranker("jev_full_pool").rank(OWNER, QUERY)

    assert ranked == [("c", 0.8), ("d", 0.3), ("a", 0.3), ("b", 0.3)]


@pytest.mark.parametrize(
    ("name", "order_without_answers"),
    [("bge_m3_jev", "bcad"), ("jev_full_pool", "dacb")],
)
def test_candidates_jev_gave_no_answer_for_rank_last(name: str, order_without_answers: str) -> None:
    rows = tied_rows()
    answered = {row.summary: None for row in rows} | {"summary of a": 0.2}
    bench = Bench(rows, answered)

    result = bench.ranker(name).rank_with_trace(OWNER, QUERY)

    unanswered = [episode_id for episode_id in order_without_answers if episode_id != "a"]
    assert result.ranked == [("a", 0.2)] + [(episode_id, -1.0) for episode_id in unanswered]
    assert result.trace.extra["n_missing"] == 3


# --------------------------------------------------------------------------- #
# Ranker 4                                                                     #
# --------------------------------------------------------------------------- #
def test_jev_full_pool_takes_no_embedder_and_scores_every_conversation() -> None:
    bench = Bench(make_rows(130))
    assert "embedder" not in inspect.signature(JevFullPoolRanker).parameters

    ranker = build_ranker("jev_full_pool", store=bench.store, jev=bench.jev)
    trace = ranker.rank_with_trace(OWNER, QUERY).trace

    assert bench.store.calls_to("vector_search") == []
    assert bench.embedder.queries == []
    assert trace.stage1_ids is None
    assert trace.pool_size == 130
    assert bench.jev.scored_summaries() == sorted(row.summary for row in bench.rows)  # the whole pool
    (fetch,) = bench.store.calls_to("get_summaries")
    assert list(fetch.episode_ids) == [row.episode_id for row in newest_first(bench.rows)]


# --------------------------------------------------------------------------- #
# Trace                                                                        #
# --------------------------------------------------------------------------- #
def test_trace_of_the_local_rankers() -> None:
    bench = Bench(make_rows(40))

    first = bench.ranker("bge_m3").rank_with_trace(OWNER, QUERY).trace
    second = bench.ranker("bge_m3_reranker").rank_with_trace(OWNER, QUERY).trace

    assert (first.ranker, first.pool_size) == ("bge_m3", 40)
    assert first.stage1_ids is None  # one stage only
    assert set(first.stage_latency_s) == {"embed_query", "vector_search"}
    assert second.ranker == "bge_m3_reranker"
    assert second.stage1_ids == [row.episode_id for row in by_cosine(bench.rows)]
    assert (second.pool_size, len(second.stage1_ids)) == (40, 40)
    assert set(second.stage_latency_s) == {
        "embed_query",
        "vector_search",
        "fetch_summaries",
        "rerank",
    }
    for trace in (first, second):
        assert trace.usage == Usage()  # local models cost nothing
        assert trace.extra == {}
        assert trace.latency_s >= 0


@pytest.mark.parametrize(
    ("name", "stages"),
    [
        ("bge_m3_jev", {"embed_query", "vector_search", "fetch_summaries", "rerank"}),
        ("jev_full_pool", {"fetch_pool", "fetch_summaries", "rerank"}),
    ],
)
def test_trace_of_the_jev_rankers_sums_usage_over_requests(name: str, stages: set[str]) -> None:
    bench = Bench(make_rows(40))

    trace = bench.ranker(name).rank_with_trace(OWNER, QUERY).trace

    # 40 candidates in batches of 16: requests of 16, 16 and 8 questions.
    assert len(bench.jev.requests) == 3
    assert trace.usage == Usage(
        input_tokens=700, output_tokens=40, cost_usd=1.5, latency_s=0.75, calls=3
    )
    assert trace.extra == {
        "n_requests": 3,
        "n_missing": 0,
        "n_cached": 0,
        "jev_model": SNAPSHOT,
        "scoring": "batched-noul-v1-b16",
    }
    assert (trace.ranker, trace.pool_size) == (name, 40)
    assert len(bench.jev.scored_summaries()) == 40  # Jev scored the whole pool
    assert set(trace.stage_latency_s) == stages


@pytest.mark.parametrize("name", config.JEV_RANKERS)
def test_jev_rankers_apply_the_scoring_config_they_were_given(name: str) -> None:
    bench = Bench(make_rows(12))
    scoring = JevScoringConfig(mode="pair", wording="v2", salt="variance-run-2")

    trace = bench.ranker(name, scoring=scoring).rank_with_trace(OWNER, QUERY).trace

    assert len(bench.jev.requests) == 12  # pair mode: one request per candidate
    assert all(salt == "variance-run-2" for _, _, salt in bench.jev.requests)
    assert all(state["query"] == QUERY for state, _, _ in bench.jev.requests)
    assert trace.extra["scoring"] == scoring.label == "pair-noul-v2"
    assert trace.extra["n_requests"] == trace.usage.calls == 12


def test_listwise_scoring_reranks_stage1_but_cannot_take_a_pool_over_255() -> None:
    bench = Bench(make_rows(256))
    listwise = JevScoringConfig(mode="listwise")

    result = bench.ranker("bge_m3_jev", scoring=listwise).rank_with_trace(OWNER, QUERY)

    assert result.ranked == expected_ranking("bge_m3_jev", bench, 30)
    assert result.trace.extra["n_requests"] == 1
    assert result.trace.extra["scoring"] == "listwise-v1"
    with pytest.raises(ValueError, match="at most 255 candidates"):
        bench.ranker("jev_full_pool", scoring=listwise).rank(OWNER, QUERY)
    assert len(bench.jev.requests) == 1  # the full-pool call never reached the API


def test_latency_covers_the_algorithm_and_leaves_out_the_pool_size_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = Bench(make_rows(12))
    monkeypatch.setattr(base, "time", bench.clock)
    stage1 = {"embed_query": 1.0, "vector_search": 2.0}
    expected = {
        "bge_m3": stage1,
        "bge_m3_reranker": {**stage1, "fetch_summaries": 4.0, "rerank": 8.0},
        "bge_m3_jev": {**stage1, "fetch_summaries": 4.0, "rerank": 16.0},
        # Reading the pool (32 s on this clock) is this ranker's first step, so it is timed.
        "jev_full_pool": {"fetch_pool": 32.0, "fetch_summaries": 4.0, "rerank": 16.0},
    }

    for name, stages in expected.items():
        trace = bench.ranker(name).rank_with_trace(OWNER, QUERY).trace

        assert trace.stage_latency_s == stages
        assert trace.latency_s == sum(stages.values())
        assert trace.pool_size == 12  # still reported by rankers 1-3, just not timed


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #
def test_build_ranker_builds_each_configured_ranker() -> None:
    bench = Bench(make_rows(3))
    classes = {
        "bge_m3": BgeM3Ranker,
        "bge_m3_reranker": BgeRerankerRanker,
        "bge_m3_jev": BgeJevRanker,
        "jev_full_pool": JevFullPoolRanker,
    }

    assert set(classes) == set(config.RANKER_NAMES)
    for name, ranker_class in classes.items():
        ranker = bench.ranker(name, k=7)
        assert type(ranker) is ranker_class
        assert ranker.k == 7
    assert bench.ranker("bge_m3").k == 30  # the deepest cut-off the report needs
    assert bench.ranker("bge_m3_reranker").stage1_k == config.STAGE1_TOP_K


def test_constructor_defaults_match_the_ticket() -> None:
    bench = Bench(make_rows(3))
    rankers: list[Any] = [
        BgeM3Ranker(bench.store, bench.embedder),
        BgeRerankerRanker(bench.store, bench.embedder, bench.scorer),
        BgeJevRanker(bench.store, bench.embedder, bench.jev),
        JevFullPoolRanker(bench.store, bench.jev),
    ]

    assert [ranker.name for ranker in rankers] == list(config.RANKER_NAMES)
    assert all(ranker.k == 30 for ranker in rankers)
    assert all(ranker.stage1_k == 100 for ranker in rankers[:3])
    assert all(ranker.scoring == JevScoringConfig() for ranker in rankers[2:])


def test_build_ranker_passes_the_jev_scoring_config_through() -> None:
    bench = Bench(make_rows(3))
    scoring = JevScoringConfig(mode="listwise")

    for name in config.JEV_RANKERS:
        assert bench.ranker(name, scoring=scoring).scoring is scoring
        assert bench.ranker(name).scoring == JevScoringConfig()


@pytest.mark.parametrize(
    ("name", "given", "missing"),
    [
        ("bge_m3", {}, "embedder"),
        ("bge_m3_reranker", {"scorer": object()}, "embedder"),
        ("bge_m3_reranker", {"embedder": object()}, "scorer"),
        ("bge_m3_jev", {"jev": object()}, "embedder"),
        ("bge_m3_jev", {"embedder": object()}, "jev"),
        ("jev_full_pool", {"embedder": object(), "scorer": object()}, "jev"),
    ],
)
def test_build_ranker_names_the_dependency_that_is_missing(
    name: str, given: dict[str, Any], missing: str
) -> None:
    store = Bench(make_rows(3)).store

    with pytest.raises(ValueError, match=rf"'{name}' needs {missing}="):
        build_ranker(name, store=store, **given)


def test_build_ranker_rejects_an_unknown_name() -> None:
    bench = Bench(make_rows(3))

    with pytest.raises(ValueError, match="unknown ranker 'bm25'"):
        bench.ranker("bm25")


@pytest.mark.parametrize("name", config.RANKER_NAMES)
def test_k_must_be_positive(name: str) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        Bench(make_rows(3)).ranker(name, k=0)


def test_stage1_k_must_be_positive() -> None:
    bench = Bench(make_rows(3))

    with pytest.raises(ValueError, match="stage1_k must be at least 1"):
        BgeM3Ranker(bench.store, bench.embedder, stage1_k=0)


# --------------------------------------------------------------------------- #
# Broken dependencies fail loudly                                              #
# --------------------------------------------------------------------------- #
def test_a_scorer_that_returns_the_wrong_number_of_scores_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = Bench(make_rows(5))
    monkeypatch.setattr(bench.scorer, "score", lambda query, texts: [0.5] * (len(texts) - 1))

    with pytest.raises(ValueError, match="4 scores for 5 candidates"):
        bench.ranker("bge_m3_reranker").rank(OWNER, QUERY)


@pytest.mark.parametrize("name", ["bge_m3_reranker", "bge_m3_jev", "jev_full_pool"])
def test_a_candidate_without_a_summary_is_an_error(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = Bench(make_rows(5))
    monkeypatch.setattr(bench.store, "get_summaries", lambda episode_ids: {})

    with pytest.raises(LookupError, match="no summary for 5 of 5 candidates"):
        bench.ranker(name).rank(OWNER, QUERY)
