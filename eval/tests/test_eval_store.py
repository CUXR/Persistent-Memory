"""Unit tests for the corpus stores.

The in-memory store is exercised end to end on a small corpus built here. The
Postgres store never sees a database: it gets a fake connection that records
each statement and replays scripted rows, which pins down the SQL it issues.
"""

from __future__ import annotations

import math
import random
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest

from eval.retrieval import config, dataset, db
from eval.retrieval import store as store_module
from eval.retrieval.interfaces import CorpusStore
from eval.retrieval.pipelines import BgeM3Ranker, BgeRerankerRanker
from eval.retrieval.schema import (
    OWNER_SPEAKER,
    Conversation,
    ConversationHeader,
    Person,
    TimeRange,
    Turn,
)
from eval.retrieval.store import InMemoryCorpusStore, PgCorpusStore

OWNER = "owner-riley"
SAM, ANA, BO = "person-sam", "person-ana", "person-bo"
PEOPLE = [
    Person(person_id=SAM, first_name="Sam", last_name="Carter", display_name="Sam Carter"),
    Person(person_id=BO, first_name="Bo", last_name="Chen", display_name=""),
    Person(person_id=ANA, first_name="Ana", last_name="Lopez", display_name="Ana Lopez"),
]
NOON = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def _conversation(episode_id: str, person_id: str, start_time: datetime) -> Conversation:
    return Conversation(
        episode_id=episode_id,
        person_id=person_id,
        start_time=start_time,
        end_time=start_time + timedelta(minutes=20),
        dialogue=[Turn("Sam", f"about {episode_id}"), Turn(OWNER_SPEAKER, "noted")],
        summary=f"summary of {episode_id}",
        meta={"topic": "climbing"},
    )


# Deliberately not in pool order; e3 and e2 share a start time.
CONVERSATIONS = [
    _conversation("e3", SAM, NOON),
    _conversation("e1", SAM, NOON - timedelta(days=2)),
    _conversation("e5", ANA, NOON + timedelta(days=1)),
    _conversation("e2", ANA, NOON),
    _conversation("e4", BO, NOON - timedelta(days=9)),
]


def _store(embeddings: Mapping[str, Sequence[float]] | None = None) -> InMemoryCorpusStore:
    return InMemoryCorpusStore(PEOPLE, CONVERSATIONS, embeddings, owner_id=OWNER)


# --------------------------------------------------------------------------- #
# In-memory store: people and pools                                            #
# --------------------------------------------------------------------------- #
def test_both_stores_satisfy_the_corpus_store_protocol() -> None:
    assert isinstance(_store(), CorpusStore)
    assert isinstance(PgCorpusStore(connection_factory=FakeDatabase()), CorpusStore)


def test_each_store_says_which_corpus_it_serves() -> None:
    # The memory store computes the fingerprint of what it holds, whatever the input order.
    expected = db.dataset_fingerprint(PEOPLE, CONVERSATIONS)
    assert _store().corpus_fingerprint() == expected
    assert InMemoryCorpusStore(PEOPLE[::-1], CONVERSATIONS[::-1], owner_id=OWNER).corpus_fingerprint() == expected
    assert _store().corpus_fingerprint() != db.dataset_fingerprint(PEOPLE, CONVERSATIONS[:-1])

    # The pg store reads what load_db recorded for the owner.
    store, connection = _pg_store([("abc123",)])
    assert store.corpus_fingerprint(OWNER) == "abc123"
    assert connection.statements == [(" ".join(db.SELECT_CORPUS_META.split()), {"owner_id": OWNER})]
    _assert_parameters_match_placeholders(connection)
    store, _ = _pg_store([])
    assert store.corpus_fingerprint(OWNER) is None  # nothing recorded for the owner
    # A database loaded before the table existed recorded nothing either.
    missing = FakeConnection(error=psycopg.errors.UndefinedTable("eval_corpus_meta does not exist"))
    assert PgCorpusStore(connection_factory=FakeDatabase(missing)).corpus_fingerprint(OWNER) is None


def test_list_people_is_ordered_by_name_then_id() -> None:
    twin = replace(PEOPLE[0], person_id="person-a-twin")  # a second "Sam Carter"
    store = InMemoryCorpusStore([*PEOPLE, twin], CONVERSATIONS, owner_id=OWNER)

    people = store.list_people(OWNER)

    # Bo has no display name and sorts as "Bo Chen".
    assert [person.person_id for person in people] == [ANA, BO, "person-a-twin", SAM]
    assert people[0] is PEOPLE[2]  # the dataset records, as given


def test_pool_is_newest_first_with_ties_in_episode_id_order() -> None:
    assert _store().pool_ids(OWNER) == ["e5", "e2", "e3", "e1", "e4"]


def test_pool_filters_by_person() -> None:
    store = _store()

    assert store.pool_ids(OWNER, person_id=SAM) == ["e3", "e1"]
    assert store.pool_ids(OWNER, person_id=ANA) == ["e5", "e2"]
    assert store.pool_ids(OWNER, person_id="person-nobody") == []


def test_pool_time_range_includes_both_bounds() -> None:
    store = _store()
    exact = TimeRange(NOON - timedelta(days=2), NOON)
    microsecond = timedelta(microseconds=1)
    inside = TimeRange(exact.start + microsecond, exact.end - microsecond)

    assert store.pool_ids(OWNER, time_range=exact) == ["e2", "e3", "e1"]
    assert store.pool_ids(OWNER, time_range=inside) == []
    assert store.pool_ids(OWNER, person_id=SAM, time_range=TimeRange(NOON, NOON)) == ["e3"]


def test_another_owner_sees_nothing() -> None:
    store = _store({conversation.episode_id: [1.0, 0.0] for conversation in CONVERSATIONS})

    assert store.list_people("owner-else") == []
    assert store.pool_ids("owner-else") == []
    assert store.vector_search("owner-else", [1.0, 0.0]) == []
    assert len(store.list_people(OWNER)) == 3
    assert len(store.pool_ids(OWNER)) == len(store.vector_search(OWNER, [1.0, 0.0])) == 5


def test_default_owner_is_the_benchmark_owner() -> None:
    store = InMemoryCorpusStore(PEOPLE, CONVERSATIONS)

    assert len(store.pool_ids(config.OWNER_ID)) == len(CONVERSATIONS)
    assert store.pool_ids(OWNER) == []


@pytest.mark.parametrize(
    ("conversations", "message"),
    [
        ([CONVERSATIONS[0], CONVERSATIONS[0]], "duplicate episode id: e3"),
        ([_conversation("e9", "person-ghost", NOON)], "unknown person: person-ghost"),
    ],
)
def test_inconsistent_records_are_rejected(conversations: list[Conversation], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InMemoryCorpusStore(PEOPLE, conversations, owner_id=OWNER)


# --------------------------------------------------------------------------- #
# In-memory store: vector search                                               #
# --------------------------------------------------------------------------- #
def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def _brute_force(
    conversations: Sequence[Conversation],
    embeddings: Mapping[str, Sequence[float]],
    query: Sequence[float],
    person_id: str | None = None,
    time_range: TimeRange | None = None,
) -> list[tuple[str, float]]:
    """Reference ranking in plain Python: filter, score every pair, sort."""

    scored = [
        (conversation.episode_id, _cosine(embeddings[conversation.episode_id], query))
        for conversation in conversations
        if (person_id is None or conversation.person_id == person_id)
        and (time_range is None or time_range.start <= conversation.start_time <= time_range.end)
    ]
    return sorted(scored, key=lambda hit: (-hit[1], hit[0]))


def _random_corpus() -> tuple[list[Conversation], dict[str, list[float]], random.Random]:
    """60 conversations over three people with unnormalized 8-d embeddings."""

    rng = random.Random(config.SEED)
    conversations = [
        _conversation(
            f"c{index:02d}",
            rng.choice([SAM, ANA, BO]),
            NOON - timedelta(hours=rng.randrange(600)),
        )
        for index in range(60)
    ]
    embeddings = {
        conversation.episode_id: [rng.gauss(0.0, 1.0) * rng.uniform(0.5, 4.0) for _ in range(8)]
        for conversation in conversations
    }
    return conversations, embeddings, rng


@pytest.mark.parametrize(
    ("person_id", "time_range"),
    [
        (None, None),
        (SAM, None),
        (None, TimeRange(NOON - timedelta(hours=300), NOON - timedelta(hours=100))),
        (ANA, TimeRange(NOON - timedelta(hours=400), NOON)),
    ],
)
def test_vector_search_matches_a_brute_force_reference(
    person_id: str | None,
    time_range: TimeRange | None,
) -> None:
    conversations, embeddings, rng = _random_corpus()
    store = InMemoryCorpusStore(PEOPLE, conversations, embeddings, owner_id=OWNER)
    query = [rng.gauss(0.0, 3.0) for _ in range(8)]
    expected = _brute_force(conversations, embeddings, query, person_id, time_range)
    assert len(expected) >= 5  # the filters leave a pool worth ranking

    hits = store.vector_search(OWNER, query, person_id=person_id, time_range=time_range)

    assert [episode_id for episode_id, _ in hits] == [episode_id for episode_id, _ in expected]
    scores = [score for _, score in hits]
    assert scores == pytest.approx([score for _, score in expected], abs=1e-12)
    assert all(type(score) is float and abs(score) <= 1.0 + 1e-12 for score in scores)
    assert {episode_id for episode_id, _ in hits} == set(
        store.pool_ids(OWNER, person_id=person_id, time_range=time_range)
    )


def test_vector_search_cuts_at_limit_without_changing_the_order() -> None:
    conversations, embeddings, rng = _random_corpus()
    store = InMemoryCorpusStore(PEOPLE, conversations, embeddings, owner_id=OWNER)
    query = [rng.gauss(0.0, 1.0) for _ in range(8)]

    everything = store.vector_search(OWNER, query)

    assert len(everything) == 60  # the default limit of 100 returns the whole pool
    assert store.vector_search(OWNER, query, limit=7) == everything[:7]
    assert store.vector_search(OWNER, query, limit=1) == everything[:1]
    with pytest.raises(ValueError, match="limit must be at least 1"):
        store.vector_search(OWNER, query, limit=0)


def test_vector_search_scores_do_not_depend_on_the_filters() -> None:
    conversations, embeddings, rng = _random_corpus()
    store = InMemoryCorpusStore(PEOPLE, conversations, embeddings, owner_id=OWNER)
    query = [rng.gauss(0.0, 1.0) for _ in range(8)]

    unfiltered = dict(store.vector_search(OWNER, query))
    filtered = store.vector_search(OWNER, query, person_id=BO)

    assert filtered and all(unfiltered[episode_id] == score for episode_id, score in filtered)


def test_equal_cosines_are_ordered_by_episode_id() -> None:
    same = [3.0, 4.0]
    embeddings = {"e1": [0.0, 1.0], "e2": same, "e3": [1.0, 0.0], "e4": same, "e5": same}

    hits = _store(embeddings).vector_search(OWNER, [6.0, 8.0])

    assert [episode_id for episode_id, _ in hits] == ["e2", "e4", "e5", "e1", "e3"]
    assert [score for _, score in hits] == pytest.approx([1.0, 1.0, 1.0, 0.8, 0.6])


def test_identical_embeddings_score_exactly_alike_wherever_they_sit() -> None:
    conversations, embeddings, rng = _random_corpus()
    twins = ["c58", "c07", "c31"]
    for episode_id in twins:
        embeddings[episode_id] = list(embeddings["c58"])
    store = InMemoryCorpusStore(PEOPLE, conversations, embeddings, owner_id=OWNER)

    hits = store.vector_search(OWNER, [rng.gauss(0.0, 1.0) for _ in range(8)])

    scores = dict(hits)
    assert scores["c07"] == scores["c31"] == scores["c58"]
    ranked = [episode_id for episode_id, _ in hits]
    start = ranked.index("c07")
    assert ranked[start : start + 3] == ["c07", "c31", "c58"]


def test_embeddings_and_query_are_normalized_defensively() -> None:
    embeddings = {
        "e1": [0.0, 0.0],
        "e2": [0.0, 250.0],
        "e3": [-0.001, 0.0],
        "e4": [5.0, 5.0],
        "e5": [2.0, 0.0],
    }
    store = _store(embeddings)

    hits = store.vector_search(OWNER, [1000.0, 0.0])

    assert [episode_id for episode_id, _ in hits] == ["e5", "e4", "e1", "e2", "e3"]
    assert [score for _, score in hits] == pytest.approx([1.0, math.sqrt(0.5), 0.0, 0.0, -1.0])
    # A zero vector has no direction: every cosine is 0 (never NaN) and the ids decide the order.
    assert store.vector_search(OWNER, [0.0, 0.0]) == [(f"e{index}", 0.0) for index in range(1, 6)]


def test_the_store_serves_the_rankers_end_to_end() -> None:
    conversations, embeddings, rng = _random_corpus()
    store = InMemoryCorpusStore(PEOPLE, conversations, embeddings, owner_id=OWNER)
    query = [rng.gauss(0.0, 1.0) for _ in range(8)]
    embedder = SimpleNamespace(model_name="fake/bi-encoder", embed_query=lambda text: query)
    by_id = SimpleNamespace(  # a cross-encoder that prefers the highest episode id
        model_name="fake/cross-encoder",
        score=lambda text, summaries: [float(summary[-2:]) for summary in summaries],
    )
    nearest = [episode_id for episode_id, _ in _brute_force(conversations, embeddings, query, SAM)]

    first = BgeM3Ranker(store, embedder, k=5).rank_with_trace(OWNER, "the trip?", person_id=SAM)
    second = BgeRerankerRanker(store, embedder, by_id, k=5, stage1_k=8).rank(
        OWNER,
        "the trip?",
        person_id=SAM,
    )

    assert [episode_id for episode_id, _ in first.ranked] == nearest[:5]
    assert first.trace.pool_size == len(nearest) > 8
    assert [episode_id for episode_id, _ in second] == sorted(nearest[:8], reverse=True)[:5]


def test_vector_search_without_embeddings_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="no embeddings"):
        _store().vector_search(OWNER, [1.0, 0.0])


def test_embedding_problems_are_reported() -> None:
    complete = {conversation.episode_id: [1.0, 0.0] for conversation in CONVERSATIONS}

    with pytest.raises(ValueError, match=r"no embedding for 2 of 5 conversations \(first: e3\)"):
        _store({"e1": [1.0, 0.0], "e5": [1.0, 0.0], "e2": [0.0, 1.0]})
    with pytest.raises(ValueError, match=r"mixed dimensions: \[2, 3\]"):
        _store({**complete, "e4": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="query embedding has shape"):
        _store(complete).vector_search(OWNER, [1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# In-memory store: record content                                              #
# --------------------------------------------------------------------------- #
def test_summaries_headers_and_conversations_skip_unknown_ids() -> None:
    store = _store()

    assert store.get_summaries(["e2", "missing", "e1"]) == {
        "e2": "summary of e2",
        "e1": "summary of e1",
    }
    assert list(store.get_headers(["missing", "e4"])) == ["e4"]
    assert list(store.get_conversations(["e4", "missing"])) == ["e4"]
    assert store.get_summaries([]) == store.get_headers([]) == store.get_conversations([]) == {}


def test_headers_carry_the_person_name_and_no_dialogue() -> None:
    headers = _store().get_headers(["e1", "e4"])

    assert headers["e1"] == ConversationHeader(
        episode_id="e1",
        person_id=SAM,
        person_name="Sam Carter",
        start_time=NOON - timedelta(days=2),
        summary="summary of e1",
    )
    assert headers["e4"].person_name == "Bo Chen"  # no display name: first and last name


def test_conversations_are_copies_with_the_person_name_added() -> None:
    store = _store()
    original = CONVERSATIONS[1]

    fetched = store.get_conversations(["e1"])["e1"]

    assert fetched.meta == {"topic": "climbing", "person_name": "Sam Carter"}
    assert (fetched.episode_id, fetched.person_id, fetched.summary) == ("e1", SAM, "summary of e1")
    assert (fetched.start_time, fetched.end_time) == (original.start_time, original.end_time)
    assert fetched.dialogue == original.dialogue
    assert fetched.dialogue_text() == 'Sam: "about e1"\nMe: "noted"'
    # Neither the added name nor a caller's edits leak into the dataset records.
    fetched.dialogue.clear()
    fetched.meta["scratch"] = True
    assert original.meta == {"topic": "climbing"}
    assert len(original.dialogue) == 2
    again = store.get_conversations(["e1"])["e1"]
    assert again.meta == {"topic": "climbing", "person_name": "Sam Carter"}
    assert again.dialogue == original.dialogue


def test_from_dataset_reads_the_committed_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "PEOPLE_PATH", tmp_path / "people.jsonl")
    monkeypatch.setattr(config, "CONVERSATIONS_PATH", tmp_path / "conversations.jsonl")
    dataset.save_people(PEOPLE)
    dataset.save_conversations(CONVERSATIONS)
    embedder = SimpleNamespace(model_name="fake-model")
    seen: list[tuple[list[str], Any]] = []

    def fake_embeddings(
        conversations: Sequence[Conversation],
        used_embedder: Any,
    ) -> dict[str, list[float]]:
        """Vectors that turn away from [1, 0] in file order, so a search returns that order."""

        seen.append(([record.episode_id for record in conversations], used_embedder))
        return {
            record.episode_id: [1.0, float(index)] for index, record in enumerate(conversations)
        }

    monkeypatch.setattr(store_module, "load_or_compute_corpus_embeddings", fake_embeddings)

    plain = InMemoryCorpusStore.from_dataset()
    searchable = InMemoryCorpusStore.from_dataset(embedder=embedder)

    file_order = ["e3", "e1", "e5", "e2", "e4"]
    assert seen == [(file_order, embedder)]  # only the second store embeds
    assert plain.pool_ids(config.OWNER_ID) == ["e5", "e2", "e3", "e1", "e4"]
    assert plain.get_headers(["e5"])["e5"].person_name == "Ana Lopez"
    with pytest.raises(RuntimeError, match="no embeddings"):
        plain.vector_search(config.OWNER_ID, [1.0, 0.0])
    hits = searchable.vector_search(config.OWNER_ID, [1.0, 0.0])
    assert [episode_id for episode_id, _ in hits] == file_order


# --------------------------------------------------------------------------- #
# Postgres store, against a fake connection                                    #
# --------------------------------------------------------------------------- #
PG_OWNER = "11111111-1111-4111-8111-111111111111"
PG_PERSON = "22222222-2222-4222-8222-222222222222"
EPISODE_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
EPISODE_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
RANGE = TimeRange(NOON - timedelta(days=8), NOON - timedelta(days=1))
OWNER_FILTER = "e.user_id = %(owner_id)s::uuid"
PERSON_FILTER = "e.person_id = %(person_id)s::uuid"
TIME_FILTER = "e.start_time >= %(start)s AND e.start_time <= %(end)s"
FILTER_CASES = [
    pytest.param(None, None, OWNER_FILTER, {}, id="owner"),
    pytest.param(
        PG_PERSON,
        None,
        f"{OWNER_FILTER} AND {PERSON_FILTER}",
        {"person_id": PG_PERSON},
        id="person",
    ),
    pytest.param(
        None,
        RANGE,
        f"{OWNER_FILTER} AND {TIME_FILTER}",
        {"start": RANGE.start, "end": RANGE.end},
        id="time",
    ),
    pytest.param(
        PG_PERSON,
        RANGE,
        f"{OWNER_FILTER} AND {PERSON_FILTER} AND {TIME_FILTER}",
        {"person_id": PG_PERSON, "start": RANGE.start, "end": RANGE.end},
        id="person-and-time",
    ),
]


class FakeCursor:
    def __init__(self, rows: Sequence[tuple[Any, ...]]) -> None:
        self._rows = list(rows)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class FakeConnection:
    """Records each statement (whitespace collapsed) and replays one scripted result for each."""

    def __init__(self, *results: Sequence[tuple[Any, ...]], error: Exception | None = None) -> None:
        self.results = list(results)
        self.error = error
        self.statements: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def execute(self, sql: str, params: Mapping[str, Any]) -> FakeCursor:
        self.statements.append((" ".join(sql.split()), dict(params)))
        if self.error is not None:
            raise self.error
        return FakeCursor(self.results.pop(0))

    def close(self) -> None:
        self.closed = True


class FakeDatabase:
    """Connection factory: hands out the given fake connections, one per connect."""

    def __init__(self, *connections: FakeConnection) -> None:
        self.connections = list(connections)
        self.connects = 0

    def __call__(self) -> FakeConnection:
        self.connects += 1
        return self.connections[self.connects - 1]


def _pg_store(*results: Sequence[tuple[Any, ...]]) -> tuple[PgCorpusStore, FakeConnection]:
    connection = FakeConnection(*results)
    return PgCorpusStore(connection_factory=FakeDatabase(connection)), connection


def _assert_parameters_match_placeholders(connection: FakeConnection) -> None:
    """psycopg rejects a statement whose named placeholders and parameters differ."""

    assert connection.statements
    for sql, params in connection.statements:
        assert set(re.findall(r"%\((\w+)\)s", sql)) == set(params)
        assert "%" not in re.sub(r"%\(\w+\)s", "", sql)


@pytest.mark.parametrize(("person_id", "time_range", "where", "extra"), FILTER_CASES)
def test_pg_pool_query(
    person_id: str | None,
    time_range: TimeRange | None,
    where: str,
    extra: dict[str, Any],
) -> None:
    store, connection = _pg_store([(uuid.UUID(EPISODE_B),), (uuid.UUID(EPISODE_A),)])

    pool = store.pool_ids(PG_OWNER, person_id=person_id, time_range=time_range)

    assert pool == [EPISODE_B, EPISODE_A]  # the database's order, ids as strings
    assert connection.statements == [
        (
            f"SELECT e.id FROM episodes e WHERE {where} ORDER BY e.start_time DESC, e.id",
            {"owner_id": PG_OWNER, **extra},
        )
    ]
    _assert_parameters_match_placeholders(connection)


@pytest.mark.parametrize(("person_id", "time_range", "where", "extra"), FILTER_CASES)
def test_pg_vector_search_query(
    person_id: str | None,
    time_range: TimeRange | None,
    where: str,
    extra: dict[str, Any],
) -> None:
    store, connection = _pg_store([(uuid.UUID(EPISODE_B), 0.25), (uuid.UUID(EPISODE_A), 1.5)])

    hits = store.vector_search(
        PG_OWNER,
        [0.6, 0.8],
        person_id=person_id,
        time_range=time_range,
        limit=7,
    )

    assert hits == [(EPISODE_B, 0.75), (EPISODE_A, -0.5)]  # score = 1 - cosine distance
    assert connection.statements == [
        (
            "SELECT e.id, m.embedding <=> %(query)s::vector AS distance FROM episodes e "
            "JOIN eval_episode_embeddings m ON m.episode_id = e.id AND m.model = %(model)s "
            f"WHERE {where} ORDER BY distance, e.id LIMIT %(limit)s",
            {
                "owner_id": PG_OWNER,
                **extra,
                "query": "[0.6,0.8]",
                "model": config.EMBEDDING_MODEL,
                "limit": 7,
            },
        )
    ]
    _assert_parameters_match_placeholders(connection)


def test_pg_vector_search_defaults_to_the_stage_one_depth_and_checks_the_limit() -> None:
    store, connection = _pg_store([])

    assert store.vector_search(PG_OWNER, [1.0]) == []
    assert connection.statements[0][1]["limit"] == config.STAGE1_TOP_K == 100
    with pytest.raises(ValueError, match="limit must be at least 1"):
        store.vector_search(PG_OWNER, [1.0], limit=0)
    assert len(connection.statements) == 1


def test_pg_list_people_sorts_like_the_in_memory_store() -> None:
    rows = [
        (uuid.UUID(PG_PERSON), "Sam", "Carter", "Sam Carter"),
        (uuid.UUID(EPISODE_B), "Bo", "Chen", None),
        (uuid.UUID(EPISODE_A), "Ana", "Lopez", "Ana Lopez"),
    ]
    store, connection = _pg_store(rows)

    people = store.list_people(PG_OWNER)

    assert people == [
        Person(EPISODE_A, first_name="Ana", last_name="Lopez", display_name="Ana Lopez"),
        Person(EPISODE_B, first_name="Bo", last_name="Chen", display_name="Bo Chen"),
        Person(PG_PERSON, first_name="Sam", last_name="Carter", display_name="Sam Carter"),
    ]
    assert connection.statements == [
        (
            "SELECT p.id, p.first_name, p.last_name, p.display_name FROM people p "
            "WHERE p.user_id = %(owner_id)s::uuid",
            {"owner_id": PG_OWNER},
        )
    ]
    _assert_parameters_match_placeholders(connection)


def test_pg_summaries() -> None:
    store, connection = _pg_store([(uuid.UUID(EPISODE_A), "summary a")])

    assert store.get_summaries([EPISODE_A, EPISODE_B]) == {EPISODE_A: "summary a"}
    assert connection.statements == [
        (
            "SELECT e.id, e.dialogue_summary FROM episodes e "
            "WHERE e.id = ANY(%(episode_ids)s::uuid[])",
            {"episode_ids": [EPISODE_A, EPISODE_B]},
        )
    ]
    _assert_parameters_match_placeholders(connection)


def test_pg_headers() -> None:
    cest = timezone(timedelta(hours=2))
    row = (
        uuid.UUID(EPISODE_A),
        uuid.UUID(PG_PERSON),
        "Sam",
        "Carter",
        None,
        datetime(2026, 10, 10, 14, 0, tzinfo=cest),
        "summary a",
    )
    store, connection = _pg_store([row])

    headers = store.get_headers((EPISODE_A,))

    assert headers == {
        EPISODE_A: ConversationHeader(
            episode_id=EPISODE_A,
            person_id=PG_PERSON,
            person_name="Sam Carter",
            start_time=NOON,
            summary="summary a",
        )
    }
    # UTC, whatever time zone the session reports timestamps in.
    assert headers[EPISODE_A].start_time.utcoffset() == timedelta(0)
    assert connection.statements == [
        (
            "SELECT e.id, e.person_id, p.first_name, p.last_name, p.display_name, e.start_time, "
            "e.dialogue_summary FROM episodes e JOIN people p ON p.id = e.person_id "
            "WHERE e.id = ANY(%(episode_ids)s::uuid[])",
            {"episode_ids": [EPISODE_A]},
        )
    ]
    _assert_parameters_match_placeholders(connection)


def test_pg_conversations() -> None:
    dialogue = [{"speaker": "Sam", "text": "The trip is on."}, {"speaker": "Me", "text": "Great."}]
    ended = NOON + timedelta(minutes=20)
    person = (uuid.UUID(PG_PERSON), "Sam", "Carter", "Sammy")
    rows = [
        (uuid.UUID(EPISODE_A), *person, NOON, ended, "summary a", dialogue),
        (uuid.UUID(EPISODE_B), *person, NOON, None, "summary b", []),
    ]
    store, connection = _pg_store(rows)

    conversations = store.get_conversations([EPISODE_A, EPISODE_B])

    assert conversations[EPISODE_A] == Conversation(
        episode_id=EPISODE_A,
        person_id=PG_PERSON,
        start_time=NOON,
        end_time=ended,
        dialogue=[Turn("Sam", "The trip is on."), Turn("Me", "Great.")],
        summary="summary a",
        meta={"person_name": "Sammy"},
    )
    assert conversations[EPISODE_B].end_time == NOON  # a missing end falls back to the start
    assert conversations[EPISODE_B].dialogue == []
    assert connection.statements == [
        (
            "SELECT e.id, e.person_id, p.first_name, p.last_name, p.display_name, e.start_time, "
            "e.end_time, e.dialogue_summary, d.dialogue FROM episodes e "
            "JOIN people p ON p.id = e.person_id "
            "JOIN eval_episode_dialogues d ON d.episode_id = e.id "
            "WHERE e.id = ANY(%(episode_ids)s::uuid[])",
            {"episode_ids": [EPISODE_A, EPISODE_B]},
        )
    ]
    _assert_parameters_match_placeholders(connection)


def test_pg_lookups_of_no_episodes_skip_the_database() -> None:
    database = FakeDatabase()
    store = PgCorpusStore(connection_factory=database)

    assert store.get_summaries([]) == {}
    assert store.get_headers([]) == {}
    assert store.get_conversations([]) == {}
    assert database.connects == 0


def test_pg_connection_opens_on_first_use_and_is_reused() -> None:
    connection = FakeConnection([], [], [])
    database = FakeDatabase(connection)
    store = PgCorpusStore(connection_factory=database)
    assert database.connects == 0

    store.pool_ids(PG_OWNER)
    store.list_people(PG_OWNER)
    store.get_summaries([EPISODE_A])

    assert database.connects == 1
    assert len(connection.statements) == 3


def test_pg_reconnects_once_when_the_connection_was_dropped() -> None:
    dropped = FakeConnection(error=psycopg.OperationalError("the connection is lost"))
    fresh = FakeConnection([(uuid.UUID(EPISODE_A),)], [(uuid.UUID(EPISODE_B),)])
    database = FakeDatabase(dropped, fresh)
    store = PgCorpusStore(connection_factory=database)

    assert store.pool_ids(PG_OWNER, person_id=PG_PERSON) == [EPISODE_A]

    assert database.connects == 2
    assert dropped.closed
    assert fresh.statements == dropped.statements  # the same statement, retried as it was
    assert store.pool_ids(PG_OWNER) == [EPISODE_B]  # and the new connection is kept
    assert database.connects == 2


def test_pg_gives_up_after_one_reconnect() -> None:
    first = FakeConnection(error=psycopg.OperationalError("the connection is lost"))
    second = FakeConnection(error=psycopg.OperationalError("still unreachable"))
    database = FakeDatabase(first, second, FakeConnection([]))
    store = PgCorpusStore(connection_factory=database)

    with pytest.raises(psycopg.OperationalError, match="still unreachable"):
        store.pool_ids(PG_OWNER)

    assert database.connects == 2
    assert len(first.statements) == len(second.statements) == 1


def test_pg_retries_when_connecting_itself_fails() -> None:
    connection = FakeConnection([(uuid.UUID(EPISODE_A),)])
    attempts: list[str] = []

    def waking_database() -> FakeConnection:
        attempts.append("connect")
        if len(attempts) == 1:
            raise psycopg.OperationalError("the compute is starting")
        return connection

    assert PgCorpusStore(connection_factory=waking_database).pool_ids(PG_OWNER) == [EPISODE_A]
    assert attempts == ["connect", "connect"]


def test_pg_does_not_retry_other_errors() -> None:
    broken = FakeConnection(error=psycopg.ProgrammingError("syntax error"))
    database = FakeDatabase(broken, FakeConnection([]))
    store = PgCorpusStore(connection_factory=database)

    with pytest.raises(psycopg.ProgrammingError):
        store.pool_ids(PG_OWNER)

    assert database.connects == 1
    assert len(broken.statements) == 1
    assert not broken.closed


def test_pg_close_and_context_manager_release_the_connection() -> None:
    first, second = FakeConnection([]), FakeConnection([])
    database = FakeDatabase(first, second)

    with PgCorpusStore(connection_factory=database) as store:
        store.pool_ids(PG_OWNER)
        assert not first.closed
    assert first.closed

    store.close()  # nothing open: a no-op
    store.pool_ids(PG_OWNER)  # a closed store reconnects when it is used again
    assert database.connects == 2
    store.close()
    assert second.closed


def test_pg_close_before_any_use_opens_nothing() -> None:
    database = FakeDatabase()

    PgCorpusStore(connection_factory=database).close()

    assert database.connects == 0


def test_pg_store_requires_a_database_url(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(database_url=None))

    with pytest.raises(RuntimeError, match="EVAL_DATABASE_URL"):
        PgCorpusStore()


def test_pg_store_connects_with_the_libpq_form_of_its_url(monkeypatch) -> None:
    connection = FakeConnection([])
    connects: list[tuple[str, dict[str, Any]]] = []

    def fake_connect(url: str, **kwargs: Any) -> FakeConnection:
        connects.append((url, kwargs))
        return connection

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    location = "riley:secret@ep-eval.example.test/neondb?sslmode=require"
    store = PgCorpusStore(f"postgresql+psycopg://{location}")
    assert connects == []  # constructing the store opens nothing

    store.pool_ids(PG_OWNER)

    assert connects == [(f"postgresql://{location}", {"autocommit": True})]
