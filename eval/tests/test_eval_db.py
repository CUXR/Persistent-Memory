"""Unit tests for the database layer and the load CLI.

No database is involved: a recording fake stands in for the psycopg connection
and the tests check the statements, their parameters and their order.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import psycopg
import pytest

from eval.retrieval import config, db
from eval.retrieval.data import load_db
from eval.retrieval.schema import OWNER_SPEAKER, Conversation, Person, Query, TimeRange, Turn

# The shape of a Neon connection string, on a reserved domain: no such host, no real password.
NEON_URL = "ep-eval-123.eu-central-1.example.test/neondb?sslmode=require&channel_binding=require"
SAM, ANA, LEE = (config.stable_id("person", index) for index in range(3))
PEOPLE = [
    Person(person_id=SAM, first_name="Sam", last_name="Carter", display_name="Sam Carter"),
    Person(person_id=ANA, first_name="Ana", last_name="Lopez", display_name="Ana Lopez"),
    Person(person_id=LEE, first_name="Lee", last_name="Park", display_name="Lee Park"),
]  # Lee has no conversations
MORNING = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)


def _conversation(index: int, person_id: str, start_time: datetime, minutes: int) -> Conversation:
    return Conversation(
        episode_id=config.stable_id("episode", index),
        person_id=person_id,
        start_time=start_time,
        end_time=start_time + timedelta(minutes=minutes),
        dialogue=[Turn("Sam", f'Café "Luna" at {index}?'), Turn(OWNER_SPEAKER, "See you there.")],
        summary=f"Plans for meeting number {index}.",
        meta={"topic": "plans"},
    )


CONVERSATIONS = [
    _conversation(0, SAM, MORNING, minutes=90),  # ends 10:30, after the next one ends
    _conversation(1, SAM, MORNING + timedelta(minutes=30), minutes=15),  # starts later, ends 09:45
    _conversation(2, ANA, MORNING - timedelta(days=3), minutes=20),
]
EMBEDDINGS = {
    conversation.episode_id: [(index + place % 5) / 8 for place in range(config.EMBEDDING_DIM)]
    for index, conversation in enumerate(CONVERSATIONS)
}
# The columns the migrated schema has at 0003_voice_discovery (and the eval tables), and those
# that are NOT NULL without a server default: an INSERT must stay within the first and supply all
# of the second.
SCHEMA_COLUMNS = {
    "users": (
        "id first_name last_name display_name username oauth_provider oauth_subject preferences "
        "created_at updated_at"
    ),
    "people": (
        "id user_id first_name last_name display_name face_embedding voice_embedding "
        "face_embedding_model voice_embedding_model face_key voice_key persona90 last_seen_at "
        "created_at updated_at"
    ),
    "episodes": (
        "id user_id person_id start_time end_time transcript dialogue_summary summary_version "
        "importance_score created_at updated_at"
    ),
    "episode_participants": "episode_id person_id created_at",
    "eval_episode_dialogues": "episode_id dialogue",
    "eval_episode_embeddings": "episode_id model embedding",
    "eval_corpus_meta": "owner_id fingerprint",
}
REQUIRED_COLUMNS = {
    "users": "id first_name last_name username preferences created_at updated_at",
    "people": "id user_id first_name last_name persona90 created_at updated_at",
    "episodes": "id user_id person_id start_time transcript dialogue_summary created_at updated_at",
    "episode_participants": "episode_id person_id created_at",
    "eval_episode_dialogues": "episode_id dialogue",
    "eval_episode_embeddings": "episode_id model embedding",
    "eval_corpus_meta": "owner_id fingerprint",
}
PRIMARY_KEYS = {
    "users": "id",
    "people": "id",
    "episodes": "id",
    "episode_participants": "episode_id person_id",
    "eval_episode_dialogues": "episode_id",
    "eval_episode_embeddings": "episode_id model",
    "eval_corpus_meta": "owner_id",
}


def _squash(sql: str) -> str:
    """Collapse whitespace so statements compare regardless of their layout."""

    return re.sub(r"\s+\)", ")", re.sub(r"\(\s+", "(", " ".join(sql.split())))


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection
        self._sql = ""

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> "RecordingCursor":
        self._sql = sql
        self._connection.record("execute", sql, params)
        return self

    def executemany(self, sql: str, params_seq: Sequence[Mapping[str, Any]]) -> None:
        self._connection.record("executemany", sql, list(params_seq))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._connection.answer(self._sql)


class RecordingConnection:
    """Stands in for a psycopg connection and logs what is done to it, in order."""

    def __init__(self, row: tuple[Any, ...] | None = None, fail_on: str | None = None) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.row = row  # what fetchone returns, unless ``answers`` has a row for the statement
        self.answers: dict[str, tuple[Any, ...] | None] = {}  # text in the statement -> its row
        self.fail_on = fail_on  # a statement containing this text raises
        self.error: Exception = psycopg.DataError("expected 1024 dimensions, not 3")  # what it raises
        self.closed = False

    def __enter__(self) -> "RecordingConnection":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True

    def record(self, kind: str, sql: str, params: Any) -> None:
        self.events.append((kind, _squash(sql), params))
        if self.fail_on is not None and self.fail_on in sql:
            raise self.error

    def answer(self, sql: str) -> tuple[Any, ...] | None:
        return next((row for text, row in self.answers.items() if text in sql), self.row)

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> RecordingCursor:
        return RecordingCursor(self).execute(sql, params)

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.events.append(("begin",))
        try:
            yield
        except BaseException:
            self.events.append(("rollback",))
            raise
        self.events.append(("commit",))

    def statements(self) -> list[tuple[Any, ...]]:
        return [event for event in self.events if len(event) == 3]


def _placeholders(sql: str) -> set[str]:
    assert "%" not in re.sub(r"%\(\w+\)s", "", sql)  # no stray percent sign for psycopg to trip on
    return set(re.findall(r"%\((\w+)\)s", sql))


# --------------------------------------------------------------------------- #
# URLs and connections                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("scheme", "location"),
    [
        ("postgresql+psycopg://", f"riley:s3cret@{NEON_URL}"),  # the form eval/.env.example shows
        ("postgresql+psycopg2://", "riley@localhost:5432/eval"),
        ("postgresql+asyncpg://", "riley@localhost/eval"),
        ("postgresql://", f"riley:s3cret@{NEON_URL}"),
        ("postgres://", "riley@localhost/eval"),
        ("PostgreSQL+psycopg://", "riley@localhost/eval"),
        ("postgresql+psycopg://", "riley:p%40ss%2Fword@localhost/eval?options=-c%20timezone%3DUTC"),
    ],
)
def test_to_libpq_url_changes_only_the_scheme(scheme: str, location: str) -> None:
    converted = db.to_libpq_url(scheme + location)

    assert converted == f"postgresql://{location}"
    assert db.to_libpq_url(converted) == converted  # already converted: unchanged


def test_to_libpq_url_ignores_surrounding_whitespace() -> None:
    assert db.to_libpq_url(" postgresql+psycopg://riley@host/db\n") == "postgresql://riley@host/db"


@pytest.mark.parametrize(
    "url",
    [
        "sqlite+pysqlite:///:memory:",
        "mysql://riley:s3cret@localhost/eval",
        "riley:s3cret@localhost/eval",
        "",
    ],
)
def test_to_libpq_url_rejects_other_urls_without_quoting_the_password(url: str) -> None:
    with pytest.raises(ValueError, match="expected a postgresql:// database URL") as raised:
        db.to_libpq_url(url)

    assert "s3cret" not in str(raised.value)


def test_resolve_url_prefers_the_argument_then_the_settings(monkeypatch) -> None:
    settings_url = "postgresql+psycopg://settings@localhost/eval"
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(database_url=settings_url))

    assert db.resolve_url("postgresql+psycopg://given@host/db") == "postgresql://given@host/db"
    assert db.resolve_url() == "postgresql://settings@localhost/eval"


def test_a_missing_database_url_names_the_variable_to_set(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(database_url=None))

    with pytest.raises(RuntimeError, match="EVAL_DATABASE_URL is not set"):
        db.resolve_url()
    with pytest.raises(RuntimeError, match="EVAL_DATABASE_URL is not set"):
        db.connect()


def test_connect_opens_an_autocommit_connection_to_the_libpq_url(monkeypatch) -> None:
    opened: list[tuple[str, dict[str, Any]]] = []
    connection = RecordingConnection()

    def fake_connect(url: str, **kwargs: Any) -> RecordingConnection:
        opened.append((url, kwargs))
        return connection

    monkeypatch.setattr(psycopg, "connect", fake_connect)

    assert db.connect(f"postgresql+psycopg://riley:s3cret@{NEON_URL}") is connection
    assert opened == [(f"postgresql://riley:s3cret@{NEON_URL}", {"autocommit": True})]


# --------------------------------------------------------------------------- #
# Vector literals                                                              #
# --------------------------------------------------------------------------- #
def test_vector_literal_is_the_pgvector_text_form() -> None:
    assert db.vector_literal([0.5, -1, 0.0, 2.5e-7]) == "[0.5,-1,0,2.5e-07]"
    assert db.vector_literal(np.array([0.25, 1.0], dtype=np.float32)) == "[0.25,1]"
    assert db.vector_literal([]) == "[]"


def test_vector_literal_round_trips_float32() -> None:
    values = np.random.default_rng(config.SEED).standard_normal(2000).astype(np.float32)
    limits = np.finfo(np.float32)
    extremes = np.array([limits.tiny, limits.max, 1 / 3, -1e-30], dtype=np.float32)
    stored = np.concatenate([values, extremes])

    literal = db.vector_literal(stored.tolist())  # as the embedder hands them over: Python floats

    parsed = np.array([float(token) for token in literal[1:-1].split(",")], dtype=np.float32)
    assert np.array_equal(parsed, stored)


# --------------------------------------------------------------------------- #
# Eval-owned tables and reset                                                  #
# --------------------------------------------------------------------------- #
def test_ensure_eval_tables_creates_the_extension_then_the_eval_tables() -> None:
    conn = RecordingConnection()

    db.ensure_eval_tables(conn)

    assert conn.events == [
        ("execute", "CREATE EXTENSION IF NOT EXISTS vector", None),
        (
            "execute",
            "CREATE TABLE IF NOT EXISTS eval_episode_embeddings ("
            "episode_id uuid REFERENCES episodes(id) ON DELETE CASCADE, "
            "model text, "
            "embedding vector(1024) NOT NULL, "
            "PRIMARY KEY (episode_id, model))",
            None,
        ),
        (
            "execute",
            "CREATE TABLE IF NOT EXISTS eval_episode_dialogues ("
            "episode_id uuid PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE, "
            "dialogue jsonb NOT NULL)",
            None,
        ),
        (
            "execute",
            "CREATE TABLE IF NOT EXISTS eval_corpus_meta ("
            "owner_id uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, "
            "fingerprint text NOT NULL)",
            None,
        ),
        (
            "execute",
            "CREATE TABLE IF NOT EXISTS eval_queries ("
            "query_id uuid PRIMARY KEY, "
            "owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
            "split text NOT NULL, "
            "type text NOT NULL, "
            "query text NOT NULL, "
            "gold_name text, "
            "gold_person_id uuid, "
            "gold_time_start timestamptz, "
            "gold_time_end timestamptz, "
            "gold_episode_ids uuid[] NOT NULL, "
            "gold_answer text, "
            "meta jsonb NOT NULL)",
            None,
        ),
    ]
    issued = [sql for _, sql, _ in conn.events]
    assert issued == [_squash(statement) for statement in db.EVAL_SCHEMA_SQL]


def test_reset_deletes_only_the_given_owner() -> None:
    conn = RecordingConnection()

    db.reset_eval_data(conn)
    db.reset_eval_data(conn, owner_id="99999999-9999-4999-8999-999999999999")

    delete = "DELETE FROM users WHERE id = %(owner_id)s::uuid"
    assert conn.events == [
        ("execute", delete, {"owner_id": config.OWNER_ID}),
        ("execute", delete, {"owner_id": "99999999-9999-4999-8999-999999999999"}),
    ]


# --------------------------------------------------------------------------- #
# load_dataset                                                                 #
# --------------------------------------------------------------------------- #
def _loaded() -> RecordingConnection:
    conn = RecordingConnection()
    db.load_dataset(conn, PEOPLE, CONVERSATIONS, EMBEDDINGS)
    return conn


def _insert_shape(sql: str) -> tuple[str, list[str], list[str], list[str], list[str]]:
    """(table, inserted columns, values, conflict target, updated columns) of an upsert."""

    match = re.fullmatch(
        r"INSERT INTO (\w+) \((.*?)\) VALUES \((.*)\) ON CONFLICT \((.*?)\) "
        r"DO (?:NOTHING|UPDATE SET (.*))",
        sql,
    )
    assert match, sql
    table, columns, values, target, updates = match.groups()
    updated: list[str] = []
    for assignment in updates.split(", ") if updates else []:
        column, _, source = assignment.partition(" = ")
        assert source == f"EXCLUDED.{column}"
        updated.append(column)
    return table, columns.split(", "), values.split(", "), target.split(", "), updated


def test_load_runs_in_one_transaction_in_dependency_order() -> None:
    conn = _loaded()

    assert conn.events[0] == ("begin",)
    assert conn.events[-1] == ("commit",)
    statements = conn.events[1:-1]
    assert [event[0] for event in statements] == ["execute", *["executemany"] * 5, "execute"]
    assert [_insert_shape(event[1])[0] for event in statements] == [
        "users",
        "people",
        "episodes",
        "episode_participants",
        "eval_episode_dialogues",
        "eval_episode_embeddings",
        "eval_corpus_meta",  # last: the fingerprint says the rows before it are there
    ]


def test_load_statements_are_upserts_within_the_migrated_schema() -> None:
    for _, sql, params in _loaded().statements():
        table, columns, values, target, updated = _insert_shape(sql)
        rows = [params] if isinstance(params, dict) else params

        assert set(columns) <= set(SCHEMA_COLUMNS[table].split())
        assert set(REQUIRED_COLUMNS[table].split()) <= set(columns)
        assert len(values) == len(columns)
        assert target == PRIMARY_KEYS[table].split()
        # A reload overwrites every column it writes, bar the key and the creation time.
        assert updated == [column for column in columns if column not in (*target, "created_at")]
        assert all(set(row) == _placeholders(sql) for row in rows)


def test_load_writes_the_owner_from_config() -> None:
    _, sql, params = _loaded().statements()[0]

    assert params == {
        "id": config.OWNER_ID,
        "first_name": "Riley",
        "last_name": "Morgan",
        "display_name": "Riley Morgan",
        "username": "eval-riley-morgan",
        "loaded_at": config.NOW,
    }
    assert "%(id)s::uuid" in sql and sql.count("%(loaded_at)s") == 2  # created_at and updated_at


def test_load_fills_the_not_null_columns_with_the_orm_defaults() -> None:
    owner, people = (_insert_shape(event[1]) for event in _loaded().statements()[:2])

    assert dict(zip(owner[1], owner[2]))["preferences"] == "'{}'::jsonb"
    assert dict(zip(people[1], people[2]))["persona90"] == "'{}'::float8[]"


def test_load_writes_people_with_their_latest_conversation_end() -> None:
    _, _, rows = _loaded().statements()[1]

    assert rows == [
        {
            "id": person.person_id,
            "user_id": config.OWNER_ID,
            "first_name": person.first_name,
            "last_name": person.last_name,
            "display_name": person.display_name,
            "last_seen_at": last_seen_at,
            "loaded_at": config.NOW,
        }
        for person, last_seen_at in zip(
            PEOPLE,
            [
                MORNING + timedelta(minutes=90),  # the latest end, not the end of the latest start
                MORNING - timedelta(days=3) + timedelta(minutes=20),
                None,
            ],
        )
    ]


def test_load_writes_episodes_participants_dialogues_and_embeddings() -> None:
    statements = _loaded().statements()
    episode_rows, participant_rows, dialogue_rows, embedding_rows = (
        event[2] for event in statements[2:6]
    )

    assert episode_rows == [
        {
            "id": conversation.episode_id,
            "user_id": config.OWNER_ID,
            "person_id": conversation.person_id,
            "start_time": conversation.start_time,
            "end_time": conversation.end_time,
            "transcript": conversation.dialogue_text(),
            "summary": conversation.summary,
            "summary_version": "eval-v1",
            "loaded_at": config.NOW,
        }
        for conversation in CONVERSATIONS
    ]
    assert episode_rows[0]["transcript"] == 'Sam: "Café "Luna" at 0?"\nMe: "See you there."'
    assert "%(transcript)s, %(summary)s, %(summary_version)s" in statements[2][1]
    assert "transcript, dialogue_summary, summary_version" in statements[2][1]

    # One participant per episode: the person the conversation was with.
    assert participant_rows == [
        {
            "episode_id": conversation.episode_id,
            "person_id": conversation.person_id,
            "loaded_at": config.NOW,
        }
        for conversation in CONVERSATIONS
    ]
    assert statements[3][1].endswith("ON CONFLICT (episode_id, person_id) DO NOTHING")

    assert [row["episode_id"] for row in dialogue_rows] == [c.episode_id for c in CONVERSATIONS]
    assert dialogue_rows[0]["dialogue"] == (
        '[{"speaker": "Sam", "text": "Café \\"Luna\\" at 0?"}, '
        '{"speaker": "Me", "text": "See you there."}]'
    )
    for row, conversation in zip(dialogue_rows, CONVERSATIONS):
        turns = [Turn(turn["speaker"], turn["text"]) for turn in json.loads(row["dialogue"])]
        assert turns == conversation.dialogue
    assert "%(dialogue)s::jsonb" in statements[4][1]

    assert embedding_rows == [
        {
            "episode_id": conversation.episode_id,
            "model": "BAAI/bge-m3",
            "embedding": db.vector_literal(EMBEDDINGS[conversation.episode_id]),
        }
        for conversation in CONVERSATIONS
    ]
    assert embedding_rows[1]["embedding"].startswith("[0.125,0.25,0.375,0.5,0.625,0.125,")
    assert embedding_rows[1]["embedding"].count(",") == config.EMBEDDING_DIM - 1
    assert "%(embedding)s::vector" in statements[5][1]


def test_loading_twice_issues_identical_statements() -> None:
    assert _loaded().events == _loaded().events


def test_a_failing_statement_rolls_the_whole_load_back() -> None:
    conn = RecordingConnection(fail_on="eval_episode_embeddings")

    with pytest.raises(psycopg.DataError):
        db.load_dataset(conn, PEOPLE, CONVERSATIONS, EMBEDDINGS)

    assert conn.events[0] == ("begin",)
    assert conn.events[-1] == ("rollback",)
    assert ("commit",) not in conn.events


def test_embeddings_are_checked_before_anything_is_written() -> None:
    conn = RecordingConnection()
    first, second = CONVERSATIONS[0].episode_id, CONVERSATIONS[1].episode_id
    without_two = {CONVERSATIONS[2].episode_id: EMBEDDINGS[CONVERSATIONS[2].episode_id]}

    with pytest.raises(ValueError, match=rf"no embedding for 2 of 3 .* \(first: {first}\)"):
        db.load_dataset(conn, PEOPLE, CONVERSATIONS, without_two)
    with pytest.raises(ValueError, match=rf"embedding of {second} has 3 dimensions, expected 1024"):
        db.load_dataset(conn, PEOPLE, CONVERSATIONS, {**EMBEDDINGS, second: [0.1, 0.2, 0.3]})

    assert conn.events == []


# --------------------------------------------------------------------------- #
# The corpus fingerprint                                                       #
# --------------------------------------------------------------------------- #
def test_load_writes_the_dataset_fingerprint_for_the_owner() -> None:
    kind, sql, params = _loaded().statements()[-1]

    assert kind == "execute"
    assert sql == (
        "INSERT INTO eval_corpus_meta (owner_id, fingerprint) VALUES (%(owner_id)s::uuid, %(fingerprint)s) "
        "ON CONFLICT (owner_id) DO UPDATE SET fingerprint = EXCLUDED.fingerprint"
    )
    assert params == {"owner_id": config.OWNER_ID, "fingerprint": db.dataset_fingerprint(PEOPLE, CONVERSATIONS)}
    assert re.fullmatch(r"[0-9a-f]{64}", params["fingerprint"])


def test_dataset_fingerprint_follows_the_loaded_content_and_not_the_row_order(monkeypatch) -> None:
    fingerprint = db.dataset_fingerprint(PEOPLE, CONVERSATIONS)
    shuffled = db.dataset_fingerprint(PEOPLE[::-1], CONVERSATIONS[::-1])
    copied = db.dataset_fingerprint(
        [Person.from_json(person.to_json()) for person in PEOPLE],
        [Conversation.from_json(conversation.to_json()) for conversation in CONVERSATIONS],
    )
    assert shuffled == copied == fingerprint

    # Anything a store serves from the database changes it; what is not loaded does not.
    first, rest = CONVERSATIONS[0], CONVERSATIONS[1:]
    changed = [
        [*PEOPLE, Person(person_id=config.stable_id("person", 9), first_name="Kim", last_name="Ng", display_name="Kim Ng")],
        [*PEOPLE[:2], Person(person_id=LEE, first_name="Lee", last_name="Parker", display_name="Lee Parker")],
    ]
    for altered in (
        Conversation.from_json({**first.to_json(), "summary": first.summary + " Edited."}),
        Conversation.from_json({**first.to_json(), "start_time": "2026-10-10T09:00:01Z"}),
        Conversation.from_json({**first.to_json(), "end_time": "2026-10-10T10:30:01Z"}),
        Conversation.from_json({**first.to_json(), "person_id": ANA}),
        Conversation.from_json({**first.to_json(), "dialogue": [{"speaker": "Sam", "text": "Changed."}]}),
    ):
        assert db.dataset_fingerprint(PEOPLE, [altered, *rest]) != fingerprint
    assert db.dataset_fingerprint(PEOPLE, rest) != fingerprint
    for people in changed:
        assert db.dataset_fingerprint(people, CONVERSATIONS) != fingerprint
    meta_only = Conversation.from_json({**first.to_json(), "meta": {"topic": "other", "recap": True}})
    assert db.dataset_fingerprint(PEOPLE, [meta_only, *rest]) == fingerprint

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "other/embedder")
    assert db.dataset_fingerprint(PEOPLE, CONVERSATIONS) != fingerprint  # the rows must be embedded again


def test_corpus_fingerprint_reads_the_owner_row_or_none() -> None:
    conn = RecordingConnection(row=("abc123",))
    assert db.corpus_fingerprint(conn) == "abc123"
    [(kind, sql, params)] = conn.events
    assert kind == "execute"
    assert sql == "SELECT fingerprint FROM eval_corpus_meta WHERE owner_id = %(owner_id)s::uuid"
    assert params == {"owner_id": config.OWNER_ID}

    other = "99999999-9999-4999-8999-999999999999"
    assert db.corpus_fingerprint(RecordingConnection(row=None), owner_id=other) is None
    assert RecordingConnection(row=None).events == []

    # A database loaded before the table existed records nothing: it has to be loaded again.
    missing = RecordingConnection(fail_on="eval_corpus_meta")
    missing.error = psycopg.errors.UndefinedTable('relation "eval_corpus_meta" does not exist')
    assert db.corpus_fingerprint(missing) is None


def test_check_corpus_passes_only_the_fingerprint_of_this_dataset() -> None:
    fingerprint = db.dataset_fingerprint(PEOPLE, CONVERSATIONS)

    assert db.check_corpus(RecordingConnection(row=(fingerprint,)), PEOPLE, CONVERSATIONS) == fingerprint

    stale = RecordingConnection(row=("0" * 64,))
    with pytest.raises(db.StaleCorpusError, match=f"fingerprint {'0' * 64} in the database, the dataset's is {fingerprint}"):
        db.check_corpus(stale, PEOPLE, CONVERSATIONS)
    with pytest.raises(db.StaleCorpusError, match="no fingerprint in the database") as caught:
        db.check_corpus(RecordingConnection(row=None), PEOPLE, CONVERSATIONS[:2])
    assert "python -m eval.retrieval.data.load_db --reset" in str(caught.value)
    assert isinstance(caught.value, RuntimeError)


# --------------------------------------------------------------------------- #
# Queries                                                                      #
# --------------------------------------------------------------------------- #
QUERIES = [
    Query(
        query_id=config.stable_id("query", "temporal:0"),
        query="What did Sam and I talk about on Saturday?",
        type="temporal",
        gold_name="Sam",
        gold_time_range=TimeRange(MORNING, MORNING + timedelta(hours=12)),
        gold_episode_ids=[CONVERSATIONS[0].episode_id, CONVERSATIONS[1].episode_id],
        gold_answer="You made plans to meet.",
        split="dev",
        gold_person_id=SAM,
        meta={"writer_model": "m", "attempt": 0},
    ),
    Query(
        query_id=config.stable_id("query", "unknown_name:0"),
        query="What did Sunita say about the café?",
        type="unknown_name",
        gold_name="Sunita",
        gold_time_range=None,
        gold_episode_ids=[],
        gold_answer=None,
    ),
]


def test_load_queries_replaces_the_owners_rows_in_one_transaction() -> None:
    conn = RecordingConnection()

    db.load_queries(conn, QUERIES)

    assert [event[0] for event in conn.events] == ["begin", "execute", "executemany", "commit"]
    _, delete, params = conn.events[1]
    assert delete == "DELETE FROM eval_queries WHERE owner_id = %(owner_id)s::uuid"
    assert params == {"owner_id": config.OWNER_ID}
    _, insert, rows = conn.events[2]
    assert insert.startswith("INSERT INTO eval_queries (query_id, owner_id, split, type, query, gold_name,")
    assert all(set(row) == _placeholders(insert) for row in rows)
    assert rows[0] == {
        "query_id": QUERIES[0].query_id,
        "owner_id": config.OWNER_ID,
        "split": "dev",
        "type": "temporal",
        "query": "What did Sam and I talk about on Saturday?",
        "gold_name": "Sam",
        "gold_person_id": SAM,
        "gold_time_start": MORNING,
        "gold_time_end": MORNING + timedelta(hours=12),
        "gold_episode_ids": [CONVERSATIONS[0].episode_id, CONVERSATIONS[1].episode_id],
        "gold_answer": "You made plans to meet.",
        "meta": '{"attempt": 0, "writer_model": "m"}',
    }
    # No name resolves, no time range, no gold conversations, no answer: NULLs and an empty array.
    assert [rows[1][key] for key in ("split", "gold_person_id", "gold_time_start", "gold_time_end")] == [
        "test",
        None,
        None,
        None,
    ]
    assert rows[1]["gold_episode_ids"] == [] and rows[1]["gold_answer"] is None and rows[1]["meta"] == "{}"
    assert "%(gold_episode_ids)s::uuid[]" in insert and "%(meta)s::jsonb" in insert


# --------------------------------------------------------------------------- #
# Row counts and the CLI                                                       #
# --------------------------------------------------------------------------- #
def test_count_rows_reads_one_count_per_table_for_the_owner() -> None:
    conn = RecordingConnection(row=(1, 40, 800, 800, 800, 799))

    counts = db.count_rows(conn)

    assert counts == {
        "users": 1,
        "people": 40,
        "episodes": 800,
        "episode_participants": 800,
        "eval_episode_dialogues": 800,
        "eval_episode_embeddings": 799,
    }
    assert list(counts) == list(db.COUNTED_TABLES)
    [(kind, sql, params)] = conn.events
    assert kind == "execute"
    assert params == {"owner_id": config.OWNER_ID, "model": "BAAI/bge-m3"}
    assert _placeholders(sql) == set(params)
    assert re.findall(r"count\(\*\) FROM (\w+)", sql) == list(db.COUNTED_TABLES)
    assert sql.count("%(owner_id)s::uuid") == 6  # every count is restricted to the owner
    assert "m.model = %(model)s" in sql


class _Cli:
    """The load CLI wired to fakes: records what it loads, embeds and connects to."""

    def __init__(self, monkeypatch, counts: tuple[int, ...]) -> None:
        self.conn = RecordingConnection(row=counts)
        # What the fingerprint read-back returns: the dataset's, as a load just wrote it.
        self.conn.answers["eval_corpus_meta"] = (db.dataset_fingerprint(PEOPLE, CONVERSATIONS),)
        self.calls: list[Any] = []
        self.embedder = SimpleNamespace(model_name="fake-bge-m3")
        monkeypatch.setattr(load_db.db, "resolve_url", self._resolve_url)
        monkeypatch.setattr(load_db.db, "connect", self._connect)
        monkeypatch.setattr(load_db.dataset, "load_people", lambda: PEOPLE)
        monkeypatch.setattr(load_db.dataset, "load_conversations", lambda: CONVERSATIONS)
        monkeypatch.setattr(load_db, "_committed_queries", lambda: None)  # no queries.jsonl yet
        monkeypatch.setattr(load_db, "BgeM3Embedder", lambda: self.embedder)
        monkeypatch.setattr(load_db, "load_or_compute_corpus_embeddings", self._embeddings)

    def _resolve_url(self) -> str:
        self.calls.append("resolve_url")
        return "postgresql://riley@localhost/eval"

    def _connect(self, database_url: str) -> RecordingConnection:
        self.calls.append(("connect", database_url))
        return self.conn

    def _embeddings(
        self,
        conversations: Sequence[Conversation],
        embedder: Any,
    ) -> dict[str, list[float]]:
        self.calls.append(("embed", len(conversations), embedder))
        return EMBEDDINGS

    def outline(self) -> list[str]:
        """What happened on the connection, in order, each statement cut down to verb and table."""

        return [
            event[0] if len(event) == 1 else re.split(r" \(| WHERE", event[1])[0]
            for event in self.conn.events
        ]


LOAD_OUTLINE = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE TABLE IF NOT EXISTS eval_episode_embeddings",
    "CREATE TABLE IF NOT EXISTS eval_episode_dialogues",
    "CREATE TABLE IF NOT EXISTS eval_corpus_meta",
    "CREATE TABLE IF NOT EXISTS eval_queries",
    "begin",  # the CLI's transaction, around the reset and the load
    "begin",  # load_dataset's own, nested inside it
    "INSERT INTO users",
    "INSERT INTO people",
    "INSERT INTO episodes",
    "INSERT INTO episode_participants",
    "INSERT INTO eval_episode_dialogues",
    "INSERT INTO eval_episode_embeddings",
    "INSERT INTO eval_corpus_meta",
    "commit",
    "commit",
    "SELECT",  # the counts, read back once the load has committed
    "SELECT fingerprint FROM eval_corpus_meta",  # and the fingerprint the load wrote
]


def test_cli_loads_and_prints_the_counts_read_back(monkeypatch, capsys) -> None:
    cli = _Cli(monkeypatch, counts=(1, 3, 3, 3, 3, 3))

    assert load_db.main([]) == 0

    assert cli.calls == [
        "resolve_url",  # before the embeddings: an unconfigured database fails at once
        ("embed", 3, cli.embedder),
        ("connect", "postgresql://riley@localhost/eval"),
    ]
    assert cli.outline() == LOAD_OUTLINE  # and no DELETE without --reset
    assert cli.conn.closed
    output = capsys.readouterr()
    assert output.err == ""
    assert f"owner {config.OWNER_ID} (Riley Morgan)" in output.out
    assert f"Corpus fingerprint: {db.dataset_fingerprint(PEOPLE, CONVERSATIONS)}" in output.out
    assert [line.split() for line in output.out.splitlines()[-6:]] == [
        ["users", "1"],
        ["people", "3"],
        ["episodes", "3"],
        ["episode_participants", "3"],
        ["eval_episode_dialogues", "3"],
        ["eval_episode_embeddings", "3"],
    ]


def test_cli_reset_deletes_the_owner_in_the_same_transaction_as_the_load(monkeypatch) -> None:
    cli = _Cli(monkeypatch, counts=(1, 3, 3, 3, 3, 3))

    assert load_db.main(["--reset"]) == 0

    # The delete sits inside the CLI's transaction, ahead of the load.
    assert cli.outline() == [*LOAD_OUTLINE[:6], "DELETE FROM users", *LOAD_OUTLINE[6:]]
    _, sql, params = cli.conn.events[6]
    assert sql == "DELETE FROM users WHERE id = %(owner_id)s::uuid"
    assert params == {"owner_id": config.OWNER_ID}


def test_cli_copies_the_queries_inside_the_load_transaction(monkeypatch, capsys) -> None:
    cli = _Cli(monkeypatch, counts=(1, 3, 3, 3, 3, 3))
    monkeypatch.setattr(load_db, "_committed_queries", lambda: QUERIES)

    assert load_db.main([]) == 0

    # After the corpus, before the CLI's commit: begin / delete / insert / commit of load_queries.
    start = LOAD_OUTLINE.index("INSERT INTO eval_corpus_meta") + 2  # past load_dataset's own commit
    assert cli.outline() == [
        *LOAD_OUTLINE[:start],
        "begin",
        "DELETE FROM eval_queries",
        "INSERT INTO eval_queries",
        "commit",
        *LOAD_OUTLINE[start:],
    ]
    assert "Queries copied into eval_queries: 2" in capsys.readouterr().out


def test_cli_says_when_there_are_no_queries_yet(monkeypatch, capsys) -> None:
    _Cli(monkeypatch, counts=(1, 3, 3, 3, 3, 3))

    assert load_db.main([]) == 0

    assert "No queries.jsonl yet" in capsys.readouterr().out


def test_cli_reset_is_rolled_back_when_the_load_fails(monkeypatch) -> None:
    cli = _Cli(monkeypatch, counts=(1, 3, 3, 3, 3, 3))
    cli.conn.fail_on = "INSERT INTO episodes"

    with pytest.raises(psycopg.DataError):
        load_db.main(["--reset"])

    assert cli.conn.events[-2:] == [("rollback",), ("rollback",)]  # the load, then the reset
    assert ("commit",) not in cli.conn.events
    assert cli.conn.closed


def test_cli_fails_when_the_database_holds_rows_the_dataset_lacks(monkeypatch, capsys) -> None:
    cli = _Cli(monkeypatch, counts=(1, 3, 5, 5, 5, 3))

    assert load_db.main([]) == 1

    output = capsys.readouterr()
    assert "episodes" in output.out and "5" in output.out  # the counts are still printed
    assert (
        "episodes (dataset: 3), episode_participants (dataset: 3), "
        "eval_episode_dialogues (dataset: 3)"
    ) in output.err
    assert "--reset" in output.err
    assert "fingerprint" not in output.err
    assert cli.conn.closed


def test_cli_fails_when_the_fingerprint_read_back_is_not_the_datasets(monkeypatch, capsys) -> None:
    cli = _Cli(monkeypatch, counts=(1, 3, 3, 3, 3, 3))
    cli.conn.answers["eval_corpus_meta"] = ("f" * 64,)

    assert load_db.main([]) == 1

    output = capsys.readouterr()
    assert f"Corpus fingerprint: {'f' * 64}" in output.out
    assert output.err.splitlines() == [
        f"The database records the corpus fingerprint {'f' * 64}, "
        f"the dataset's is {db.dataset_fingerprint(PEOPLE, CONVERSATIONS)}."
    ]
    assert cli.conn.closed


def test_cli_stops_before_embedding_when_no_database_is_configured(monkeypatch) -> None:
    embedded: list[int] = []
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(database_url=None))
    monkeypatch.setattr(load_db.dataset, "load_people", lambda: PEOPLE)
    monkeypatch.setattr(load_db.dataset, "load_conversations", lambda: CONVERSATIONS)
    monkeypatch.setattr(
        load_db,
        "load_or_compute_corpus_embeddings",
        lambda conversations, embedder: embedded.append(len(conversations)),
    )

    with pytest.raises(RuntimeError, match="EVAL_DATABASE_URL is not set"):
        load_db.main([])

    assert embedded == []
