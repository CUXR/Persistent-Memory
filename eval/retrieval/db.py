"""Postgres access for the benchmark: connections, the eval-owned tables, the dataset load.

The benchmark database is a Neon branch migrated with ``alembic upgrade head``
(``origin/main``'s migrations, head ``0003_voice_discovery``). This module talks
to the migrated tables (``users``, ``people``, ``episodes``,
``episode_participants``) in plain SQL, writing them the way the backend's
ingestion does: the dialogue goes into ``episodes.transcript`` as
``Sam: "..."`` lines and each episode gets its participant row. The schema has
nowhere to keep a 1024-d summary embedding, so the embedding, the dialogue's
structured turns and the fingerprint of the loaded dataset live in tables of
its own, and so does a readable copy of the benchmark queries. They are
created here rather than by a migration, so the backend schema stays as it is.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import psycopg

from . import config
from .schema import Conversation, Person, Query, parse_dt

SUMMARY_VERSION = "eval-v1"
LOAD_COMMAND = "python -m eval.retrieval.data.load_db --reset"

EVAL_SCHEMA_SQL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS eval_episode_embeddings (
        episode_id uuid REFERENCES episodes(id) ON DELETE CASCADE,
        model text,
        embedding vector({config.EMBEDDING_DIM}) NOT NULL,
        PRIMARY KEY (episode_id, model)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_episode_dialogues (
        episode_id uuid PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
        dialogue jsonb NOT NULL
    )
    """,
    # One row per owner: the fingerprint of the dataset the last load wrote, so a
    # run can tell a database loaded from an older conversations.jsonl (the ids
    # are stable, so stale summaries and dialogues would otherwise go unnoticed).
    """
    CREATE TABLE IF NOT EXISTS eval_corpus_meta (
        owner_id uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        fingerprint text NOT NULL
    )
    """,
    # A copy of queries.jsonl for people who read the eval branch. The runner
    # reads the file, never this table.
    """
    CREATE TABLE IF NOT EXISTS eval_queries (
        query_id uuid PRIMARY KEY,
        owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        split text NOT NULL,
        type text NOT NULL,
        query text NOT NULL,
        gold_name text,
        gold_person_id uuid,
        gold_time_start timestamptz,
        gold_time_end timestamptz,
        gold_episode_ids uuid[] NOT NULL,
        gold_answer text,
        meta jsonb NOT NULL
    )
    """,
)

# created_at / updated_at have no server default. Every upsert writes the
# benchmark's fixed clock there, so reloading the same files leaves the rows
# exactly as they were. preferences and persona90 are NOT NULL without a server
# default either: they get what the ORM defaults to, an empty object and an
# empty array.
_UPSERT_OWNER = """
    INSERT INTO users (id, first_name, last_name, display_name, username, preferences,
                       created_at, updated_at)
    VALUES (%(id)s::uuid, %(first_name)s, %(last_name)s, %(display_name)s, %(username)s,
            '{}'::jsonb, %(loaded_at)s, %(loaded_at)s)
    ON CONFLICT (id) DO UPDATE SET
        first_name = EXCLUDED.first_name,
        last_name = EXCLUDED.last_name,
        display_name = EXCLUDED.display_name,
        username = EXCLUDED.username,
        preferences = EXCLUDED.preferences,
        updated_at = EXCLUDED.updated_at
"""

_UPSERT_PERSON = """
    INSERT INTO people (id, user_id, first_name, last_name, display_name, persona90,
                        last_seen_at, created_at, updated_at)
    VALUES (%(id)s::uuid, %(user_id)s::uuid, %(first_name)s, %(last_name)s, %(display_name)s,
            '{}'::float8[], %(last_seen_at)s, %(loaded_at)s, %(loaded_at)s)
    ON CONFLICT (id) DO UPDATE SET
        user_id = EXCLUDED.user_id,
        first_name = EXCLUDED.first_name,
        last_name = EXCLUDED.last_name,
        display_name = EXCLUDED.display_name,
        persona90 = EXCLUDED.persona90,
        last_seen_at = EXCLUDED.last_seen_at,
        updated_at = EXCLUDED.updated_at
"""

_UPSERT_EPISODE = """
    INSERT INTO episodes (id, user_id, person_id, start_time, end_time, transcript,
                          dialogue_summary, summary_version, created_at, updated_at)
    VALUES (%(id)s::uuid, %(user_id)s::uuid, %(person_id)s::uuid, %(start_time)s, %(end_time)s,
            %(transcript)s, %(summary)s, %(summary_version)s, %(loaded_at)s, %(loaded_at)s)
    ON CONFLICT (id) DO UPDATE SET
        user_id = EXCLUDED.user_id,
        person_id = EXCLUDED.person_id,
        start_time = EXCLUDED.start_time,
        end_time = EXCLUDED.end_time,
        transcript = EXCLUDED.transcript,
        dialogue_summary = EXCLUDED.dialogue_summary,
        summary_version = EXCLUDED.summary_version,
        updated_at = EXCLUDED.updated_at
"""

# One participant per episode, as the backend's ingestion writes it. The row is
# all key and creation time, so a reload has nothing to overwrite. An episode
# whose person changed between loads keeps its old row until --reset.
_UPSERT_PARTICIPANT = """
    INSERT INTO episode_participants (episode_id, person_id, created_at)
    VALUES (%(episode_id)s::uuid, %(person_id)s::uuid, %(loaded_at)s)
    ON CONFLICT (episode_id, person_id) DO NOTHING
"""

_UPSERT_DIALOGUE = """
    INSERT INTO eval_episode_dialogues (episode_id, dialogue)
    VALUES (%(episode_id)s::uuid, %(dialogue)s::jsonb)
    ON CONFLICT (episode_id) DO UPDATE SET dialogue = EXCLUDED.dialogue
"""

_UPSERT_EMBEDDING = """
    INSERT INTO eval_episode_embeddings (episode_id, model, embedding)
    VALUES (%(episode_id)s::uuid, %(model)s, %(embedding)s::vector)
    ON CONFLICT (episode_id, model) DO UPDATE SET embedding = EXCLUDED.embedding
"""

_UPSERT_CORPUS_META = """
    INSERT INTO eval_corpus_meta (owner_id, fingerprint)
    VALUES (%(owner_id)s::uuid, %(fingerprint)s)
    ON CONFLICT (owner_id) DO UPDATE SET fingerprint = EXCLUDED.fingerprint
"""

SELECT_CORPUS_META = "SELECT fingerprint FROM eval_corpus_meta WHERE owner_id = %(owner_id)s::uuid"

_DELETE_QUERIES = "DELETE FROM eval_queries WHERE owner_id = %(owner_id)s::uuid"

_INSERT_QUERY = """
    INSERT INTO eval_queries (query_id, owner_id, split, type, query, gold_name, gold_person_id,
                              gold_time_start, gold_time_end, gold_episode_ids, gold_answer, meta)
    VALUES (%(query_id)s::uuid, %(owner_id)s::uuid, %(split)s, %(type)s, %(query)s, %(gold_name)s,
            %(gold_person_id)s::uuid, %(gold_time_start)s, %(gold_time_end)s,
            %(gold_episode_ids)s::uuid[], %(gold_answer)s, %(meta)s::jsonb)
"""

_DELETE_OWNER = "DELETE FROM users WHERE id = %(owner_id)s::uuid"

_COUNT_ROWS = """
    SELECT
        (SELECT count(*) FROM users WHERE id = %(owner_id)s::uuid),
        (SELECT count(*) FROM people WHERE user_id = %(owner_id)s::uuid),
        (SELECT count(*) FROM episodes WHERE user_id = %(owner_id)s::uuid),
        (SELECT count(*) FROM episode_participants ep
            JOIN episodes e ON e.id = ep.episode_id
            WHERE e.user_id = %(owner_id)s::uuid),
        (SELECT count(*) FROM eval_episode_dialogues d
            JOIN episodes e ON e.id = d.episode_id
            WHERE e.user_id = %(owner_id)s::uuid),
        (SELECT count(*) FROM eval_episode_embeddings m
            JOIN episodes e ON e.id = m.episode_id
            WHERE e.user_id = %(owner_id)s::uuid AND m.model = %(model)s)
"""
COUNTED_TABLES: tuple[str, ...] = (
    "users",
    "people",
    "episodes",
    "episode_participants",
    "eval_episode_dialogues",
    "eval_episode_embeddings",
)


def to_libpq_url(url: str) -> str:
    """Rewrite a SQLAlchemy URL (``postgresql+psycopg://...``) as one psycopg accepts.

    Only the scheme changes, so the credentials, the host and the query string
    (``sslmode=require``) pass through untouched.
    """

    scheme, separator, rest = url.strip().partition("://")
    # The messages never quote the URL itself: it carries the password.
    if not separator:
        raise ValueError("expected a postgresql:// database URL, found no scheme")
    if scheme.partition("+")[0].lower() not in ("postgresql", "postgres"):
        raise ValueError(f"expected a postgresql:// database URL, got scheme {scheme!r}")
    return f"postgresql://{rest}"


def resolve_url(database_url: str | None = None) -> str:
    """The libpq URL of the eval database: the argument, else ``EVAL_DATABASE_URL``."""

    url = database_url or config.get_settings().database_url
    if not url:
        raise RuntimeError(
            "EVAL_DATABASE_URL is not set: point it at the Neon eval branch (see eval/.env.example)"
        )
    return to_libpq_url(url)


def connect(database_url: str | None = None) -> psycopg.Connection[Any]:
    """Open an autocommit connection to the eval database.

    Autocommit, so reads never leave a transaction open on an idle connection
    and a failed statement does not poison the ones after it. Writers open
    their transaction explicitly.
    """

    return psycopg.connect(resolve_url(database_url), autocommit=True)


def vector_literal(values: Sequence[float]) -> str:
    """An embedding in pgvector's text form, for a ``%(name)s::vector`` parameter.

    Sending text needs no adapter registered on the connection. Nine
    significant digits round-trip a float32, which is what pgvector stores.
    """

    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def ensure_eval_tables(conn: psycopg.Connection[Any]) -> None:
    """Create the pgvector extension and the eval-owned tables where missing."""

    for statement in EVAL_SCHEMA_SQL:
        conn.execute(statement)


def reset_eval_data(conn: psycopg.Connection[Any], owner_id: str = config.OWNER_ID) -> None:
    """Delete everything the benchmark loaded for ``owner_id``, and nothing else.

    Removing the owner's ``users`` row cascades to their people and episodes,
    and from the episodes to their participant rows and the two eval tables.
    Other owners are untouched.
    """

    conn.execute(_DELETE_OWNER, {"owner_id": owner_id})


def load_dataset(
    conn: psycopg.Connection[Any],
    people: Sequence[Person],
    conversations: Sequence[Conversation],
    embeddings: Mapping[str, Sequence[float]],
) -> None:
    """Upsert the eval owner, their people and every conversation, in one transaction.

    Safe to repeat: rows are keyed by the dataset's stable ids and overwritten
    in place. Rows the dataset no longer contains are left alone; use
    :func:`reset_eval_data` first to drop them. Each table is written with one
    ``executemany``, which psycopg sends as a single pipelined batch. The
    dataset's fingerprint (:func:`dataset_fingerprint`) is written last, in the
    same transaction, so it is there exactly when the rows are.
    """

    _check_embeddings(conversations, embeddings)
    owner_id = config.OWNER_ID
    loaded_at = config.NOW
    meta_row = {"owner_id": owner_id, "fingerprint": dataset_fingerprint(people, conversations)}
    last_seen: dict[str, datetime] = {}
    for conversation in conversations:
        seen = last_seen.get(conversation.person_id)
        if seen is None or conversation.end_time > seen:
            last_seen[conversation.person_id] = conversation.end_time

    owner_row = {
        "id": owner_id,
        "first_name": config.OWNER_FIRST_NAME,
        "last_name": config.OWNER_LAST_NAME,
        "display_name": config.OWNER_DISPLAY_NAME,
        "username": config.OWNER_USERNAME,
        "loaded_at": loaded_at,
    }
    person_rows = [
        {
            "id": person.person_id,
            "user_id": owner_id,
            "first_name": person.first_name,
            "last_name": person.last_name,
            "display_name": person.display_name,
            "last_seen_at": last_seen.get(person.person_id),
            "loaded_at": loaded_at,
        }
        for person in people
    ]
    episode_rows = [
        {
            "id": conversation.episode_id,
            "user_id": owner_id,
            "person_id": conversation.person_id,
            "start_time": conversation.start_time,
            "end_time": conversation.end_time,
            "transcript": conversation.dialogue_text(),
            "summary": conversation.summary,
            "summary_version": SUMMARY_VERSION,
            "loaded_at": loaded_at,
        }
        for conversation in conversations
    ]
    participant_rows = [
        {
            "episode_id": conversation.episode_id,
            "person_id": conversation.person_id,
            "loaded_at": loaded_at,
        }
        for conversation in conversations
    ]
    dialogue_rows = [
        {
            "episode_id": conversation.episode_id,
            "dialogue": json.dumps(
                [{"speaker": turn.speaker, "text": turn.text} for turn in conversation.dialogue],
                ensure_ascii=False,
            ),
        }
        for conversation in conversations
    ]
    embedding_rows = [
        {
            "episode_id": conversation.episode_id,
            "model": config.EMBEDDING_MODEL,
            "embedding": vector_literal(embeddings[conversation.episode_id]),
        }
        for conversation in conversations
    ]

    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute(_UPSERT_OWNER, owner_row)
        cursor.executemany(_UPSERT_PERSON, person_rows)
        cursor.executemany(_UPSERT_EPISODE, episode_rows)
        cursor.executemany(_UPSERT_PARTICIPANT, participant_rows)
        cursor.executemany(_UPSERT_DIALOGUE, dialogue_rows)
        cursor.executemany(_UPSERT_EMBEDDING, embedding_rows)
        cursor.execute(_UPSERT_CORPUS_META, meta_row)


def load_queries(
    conn: psycopg.Connection[Any],
    queries: Sequence[Query],
    owner_id: str = config.OWNER_ID,
) -> None:
    """Make ``eval_queries`` hold exactly ``queries`` for ``owner_id``, in one transaction.

    The owner's earlier rows are deleted first, so the table always mirrors the
    file it was loaded from (label pooling rewrites ``gold_episode_ids``; load
    again afterwards). The owner row must exist: load the dataset first.
    """

    rows = [
        {
            "query_id": query.query_id,
            "owner_id": owner_id,
            "split": query.split,
            "type": query.type,
            "query": query.query,
            "gold_name": query.gold_name,
            "gold_person_id": query.gold_person_id,
            "gold_time_start": query.gold_time_range.start if query.gold_time_range else None,
            "gold_time_end": query.gold_time_range.end if query.gold_time_range else None,
            "gold_episode_ids": list(query.gold_episode_ids),
            "gold_answer": query.gold_answer,
            "meta": json.dumps(query.meta, ensure_ascii=False, sort_keys=True),
        }
        for query in queries
    ]
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute(_DELETE_QUERIES, {"owner_id": owner_id})
        cursor.executemany(_INSERT_QUERY, rows)


def dataset_fingerprint(people: Sequence[Person], conversations: Sequence[Conversation]) -> str:
    """A sha256 over what :func:`load_dataset` writes, the same for the same dataset on every machine.

    It covers each person's ids and names and each conversation's ids, times,
    summary and dialogue (in id order, so the files' row order does not
    matter), plus the embedding model the rows are searched with. Anything a
    store serves from the database changes it; the records' ``meta`` does not,
    since none of it is loaded.
    """

    payload = {
        "embedding_model": config.EMBEDDING_MODEL,
        "summary_version": SUMMARY_VERSION,
        "people": [
            [person.person_id, person.first_name, person.last_name, person.display_name]
            for person in sorted(people, key=lambda person: person.person_id)
        ],
        "conversations": [
            [
                conversation.episode_id,
                conversation.person_id,
                parse_dt(conversation.start_time).isoformat(),
                parse_dt(conversation.end_time).isoformat(),
                conversation.summary,
                [[turn.speaker, turn.text] for turn in conversation.dialogue],
            ]
            for conversation in sorted(conversations, key=lambda conversation: conversation.episode_id)
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def corpus_fingerprint(conn: psycopg.Connection[Any], owner_id: str = config.OWNER_ID) -> str | None:
    """The fingerprint the last load wrote for ``owner_id``, or None when nothing records one.

    None also when the ``eval_corpus_meta`` table does not exist yet: a
    database loaded before the fingerprint was introduced has no record of
    what it holds, and must be loaded again to get one.
    """

    try:
        row = conn.execute(SELECT_CORPUS_META, {"owner_id": owner_id}).fetchone()
    except psycopg.errors.UndefinedTable:
        return None
    return None if row is None else str(row[0])


class StaleCorpusError(RuntimeError):
    """The database does not hold the dataset the run reads from the JSONL files."""


def check_corpus(
    conn: psycopg.Connection[Any],
    people: Sequence[Person],
    conversations: Sequence[Conversation],
    owner_id: str = config.OWNER_ID,
) -> str:
    """Fail unless the database holds exactly this dataset; return the fingerprint they share.

    The ids in the dataset are stable, so a regenerated ``conversations.jsonl``
    that was never loaded leaves every gold id resolving against old summaries
    and dialogues: a run over the pg store must call this before ranking.
    """

    return check_fingerprint(corpus_fingerprint(conn, owner_id), dataset_fingerprint(people, conversations))


def check_fingerprint(stored: str | None, expected: str) -> str:
    """Fail unless ``stored`` (what the database recorded) is ``expected`` (the dataset's); return it."""

    if stored != expected:
        held = "no fingerprint" if stored is None else f"fingerprint {stored}"
        raise StaleCorpusError(
            f"the eval database does not hold the committed dataset ({held} in the database, "
            f"the dataset's is {expected}); load it with: {LOAD_COMMAND}"
        )
    return expected


def count_rows(conn: psycopg.Connection[Any], owner_id: str = config.OWNER_ID) -> dict[str, int]:
    """Rows the database holds for ``owner_id``, per table in :data:`COUNTED_TABLES`.

    Embeddings are counted for ``config.EMBEDDING_MODEL`` only, the model the
    stores search with.
    """

    params = {"owner_id": owner_id, "model": config.EMBEDDING_MODEL}
    row = conn.execute(_COUNT_ROWS, params).fetchone()
    return dict(zip(COUNTED_TABLES, (int(count) for count in row)))


def _check_embeddings(
    conversations: Sequence[Conversation],
    embeddings: Mapping[str, Sequence[float]],
) -> None:
    """Fail before the first statement if a conversation lacks a usable embedding."""

    missing = [c.episode_id for c in conversations if c.episode_id not in embeddings]
    if missing:
        raise ValueError(
            f"no embedding for {len(missing)} of {len(conversations)} conversations "
            f"(first: {missing[0]})"
        )
    for conversation in conversations:
        dimension = len(embeddings[conversation.episode_id])
        if dimension != config.EMBEDDING_DIM:
            raise ValueError(
                f"the embedding of {conversation.episode_id} has {dimension} dimensions, "
                f"expected {config.EMBEDDING_DIM}"
            )
