"""The two corpus stores: one owner's conversations, as rankers and the pipeline read them.

``InMemoryCorpusStore`` serves dataset records held in memory (unit tests,
offline runs); ``PgCorpusStore`` serves the same data from the Neon eval branch
(the real benchmark). Both honour ``interfaces.CorpusStore`` with the same
orderings: the pool newest first then by episode id, a search by cosine
descending then by episode id. Equal start times and equal cosines therefore
fall in the same order, and are cut at the same place, whichever store a ranker
reads from.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from functools import partial
from types import TracebackType
from typing import Any

import numpy as np
import psycopg

from . import config, dataset, db
from .interfaces import Embedder
from .local_models import load_or_compute_corpus_embeddings
from .schema import Conversation, ConversationHeader, Person, TimeRange, Turn, parse_dt


def _display_name(first_name: str, last_name: str, display_name: str | None) -> str:
    """The name a conversation is attributed to: the display name, else first and last name."""

    return display_name or f"{first_name} {last_name}".strip()


def _person_name(person: Person) -> str:
    """:func:`_display_name` of a dataset person."""

    return _display_name(person.first_name, person.last_name, person.display_name)


def _sorted_people(people: Iterable[Person]) -> list[Person]:
    """People by name, then id.

    Both stores sort here, in Python, so the order never depends on the
    database collation.
    """

    return sorted(people, key=lambda person: (_person_name(person), person.person_id))


def _check_limit(limit: int) -> None:
    """Reject a limit that Python would read as a slice from the end and Postgres would refuse."""

    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")


def _unit_rows(
    episode_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """The embeddings as L2-normalized matrix rows, and the row to read for each episode id.

    A zero vector stays zero: its cosine with anything is then 0, not NaN.
    Conversations with identical embeddings read the same row, so they score
    exactly alike and the episode id breaks the tie, as it does in Postgres;
    BLAS does not promise equal results for equal rows in different places.
    """

    missing = [episode_id for episode_id in episode_ids if episode_id not in embeddings]
    if missing:
        raise ValueError(
            f"no embedding for {len(missing)} of {len(episode_ids)} conversations "
            f"(first: {missing[0]})"
        )
    dimensions = {len(embeddings[episode_id]) for episode_id in episode_ids}
    if len(dimensions) > 1:
        raise ValueError(f"embeddings have mixed dimensions: {sorted(dimensions)}")
    matrix = np.zeros((len(episode_ids), dimensions.pop() if dimensions else 0))
    rows: dict[str, int] = {}
    first_row_of: dict[bytes, int] = {}
    for row, episode_id in enumerate(episode_ids):
        matrix[row] = embeddings[episode_id]
        rows[episode_id] = first_row_of.setdefault(matrix[row].tobytes(), row)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0), rows


class InMemoryCorpusStore:
    """``CorpusStore`` over dataset records held in memory.

    It holds a single owner's data: calls for any other ``owner_id`` find
    nothing. ``embeddings`` (episode id -> vector) is needed only by
    ``vector_search``.
    """

    def __init__(
        self,
        people: Sequence[Person],
        conversations: Sequence[Conversation],
        embeddings: Mapping[str, Sequence[float]] | None = None,
        owner_id: str = config.OWNER_ID,
    ) -> None:
        self._owner_id = owner_id
        self._people = _sorted_people(people)
        self._person_names = {person.person_id: _person_name(person) for person in self._people}
        self._conversations: dict[str, Conversation] = {}
        for conversation in conversations:
            if conversation.episode_id in self._conversations:
                raise ValueError(f"duplicate episode id: {conversation.episode_id}")
            if conversation.person_id not in self._person_names:
                raise ValueError(
                    f"conversation {conversation.episode_id} is with an unknown person: "
                    f"{conversation.person_id}"
                )
            self._conversations[conversation.episode_id] = conversation
        # Two stable sorts: newest first, equal start times in episode-id order.
        by_id = sorted(self._conversations.values(), key=lambda record: record.episode_id)
        self._newest_first = sorted(by_id, key=lambda record: record.start_time, reverse=True)
        self._unit_embeddings: np.ndarray | None = None
        self._rows: dict[str, int] = {}
        if embeddings is not None:
            self._unit_embeddings, self._rows = _unit_rows(list(self._conversations), embeddings)

    @classmethod
    def from_dataset(cls, embedder: Embedder | None = None) -> "InMemoryCorpusStore":
        """A store over the committed JSONL files.

        With an ``embedder`` the summaries are embedded (or read back from the
        on-disk cache), which enables ``vector_search``.
        """

        people = dataset.load_people()
        conversations = dataset.load_conversations()
        embeddings = None
        if embedder is not None:
            embeddings = load_or_compute_corpus_embeddings(conversations, embedder)
        return cls(people, conversations, embeddings)

    def corpus_fingerprint(self) -> str:
        """``db.dataset_fingerprint`` of the people and conversations this store holds."""

        return db.dataset_fingerprint(self._people, list(self._conversations.values()))

    def list_people(self, owner_id: str) -> list[Person]:
        return list(self._people) if owner_id == self._owner_id else []

    def pool_ids(
        self,
        owner_id: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[str]:
        if owner_id != self._owner_id:
            return []
        return [
            conversation.episode_id
            for conversation in self._newest_first
            if (person_id is None or conversation.person_id == person_id)
            and (time_range is None or time_range.contains(conversation.start_time))
        ]

    def vector_search(
        self,
        owner_id: str,
        query_embedding: Sequence[float],
        person_id: str | None = None,
        time_range: TimeRange | None = None,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        if self._unit_embeddings is None:
            raise RuntimeError(
                "this store has no embeddings, so it cannot search: build it with embeddings=... "
                "or InMemoryCorpusStore.from_dataset(embedder=...)"
            )
        _check_limit(limit)
        pool = self.pool_ids(owner_id, person_id=person_id, time_range=time_range)
        if not pool:
            return []
        query = np.asarray(query_embedding, dtype=float)
        dimension = self._unit_embeddings.shape[1]
        if query.shape != (dimension,):
            raise ValueError(
                f"the query embedding has shape {query.shape}, "
                f"the corpus embeddings have {dimension} dimensions"
            )
        norm = float(np.linalg.norm(query))
        # Scored against the whole corpus, then cut to the pool: a conversation's
        # score for a query is then the same number under every filter.
        cosines = self._unit_embeddings @ (query / norm if norm > 0.0 else query)
        hits = [(episode_id, float(cosines[self._rows[episode_id]])) for episode_id in pool]
        hits.sort(key=lambda hit: (-hit[1], hit[0]))
        return hits[:limit]

    def get_summaries(self, episode_ids: Sequence[str]) -> dict[str, str]:
        return {
            episode_id: self._conversations[episode_id].summary
            for episode_id in episode_ids
            if episode_id in self._conversations
        }

    def get_headers(self, episode_ids: Sequence[str]) -> dict[str, ConversationHeader]:
        headers: dict[str, ConversationHeader] = {}
        for episode_id in episode_ids:
            conversation = self._conversations.get(episode_id)
            if conversation is not None:
                headers[episode_id] = ConversationHeader(
                    episode_id=episode_id,
                    person_id=conversation.person_id,
                    person_name=self._person_names[conversation.person_id],
                    start_time=conversation.start_time,
                    summary=conversation.summary,
                )
        return headers

    def get_conversations(self, episode_ids: Sequence[str]) -> dict[str, Conversation]:
        """Full records, each a copy with ``meta['person_name']`` filled in.

        Copies, so that neither the added name nor anything a caller does to a
        record reaches the dataset records this store was built from.
        """

        found: dict[str, Conversation] = {}
        for episode_id in episode_ids:
            conversation = self._conversations.get(episode_id)
            if conversation is not None:
                person_name = self._person_names[conversation.person_id]
                found[episode_id] = replace(
                    conversation,
                    dialogue=list(conversation.dialogue),
                    meta={**conversation.meta, "person_name": person_name},
                )
        return found


_PEOPLE_SQL = """
    SELECT p.id, p.first_name, p.last_name, p.display_name
    FROM people p
    WHERE p.user_id = %(owner_id)s::uuid
"""

_POOL_SQL = """
    SELECT e.id
    FROM episodes e
    WHERE {filters}
    ORDER BY e.start_time DESC, e.id
"""

# Exact nearest neighbours: the pool is filtered and small, so there is no ANN
# index to go through. ``<=>`` is pgvector's cosine distance.
_SEARCH_SQL = """
    SELECT e.id, m.embedding <=> %(query)s::vector AS distance
    FROM episodes e
    JOIN eval_episode_embeddings m ON m.episode_id = e.id AND m.model = %(model)s
    WHERE {filters}
    ORDER BY distance, e.id
    LIMIT %(limit)s
"""

_SUMMARIES_SQL = """
    SELECT e.id, e.dialogue_summary
    FROM episodes e
    WHERE e.id = ANY(%(episode_ids)s::uuid[])
"""

_HEADERS_SQL = """
    SELECT e.id, e.person_id, p.first_name, p.last_name, p.display_name, e.start_time,
           e.dialogue_summary
    FROM episodes e
    JOIN people p ON p.id = e.person_id
    WHERE e.id = ANY(%(episode_ids)s::uuid[])
"""

_CONVERSATIONS_SQL = """
    SELECT e.id, e.person_id, p.first_name, p.last_name, p.display_name, e.start_time,
           e.end_time, e.dialogue_summary, d.dialogue
    FROM episodes e
    JOIN people p ON p.id = e.person_id
    JOIN eval_episode_dialogues d ON d.episode_id = e.id
    WHERE e.id = ANY(%(episode_ids)s::uuid[])
"""


def _pool_filters(
    owner_id: str,
    person_id: str | None,
    time_range: TimeRange | None,
) -> tuple[str, dict[str, Any]]:
    """The WHERE clause that selects a filtered pool from ``episodes e``, and its parameters.

    The clause is assembled from fixed fragments; every value travels as a parameter.
    """

    clauses = ["e.user_id = %(owner_id)s::uuid"]
    params: dict[str, Any] = {"owner_id": owner_id}
    if person_id is not None:
        clauses.append("e.person_id = %(person_id)s::uuid")
        params["person_id"] = person_id
    if time_range is not None:
        clauses.append("e.start_time >= %(start)s AND e.start_time <= %(end)s")
        params["start"] = time_range.start
        params["end"] = time_range.end
    return " AND ".join(clauses), params


class PgCorpusStore:
    """``CorpusStore`` over the Neon eval branch, in plain SQL on psycopg 3.

    It reads the migrated ``people`` and ``episodes`` tables and the two
    eval-owned tables (see ``db.py``), so records carry what those hold: a
    person has ids and names, a conversation has no ``meta`` beyond the
    person's name. One autocommit connection is opened on first use and shared
    by every call; the store may be used from several threads.
    """

    def __init__(
        self,
        database_url: str | None = None,
        connection_factory: Callable[[], psycopg.Connection[Any]] | None = None,
    ) -> None:
        if connection_factory is None:
            # Resolved now, so a missing EVAL_DATABASE_URL fails here and not mid-run.
            connection_factory = partial(db.connect, db.resolve_url(database_url))
        self._connect = connection_factory
        self._connection: psycopg.Connection[Any] | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "PgCorpusStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection, if one is open. A later call opens a new one."""

        with self._lock:
            self._drop_connection()

    def corpus_fingerprint(self, owner_id: str = config.OWNER_ID) -> str | None:
        """The fingerprint ``load_db`` recorded for the owner's corpus (``db.corpus_fingerprint``).

        None when no load recorded one, the table included: such a database
        must be loaded again before a run can trust it.
        """

        try:
            rows = self._fetch(db.SELECT_CORPUS_META, {"owner_id": owner_id})
        except psycopg.errors.UndefinedTable:
            return None
        return str(rows[0][0]) if rows else None

    def list_people(self, owner_id: str) -> list[Person]:
        rows = self._fetch(_PEOPLE_SQL, {"owner_id": owner_id})
        return _sorted_people(
            Person(
                person_id=str(person_id),
                first_name=first_name,
                last_name=last_name,
                display_name=_display_name(first_name, last_name, display_name),
            )
            for person_id, first_name, last_name, display_name in rows
        )

    def pool_ids(
        self,
        owner_id: str,
        person_id: str | None = None,
        time_range: TimeRange | None = None,
    ) -> list[str]:
        filters, params = _pool_filters(owner_id, person_id, time_range)
        rows = self._fetch(_POOL_SQL.format(filters=filters), params)
        return [str(episode_id) for (episode_id,) in rows]

    def vector_search(
        self,
        owner_id: str,
        query_embedding: Sequence[float],
        person_id: str | None = None,
        time_range: TimeRange | None = None,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        _check_limit(limit)
        filters, params = _pool_filters(owner_id, person_id, time_range)
        params.update(
            query=db.vector_literal(query_embedding),
            model=config.EMBEDDING_MODEL,
            limit=limit,
        )
        rows = self._fetch(_SEARCH_SQL.format(filters=filters), params)
        return [(str(episode_id), 1.0 - float(distance)) for episode_id, distance in rows]

    def get_summaries(self, episode_ids: Sequence[str]) -> dict[str, str]:
        rows = self._fetch_episodes(_SUMMARIES_SQL, episode_ids)
        return {str(episode_id): summary for episode_id, summary in rows}

    def get_headers(self, episode_ids: Sequence[str]) -> dict[str, ConversationHeader]:
        headers: dict[str, ConversationHeader] = {}
        for row in self._fetch_episodes(_HEADERS_SQL, episode_ids):
            episode_id, person_id, first_name, last_name, display_name, start_time, summary = row
            headers[str(episode_id)] = ConversationHeader(
                episode_id=str(episode_id),
                person_id=str(person_id),
                person_name=_display_name(first_name, last_name, display_name),
                start_time=parse_dt(start_time),
                summary=summary,
            )
        return headers

    def get_conversations(self, episode_ids: Sequence[str]) -> dict[str, Conversation]:
        conversations: dict[str, Conversation] = {}
        for row in self._fetch_episodes(_CONVERSATIONS_SQL, episode_ids):
            (
                episode_id,
                person_id,
                first_name,
                last_name,
                display_name,
                start_time,
                end_time,
                summary,
                dialogue,
            ) = row
            conversations[str(episode_id)] = Conversation(
                episode_id=str(episode_id),
                person_id=str(person_id),
                start_time=parse_dt(start_time),
                end_time=parse_dt(end_time or start_time),  # the column is nullable
                dialogue=[Turn(speaker=turn["speaker"], text=turn["text"]) for turn in dialogue],
                summary=summary,
                meta={"person_name": _display_name(first_name, last_name, display_name)},
            )
        return conversations

    def _fetch_episodes(self, sql: str, episode_ids: Sequence[str]) -> list[tuple[Any, ...]]:
        """Rows of ``sql`` for the given episodes; no round trip when there are none."""

        if not episode_ids:
            return []
        return self._fetch(sql, {"episode_ids": list(episode_ids)})

    def _fetch(self, sql: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """Run one read-only statement and return all of its rows.

        Neon suspends an idle compute and drops its connections. A statement
        that fails with ``OperationalError`` is therefore retried once on a new
        connection; every statement here is a read, so repeating it is safe.
        """

        with self._lock:
            try:
                return self._execute(sql, params)
            except psycopg.OperationalError:
                self._drop_connection()
                return self._execute(sql, params)

    def _execute(self, sql: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """One attempt at a statement, on the shared connection (opened here if there is none)."""

        if self._connection is None:
            self._connection = self._connect()
        return self._connection.execute(sql, params).fetchall()

    def _drop_connection(self) -> None:
        """Forget the shared connection, closing it as far as its state allows."""

        connection, self._connection = self._connection, None
        if connection is not None:
            with suppress(psycopg.Error):
                connection.close()
