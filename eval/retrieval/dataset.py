"""Read and write the committed dataset files in ``eval/retrieval/data``."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import config
from .schema import Conversation, Person, Query


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_people(path: Path | None = None) -> list[Person]:
    return [Person.from_json(row) for row in read_jsonl(path or config.PEOPLE_PATH)]


def load_conversations(path: Path | None = None) -> list[Conversation]:
    return [Conversation.from_json(row) for row in read_jsonl(path or config.CONVERSATIONS_PATH)]


def load_queries(split: str | None = None, path: Path | None = None) -> list[Query]:
    """Load queries, optionally only one split (``"dev"`` or ``"test"``)."""

    queries = [Query.from_json(row) for row in read_jsonl(path or config.QUERIES_PATH)]
    if split is None:
        return queries
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    return [query for query in queries if query.split == split]


def save_people(people: Iterable[Person], path: Path | None = None) -> None:
    write_jsonl(path or config.PEOPLE_PATH, (person.to_json() for person in people))


def save_conversations(conversations: Iterable[Conversation], path: Path | None = None) -> None:
    write_jsonl(path or config.CONVERSATIONS_PATH, (c.to_json() for c in conversations))


def save_queries(queries: Iterable[Query], path: Path | None = None) -> None:
    write_jsonl(path or config.QUERIES_PATH, (query.to_json() for query in queries))
