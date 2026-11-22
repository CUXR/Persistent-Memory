"""On-disk cache for hosted-API responses.

Every LLM and Jev call is stored under a key derived from its full request,
so an interrupted run resumes without paying twice and a re-run reproduces the
numbers of the run before it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config


def make_key(*parts: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of ``parts``.

    Canonical means sorted object keys, no insignificant whitespace and ASCII
    escapes, so requests that differ only in dict ordering share a key. Parts
    must be plain JSON data: anything else raises rather than hash a repr that
    changes between runs.
    """

    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass
class _KeyLock:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0  # threads holding or waiting for the lock


class DiskCache:
    """Thread-safe SQLite store of JSON values under string keys.

    One connection serves every thread, serialized by a lock. Each ``set``
    commits before it returns, so the file is complete after a crash and a
    later process (a resumed or repeated run) reads everything written so far.

    Without ``path`` the file is ``api_cache.sqlite`` under ``config.CACHE_DIR``,
    resolved when the cache is built rather than at import, so a test that
    redirects the cache directory never touches the real cache.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else config.CACHE_DIR / "api_cache.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(
            self.path,
            timeout=30.0,  # wait, rather than fail, if another process is mid-write
            isolation_level=None,  # autocommit: every statement is its own transaction
            check_same_thread=False,
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._key_guard = threading.Lock()
        self._key_locks: dict[str, _KeyLock] = {}

    def get(self, key: str) -> Any:
        """The value stored under ``key``, or None on a miss.

        A stored JSON null reads as None too: the clients store the payload
        objects they got, so to them None is always a miss.
        """

        with self._db_lock:
            row = self._db.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` (plain JSON data) under ``key``, replacing any earlier value."""

        encoded = json.dumps(value)
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO entries (key, value) VALUES (?, ?)", (key, encoded)
            )

    @contextmanager
    def lock(self, key: str) -> Iterator[None]:
        """Hold the in-process lock for ``key`` around "get, compute on a miss, set".

        Threads that want the same missing entry at the same moment then make
        one hosted call between them: the rest wait and read what the first
        one stored. Without it each would pay for its own response, and since
        only the last one written survives, a re-run would not reproduce what
        the others saw. Different keys never wait for each other.
        """

        with self._key_guard:
            entry = self._key_locks.setdefault(key, _KeyLock())
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._key_guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._key_locks[key]

    def close(self) -> None:
        """Close the connection; the cache cannot be used afterwards."""

        with self._db_lock:
            self._db.close()
