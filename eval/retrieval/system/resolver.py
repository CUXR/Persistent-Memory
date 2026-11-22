"""Step 2 of the query flow: match the extracted name to one of the owner's people.

The matching itself is the backend's ``PersonResolver`` (display-name pass,
then first-name pass, 0/1/N branching), so the benchmark measures the resolver
that ships. This module only adapts the benchmark's ``Person`` records to the
two store methods that resolver reads.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from .. import config
from ..schema import Person

ResolutionStatus = Literal["resolved", "ambiguous", "unknown"]


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a name: one person, several candidates, or nobody."""

    status: ResolutionStatus
    person_id: str | None  # set only when resolved
    candidates: list[Person]  # the matches when ambiguous (ordered by display name), else empty


@dataclass(frozen=True)
class _ResolverPerson:
    """The attributes ``PersonResolver`` reads from a backend ``Person`` row."""

    id: uuid.UUID
    first_name: str
    last_name: str
    display_name: str


class _PeopleAdapter:
    """Stands in for the backend ``MemoryStore``: the two methods ``PersonResolver`` calls."""

    def __init__(self, rows: list[_ResolverPerson]) -> None:
        self._rows = rows

    def list_people(self) -> list[_ResolverPerson]:
        return list(self._rows)

    def get_disambiguation_hints(self, person_id: uuid.UUID) -> dict[str, list[str]]:
        """No hints: the benchmark stores no person facts, and returns every candidate as is."""

        return {}


@lru_cache(maxsize=1)
def load_person_resolver() -> type[Any]:
    """Import the backend's ``PersonResolver`` class on first use.

    ``app.*`` is importable only with ``backend/`` on ``sys.path``, and its
    settings object refuses to load without a database URL and an OpenAI key.
    Nothing here connects to either, so placeholders are enough. Done lazily so
    that importing the eval package never pulls in the backend. The import
    takes a few seconds, so a caller that times queries calls this once up
    front.
    """

    backend_root = str(config.BACKEND_ROOT)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ.setdefault("OPENAI_API_KEY", "unused")

    from app.crud.person_resolver import PersonResolver

    return PersonResolver


def _resolver_id(person_id: str) -> uuid.UUID:
    """A UUID handle for a benchmark person id.

    ``PersonResolver`` needs UUID primary keys; deriving them keeps this module
    independent of how a store spells its ids. Results are mapped back to the
    original ``Person`` records, so the handle never leaves this module.
    """

    return uuid.uuid5(config.ID_NAMESPACE, person_id)


def resolve_name(name: str, people: Sequence[Person]) -> Resolution:
    """Resolve an extracted name against the owner's people with the backend resolver."""

    by_resolver_id = {_resolver_id(person.person_id): person for person in people}
    rows = [
        _ResolverPerson(
            id=resolver_id,
            first_name=person.first_name,
            last_name=person.last_name,
            display_name=person.display_name,
        )
        for resolver_id, person in by_resolver_id.items()
    ]
    resolver = load_person_resolver()(_PeopleAdapter(rows))
    result = resolver.resolve_person_from_query(name)
    if result.person_id is not None:
        return Resolution("resolved", by_resolver_id[result.person_id].person_id, [])
    if result.is_ambiguous:
        candidates = [by_resolver_id[candidate.person_id] for candidate in result.candidates]
        return Resolution("ambiguous", None, candidates)
    return Resolution("unknown", None, [])
