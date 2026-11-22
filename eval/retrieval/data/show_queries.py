"""Print a plain-text review sheet of the queries, for a human sanity check.

    python -m eval.retrieval.data.show_queries [--split dev] [--type T] [--ids ID ...]
        [--out PATH] [--queries PATH]

The ticket uses the 30 dev queries to check that the queries make sense. Each
query is shown with its labels and the material behind them: the gold
conversations' date, person and summary; for a detail-only query also the
dialogue turns that carry the answer; for an unanswerable query the person's
storyline titles, which say what the corpus does cover about them. No model,
gateway or database is needed: the sheet reads the committed dataset files.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

from .. import config, dataset
from ..interfaces import CorpusStore
from ..runtime import Runtime
from ..schema import QUERY_TYPES, Conversation, Person, Query
from ..timeutil import describe_moment

MAX_LINES = 60  # per query: a sheet is read top to bottom, so no query may swamp it
MAX_TURNS = 6  # matching dialogue turns shown for a detail-only query
WIDTH = 110
MIN_WORD_LENGTH = 4  # shorter words of a gold answer ("the", "at") match every turn
_WORD = re.compile(r"[a-z0-9]+")
_INDENT = " " * 5


def answer_words(gold_answer: str | None) -> frozenset[str]:
    """The words of a gold answer worth matching a dialogue turn on: four letters or more."""

    return frozenset(word for word in _WORD.findall((gold_answer or "").lower()) if len(word) >= MIN_WORD_LENGTH)


def matching_turns(conversation: Conversation, gold_answer: str | None, limit: int = MAX_TURNS) -> list[str]:
    """The first ``limit`` dialogue turns that contain any word of the gold answer, rendered."""

    wanted = answer_words(gold_answer)
    matched: list[str] = []
    for turn in conversation.dialogue:
        if wanted & set(_WORD.findall(turn.text.lower())):
            matched.append(f'{turn.speaker}: "{turn.text}"')
            if len(matched) == limit:
                break
    return matched


def _wrapped(label: str, text: str) -> list[str]:
    """``label`` then ``text`` wrapped under it, continuation lines indented to the text."""

    return textwrap.wrap(f"{label}{text}", width=WIDTH, subsequent_indent=" " * len(label)) or [label.rstrip()]


def _time_range(query: Query) -> str:
    window = query.gold_time_range
    if window is None:
        return "-"
    return f"{describe_moment(window.start)} to {describe_moment(window.end)}"


def render_conversation(index: int, conversation: Conversation, query: Query) -> list[str]:
    """One gold conversation: date, person, id, summary, and for detail-only queries the turns."""

    person_name = conversation.meta.get("person_name") or conversation.person_id
    lines = [f"  {index}. {describe_moment(conversation.start_time)}, {person_name} [{conversation.episode_id}]"]
    lines.extend(_wrapped(_INDENT, conversation.summary))
    if query.type == "detail_only":
        turns = matching_turns(conversation, query.gold_answer)
        lines.append(f"{_INDENT}matching turns ({len(turns)}):")
        for turn in turns:
            lines.extend(_wrapped(_INDENT + "  ", turn))
    return lines


def render_storylines(person: Person | None) -> list[str]:
    """The storyline titles of the query's person: what the corpus does cover about them."""

    if person is None:
        return ["storylines: - (the query resolves to nobody)"]
    titles = [str(storyline.get("title", "")) for storyline in person.meta.get("storylines", [])]
    lines = [f"storylines of {person.display_name} ({len(titles)}):"]
    lines.extend(f"  - {title}" for title in titles)
    return lines


def render_query(query: Query, store: CorpusStore, people: dict[str, Person]) -> str:
    """The review block of one query, at most ``MAX_LINES`` lines."""

    person = people.get(query.gold_person_id or "")
    lines = [
        f"== {query.query_id}  [{query.type}/{query.meta.get('subkind', '-')}, {query.split}]",
        *_wrapped("query:        ", query.query),
        f"gold name:    {query.gold_name or '-'}  (person: {person.display_name if person else '-'})",
        *_wrapped("time range:   ", _time_range(query)),
        f"pool size:    {query.meta.get('pool_size', '-')}",
        *_wrapped("gold answer:  ", query.gold_answer or "-"),
    ]
    if query.type == "unanswerable":
        lines.extend(render_storylines(person))
    else:
        conversations = store.get_conversations(query.gold_episode_ids)
        lines.append(f"gold conversations ({len(query.gold_episode_ids)}):")
        for index, episode_id in enumerate(query.gold_episode_ids, start=1):
            conversation = conversations.get(episode_id)
            if conversation is None:
                lines.append(f"  {index}. [{episode_id}] is not in the corpus")
            else:
                lines.extend(render_conversation(index, conversation, query))
    if len(lines) > MAX_LINES:
        cut = len(lines) - (MAX_LINES - 1)
        lines = [*lines[: MAX_LINES - 1], f"  ... {cut} more lines; see the dataset files"]
    return "\n".join(lines)


def render(queries: Iterable[Query], store: CorpusStore, owner_id: str = config.OWNER_ID) -> str:
    """The whole sheet: one block per query, separated by a blank line."""

    people = {person.person_id: person for person in store.list_people(owner_id)}
    blocks = [render_query(query, store, people) for query in queries]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def select(
    queries: Sequence[Query], *, split: str | None, query_type: str | None, query_ids: Sequence[str]
) -> list[Query]:
    """The queries matching every filter given, in file order; unknown ids are an error."""

    known = {query.query_id for query in queries}
    unknown = [query_id for query_id in query_ids if query_id not in known]
    if unknown:
        raise ValueError(f"unknown query ids: {', '.join(unknown)}")
    wanted = set(query_ids)
    return [
        query
        for query in queries
        if (split is None or query.split == split)
        and (query_type is None or query.type == query_type)
        and (not wanted or query.query_id in wanted)
    ]


def _dataset_store() -> CorpusStore:
    """The committed corpus in memory, with no embeddings: the sheet never searches."""

    from ..store import InMemoryCorpusStore

    return InMemoryCorpusStore(dataset.load_people(), dataset.load_conversations())


def main(argv: Sequence[str] | None = None, runtime: Runtime | None = None) -> int:
    """Print the sheet (and write it to ``--out``). Exit 1 for an unknown query id."""

    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.data.show_queries",
        description="Print a plain-text review sheet of the benchmark queries.",
    )
    parser.add_argument("--split", choices=("dev", "test"), default=None, help="only this split (default: both)")
    parser.add_argument("--type", choices=QUERY_TYPES, default=None, help="only this query type")
    parser.add_argument("--ids", nargs="+", default=[], metavar="ID", help="only these query ids")
    parser.add_argument("--out", type=Path, default=None, help="also write the sheet to this file")
    parser.add_argument("--queries", type=Path, default=config.QUERIES_PATH, help="query file to read")
    args = parser.parse_args(argv)

    queries = dataset.load_queries(path=args.queries)
    try:
        chosen = select(queries, split=args.split, query_type=args.type, query_ids=args.ids)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    store = runtime.store if runtime is not None else _dataset_store()
    sheet = render(chosen, store)
    print(sheet, end="")
    print(f"({len(chosen)} of {len(queries)} queries shown)")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(sheet, encoding="utf-8")
        print(f"wrote the sheet to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
