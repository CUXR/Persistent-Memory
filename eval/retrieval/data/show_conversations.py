"""Write a reading copy of the corpus: one Markdown file per person.

    python -m eval.retrieval.data.show_conversations [--out-dir DIR]

Each file lists that person's conversations oldest first, with the summary,
the key facts and minor details the dialogue was written from, and the full
dialogue. ``README.md`` in the directory indexes the people, most
conversations first. The sheets are derived from ``people.jsonl`` and
``conversations.jsonl`` and say nothing those files do not: they exist so that
a person can read the dataset. Rerun the command after regenerating the corpus.

The query sheet that sits next to them is written by
``python -m eval.retrieval.data.show_queries --out <DIR>/queries.txt``.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .. import config, dataset
from ..schema import Conversation, Person

REVIEW_DIR = config.DATA_DIR / "review"
CONVERSATIONS_SUBDIR = "conversations"
QUERY_SHEET_NAME = "queries.txt"


def slug(name: str) -> str:
    """A file name for a person: 'Sam Okafor' -> 'sam-okafor'."""

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "person"


def file_names(people: Sequence[Person]) -> dict[str, str]:
    """person id -> file name, with the id appended where two people share a display name."""

    slugs = [slug(person.display_name) for person in people]
    return {
        person.person_id: f"{name}.md" if slugs.count(name) == 1 else f"{name}-{person.person_id[:8]}.md"
        for person, name in zip(people, slugs)
    }


def render_person(person: Person, conversations: Sequence[Conversation]) -> str:
    """The Markdown sheet of one person's conversations, oldest first."""

    ordered = sorted(conversations, key=lambda conversation: conversation.start_time)
    lines = [f"# {person.display_name}", ""]
    count = f"{len(ordered)} conversation{'' if len(ordered) == 1 else 's'}"
    lines += [f"{count}. Relationship: {person.relationship or 'not recorded'}. Person id `{person.person_id}`.", ""]
    for conversation in ordered:
        meta = conversation.meta
        title = meta.get("storyline_title")
        heading = f"## {conversation.start_time:%Y-%m-%d %H:%M} UTC"
        if title:
            heading += f" · {title} (part {int(meta.get('beat_index', 0)) + 1})"
        lines += [heading, "", f"`{conversation.episode_id}`", ""]
        lines += [f"**Summary ({len(conversation.summary.split())} words):** {conversation.summary}", ""]
        for label, key in (("Key facts", "key_facts"), ("Minor details", "minor_details")):
            if meta.get(key):
                lines += [f"**{label}:** " + " | ".join(str(item) for item in meta[key]), ""]
        unstated = [str(item) for key in ("unstated_facts", "unstated_details") for item in meta.get(key) or ()]
        if unstated:
            lines += ["**Flagged as not said in the dialogue:** " + " | ".join(unstated), ""]
        # Two trailing spaces: a Markdown line break, so each turn keeps its own line.
        lines += [f'{turn.speaker}: "{turn.text}"  ' for turn in conversation.dialogue]
        lines += ["", "---", ""]
    return "\n".join(lines)


def render_index(people: Sequence[Person], by_person: dict[str, list[Conversation]], names: dict[str, str]) -> str:
    """The README of the review directory: every person with conversations, most first."""

    total = sum(len(conversations) for conversations in by_person.values())
    lines = [
        "# Synthetic corpus, for reading",
        "",
        f"{len(by_person)} people and {total} conversations, generated from `people.jsonl` and "
        "`conversations.jsonl` by `python -m eval.retrieval.data.show_conversations`. "
        "Everyone here is fictional. The owner of the assistant, Riley Morgan, speaks as \"Me\".",
        "",
        f"The benchmark queries, each with its gold answer and gold conversations, are in "
        f"[{QUERY_SHEET_NAME}]({QUERY_SHEET_NAME}).",
        "",
        "| Person | Relationship | Conversations | First | Last |",
        "|---|---|---|---|---|",
    ]
    listed = [person for person in people if by_person.get(person.person_id)]
    for person in sorted(listed, key=lambda person: (-len(by_person[person.person_id]), person.display_name)):
        conversations = sorted(by_person[person.person_id], key=lambda conversation: conversation.start_time)
        link = f"[{person.display_name}]({CONVERSATIONS_SUBDIR}/{names[person.person_id]})"
        lines.append(
            f"| {link} | {person.relationship or ''} | {len(conversations)} | "
            f"{conversations[0].start_time:%Y-%m-%d} | {conversations[-1].start_time:%Y-%m-%d} |"
        )
    return "\n".join(lines) + "\n"


def write_sheets(people: Sequence[Person], conversations: Sequence[Conversation], out_dir: Path) -> list[Path]:
    """Write the index and one file per person under ``out_dir``; return the files written.

    Person files left by an earlier run for people who are no longer in the
    corpus are removed, so the directory always mirrors the dataset.
    """

    by_person: dict[str, list[Conversation]] = defaultdict(list)
    for conversation in conversations:
        by_person[conversation.person_id].append(conversation)
    names = file_names(people)
    folder = out_dir / CONVERSATIONS_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for person in people:
        if by_person.get(person.person_id):
            path = folder / names[person.person_id]
            path.write_text(render_person(person, by_person[person.person_id]), encoding="utf-8")
            written.append(path)
    for stale in set(folder.glob("*.md")) - set(written):
        stale.unlink()
    index = out_dir / "README.md"
    index.write_text(render_index(people, by_person, names), encoding="utf-8")
    return [index, *written]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.data.show_conversations",
        description="Write a Markdown reading copy of the benchmark corpus, one file per person.",
    )
    parser.add_argument("--out-dir", type=Path, default=REVIEW_DIR, help="directory to write (default: data/review)")
    args = parser.parse_args(argv)
    written = write_sheets(dataset.load_people(), dataset.load_conversations(), args.out_dir)
    print(f"wrote {len(written) - 1} person files and {written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
