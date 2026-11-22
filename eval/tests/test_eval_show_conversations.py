"""Unit tests for the Markdown reading copy of the corpus."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from eval.retrieval import config
from eval.retrieval.data import show_conversations as sheets
from eval.retrieval.schema import OWNER_SPEAKER, Conversation, Person, Turn

SAM, ANA, LEE = (config.stable_id("person", index) for index in range(3))
PEOPLE = [
    Person(person_id=SAM, first_name="Sam", last_name="Okafor", display_name="Sam Okafor", relationship="colleague"),
    Person(person_id=ANA, first_name="Ana", last_name="Lopez", display_name="Ana Lopez"),
    Person(person_id=LEE, first_name="Lee", last_name="Park", display_name="Lee Park"),  # no conversations
]
MORNING = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)


def _conversation(index: int, person_id: str, start_time: datetime, **meta: object) -> Conversation:
    return Conversation(
        episode_id=config.stable_id("episode", index),
        person_id=person_id,
        start_time=start_time,
        end_time=start_time + timedelta(minutes=20),
        dialogue=[Turn("Sam", f'Café "Luna" at {index}?'), Turn(OWNER_SPEAKER, "See you there.")],
        summary=f"Plans for meeting number {index}.",
        meta=dict(meta),
    )


CONVERSATIONS = [
    _conversation(0, SAM, MORNING + timedelta(days=2), storyline_title="Coffee plans", beat_index=1),
    _conversation(
        1,
        SAM,
        MORNING,
        storyline_title="Coffee plans",
        beat_index=0,
        key_facts=["They meet at Luna.", "Sam pays."],
        minor_details=["It rained."],
        unstated_details=["It rained."],
    ),
    _conversation(2, ANA, MORNING - timedelta(days=3)),
]


def test_a_person_sheet_lists_the_conversations_oldest_first_with_everything_to_read() -> None:
    sheet = sheets.render_person(PEOPLE[0], CONVERSATIONS[:2])

    assert sheet.startswith("# Sam Okafor\n\n2 conversations. Relationship: colleague. Person id `")
    older, newer = sheet.index("## 2026-10-10 09:00 UTC"), sheet.index("## 2026-10-12 09:00 UTC")
    assert older < newer
    assert "## 2026-10-10 09:00 UTC · Coffee plans (part 1)" in sheet
    assert "## 2026-10-12 09:00 UTC · Coffee plans (part 2)" in sheet
    assert f"`{CONVERSATIONS[1].episode_id}`" in sheet
    assert "**Summary (5 words):** Plans for meeting number 1." in sheet
    assert "**Key facts:** They meet at Luna. | Sam pays." in sheet
    assert "**Minor details:** It rained." in sheet
    assert "**Flagged as not said in the dialogue:** It rained." in sheet
    assert 'Sam: "Café "Luna" at 1?"  \nMe: "See you there."  \n' in sheet
    # The other conversation has no facts recorded: no empty labels.
    assert sheet.count("**Key facts:**") == 1 and sheet.count("Flagged") == 1


def test_a_sheet_copes_with_a_conversation_that_records_no_storyline() -> None:
    sheet = sheets.render_person(PEOPLE[1], CONVERSATIONS[2:])

    assert "1 conversation. Relationship: not recorded." in sheet
    assert "## 2026-10-07 09:00 UTC\n" in sheet


def test_write_sheets_mirrors_the_dataset(tmp_path) -> None:
    folder = tmp_path / "conversations"
    folder.mkdir()
    (folder / "gone-person.md").write_text("left by an earlier run")

    written = sheets.write_sheets(PEOPLE, CONVERSATIONS, tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in written] == [
        "README.md",
        "conversations/sam-okafor.md",
        "conversations/ana-lopez.md",
    ]
    assert sorted(path.name for path in folder.iterdir()) == ["ana-lopez.md", "sam-okafor.md"]  # no Lee, no stale file
    index = (tmp_path / "README.md").read_text()
    assert "2 people and 3 conversations" in index
    assert "[queries.txt](queries.txt)" in index
    rows = [line for line in index.splitlines() if line.startswith("| [")]
    assert rows == [
        "| [Sam Okafor](conversations/sam-okafor.md) | colleague | 2 | 2026-10-10 | 2026-10-12 |",
        "| [Ana Lopez](conversations/ana-lopez.md) |  | 1 | 2026-10-07 | 2026-10-07 |",
    ]


def test_people_who_share_a_display_name_get_distinct_files() -> None:
    twin = Person(person_id=LEE, first_name="Sam", last_name="Okafor", display_name="Sam Okafor")

    names = sheets.file_names([PEOPLE[0], twin, PEOPLE[1]])

    assert names == {SAM: f"sam-okafor-{SAM[:8]}.md", LEE: f"sam-okafor-{LEE[:8]}.md", ANA: "ana-lopez.md"}


def test_cli_writes_the_committed_dataset_to_the_directory_asked_for(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(sheets.dataset, "load_people", lambda: PEOPLE)
    monkeypatch.setattr(sheets.dataset, "load_conversations", lambda: CONVERSATIONS)

    assert sheets.main(["--out-dir", str(tmp_path / "review")]) == 0

    assert (tmp_path / "review" / "conversations" / "sam-okafor.md").exists()
    assert "wrote 2 person files" in capsys.readouterr().out
