"""Person briefings: by person ID, by name (resolver), and fact pages."""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.crud.briefing_queries import CURSOR_VERSION, FactCursor, InvalidCursorError, PersonNotFoundError
from app.models.person import PersonFact
from app.services.briefing import BriefingService
from briefing_helpers import T0, add_episode, add_facts, add_person, add_user, make_store

ROOT = Path(__file__).resolve().parents[2]


def texts(page):
    return [fact.text for fact in page.items]


@pytest.fixture
def store():
    s = make_store()
    yield s
    s.close()


@pytest.fixture
def service(store):
    return BriefingService(store)


@pytest.fixture
def sam(store):
    person_id = add_person(store, "Sam", "Lee")
    add_facts(store, person_id, "hobby", [f"hobby {i}" for i in range(7)])
    add_facts(store, person_id, "affiliation", [f"affiliation {i}" for i in range(7)])
    add_facts(store, person_id, "visual_descriptor", [f"visual {i}" for i in range(7)])
    return person_id


# ── By person ID ─────────────────────────────────────────────


def test_brief_caps_each_category_at_max_facts(service, sam):
    brief = service.brief(sam)
    assert len(brief.hobbies.items) == 5
    assert len(brief.affiliations.items) == 5
    small = service.brief(sam, max_facts=2)
    assert texts(small.hobbies) == ["hobby 6", "hobby 5"]
    assert texts(small.affiliations) == ["affiliation 6", "affiliation 5"]


def test_brief_includes_every_category(service, sam):
    # The client decides what to display, e.g. hiding looks for a person in view.
    brief = service.brief(sam, max_facts=1)
    assert [texts(brief.hobbies), texts(brief.affiliations), texts(brief.visual_descriptors)] == [
        ["hobby 6"], ["affiliation 6"], ["visual 6"]]


def test_brief_names_a_named_person(service, sam):
    brief = service.brief(sam)
    assert brief.person_id == sam
    assert brief.display_name == "Sam Lee"
    assert brief.is_unnamed is False


def test_unnamed_person_has_no_fake_name(store, service):
    person_id = add_person(store, "Unknown", unnamed=True)
    brief = service.brief(person_id)
    assert brief.is_unnamed is True
    assert brief.display_name is None
    assert texts(brief.hobbies) == texts(brief.affiliations) == texts(brief.visual_descriptors) == []
    assert brief.hobbies.next_cursor is None
    assert brief.last_interaction is None


def test_brief_includes_latest_interaction(store, service, sam):
    add_episode(store, sam, T0, "Talked about climbing.")
    add_episode(store, sam, T0 + timedelta(days=3), "Talked about her internship.")
    last = service.brief(sam).last_interaction
    assert last.summary == "Talked about her internship."
    assert last.date == T0 + timedelta(days=3)


def test_brief_raises_not_found_for_missing_person(service):
    with pytest.raises(PersonNotFoundError):
        service.brief(uuid4())


def test_brief_raises_not_found_for_other_owners_person(store, service):
    other_owner = uuid4()
    add_user(store, other_owner)
    theirs = add_person(store, "Sam", "Lee", owner_id=other_owner)
    with pytest.raises(PersonNotFoundError):
        service.brief(theirs)


@pytest.mark.parametrize("max_facts", [0, -1])
def test_max_facts_below_one_is_rejected(service, sam, max_facts):
    with pytest.raises(ValueError):
        service.brief(sam, max_facts=max_facts)
    with pytest.raises(ValueError):
        service.brief_by_name("Sam", max_facts=max_facts)
    with pytest.raises(ValueError):
        service.facts(sam, "hobby", limit=max_facts)


# ── By name ──────────────────────────────────────────────────


def test_name_match_returns_the_same_brief_as_the_id(store, service, sam):
    add_episode(store, sam, T0, "Talked about climbing.")
    result = service.brief_by_name("Sam", max_facts=3)
    assert result.status == "found"
    assert result.brief == service.brief(sam, max_facts=3)
    assert texts(result.brief.visual_descriptors) == ["visual 6", "visual 5", "visual 4"]


def test_identical_names_are_disambiguated_by_id(store, service):
    first, second = add_person(store, "Alex", "Smith"), add_person(store, "Alex", "Smith")
    add_facts(store, first, "visual_descriptor", ["Red hair"])
    add_facts(store, second, "visual_descriptor", ["Tall, glasses"])
    result = service.brief_by_name("Alex Smith")
    assert result.status == "ambiguous"
    assert {c.person_id for c in result.candidates} == {first, second}
    # The wearer picks one; the app then briefs that ID.
    assert texts(service.brief(second).visual_descriptors) == ["Tall, glasses"]


def test_name_path_returns_candidates_when_ambiguous(store, service):
    smith = add_person(store, "Alex", "Smith")
    johnson = add_person(store, "Alex", "Johnson")
    add_facts(store, smith, "affiliation", ["Works at Acme"])
    result = service.brief_by_name("Alex")
    assert result.status == "ambiguous"
    assert {c.person_id for c in result.candidates} == {smith, johnson}
    hints = {c.person_id: c.hints for c in result.candidates}
    assert hints[smith] == {"affiliation": ["Works at Acme"]}


def test_ambiguous_candidate_hints_are_capped_at_max_facts(store, service):
    smith = add_person(store, "Alex", "Smith")
    add_person(store, "Alex", "Johnson")
    add_facts(store, smith, "hobby", [f"hobby {i}" for i in range(7)])
    add_facts(store, smith, "affiliation", ["Works at Acme"])

    def smith_hints(**kwargs):
        result = service.brief_by_name("Alex", **kwargs)
        return next(c.hints for c in result.candidates if c.person_id == smith)

    assert smith_hints() == {
        "affiliation": ["Works at Acme"],
        "hobby": ["hobby 6", "hobby 5", "hobby 4", "hobby 3", "hobby 2"],
    }
    assert smith_hints(max_facts=2)["hobby"] == ["hobby 6", "hobby 5"]


def test_name_path_reports_unknown_name(service, sam):
    result = service.brief_by_name("  Zara  ")
    assert result.status == "not_found"
    assert result.message == "There is no one in your memory named Zara."


@pytest.mark.parametrize("query", ["Unknown", "Unknown person"])
def test_name_path_never_matches_unnamed_people(store, service, query):
    add_person(store, "Unknown", unnamed=True)
    add_person(store, "Unknown", unnamed=True)
    assert service.brief_by_name(query).status == "not_found"


def test_name_path_never_matches_unnamed_full_display_name(store, service):
    person_id = add_person(store, "Unknown", unnamed=True)
    assert service.brief_by_name(f"Unknown person {person_id}").status == "not_found"


def test_name_path_ignores_other_owners_people(store, service):
    other_owner = uuid4()
    add_user(store, other_owner)
    add_person(store, "Sam", "Lee", owner_id=other_owner)
    assert service.brief_by_name("Sam").status == "not_found"


# ── Pagination ───────────────────────────────────────────


def walk(service, person_id, first_page, limit, category="hobby"):
    seen, page = texts(first_page), first_page
    while page.next_cursor:
        page = service.facts(person_id, category, cursor=page.next_cursor, limit=limit)
        seen += texts(page)
    return seen


@pytest.mark.parametrize("limit", [1, 2, 3, 7, 50])
def test_pages_cover_every_fact_once_in_rank_order(service, sam, limit):
    first = service.brief(sam, max_facts=limit).hobbies
    assert walk(service, sam, first, limit) == [f"hobby {i}" for i in range(6, -1, -1)]


def test_facts_without_cursor_start_at_the_first_page(service, sam):
    page = service.facts(sam, "affiliation", limit=2)
    assert texts(page) == ["affiliation 6", "affiliation 5"]
    assert walk(service, sam, page, 2, "affiliation") == [f"affiliation {i}" for i in range(6, -1, -1)]


def test_pages_break_timestamp_ties_without_skips_or_repeats(store, service):
    person_id = add_person(store, "Tie")
    with store.Session.begin() as session:
        for i in range(9):
            session.add(PersonFact(person_id=person_id, fact_text=f"same {i}",
                                   fact_category="hobby", created_at=T0))
    first = service.brief(person_id, max_facts=2).hobbies
    assert sorted(walk(service, person_id, first, 2)) == sorted(f"same {i}" for i in range(9))


def test_last_page_has_no_cursor(service, sam):
    assert service.brief(sam, max_facts=7).hobbies.next_cursor is None
    assert service.brief(sam, max_facts=6).hobbies.next_cursor is not None


def test_facts_page_reports_person_and_category(service, sam):
    cursor = service.brief(sam, max_facts=2).visual_descriptors.next_cursor
    page = service.facts(sam, "visual_descriptor", cursor=cursor, limit=2)
    assert (page.person_id, page.category) == (sam, "visual_descriptor")
    assert texts(page) == ["visual 4", "visual 3"]


def test_cursor_for_another_category_or_person_is_rejected(store, service, sam):
    other = add_person(store, "Priya")
    cursor = service.brief(sam, max_facts=1).hobbies.next_cursor
    with pytest.raises(InvalidCursorError):
        service.facts(sam, "affiliation", cursor=cursor)
    with pytest.raises(InvalidCursorError):
        service.facts(other, "hobby", cursor=cursor)


def test_page_size_is_bounded(service, sam):
    with pytest.raises(ValueError):
        service.brief(sam, max_facts=51)
    with pytest.raises(ValueError):
        service.facts(sam, "hobby", limit=51)


@pytest.mark.parametrize("cursor", [
    "not base64!", "e30", "W10",  # garbage, {}, []
    FactCursor(uuid4(), "hobby", T0, uuid4()).encode().replace("A", "B", 1)[:-3],
])
def test_malformed_cursor_is_rejected(service, sam, cursor):
    with pytest.raises(InvalidCursorError):
        service.facts(sam, "hobby", cursor=cursor)


def test_cursor_from_other_ordering_version_is_rejected(service, sam, monkeypatch):
    cursor = service.brief(sam, max_facts=1).hobbies.next_cursor
    monkeypatch.setattr("app.crud.briefing_queries.CURSOR_VERSION", CURSOR_VERSION + 1)
    with pytest.raises(InvalidCursorError):
        service.facts(sam, "hobby", cursor=cursor)


# ── CLI ──────────────────────────────────────────────────────


def test_cli_prints_briefs_and_fact_pages(tmp_path, monkeypatch, capsys):
    from app.cli import briefing as cli

    url = f"sqlite+pysqlite:///{tmp_path / 'cli.sqlite'}"
    store = make_store(url)
    person_id = add_person(store, "Sam", "Lee")
    add_facts(store, person_id, "hobby", ["Climbs", "Runs"])
    add_facts(store, person_id, "visual_descriptor", ["Red jacket"])
    store.close()
    monkeypatch.setattr("app.crud.memory_store.settings", SimpleNamespace(database_url=url))
    owner = ["--owner-id", str(store.owner_user_id), "--max-facts", "1"]

    monkeypatch.setattr(sys, "argv", ["briefing", *owner, "brief", str(person_id)])
    cli.main()
    face = json.loads(capsys.readouterr().out)
    assert [f["text"] for f in face["hobbies"]["items"]] == ["Runs"]
    assert [f["text"] for f in face["visual_descriptors"]["items"]] == ["Red jacket"]

    monkeypatch.setattr(sys, "argv", ["briefing", *owner, "by-name", "Sam"])
    cli.main()
    named = json.loads(capsys.readouterr().out)
    assert named["status"] == "found"
    assert [f["text"] for f in named["brief"]["visual_descriptors"]["items"]] == ["Red jacket"]

    monkeypatch.setattr(sys, "argv", ["briefing", *owner, "facts", str(person_id), "hobby",
                                      "--cursor", face["hobbies"]["next_cursor"]])
    cli.main()
    assert [f["text"] for f in json.loads(capsys.readouterr().out)["items"]] == ["Climbs"]

    monkeypatch.setattr(sys, "argv", ["briefing", *owner, "brief", str(uuid4())])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "not found" in str(exc.value)


def test_cli_help_does_not_require_database_configuration():
    environment = {key: value for key, value in os.environ.items() if key != "DATABASE_URL"}
    result = subprocess.run(
        [sys.executable, "-m", "app.cli.briefing", "--help"], cwd=ROOT,
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "by-name" in result.stdout
