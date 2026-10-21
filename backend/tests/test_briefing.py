"""Person briefings: face path (person_id) and name path (resolver)."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.crud.briefing_queries import PersonNotFoundError
from app.crud.memory_store import MemoryStore
from app.models.episode import Episode
from app.models.memory import EpisodeParticipant
from app.models.person import Person, PersonFact
from app.models.user import User
from app.services.briefing import BriefingService

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_store(url="sqlite+pysqlite:///:memory:"):
    store = MemoryStore(url, owner_user_id=uuid4())
    store.initialize()
    add_user(store, store.owner_user_id)
    return store


def add_user(store, user_id):
    with store.Session.begin() as session:
        session.add(User(id=user_id, first_name="Test", last_name="Owner",
                         display_name="Test Owner", username=f"owner-{user_id}"))


def add_person(store, first, last="", *, owner_id=None, unnamed=False):
    person_id = uuid4()
    with store.Session.begin() as session:
        session.add(Person(
            id=person_id, user_id=owner_id or store.owner_user_id, first_name=first, last_name=last,
            display_name=f"Unknown person {person_id}" if unnamed else f"{first} {last}".strip(),
        ))
    return person_id


def add_facts(store, person_id, category, texts):
    """Add facts oldest first, so the last text is the newest."""
    with store.Session.begin() as session:
        for i, text in enumerate(texts):
            session.add(PersonFact(person_id=person_id, fact_text=text, fact_category=category,
                                   created_at=T0 + timedelta(minutes=i)))


def add_episode(store, person_id, start, summary, *, owner_id=None):
    with store.Session.begin() as session:
        episode = Episode(user_id=owner_id or store.owner_user_id, person_id=person_id,
                          start_time=start, dialogue_summary=summary)
        session.add(episode)
        session.flush()
        session.add(EpisodeParticipant(episode_id=episode.id, person_id=person_id))


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


# ── Face path ────────────────────────────────────────────────


def test_brief_caps_each_category_at_max_facts(service, sam):
    brief = service.brief(sam)
    assert len(brief.hobbies) == 5
    assert len(brief.affiliations) == 5
    small = service.brief(sam, max_facts=2)
    assert small.hobbies == ["hobby 6", "hobby 5"]
    assert small.affiliations == ["affiliation 6", "affiliation 5"]
    assert service.brief(sam, max_facts=0).hobbies == []


def test_brief_returns_facts_in_rank_order(service, sam):
    assert service.brief(sam).hobbies == ["hobby 6", "hobby 5", "hobby 4", "hobby 3", "hobby 2"]


def test_face_path_never_includes_visual_descriptors(service, sam):
    assert service.brief(sam).visual_descriptors == []


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
    assert brief.hobbies == brief.affiliations == []
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


def test_negative_max_facts_is_rejected(service, sam):
    with pytest.raises(ValueError):
        service.brief(sam, max_facts=-1)
    with pytest.raises(ValueError):
        service.brief_by_name("Sam", max_facts=-1)


# ── Name path ────────────────────────────────────────────────


def test_name_path_returns_brief_with_visual_descriptors(service, sam):
    result = service.brief_by_name("Sam", max_facts=3)
    assert result.status == "found"
    assert result.brief.person_id == sam
    assert result.brief.visual_descriptors == ["visual 6", "visual 5", "visual 4"]
    assert len(result.brief.hobbies) == len(result.brief.affiliations) == 3


def test_name_path_returns_candidates_when_ambiguous(store, service):
    smith = add_person(store, "Alex", "Smith")
    johnson = add_person(store, "Alex", "Johnson")
    add_facts(store, smith, "affiliation", ["Works at Acme"])
    result = service.brief_by_name("Alex")
    assert result.status == "ambiguous"
    assert {c.person_id for c in result.candidates} == {smith, johnson}
    hints = {c.person_id: c.hints for c in result.candidates}
    assert hints[smith] == {"affiliation": ["Works at Acme"]}


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


# ── CLI ──────────────────────────────────────────────────────


def test_cli_prints_both_paths(tmp_path, monkeypatch, capsys):
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
    assert face["hobbies"] == ["Runs"]
    assert face["visual_descriptors"] == []

    monkeypatch.setattr(sys, "argv", ["briefing", *owner, "by-name", "Sam"])
    cli.main()
    named = json.loads(capsys.readouterr().out)
    assert named["status"] == "found"
    assert named["brief"]["visual_descriptors"] == ["Red jacket"]

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
