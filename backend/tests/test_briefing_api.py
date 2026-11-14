"""HTTP briefings: owner scoping, owner isolation, pagination, contract fixtures.

Contract fixtures in docs/contracts/briefings/ are checked against live responses.
After an intended contract change, regenerate them with:

    UPDATE_CONTRACT_FIXTURES=1 python -m pytest backend/tests/test_briefing_api.py
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
import pytest

from app.api.deps import get_session_factory
from app.crud.memory_store import MemoryStore
from app.main import app
from app.models.episode import Episode
from app.models.memory import EpisodeParticipant
from app.models.person import Person, PersonFact
from app.models.user import User

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "contracts" / "briefings"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fixed(n):
    """A stable, readable UUID so fixtures do not change between runs."""
    return UUID(f"00000000-0000-4000-8000-{n:012x}")


OWNER = fixed(0x1)
SAM, UNNAMED, ALEX_S, ALEX_J, PRIYA = (fixed(n) for n in (0xA1, 0xA2, 0xA3, 0xA4, 0xA5))


class Api:
    """Synchronous wrapper over httpx's ASGI transport (no live server)."""

    def __init__(self, owner_id=None):
        self.owner_id = owner_id

    def __call__(self, method, path, *, owner=..., **kwargs):
        owner = self.owner_id if owner is ... else owner
        headers = kwargs.pop("headers", {})
        if owner is not None:
            headers["X-Owner-Id"] = str(owner)

        async def send():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, headers=headers, **kwargs)

        return asyncio.run(send())


def add_user(store, user_id):
    with store.Session.begin() as session:
        session.add(User(id=user_id, first_name="Test", last_name="Owner",
                         display_name="Test Owner", username=f"owner-{user_id}"))


def add_person(store, person_id, first, last="", *, owner_id, unnamed=False):
    with store.Session.begin() as session:
        session.add(Person(
            id=person_id, user_id=owner_id, first_name=first, last_name=last,
            display_name=f"Unknown person {person_id}" if unnamed else f"{first} {last}".strip(),
        ))


def add_facts(store, person_id, category, texts, *, id_base=None):
    """Add facts oldest first, so the last text ranks first."""
    with store.Session.begin() as session:
        for i, text in enumerate(texts):
            fact_id = fixed(id_base + i) if id_base is not None else uuid4()
            session.add(PersonFact(id=fact_id, person_id=person_id, fact_text=text,
                                   fact_category=category, created_at=T0 + timedelta(minutes=i)))


@pytest.fixture
def store(tmp_path):
    # A file database: the app serves sync routes from a worker thread.
    s = MemoryStore(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite'}", owner_user_id=OWNER)
    s.initialize()
    add_user(s, OWNER)
    app.dependency_overrides[get_session_factory] = lambda: s.Session
    yield s
    app.dependency_overrides.clear()
    s.close()


@pytest.fixture
def api(store):
    return Api(OWNER)


@pytest.fixture
def scenario(store):
    """The deterministic data behind the contract fixtures."""
    add_person(store, SAM, "Sam", "Lee", owner_id=OWNER)
    add_facts(store, SAM, "hobby", ["Bakes sourdough", "Trail running", "Rock climbing"], id_base=0xF10)
    add_facts(store, SAM, "affiliation", ["Works at Acme Robotics"], id_base=0xF20)
    add_facts(store, SAM, "visual_descriptor", ["Curly dark hair", "Red rain jacket"], id_base=0xF30)
    with store.Session.begin() as session:
        episode = Episode(id=fixed(0xE1), user_id=OWNER, person_id=SAM,
                          start_time=T0 + timedelta(days=4, hours=15, minutes=30),
                          dialogue_summary="Sam described a climbing trip to the Gunks.")
        session.add(episode)
        session.flush()
        session.add(EpisodeParticipant(episode_id=episode.id, person_id=SAM))
    add_person(store, UNNAMED, "Unknown", owner_id=OWNER, unnamed=True)
    add_person(store, ALEX_S, "Alex", "Smith", owner_id=OWNER)
    add_facts(store, ALEX_S, "affiliation", ["Works at Acme Robotics"], id_base=0xF40)
    add_person(store, ALEX_J, "Alex", "Johnson", owner_id=OWNER)
    add_facts(store, ALEX_J, "visual_descriptor", ["Tall, glasses"], id_base=0xF50)
    add_person(store, PRIYA, "Priya", "Sharma", owner_id=OWNER)


# ── Owner scoping ────────────────────────────────────────────

ROUTES = [
    f"/people/{SAM}/briefing",
    "/people/briefing?name=Sam",
    f"/people/{SAM}/facts?category=hobby",
]


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("owner", [None, "", "not-a-uuid", uuid4()])
def test_every_route_needs_an_existing_owner(store, scenario, path, owner):
    response = Api(owner)("GET", path)
    assert response.status_code == 401
    assert response.json() == {"detail": "X-Owner-Id must name an existing user"}


# ── Routing ──────────────────────────────────────────────────


def test_name_lookup_and_id_brief_agree(api, scenario):
    by_name = api("GET", "/people/briefing?name=Sam").json()
    assert by_name["status"] == "found"
    assert by_name["brief"] == api("GET", f"/people/{SAM}/briefing").json()


def test_non_uuid_person_paths_do_not_exist(api, scenario):
    assert api("GET", "/people/sam/briefing").status_code == 404
    assert api("GET", "/people/sam/facts?category=hobby").status_code == 404


# ── Owner isolation ──────────────────────────────────────────


@pytest.fixture
def other(store, scenario):
    """A second owner with their own Sam."""
    other_owner, their_sam = uuid4(), uuid4()
    add_user(store, other_owner)
    add_person(store, their_sam, "Sam", "Lee", owner_id=other_owner)
    add_facts(store, their_sam, "hobby", ["Theirs 1", "Theirs 2", "Theirs 3"])
    return Api(other_owner), their_sam


def test_briefs_hide_other_owners_people(api, other):
    their_api, their_sam = other
    assert api("GET", f"/people/{their_sam}/briefing").status_code == 404
    assert their_api("GET", f"/people/{SAM}/briefing").status_code == 404
    # Indistinguishable from an ID that does not exist.
    missing = api("GET", f"/people/{uuid4()}/briefing")
    assert missing.json() == api("GET", f"/people/{their_sam}/briefing").json()


def test_name_lookup_stays_within_the_owner(api, other):
    their_api, their_sam = other
    assert api("GET", "/people/briefing?name=Sam").json()["brief"]["person_id"] == str(SAM)
    assert their_api("GET", "/people/briefing?name=Sam").json()["brief"]["person_id"] == str(their_sam)
    assert their_api("GET", "/people/briefing?name=Priya").json()["status"] == "not_found"


def test_facts_do_not_cross_owners(api, other):
    their_api, their_sam = other
    cursor = their_api("GET", f"/people/{their_sam}/briefing?max_facts=1").json()["hobbies"]["next_cursor"]
    path = f"/people/{their_sam}/facts?category=hobby&cursor={cursor}"
    assert their_api("GET", path).status_code == 200
    assert api("GET", path).status_code == 404
    assert api("GET", f"/people/{their_sam}/facts?category=hobby").status_code == 404


# ── Pagination ───────────────────────────────────────────────


def test_pages_walk_an_arbitrary_number_of_facts(store, api):
    add_person(store, SAM, "Sam", "Lee", owner_id=OWNER)
    add_facts(store, SAM, "hobby", [f"hobby {i}" for i in range(23)])
    page = api("GET", f"/people/{SAM}/briefing?max_facts=5").json()["hobbies"]
    seen = [f["text"] for f in page["items"]]
    while page["next_cursor"]:
        page = api("GET", f"/people/{SAM}/facts?category=hobby&cursor={page['next_cursor']}&limit=5").json()
        assert len(page["items"]) <= 5
        seen += [f["text"] for f in page["items"]]
    assert seen == [f"hobby {i}" for i in range(22, -1, -1)]


def test_cursor_must_match_person_and_category(api, scenario):
    cursor = api("GET", f"/people/{SAM}/briefing?max_facts=1").json()["hobbies"]["next_cursor"]
    assert api("GET", f"/people/{SAM}/facts?category=affiliation&cursor={cursor}").status_code == 400
    assert api("GET", f"/people/{PRIYA}/facts?category=hobby&cursor={cursor}").status_code == 400


@pytest.mark.parametrize("query", ["max_facts=0", "max_facts=51", "max_facts=two"])
def test_page_size_is_bounded(api, scenario, query):
    assert api("GET", f"/people/{SAM}/briefing?{query}").status_code == 422
    assert api("GET", f"/people/briefing?name=Sam&{query}").status_code == 422


@pytest.mark.parametrize("query", ["limit=0", "limit=51", "category=pets", ""])
def test_facts_query_is_validated(api, scenario, query):
    if "category" not in query:
        query = f"category=hobby&{query}" if query else ""
    assert api("GET", f"/people/{SAM}/facts?{query}").status_code == 422


def test_default_page_size_is_five(store, api):
    add_person(store, SAM, "Sam", "Lee", owner_id=OWNER)
    add_facts(store, SAM, "hobby", [f"hobby {i}" for i in range(8)])
    assert len(api("GET", f"/people/{SAM}/briefing").json()["hobbies"]["items"]) == 5
    assert len(api("GET", f"/people/{SAM}/facts?category=hobby").json()["items"]) == 5


def test_blank_name_is_rejected(api, scenario):
    assert api("GET", "/people/briefing?name=%20%20").status_code == 422
    assert api("GET", "/people/briefing").status_code == 422


# ── Contract fixtures ────────────────────────────────────────

# Each case: fixture name, description, path, and whether to send X-Owner-Id. Cursors are filled in from earlier responses.
CASES = [
    ("brief", "A known person, e.g. from face recognition: the first page of every category. "
     "The client decides what to display.", f"/people/{SAM}/briefing?max_facts=2", True),
    ("facts_next_page", "Continue Sam's hobbies from brief's hobbies.next_cursor. "
     "next_cursor is null on the last page.", f"/people/{SAM}/facts?category=hobby&cursor={{hobby_cursor}}&limit=2", True),
    ("facts_first_page", "A category's first page, without a cursor, e.g. for a person detail screen.",
     f"/people/{SAM}/facts?category=visual_descriptor&limit=1", True),
    ("brief_unnamed", "A face-created person nobody has named: display_name is null.",
     f"/people/{UNNAMED}/briefing", True),
    ("brief_no_memories", "A named person with no facts or conversations: empty pages.",
     f"/people/{PRIYA}/briefing", True),
    ("by_name_found", "Name lookup, one match: the same brief as /people/{id}/briefing.",
     "/people/briefing?name=Sam&max_facts=2", True),
    ("by_name_ambiguous", "Name lookup, several matches: candidates with hints capped at max_facts per "
     "category. After the wearer picks, call /people/{id}/briefing.", "/people/briefing?name=Alex", True),
    ("by_name_not_found", "Name lookup, no match. Unnamed people never match a name.",
     "/people/briefing?name=Zara", True),
    ("error_unknown_owner", "Missing or malformed X-Owner-Id, or one that names no user (any route).",
     f"/people/{SAM}/briefing", False),
    ("error_person_not_found", "Unknown person ID, or one belonging to another owner.",
     f"/people/{fixed(0xDEAD)}/briefing", True),
    ("error_invalid_cursor", "Malformed cursor, one from an older fact ordering, or one issued for a "
     "different person or category.", f"/people/{SAM}/facts?category=hobby&cursor=bm90LWEtY3Vyc29y", True),
]


def test_contract_fixtures_match_live_responses(api, scenario):
    update = os.environ.get("UPDATE_CONTRACT_FIXTURES") == "1"
    values, stale = {}, []
    for name, description, path, with_owner in CASES:
        path = path.format(**{k: quote(v) for k, v in values.items()})
        response = api("GET", path, owner=... if with_owner else None)
        if name == "brief":
            values["hobby_cursor"] = response.json()["hobbies"]["next_cursor"]
        request = {"method": "GET", "path": path}
        if with_owner:
            request["headers"] = {"X-Owner-Id": str(OWNER)}
        fixture = {"description": description, "request": request,
                   "response": {"status": response.status_code, "body": response.json()}}
        target = CONTRACTS / f"{name}.json"
        if update:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(fixture, indent=2) + "\n")
        elif not target.exists() or json.loads(target.read_text()) != fixture:
            stale.append(name)
    assert not stale, f"Contract fixtures differ from live responses: {stale}. See this module's docstring."
    listed = {p.stem for p in CONTRACTS.glob("*.json")}
    assert listed == {case[0] for case in CASES}, "Remove fixtures that no longer have a case"
