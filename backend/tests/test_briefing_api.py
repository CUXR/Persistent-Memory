"""HTTP briefings: bearer auth, owner isolation, pagination, display mode, contract fixtures.

Contract fixtures in docs/contracts/briefings/ are checked against live responses.
After an intended contract change, regenerate them with:

    UPDATE_CONTRACT_FIXTURES=1 python -m pytest backend/tests/test_briefing_api.py
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
import pytest

from app.api.deps import get_session_factory
from app.crud.api_tokens import issue_token, owner_for_token, revoke_token
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

    def __init__(self, token=None):
        self.token = token

    def __call__(self, method, path, *, token=..., **kwargs):
        token = self.token if token is ... else token
        headers = kwargs.pop("headers", {})
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"

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


def token_for(store, owner_id, label="test client"):
    with store.Session.begin() as session:
        return issue_token(session, owner_id, label)[1]


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
    return Api(token_for(store, OWNER))


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


# ── Authentication ───────────────────────────────────────────

ROUTES = [
    ("GET", f"/briefings/people/{SAM}"),
    ("GET", "/briefings/by-name?name=Sam"),
    ("GET", "/briefings/facts?cursor=abc"),
    ("GET", "/briefings/display-mode"),
    ("PUT", "/briefings/display-mode"),
]


@pytest.mark.parametrize("method,path", ROUTES)
def test_every_route_requires_a_bearer_token(store, scenario, method, path):
    response = Api()(method, path, json={"display_mode": "name"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("method,path", ROUTES)
def test_unknown_token_is_rejected(store, scenario, method, path):
    assert Api("pm_not-a-real-token")(method, path, json={"display_mode": "name"}).status_code == 401


def test_non_bearer_scheme_is_rejected(store, scenario, api):
    response = api("GET", f"/briefings/people/{SAM}", token=None,
                   headers={"Authorization": f"Basic {api.token}"})
    assert response.status_code == 401


def test_revoked_token_is_rejected(store, scenario):
    with store.Session.begin() as session:
        token, raw = issue_token(session, OWNER, "lost phone")
    assert Api(raw)("GET", f"/briefings/people/{SAM}").status_code == 200
    with store.Session.begin() as session:
        assert revoke_token(session, token.id)
    assert Api(raw)("GET", f"/briefings/people/{SAM}").status_code == 401


def test_only_the_token_hash_is_stored(store):
    with store.Session.begin() as session:
        token, raw = issue_token(session, OWNER, "glasses")
    assert raw.startswith("pm_") and raw not in token.token_hash
    with store.Session() as session:
        assert owner_for_token(session, raw) == OWNER


def test_tokens_require_an_existing_user(store):
    with store.Session.begin() as session, pytest.raises(ValueError):
        issue_token(session, uuid4(), "glasses")


def test_token_cli_issues_lists_and_revokes(store, scenario):
    env = {**os.environ, "DATABASE_URL": str(store._db_url)}

    def cli(*args):
        result = subprocess.run([sys.executable, "-m", "app.cli.tokens", *args], cwd=ROOT,
                                env=env, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    code, out = cli("create", "--owner-id", str(OWNER), "--label", "RayNeo glasses")
    assert code == 0, out
    fields = dict(line.split(":", 1) for line in out.strip().splitlines())
    token_id, raw = fields["token_id"].strip(), fields["token"].strip()
    assert Api(raw)("GET", f"/briefings/people/{SAM}").status_code == 200
    code, out = cli("list", "--owner-id", str(OWNER))
    assert token_id in out and "active" in out and "RayNeo glasses" in out
    assert cli("revoke", token_id)[0] == 0
    assert "revoked" in cli("list", "--owner-id", str(OWNER))[1]
    assert Api(raw)("GET", f"/briefings/people/{SAM}").status_code == 401
    assert cli("revoke", str(uuid4()))[0] != 0
    assert cli("create", "--owner-id", str(uuid4()), "--label", "x")[0] != 0


# ── Owner isolation ──────────────────────────────────────────


@pytest.fixture
def other(store, scenario):
    """A second owner with their own Sam."""
    other_owner, their_sam = uuid4(), uuid4()
    add_user(store, other_owner)
    add_person(store, their_sam, "Sam", "Lee", owner_id=other_owner)
    add_facts(store, their_sam, "hobby", ["Theirs 1", "Theirs 2", "Theirs 3"])
    return Api(token_for(store, other_owner)), their_sam


def test_face_path_hides_other_owners_people(api, other):
    their_api, their_sam = other
    assert api("GET", f"/briefings/people/{their_sam}").status_code == 404
    assert their_api("GET", f"/briefings/people/{SAM}").status_code == 404
    # Indistinguishable from an ID that does not exist.
    missing = api("GET", f"/briefings/people/{uuid4()}")
    assert missing.json() == api("GET", f"/briefings/people/{their_sam}").json()


def test_name_path_resolves_within_the_owner(api, other):
    their_api, their_sam = other
    assert api("GET", "/briefings/by-name?name=Sam").json()["brief"]["person_id"] == str(SAM)
    assert their_api("GET", "/briefings/by-name?name=Sam").json()["brief"]["person_id"] == str(their_sam)
    assert their_api("GET", "/briefings/by-name?name=Priya").json()["status"] == "not_found"


def test_cursor_does_not_cross_owners(api, other):
    their_api, their_sam = other
    cursor = their_api("GET", f"/briefings/people/{their_sam}?max_facts=1").json()["hobbies"]["next_cursor"]
    assert their_api("GET", f"/briefings/facts?cursor={cursor}").status_code == 200
    assert api("GET", f"/briefings/facts?cursor={cursor}").status_code == 404


def test_display_mode_is_per_owner(api, other):
    their_api, _ = other
    assert api("PUT", "/briefings/display-mode", json={"display_mode": "nothing"}).status_code == 200
    assert their_api("GET", "/briefings/display-mode").json() == {"display_mode": "name_and_facts"}


# ── Pagination ───────────────────────────────────────────────


def test_pages_walk_an_arbitrary_number_of_facts(store, api):
    add_person(store, SAM, "Sam", "Lee", owner_id=OWNER)
    add_facts(store, SAM, "hobby", [f"hobby {i}" for i in range(23)])
    page = api("GET", f"/briefings/people/{SAM}?max_facts=5").json()["hobbies"]
    seen = [f["text"] for f in page["items"]]
    while page["next_cursor"]:
        page = api("GET", f"/briefings/facts?cursor={page['next_cursor']}&limit=5").json()
        assert len(page["items"]) <= 5
        seen += [f["text"] for f in page["items"]]
    assert seen == [f"hobby {i}" for i in range(22, -1, -1)]


@pytest.mark.parametrize("query", ["max_facts=0", "max_facts=51", "max_facts=two"])
def test_page_size_is_bounded(api, scenario, query):
    assert api("GET", f"/briefings/people/{SAM}?{query}").status_code == 422
    assert api("GET", f"/briefings/by-name?name=Sam&{query}").status_code == 422


def test_default_page_size_is_five(store, api):
    add_person(store, SAM, "Sam", "Lee", owner_id=OWNER)
    add_facts(store, SAM, "hobby", [f"hobby {i}" for i in range(8)])
    assert len(api("GET", f"/briefings/people/{SAM}").json()["hobbies"]["items"]) == 5


def test_continuation_limit_is_bounded(api, scenario):
    cursor = api("GET", f"/briefings/people/{SAM}?max_facts=1").json()["hobbies"]["next_cursor"]
    assert api("GET", f"/briefings/facts?cursor={cursor}&limit=51").status_code == 422


def test_blank_name_is_rejected(api, scenario):
    assert api("GET", "/briefings/by-name?name=%20%20").status_code == 422
    assert api("GET", "/briefings/by-name").status_code == 422


# ── Display mode ─────────────────────────────────────────────


def test_display_mode_round_trips_and_drives_the_face_path(api, scenario):
    for mode, has_name, has_facts in [("nothing", False, False), ("name", True, False),
                                      ("name_and_facts", True, True)]:
        assert api("PUT", "/briefings/display-mode", json={"display_mode": mode}).json() == {"display_mode": mode}
        assert api("GET", "/briefings/display-mode").json() == {"display_mode": mode}
        brief = api("GET", f"/briefings/people/{SAM}").json()
        assert brief["display_mode"] == mode
        assert (brief["display_name"] is not None) == has_name
        assert (brief["hobbies"] is not None) == has_facts


def test_display_mode_does_not_limit_the_name_path(api, scenario):
    api("PUT", "/briefings/display-mode", json={"display_mode": "nothing"})
    brief = api("GET", "/briefings/by-name?name=Sam").json()["brief"]
    assert brief["display_name"] == "Sam Lee"
    assert brief["visual_descriptors"]["items"]


@pytest.mark.parametrize("body", [{"display_mode": "one_fact"}, {}, {"display_mode": None}])
def test_unknown_display_mode_is_rejected(api, scenario, body):
    assert api("PUT", "/briefings/display-mode", json=body).status_code == 422


# ── Contract fixtures ────────────────────────────────────────

# Each case: fixture name, description, request, and an optional display mode
# set (and then reset) around it. Cursors are filled in from earlier responses.
CASES = [
    ("face_name_and_facts", "Face path, mode name_and_facts: first page per category. "
     "No visual_descriptors on the face path.", "GET", f"/briefings/people/{SAM}?max_facts=2", None, None),
    ("facts_next_page", "Continue Sam's hobbies from face_name_and_facts's hobbies.next_cursor. "
     "next_cursor is null on the last page.", "GET", "/briefings/facts?cursor={hobby_cursor}&limit=2", None, None),
    ("face_name_only", "Face path with the owner's mode set to name: no facts or last interaction.",
     "GET", f"/briefings/people/{SAM}", None, "name"),
    ("face_nothing", "Face path with the owner's mode set to nothing: no name either.",
     "GET", f"/briefings/people/{SAM}", None, "nothing"),
    ("face_unnamed", "Face path for a face-created person nobody has named: display_name is null.",
     "GET", f"/briefings/people/{UNNAMED}", None, None),
    ("face_no_memories", "Face path for a named person with no facts or conversations: empty pages.",
     "GET", f"/briefings/people/{PRIYA}", None, None),
    ("by_name_found", "Name path, one match: full brief including visual_descriptors, regardless of mode.",
     "GET", "/briefings/by-name?name=Sam&max_facts=2", None, None),
    ("by_name_ambiguous", "Name path, several matches: candidates with hints capped at max_facts per category.",
     "GET", "/briefings/by-name?name=Alex", None, None),
    ("by_name_not_found", "Name path, no match. Unnamed people never match a name.",
     "GET", "/briefings/by-name?name=Zara", None, None),
    ("display_mode_get", "The owner's glasses display mode. Defaults to name_and_facts.",
     "GET", "/briefings/display-mode", None, None),
    ("display_mode_put", "Set the owner's glasses display mode: nothing, name, or name_and_facts.",
     "PUT", "/briefings/display-mode", {"display_mode": "name"}, None),
    ("error_unauthorized", "Missing, unknown, or revoked bearer token (any route).",
     "GET", f"/briefings/people/{SAM}", None, "no-token"),
    ("error_person_not_found", "Unknown person ID, or one belonging to another owner.",
     "GET", f"/briefings/people/{fixed(0xDEAD)}", None, None),
    ("error_invalid_cursor", "Malformed cursor, or one from an older fact ordering.",
     "GET", "/briefings/facts?cursor=bm90LWEtY3Vyc29y", None, None),
]


def test_contract_fixtures_match_live_responses(api, scenario):
    update = os.environ.get("UPDATE_CONTRACT_FIXTURES") == "1"
    values, stale = {}, []
    for name, description, method, path, body, mode in CASES:
        if mode not in (None, "no-token"):
            api("PUT", "/briefings/display-mode", json={"display_mode": mode})
        path = path.format(**{k: quote(v) for k, v in values.items()})
        response = api(method, path, json=body, token=None if mode == "no-token" else ...)
        if mode not in (None, "no-token"):
            api("PUT", "/briefings/display-mode", json={"display_mode": "name_and_facts"})
        if name == "face_name_and_facts":
            values["hobby_cursor"] = response.json()["hobbies"]["next_cursor"]
        request = {"method": method, "path": path}
        if mode != "no-token":
            request["headers"] = {"Authorization": "Bearer <token>"}
        if body is not None:
            request["body"] = body
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
