"""Data builders shared by the briefing service and HTTP tests."""
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.crud.memory_store import MemoryStore
from app.models.episode import Episode
from app.models.memory import EpisodeParticipant
from app.models.person import Person, PersonFact
from app.models.user import User

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fixed(n):
    """A stable, readable UUID so fixtures do not change between runs."""
    return UUID(f"00000000-0000-4000-8000-{n:012x}")


def make_store(url="sqlite+pysqlite:///:memory:", *, owner_id=None):
    store = MemoryStore(url, owner_user_id=owner_id or uuid4())
    store.initialize()
    add_user(store, store.owner_user_id)
    return store


def add_user(store, user_id):
    with store.Session.begin() as session:
        session.add(User(id=user_id, first_name="Test", last_name="Owner",
                         display_name="Test Owner", username=f"owner-{user_id}"))


def add_person(store, first, last="", *, person_id=None, owner_id=None, unnamed=False):
    """Add a person, by default with a new ID and the store's owner. Returns the ID."""
    person_id = person_id or uuid4()
    with store.Session.begin() as session:
        session.add(Person(
            id=person_id, user_id=owner_id or store.owner_user_id, first_name=first, last_name=last,
            display_name=f"Unknown person {person_id}" if unnamed else f"{first} {last}".strip(),
        ))
    return person_id


def add_facts(store, person_id, category, texts, *, id_base=None):
    """Add facts oldest first, so the last text ranks first. `id_base` gives them fixed IDs."""
    with store.Session.begin() as session:
        for i, text in enumerate(texts):
            fact_id = fixed(id_base + i) if id_base is not None else uuid4()
            session.add(PersonFact(id=fact_id, person_id=person_id, fact_text=text,
                                   fact_category=category, created_at=T0 + timedelta(minutes=i)))


def add_episode(store, person_id, start, summary, *, owner_id=None):
    with store.Session.begin() as session:
        episode = Episode(user_id=owner_id or store.owner_user_id, person_id=person_id,
                          start_time=start, dialogue_summary=summary)
        session.add(episode)
        session.flush()
        session.add(EpisodeParticipant(episode_id=episode.id, person_id=person_id))
