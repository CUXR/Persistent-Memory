"""Insert a small deterministic dataset through MemoryStore."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.database import Base
from app.crud.memory_store import MemoryStore
from app.models.user import User


def _get_or_create_seed_owner(db_url: str):
    """Return the first user's UUID, creating a seed user if none exists."""

    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with Session() as session:
        owner = session.scalar(select(User).order_by(User.created_at.asc()).limit(1))
        if owner is None:
            with session.begin():
                owner = User(
                    first_name="Seed",
                    last_name="Owner",
                    display_name="Seed Owner",
                    username="seed-owner",
                )
                session.add(owner)
                session.flush()
                print(f"Created seed owner user id={owner.id}")
        else:
            print(f"Using existing owner user id={owner.id}")
        owner_id = owner.id

    engine.dispose()
    return owner_id


def main() -> None:
    """Seed the configured database with a minimal memory dataset."""

    settings = get_settings()
    owner_id = _get_or_create_seed_owner(settings.database_url)

    store = MemoryStore(owner_user_id=owner_id)
    store.initialize(create_schema=False)

    try:
        emily = store.upsert_person(
            name="Emily Chen",
            face_key="face_emily_demo",
            voice_key="voice_emily_demo",
            persona90=[0.5] * 90,
        )
        john = store.upsert_person(
            name="John Rivera",
            face_key="face_john_demo",
        )

        now = datetime.now(timezone.utc)
        episode_id = store.write_episode(
            time_start=now - timedelta(minutes=10),
            time_end=now,
            transcript="Emily and John discussed weekend plans.",
            summary="Conversation about hobbies and weekend plans.",
            participants=[emily.id, john.id],
        )
        store.write_fact(emily.id, "likes swimming", confidence=0.9, episode_id=episode_id)
        store.write_pref(emily.id, "energy: high", confidence=0.8, episode_id=episode_id)
        store.write_summary(
            emily.id,
            "Emily discussed swimming and hiking plans.",
            episode_time_start=now - timedelta(minutes=10),
            episode_time_end=now,
            episode_id=episode_id,
        )
        store.write_edge(emily.id, "friend", john.id, confidence=0.85, episode_id=episode_id)

        profile = store.get_profile_context(emily.id)
        print(f"Seeded person: {emily.id}")
        print(f"Episode: {episode_id}")
        print(profile.model_dump_json(indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
