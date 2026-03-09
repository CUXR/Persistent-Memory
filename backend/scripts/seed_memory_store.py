"""Insert a small deterministic dataset through MemoryStore."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.crud.memory_store import MemoryStore


def main() -> None:
    """Seed the configured database with a minimal memory dataset."""

    store = MemoryStore()
    store.initialize()

    try:
        emily = store.upsert_person(
            name="Emily Chen",
            aliases=["Em", "Dr. Chen"],
            face_key="face_emily_demo",
            voice_key="voice_emily_demo",
            persona90=[0.5] * 90,
        )
        john = store.upsert_person(
            name="John Rivera",
            aliases=["Johnny"],
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
