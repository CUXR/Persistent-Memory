"""Live smoke-test for the conversation ingestion pipeline.

Runs against the **real OpenAI API** using the key in your .env file.
Uses an in-memory SQLite database, so no Postgres setup is required.

Usage
-----
From the ``backend/`` directory::

    python scripts/live_ingestion.py

What it tests
-------------
Scenario 1 — New interlocutor
    Ingests a first conversation with "Jordan Lee".  Verifies that a person
    record is created and that facts / summary are written.

Scenario 2 — Repeat interlocutor (deduplication)
    Ingests a second conversation with the same "Jordan Lee".  Facts from the
    first conversation are passed to the LLM as context; newly extracted facts
    that duplicate already-stored ones should appear in
    ``facts_skipped_as_duplicate``.

Scenario 3 — Relationship edge extraction
    Pre-creates a second person ("Acme Corp") and ingests a conversation where
    Jordan mentions working there.  Verifies that an edge is written.

Inspect the printed output to judge whether the LLM's summaries and extracted
facts make sense — there are no hard assertions on content.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the backend package root is on sys.path when run directly
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.crud.memory_store import MemoryStore
from app.models.user import UserFact
from app.services.conversation_ingestion import ingest_conversation
from app.services.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Sample transcripts
# ---------------------------------------------------------------------------

TRANSCRIPT_1 = textwrap.dedent("""\
    Alex: Hey Jordan, good to finally catch up!
    Jordan: Likewise! I've been flat-out lately — just finished a triathlon last weekend.
    Alex: Wow, that's impressive. Do you do them regularly?
    Jordan: Yeah, I try to race twice a year. I also coach a junior swim team on Saturdays.
    Alex: That's a lot. What do you do for work these days?
    Jordan: I'm a senior data engineer at Acme Corp. Been there about three years now.
    Alex: Nice. Are you still into rock climbing too?
    Jordan: Absolutely — I go bouldering at the local gym every Tuesday.
    Alex: We should go sometime, I've just started climbing myself.
    Jordan: Definitely, let's plan it!
""")

TRANSCRIPT_2 = textwrap.dedent("""\
    Alex: Hey Jordan, how's training going?
    Jordan: Really well — I'm targeting the city triathlon in June.
    Alex: Nice. Still coaching the swim team?
    Jordan: Every Saturday without fail. We have a junior competition next month.
    Alex: How's work at Acme Corp treating you?
    Jordan: Busy but good. I just got promoted to lead engineer.
    Alex: Congratulations! By the way, do you know anyone in the data science space?
    Jordan: Actually yes — my colleague Priya Sharma is a data scientist there.
      She's great if you ever need an intro.
    Alex: That would be amazing, thanks!
""")

TRANSCRIPT_3 = textwrap.dedent("""\
    Alex: Jordan, I finally took your advice and signed up for a bouldering class.
    Jordan: That's great! Which gym?
    Alex: The one downtown you recommended.
    Jordan: Perfect — they have a really good beginner wall.
    Alex: I also checked out your company's open positions.
    Jordan: Oh nice! Acme Corp is hiring a few ML engineers right now actually.
    Alex: I might apply. Do you enjoy the culture there?
    Jordan: I love it. Very flexible, good benefits, strong engineering culture.
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 70


def _banner(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _section(label: str) -> None:
    print(f"\n  ── {label}")


def _print_result(result) -> None:
    _section("IngestionResult")
    print(f"    episode_id  : {result.episode_id}")
    print(f"    person_id   : {result.person_id}")
    print(f"    importance  : {result.importance_score:.2f}")
    print(f"    summary     :\n{textwrap.indent(result.summary, '      ')}")

    _section(f"Facts written ({len(result.facts_written)})")
    for uid in result.facts_written:
        print(f"    + {uid}")

    if result.facts_skipped_as_duplicate:
        _section(f"Facts skipped as duplicate ({len(result.facts_skipped_as_duplicate)})")
        for text in result.facts_skipped_as_duplicate:
            print(f"    ~ {text!r}")

    if result.edges_written:
        _section(f"Edges written ({len(result.edges_written)})")
        for uid in result.edges_written:
            print(f"    → {uid}")


def _print_profile(store: MemoryStore, person_id, label: str) -> None:
    profile = store.get_profile_context(person_id)
    _section(f"Profile context after {label}")
    print(f"    facts ({len(profile.facts)}):")
    for f in profile.facts:
        print(f"      [{f.confidence:.2f}] {f.fact_text}")
    print(f"    summaries ({len(profile.summaries)}):")
    for s in profile.summaries:
        print(f"      {s.summary_text[:120]}")
    print(f"    edges_from ({len(profile.edges_from)}):")
    for e in profile.edges_from:
        print(f"      -{e.relation}-> {e.dst_name} [{e.confidence:.2f}]")


def _seed_user_facts(store: MemoryStore, facts: list[str]) -> None:
    """Add wearer facts so the LLM can detect shared interests."""
    store.upsert_person(name="__owner_init__", aliases=[])  # trigger owner creation
    with store.Session() as session:
        with session.begin():
            from sqlalchemy import select
            from app.models.user import User
            owner = session.scalar(select(User).limit(1))
            for text in facts:
                session.add(UserFact(user_id=owner.id, fact_text=text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    settings = get_settings()

    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-..."):
        print("ERROR: OPENAI_API_KEY is not set in your .env file.")
        print("       Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    print(f"\nUsing model : {settings.openai_model}")
    print(f"Max retries : {settings.openai_max_retries}")

    # Shared in-memory store and LLM client for all scenarios
    store = MemoryStore("sqlite+pysqlite:///:memory:")
    store.initialize()
    client = LLMClient()

    # Seed wearer facts so shared-interest detection has material to work with
    _seed_user_facts(store, [
        "Enjoys rock climbing and bouldering",
        "Interested in data engineering and machine learning",
        "Likes endurance sports",
    ])

    now = datetime.now(timezone.utc)

    try:
        # ------------------------------------------------------------------ #
        # Scenario 1 — New interlocutor                                       #
        # ------------------------------------------------------------------ #
        _banner("SCENARIO 1 — New interlocutor: Jordan Lee")
        print("  Ingesting first conversation.  Expect: new person created,")
        print("  facts extracted (triathlon, swimming, climbing, Acme Corp, ...)")

        result1 = await ingest_conversation(
            transcript=TRANSCRIPT_1,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            time_start=now - timedelta(days=7),
            time_end=now - timedelta(days=7) + timedelta(minutes=20),
            store=store,
            llm_client=client,
        )
        _print_result(result1)
        _print_profile(store, result1.person_id, "scenario 1")

        # ------------------------------------------------------------------ #
        # Scenario 2 — Repeat interlocutor (deduplication)                   #
        # ------------------------------------------------------------------ #
        _banner("SCENARIO 2 — Repeat interlocutor (deduplication check)")
        print("  Ingesting second conversation with same Jordan Lee.")
        print("  Expect: known facts (triathlon, Acme Corp, ...) skipped as")
        print("  duplicates; only genuinely new facts written (promotion, Priya).")

        result2 = await ingest_conversation(
            transcript=TRANSCRIPT_2,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            time_start=now - timedelta(days=3),
            time_end=now - timedelta(days=3) + timedelta(minutes=15),
            store=store,
            llm_client=client,
        )
        _print_result(result2)
        _print_profile(store, result2.person_id, "scenario 2")

        # ------------------------------------------------------------------ #
        # Scenario 3 — Relationship edge to a known person/org               #
        # ------------------------------------------------------------------ #
        _banner("SCENARIO 3 — Edge extraction to pre-existing person")
        print("  Pre-creating 'Acme Corp' as a person/org in the store.")
        print("  Ingesting a third conversation where Jordan mentions Acme.")
        print("  Expect: a works_at (or similar) edge written to Acme Corp.")

        store.upsert_person(name="Acme Corp", aliases=["Acme"])

        result3 = await ingest_conversation(
            transcript=TRANSCRIPT_3,
            wearer_name="Alex",
            interlocutor_name="Jordan Lee",
            time_start=now - timedelta(hours=2),
            time_end=now - timedelta(hours=2) + timedelta(minutes=10),
            store=store,
            llm_client=client,
        )
        _print_result(result3)
        _print_profile(store, result3.person_id, "scenario 3")

        # ------------------------------------------------------------------ #
        # Summary                                                              #
        # ------------------------------------------------------------------ #
        _banner("SUMMARY")
        profile_final = store.get_profile_context(result1.person_id)
        print(f"  Total facts stored for Jordan Lee : {len(profile_final.facts)}")
        print(f"  Total summaries                   : {len(profile_final.summaries)}")
        print(f"  Total edges from Jordan           : {len(profile_final.edges_from)}")
        print()

    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
