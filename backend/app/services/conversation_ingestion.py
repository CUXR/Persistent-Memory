"""Extract memories from a named transcript and commit them as one transaction."""
from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import UUID

from ..crud.conversations import commit_conversation
from ..crud.memory_store import MemoryStore
from ..schema.ingestion import IngestionResult
from ..schema.memory import EdgeIn, FactIn
from .llm_client import LLMClient


async def ingest_conversation(
    transcript: str, wearer_name: str, interlocutor_name: str,
    time_start: datetime, time_end: datetime, store: MemoryStore,
    llm_client: LLMClient | None = None, *,
    person_id: UUID | None = None, recording_id: UUID | None = None,
    attempt_id: UUID | None = None,
) -> IngestionResult:
    """Audio callers supply the selected person ID and a claimed recording job.

    Text callers resolve/create by the person's single display name. No model
    calls occur inside the transaction that persists the resulting memories.
    """
    if not transcript.strip():
        raise ValueError("A conversation transcript is required")
    if time_end < time_start:
        raise ValueError("time_end must be >= time_start")
    person = store.get_person(person_id) if person_id is not None else store.resolve_person_by_name(interlocutor_name)
    if person is None:
        person = store.upsert_person(interlocutor_name)
    interlocutor_name = person.name
    profile = store.get_profile_context(person.id)
    client = llm_client or LLMClient()
    summary, extraction = await asyncio.gather(
        client.generate_summary(transcript, wearer_name, interlocutor_name),
        client.extract_facts(transcript, wearer_name, interlocutor_name,
                             [fact.fact_text for fact in profile.facts], store.get_user_facts()),
    )
    facts = [FactIn(person_id=person.id, fact_text=fact.fact_text,
                    confidence=fact.confidence, fact_category=fact.category)
             for fact in extraction.facts]
    edges = []
    for edge in extraction.edges:
        target = store.resolve_person_by_name(edge.target_name)
        if target is not None and target.id != person.id:
            edges.append(EdgeIn(src_id=person.id, dst_id=target.id,
                                relation=edge.relation, confidence=edge.confidence))
    return commit_conversation(
        store, person_id=person.id, transcript=transcript, time_start=time_start,
        time_end=time_end, summary=summary, facts=facts, edges=edges,
        recording_id=recording_id, attempt_id=attempt_id,
    )
