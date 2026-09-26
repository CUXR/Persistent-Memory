"""Commit a validated extraction and its audio-job completion atomically."""
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from ..models import AudioJob, Edge, Episode, EpisodeParticipant, Person, PersonFact, Summary
from ..schema.ingestion import EpisodeSummaryLLMResponse, IngestionResult
from ..schema.memory import EdgeIn, EpisodeIn, FactIn
from .memory_store import MemoryStore


def commit_conversation(
    store: MemoryStore, *, person_id: UUID, transcript: str,
    time_start: datetime, time_end: datetime, summary: EpisodeSummaryLLMResponse,
    facts: list[FactIn], edges: list[EdgeIn],
    recording_id: UUID | None = None, attempt_id: UUID | None = None,
) -> IngestionResult:
    data = EpisodeIn(time_start=time_start, time_end=time_end, transcript=transcript,
                     summary=summary.summary, participant_ids=[person_id])
    with store.Session.begin() as session:
        # Serialize commits for the same person before rechecking existing facts.
        person = session.scalar(select(Person).where(
            Person.id == person_id, Person.user_id == store.owner_user_id,
        ).with_for_update())
        if person is None:
            raise ValueError("Person not found for this owner")
        job = None
        if recording_id is not None:
            job = session.scalar(select(AudioJob).where(
                AudioJob.id == recording_id, AudioJob.user_id == store.owner_user_id,
            ).with_for_update())
            if job is None or job.person_id != person_id:
                raise ValueError("Audio job does not belong to this conversation")
            if job.status == "complete":
                return IngestionResult.model_validate(job.result)
            if job.status != "processing" or job.attempt_id != attempt_id:
                raise ValueError("Audio processing attempt is no longer current")
        episode = Episode(user_id=store.owner_user_id, person_id=person_id,
                          start_time=data.time_start, end_time=data.time_end,
                          transcript=transcript, dialogue_summary=summary.summary,
                          importance_score=summary.importance_score)
        session.add(episode)
        session.flush()
        session.add(EpisodeParticipant(episode_id=episode.id, person_id=person_id))
        session.add(Summary(person_id=person_id, episode_id=episode.id,
                            summary_text=summary.summary, episode_time_start=time_start,
                            episode_time_end=time_end))
        # Recheck deduplication at commit time, including repeated facts within a response.
        def normalize(text):
            return " ".join(text.lower().split())
        seen = {normalize(text) for text in session.scalars(
            select(PersonFact.fact_text).where(PersonFact.person_id == person_id)
        )}
        written, skipped, relationships = [], [], []
        for fact in facts:
            if fact.person_id != person_id:
                raise ValueError("Extracted fact targets the wrong person")
            if normalize(fact.fact_text) in seen:
                skipped.append(fact.fact_text)
                continue
            row = PersonFact(person_id=person_id, fact_text=fact.fact_text,
                             fact_category=fact.fact_category, confidence=fact.confidence,
                             source_episode_id=episode.id)
            session.add(row)
            session.flush()
            written.append(row.id)
            seen.add(normalize(fact.fact_text))
        for edge in edges:
            if edge.src_id != person_id:
                raise ValueError("Extracted relationship targets the wrong person")
            store._assert_person_exists(session, edge.dst_id, store.owner_user_id)
            row = session.scalar(select(Edge).where(
                Edge.src_id == person_id, Edge.dst_id == edge.dst_id, Edge.relation == edge.relation,
            ))
            if row is None:
                row = Edge(src_id=person_id, dst_id=edge.dst_id, relation=edge.relation)
                session.add(row)
            row.confidence, row.episode_id = edge.confidence, episode.id
            session.flush()
            relationships.append(row.id)
        # Never move last_seen backwards when importing an older recording.
        previous = person.last_seen_at
        def utc(value):
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        if previous is None or utc(time_end) > utc(previous):
            person.last_seen_at = time_end
        result = IngestionResult(
            episode_id=episode.id, person_id=person_id, summary=summary.summary,
            importance_score=summary.importance_score, facts_written=written,
            facts_skipped_as_duplicate=skipped, edges_written=relationships,
        )
        if job is not None:
            job.status, job.stage, job.error = "complete", "complete", None
            job.episode_id, job.result = episode.id, result.model_dump(mode="json")
        return result
