"""
backend/app/crud/memory_store.py
--------------------------------
Typed SQLAlchemy service for all memory operations.

Deterministic data access only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Optional

from sqlalchemy import delete, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased, sessionmaker
from sqlalchemy import create_engine

from ..schema.memory import (
    EdgeIn,
    EdgeOut,
    EpisodeIn,
    FactIn,
    FactOut,
    PersonIn,
    PersonOut,
    PrefIn,
    PrefOut,
    ProfileContext,
    SummaryIn,
    SummaryOut,
)
from ..models.memory import Alias, Base, Edge, Episode, EpisodeParticipant, Fact, Person, Pref, Summary

logger = logging.getLogger("app.crud.memory_store")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.isoformat()


class MemoryStore:
    """
    Typed, validated access to memory data via SQLAlchemy.

    Public methods match the project spec method signatures and use Pydantic
    input models internally for validation.
    """

    def __init__(self, db_url: str = "sqlite+pysqlite:///:memory:") -> None:
        self._db_url = db_url
        self._engine: Optional[Engine] = None
        self._Session: Optional[sessionmaker[Session]] = None
        logger.info("MemoryStore created (db_url=%s)", db_url)

    def initialize(self, create_schema: bool = True) -> None:
        """Initialize engine/session factory and optionally create tables."""
        self._engine = create_engine(self._db_url, future=True)
        self._Session = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        if create_schema:
            Base.metadata.create_all(self._engine)
        logger.info("Database initialized")

    def close(self) -> None:
        """Dispose SQLAlchemy engine."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._Session = None
            logger.info("Database connection closed")

    @property
    def Session(self) -> sessionmaker[Session]:
        if self._Session is None:
            raise RuntimeError("MemoryStore not initialized - call .initialize() first")
        return self._Session

    def upsert_person(
        self,
        name: str,
        aliases: list[str],
        face_key: Optional[str] = None,
        voice_key: Optional[str] = None,
        persona90: Optional[list[float]] = None,
    ) -> PersonOut:
        data = PersonIn(
            name=name,
            aliases=aliases,
            face_key=face_key,
            voice_key=voice_key,
            persona90=persona90 or [],
        )
        persona_supplied = persona90 is not None

        with self.Session() as session:
            try:
                with session.begin():
                    person = session.scalar(
                        select(Person).where(func.lower(Person.name) == data.name.lower())
                    )

                    if person is None:
                        person = Person(
                            name=data.name,
                            face_key=data.face_key,
                            voice_key=data.voice_key,
                            persona90=data.persona90,
                        )
                        session.add(person)
                        session.flush()
                        logger.debug("Inserted person id=%d name=%s", person.id, data.name)
                    else:
                        if data.face_key is not None:
                            person.face_key = data.face_key
                        if data.voice_key is not None:
                            person.voice_key = data.voice_key
                        if persona_supplied:
                            person.persona90 = data.persona90
                        person.updated_at = datetime.now(timezone.utc)
                        session.flush()
                        logger.debug("Updated person id=%d name=%s", person.id, data.name)

                    session.execute(delete(Alias).where(Alias.person_id == person.id))
                    for alias in data.aliases:
                        session.add(Alias(person_id=person.id, alias=alias))

                return self._load_person(session, person.id)
            except IntegrityError as exc:
                raise ValueError(f"Unable to upsert person '{name}': {exc.orig}") from exc

    def resolve_person_by_name_or_alias(self, text: str) -> Optional[PersonOut]:
        cleaned = text.strip()
        if not cleaned:
            return None

        with self.Session() as session:
            person = session.scalar(
                select(Person)
                .outerjoin(Alias, Alias.person_id == Person.id)
                .where(
                    or_(
                        func.lower(Person.name) == cleaned.lower(),
                        func.lower(Alias.alias) == cleaned.lower(),
                    )
                )
                .limit(1)
            )

            if person is None:
                logger.debug("resolve_person: no match for '%s'", cleaned)
                return None

            logger.debug("resolve_person: '%s' -> id=%d", cleaned, person.id)
            return self._load_person(session, person.id)

    def write_episode(
        self,
        time_start: datetime,
        time_end: datetime,
        transcript: str = "",
        summary: str = "",
        participants: Optional[list[int]] = None,
    ) -> int:
        data = EpisodeIn(
            time_start=time_start,
            time_end=time_end,
            transcript=transcript,
            summary=summary,
            participant_ids=participants or [],
        )

        with self.Session() as session:
            with session.begin():
                self._assert_people_exist(session, data.participant_ids)

                episode = Episode(
                    time_start=data.time_start,
                    time_end=data.time_end,
                    transcript=data.transcript,
                    summary=data.summary,
                )
                session.add(episode)
                session.flush()

                for person_id in sorted(set(data.participant_ids)):
                    session.add(EpisodeParticipant(episode_id=episode.id, person_id=person_id))

                episode_id = episode.id

            logger.debug(
                "Wrote episode id=%d (%s -> %s, %d participants)",
                episode_id,
                data.time_start.isoformat(),
                data.time_end.isoformat(),
                len(data.participant_ids),
            )
            return episode_id

    def write_fact(
        self,
        person_id: int,
        fact_text: str,
        confidence: float = 1.0,
        episode_id: Optional[int] = None,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ) -> int:
        data = FactIn(
            person_id=person_id,
            fact_text=fact_text,
            confidence=confidence,
            episode_id=episode_id,
            valid_from=valid_from,
            valid_to=valid_to,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.person_id)
                self._assert_episode_exists(session, data.episode_id)

                row = Fact(
                    person_id=data.person_id,
                    fact_text=data.fact_text,
                    confidence=data.confidence,
                    episode_id=data.episode_id,
                    valid_from=data.valid_from,
                    valid_to=data.valid_to,
                )
                session.add(row)
                session.flush()
                fact_id = row.id

            logger.debug("Wrote fact id=%d for person=%d", fact_id, data.person_id)
            return fact_id

    def write_pref(
        self,
        person_id: int,
        pref_text: str,
        confidence: float = 1.0,
        episode_id: Optional[int] = None,
    ) -> int:
        data = PrefIn(
            person_id=person_id,
            pref_text=pref_text,
            confidence=confidence,
            episode_id=episode_id,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.person_id)
                self._assert_episode_exists(session, data.episode_id)

                row = Pref(
                    person_id=data.person_id,
                    pref_text=data.pref_text,
                    confidence=data.confidence,
                    episode_id=data.episode_id,
                )
                session.add(row)
                session.flush()
                pref_id = row.id

            logger.debug("Wrote pref id=%d for person=%d", pref_id, data.person_id)
            return pref_id

    def write_summary(
        self,
        person_id: int,
        summary_text: str,
        episode_time_start: Optional[datetime] = None,
        episode_time_end: Optional[datetime] = None,
        episode_id: Optional[int] = None,
    ) -> int:
        data = SummaryIn(
            person_id=person_id,
            summary_text=summary_text,
            episode_time_start=episode_time_start,
            episode_time_end=episode_time_end,
            episode_id=episode_id,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.person_id)
                self._assert_episode_exists(session, data.episode_id)

                row = Summary(
                    person_id=data.person_id,
                    summary_text=data.summary_text,
                    episode_time_start=data.episode_time_start,
                    episode_time_end=data.episode_time_end,
                    episode_id=data.episode_id,
                )
                session.add(row)
                session.flush()
                summary_id = row.id

            logger.debug("Wrote summary id=%d for person=%d", summary_id, data.person_id)
            return summary_id

    def write_edge(
        self,
        src_id: int,
        relation: str,
        dst_id: int,
        confidence: float = 1.0,
        episode_id: Optional[int] = None,
    ) -> int:
        data = EdgeIn(
            src_id=src_id,
            relation=relation,
            dst_id=dst_id,
            confidence=confidence,
            episode_id=episode_id,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.src_id)
                self._assert_person_exists(session, data.dst_id)
                self._assert_episode_exists(session, data.episode_id)

                edge = session.scalar(
                    select(Edge).where(
                        Edge.src_id == data.src_id,
                        Edge.relation == data.relation,
                        Edge.dst_id == data.dst_id,
                    )
                )

                if edge is None:
                    edge = Edge(
                        src_id=data.src_id,
                        relation=data.relation,
                        dst_id=data.dst_id,
                        confidence=data.confidence,
                        episode_id=data.episode_id,
                    )
                    session.add(edge)
                    session.flush()
                else:
                    edge.confidence = data.confidence
                    edge.episode_id = data.episode_id
                    session.flush()

                edge_id = edge.id

            logger.debug(
                "Wrote edge id=%d: person %d -[%s]-> person %d",
                edge_id,
                data.src_id,
                data.relation,
                data.dst_id,
            )
            return edge_id

    def get_profile_context(self, person_id: int) -> ProfileContext:
        with self.Session() as session:
            person = self._get_person(session, person_id)

            facts_rows = session.scalars(
                select(Fact).where(Fact.person_id == person_id).order_by(Fact.created_at.desc())
            ).all()

            prefs_rows = session.scalars(
                select(Pref).where(Pref.person_id == person_id).order_by(Pref.created_at.desc())
            ).all()

            summaries_rows = session.scalars(
                select(Summary)
                .where(Summary.person_id == person_id)
                .order_by(Summary.created_at.desc())
            ).all()

            dst_person = aliased(Person)
            edge_rows = session.execute(
                select(Edge, dst_person.name)
                .join(dst_person, dst_person.id == Edge.dst_id)
                .where(Edge.src_id == person_id)
                .order_by(Edge.created_at.desc())
            ).all()

            facts = [
                FactOut(
                    id=row.id,
                    fact_text=row.fact_text,
                    confidence=row.confidence,
                    episode_id=row.episode_id,
                    valid_from=_iso(row.valid_from),
                    valid_to=_iso(row.valid_to),
                    created_at=_iso(row.created_at) or "",
                )
                for row in facts_rows
            ]

            prefs = [
                PrefOut(
                    id=row.id,
                    pref_text=row.pref_text,
                    confidence=row.confidence,
                    episode_id=row.episode_id,
                    created_at=_iso(row.created_at) or "",
                )
                for row in prefs_rows
            ]

            summaries = [
                SummaryOut(
                    id=row.id,
                    summary_text=row.summary_text,
                    episode_time_start=_iso(row.episode_time_start),
                    episode_time_end=_iso(row.episode_time_end),
                    episode_id=row.episode_id,
                    created_at=_iso(row.created_at) or "",
                )
                for row in summaries_rows
            ]

            edges = [
                EdgeOut(
                    id=edge.id,
                    relation=edge.relation,
                    dst_id=edge.dst_id,
                    dst_name=dst_name,
                    confidence=edge.confidence,
                    episode_id=edge.episode_id,
                    created_at=_iso(edge.created_at) or "",
                )
                for edge, dst_name in edge_rows
            ]

            logger.debug(
                "Profile context for person=%d: %d facts, %d prefs, %d summaries, %d edges",
                person_id,
                len(facts),
                len(prefs),
                len(summaries),
                len(edges),
            )

            return ProfileContext(
                facts=facts,
                prefs=prefs,
                summaries=summaries,
                edges_from=edges,
                persona90=person.persona90 or [],
            )

    def _load_person(self, session: Session, person_id: int) -> PersonOut:
        person = self._get_person(session, person_id)
        aliases = session.scalars(
            select(Alias.alias).where(Alias.person_id == person_id).order_by(Alias.alias.asc())
        ).all()

        return PersonOut(
            id=person.id,
            name=person.name,
            face_key=person.face_key,
            voice_key=person.voice_key,
            persona90=person.persona90 or [],
            aliases=list(aliases),
            created_at=_iso(person.created_at) or "",
            updated_at=_iso(person.updated_at) or "",
        )

    def _get_person(self, session: Session, person_id: int) -> Person:
        person = session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found")
        return person

    def _assert_person_exists(self, session: Session, person_id: int) -> None:
        if session.get(Person, person_id) is None:
            raise ValueError(f"Person id={person_id} not found")

    def _assert_episode_exists(self, session: Session, episode_id: Optional[int]) -> None:
        if episode_id is not None and session.get(Episode, episode_id) is None:
            raise ValueError(f"Episode id={episode_id} not found")

    def _assert_people_exist(self, session: Session, person_ids: list[int]) -> None:
        if not person_ids:
            return
        unique_ids = sorted(set(person_ids))
        existing_ids = set(
            session.scalars(select(Person.id).where(Person.id.in_(unique_ids))).all()
        )
        missing = [pid for pid in unique_ids if pid not in existing_ids]
        if missing:
            missing_text = ", ".join(str(pid) for pid in missing)
            raise ValueError(f"Participant person id(s) not found: {missing_text}")
