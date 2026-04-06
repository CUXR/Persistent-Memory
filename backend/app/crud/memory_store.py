from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import logging
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, aliased, sessionmaker

from ..core.config import get_settings
from ..core.database import Base
from ..models.episode import Episode
from ..models.memory import Edge, EpisodeParticipant, Summary
from ..models.person import Person, PersonFact
from ..models.user import User, UserFact
from ..schema.memory import (
    EdgeIn,
    EdgeOut,
    EpisodeIn,
    FactIn,
    FactOut,
    PersonIn,
    PersonOut,
    ProfileContext,
    SummaryIn,
    SummaryOut,
)

logger = logging.getLogger("app.crud.memory_store")
settings = get_settings()

# Priority order for fact categories used in disambiguation
FACT_CATEGORY_PRIORITY = ("visual_descriptor", "affiliation", "hobby")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.isoformat()


def _decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _float(value: Decimal | None) -> float:
    if value is None:
        return 1.0
    return float(value)


def _split_name(name: str) -> tuple[str, str, str]:
    cleaned = name.strip()
    parts = cleaned.split(maxsplit=1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else ""
    return first_name, last_name, cleaned


class MemoryStore:
    """Typed, deterministic memory access layer over the project schema."""

    def __init__(self, db_url: str | None = None, *, owner_user_id: UUID) -> None:
        """Create a store bound to a database URL and owner user.

        Args:
            db_url: Override for the configured database URL.
            owner_user_id: UUID of the user that owns all memory records
                created and queried through this store.  The caller is
                responsible for ensuring this user exists in the database.
        """

        self._db_url = db_url or settings.database_url
        self._owner_user_id = owner_user_id
        self._engine: Optional[Engine] = None
        self._Session: Optional[sessionmaker[Session]] = None
        logger.info("MemoryStore created (db_url=%s, owner=%s)", self._db_url, owner_user_id)

    @property
    def owner_user_id(self) -> UUID:
        """The UUID of the owner user scoping all operations."""
        return self._owner_user_id

    def initialize(self, create_schema: bool = True) -> None:
        """Initialize the SQLAlchemy engine and create tables when requested."""

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
        """Dispose the SQLAlchemy engine and clear the session factory."""

        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._Session = None
            logger.info("Database connection closed")

    @property
    def Session(self) -> sessionmaker[Session]:
        """Return the configured session factory after initialization."""

        if self._Session is None:
            raise RuntimeError("MemoryStore not initialized - call .initialize() first")
        return self._Session

    # ── Person CRUD ──────────────────────────────────────────

    def upsert_person(
        self,
        name: str,
        face_key: Optional[str] = None,
        voice_key: Optional[str] = None,
        persona90: Optional[list[float]] = None,
    ) -> PersonOut:
        """Create or update an interlocutor for the current owner.

        The lookup is case-insensitive on the person's display name within the
        active owner scope. When a matching person already exists, the supplied
        keys and persona vector are updated.

        Returns:
            The normalized stored person record.
        """

        data = PersonIn(
            name=name,
            face_key=face_key,
            voice_key=voice_key,
            persona90=persona90 or [],
        )
        persona_supplied = persona90 is not None

        with self.Session() as session:
            with session.begin():
                person = session.scalar(
                    select(Person).where(
                        Person.user_id == self._owner_user_id,
                        func.lower(func.coalesce(Person.display_name, "")) == data.name.lower(),
                    )
                )

                if person is None:
                    first_name, last_name, display_name = _split_name(data.name)
                    person = Person(
                        user_id=self._owner_user_id,
                        first_name=first_name,
                        last_name=last_name,
                        display_name=display_name,
                        face_key=data.face_key,
                        voice_key=data.voice_key,
                        persona90=data.persona90,
                    )
                    session.add(person)
                    session.flush()
                    logger.debug("Inserted person id=%s name=%s", person.id, data.name)
                else:
                    if data.face_key is not None:
                        person.face_key = data.face_key
                    if data.voice_key is not None:
                        person.voice_key = data.voice_key
                    if persona_supplied:
                        person.persona90 = data.persona90
                    first_name, last_name, display_name = _split_name(data.name)
                    person.first_name = first_name
                    person.last_name = last_name
                    person.display_name = display_name
                    person.updated_at = datetime.now(timezone.utc)
                    session.flush()
                    logger.debug("Updated person id=%s name=%s", person.id, data.name)

            return self._load_person(session, person.id)

    def list_people(self) -> list[Person]:
        """Return all people for the current owner.

        Returns:
            Person ORM instances.
            An empty list when no people exist for the owner.
        """

        with self.Session() as session:
            people = list(
                session.scalars(
                    select(Person).where(Person.user_id == self._owner_user_id)
                ).all()
            )
            logger.debug("list_people: %d people for owner=%s", len(people), self._owner_user_id)
            return people

    def resolve_person_by_name(self, text: str) -> Optional[PersonOut]:
        """Resolve a person by display name for the current owner.

        Case-insensitive match on display_name. Used by the ingestion pipeline
        for exact-name lookups. For query-based resolution with ambiguity
        handling, use PersonResolver instead.

        Returns:
            The matching person record, or ``None`` when no match exists.
        """

        cleaned = text.strip()
        if not cleaned:
            return None

        with self.Session() as session:
            person = session.scalar(
                select(Person).where(
                    Person.user_id == self._owner_user_id,
                    func.lower(func.coalesce(Person.display_name, "")) == cleaned.lower(),
                )
            )

            if person is None:
                logger.debug("resolve_person: no match for '%s'", cleaned)
                return None

            logger.debug("resolve_person: '%s' -> id=%s", cleaned, person.id)
            return self._load_person(session, person.id)

    # ── Episode / Fact / Summary / Edge writes ───────────────

    def write_episode(
        self,
        time_start: datetime,
        time_end: datetime,
        transcript: str = "",
        summary: str = "",
        participants: Optional[list[UUID]] = None,
    ) -> UUID:
        """Insert a conversation episode and its participant links.

        Returns:
            The UUID of the created episode row.
        """

        data = EpisodeIn(
            time_start=time_start,
            time_end=time_end,
            transcript=transcript,
            summary=summary,
            participant_ids=participants or [],
        )

        with self.Session() as session:
            with session.begin():
                ordered_participants = list(dict.fromkeys(data.participant_ids))
                if not ordered_participants:
                    raise ValueError("write_episode requires at least one participant")
                self._assert_people_exist(session, ordered_participants, self._owner_user_id)

                episode = Episode(
                    user_id=self._owner_user_id,
                    person_id=ordered_participants[0],
                    start_time=data.time_start,
                    end_time=data.time_end,
                    transcript=data.transcript,
                    dialogue_summary=data.summary,
                )
                session.add(episode)
                session.flush()

                for person_id in ordered_participants:
                    session.add(EpisodeParticipant(episode_id=episode.id, person_id=person_id))

                episode_id = episode.id

            logger.debug("Wrote episode id=%s", episode_id)
            return episode_id

    def write_fact(
        self,
        person_id: UUID,
        fact_text: str,
        confidence: float = 1.0,
        fact_category: Optional[str] = None,
        episode_id: Optional[UUID] = None,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ) -> UUID:
        """Persist a structured fact about a person.

        Returns:
            The UUID of the created fact row.
        """

        data = FactIn(
            person_id=person_id,
            fact_text=fact_text,
            confidence=confidence,
            fact_category=fact_category,
            episode_id=episode_id,
            valid_from=valid_from,
            valid_to=valid_to,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.person_id, self._owner_user_id)
                self._assert_episode_exists(session, data.episode_id, self._owner_user_id)

                row = PersonFact(
                    person_id=data.person_id,
                    fact_text=data.fact_text,
                    fact_category=data.fact_category,
                    confidence=_decimal(data.confidence),
                    source_episode_id=data.episode_id,
                    valid_from=data.valid_from,
                    valid_to=data.valid_to,
                )
                session.add(row)
                session.flush()
                fact_id = row.id

            logger.debug("Wrote fact id=%s for person=%s", fact_id, data.person_id)
            return fact_id

    def write_summary(
        self,
        person_id: UUID,
        summary_text: str,
        episode_time_start: Optional[datetime] = None,
        episode_time_end: Optional[datetime] = None,
        episode_id: Optional[UUID] = None,
    ) -> UUID:
        """Persist a summary slice for a person.

        Returns:
            The UUID of the created summary row.
        """

        data = SummaryIn(
            person_id=person_id,
            summary_text=summary_text,
            episode_time_start=episode_time_start,
            episode_time_end=episode_time_end,
            episode_id=episode_id,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.person_id, self._owner_user_id)
                self._assert_episode_exists(session, data.episode_id, self._owner_user_id)

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

            logger.debug("Wrote summary id=%s for person=%s", summary_id, data.person_id)
            return summary_id

    def write_edge(
        self,
        src_id: UUID,
        relation: str,
        dst_id: UUID,
        confidence: float = 1.0,
        episode_id: Optional[UUID] = None,
    ) -> UUID:
        """Create or update a directed relationship between two people.

        Edges are unique by ``(src_id, relation, dst_id)``.

        Returns:
            The UUID of the inserted or updated edge row.
        """

        data = EdgeIn(
            src_id=src_id,
            relation=relation,
            dst_id=dst_id,
            confidence=confidence,
            episode_id=episode_id,
        )

        with self.Session() as session:
            with session.begin():
                self._assert_person_exists(session, data.src_id, self._owner_user_id)
                self._assert_person_exists(session, data.dst_id, self._owner_user_id)
                self._assert_episode_exists(session, data.episode_id, self._owner_user_id)

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
                        confidence=_decimal(data.confidence),
                        episode_id=data.episode_id,
                    )
                    session.add(edge)
                    session.flush()
                else:
                    edge.confidence = _decimal(data.confidence)
                    edge.episode_id = data.episode_id
                    session.flush()

                edge_id = edge.id

            logger.debug("Wrote edge id=%s: %s -[%s]-> %s", edge_id, data.src_id, data.relation, data.dst_id)
            return edge_id

    # ── Read methods ─────────────────────────────────────────

    def get_user_facts(self) -> list[str]:
        """Return all fact texts stored for the current owner user.

        Used by the ingestion pipeline to provide wearer context.
        """

        with self.Session() as session:
            rows = session.scalars(
                select(UserFact.fact_text)
                .where(UserFact.user_id == self._owner_user_id)
                .order_by(UserFact.created_at.asc())
            ).all()
            return list(rows)

    def get_disambiguation_hints(self, person_id: UUID) -> dict[str, list[str]]:
        """Return categorized fact texts for person disambiguation.

        Groups facts by ``fact_category`` in priority order:
        visual_descriptor, affiliation, hobby.

        Returns:
            Dict keyed by category with lists of fact_text strings.
            Categories with no facts are omitted.
        """

        with self.Session() as session:
            rows = session.execute(
                select(PersonFact.fact_category, PersonFact.fact_text)
                .where(
                    PersonFact.person_id == person_id,
                    PersonFact.fact_category.isnot(None),
                )
                .order_by(PersonFact.created_at.desc())
            ).all()

            hints: dict[str, list[str]] = {}
            for category in FACT_CATEGORY_PRIORITY:
                texts = [text for cat, text in rows if cat == category]
                if texts:
                    hints[category] = texts

            return hints

    def get_profile_context(self, person_id: UUID) -> ProfileContext:
        """Return the profile context bundle for one person.

        The returned structure contains facts, summaries, outgoing edges,
        and the stored ``persona90`` vector ordered newest-first.

        Returns:
            A ``ProfileContext`` ready for deterministic retrieval use.
        """

        with self.Session() as session:
            person = self._get_person(session, person_id, self._owner_user_id)

            facts_rows = session.scalars(
                select(PersonFact)
                .where(PersonFact.person_id == person_id)
                .order_by(PersonFact.created_at.desc())
            ).all()

            summaries_rows = session.scalars(
                select(Summary)
                .where(Summary.person_id == person_id)
                .order_by(Summary.created_at.desc())
            ).all()

            dst_person = aliased(Person)
            edge_rows = session.execute(
                select(Edge, func.coalesce(dst_person.display_name, dst_person.first_name))
                .join(dst_person, dst_person.id == Edge.dst_id)
                .where(Edge.src_id == person_id)
                .order_by(Edge.created_at.desc())
            ).all()

            facts = [
                FactOut(
                    id=row.id,
                    fact_text=row.fact_text,
                    confidence=_float(row.confidence),
                    fact_category=row.fact_category,
                    episode_id=row.source_episode_id,
                    valid_from=_iso(row.valid_from),
                    valid_to=_iso(row.valid_to),
                    created_at=_iso(row.created_at) or "",
                )
                for row in facts_rows
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
                    dst_name=dst_name or "",
                    confidence=_float(edge.confidence),
                    episode_id=edge.episode_id,
                    created_at=_iso(edge.created_at) or "",
                )
                for edge, dst_name in edge_rows
            ]

            return ProfileContext(
                facts=facts,
                summaries=summaries,
                edges_from=edges,
                persona90=person.persona90 or [],
            )

    # ── Private helpers ──────────────────────────────────────

    def _load_person(self, session: Session, person_id: UUID) -> PersonOut:
        person = self._get_person(session, person_id, person_user_id=None)

        return PersonOut(
            id=person.id,
            name=person.display_name or person.first_name,
            face_key=person.face_key,
            voice_key=person.voice_key,
            persona90=person.persona90 or [],
            created_at=_iso(person.created_at) or "",
            updated_at=_iso(person.updated_at) or "",
        )

    def _get_person(self, session: Session, person_id: UUID, person_user_id: UUID | None) -> Person:
        person = session.get(Person, person_id)
        if person is None or (person_user_id is not None and person.user_id != person_user_id):
            raise ValueError(f"Person id={person_id} not found")
        return person

    def _assert_person_exists(self, session: Session, person_id: UUID, owner_id: UUID) -> None:
        person = session.get(Person, person_id)
        if person is None or person.user_id != owner_id:
            raise ValueError(f"Person id={person_id} not found")

    def _assert_episode_exists(
        self,
        session: Session,
        episode_id: Optional[UUID],
        owner_id: UUID,
    ) -> None:
        if episode_id is None:
            return
        episode = session.get(Episode, episode_id)
        if episode is None or episode.user_id != owner_id:
            raise ValueError(f"Episode id={episode_id} not found")

    def _assert_people_exist(self, session: Session, person_ids: list[UUID], owner_id: UUID) -> None:
        if not person_ids:
            return
        unique_ids = list(dict.fromkeys(person_ids))
        existing_ids = set(
            session.scalars(
                select(Person.id).where(Person.user_id == owner_id, Person.id.in_(unique_ids))
            ).all()
        )
        missing = [pid for pid in unique_ids if pid not in existing_ids]
        if missing:
            missing_text = ", ".join(str(pid) for pid in missing)
            raise ValueError(f"Participant person id(s) not found: {missing_text}")
