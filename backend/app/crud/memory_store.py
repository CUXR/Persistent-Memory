from __future__ import annotations

from datetime import datetime, timezone
import logging
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import create_engine, delete, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased, selectinload, sessionmaker

from ..core.config import get_settings
from ..core.database import Base
from ..models.episode import Episode
from ..models.memory import Alias, Edge, EpisodeParticipant, Pref, Summary
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
    PrefIn,
    PrefOut,
    ProfileContext,
    SummaryIn,
    SummaryOut,
)

logger = logging.getLogger("app.crud.memory_store")


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

    def __init__(self, db_url: str | None = None, owner_user_id: UUID | None = None) -> None:
        """Create a store bound to an optional database URL and owner user.

        Args:
            db_url: Override for the configured database URL.
            owner_user_id: Existing user UUID that owns all memory records created
                through this store. When omitted, the store will reuse the first
                available user or create a default owner on first write.
        """

        self._db_url = db_url or get_settings().database_url
        self._owner_user_id = owner_user_id
        # Track whether the owner UUID was explicitly supplied by the caller.
        # When True, a missing owner is always a programming error and raises.
        # When False, a stale cache (e.g. from a rolled-back read-only session)
        # should silently recover by re-discovering or re-creating the owner.
        self._explicit_owner: bool = owner_user_id is not None
        self._engine: Optional[Engine] = None
        self._Session: Optional[sessionmaker[Session]] = None
        logger.info("MemoryStore created (db_url=%s)", self._db_url)

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

    def upsert_person(
        self,
        name: str,
        aliases: list[str],
        face_key: Optional[str] = None,
        voice_key: Optional[str] = None,
        persona90: Optional[list[float]] = None,
    ) -> PersonOut:
        """Create or update an interlocutor for the current owner.

        The lookup is case-insensitive on the person's display name within the
        active owner scope. When a matching person already exists, the supplied
        keys and persona vector are updated and the alias collection is replaced
        with the provided set.

        Returns:
            The normalized stored person record.
        """

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
                    owner = self._get_or_create_owner(session)
                    person = session.scalar(
                        select(Person).where(
                            Person.user_id == owner.id,
                            func.lower(func.coalesce(Person.display_name, "")) == data.name.lower(),
                        )
                    )

                    if person is None:
                        first_name, last_name, display_name = _split_name(data.name)
                        person = Person(
                            user_id=owner.id,
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

                    session.execute(delete(Alias).where(Alias.person_id == person.id))
                    for alias in data.aliases:
                        session.add(Alias(person_id=person.id, alias=alias))

                return self._load_person(session, person.id)
            except IntegrityError as exc:
                raise ValueError(f"Unable to upsert person '{name}': {exc.orig}") from exc

    def list_people(self) -> list[Person]:
        """Return all people for the current owner with aliases eagerly loaded.

        This is a read-only method intended for resolution and search use cases
        where the caller needs access to ``display_name``, ``first_name``, and
        ``aliases`` on each person.  Unlike write methods, this will never
        create a default owner; it returns an empty list when no owner exists.

        Returns:
            Person ORM instances with their ``aliases`` relationship populated.
            An empty list when no owner or people exist.
        """

        with self.Session() as session:
            owner_id = self._find_owner_id(session)
            if owner_id is None:
                return []
            people = list(
                session.scalars(
                    select(Person)
                    .options(selectinload(Person.aliases))
                    .where(Person.user_id == owner_id)
                ).all()
            )
            logger.debug("list_people: %d people for owner=%s", len(people), owner_id)
            return people

    def resolve_person_by_name_or_alias(self, text: str) -> Optional[PersonOut]:
        """Resolve a person by display name or alias for the current owner.

        The match is case-insensitive and restricted to people owned by the
        store's active user. Blank input returns ``None`` instead of raising.

        Returns:
            The matching person record, or ``None`` when no match exists.
        """

        cleaned = text.strip()
        if not cleaned:
            return None

        with self.Session() as session:
            owner = self._get_or_create_owner(session)
            person = session.scalar(
                select(Person)
                .outerjoin(Alias, Alias.person_id == Person.id)
                .where(
                    Person.user_id == owner.id,
                    or_(
                        func.lower(func.coalesce(Person.display_name, "")) == cleaned.lower(),
                        func.lower(Alias.alias) == cleaned.lower(),
                    )
                )
                .limit(1)
            )

            if person is None:
                logger.debug("resolve_person: no match for '%s'", cleaned)
                return None

            logger.debug("resolve_person: '%s' -> id=%s", cleaned, person.id)
            return self._load_person(session, person.id)

    def write_episode(
        self,
        time_start: datetime,
        time_end: datetime,
        transcript: str = "",
        summary: str = "",
        participants: Optional[list[UUID]] = None,
    ) -> UUID:
        """Insert a conversation episode and its participant links.

        The first participant becomes the primary ``Episode.person_id`` so the
        episode remains compatible with the main schema, while every supplied
        participant is also written to ``EpisodeParticipant``.

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
                owner = self._get_or_create_owner(session)
                ordered_participants = list(dict.fromkeys(data.participant_ids))
                if not ordered_participants:
                    raise ValueError("write_episode requires at least one participant")
                self._assert_people_exist(session, ordered_participants, owner.id)

                episode = Episode(
                    user_id=owner.id,
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

            logger.debug(
                "Wrote episode id=%s (%s -> %s, %d participants)",
                episode_id,
                data.time_start.isoformat(),
                data.time_end.isoformat(),
                len(ordered_participants),
            )
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

        The person must belong to the current owner. When ``episode_id`` is
        provided, the fact is linked back to the episode where it was learned.

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
                owner = self._get_or_create_owner(session)
                self._assert_person_exists(session, data.person_id, owner.id)
                self._assert_episode_exists(session, data.episode_id, owner.id)

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

    def write_pref(
        self,
        person_id: UUID,
        pref_text: str,
        confidence: float = 1.0,
        episode_id: Optional[UUID] = None,
    ) -> UUID:
        """Persist a preference remembered about a person.

        Preferences are owner-scoped through the referenced person and may be
        associated with an originating episode.

        Returns:
            The UUID of the created preference row.
        """

        data = PrefIn(
            person_id=person_id,
            pref_text=pref_text,
            confidence=confidence,
            episode_id=episode_id,
        )

        with self.Session() as session:
            with session.begin():
                owner = self._get_or_create_owner(session)
                self._assert_person_exists(session, data.person_id, owner.id)
                self._assert_episode_exists(session, data.episode_id, owner.id)

                row = Pref(
                    person_id=data.person_id,
                    pref_text=data.pref_text,
                    confidence=_decimal(data.confidence),
                    episode_id=data.episode_id,
                )
                session.add(row)
                session.flush()
                pref_id = row.id

            logger.debug("Wrote pref id=%s for person=%s", pref_id, data.person_id)
            return pref_id

    def write_summary(
        self,
        person_id: UUID,
        summary_text: str,
        episode_time_start: Optional[datetime] = None,
        episode_time_end: Optional[datetime] = None,
        episode_id: Optional[UUID] = None,
    ) -> UUID:
        """Persist a summary slice for a person.

        This stores the human-readable summary text plus optional time bounds
        describing the source interaction window and an optional backing episode.

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
                owner = self._get_or_create_owner(session)
                self._assert_person_exists(session, data.person_id, owner.id)
                self._assert_episode_exists(session, data.episode_id, owner.id)

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

        Edges are unique by ``(src_id, relation, dst_id)``. Rewriting an
        existing edge updates its confidence and optional episode reference
        instead of creating a duplicate row.

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
                owner = self._get_or_create_owner(session)
                self._assert_person_exists(session, data.src_id, owner.id)
                self._assert_person_exists(session, data.dst_id, owner.id)
                self._assert_episode_exists(session, data.episode_id, owner.id)

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

            logger.debug(
                "Wrote edge id=%s: person %s -[%s]-> person %s",
                edge_id,
                data.src_id,
                data.relation,
                data.dst_id,
            )
            return edge_id

    def get_user_facts(self) -> list[str]:
        """Return all fact texts stored for the current owner user.

        Used by the ingestion pipeline to provide wearer context when detecting
        shared interests between the wearer and an interlocutor.
        """

        with self.Session() as session:
            owner = self._get_or_create_owner(session)
            rows = session.scalars(
                select(UserFact.fact_text)
                .where(UserFact.user_id == owner.id)
                .order_by(UserFact.created_at.asc())
            ).all()
            return list(rows)

    def get_profile_context(self, person_id: UUID) -> ProfileContext:
        """Return the issue-specified profile context bundle for one person.

        The returned structure contains facts, preferences, summaries, outgoing
        edges, and the stored ``persona90`` vector ordered newest-first within
        each collection.

        Returns:
            A ``ProfileContext`` ready for deterministic retrieval use.
        """

        with self.Session() as session:
            owner = self._get_or_create_owner(session)
            person = self._get_person(session, person_id, owner.id)

            facts_rows = session.scalars(
                select(PersonFact)
                .where(PersonFact.person_id == person_id)
                .order_by(PersonFact.created_at.desc())
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

            prefs = [
                PrefOut(
                    id=row.id,
                    pref_text=row.pref_text,
                    confidence=_float(row.confidence),
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
                    dst_name=dst_name or "",
                    confidence=_float(edge.confidence),
                    episode_id=edge.episode_id,
                    created_at=_iso(edge.created_at) or "",
                )
                for edge, dst_name in edge_rows
            ]

            logger.debug(
                "Profile context for person=%s: %d facts, %d prefs, %d summaries, %d edges",
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

    def _load_person(self, session: Session, person_id: UUID) -> PersonOut:
        person = self._get_person(session, person_id, person_user_id=None)
        aliases = session.scalars(
            select(Alias.alias).where(Alias.person_id == person_id).order_by(Alias.alias.asc())
        ).all()

        return PersonOut(
            id=person.id,
            name=person.display_name or person.first_name,
            face_key=person.face_key,
            voice_key=person.voice_key,
            persona90=person.persona90 or [],
            aliases=list(aliases),
            created_at=_iso(person.created_at) or "",
            updated_at=_iso(person.updated_at) or "",
        )

    def _find_owner_id(self, session: Session) -> UUID | None:
        """Return the active owner's UUID without creating a default user.

        This is the read-safe counterpart to ``_get_or_create_owner``.  When
        the cached ``_owner_user_id`` points to a deleted or rolled-back user,
        the stale reference is cleared and the lookup falls through to the
        first available user.  Returns ``None`` when no user exists at all.
        """

        if self._owner_user_id is not None:
            if session.get(User, self._owner_user_id) is not None:
                return self._owner_user_id
            # Stale reference -- clear and fall through
            self._owner_user_id = None

        owner = session.scalar(
            select(User).order_by(User.created_at.asc()).limit(1)
        )
        if owner is not None:
            self._owner_user_id = owner.id
            return owner.id
        return None

    def _get_or_create_owner(self, session: Session) -> User:
        if self._owner_user_id is not None:
            owner = session.get(User, self._owner_user_id)
            if owner is not None:
                return owner
            if self._explicit_owner:
                # Caller supplied a specific UUID — a missing row is always an error.
                raise ValueError(f"Owner user id={self._owner_user_id} not found")
            # Auto-cached UUID is stale (e.g. from a rolled-back read-only session).
            # Clear the cache and fall through to re-discover / re-create below.
            logger.debug(
                "_get_or_create_owner: stale cached id=%s, re-discovering owner",
                self._owner_user_id,
            )
            self._owner_user_id = None

        owner = session.scalar(select(User).order_by(User.created_at.asc()).limit(1))
        if owner is None:
            owner = User(
                first_name="Memory",
                last_name="Owner",
                display_name="Memory Store Owner",
                username="memory-store-owner",
            )
            session.add(owner)
            session.flush()
            logger.debug("Created default owner user id=%s", owner.id)

        self._owner_user_id = owner.id
        return owner

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
