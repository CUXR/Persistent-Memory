from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from uuid import UUID as UUIDType
from uuid import uuid4

from sqlalchemy import DateTime, MetaData, Uuid, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, declared_attr, mapped_column, sessionmaker

from .config import get_settings

settings = get_settings()

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


class TimestampMixin:
    """Created/updated timestamps stored in UTC."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """UUID primary key mixin compatible with SQLite and PostgreSQL."""

    id: Mapped[UUIDType] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )


def make_engine(database_url: str | None = None, *, echo: bool | None = None):
    """Create a SQLAlchemy engine for the configured database."""

    return create_engine(
        database_url or settings.database_url,
        echo=settings.db_echo if echo is None else echo,
        future=True,
    )


def make_session_factory(engine) -> sessionmaker[Session]:
    """Create a configured session factory."""

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


try:
    engine = make_engine()
    SessionLocal = make_session_factory(engine)
except ModuleNotFoundError:
    engine = None
    SessionLocal = None


def get_db() -> Generator[Session, None, None]:
    """Yield a scoped database session for FastAPI dependencies."""

    if SessionLocal is None:
        raise RuntimeError("Database driver not available for configured database_url")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
