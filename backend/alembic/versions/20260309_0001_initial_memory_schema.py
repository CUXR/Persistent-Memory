"""Initial persistent memory schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

settings = get_settings()

# revision identifiers, used by Alembic.
revision = "20260309_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("first_name", sa.String(length=100), nullable=False),
        sa.Column("last_name", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("oauth_provider", sa.String(length=100), nullable=True),
        sa.Column("oauth_subject", sa.String(length=255), nullable=True),
        sa.Column(
            "preferences",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("oauth_provider", "oauth_subject", name="uq_users_oauth_identity"),
    )
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    op.create_table(
        "people",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("first_name", sa.String(length=100), nullable=False),
        sa.Column("last_name", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column("face_embedding", Vector(dim=settings.embedding_dimension), nullable=True),
        sa.Column("voice_embedding", Vector(dim=settings.embedding_dimension), nullable=True),
        sa.Column("face_embedding_model", sa.String(length=100), nullable=True),
        sa.Column("voice_embedding_model", sa.String(length=100), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_people_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_people")),
    )
    op.create_index(op.f("ix_people_user_id"), "people", ["user_id"], unique=False)
    op.create_index(op.f("ix_people_last_seen_at"), "people", ["last_seen_at"], unique=False)

    op.create_table(
        "episodes",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dialogue_summary", sa.Text(), nullable=False),
        sa.Column("summary_version", sa.String(length=100), nullable=True),
        sa.Column("importance_score", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["person_id"], ["people.id"], name=op.f("fk_episodes_person_id_people"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_episodes_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_episodes")),
    )
    op.create_index(op.f("ix_episodes_person_id"), "episodes", ["person_id"], unique=False)
    op.create_index(op.f("ix_episodes_person_id_start_time"), "episodes", ["person_id", "start_time"], unique=False)
    op.create_index(op.f("ix_episodes_user_id"), "episodes", ["user_id"], unique=False)
    op.create_index(op.f("ix_episodes_user_id_start_time"), "episodes", ["user_id", "start_time"], unique=False)

    op.create_table(
        "person_facts",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_episode_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("fact_text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["person_id"], ["people.id"], name=op.f("fk_person_facts_person_id_people"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_episode_id"],
            ["episodes.id"],
            name=op.f("fk_person_facts_source_episode_id_episodes"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_facts")),
    )
    op.create_index(op.f("ix_person_facts_person_id"), "person_facts", ["person_id"], unique=False)
    op.create_index(op.f("ix_person_facts_source_episode_id"), "person_facts", ["source_episode_id"], unique=False)

    op.create_table(
        "user_facts",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fact_text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_user_facts_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_facts")),
    )
    op.create_index(op.f("ix_user_facts_user_id"), "user_facts", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_user_facts_user_id"), table_name="user_facts")
    op.drop_table("user_facts")

    op.drop_index(op.f("ix_person_facts_source_episode_id"), table_name="person_facts")
    op.drop_index(op.f("ix_person_facts_person_id"), table_name="person_facts")
    op.drop_table("person_facts")

    op.drop_index(op.f("ix_episodes_user_id_start_time"), table_name="episodes")
    op.drop_index(op.f("ix_episodes_user_id"), table_name="episodes")
    op.drop_index(op.f("ix_episodes_person_id_start_time"), table_name="episodes")
    op.drop_index(op.f("ix_episodes_person_id"), table_name="episodes")
    op.drop_table("episodes")

    op.drop_index(op.f("ix_people_last_seen_at"), table_name="people")
    op.drop_index(op.f("ix_people_user_id"), table_name="people")
    op.drop_table("people")

    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_table("users")

    op.execute("DROP EXTENSION IF EXISTS vector")
