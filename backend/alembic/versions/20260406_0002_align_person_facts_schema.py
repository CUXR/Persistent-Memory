"""Align person_facts with retrieval metadata schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

settings = get_settings()

# revision identifiers, used by Alembic.
revision = "20260406_0002"
down_revision = "20260309_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "person_facts",
        sa.Column("fact_category", sa.String(length=50), nullable=False, server_default="general"),
    )
    op.add_column(
        "person_facts",
        sa.Column("embedding", Vector(dim=settings.retrieval_embedding_dimension), nullable=True),
    )
    op.add_column(
        "person_facts",
        sa.Column("source_tmp", postgresql.UUID(as_uuid=True), nullable=True),
    )

    op.execute("UPDATE person_facts SET source_tmp = source_episode_id")

    op.drop_constraint(
        op.f("fk_person_facts_source_episode_id_episodes"),
        "person_facts",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_person_facts_source_episode_id"), table_name="person_facts")
    op.drop_column("person_facts", "source_episode_id")
    op.drop_column("person_facts", "source")

    op.alter_column(
        "person_facts",
        "source_tmp",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
        new_column_name="source",
    )
    op.create_foreign_key(
        op.f("fk_person_facts_source_episodes"),
        "person_facts",
        "episodes",
        ["source"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(op.f("ix_person_facts_source"), "person_facts", ["source"], unique=False)
    op.alter_column("person_facts", "fact_category", server_default=None)


def downgrade() -> None:
    op.add_column(
        "person_facts",
        sa.Column("source_label_tmp", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "person_facts",
        sa.Column("source_episode_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    op.execute("UPDATE person_facts SET source_episode_id = source")

    op.create_foreign_key(
        op.f("fk_person_facts_source_episode_id_episodes"),
        "person_facts",
        "episodes",
        ["source_episode_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_person_facts_source_episode_id"),
        "person_facts",
        ["source_episode_id"],
        unique=False,
    )

    op.drop_constraint(
        op.f("fk_person_facts_source_episodes"),
        "person_facts",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_person_facts_source"), table_name="person_facts")
    op.drop_column("person_facts", "source")
    op.alter_column(
        "person_facts",
        "source_label_tmp",
        existing_type=sa.String(length=255),
        nullable=True,
        new_column_name="source",
    )
    op.drop_column("person_facts", "embedding")
    op.drop_column("person_facts", "fact_category")
