"""Add missing schema tables/columns and PersonFact.fact_category.

The initial migration (20260309_0001) created the core tables but omitted:
  - episodes.transcript
  - people.face_key, voice_key, persona90
  - person_aliases, episode_participants, person_prefs, person_summaries, person_edges

This migration adds all of the above plus the new fact_category column on
person_facts as defined in DB_STRUCTURE.md (values: visual_descriptor,
affiliation, hobby).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

settings = get_settings()

# revision identifiers, used by Alembic.
revision = "20260316_0002"
down_revision = "20260309_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # Extend existing tables with missing columns                         #
    # ------------------------------------------------------------------ #

    # episodes — transcript was in the model but not the initial migration
    op.add_column(
        "episodes",
        sa.Column("transcript", sa.Text(), nullable=False, server_default=""),
    )
    # Remove the server_default now that existing rows have been back-filled
    op.alter_column("episodes", "transcript", server_default=None)

    # people — face_key, voice_key, and persona90 vector
    op.add_column("people", sa.Column("face_key", sa.String(255), nullable=True))
    op.add_column("people", sa.Column("voice_key", sa.String(255), nullable=True))
    op.add_column(
        "people",
        sa.Column(
            "persona90",
            postgresql.ARRAY(sa.Float()),
            nullable=True,
        ),
    )

    # person_facts — fact_category with CHECK constraint
    op.add_column(
        "person_facts",
        sa.Column("fact_category", sa.String(50), nullable=True),
    )
    op.create_check_constraint(
        "ck_person_facts_fact_category",
        "person_facts",
        "fact_category IN ('visual_descriptor', 'affiliation', 'hobby')",
    )

    # person_facts — valid_from / valid_to were also missing from the initial migration
    op.add_column(
        "person_facts",
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "person_facts",
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
    )

    # ------------------------------------------------------------------ #
    # New tables                                                          #
    # ------------------------------------------------------------------ #

    op.create_table(
        "person_aliases",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alias", sa.String(200), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_person_aliases_person_id_people"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_aliases")),
    )
    op.create_index(op.f("ix_person_aliases_person_id"), "person_aliases", ["person_id"], unique=False)
    # Case-insensitive unique index on alias
    op.create_index(
        "ix_person_aliases_alias_ci_unique",
        "person_aliases",
        [sa.text("lower(alias)")],
        unique=True,
    )

    op.create_table(
        "episode_participants",
        sa.Column("episode_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.id"],
            name=op.f("fk_episode_participants_episode_id_episodes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_episode_participants_person_id_people"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("episode_id", "person_id", name=op.f("pk_episode_participants")),
    )

    op.create_table(
        "person_prefs",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pref_text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("episode_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.id"],
            name=op.f("fk_person_prefs_episode_id_episodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_person_prefs_person_id_people"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_prefs")),
    )
    op.create_index(op.f("ix_person_prefs_person_id"), "person_prefs", ["person_id"], unique=False)
    op.create_index(op.f("ix_person_prefs_episode_id"), "person_prefs", ["episode_id"], unique=False)

    op.create_table(
        "person_summaries",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("episode_time_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("episode_time_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("episode_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.id"],
            name=op.f("fk_person_summaries_episode_id_episodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_person_summaries_person_id_people"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_summaries")),
    )
    op.create_index(op.f("ix_person_summaries_person_id"), "person_summaries", ["person_id"], unique=False)
    op.create_index(op.f("ix_person_summaries_episode_id"), "person_summaries", ["episode_id"], unique=False)

    op.create_table(
        "person_edges",
        sa.Column("src_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation", sa.String(100), nullable=False),
        sa.Column("dst_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("episode_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dst_id"],
            ["people.id"],
            name=op.f("fk_person_edges_dst_id_people"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.id"],
            name=op.f("fk_person_edges_episode_id_episodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["src_id"],
            ["people.id"],
            name=op.f("fk_person_edges_src_id_people"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_edges")),
        sa.UniqueConstraint("src_id", "relation", "dst_id", name="uq_person_edges_src_relation_dst"),
    )
    op.create_index(op.f("ix_person_edges_src_id"), "person_edges", ["src_id"], unique=False)
    op.create_index(op.f("ix_person_edges_dst_id"), "person_edges", ["dst_id"], unique=False)
    op.create_index(op.f("ix_person_edges_episode_id"), "person_edges", ["episode_id"], unique=False)


def downgrade() -> None:
    # Drop new tables (reverse order of creation)
    op.drop_index(op.f("ix_person_edges_episode_id"), table_name="person_edges")
    op.drop_index(op.f("ix_person_edges_dst_id"), table_name="person_edges")
    op.drop_index(op.f("ix_person_edges_src_id"), table_name="person_edges")
    op.drop_table("person_edges")

    op.drop_index(op.f("ix_person_summaries_episode_id"), table_name="person_summaries")
    op.drop_index(op.f("ix_person_summaries_person_id"), table_name="person_summaries")
    op.drop_table("person_summaries")

    op.drop_index(op.f("ix_person_prefs_episode_id"), table_name="person_prefs")
    op.drop_index(op.f("ix_person_prefs_person_id"), table_name="person_prefs")
    op.drop_table("person_prefs")

    op.drop_table("episode_participants")

    op.drop_index("ix_person_aliases_alias_ci_unique", table_name="person_aliases")
    op.drop_index(op.f("ix_person_aliases_person_id"), table_name="person_aliases")
    op.drop_table("person_aliases")

    # Remove added columns (reverse order)
    op.drop_column("person_facts", "valid_to")
    op.drop_column("person_facts", "valid_from")
    op.drop_constraint("ck_person_facts_fact_category", "person_facts", type_="check")
    op.drop_column("person_facts", "fact_category")

    op.drop_column("people", "persona90")
    op.drop_column("people", "voice_key")
    op.drop_column("people", "face_key")

    op.drop_column("episodes", "transcript")
