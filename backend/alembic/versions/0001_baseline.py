"""Baseline schema before durable audio processing."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy.vector

revision = '0001_baseline'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('users',
    sa.Column('first_name', sa.String(length=100), nullable=False),
    sa.Column('last_name', sa.String(length=100), nullable=False),
    sa.Column('display_name', sa.String(length=200), nullable=True),
    sa.Column('username', sa.String(length=100), nullable=False),
    sa.Column('oauth_provider', sa.String(length=100), nullable=True),
    sa.Column('oauth_subject', sa.String(length=255), nullable=True),
    sa.Column('preferences', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users'))
    )
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('people',
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('first_name', sa.String(length=100), nullable=False),
    sa.Column('last_name', sa.String(length=100), nullable=False),
    sa.Column('display_name', sa.String(length=200), nullable=True),
    sa.Column('face_embedding', pgvector.sqlalchemy.vector.VECTOR(dim=512).with_variant(sa.JSON(), 'sqlite'), nullable=True),
    sa.Column('voice_embedding', pgvector.sqlalchemy.vector.VECTOR(dim=512).with_variant(sa.JSON(), 'sqlite'), nullable=True),
    sa.Column('face_embedding_model', sa.String(length=100), nullable=True),
    sa.Column('voice_embedding_model', sa.String(length=100), nullable=True),
    sa.Column('face_key', sa.String(length=255), nullable=True),
    sa.Column('voice_key', sa.String(length=255), nullable=True),
    sa.Column('persona90', sa.ARRAY(sa.Float()).with_variant(sa.JSON(), "sqlite"), nullable=False),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_people_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_people'))
    )
    op.create_index(op.f('ix_people_last_seen_at'), 'people', ['last_seen_at'], unique=False)
    op.create_index(op.f('ix_people_user_id'), 'people', ['user_id'], unique=False)
    op.create_table('user_facts',
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('fact_text', sa.Text(), nullable=False),
    sa.Column('source', sa.String(length=255), nullable=True),
    sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_facts_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_facts'))
    )
    op.create_index(op.f('ix_user_facts_user_id'), 'user_facts', ['user_id'], unique=False)
    op.create_table('episodes',
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
    sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
    sa.Column('transcript', sa.Text(), nullable=False),
    sa.Column('dialogue_summary', sa.Text(), nullable=False),
    sa.Column('summary_version', sa.String(length=100), nullable=True),
    sa.Column('importance_score', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_episodes_person_id_people'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_episodes_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_episodes'))
    )
    op.create_index(op.f('ix_episodes_person_id'), 'episodes', ['person_id'], unique=False)
    op.create_index('ix_episodes_person_id_start_time', 'episodes', ['person_id', 'start_time'], unique=False)
    op.create_index(op.f('ix_episodes_user_id'), 'episodes', ['user_id'], unique=False)
    op.create_index('ix_episodes_user_id_start_time', 'episodes', ['user_id', 'start_time'], unique=False)
    op.create_table('person_aliases',
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('alias', sa.String(length=200), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_person_aliases_person_id_people'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_person_aliases'))
    )
    op.create_index('ix_person_aliases_alias_ci_unique', 'person_aliases', [sa.text('lower(alias)')], unique=True)
    op.create_index(op.f('ix_person_aliases_person_id'), 'person_aliases', ['person_id'], unique=False)
    op.create_table('episode_participants',
    sa.Column('episode_id', sa.Uuid(), nullable=False),
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], name=op.f('fk_episode_participants_episode_id_episodes'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_episode_participants_person_id_people'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('episode_id', 'person_id', name=op.f('pk_episode_participants'))
    )
    op.create_table('person_edges',
    sa.Column('src_id', sa.Uuid(), nullable=False),
    sa.Column('relation', sa.String(length=100), nullable=False),
    sa.Column('dst_id', sa.Uuid(), nullable=False),
    sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('episode_id', sa.Uuid(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['dst_id'], ['people.id'], name=op.f('fk_person_edges_dst_id_people'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], name=op.f('fk_person_edges_episode_id_episodes'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['src_id'], ['people.id'], name=op.f('fk_person_edges_src_id_people'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_person_edges')),
    sa.UniqueConstraint('src_id', 'relation', 'dst_id', name='uq_person_edges_src_relation_dst')
    )
    op.create_index(op.f('ix_person_edges_dst_id'), 'person_edges', ['dst_id'], unique=False)
    op.create_index(op.f('ix_person_edges_episode_id'), 'person_edges', ['episode_id'], unique=False)
    op.create_index(op.f('ix_person_edges_src_id'), 'person_edges', ['src_id'], unique=False)
    op.create_table('person_facts',
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('source_episode_id', sa.Uuid(), nullable=True),
    sa.Column('fact_text', sa.Text(), nullable=False),
    sa.Column('fact_category', sa.String(length=50), nullable=True),
    sa.Column('source', sa.String(length=255), nullable=True),
    sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('valid_from', sa.DateTime(timezone=True), nullable=True),
    sa.Column('valid_to', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("fact_category IN ('visual_descriptor', 'affiliation', 'hobby')", name=op.f('ck_person_facts_ck_person_facts_fact_category')),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_person_facts_person_id_people'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_episode_id'], ['episodes.id'], name=op.f('fk_person_facts_source_episode_id_episodes'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_person_facts'))
    )
    op.create_index(op.f('ix_person_facts_person_id'), 'person_facts', ['person_id'], unique=False)
    op.create_index(op.f('ix_person_facts_source_episode_id'), 'person_facts', ['source_episode_id'], unique=False)
    op.create_table('person_prefs',
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('pref_text', sa.Text(), nullable=False),
    sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('episode_id', sa.Uuid(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], name=op.f('fk_person_prefs_episode_id_episodes'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_person_prefs_person_id_people'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_person_prefs'))
    )
    op.create_index(op.f('ix_person_prefs_episode_id'), 'person_prefs', ['episode_id'], unique=False)
    op.create_index(op.f('ix_person_prefs_person_id'), 'person_prefs', ['person_id'], unique=False)
    op.create_table('person_summaries',
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('summary_text', sa.Text(), nullable=False),
    sa.Column('episode_time_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('episode_time_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('episode_id', sa.Uuid(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], name=op.f('fk_person_summaries_episode_id_episodes'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_person_summaries_person_id_people'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_person_summaries'))
    )
    op.create_index(op.f('ix_person_summaries_episode_id'), 'person_summaries', ['episode_id'], unique=False)
    op.create_index(op.f('ix_person_summaries_person_id'), 'person_summaries', ['person_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_person_summaries_person_id'), table_name='person_summaries')
    op.drop_index(op.f('ix_person_summaries_episode_id'), table_name='person_summaries')
    op.drop_table('person_summaries')
    op.drop_index(op.f('ix_person_prefs_person_id'), table_name='person_prefs')
    op.drop_index(op.f('ix_person_prefs_episode_id'), table_name='person_prefs')
    op.drop_table('person_prefs')
    op.drop_index(op.f('ix_person_facts_source_episode_id'), table_name='person_facts')
    op.drop_index(op.f('ix_person_facts_person_id'), table_name='person_facts')
    op.drop_table('person_facts')
    op.drop_index(op.f('ix_person_edges_src_id'), table_name='person_edges')
    op.drop_index(op.f('ix_person_edges_episode_id'), table_name='person_edges')
    op.drop_index(op.f('ix_person_edges_dst_id'), table_name='person_edges')
    op.drop_table('person_edges')
    op.drop_table('episode_participants')
    op.drop_index('ix_person_aliases_alias_ci_unique', table_name='person_aliases')
    op.drop_index(op.f('ix_person_aliases_person_id'), table_name='person_aliases')
    op.drop_table('person_aliases')
    op.drop_index('ix_episodes_user_id_start_time', table_name='episodes')
    op.drop_index(op.f('ix_episodes_user_id'), table_name='episodes')
    op.drop_index('ix_episodes_person_id_start_time', table_name='episodes')
    op.drop_index(op.f('ix_episodes_person_id'), table_name='episodes')
    op.drop_table('episodes')
    op.drop_index(op.f('ix_user_facts_user_id'), table_name='user_facts')
    op.drop_table('user_facts')
    op.drop_index(op.f('ix_people_user_id'), table_name='people')
    op.drop_index(op.f('ix_people_last_seen_at'), table_name='people')
    op.drop_table('people')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_table('users')
