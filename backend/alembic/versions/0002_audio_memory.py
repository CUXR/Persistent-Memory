"""Wearer voice profiles and retained audio processing jobs."""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy.vector

revision = '0002_audio_memory'
down_revision = '0001_baseline'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('audio_jobs',
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('person_id', sa.Uuid(), nullable=False),
    sa.Column('recording', sa.JSON(), nullable=False),
    sa.Column('status', sa.String(length=30), nullable=False),
    sa.Column('stage', sa.String(length=30), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('attempt_id', sa.Uuid(), nullable=True),
    sa.Column('dialog', sa.JSON(), nullable=True),
    sa.Column('transcript', sa.Text(), nullable=False),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.Column('episode_id', sa.Uuid(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], name=op.f('fk_audio_jobs_episode_id_episodes')),
    sa.ForeignKeyConstraint(['person_id'], ['people.id'], name=op.f('fk_audio_jobs_person_id_people')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_audio_jobs_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audio_jobs'))
    )
    op.create_index(op.f('ix_audio_jobs_person_id'), 'audio_jobs', ['person_id'], unique=False)
    op.create_index(op.f('ix_audio_jobs_user_id'), 'audio_jobs', ['user_id'], unique=False)
    op.add_column('users', sa.Column('voice_embedding', pgvector.sqlalchemy.vector.VECTOR(dim=512).with_variant(sa.JSON(), 'sqlite'), nullable=True))
    op.add_column('users', sa.Column('voice_embedding_model', sa.String(length=100), nullable=True))


def downgrade():
    op.drop_column('users', 'voice_embedding_model')
    op.drop_column('users', 'voice_embedding')
    op.drop_index(op.f('ix_audio_jobs_user_id'), table_name='audio_jobs')
    op.drop_index(op.f('ix_audio_jobs_person_id'), table_name='audio_jobs')
    op.drop_table('audio_jobs')
