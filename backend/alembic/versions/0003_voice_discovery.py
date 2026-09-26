"""Track voice profiles learned naturally from a conversation."""
from alembic import op
import sqlalchemy as sa

revision = "0003_voice_discovery"
down_revision = "0002_audio_memory"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("audio_jobs", sa.Column("voice_discovery", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("audio_jobs", "voice_discovery")
