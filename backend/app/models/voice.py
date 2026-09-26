"""Voice vectors use their own fixed dimension, independent of face settings."""
from pgvector.sqlalchemy import Vector
from sqlalchemy.types import JSON

from audio_pipeline.segment import VOICE_DIMENSION

VOICE_VECTOR_TYPE = Vector(VOICE_DIMENSION).with_variant(JSON(), "sqlite")
