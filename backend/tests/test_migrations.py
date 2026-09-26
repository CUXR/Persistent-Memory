"""Exercise real Alembic upgrades separately from create_all-based fixtures."""
import os
from pathlib import Path
import subprocess
import sys

from sqlalchemy import create_engine, inspect, text


ROOT = Path(__file__).resolve().parents[2]


def migrate(url, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "backend/alembic.ini", *arguments],
        cwd=ROOT, env={**os.environ, "DATABASE_URL": url},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_empty_database_upgrade_and_downgrade(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'fresh.sqlite'}"
    migrate(url, "upgrade", "head")
    engine = create_engine(url)
    assert "audio_jobs" in inspect(engine).get_table_names()
    assert "voice_discovery" in {column["name"] for column in inspect(engine).get_columns("audio_jobs")}
    assert "voice_embedding" in {column["name"] for column in inspect(engine).get_columns("users")}
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0003_voice_discovery"
    engine.dispose()
    seed = subprocess.run(
        [sys.executable, "backend/scripts/seed_memory_store.py"], cwd=ROOT,
        env={**os.environ, "DATABASE_URL": url}, capture_output=True, text=True,
    )
    assert seed.returncode == 0, seed.stderr
    assert "Seeded person:" in seed.stdout
    migrate(url, "downgrade", "base")
    engine = create_engine(url)
    assert inspect(engine).get_table_names() == ["alembic_version"]
    engine.dispose()


def test_audio_upgrade_preserves_existing_users_and_partner_voices(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'existing.sqlite'}"
    migrate(url, "upgrade", "0001_baseline")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO users (id, first_name, last_name, username, preferences, created_at, updated_at)
            VALUES ('11111111111111111111111111111111', 'Seed', 'Owner', 'seed', '{}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """))
        connection.execute(text("""
            INSERT INTO people (id, user_id, first_name, last_name, persona90, voice_embedding,
                                voice_embedding_model, created_at, updated_at)
            VALUES ('22222222222222222222222222222222', '11111111111111111111111111111111',
                    'Emily', 'Chen', '[]', '[1, 0]', 'legacy', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """))
    engine.dispose()
    migrate(url, "upgrade", "head")
    engine = create_engine(url)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT username FROM users")) == "seed"
        assert connection.scalar(text("SELECT voice_embedding FROM users")) is None
        assert connection.scalar(text("SELECT voice_embedding FROM people")) == "[1, 0]"
        assert connection.scalar(text("SELECT voice_embedding_model FROM people")) == "legacy"
    engine.dispose()


def test_postgres_migration_compiles_vector_schema():
    sql = migrate("postgresql+psycopg://unused:unused@localhost/unused", "upgrade", "head", "--sql")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE audio_jobs" in sql
    assert "ALTER TABLE users ADD COLUMN voice_embedding VECTOR(512)" in sql


def test_audio_cli_help_does_not_require_database_configuration():
    environment = {key: value for key, value in os.environ.items() if key != "DATABASE_URL"}
    result = subprocess.run(
        [sys.executable, "-m", "app.cli.audio", "--help"], cwd=ROOT,
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "capture-file" in result.stdout
