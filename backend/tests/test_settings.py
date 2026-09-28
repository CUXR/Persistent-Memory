"""Settings parsing that must go through the real environment / .env sources."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings, parse_cors_origins


def test_cors_origins_default_is_the_two_dev_origins(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    assert Settings(_env_file=None).cors_allowed_origins_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_cors_origins_accept_comma_separated_env_value(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", " http://a:1 , http://b:2 ,")

    assert Settings(_env_file=None).cors_allowed_origins_list == ["http://a:1", "http://b:2"]


def test_cors_origins_accept_json_list_env_value(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", '["http://a:1", "http://b:2"]')

    assert Settings(_env_file=None).cors_allowed_origins_list == ["http://a:1", "http://b:2"]


def test_cors_origins_reject_wildcard(monkeypatch):
    for value in ("*", "http://a:1,*", '["*"]'):
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", value)
        with pytest.raises(ValidationError, match="explicit origins"):
            Settings(_env_file=None)


def test_shipped_env_example_loads(tmp_path, monkeypatch):
    """Copying backend/.env.example to .env must not crash startup."""

    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("ALLOW_DEV_AUTH_FALLBACK", raising=False)
    example = (BACKEND_ROOT / ".env.example").read_text()
    env_file = tmp_path / ".env"
    env_file.write_text(example.replace("postgresql+psycopg://user:password@host:5432/persistent_memory", "sqlite+pysqlite:///:memory:"))

    settings = Settings(_env_file=env_file)

    assert settings.cors_allowed_origins_list == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert settings.allow_dev_auth_fallback is False
    assert settings.database_url == "sqlite+pysqlite:///:memory:"


def test_dev_auth_fallback_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv("ALLOW_DEV_AUTH_FALLBACK", raising=False)

    assert Settings(_env_file=None).allow_dev_auth_fallback is False


def test_dev_auth_fallback_parses_booleans(monkeypatch):
    monkeypatch.setenv("ALLOW_DEV_AUTH_FALLBACK", "true")
    assert Settings(_env_file=None).allow_dev_auth_fallback is True
    monkeypatch.setenv("ALLOW_DEV_AUTH_FALLBACK", "0")
    assert Settings(_env_file=None).allow_dev_auth_fallback is False


def test_parse_cors_origins_helper():
    assert parse_cors_origins("") == []
    assert parse_cors_origins(["http://a:1", " http://b:2 "]) == ["http://a:1", "http://b:2"]
    with pytest.raises(ValueError):
        parse_cors_origins("[not json")
