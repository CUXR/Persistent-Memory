"""Shared setup for the eval unit tests.

These tests must run in the plain backend environment: no torch, no model
download, no database, no network. Anything heavier is injected as a fake.
"""

import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Never read a developer's eval/.env (real keys) from unit tests.
os.environ["EVAL_DISABLE_DOTENV"] = "1"
# The backend settings object needs these to import; nothing connects to them.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-tests")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail fast if a unit test tries to open a network connection."""

    def _blocked(self, address, *args, **kwargs):
        raise RuntimeError(f"eval unit tests must not use the network (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
