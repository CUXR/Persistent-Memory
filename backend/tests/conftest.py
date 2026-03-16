"""Pytest configuration and shared fixtures.

Environment variables are set here at the very top — before any app module is
imported — so that pydantic-settings can validate required fields during
collection.  These values are test-only stubs; the MemoryStore tests always
pass their own ``db_url`` directly, and LLM calls are fully mocked.
"""

import os

# Provide the required settings fields so pydantic-settings validates cleanly.
# Tests always override these at the MemoryStore/LLMClient constructor level,
# so these stubs are never actually used to open a connection or make API calls.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-tests")
