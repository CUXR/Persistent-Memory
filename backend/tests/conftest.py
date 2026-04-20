"""Pytest configuration and shared fixtures.

Environment variables are set here at the very top — before any app module is
imported — so that pydantic-settings can validate required fields during
collection. These values are test-only stubs; the MemoryStore tests always
pass their own ``db_url`` directly, and LLM calls are fully mocked.

This file also includes a tiny async-test runner so ``async def`` tests can
run without requiring ``pytest-asyncio`` to be installed in every environment.
"""

import asyncio
import inspect
import os

import pytest

# Provide the required settings fields so pydantic-settings validates cleanly.
# Tests always override these at the MemoryStore/LLMClient constructor level,
# so these stubs are never actually used to open a connection or make API calls.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-tests")


def pytest_configure(config):
    """Register the asyncio marker used by async ingestion tests."""

    config.addinivalue_line("markers", "asyncio: mark test to run in an asyncio event loop")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run coroutine tests with ``asyncio.run`` when no async plugin is present."""

    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    funcargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(test_function(**funcargs))
    return True
