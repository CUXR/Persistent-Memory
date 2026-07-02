"""Repo-root pytest configuration.

The `pytest_addoption` hook MUST live at or above the rootdir so the option
is registered before pytest parses command-line arguments. A conftest.py
inside ``backend/tests/`` is discovered too late for ``--runslow`` to be
visible to argparse, even though `pytest_collection_modifyitems` from that
conftest still runs during collection.

Test environment setup (env vars, fixtures) stays in
``backend/tests/conftest.py`` so it loads only when those tests are
collected.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (load real ML models, etc.)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
