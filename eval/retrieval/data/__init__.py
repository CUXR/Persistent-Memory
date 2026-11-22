"""Synthetic conversation dataset: generation scripts and the committed JSONL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """The two messages of one model call, as the corpus and query prompt modules build them."""

    system: str
    user: str
