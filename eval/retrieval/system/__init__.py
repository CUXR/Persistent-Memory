"""The system under test: the query flow from the ticket.

extract (one LLM call) -> resolve the name (backend ``PersonResolver``) -> pad
the time range -> rank -> answer from summaries, falling back to dialogues.
"""

from __future__ import annotations

from .answering import AnswerAttempt, answer_from_summaries, answer_with_dialogues
from .extraction import Extraction, extract
from .pipeline import (
    AMBIGUOUS_NAME_TEMPLATE,
    STAGES,
    Answered,
    Prepared,
    SystemPipeline,
    SystemResult,
)
from .resolver import Resolution, resolve_name

__all__ = [
    "AMBIGUOUS_NAME_TEMPLATE",
    "STAGES",
    "AnswerAttempt",
    "Answered",
    "Extraction",
    "Prepared",
    "Resolution",
    "SystemPipeline",
    "SystemResult",
    "answer_from_summaries",
    "answer_with_dialogues",
    "extract",
    "resolve_name",
]
