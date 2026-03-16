"""OpenAI wrapper for the conversation ingestion pipeline.

Encapsulates all SDK interaction so the ingestion service stays decoupled from
the underlying LLM provider.  Both public methods use OpenAI's structured-output
feature, parsing responses directly into Pydantic models for type-safe consumption.

Prompt constants are module-level so they can be tuned without touching service
logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from ..core.config import get_settings
from ..schema.ingestion import (
    EpisodeSummaryLLMResponse,
    FactExtractionLLMResponse,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("app.services.llm_client")

# ---------------------------------------------------------------------------
# Prompt templates — edit here to tune extraction behaviour
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "You are the memory system for a pair of smart glasses worn by {wearer_name}. "
    "You receive raw conversation transcripts and produce concise, factual episode summaries.\n\n"
    "Guidelines:\n"
    "- Summarise the key topics, decisions, and any action items in 2-4 sentences.\n"
    "- Write in third person (e.g. '{wearer_name} discussed …').\n"
    "- Rate importance_score: 0.0 = trivial small talk, 1.0 = life-changing event.\n"
    "- Be objective; do not infer emotions unless explicitly stated."
)

_SUMMARY_USER = (
    "Conversation between {wearer_name} and {interlocutor_name}:\n\n{transcript}"
)

_FACTS_SYSTEM = (
    "You are an information extraction engine for a personal memory assistant.\n\n"
    "Your task: extract NEW facts about {interlocutor_name} from the conversation below.\n\n"
    "Categories (use exactly one per fact):\n"
    "  visual_descriptor – physical appearance, distinguishing features, style\n"
    "  affiliation       – employers, companies, clubs, teams, or groups they belong to\n"
    "  hobby             – leisure activities, sports, interests, and topics they care about\n\n"
    "Shared-interest detection:\n"
    "  The wearer's known facts are listed below. Set is_shared_interest=true for any "
    "fact where {interlocutor_name} shares the same hobby or interest as the wearer.\n\n"
    "Deduplication:\n"
    "  The following facts about {interlocutor_name} are ALREADY stored — do NOT re-extract them:\n"
    "{existing_facts_block}\n\n"
    "Wearer's facts (for shared-interest detection):\n"
    "{user_facts_block}\n\n"
    "Relationship extraction:\n"
    "  Also extract directed relationships using snake_case labels "
    "(e.g. works_at, member_of, knows, is_married_to). "
    "Only include relationships that are clearly stated or strongly implied.\n\n"
    "Return ONLY facts not already listed in the deduplication block."
)

_FACTS_USER = (
    "Conversation between {wearer_name} and {interlocutor_name}:\n\n{transcript}"
)


def _bullet_list(items: list[str]) -> str:
    """Format a list as a markdown bullet list, or '(none)' when empty."""
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


class LLMClient:
    """Async OpenAI client scoped to the ingestion pipeline.

    Args:
        api_key: OpenAI secret key.  Defaults to ``settings.openai_api_key``.
        model: Model identifier.  Defaults to ``settings.openai_model``.
        max_retries: SDK-level retry count for transient network errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.openai_model
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            max_retries=max_retries if max_retries is not None else settings.openai_max_retries,
        )
        logger.info("LLMClient initialised (model=%s)", self._model)

    async def generate_summary(
        self,
        transcript: str,
        wearer_name: str,
        interlocutor_name: str,
    ) -> EpisodeSummaryLLMResponse:
        """Generate a concise episode summary with an importance score.

        Args:
            transcript: Raw conversation text in ``Speaker: utterance`` format.
            wearer_name: Display name of the smart-glasses wearer.
            interlocutor_name: Display name of the primary conversation partner.

        Returns:
            Parsed :class:`~app.schema.ingestion.EpisodeSummaryLLMResponse`.
        """

        system_msg = _SUMMARY_SYSTEM.format(
            wearer_name=wearer_name,
            interlocutor_name=interlocutor_name,
        )
        user_msg = _SUMMARY_USER.format(
            wearer_name=wearer_name,
            interlocutor_name=interlocutor_name,
            transcript=transcript,
        )

        logger.debug("generate_summary: calling %s for interlocutor=%s", self._model, interlocutor_name)

        completion = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format=EpisodeSummaryLLMResponse,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("OpenAI returned a null parsed result for summary generation")

        logger.debug(
            "generate_summary: importance_score=%.2f summary_len=%d",
            result.importance_score,
            len(result.summary),
        )
        return result

    async def extract_facts(
        self,
        transcript: str,
        wearer_name: str,
        interlocutor_name: str,
        existing_facts: list[str],
        user_facts: list[str],
    ) -> FactExtractionLLMResponse:
        """Extract candidate facts and relationships about the interlocutor.

        The prompt includes both the wearer's stored facts (for shared-interest
        flagging) and the interlocutor's existing facts (for deduplication).

        Args:
            transcript: Raw conversation text.
            wearer_name: Display name of the smart-glasses wearer.
            interlocutor_name: Display name of the conversation partner.
            existing_facts: ``fact_text`` values already stored for the interlocutor.
            user_facts: ``fact_text`` values stored for the wearer (owner).

        Returns:
            Parsed :class:`~app.schema.ingestion.FactExtractionLLMResponse`.
        """

        system_msg = _FACTS_SYSTEM.format(
            interlocutor_name=interlocutor_name,
            existing_facts_block=_bullet_list(existing_facts),
            user_facts_block=_bullet_list(user_facts),
        )
        user_msg = _FACTS_USER.format(
            wearer_name=wearer_name,
            interlocutor_name=interlocutor_name,
            transcript=transcript,
        )

        logger.debug(
            "extract_facts: calling %s (existing=%d, user_facts=%d)",
            self._model,
            len(existing_facts),
            len(user_facts),
        )

        completion = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format=FactExtractionLLMResponse,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("OpenAI returned a null parsed result for fact extraction")

        logger.debug(
            "extract_facts: %d facts, %d edges extracted",
            len(result.facts),
            len(result.edges),
        )
        return result
