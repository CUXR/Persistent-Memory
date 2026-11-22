"""The query flow under test: extract -> resolve -> pad -> rank -> answer.

``prepare`` (steps 1-3) and ``answer`` (step 5) are separate and keep no state
between calls, so the runner can prepare a query once and reuse it for every
ranker and every (k, k') cell. ``run`` chains them around one ranker, and
``compose_system_result`` assembles the result from the parts either way.

LLM stage latencies are wall clock, with one correction for the response cache.
A hosted call replayed from the cache returns at once, while its ``Usage`` still
holds the latency measured when the call was really made, so an LLM stage
reports whichever of the two is longer. Latency then follows the accounting the
cache already gives tokens and cost: a resumed run, and the (k, k') cells that
share one summary attempt, report what the stage took when its call was made.
The rank stage is the ranker's own ``trace.latency_s``: the ranking without the
pool-size diagnostic the local rankers take afterwards, so it is the same
quantity a ``rankers``-stage record stores, whether the ranking ran now or is
reused from such a record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

from .. import config
from ..interfaces import ChatLLM, CorpusStore, Ranker, RankResult, Usage
from ..schema import SYSTEM_STAGES, TimeRange
from ..timeutil import pad_time_range
from .answering import AnswerAttempt, answer_from_summaries, answer_with_dialogues
from .extraction import Extraction, extract
from .resolver import Resolution, load_person_resolver, resolve_name

STAGES = SYSTEM_STAGES  # the stages of the flow, in order (schema.SYSTEM_STAGES)
AMBIGUOUS_NAME_TEMPLATE = config.AMBIGUOUS_NAME_TEMPLATE


@dataclass
class Prepared:
    """Steps 1-3 for one query: the extraction, who it resolved to, and the padded time filter."""

    extraction: Extraction
    resolution: Resolution | None  # None when no name was extracted
    person_id: str | None  # the resolved person; None searches all of the owner's conversations
    time_range: TimeRange | None  # padded; None means no time filter
    early_answer: str | None  # the reply when the flow stops here; None when it continues
    path: str | None  # "unknown_name" | "ambiguous"; None when the flow continues
    usage: Usage
    latency_s: float


@dataclass
class Answered:
    """Step 5 for one ranked list at one (k, k')."""

    final_answer: str
    path: str  # "summary" | "fallback" | "declined" | "empty_pool"
    summary_attempt: AnswerAttempt | None  # None for an empty pool
    fallback_attempt: AnswerAttempt | None  # None unless the summary attempt could not answer
    usage_by_stage: dict[str, Usage]  # "answer_summary" and "answer_dialogue"
    latency_by_stage: dict[str, float]  # same keys


@dataclass
class SystemResult:
    """One query through the whole flow with one ranker at one (k, k')."""

    query: str
    final_answer: str
    path: str  # ``Prepared.path`` when the flow stopped early, else ``Answered.path``
    prepared: Prepared
    rank: RankResult | None  # None when the flow stopped before ranking
    answered: Answered | None  # None when the flow stopped before ranking
    usage_by_stage: dict[str, Usage]  # every stage in ``STAGES``; zero for a stage that did not run
    latency_by_stage: dict[str, float]  # same keys; "extraction" is the whole of ``prepare``
    usage: Usage  # total over the stages
    latency_s: float  # total over the stages


class SystemPipeline:
    """Runs the query flow for one owner against a store and a chat model."""

    def __init__(
        self,
        store: CorpusStore,
        llm: ChatLLM,
        system_model: str,
        owner_id: str = config.OWNER_ID,
        now: datetime = config.NOW,
    ) -> None:
        self.store = store
        self.llm = llm
        self.system_model = system_model
        self.owner_id = owner_id
        self.now = now
        # The backend resolver takes seconds to import. Pay for that here rather
        # than inside the measured latency of the first query that names someone.
        load_person_resolver()

    def prepare(self, query: str) -> Prepared:
        """Extract the name and time range, resolve the name, and pad the range.

        An extracted name that matches nobody, or more than one person, ends the
        flow here with ``early_answer`` set.
        """

        started = perf_counter()
        extraction = extract(query, self.llm, self.system_model, now=self.now)
        resolution: Resolution | None = None
        person_id: str | None = None
        early_answer: str | None = None
        path: str | None = None
        if extraction.name is not None:
            resolution = resolve_name(extraction.name, self.store.list_people(self.owner_id))
            if resolution.status == "unknown":
                early_answer = config.UNKNOWN_NAME_TEMPLATE.format(name=extraction.name)
                path = "unknown_name"
            elif resolution.status == "ambiguous":
                candidates = ", ".join(person.display_name for person in resolution.candidates)
                early_answer = AMBIGUOUS_NAME_TEMPLATE.format(
                    name=extraction.name, candidates=candidates
                )
                path = "ambiguous"
            else:
                person_id = resolution.person_id
        time_range: TimeRange | None = None
        if extraction.time_range is not None:
            time_range = pad_time_range(extraction.time_range, self.now)
        return Prepared(
            extraction=extraction,
            resolution=resolution,
            person_id=person_id,
            time_range=time_range,
            early_answer=early_answer,
            path=path,
            usage=extraction.usage,
            latency_s=_llm_stage_latency(started, extraction.usage),
        )

    def answer(self, query: str, ranked_ids: Sequence[str], k: int, k_prime: int) -> Answered:
        """Answer from the first ``k`` summaries, then from the first ``k_prime`` dialogues.

        ``ranked_ids`` is a ranker's output, best first. The dialogue attempt
        runs only when the summary attempt cannot answer; it sees the same
        ``k`` summaries again. ``k_prime`` may exceed ``k``. An empty ranking
        declines without an LLM call.
        """

        if k < 1 or k_prime < 1:
            raise ValueError(f"k and k_prime must be at least 1, got k={k} and k_prime={k_prime}")
        usage = {"answer_summary": Usage(), "answer_dialogue": Usage()}
        latency = {"answer_summary": 0.0, "answer_dialogue": 0.0}
        if not ranked_ids:
            return Answered(
                final_answer=config.DECLINE_ANSWER,
                path="empty_pool",
                summary_attempt=None,
                fallback_attempt=None,
                usage_by_stage=usage,
                latency_by_stage=latency,
            )

        started = perf_counter()
        summary_ids = list(ranked_ids[:k])
        headers_by_id = self.store.get_headers(summary_ids)
        headers = [headers_by_id[episode_id] for episode_id in summary_ids]
        summary_attempt = answer_from_summaries(
            query, headers, self.llm, self.system_model, self.now
        )
        usage["answer_summary"] = summary_attempt.usage
        latency["answer_summary"] = _llm_stage_latency(started, summary_attempt.usage)
        if summary_attempt.can_answer:
            return Answered(
                final_answer=summary_attempt.answer,
                path="summary",
                summary_attempt=summary_attempt,
                fallback_attempt=None,
                usage_by_stage=usage,
                latency_by_stage=latency,
            )

        started = perf_counter()
        dialogue_ids = list(ranked_ids[:k_prime])
        conversations_by_id = self.store.get_conversations(dialogue_ids)
        conversations = [conversations_by_id[episode_id] for episode_id in dialogue_ids]
        fallback_attempt = answer_with_dialogues(
            query, headers, conversations, self.llm, self.system_model, self.now
        )
        usage["answer_dialogue"] = fallback_attempt.usage
        latency["answer_dialogue"] = _llm_stage_latency(started, fallback_attempt.usage)
        if fallback_attempt.can_answer:
            final_answer, path = fallback_attempt.answer, "fallback"
        else:
            final_answer, path = config.DECLINE_ANSWER, "declined"
        return Answered(
            final_answer=final_answer,
            path=path,
            summary_attempt=summary_attempt,
            fallback_attempt=fallback_attempt,
            usage_by_stage=usage,
            latency_by_stage=latency,
        )

    def run(self, query: str, ranker: Ranker, k: int, k_prime: int) -> SystemResult:
        """Run the whole flow for one query: prepare, rank with ``ranker``, answer."""

        prepared = self.prepare(query)
        if prepared.early_answer is not None:
            return compose_system_result(query, prepared)
        rank_result = ranker.rank_with_trace(
            self.owner_id, query, prepared.person_id, prepared.time_range
        )
        ranked_ids = [episode_id for episode_id, _ in rank_result.ranked]
        answered = self.answer(query, ranked_ids, k, k_prime)
        return compose_system_result(query, prepared, rank_result, answered)


def compose_system_result(
    query_text: str,
    prepared: Prepared,
    rank: RankResult | None = None,
    answered: Answered | None = None,
) -> SystemResult:
    """Assemble a ``SystemResult`` from the parts of the flow.

    ``SystemPipeline.run`` uses it around one ranker; the runner prepares a
    query once and answers it per (ranker, k, k') cell, so it calls it per cell.
    The accounting is the same: the extraction stage is the whole of
    ``prepare``, the rank stage is the ranker's own trace latency (see the
    module docstring), and the answer stages come from ``answer``. Without a
    ranking the flow must have stopped early.
    """

    if (rank is None) != (answered is None):
        raise ValueError("a ranking and an answer go together")
    usage = {stage: Usage() for stage in STAGES}
    latency = {stage: 0.0 for stage in STAGES}
    usage["extraction"] = prepared.usage
    latency["extraction"] = prepared.latency_s
    final_answer, path = prepared.early_answer, prepared.path
    if rank is not None and answered is not None:
        latency["rank"] = rank.trace.latency_s
        usage["rank"] = rank.trace.usage
        usage.update(answered.usage_by_stage)
        latency.update(answered.latency_by_stage)
        final_answer, path = answered.final_answer, answered.path
    if final_answer is None or path is None:
        raise ValueError("the flow did not stop early, so it needs a ranking and an answer")
    return SystemResult(
        query=query_text,
        final_answer=final_answer,
        path=path,
        prepared=prepared,
        rank=rank,
        answered=answered,
        usage_by_stage=usage,
        latency_by_stage=latency,
        usage=sum(usage.values(), Usage()),
        latency_s=sum(latency.values()),
    )


def _llm_stage_latency(started: float, usage: Usage) -> float:
    """Seconds since ``started``, or the stage's recorded LLM latency when that is longer.

    The recorded latency is longer only when the call was replayed from the
    response cache (see the module docstring).
    """

    return max(perf_counter() - started, usage.latency_s)
