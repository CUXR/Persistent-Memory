"""Turn pipeline results into the per-query records that ``metrics`` aggregates.

The conventions live here, in one place, so the runner, the label-pooling
script and the Jev study produce records the aggregates can trust:

* Ranker records are produced under the *gold* filters (the gold person and
  the padded gold time range), so ranking is measured apart from extraction.
* ``SystemRecord.pool_size`` is also the gold-filter pool (``None`` only for
  unknown-name queries), so both evaluations slice by the same pool buckets.
* ``needed_dialogue`` is true whenever the summary attempt could not answer,
  whether the fallback then answered or the system declined.
* ``time_filter_ok`` is ``None`` without a gold time range; with one, it says
  whether the padded filter the system applied kept every gold conversation.
  No filter at all (the extraction returned no time range) is ``False``: the
  metric is the share of temporal queries whose padded range contains every
  gold conversation, and a missing range contains nothing. ``metrics`` tells a
  missed range from a range that dropped a gold conversation through
  ``extracted_time_range`` (``time_range_extracted_rate`` and the
  ``missed_time_ranges`` failure count).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from . import config, metrics
from .interfaces import RankResult
from .schema import Query, TimeRange
from .system.pipeline import SystemResult
from .timeutil import pad_time_range


def gold_filters(query: Query, now: datetime = config.NOW) -> tuple[str | None, TimeRange | None]:
    """The person filter and padded time filter a perfect extraction would produce."""

    padded = pad_time_range(query.gold_time_range, now) if query.gold_time_range else None
    return query.gold_person_id, padded


def ranker_record(query: Query, result: RankResult, run: int = 0) -> metrics.RankerRecord:
    """One ranker's ranking for one query (under the gold filters)."""

    trace = result.trace
    return metrics.RankerRecord(
        query_id=query.query_id,
        type=query.type,
        pool_size=trace.pool_size,
        gold_ids=list(query.gold_episode_ids),
        ranker=trace.ranker,
        run=run,
        ranked_ids=[episode_id for episode_id, _ in result.ranked],
        scores=[score for _, score in result.ranked],
        stage1_ids=trace.stage1_ids,
        latency_s=trace.latency_s,
        stage_latency_s=dict(trace.stage_latency_s),
        usage=trace.usage,
        extra=dict(trace.extra),
    )


def time_filter_ok(
    query: Query,
    applied_range: TimeRange | None,
    start_times: Mapping[str, datetime],
) -> bool | None:
    """Whether the filter the system applied kept every gold conversation.

    None without a gold time range. False when the query has one and no filter
    was applied: the extraction missed the time expression, so no padded range
    contains the gold conversations.
    """

    if query.gold_time_range is None:
        return None
    if applied_range is None:
        return False
    return all(applied_range.contains(start_times[episode_id]) for episode_id in query.gold_episode_ids)


def system_record(
    query: Query,
    ranker_name: str,
    k: int,
    k_prime: int,
    result: SystemResult,
    pool_size: int | None,
    verdict: str | None,
    start_times: Mapping[str, datetime],
) -> metrics.SystemRecord:
    """One grid cell (ranker, k, k') of the system eval for one query."""

    ranked_ids = [] if result.rank is None else [episode_id for episode_id, _ in result.rank.ranked]
    extracted_name = result.prepared.extraction.name
    return metrics.SystemRecord(
        query_id=query.query_id,
        type=query.type,
        pool_size=pool_size,
        ranker=ranker_name,
        k=k,
        k_prime=k_prime,
        path=result.path,
        needed_dialogue=result.answered is not None and result.answered.fallback_attempt is not None,
        final_answer=result.final_answer,
        verdict=verdict,
        gold_ids=list(query.gold_episode_ids),
        retrieved_ids=ranked_ids[:k],
        gold_name=query.gold_name,
        extracted_name=extracted_name,
        name_correct=(extracted_name or "").casefold() == (query.gold_name or "").casefold(),
        person_correct=result.prepared.person_id == query.gold_person_id,
        extracted_time_range=result.prepared.extraction.time_range,
        time_filter_ok=time_filter_ok(query, result.prepared.time_range, start_times),
        unknown_name_exact=(
            result.final_answer == config.UNKNOWN_NAME_TEMPLATE.format(name=query.gold_name)
            if query.type == "unknown_name"
            else None
        ),
        usage_by_stage=dict(result.usage_by_stage),
        latency_by_stage=dict(result.latency_by_stage),
        rank_cached=result.rank is not None and (result.rank.trace.extra.get("n_cached") or 0) > 0,
    )
