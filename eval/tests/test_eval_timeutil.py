"""Time-filter padding and the time-expression conventions shared with the gold labels."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest

from eval.retrieval import config
from eval.retrieval.schema import TimeRange
from eval.retrieval.timeutil import (
    EXTRACTION_TIME_RULES,
    WEEKDAY_NAMES,
    describe_moment,
    pad_time_range,
    time_expressions,
)

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
SECOND = timedelta(seconds=1)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def day_range(first: datetime, last: datetime) -> TimeRange:
    """Whole days from 00:00 on ``first`` through 23:59:59 on ``last``."""

    return TimeRange(first, last + DAY - SECOND)


def by_key(now: datetime) -> dict[str, TimeRange]:
    return {expression.key: expression.time_range for expression in time_expressions(now)}


# --------------------------------------------------------------------------- #
# Padding                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("length", "expected_pad"),
    [
        (7 * DAY, DAY),  # the ticket's "last week": 1.4 days rounds to 1 day
        (4 * HOUR, 4 * HOUR),  # the ticket's "this morning": 48 minutes is lifted to 4h
        (timedelta(0), 4 * HOUR),  # a single instant still gets the minimum
        (20 * HOUR - SECOND, 4 * HOUR),  # just under the minimum
        (20 * HOUR, 4 * HOUR),  # 20% is exactly the minimum
        (21 * HOUR, timedelta(hours=4, minutes=12)),  # above the minimum: the plain 20%
        (4 * DAY, timedelta(hours=19, minutes=12)),  # under a day is never rounded
        (5 * DAY, DAY),  # exactly one day stays one day
        (5 * DAY + SECOND, DAY),  # just over a day rounds down to it
        (7.5 * DAY - 5 * SECOND, DAY),  # just under a day and a half rounds down
        (7.5 * DAY, 2 * DAY),  # a day and a half rounds up
        (12 * DAY, 2 * DAY),  # 2.4 days
        (12.5 * DAY, 3 * DAY),  # 2.5 days rounds up
        (31 * DAY, 6 * DAY),  # a calendar month: 6.2 days
    ],
)
def test_pad_is_a_fifth_of_the_length_with_a_floor_and_whole_days(length, expected_pad):
    start = utc(2026, 6, 1, 9, 30)  # far enough back that no pad reaches the cap at now
    padded = pad_time_range(TimeRange(start, start + length))

    assert start - padded.start == expected_pad
    assert padded.end - (start + length) == expected_pad


def test_ticket_examples_on_the_benchmark_clock():
    expressions = by_key(config.NOW)

    assert pad_time_range(expressions["last_week"]) == TimeRange(utc(2026, 10, 25), utc(2026, 11, 2, 23, 59, 59))
    assert pad_time_range(expressions["this_morning"]) == TimeRange(utc(2026, 11, 4, 2), utc(2026, 11, 4, 16))


def test_padded_end_is_capped_at_now():
    now = config.NOW
    still_running = pad_time_range(TimeRange(utc(2026, 11, 4), now))  # "today"
    ends_near_now = pad_time_range(TimeRange(now - 5 * HOUR, now - HOUR))
    ends_after_now = pad_time_range(TimeRange(utc(2026, 11, 2), utc(2026, 11, 8, 23, 59, 59)))  # a whole week

    assert still_running == TimeRange(utc(2026, 11, 3, 20), now)
    assert ends_near_now == TimeRange(now - 9 * HOUR, now)
    assert ends_after_now == TimeRange(utc(2026, 11, 1), now)


def test_padding_uses_the_now_it_is_given():
    time_range = TimeRange(utc(2026, 3, 9), utc(2026, 3, 10))  # 24h: a pad of 4h48m
    an_hour_later = utc(2026, 3, 10, 1)

    assert pad_time_range(time_range, now=an_hour_later) == TimeRange(utc(2026, 3, 8, 19, 12), an_hour_later)
    assert pad_time_range(time_range) == TimeRange(utc(2026, 3, 8, 19, 12), utc(2026, 3, 10, 4, 48))


def test_range_in_the_future_matches_nothing():
    now = config.NOW
    beyond_the_pad = pad_time_range(TimeRange(now + 2 * DAY, now + 3 * DAY))
    within_the_pad = pad_time_range(TimeRange(now + HOUR, now + 2 * HOUR))
    pad_lands_on_now = pad_time_range(TimeRange(now + 4 * HOUR, now + 5 * HOUR))

    assert beyond_the_pad == TimeRange(now, now)
    assert within_the_pad == TimeRange(now - 3 * HOUR, now)  # its padded start is in the past
    assert pad_lands_on_now == TimeRange(now, now)


def test_ranges_at_the_edge_of_the_calendar_pad_without_overflowing():
    now = config.NOW
    earliest = datetime.min.replace(tzinfo=timezone.utc)
    latest = datetime.max.replace(tzinfo=timezone.utc)
    since_forever = pad_time_range(TimeRange(earliest, now))  # the start cannot move back at all
    open_ended = pad_time_range(TimeRange(utc(2026, 1, 1), latest))  # the end cannot move forward
    start_stops_short = pad_time_range(TimeRange(earliest + 2 * HOUR, earliest + 3 * HOUR))  # room for 2h of the 4h

    assert since_forever == TimeRange(earliest, now)
    assert open_ended.end == now
    assert utc(2026, 1, 1) - open_ended.start > 365_000 * DAY  # a fifth of some 8,000 years
    assert pad_time_range(TimeRange(earliest, latest)) == TimeRange(earliest, now)
    assert start_stops_short == TimeRange(earliest, earliest + 7 * HOUR)


# --------------------------------------------------------------------------- #
# Time expressions                                                             #
# --------------------------------------------------------------------------- #
BENCHMARK_NOW = {  # Wednesday 2026-11-04 18:00
    "today": TimeRange(utc(2026, 11, 4), utc(2026, 11, 4, 18)),
    "this_morning": TimeRange(utc(2026, 11, 4, 6), utc(2026, 11, 4, 12)),
    "yesterday": day_range(utc(2026, 11, 3), utc(2026, 11, 3)),
    "this_week": TimeRange(utc(2026, 11, 2), utc(2026, 11, 4, 18)),
    "last_week": day_range(utc(2026, 10, 26), utc(2026, 11, 1)),
    "two_weeks_ago": day_range(utc(2026, 10, 19), utc(2026, 10, 25)),
    "last_weekend": day_range(utc(2026, 10, 31), utc(2026, 11, 1)),
    "this_month": TimeRange(utc(2026, 11, 1), utc(2026, 11, 4, 18)),
    "last_month": day_range(utc(2026, 10, 1), utc(2026, 10, 31)),
    "in_september": day_range(utc(2026, 9, 1), utc(2026, 9, 30)),
    "in_august": day_range(utc(2026, 8, 1), utc(2026, 8, 31)),
    "in_july": day_range(utc(2026, 7, 1), utc(2026, 7, 31)),
    "in_june": day_range(utc(2026, 6, 1), utc(2026, 6, 30)),
    "in_may": day_range(utc(2026, 5, 1), utc(2026, 5, 31)),
    "in_april": day_range(utc(2026, 4, 1), utc(2026, 4, 30)),
    "in_march": day_range(utc(2026, 3, 1), utc(2026, 3, 31)),
    "in_february": day_range(utc(2026, 2, 1), utc(2026, 2, 28)),
    "in_january": day_range(utc(2026, 1, 1), utc(2026, 1, 31)),
    "in_december": day_range(utc(2025, 12, 1), utc(2025, 12, 31)),
    "in_november": day_range(utc(2025, 11, 1), utc(2025, 11, 30)),  # not the month still running
    "past_3_days": TimeRange(utc(2026, 11, 1, 18), utc(2026, 11, 4, 18)),
    "past_10_days": TimeRange(utc(2026, 10, 25, 18), utc(2026, 11, 4, 18)),
    "past_30_days": TimeRange(utc(2026, 10, 5, 18), utc(2026, 11, 4, 18)),
    "on_tuesday": day_range(utc(2026, 11, 3), utc(2026, 11, 3)),
    "on_monday": day_range(utc(2026, 11, 2), utc(2026, 11, 2)),
    "on_sunday": day_range(utc(2026, 11, 1), utc(2026, 11, 1)),
    "on_saturday": day_range(utc(2026, 10, 31), utc(2026, 10, 31)),
    "on_friday": day_range(utc(2026, 10, 30), utc(2026, 10, 30)),
    "on_thursday": day_range(utc(2026, 10, 29), utc(2026, 10, 29)),
}

NEW_YEARS_MORNING = utc(2026, 1, 1, 9, 30)  # a Thursday: every look-back crosses the year
NEW_YEAR = {
    "today": TimeRange(utc(2026, 1, 1), NEW_YEARS_MORNING),
    "this_morning": TimeRange(utc(2026, 1, 1, 6), NEW_YEARS_MORNING),  # the morning is still running
    "yesterday": day_range(utc(2025, 12, 31), utc(2025, 12, 31)),
    "this_week": TimeRange(utc(2025, 12, 29), NEW_YEARS_MORNING),
    "last_week": day_range(utc(2025, 12, 22), utc(2025, 12, 28)),
    "two_weeks_ago": day_range(utc(2025, 12, 15), utc(2025, 12, 21)),
    "last_weekend": day_range(utc(2025, 12, 27), utc(2025, 12, 28)),
    "this_month": TimeRange(utc(2026, 1, 1), NEW_YEARS_MORNING),
    "last_month": day_range(utc(2025, 12, 1), utc(2025, 12, 31)),
    "in_november": day_range(utc(2025, 11, 1), utc(2025, 11, 30)),
    "in_october": day_range(utc(2025, 10, 1), utc(2025, 10, 31)),
    "in_september": day_range(utc(2025, 9, 1), utc(2025, 9, 30)),
    "in_august": day_range(utc(2025, 8, 1), utc(2025, 8, 31)),
    "in_july": day_range(utc(2025, 7, 1), utc(2025, 7, 31)),
    "in_june": day_range(utc(2025, 6, 1), utc(2025, 6, 30)),
    "in_may": day_range(utc(2025, 5, 1), utc(2025, 5, 31)),
    "in_april": day_range(utc(2025, 4, 1), utc(2025, 4, 30)),
    "in_march": day_range(utc(2025, 3, 1), utc(2025, 3, 31)),
    "in_february": day_range(utc(2025, 2, 1), utc(2025, 2, 28)),
    "in_january": day_range(utc(2025, 1, 1), utc(2025, 1, 31)),
    "past_3_days": TimeRange(utc(2025, 12, 29, 9, 30), NEW_YEARS_MORNING),
    "past_10_days": TimeRange(utc(2025, 12, 22, 9, 30), NEW_YEARS_MORNING),
    "past_30_days": TimeRange(utc(2025, 12, 2, 9, 30), NEW_YEARS_MORNING),
    "on_wednesday": day_range(utc(2025, 12, 31), utc(2025, 12, 31)),
    "on_tuesday": day_range(utc(2025, 12, 30), utc(2025, 12, 30)),
    "on_monday": day_range(utc(2025, 12, 29), utc(2025, 12, 29)),
    "on_sunday": day_range(utc(2025, 12, 28), utc(2025, 12, 28)),
    "on_saturday": day_range(utc(2025, 12, 27), utc(2025, 12, 27)),
    "on_friday": day_range(utc(2025, 12, 26), utc(2025, 12, 26)),
}


@pytest.mark.parametrize(
    ("now", "expected"),
    [(config.NOW, BENCHMARK_NOW), (NEW_YEARS_MORNING, NEW_YEAR)],
    ids=["benchmark-clock", "new-year-rollover"],
)
def test_every_time_expression_range(now, expected):
    expressions = time_expressions(now)

    assert [expression.key for expression in expressions] == list(expected)  # same set, same order
    assert {expression.key: expression.time_range for expression in expressions} == expected


def test_phrases_read_as_written_in_a_question():
    phrases = {expression.key: expression.phrase for expression in time_expressions()}

    assert phrases["this_morning"] == "this morning"
    assert phrases["two_weeks_ago"] == "two weeks ago"
    assert phrases["in_february"] == "in February"
    assert phrases["past_10_days"] == "in the past 10 days"
    assert phrases["on_thursday"] == "on Thursday"
    assert len(set(phrases.values())) == len(phrases) == 29


def test_sunday_before_dawn_in_a_leap_year():
    expressions = by_key(utc(2028, 3, 5, 3))  # Sunday 03:00

    assert "this_morning" not in expressions  # it has not started
    assert expressions["this_week"] == TimeRange(utc(2028, 2, 28), utc(2028, 3, 5, 3))
    assert expressions["last_week"] == day_range(utc(2028, 2, 21), utc(2028, 2, 27))
    assert expressions["last_weekend"] == day_range(utc(2028, 2, 26), utc(2028, 2, 27))  # not the one under way
    assert expressions["last_month"] == day_range(utc(2028, 2, 1), utc(2028, 2, 29))
    assert expressions["in_march"] == day_range(utc(2027, 3, 1), utc(2027, 3, 31))
    assert "in_february" not in expressions  # that is "last month"


def test_week_boundaries_on_monday_midnight_and_on_a_saturday():
    monday = by_key(utc(2026, 11, 2))
    saturday = by_key(utc(2026, 11, 7, 12))

    assert monday["this_week"] == TimeRange(utc(2026, 11, 2), utc(2026, 11, 2))
    assert monday["last_week"] == day_range(utc(2026, 10, 26), utc(2026, 11, 1))
    assert monday["last_weekend"] == day_range(utc(2026, 10, 31), utc(2026, 11, 1))
    assert saturday["this_week"] == TimeRange(utc(2026, 11, 2), utc(2026, 11, 7, 12))
    assert saturday["last_weekend"] == day_range(utc(2026, 10, 31), utc(2026, 11, 1))
    assert saturday["this_morning"] == TimeRange(utc(2026, 11, 7, 6), utc(2026, 11, 7, 12))


@pytest.mark.parametrize(
    "now",
    [config.NOW, NEW_YEARS_MORNING, utc(2028, 3, 5, 3), utc(2026, 11, 2), utc(2027, 3, 31, 23, 59, 59)],
)
def test_ranges_are_closed_whole_periods_that_never_pass_now(now):
    expressions = time_expressions(now)
    ranges = {expression.key: expression.time_range for expression in expressions}

    assert len(ranges) == len(expressions)  # keys are unique
    for key, time_range in ranges.items():
        assert time_range.start <= time_range.end <= now, key
        if key not in ("this_morning", "past_3_days", "past_10_days", "past_30_days"):
            assert time_range.start.time() == time(0, 0), key
            assert time_range.end == now or time_range.end.time() == time(23, 59, 59), key

    # "last month" and the eleven named months are the twelve months before this one, gap-free.
    months = [ranges["last_month"]] + [ranges[key] for key in ranges if key.startswith("in_")]
    assert len(months) == 12
    assert months[0].end + SECOND == ranges["this_month"].start
    for later, earlier in zip(months, months[1:]):
        assert earlier.end + SECOND == later.start
    assert months[-1].start == ranges["this_month"].start.replace(year=ranges["this_month"].start.year - 1)

    # The named weekdays are the six days before today, so today's own name is never among them.
    weekdays = [ranges[key] for key in ranges if key.startswith("on_")]
    assert [day.start for day in weekdays] == [ranges["today"].start - n * DAY for n in range(1, 7)]
    assert f"on_{WEEKDAY_NAMES[now.weekday()].lower()}" not in ranges


def test_extraction_rules_state_each_convention():
    for convention in (
        "Weeks start on Monday",
        '"last week" is the previous calendar week',
        "rolling window",
        "Months are calendar months",
        "most recent calendar month of that name that has fully ended",
        "most recent day of that name before today",
        "Mornings are 06:00 to 12:00",
        "never extends past the current time",
    ):
        assert convention in EXTRACTION_TIME_RULES


def test_describe_moment_names_the_weekday_in_utc():
    tokyo = timezone(timedelta(hours=9))

    assert describe_moment(config.NOW) == "Wednesday 2026-11-04 18:00 UTC"
    assert describe_moment(datetime(2026, 11, 5, 1, 30, tzinfo=tokyo)) == "Wednesday 2026-11-04 16:30 UTC"
