"""Time rules shared by the query flow and the dataset's gold labels.

Two things live here so that the system under test and the gold labels follow
one rule set: the ticket's time-filter padding, and the conventions that turn a
relative time expression ("last week", "in March") into a concrete range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import config
from .schema import TimeRange, parse_dt

ONE_DAY = timedelta(days=1)
ONE_SECOND = timedelta(seconds=1)
EARLIEST = datetime.min.replace(tzinfo=timezone.utc)  # the furthest back a padded start can reach
MORNING_START_HOUR = 6
MORNING_END_HOUR = 12
ROLLING_WINDOW_DAYS: tuple[int, ...] = (3, 10, 30)
PAST_MONTHS_BY_NAME = 11  # the calendar months before last month that "in <Month>" covers
PAST_DAYS_BY_NAME = 6  # the fully past days that "on <Weekday>" covers

# Spelled out rather than taken from strftime/calendar, which follow the locale:
# prompts and gold phrases must read the same on every machine.
WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# The conventions of ``time_expressions`` in plain English, for the extraction
# prompt. Change the two together.
EXTRACTION_TIME_RULES = "\n".join(
    (
        "- Times are UTC. A range is closed: it starts at 00:00:00 on its first day and ends at "
        "23:59:59 on its last day, unless a rule below says otherwise.",
        "- A range never extends past the current time. A period that is still running "
        '("today", "this week", "this month") ends at the current time.',
        '- "today" starts at 00:00 today. "yesterday" is the whole previous day.',
        '- Mornings are 06:00 to 12:00, so "this morning" is 06:00 to 12:00 today.',
        '- Weeks start on Monday. "this week" starts on this week\'s Monday at 00:00. '
        '"last week" is the previous calendar week, Monday 00:00 to Sunday 23:59:59. '
        '"two weeks ago" is the calendar week before last week.',
        '- "last weekend" is the most recent Saturday 00:00 to Sunday 23:59:59 that has '
        "fully ended.",
        '- Months are calendar months. "this month" starts on the 1st at 00:00. "last month" '
        "is the whole previous calendar month.",
        '- A bare month name ("in March") is the most recent calendar month of that name that '
        'has fully ended. A bare weekday ("on Monday") is the most recent day of that name '
        "before today.",
        '- "the past N days" is a rolling window: exactly N times 24 hours, ending at the '
        "current time.",
    )
)


@dataclass(frozen=True)
class TimeExpression:
    """A relative time phrase and the range it denotes at one ``now``."""

    key: str  # stable identifier: "last_week", "in_march", "past_3_days", "on_monday"
    phrase: str  # an adverbial that can close a question as is: "last week", "in March"
    time_range: TimeRange


def describe_moment(moment: datetime) -> str:
    """Render a moment for a prompt, weekday included: ``Wednesday 2026-11-04 18:00 UTC``."""

    moment = parse_dt(moment)
    return f"{WEEKDAY_NAMES[moment.weekday()]} {moment:%Y-%m-%d %H:%M} UTC"


def pad_time_range(time_range: TimeRange, now: datetime = config.NOW) -> TimeRange:
    """Widen an extracted time range into the filter that the ranker's pool uses.

    The ticket's rule: each side moves out by 20% of the range's length, by at
    least 4 hours, and a pad over one day rounds to the nearest whole day (half
    rounds up). The end never passes ``now``. A range that still starts after
    ``now`` once padded collapses to ``[now, now]``, which matches nothing in
    the past.

    Every valid range pads without an error. An extracted range can reach as
    far as the calendar goes ("since forever" as year 1, "until further notice"
    as year 9999), where moving a bound out by the pad would overflow, so the
    start stops at ``EARLIEST`` and the end is capped before it is moved.
    """

    now = parse_dt(now)
    pad = _pad_length(time_range.end - time_range.start)
    start = time_range.start - min(pad, time_range.start - EARLIEST)
    if start > now:
        return TimeRange(start=now, end=now)
    end = now if now - time_range.end <= pad else time_range.end + pad
    return TimeRange(start=start, end=end)


def _pad_length(length: timedelta) -> timedelta:
    pad = length * config.PAD_FRACTION
    minimum = timedelta(hours=config.PAD_MIN_HOURS)
    if pad < minimum:
        return minimum
    if pad > ONE_DAY:
        whole_days, remainder = divmod(pad, ONE_DAY)
        if remainder * 2 >= ONE_DAY:
            whole_days += 1
        return ONE_DAY * whole_days
    return pad


def time_expressions(now: datetime = config.NOW) -> list[TimeExpression]:
    """Every relative time expression the benchmark uses, resolved against ``now``.

    The dataset generator takes gold ranges from here and the extraction prompt
    states the same conventions (``EXTRACTION_TIME_RULES``). Ranges are closed,
    end at 23:59:59 of their last day unless they end at ``now``, and never
    extend past ``now``. "this morning" is left out while ``now`` is before
    06:00, when there is no morning to look back on yet.
    """

    now = parse_dt(now)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = today - ONE_DAY * today.weekday()
    month = today.replace(day=1)

    expressions = [TimeExpression("today", "today", TimeRange(today, now))]
    morning_start = today + timedelta(hours=MORNING_START_HOUR)
    if morning_start <= now:
        morning = TimeRange(morning_start, min(today + timedelta(hours=MORNING_END_HOUR), now))
        expressions.append(TimeExpression("this_morning", "this morning", morning))
    expressions += [
        TimeExpression("yesterday", "yesterday", _whole_days(today - ONE_DAY, 1)),
        TimeExpression("this_week", "this week", TimeRange(monday, now)),
        TimeExpression("last_week", "last week", _whole_days(monday - ONE_DAY * 7, 7)),
        TimeExpression("two_weeks_ago", "two weeks ago", _whole_days(monday - ONE_DAY * 14, 7)),
        TimeExpression("last_weekend", "last weekend", _whole_days(monday - ONE_DAY * 2, 2)),
        TimeExpression("this_month", "this month", TimeRange(month, now)),
        TimeExpression("last_month", "last month", _calendar_month(month, 1)),
    ]
    for months_back in range(2, 2 + PAST_MONTHS_BY_NAME):
        time_range = _calendar_month(month, months_back)
        name = MONTH_NAMES[time_range.start.month - 1]
        expressions.append(TimeExpression(f"in_{name.lower()}", f"in {name}", time_range))
    for days in ROLLING_WINDOW_DAYS:
        window = TimeRange(now - ONE_DAY * days, now)
        expressions.append(TimeExpression(f"past_{days}_days", f"in the past {days} days", window))
    for days_back in range(1, 1 + PAST_DAYS_BY_NAME):
        day = today - ONE_DAY * days_back
        name = WEEKDAY_NAMES[day.weekday()]
        expressions.append(TimeExpression(f"on_{name.lower()}", f"on {name}", _whole_days(day, 1)))
    return expressions


def _whole_days(first_day: datetime, n_days: int) -> TimeRange:
    """``n_days`` calendar days from midnight of ``first_day`` through 23:59:59 of the last."""

    return TimeRange(first_day, first_day + ONE_DAY * n_days - ONE_SECOND)


def _calendar_month(current_month: datetime, months_back: int) -> TimeRange:
    """The calendar month ``months_back`` months before the one starting at ``current_month``."""

    start = _months_before(current_month, months_back)
    return TimeRange(start, _months_before(current_month, months_back - 1) - ONE_SECOND)


def _months_before(month_start: datetime, months: int) -> datetime:
    index = month_start.year * 12 + month_start.month - 1 - months
    return month_start.replace(year=index // 12, month=index % 12 + 1)
