"""Turn the roster into a schedule (pure code).

``build_timeline`` gives every conversation a start and an end. ``build_schedule``
then shapes every person's storylines (how many, how long, which shared subject)
and decides which storyline beat each conversation carries. All of it needs
only the roster, so an outline is written knowing exactly when each of its
storylines runs. Both are deterministic from ``config.SEED``.

The benchmark treats UTC as the owner's wall clock: 07:00 here is seven in the
morning for Riley, and "this morning" in a query means before 12:00 UTC.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from .. import config
from .roster import RosterEntry, seeded

Slot = tuple[datetime, datetime]  # start, end

MIN_DURATION_S = 3 * 60
MAX_DURATION_S = 25 * 60
TYPICAL_DURATION_S = 8 * 60
MIN_GAP_S = 5 * 60  # the owner gets at least this long between two conversations
DAY_START_S = 7 * 3600
DAY_END_S = 22 * 3600
_NOON_S = 12 * 3600
_NOON = time(12, 0)

RECENT_DAYS = 7
LIGHT_WINDOW_DAYS = (30, 240)  # a light person's active stretch: one to eight months
SCATTERED_SHARE = 0.3  # light people whose conversations are spread over the whole year instead
PARALLEL_STORYLINES = 5  # storylines of one person that run side by side
MONTHLY_COVERAGE_RATE = 3  # conversations per calendar month from which every month is reached


def _clock(hour: int, minute: int = 0) -> int:
    return hour * 3600 + minute * 60


# When the owner talks to someone, by where they usually meet, as seconds after
# midnight. All of it lies inside DAY_START_S..DAY_END_S.
_WEEKDAY_WINDOWS: dict[str, tuple[tuple[int, int], ...]] = {
    "home": ((DAY_START_S, _clock(9)), (_clock(17, 30), DAY_END_S)),
    "work": ((_clock(8), _clock(18, 30)),),
    "social": ((_clock(12), _clock(13, 30)), (_clock(17), DAY_END_S)),
}
_WEEKEND_WINDOWS: dict[str, tuple[tuple[int, int], ...]] = {
    "home": ((DAY_START_S, DAY_END_S),),
    "work": (),
    "social": ((_clock(9), DAY_END_S),),
}


@dataclass(frozen=True)
class StorylineShape:
    """One storyline's size, shared subject and stretch of the timeline, decided in code.

    The outline call fills in the title and premise, so a premise can fit the
    dates its conversations actually fall on instead of the whole year.
    """

    n_beats: int
    shared_topic: str | None
    first: datetime  # start of its first conversation
    last: datetime  # start of its last conversation


@dataclass(frozen=True)
class Storyline:
    """A shape filled in: one subject that runs through ``n_beats`` of a person's conversations."""

    storyline_id: str
    title: str
    premise: str
    shared_topic: str | None
    n_beats: int


@dataclass(frozen=True)
class StorylineBudget:
    """How many storylines a person may have and how long each may run."""

    min_storylines: int
    max_storylines: int
    min_beats: int
    max_beats: int
    close_pairs: int  # pairs of storylines that must be deliberately close in subject


@dataclass(frozen=True)
class PlannedConversation:
    """When one conversation happens and which storyline beat it carries."""

    episode_id: str
    person_index: int  # index into the roster
    position: int  # chronological index among that person's conversations
    start_time: datetime
    end_time: datetime
    storyline_index: int  # index into that person's storylines
    beat_index: int  # 0-based within the storyline


@dataclass(frozen=True)
class Schedule:
    """Every person's storyline shapes and every conversation's time and beat."""

    storylines: list[list[StorylineShape]]  # per roster entry, in order of first conversation
    conversations: list[PlannedConversation]  # chronological


@dataclass(frozen=True)
class CoverageTargets:
    """Minimum recent activity, so temporal queries have something to find."""

    today_morning: int  # on NOW's date, starting before 12:00
    yesterday: int  # on the day before NOW's date
    last_week: int  # starting in the 7 days before NOW


_FULL_COVERAGE = CoverageTargets(today_morning=2, yesterday=3, last_week=14)
_FULL_HEAVY_BUDGET = StorylineBudget(
    min_storylines=14, max_storylines=20, min_beats=3, max_beats=14, close_pairs=2
)
LIGHT_MAX_STORYLINES = 3


# --------------------------------------------------------------------------- #
# Storyline sizes                                                              #
# --------------------------------------------------------------------------- #
def storyline_budget(entry: RosterEntry) -> StorylineBudget:
    """The outline shape for one person: 14-20 storylines if heavy, 1-3 if light."""

    n, topics = entry.n_conversations, len(entry.shared_topics)
    if not entry.person.is_heavy:
        fewest = max(1, topics)
        most = max(fewest, min(LIGHT_MAX_STORYLINES, n))
        # One storyline can take whatever the fewest allowed storylines leave over.
        return StorylineBudget(fewest, most, 1, n - fewest + 1, 0)
    full = _FULL_HEAVY_BUDGET
    # The full-size budget, whenever any storyline count it allows can hold n conversations.
    if full.max_storylines * full.min_beats <= n <= full.min_storylines * full.max_beats:
        return full
    # A heavy person in a reduced roster (the smoke run, tests): the same kind of
    # outline, sized so that the storylines can hold exactly n conversations.
    most_beats = min(full.max_beats, max(2, n // 2))
    fewest = max(min(n, topics + 1), math.ceil(n / most_beats))
    return StorylineBudget(fewest, max(fewest, min(full.max_storylines, n // 2)), 1, most_beats, 1)


def draw_beat_counts(rng: random.Random, total: int, budget: StorylineBudget) -> list[int]:
    """Beat counts for a person's storylines: as many as the budget allows, summing to ``total``.

    Each count is drawn uniformly from the budget's range, so a heavy person
    gets a spread of long threads and short subjects, then the counts are
    nudged one at a time until they add up. ``storyline_budget`` guarantees
    that every storyline count it allows can hold ``total``.
    """

    low, high = budget.min_beats, budget.max_beats
    n_storylines = rng.randint(budget.min_storylines, budget.max_storylines)
    if not n_storylines * low <= total <= n_storylines * high:
        raise ValueError(
            f"{n_storylines} storylines of {low}-{high} beats cannot hold {total} conversations"
        )
    counts = [rng.randint(low, high) for _ in range(n_storylines)]
    while (missing := total - sum(counts)) != 0:
        index = rng.randrange(n_storylines)
        nudged = counts[index] + (1 if missing > 0 else -1)
        if low <= nudged <= high:
            counts[index] = nudged
    return counts


# --------------------------------------------------------------------------- #
# Recent coverage                                                              #
# --------------------------------------------------------------------------- #
def coverage_targets(total: int) -> CoverageTargets:
    """The ticket's minimums for the full corpus, scaled down for a reduced roster."""

    scale = min(1.0, total / config.N_CONVERSATIONS_TARGET)
    return CoverageTargets(
        today_morning=math.ceil(_FULL_COVERAGE.today_morning * scale),
        yesterday=math.ceil(_FULL_COVERAGE.yesterday * scale),
        last_week=math.ceil(_FULL_COVERAGE.last_week * scale),
    )


def calendar_months() -> list[tuple[int, int]]:
    """(year, month) of every calendar month the dataset span touches."""

    year, month = config.DATASET_START.year, config.DATASET_START.month
    last = config.NOW - timedelta(seconds=1)
    months = []
    while (year, month) <= (last.year, last.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def guarantees_monthly_coverage(n_conversations: int) -> bool:
    """Whether a heavy person talks to the owner often enough to appear in every calendar month."""

    return n_conversations >= MONTHLY_COVERAGE_RATE * len(calendar_months())


def coverage_problems(
    roster: Sequence[RosterEntry], starts: Sequence[Sequence[datetime]]
) -> list[str]:
    """Ways the start times (one list per roster entry) fall short of the coverage guarantees."""

    targets = coverage_targets(sum(entry.n_conversations for entry in roster))
    today = config.NOW.date()
    week_start = config.NOW - timedelta(days=RECENT_DAYS)
    everyone = [start for own in starts for start in own]
    morning = sum(start.date() == today and start.time() < _NOON for start in everyone)
    yesterday = sum(start.date() == today - timedelta(days=1) for start in everyone)
    week = sum(week_start <= start < config.NOW for start in everyone)
    found = (
        ("on the last day before 12:00", morning, targets.today_morning),
        ("on the day before the last", yesterday, targets.yesterday),
        (f"in the last {RECENT_DAYS} days", week, targets.last_week),
    )
    problems = [
        f"{count} conversations {label}, need at least {target}"
        for label, count, target in found
        if count < target
    ]
    for entry, own in zip(roster, starts, strict=True):
        if not entry.person.is_heavy or not guarantees_monthly_coverage(entry.n_conversations):
            continue
        present = {(start.year, start.month) for start in own}
        missing = [
            f"{year}-{month:02d}"
            for year, month in calendar_months()
            if (year, month) not in present
        ]
        if missing:
            name = entry.person.display_name
            problems.append(f"{name} has no conversation in {', '.join(missing)}")
    return problems


# --------------------------------------------------------------------------- #
# Timeline                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Request:
    """A conversation that still needs a time of day."""

    person: int
    day: date
    morning: bool = False  # must end by 12:00


def build_timeline(roster: Sequence[RosterEntry]) -> list[list[Slot]]:
    """Start and end of every conversation: one chronological list per roster entry.

    Every conversation lies inside the dataset span and inside the hours the
    owner sees that person, lasts 3-25 minutes, and keeps ``MIN_GAP_S`` from
    every other conversation of the owner's day.
    """

    rng = seeded("plan:times")
    taken: dict[date, list[tuple[int, int]]] = {}
    slots: list[list[Slot]] = [[] for _ in roster]
    # Morning-pinned conversations have the narrowest windows, so they go first.
    for request in sorted(_draw_days(roster), key=lambda r: (not r.morning, r.day, r.person)):
        duration = int(rng.triangular(MIN_DURATION_S, MAX_DURATION_S, TYPICAL_DURATION_S))
        setting = roster[request.person].setting
        windows = _windows(setting, request.day, before=_NOON_S if request.morning else None)
        busy = taken.setdefault(request.day, [])
        start = _pick_start(rng, windows, duration, busy)
        busy.append((start, start + duration))
        midnight = datetime.combine(request.day, time(0), tzinfo=timezone.utc)
        begins = midnight + timedelta(seconds=start)
        slots[request.person].append((begins, begins + timedelta(seconds=duration)))
    for own in slots:
        own.sort()

    problems = coverage_problems(roster, [[start for start, _ in own] for own in slots])
    if problems:
        raise ValueError("the schedule misses its coverage guarantees: " + "; ".join(problems))
    return slots


def _span_days() -> list[date]:
    first, last = config.DATASET_START.date(), (config.NOW - timedelta(seconds=1)).date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _windows(setting: str, day: date, before: int | None = None) -> list[tuple[int, int]]:
    """Seconds after midnight during which a conversation in ``setting`` can run on ``day``.

    Clipped to the dataset span and to ``before``. A window too short for the
    longest conversation is dropped, so any drawn duration fits the ones returned.
    """

    table = _WEEKEND_WINDOWS if day.weekday() >= 5 else _WEEKDAY_WINDOWS
    midnight = datetime.combine(day, time(0), tzinfo=timezone.utc)
    earliest = max(0, int((config.DATASET_START - midnight).total_seconds()))
    latest = min(DAY_END_S, int((config.NOW - midnight).total_seconds()))
    if before is not None:
        latest = min(latest, before)
    clipped = [(max(low, earliest), min(high, latest)) for low, high in table[setting]]
    return [(low, high) for low, high in clipped if high - low >= MAX_DURATION_S]


def _draw_days(roster: Sequence[RosterEntry]) -> list[_Request]:
    """The day of every conversation.

    Heavy people are spread evenly over the year with jitter (one conversation
    per equal slice of their eligible days), then pinned to the last week.
    Light people are active in one stretch of the year or scattered across it.
    """

    days = _span_days()
    recent = days[-RECENT_DAYS:]
    pinned = _recent_requests(roster, recent)
    requests: list[_Request] = []
    for index, entry in enumerate(roster):
        rng = seeded(f"plan:days:{index}")
        eligible = [day for day in days if _windows(entry.setting, day)]
        if entry.person.is_heavy:
            earlier = [day for day in eligible if day < recent[0]]
            n_drawn = entry.n_conversations - len(pinned[index])
            width = len(earlier) / max(n_drawn, 1)
            drawn = [earlier[int((part + rng.random()) * width)] for part in range(n_drawn)]
            requests += [_Request(index, day) for day in drawn] + pinned[index]
        else:
            drawn = _light_days(rng, days, eligible, entry.n_conversations)
            requests += [_Request(index, day) for day in drawn]
    return requests


def _light_days(
    rng: random.Random, days: Sequence[date], eligible: Sequence[date], n: int
) -> list[date]:
    """``n`` distinct days for a light person: inside one stretch of the year, or anywhere in it."""

    if n > 1 and rng.random() >= SCATTERED_SHARE:
        length = rng.randint(*LIGHT_WINDOW_DAYS)
        first = days[rng.randrange(len(days) - length + 1)]
        eligible = [day for day in eligible if first <= day < first + timedelta(days=length)]
    return sorted(rng.sample(eligible, n))


def _recent_requests(
    roster: Sequence[RosterEntry], recent: Sequence[date]
) -> dict[int, list[_Request]]:
    """Conversations pinned to the last week so the coverage targets hold by construction.

    Heavy people take turns, which gives each about the three or four
    conversations a week their usual rate would, and week days fill up evenly.
    """

    targets = coverage_targets(sum(entry.n_conversations for entry in roster))
    today, yesterday, before = recent[-1], recent[-2], recent[:-2]
    kinds = ["morning"] * targets.today_morning + ["yesterday"] * targets.yesterday
    kinds += ["week"] * (targets.last_week - len(kinds))
    heavy = [index for index, entry in enumerate(roster) if entry.person.is_heavy]
    rng = seeded("plan:recent")
    pinned: dict[int, list[_Request]] = {index: [] for index in heavy}
    load = dict.fromkeys(recent, 0)
    turn = 0
    for kind in kinds:
        for offset in range(len(heavy)):
            index = heavy[(turn + offset) % len(heavy)]
            setting = roster[index].setting
            if kind == "morning":
                options = [today] if _windows(setting, today, before=_NOON_S) else []
            elif kind == "yesterday":
                options = [yesterday] if _windows(setting, yesterday) else []
            else:
                options = [day for day in before if _windows(setting, day)]
            if options and len(pinned[index]) < roster[index].n_conversations:
                break
        else:
            raise ValueError(f"no heavy person can take a {kind!r} conversation in the last week")
        turn = (turn + offset + 1) % len(heavy)
        lightest = min(load[day] for day in options)
        day = rng.choice([day for day in options if load[day] == lightest])
        load[day] += 1
        pinned[index].append(_Request(index, day, morning=kind == "morning"))
    return pinned


def _pick_start(
    rng: random.Random,
    windows: Sequence[tuple[int, int]],
    duration: int,
    busy: Sequence[tuple[int, int]],
) -> int:
    """A uniformly drawn start second that fits a window and keeps the gap from ``busy``."""

    free: list[tuple[int, int]] = []  # closed ranges of allowed starts
    for low, high in windows:
        ranges = [(low, high - duration)]
        for start, end in busy:
            # Starts from which the conversation would run into this one or its gap.
            blocked = (start - MIN_GAP_S - duration + 1, end + MIN_GAP_S - 1)
            ranges = [piece for span in ranges for piece in _without(span, blocked)]
        free += ranges
    choice = rng.randrange(sum(last - first + 1 for first, last in free)) if free else -1
    for first, last in free:
        if choice <= last - first:
            return first + choice
        choice -= last - first + 1
    raise ValueError("a day is too full to fit another conversation")


def _without(span: tuple[int, int], blocked: tuple[int, int]) -> list[tuple[int, int]]:
    """What is left of a closed integer range after another is cut out of it."""

    pieces = ((span[0], min(span[1], blocked[0] - 1)), (max(span[0], blocked[1] + 1), span[1]))
    return [(first, last) for first, last in pieces if first <= last]


# --------------------------------------------------------------------------- #
# Beats                                                                        #
# --------------------------------------------------------------------------- #
def build_schedule(
    roster: Sequence[RosterEntry], timeline: Sequence[Sequence[Slot]] | None = None
) -> Schedule:
    """Every person's storyline shapes and every conversation's time and beat.

    ``timeline`` is ``build_timeline(roster)`` when the caller already has it.
    """

    timeline = build_timeline(roster) if timeline is None else timeline
    storylines: list[list[StorylineShape]] = []
    planned: list[PlannedConversation] = []
    for index, (entry, slots) in enumerate(zip(roster, timeline, strict=True)):
        shapes, order = _shape_storylines(entry, index, slots)
        storylines.append(shapes)
        carried = zip(slots, order, strict=True)
        for position, ((start, end), (storyline, beat)) in enumerate(carried):
            planned.append(
                PlannedConversation(
                    episode_id=config.stable_id("episode", f"{index}:{position}"),
                    person_index=index,
                    position=position,
                    start_time=start,
                    end_time=end,
                    storyline_index=storyline,
                    beat_index=beat,
                )
            )
    planned.sort(key=lambda conversation: conversation.start_time)
    return Schedule(storylines=storylines, conversations=planned)


def _shape_storylines(
    entry: RosterEntry, index: int, slots: Sequence[Slot]
) -> tuple[list[StorylineShape], list[tuple[int, int]]]:
    """A person's storyline shapes and the (storyline, beat) each of their conversations carries.

    Storylines are numbered in the order their first conversations happen, so
    an outline reads as a timeline. Each shared subject goes to one storyline.
    """

    rng = seeded(f"plan:storylines:{index}")
    drawn = draw_beat_counts(rng, entry.n_conversations, storyline_budget(entry))
    dealt = _beat_order(drawn, seeded(f"plan:beats:{index}"))
    first_seen = sorted(range(len(drawn)), key=lambda which: dealt.index((which, 0)))
    renumbered = {old: new for new, old in enumerate(first_seen)}
    order = [(renumbered[which], beat) for which, beat in dealt]
    counts = [drawn[old] for old in first_seen]
    holders = rng.sample(range(len(counts)), len(entry.shared_topics))
    topics = dict(zip(holders, entry.shared_topics, strict=True))
    shapes = []
    for which, n_beats in enumerate(counts):
        starts = [slots[position][0] for position, (owner, _) in enumerate(order) if owner == which]
        shapes.append(StorylineShape(n_beats, topics.get(which), starts[0], starts[-1]))
    return shapes, order


def _beat_order(counts: Sequence[int], rng: random.Random) -> list[tuple[int, int]]:
    """The (storyline index, beat index) carried by each of a person's conversations, in time order.

    Storylines (``counts`` beats each) are dealt into a few lanes of equal load
    and laid end to end inside their lane, so each is active over its own
    stretch of the person's timeline (in proportion to its beats: a few weeks
    for a short one, several months for a long one) while the storylines of
    the other lanes run alongside. A storyline's beats get jittered places
    inside its stretch, in order.
    """

    n_lanes = min(len(counts), PARALLEL_STORYLINES)
    lanes: list[list[int]] = [[] for _ in range(n_lanes)]
    loads = [0] * n_lanes
    longest_first = sorted(range(len(counts)), key=lambda index: (-counts[index], index))
    for index in longest_first:
        lane = min(range(n_lanes), key=lambda lane: (loads[lane], lane))
        lanes[lane].append(index)
        loads[lane] += counts[index]

    places: list[tuple[float, int, int]] = []
    for lane, members in enumerate(lanes):
        rng.shuffle(members)
        done = 0
        for index in members:
            for beat in range(counts[index]):
                places.append(((done + beat + rng.random()) / loads[lane], index, beat))
            done += counts[index]
    return [(index, beat) for _, index, beat in sorted(places)]
