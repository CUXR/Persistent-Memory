"""The benchmark's cast: the owner, the people they talk to, and the subjects several of them share.

Pure code. Names and relationships are hand-picked; conversation counts and the
shared-topic assignment are drawn from ``random.Random`` seeded from
``config.SEED``, so the roster is the same on every machine.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .. import config
from ..schema import Person

# Stated in every generation prompt so that independently written dialogues agree
# on who the owner is. The summary prompt never sees it.
OWNER_BIO = (
    "Riley Morgan is in their early thirties and works as a product designer at Harborline, "
    "a logistics-software company in Seattle. Riley lives in an apartment in the Ballard "
    "neighbourhood with their partner, Maya Chen. Outside work Riley boulders at a climbing gym, "
    "bakes bread at home and plays board games with friends. Riley grew up in Spokane, where "
    "their father, Walter, still lives; their older sister, Hannah Blake, lives in Tacoma with "
    "her husband, Owen."
)

# Subjects that several people talk to the owner about. Cross-person queries
# ("who did I talk to about climbing?") are written against these.
SHARED_TOPICS: tuple[str, ...] = (
    "rock climbing",
    "sourdough baking",
    "half-marathon training",
    "a tabletop-game group",
    "apartment hunting",
    "learning Italian",
    "a rescue dog",
    "a kitchen renovation",
    "a ski trip",
    "a pottery class",
    "buying an electric car",
    "a community garden",
)

# People who do not exist in the owner's memory, for unknown-name queries. Neither
# their first nor their last names occur in the roster or in the owner's name.
UNKNOWN_NAMES: tuple[str, ...] = (
    "Caleb Thornton",
    "Stella Kowalski",
    "Rashid Anwar",
    "Penelope Ashworth",
    "Diego Fuentes",
    "Cordelia Lowell",
    "Brandon Pike",
    "Sunita Kapoor",
    "Felix Baumann",
    "Gwen Hartley",
    "Tariq Mansour",
    "Juliet Moreau",
    "Isaac Sterling",
    "Paloma Estrada",
    "Russell Dunn",
    "Keiko Sato",
    "Vanessa Holt",
    "Quentin Marsh",
    "Ngozi Eze",
    "Lorenzo Bianchi",
)

# Where the owner usually talks to someone; the schedule picks weekdays and hours from it.
SETTINGS: tuple[str, ...] = ("home", "work", "social")

TOPICS_PER_HEAVY = 3
MAX_TOPICS_PER_LIGHT = 2  # and never more than the person has conversations
PEOPLE_PER_TOPIC = (3, 4)
# Conversations for the light person who shares a heavy person's first name: enough
# that a bare "Sam" is ambiguous in practice, few enough to stay a light person.
SHARED_NAME_RANGE = (4, 8)


@dataclass(frozen=True)
class RosterEntry:
    """One person, how many conversations they get, and the shared subjects they cover."""

    person: Person
    n_conversations: int
    shared_topics: tuple[str, ...]
    setting: str  # one of SETTINGS


@dataclass(frozen=True)
class _Cast:
    first_name: str
    last_name: str
    relationship: str
    setting: str = "social"
    topic: str | None = None  # a shared topic implied by how the owner knows them


# In order of config.HEAVY_PERSON_COUNTS.
_HEAVY: tuple[_Cast, ...] = (
    _Cast("Maya", "Chen", "partner; they live together", "home"),
    _Cast("Sam", "Okafor", "teammate at Harborline and work friend", "work"),
    _Cast("Daniel", "Reyes", "best friend since college"),
    _Cast("Hannah", "Blake", "older sister"),
)

# The first four make up the smoke roster: the second Sam, a work contact, a
# person bound to a shared topic, and a relative.
_LIGHT: tuple[_Cast, ...] = (
    _Cast("Sam", "Whitfield", "neighbour two floors up in Riley's apartment building"),
    _Cast("Priya", "Raman", "manager at Harborline", "work"),
    _Cast("Andre", "Baptiste", "climbing-gym friend", topic="rock climbing"),
    _Cast("Walter", "Morgan", "father"),
    _Cast("Mateo", "Herrera", "engineering lead on Riley's product team at Harborline", "work"),
    _Cast("Grace", "Lindqvist", "product manager on Riley's team at Harborline", "work"),
    _Cast("Kofi", "Mensah", "fellow product designer at Harborline", "work"),
    _Cast("Elena", "Petrova", "user researcher at Harborline", "work"),
    _Cast("Jamal", "Carter", "customer-success lead at Harborline", "work"),
    _Cast("Wei", "Zhang", "data analyst at Harborline", "work"),
    _Cast("Nora", "Feldman", "junior designer Riley is onboarding at Harborline", "work"),
    _Cast("Victor", "Almeida", "former colleague from Riley's previous job"),
    _Cast("Ingrid", "Solberg", "design mentor and former boss"),
    _Cast("Arjun", "Mehta", "freelance developer friend from a local design meetup"),
    _Cast("Naomi", "Adler", "recruiter who has kept in touch with Riley for years"),
    _Cast("Chloe", "Bennett", "college friend who now lives in Portland"),
    _Cast("Marcus", "Hill", "college roommate"),
    _Cast("Yuki", "Tanaka", "college friend"),
    _Cast("Aisha", "Rahman", "close friend since Riley's first year in Seattle"),
    _Cast("Oliver", "Grant", "regular at Riley's board-game night", topic="a tabletop-game group"),
    _Cast("Lucia", "Romano", "friend from a bread-baking workshop", topic="sourdough baking"),
    _Cast("Fatima", "Haddad", "book-club friend"),
    _Cast("Dolores", "Vega", "longtime next-door neighbour"),
    _Cast("Omid", "Karimi", "landlord"),
    _Cast("Zoe", "Castellanos", "Maya's closest friend, now Riley's friend too"),
    _Cast("Owen", "Blake", "brother-in-law, married to Riley's sister Hannah"),
    _Cast("Tessa", "Morgan", "cousin"),
    _Cast("Beatrice", "Morgan", "grandmother"),
    _Cast("Lian", "Chen", "Maya's mother"),
    _Cast("Kevin", "Chen", "Maya's younger brother"),
    _Cast("Carmen", "Ortiz", "hairdresser Riley has gone to for years"),
    _Cast("Jonas", "Weber", "physiotherapist Riley sees for a recurring back problem"),
    _Cast("Rosa", "Delgado", "owner of the neighbourhood cafe where Riley is a regular"),
    _Cast("Theo", "Marchetti", "old friend from high school in Spokane"),
    _Cast("Amara", "Nwosu", "friend who lives a few blocks away"),
    _Cast("Gabriel", "Santos", "former roommate from Riley's first Seattle apartment"),
)


def seeded(label: str) -> random.Random:
    """A generator for one named decision; every random choice in the corpus comes from one."""

    return random.Random(f"{config.SEED}:{label}")


def build_roster(
    n_people: int = config.N_PEOPLE,
    heavy_counts: Sequence[int] = config.HEAVY_PERSON_COUNTS,
    total: int = config.N_CONVERSATIONS_TARGET,
) -> list[RosterEntry]:
    """The cast with conversation counts that sum to exactly ``total``.

    Heavy people come first, in the order of ``heavy_counts``. Smaller arguments
    give a reduced roster (the smoke run, tests) cut from the front of the same cast.
    """

    heavy_counts = tuple(heavy_counts)
    n_heavy = len(heavy_counts)
    n_light = n_people - n_heavy
    if not 1 <= n_heavy <= len(_HEAVY) or not 0 <= n_light <= len(_LIGHT):
        raise ValueError(
            f"the cast has {len(_HEAVY)} heavy and {len(_LIGHT)} light people; "
            f"cannot build {n_heavy} heavy and {n_light} light"
        )
    if min(heavy_counts) <= TOPICS_PER_HEAVY:
        raise ValueError(
            f"a heavy person needs more than {TOPICS_PER_HEAVY} conversations "
            f"(their shared topics plus a thread of their own), got {heavy_counts}"
        )

    cast = _HEAVY[:n_heavy] + _LIGHT[:n_light]
    counts = list(heavy_counts) + _light_counts(cast[n_heavy:], total - sum(heavy_counts))
    topics = _assign_topics(cast, counts, n_heavy)
    return [
        RosterEntry(
            person=Person(
                person_id=config.stable_id("person", index),
                first_name=member.first_name,
                last_name=member.last_name,
                display_name=f"{member.first_name} {member.last_name}",
                relationship=member.relationship,
                is_heavy=index < n_heavy,
            ),
            n_conversations=counts[index],
            shared_topics=topics[index],
            setting=member.setting,
        )
        for index, member in enumerate(cast)
    ]


def _light_counts(light: Sequence[_Cast], light_total: int) -> list[int]:
    """Seeded draws within each light person's range, nudged until they sum to ``light_total``."""

    heavy_first_names = {member.first_name for member in _HEAVY}
    ranges = [
        SHARED_NAME_RANGE if member.first_name in heavy_first_names else config.LIGHT_PERSON_RANGE
        for member in light
    ]
    lowest, highest = sum(low for low, _ in ranges), sum(high for _, high in ranges)
    if not lowest <= light_total <= highest:
        raise ValueError(
            f"{len(light)} light people can hold {lowest}-{highest} conversations, "
            f"not the {light_total} left after the heavy people"
        )

    rng = seeded("roster:counts")
    counts = [rng.randint(low, high) for low, high in ranges]
    while (missing := light_total - sum(counts)) != 0:
        index = rng.randrange(len(counts))
        nudged = counts[index] + (1 if missing > 0 else -1)
        if ranges[index][0] <= nudged <= ranges[index][1]:
            counts[index] = nudged
    return counts


def _assign_topics(
    cast: Sequence[_Cast], counts: Sequence[int], n_heavy: int
) -> list[tuple[str, ...]]:
    """Give every shared topic to several people.

    Each heavy person gets ``TOPICS_PER_HEAVY`` distinct topics, a light person
    who shares a heavy person's first name gets one of that person's topics, and
    light people then fill each topic up to three or four holders. A reduced
    roster runs the same procedure and simply ends up with fewer holders per topic.
    """

    rng = seeded("roster:topics")
    held: list[list[str]] = [[member.topic] if member.topic else [] for member in cast]
    deck = rng.sample(SHARED_TOPICS, len(SHARED_TOPICS))
    for heavy in range(n_heavy):
        held[heavy] += deck[heavy * TOPICS_PER_HEAVY : (heavy + 1) * TOPICS_PER_HEAVY]

    light = range(n_heavy, len(cast))
    for index in light:
        for heavy in range(n_heavy):
            same_name = cast[heavy].first_name == cast[index].first_name
            if same_name and not set(held[heavy]) & set(held[index]):
                held[index].append(rng.choice(held[heavy]))

    for topic in deck:
        wanted = rng.choice(PEOPLE_PER_TOPIC) - sum(topic in topics for topics in held)
        open_to = [
            index
            for index in light
            if topic not in held[index]
            and len(held[index]) < min(MAX_TOPICS_PER_LIGHT, counts[index])
        ]
        for index in rng.sample(open_to, min(max(wanted, 0), len(open_to))):
            held[index].append(topic)

    return [tuple(sorted(topics, key=SHARED_TOPICS.index)) for topics in held]
