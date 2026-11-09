import type {
  Brief,
  DataSource,
  DisplayMode,
  DisplaySettings,
  Episode,
  EpisodeSummary,
  Fact,
  FactCategory,
  HistoryItem,
  Page,
  Person,
  PersonRef,
  Recording,
  RecordingStatus,
  RecordingSummary,
  SpeakerRole,
  TranscriptTurn,
} from "./types";

export type FixtureScenario =
  | "normal"
  | "empty"
  | "error"
  | "slow"
  | "saveFails"
  | "glassesOffline";

export const FIXTURE_SCENARIOS: { value: FixtureScenario; label: string }[] = [
  { value: "normal", label: "Normal" },
  { value: "empty", label: "Empty" },
  { value: "error", label: "Server error" },
  { value: "slow", label: "Slow network" },
  { value: "saveFails", label: "Saves fail" },
  { value: "glassesOffline", label: "Glasses offline" },
];

const HISTORY_PAGE_SIZE = 15;
const PEOPLE_PAGE_SIZE = 10;
const FACT_PAGE_SIZE = 8;
const BRIEF_MAX_FACTS = 3;

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

const NAMED_PEOPLE: [string, string][] = [
  ["Maya", "Chen"],
  ["Sam", "Ortiz"],
  ["Priya", "Raman"],
  ["Jordan", "Lee"],
  ["Elena", "Petrova"],
  ["Marcus", "Johnson"],
  ["Aiko", "Tanaka"],
  ["Dev", "Patel"],
  ["Grace", "Kim"],
  ["Tom", "Becker"],
  ["Nadia", "Haddad"],
  ["Luis", "Romero"],
];
const UNNAMED_COUNT = 3;

const FACT_POOL: Record<FactCategory, string[]> = {
  hobby: [
    "Runs with a Saturday morning running club",
    "Plays bass in a jazz band",
    "Goes bouldering a few times a week",
    "Bakes sourdough on weekends",
    "Is learning Japanese",
    "Collects vinyl records",
    "Coaches a youth soccer team",
    "Plays chess online most nights",
    "Takes pottery classes",
    "Hikes in the Finger Lakes most weekends",
  ],
  affiliation: [
    "Works at Stripe as a product designer",
    "PhD student in the Cornell CS department",
    "Member of the Ithaca climbing gym",
    "Volunteers at the food bank on Saturdays",
    "On the board of the neighborhood association",
    "Nurse on night shifts at Cayuga Medical",
    "Product manager at a robotics startup",
    "Plays violin in the city orchestra",
    "Teaches 8th grade science",
    "Member of Cornell XR",
  ],
  visual_descriptor: [
    "Usually wears round tortoiseshell glasses",
    "Has curly red hair",
    "Often wears a green rain jacket",
    "Has a small tattoo on the left wrist",
    "Tall, with a short beard",
    "Wears a silver nose ring",
  ],
};

const TOPICS = [
  "talked about training for an upcoming half marathon.",
  "shared news about starting a new job and the move that comes with it.",
  "caught up about a recent trip to Lisbon and recommended a few restaurants.",
  "discussed a project deadline that keeps slipping.",
  "talked about adopting a dog and asked for vet recommendations.",
  "made plans to grab lunch next Thursday.",
  "talked about a book they lent you and how it ended.",
  "mentioned helping a sibling move apartments this weekend.",
  "talked about the jazz show downtown on Friday.",
  "asked how your classes are going and talked about their own research.",
];

const DIALOGUE: [SpeakerRole, string][] = [
  ["interlocutor", "Hey! It's been a while, how have you been?"],
  ["user", "Pretty good, honestly. Busy, but in a good way. You?"],
  ["interlocutor", "Same. I just got back from a trip last week."],
  ["user", "Oh nice, where did you go?"],
  ["interlocutor", "Lisbon. The food was unreal, I'll send you a list."],
  ["user", "Please do. Are you still running in the mornings?"],
  ["interlocutor", "Trying to. The half marathon is in April so I can't skip much."],
  ["user", "That's impressive. We should grab lunch sometime next week."],
  ["interlocutor", "Definitely, Thursday works for me."],
  ["user", "Perfect. How's the new job going?"],
  ["interlocutor", "Busy, but the team is great. Still learning where everything is."],
  ["user", "That's the hardest part. Let me know if you need help with the move."],
  ["interlocutor", "That would be huge, thank you."],
  ["user", "Of course. Did you end up finishing that book?"],
  ["interlocutor", "I did! The ending was not what I expected at all."],
];

const RELATIONSHIPS: [number, number, string][] = [
  [0, 1, "works_with"],
  [0, 4, "sibling_of"],
  [1, 0, "works_with"],
  [2, 3, "partner_of"],
  [3, 2, "partner_of"],
  [3, 0, "knows"],
  [6, 7, "roommate_of"],
  [5, 9, "teammate_of"],
];

// Deterministic PRNG so the fixture data is the same every launch.
function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type StoredPerson = { id: string; firstName: string; lastName: string; unnamed: boolean };
type StoredTurn = Omit<TranscriptTurn, "speakerLabel">;
type StoredEpisode = {
  id: string;
  personIds: string[];
  startTime: number;
  turns: StoredTurn[];
  topic: number;
  importanceScore: number;
};
type StoredFact = Fact & { personId: string };
type StoredRecording = {
  id: string;
  personId: string;
  startTime: number;
  status: RecordingStatus;
  stage: string;
  error?: string;
  turns: StoredTurn[];
  failedSegmentCount: number;
};

function buildStore(now: number) {
  const rand = mulberry32(48);
  const int = (min: number, max: number) => min + Math.floor(rand() * (max - min + 1));
  const hex = (n: number) => Array.from({ length: n }, () => Math.floor(rand() * 16).toString(16)).join("");
  const uuid = () => `${hex(8)}-${hex(4)}-4${hex(3)}-${hex(4)}-${hex(12)}`;

  const people: StoredPerson[] = [
    ...NAMED_PEOPLE.map(([firstName, lastName]) => ({ id: uuid(), firstName, lastName, unnamed: false })),
    ...Array.from({ length: UNNAMED_COUNT }, () => ({ id: uuid(), firstName: "", lastName: "", unnamed: true })),
  ];

  const makeTurns = (personId: string, count: number, offset: number): StoredTurn[] => {
    let clock = int(0, 4);
    return Array.from({ length: count }, (_, j) => {
      const [speaker, text] = DIALOGUE[(offset + j) % DIALOGUE.length];
      const length = int(3, 11);
      const turn: StoredTurn = {
        id: uuid(),
        speaker,
        personId: speaker === "interlocutor" ? personId : undefined,
        text,
        startSeconds: clock,
        endSeconds: clock + length,
        asrConfidence: Math.round((0.72 + rand() * 0.27) * 100) / 100,
      };
      clock += length + int(0, 2);
      return turn;
    });
  };

  const episodes: StoredEpisode[] = [];
  let clock = now - 25 * MINUTE;
  for (let i = 0; i < 38; i++) {
    const primary = people[i % people.length];
    const others = people.filter((p) => p !== primary && !p.unnamed);
    const personIds = rand() < 0.2 ? [primary.id, others[int(0, others.length - 1)].id] : [primary.id];
    // Dialogue offsets stay even so every transcript opens with the other person speaking.
    episodes.push({
      id: uuid(),
      personIds,
      startTime: clock,
      turns: makeTurns(primary.id, int(6, 15), int(0, 6) * 2),
      topic: int(0, TOPICS.length - 1),
      importanceScore: Math.round(rand() * 100) / 100,
    });
    clock -= int(40 * MINUTE, Math.round(1.4 * DAY));
  }

  const facts: StoredFact[] = [];
  const categories: FactCategory[] = ["hobby", "affiliation", "hobby", "visual_descriptor", "affiliation"];
  people.forEach((person, i) => {
    const count = person.unnamed ? int(0, 1) : i === 0 ? 20 : int(2, 8);
    const own = episodes.filter((e) => e.personIds[0] === person.id);
    // Walk each category's pool in order so a person never gets the same fact twice.
    const used: Record<FactCategory, number> = { hobby: 0, affiliation: 0, visual_descriptor: 0 };
    for (let j = 0; j < count; j++) {
      const category = person.unnamed ? "visual_descriptor" : categories[(i + j) % categories.length];
      const pool = FACT_POOL[category];
      if (used[category] >= pool.length) continue;
      const source = own.length ? own[j % own.length] : undefined;
      facts.push({
        id: uuid(),
        personId: person.id,
        text: pool[(i * 3 + used[category]++) % pool.length],
        category,
        confidence: Math.round((0.55 + rand() * 0.43) * 100) / 100,
        sourceEpisodeId: source?.id,
        createdAt: new Date(source ? source.startTime : now - int(1, 60) * DAY).toISOString(),
      });
    }
  });
  facts.sort((a, b) => b.createdAt.localeCompare(a.createdAt));

  const recordings: StoredRecording[] = [
    {
      id: uuid(),
      personId: people[0].id,
      startTime: now - 3 * MINUTE,
      status: "processing",
      stage: "asr",
      turns: makeTurns(people[0].id, 4, 0).map((t, j) => (j === 3 ? { ...t, asrConfidence: 0.41 } : t)),
      failedSegmentCount: 0,
    },
    {
      id: uuid(),
      personId: people[5].id,
      startTime: now - 9 * MINUTE,
      status: "pending",
      stage: "queued",
      turns: [],
      failedSegmentCount: 0,
    },
    {
      id: uuid(),
      personId: people[NAMED_PEOPLE.length].id,
      startTime: now - DAY - 2 * HOUR,
      status: "needs_review",
      stage: "attribution",
      error: "Two unknown speakers were detected, so the server couldn't tell who said what.",
      turns: makeTurns(people[NAMED_PEOPLE.length].id, 8, 4).map((t, j) =>
        j % 3 === 2 ? { ...t, speaker: "unknown", personId: undefined } : t,
      ),
      failedSegmentCount: 0,
    },
    {
      id: uuid(),
      personId: people[2].id,
      startTime: now - 2 * DAY - 5 * HOUR,
      status: "failed",
      stage: "asr",
      error: "Transcription failed for 2 audio segments.",
      turns: makeTurns(people[2].id, 3, 8),
      failedSegmentCount: 2,
    },
  ];

  return {
    people,
    episodes,
    facts,
    recordings,
    display: { mode: "name_and_facts" as DisplayMode, updatedAt: now - 2 * DAY },
  };
}

function paginate<T>(items: T[], cursor: string | undefined, pageSize: number): Page<T> {
  const start = cursor ? Number(cursor) : 0;
  const end = start + pageSize;
  return {
    items: items.slice(start, end),
    nextCursor: end < items.length ? String(end) : undefined,
  };
}

const iso = (ms: number) => new Date(ms).toISOString();
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function createFixtureDataSource(getScenario: () => FixtureScenario): DataSource {
  const store = buildStore(Date.now());

  const findPerson = (id: string) => store.people.find((p) => p.id === id);

  const displayName = (p: StoredPerson) =>
    p.unnamed ? `Unknown person ${p.id}` : `${p.firstName} ${p.lastName}`.trim();

  const personRef = (id: string): PersonRef => {
    const p = findPerson(id);
    return p ? { id, displayName: displayName(p), isUnnamed: p.unnamed } : { id, displayName: "Unknown person", isUnnamed: true };
  };

  const speakerLabel = (turn: StoredTurn) => {
    if (turn.speaker === "user") return "You";
    if (turn.speaker === "unknown" || !turn.personId) return "Unknown speaker";
    const ref = personRef(turn.personId);
    return ref.isUnnamed ? "Them" : ref.displayName;
  };

  const withLabels = (turns: StoredTurn[]): TranscriptTurn[] =>
    turns.map((t) => ({ ...t, speakerLabel: speakerLabel(t) }));

  const durationMs = (turns: StoredTurn[]) => (turns.length ? turns[turns.length - 1].endSeconds * 1000 : 0);

  const episodeSummary = (e: StoredEpisode): EpisodeSummary => {
    const lead = personRef(e.personIds[0]);
    const subject = lead.isUnnamed ? "They" : lead.displayName.split(" ")[0];
    return {
      kind: "episode",
      id: e.id,
      startTime: iso(e.startTime),
      endTime: iso(e.startTime + durationMs(e.turns)),
      people: e.personIds.map(personRef),
      summary: `${subject} ${TOPICS[e.topic]}`,
      importanceScore: e.importanceScore,
    };
  };

  const recordingSummary = (r: StoredRecording): RecordingSummary => ({
    kind: "recording",
    id: r.id,
    startTime: iso(r.startTime),
    durationSeconds: r.turns.length ? r.turns[r.turns.length - 1].endSeconds : 95,
    person: personRef(r.personId),
    status: r.status,
    stage: r.stage,
    transcriptPreview: r.turns[0]?.text ?? "",
  });

  const toPerson = (p: StoredPerson): Person => {
    const own = store.episodes.filter((e) => e.personIds.includes(p.id));
    return {
      ...personRef(p.id),
      firstName: p.firstName,
      lastName: p.lastName,
      lastSeenAt: own[0] ? iso(own[0].startTime) : undefined,
      episodeCount: own.length,
    };
  };

  const toFact = (f: StoredFact): Fact => ({
    id: f.id,
    text: f.text,
    category: f.category,
    confidence: f.confidence,
    sourceEpisodeId: f.sourceEpisodeId,
    createdAt: f.createdAt,
  });

  const toDisplaySettings = (): DisplaySettings => ({
    mode: store.display.mode,
    updatedAt: iso(store.display.updatedAt),
    appliedToGlasses: getScenario() !== "glassesOffline",
  });

  async function simulate(kind: "read" | "write") {
    const scenario = getScenario();
    const delay = scenario === "slow" ? 2500 + Math.random() * 1500 : 300 + Math.random() * 500;
    await sleep(delay);
    if (scenario === "error") throw new Error("Couldn't reach the server. Check your connection and try again.");
    if (kind === "write" && scenario === "saveFails") throw new Error("The server couldn't save this change.");
  }

  const isEmpty = () => getScenario() === "empty";
  const notFound = (what: string) => new Error(`This ${what} no longer exists.`);

  return {
    async listHistory({ query, cursor }) {
      await simulate("read");
      if (isEmpty()) return { items: [] };
      const q = query?.trim().toLowerCase();
      const matchesTurns = (turns: StoredTurn[]) => turns.some((t) => t.text.toLowerCase().includes(q!));
      const matchesPeople = (ids: string[]) => ids.some((id) => personRef(id).displayName.toLowerCase().includes(q!));

      const items: HistoryItem[] = [
        ...store.recordings
          .filter((r) => !q || matchesPeople([r.personId]) || matchesTurns(r.turns))
          .map(recordingSummary),
        ...store.episodes
          .map((e) => ({ e, summary: episodeSummary(e) }))
          .filter(
            ({ e, summary }) =>
              !q || matchesPeople(e.personIds) || matchesTurns(e.turns) || summary.summary.toLowerCase().includes(q),
          )
          .map(({ summary }) => summary),
      ].sort((a, b) => b.startTime.localeCompare(a.startTime));

      return paginate(items, cursor, HISTORY_PAGE_SIZE);
    },

    async getEpisode(id) {
      await simulate("read");
      const e = store.episodes.find((x) => x.id === id);
      if (!e || isEmpty()) throw notFound("conversation");
      const episode: Episode = {
        ...episodeSummary(e),
        transcript: withLabels(e.turns),
        learnedFacts: store.facts.filter((f) => f.sourceEpisodeId === id).map(toFact),
      };
      return episode;
    },

    async getRecording(id) {
      await simulate("read");
      const r = store.recordings.find((x) => x.id === id);
      if (!r || isEmpty()) throw notFound("recording");
      const recording: Recording = {
        ...recordingSummary(r),
        error: r.error,
        transcript: withLabels(r.turns),
        failedSegmentCount: r.failedSegmentCount,
      };
      return recording;
    },

    async listPeople({ query, unnamedOnly, cursor }) {
      await simulate("read");
      if (isEmpty()) return { items: [] };
      const q = query?.trim().toLowerCase();
      const people = store.people
        .filter((p) => !unnamedOnly || p.unnamed)
        .map(toPerson)
        .filter((p) => !q || (!p.isUnnamed && p.displayName.toLowerCase().includes(q)))
        .sort((a, b) => (b.lastSeenAt ?? "").localeCompare(a.lastSeenAt ?? ""));
      return paginate(people, cursor, PEOPLE_PAGE_SIZE);
    },

    async getPerson(id) {
      await simulate("read");
      const p = findPerson(id);
      if (!p || isEmpty()) throw notFound("person");
      const own = store.episodes.filter((e) => e.personIds.includes(id));
      const latest = own[0];
      const index = store.people.indexOf(p);
      return {
        ...toPerson(p),
        latestSummary: latest
          ? {
              id: `${latest.id}-summary`,
              text: episodeSummary(latest).summary,
              episodeTimeStart: iso(latest.startTime),
              episodeTimeEnd: iso(latest.startTime + durationMs(latest.turns)),
              episodeId: latest.id,
            }
          : undefined,
        relationships: RELATIONSHIPS.filter(([src]) => src === index).map(([, dst, relation]) => ({
          id: `${id}-${relation}-${dst}`,
          relation,
          target: personRef(store.people[dst].id),
          confidence: 0.8,
        })),
        recentEpisodes: own.slice(0, 3).map(episodeSummary),
      };
    },

    async listFacts(personId, { category, cursor }) {
      await simulate("read");
      const facts = store.facts
        .filter((f) => f.personId === personId && (!category || f.category === category))
        .map(toFact);
      return paginate(facts, cursor, FACT_PAGE_SIZE);
    },

    async renamePerson(id, { firstName, lastName }) {
      await simulate("write");
      const p = findPerson(id);
      if (!p) throw notFound("person");
      p.firstName = firstName.trim();
      p.lastName = lastName.trim();
      p.unnamed = false;
      return toPerson(p);
    },

    async getBrief(personId) {
      await simulate("read");
      const p = findPerson(personId);
      if (!p) throw notFound("person");
      const top = (category: FactCategory) =>
        store.facts
          .filter((f) => f.personId === personId && f.category === category)
          .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
          .slice(0, BRIEF_MAX_FACTS)
          .map((f) => f.text);
      const latest = store.episodes.find((e) => e.personIds.includes(personId));
      const brief: Brief = {
        personId,
        displayName: p.unnamed ? undefined : displayName(p),
        isUnnamed: p.unnamed,
        hobbies: top("hobby"),
        affiliations: top("affiliation"),
        lastInteraction: latest
          ? { date: iso(latest.startTime), summary: episodeSummary(latest).summary }
          : undefined,
      };
      return brief;
    },

    async getDisplaySettings() {
      await simulate("read");
      return toDisplaySettings();
    },

    async setDisplayMode(mode) {
      await simulate("write");
      store.display = { mode, updatedAt: Date.now() };
      return toDisplaySettings();
    },

    async getConnectionStatus() {
      await simulate("read");
      if (getScenario() === "glassesOffline") {
        return { glasses: "disconnected", glassesLastSeenAt: iso(Date.now() - 3 * HOUR) };
      }
      return { glasses: "connected", glassesLastSeenAt: iso(Date.now() - MINUTE) };
    },
  };
}
