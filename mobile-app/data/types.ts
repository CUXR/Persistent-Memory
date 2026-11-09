// Shared contract between the UI (#48) and app logic / backend integration (#51).
// Screens only talk to `DataSource`; #51 provides a live implementation, fixtures.ts a fake one.
// Shapes mirror backend/app/models and backend/app/schema, camelCased.

export type ISODateString = string;

export type Page<T> = {
  items: T[];
  nextCursor?: string;
};

// ── People (models/person.py: Person) ───────────────────────────

export type PersonRef = {
  id: string;
  displayName: string;
  // Faces the recognizer saved automatically still carry the "Unknown person <UUID>" placeholder.
  isUnnamed: boolean;
};

export type Person = PersonRef & {
  firstName: string;
  lastName: string;
  lastSeenAt?: ISODateString;
  episodeCount: number;
};

// models/person.py: PersonFact.fact_category
export type FactCategory = "hobby" | "affiliation" | "visual_descriptor";

export type Fact = {
  id: string;
  text: string;
  category: FactCategory | null;
  confidence: number | null;
  sourceEpisodeId?: string;
  createdAt: ISODateString;
};

// models/memory.py: Summary
export type PersonSummary = {
  id: string;
  text: string;
  episodeTimeStart?: ISODateString;
  episodeTimeEnd?: ISODateString;
  episodeId?: string;
};

// models/memory.py: Edge, e.g. relation "works_with" → target person.
export type Relationship = {
  id: string;
  relation: string;
  target: PersonRef;
  confidence: number | null;
};

export type PersonDetail = Person & {
  latestSummary?: PersonSummary;
  relationships: Relationship[];
  recentEpisodes: EpisodeSummary[];
};

// ── Transcripts (schema/asr.py: DialogTurn) ─────────────────────

export type SpeakerRole = "user" | "interlocutor" | "unknown";

export type TranscriptTurn = {
  id: string;
  speaker: SpeakerRole;
  personId?: string;
  speakerLabel: string;
  text: string;
  // Seconds from the start of the recording.
  startSeconds: number;
  endSeconds: number;
  asrConfidence?: number;
};

// ── Saved memories (models/episode.py: Episode) ─────────────────

export type EpisodeSummary = {
  kind: "episode";
  id: string;
  startTime: ISODateString;
  endTime?: ISODateString;
  // The glasses only support one other person per conversation.
  person: PersonRef;
  summary: string;
  importanceScore?: number;
};

export type Episode = EpisodeSummary & {
  transcript: TranscriptTurn[];
  learnedFacts: Fact[];
};

// ── Provisional transcripts (schema/audio_job.py: AudioJobOut) ──
// Audio jobs that haven't produced an episode. `complete` jobs appear as episodes
// and `no_speech` jobs are hidden, so neither is listed here.

export type RecordingStatus = "pending" | "processing" | "needs_review" | "failed";

export type RecordingSummary = {
  kind: "recording";
  id: string;
  startTime: ISODateString;
  durationSeconds: number;
  person: PersonRef;
  status: RecordingStatus;
  stage: string;
  transcriptPreview: string;
};

export type Recording = RecordingSummary & {
  error?: string;
  transcript: TranscriptTurn[];
  failedSegmentCount: number;
};

export type HistoryItem = EpisodeSummary | RecordingSummary;

// ── Glasses ─────────────────────────────────────────────────────

export type DisplayMode = "off" | "name" | "name_and_facts";

export type DisplaySettings = {
  mode: DisplayMode;
  updatedAt: ISODateString;
  // False when the server saved the setting but the glasses haven't picked it up yet.
  appliedToGlasses: boolean;
};

// schema/briefing.py: Brief (face path), the facts the server selects for the glasses.
export type Brief = {
  personId: string;
  displayName?: string;
  isUnnamed: boolean;
  hobbies: string[];
  affiliations: string[];
  lastInteraction?: { date: ISODateString; summary?: string };
};

// A rejected getConnectionStatus() means the server itself is unreachable.
export type ConnectionStatus = {
  glasses: "connected" | "disconnected";
  glassesLastSeenAt?: ISODateString;
};

export type DataSource = {
  listHistory(params: { query?: string; cursor?: string }): Promise<Page<HistoryItem>>;
  getEpisode(id: string): Promise<Episode>;
  getRecording(id: string): Promise<Recording>;
  listPeople(params: { query?: string; unnamedOnly?: boolean; cursor?: string }): Promise<Page<Person>>;
  getPerson(id: string): Promise<PersonDetail>;
  listFacts(personId: string, params: { category?: FactCategory; cursor?: string }): Promise<Page<Fact>>;
  renamePerson(id: string, name: { firstName: string; lastName: string }): Promise<Person>;
  getBrief(personId: string): Promise<Brief>;
  getDisplaySettings(): Promise<DisplaySettings>;
  setDisplayMode(mode: DisplayMode): Promise<DisplaySettings>;
  getConnectionStatus(): Promise<ConnectionStatus>;
};

// people.first_name / people.last_name are String(100).
export const NAME_PART_MAX_LENGTH = 100;
