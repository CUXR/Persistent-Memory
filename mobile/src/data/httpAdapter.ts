import type {
  ConversationDetail,
  ConversationId,
  ConversationListItem,
  ConversationParticipant,
  DisplayMode,
  DisplayModeSetting,
  Page,
  PageRequest,
  PersonFact,
  PersonId,
  PersonListItem,
  PersonProfile,
  ProvisionalStatus,
  RenamePersonRequest,
  RenamePersonResult,
  Transcript,
  TranscriptTurn,
} from '../domain';
import type { MemoryDataAdapter } from './adapter';
import { createHttpClient, type HttpClientOptions } from './httpClient';

export type HttpAdapterOptions = HttpClientOptions;

/*
 * Wire shapes below are a PLACEHOLDER contract, not an agreed one — see
 * "Open dependencies/risks" in README.md. PersonFact's `fact_category`/
 * `fact_text` names are reused from the real (unmerged) browser-UI branch's
 * FactSummaryOut, since that shape already exists; everything else here
 * (pagination envelope, provisional_status, rename, display-mode) is
 * mobile's own guess, pending agreement with #35/#49/#39.
 */

interface PageWire<T> {
  items: T[];
  next_cursor: string | null;
  total_count?: number;
}

interface ConversationParticipantWire {
  id: string;
  name: string;
}

interface ConversationListItemWire {
  id: string;
  started_at: string;
  ended_at: string | null;
  summary: string;
  participants: ConversationParticipantWire[];
  provisional_status: ProvisionalStatus;
}

interface TranscriptTurnWire {
  speaker_person_id: string | null;
  speaker_label: string;
  text: string;
  start_ms: number | null;
  end_ms: number | null;
  confidence: number | null;
}

interface ConversationDetailWire extends ConversationListItemWire {
  transcript: { turns: TranscriptTurnWire[]; status: ProvisionalStatus };
}

interface PersonListItemWire {
  id: string;
  name: string;
  aliases: string[];
  last_seen_at: string | null;
  top_facts: string[];
  fact_count: number;
}

interface PersonFactWire {
  id: string;
  fact_category: string;
  fact_text: string;
  confidence: number;
  created_at: string;
  valid_from: string | null;
  valid_to: string | null;
}

interface PersonProfileWire {
  id: string;
  name: string;
  aliases: string[];
  last_seen_at: string | null;
  summary: string | null;
  facts: PageWire<PersonFactWire>;
}

interface RenamePersonResultWire {
  person: PersonListItemWire;
}

interface DisplayModeSettingWire {
  mode: DisplayMode;
  sync_status: DisplayModeSetting['syncStatus'];
  updated_at: string;
}

function mapPage<Wire, Mapped>(wire: PageWire<Wire>, mapItem: (item: Wire) => Mapped): Page<Mapped> {
  return { items: wire.items.map(mapItem), nextCursor: wire.next_cursor, totalCount: wire.total_count };
}

function mapParticipant(wire: ConversationParticipantWire): ConversationParticipant {
  return { id: wire.id, name: wire.name };
}

function mapConversationListItem(wire: ConversationListItemWire): ConversationListItem {
  return {
    id: wire.id,
    startedAt: wire.started_at,
    endedAt: wire.ended_at,
    summary: wire.summary,
    participants: wire.participants.map(mapParticipant),
    provisionalStatus: wire.provisional_status,
  };
}

function mapTranscriptTurn(wire: TranscriptTurnWire): TranscriptTurn {
  return {
    speakerPersonId: wire.speaker_person_id,
    speakerLabel: wire.speaker_label,
    text: wire.text,
    startMs: wire.start_ms,
    endMs: wire.end_ms,
    confidence: wire.confidence,
  };
}

function mapConversationDetail(wire: ConversationDetailWire): ConversationDetail {
  const transcript: Transcript = { turns: wire.transcript.turns.map(mapTranscriptTurn), status: wire.transcript.status };
  return { ...mapConversationListItem(wire), transcript };
}

function mapPersonListItem(wire: PersonListItemWire): PersonListItem {
  return {
    id: wire.id,
    name: wire.name,
    aliases: wire.aliases,
    lastSeenAt: wire.last_seen_at,
    topFacts: wire.top_facts,
    factCount: wire.fact_count,
  };
}

function mapPersonFact(wire: PersonFactWire): PersonFact {
  return {
    id: wire.id,
    category: wire.fact_category,
    text: wire.fact_text,
    confidence: wire.confidence,
    createdAt: wire.created_at,
    validFrom: wire.valid_from,
    validTo: wire.valid_to,
  };
}

function mapPersonProfile(wire: PersonProfileWire): PersonProfile {
  return {
    id: wire.id,
    name: wire.name,
    aliases: wire.aliases,
    lastSeenAt: wire.last_seen_at,
    summary: wire.summary,
    facts: mapPage(wire.facts, mapPersonFact),
  };
}

function mapDisplayModeSetting(wire: DisplayModeSettingWire): DisplayModeSetting {
  return { mode: wire.mode, syncStatus: wire.sync_status, updatedAt: wire.updated_at };
}

function toQueryString(params: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') {
      search.set(key, String(value));
    }
  }
  const query = search.toString();
  return query ? `?${query}` : '';
}

/**
 * Implements MemoryDataAdapter over HTTP. Targets a placeholder REST
 * contract (see the wire-type comment above) — swap in the agreed shapes
 * from #35/#49/#39 once they exist; screens calling MemoryDataAdapter don't
 * need to change either way.
 */
export class HttpAdapter implements MemoryDataAdapter {
  private readonly client: ReturnType<typeof createHttpClient>;

  constructor(options: HttpAdapterOptions) {
    this.client = createHttpClient(options);
  }

  async listConversations(request: PageRequest & { query?: string }): Promise<Page<ConversationListItem>> {
    const path = `/conversations${toQueryString({ cursor: request.cursor, limit: request.limit, query: request.query })}`;
    const wire = await this.client.request<PageWire<ConversationListItemWire>>(path);
    return mapPage(wire, mapConversationListItem);
  }

  async getConversation(id: ConversationId): Promise<ConversationDetail> {
    const wire = await this.client.request<ConversationDetailWire>(`/conversations/${encodeURIComponent(id)}`);
    return mapConversationDetail(wire);
  }

  async listPeople(request: PageRequest & { query?: string }): Promise<Page<PersonListItem>> {
    const path = `/people${toQueryString({ cursor: request.cursor, limit: request.limit, query: request.query })}`;
    const wire = await this.client.request<PageWire<PersonListItemWire>>(path);
    return mapPage(wire, mapPersonListItem);
  }

  async getPersonProfile(id: PersonId): Promise<PersonProfile> {
    const wire = await this.client.request<PersonProfileWire>(`/people/${encodeURIComponent(id)}/profile`);
    return mapPersonProfile(wire);
  }

  async listPersonFacts(id: PersonId, request: PageRequest): Promise<Page<PersonFact>> {
    const path = `/people/${encodeURIComponent(id)}/facts${toQueryString({ cursor: request.cursor, limit: request.limit })}`;
    const wire = await this.client.request<PageWire<PersonFactWire>>(path);
    return mapPage(wire, mapPersonFact);
  }

  async renamePerson(request: RenamePersonRequest): Promise<RenamePersonResult> {
    const wire = await this.client.request<RenamePersonResultWire>(`/people/${encodeURIComponent(request.personId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ display_name: request.displayName }),
    });
    return { person: mapPersonListItem(wire.person) };
  }

  async getDisplayMode(): Promise<DisplayModeSetting> {
    const wire = await this.client.request<DisplayModeSettingWire>('/settings/display-mode');
    return mapDisplayModeSetting(wire);
  }

  async setDisplayMode(mode: DisplayMode): Promise<DisplayModeSetting> {
    const wire = await this.client.request<DisplayModeSettingWire>('/settings/display-mode', {
      method: 'PUT',
      body: JSON.stringify({ mode }),
    });
    return mapDisplayModeSetting(wire);
  }
}
