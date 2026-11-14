import type {
  AppError,
  ConversationDetail,
  ConversationId,
  ConversationListItem,
  DisplayMode,
  DisplayModeSetting,
  Page,
  PageRequest,
  PersonFact,
  PersonId,
  PersonListItem,
  PersonProfile,
  RenamePersonRequest,
  RenamePersonResult,
} from '../domain';
import type { MemoryDataAdapter } from './adapter';
import { populatedConversationDetails, populatedConversations } from './fixtures/conversations.fixtures';
import { initialDisplayMode } from './fixtures/displayMode.fixtures';
import { populatedPeople, populatedPersonFacts } from './fixtures/people.fixtures';
import type { DataKey, Scenario } from './fixtures/scenarios';

export interface FixtureAdapterOptions {
  /** Defaults to 'populated' for every key not listed. */
  scenarios?: Partial<Record<DataKey, Scenario>>;
  /** Artificial latency so a 'loading' state is visibly reachable. Defaults to 0. */
  delayMs?: number;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function paginate<T>(items: T[], request: PageRequest): Page<T> {
  const limit = request.limit ?? 20;
  const offset = request.cursor ? Number(request.cursor) : 0;
  const slice = items.slice(offset, offset + limit);
  const nextOffset = offset + limit;
  return {
    items: slice,
    nextCursor: nextOffset < items.length ? String(nextOffset) : null,
    totalCount: items.length,
  };
}

function errorFor(key: DataKey): AppError {
  return { message: `Fixture error scenario for "${key}"`, retryable: true };
}

function notFound(message: string): AppError {
  return { message, retryable: false };
}

/**
 * Implements MemoryDataAdapter entirely from in-memory fixture data.
 *
 * Scenario selection lives on the adapter instance (constructor + setScenario),
 * never as a per-call argument — that would leak a fixture-only concept into
 * call sites that must also work against a future HttpAdapter. Scenarios are
 * tracked per DataKey so a screen can show a populated pane next to an
 * errored one at the same time.
 */
export class FixtureAdapter implements MemoryDataAdapter {
  private readonly scenarios: Partial<Record<DataKey, Scenario>>;
  private readonly delayMs: number;

  private readonly conversations: ConversationListItem[];
  private readonly conversationDetails: Record<string, ConversationDetail>;
  private readonly people: Map<PersonId, PersonListItem>;
  private readonly personFacts: Record<string, PersonFact[]>;
  private displayMode: DisplayModeSetting;

  constructor(options: FixtureAdapterOptions = {}) {
    this.scenarios = { ...options.scenarios };
    this.delayMs = options.delayMs ?? 0;

    this.conversations = clone(populatedConversations);
    this.conversationDetails = clone(populatedConversationDetails);
    this.people = new Map(clone(populatedPeople).map((person) => [person.id, person] as const));
    this.personFacts = clone(populatedPersonFacts);
    this.displayMode = clone(initialDisplayMode);
  }

  /** Runtime flip, e.g. from the smoke screen's scenario buttons. */
  setScenario(key: DataKey, scenario: Scenario): void {
    this.scenarios[key] = scenario;
  }

  private scenarioFor(key: DataKey): Scenario {
    return this.scenarios[key] ?? 'populated';
  }

  private async wait(): Promise<void> {
    if (this.delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.delayMs));
    }
  }

  private async settle<T>(key: DataKey, produce: () => T): Promise<T> {
    await this.wait();
    if (this.scenarioFor(key) === 'error') {
      throw errorFor(key);
    }
    return produce();
  }

  async listConversations(request: PageRequest & { query?: string }): Promise<Page<ConversationListItem>> {
    return this.settle('conversations', () => {
      if (this.scenarioFor('conversations') === 'empty') {
        return { items: [], nextCursor: null, totalCount: 0 };
      }
      const query = request.query?.trim().toLowerCase();
      const filtered = query
        ? this.conversations.filter(
            (conversation) =>
              conversation.summary.toLowerCase().includes(query) ||
              conversation.participants.some((participant) => participant.name.toLowerCase().includes(query)),
          )
        : this.conversations;
      return paginate(filtered, request);
    });
  }

  async getConversation(id: ConversationId): Promise<ConversationDetail> {
    return this.settle('conversationDetail', () => {
      if (this.scenarioFor('conversationDetail') === 'empty') {
        return {
          id,
          startedAt: new Date(0).toISOString(),
          endedAt: null,
          summary: '',
          participants: [],
          provisionalStatus: 'committed',
          transcript: { turns: [], status: 'committed' },
        };
      }
      const detail = this.conversationDetails[id];
      if (!detail) {
        throw notFound(`Conversation "${id}" not found`);
      }
      return detail;
    });
  }

  async listPeople(request: PageRequest & { query?: string }): Promise<Page<PersonListItem>> {
    return this.settle('people', () => {
      if (this.scenarioFor('people') === 'empty') {
        return { items: [], nextCursor: null, totalCount: 0 };
      }
      const query = request.query?.trim().toLowerCase();
      const all = Array.from(this.people.values());
      const filtered = query
        ? all.filter(
            (person) =>
              person.name.toLowerCase().includes(query) ||
              person.aliases.some((alias) => alias.toLowerCase().includes(query)),
          )
        : all;
      return paginate(filtered, request);
    });
  }

  async getPersonProfile(id: PersonId): Promise<PersonProfile> {
    return this.settle('personProfile', () => {
      if (this.scenarioFor('personProfile') === 'empty') {
        return {
          id,
          name: '',
          aliases: [],
          lastSeenAt: null,
          summary: null,
          facts: { items: [], nextCursor: null, totalCount: 0 },
        };
      }
      const person = this.people.get(id);
      if (!person) {
        throw notFound(`Person "${id}" not found`);
      }
      const facts = this.personFacts[id] ?? [];
      return {
        id: person.id,
        name: person.name,
        aliases: person.aliases,
        lastSeenAt: person.lastSeenAt,
        summary: person.topFacts.length ? person.topFacts.join('. ') : null,
        facts: paginate(facts, {}),
      };
    });
  }

  async listPersonFacts(id: PersonId, request: PageRequest): Promise<Page<PersonFact>> {
    return this.settle('personFacts', () => {
      if (this.scenarioFor('personFacts') === 'empty') {
        return { items: [], nextCursor: null, totalCount: 0 };
      }
      return paginate(this.personFacts[id] ?? [], request);
    });
  }

  async renamePerson(request: RenamePersonRequest): Promise<RenamePersonResult> {
    await this.wait();
    const existing = this.people.get(request.personId);
    if (!existing) {
      throw notFound(`Person "${request.personId}" not found`);
    }
    const updated: PersonListItem = { ...existing, name: request.displayName };
    this.people.set(request.personId, updated);
    return { person: clone(updated) };
  }

  async getDisplayMode(): Promise<DisplayModeSetting> {
    return this.settle('displayMode', () => {
      if (this.scenarioFor('displayMode') === 'empty') {
        return { mode: 'none', syncStatus: 'synced', updatedAt: new Date(0).toISOString() };
      }
      return clone(this.displayMode);
    });
  }

  async setDisplayMode(mode: DisplayMode): Promise<DisplayModeSetting> {
    this.displayMode = { ...this.displayMode, mode, syncStatus: 'pending', updatedAt: new Date().toISOString() };
    await this.wait();
    if (this.scenarioFor('displayMode') === 'error') {
      this.displayMode = { ...this.displayMode, syncStatus: 'error' };
      throw errorFor('displayMode');
    }
    this.displayMode = { ...this.displayMode, syncStatus: 'synced' };
    return clone(this.displayMode);
  }
}
