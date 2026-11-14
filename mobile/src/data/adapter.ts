import type {
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

/**
 * The data contract screens are written against. Every method is a plain
 * promise with no fixture-only parameters, so FixtureAdapter today and a
 * future HttpAdapter satisfy this identically — retries and loading/error
 * presentation are layered on top by whatever calls the adapter, not baked
 * in here.
 */
export interface MemoryDataAdapter {
  listConversations(request: PageRequest & { query?: string }): Promise<Page<ConversationListItem>>;
  getConversation(id: ConversationId): Promise<ConversationDetail>;

  listPeople(request: PageRequest & { query?: string }): Promise<Page<PersonListItem>>;
  getPersonProfile(id: PersonId): Promise<PersonProfile>;
  listPersonFacts(id: PersonId, request: PageRequest): Promise<Page<PersonFact>>;

  renamePerson(request: RenamePersonRequest): Promise<RenamePersonResult>;

  getDisplayMode(): Promise<DisplayModeSetting>;
  setDisplayMode(mode: DisplayMode): Promise<DisplayModeSetting>;
}
