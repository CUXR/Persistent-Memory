import type { Page } from './pagination';
import type { PersonId } from './conversation';

export interface PersonListItem {
  id: PersonId;
  name: string;
  aliases: string[];
  lastSeenAt: string | null;
  topFacts: string[];
  factCount: number;
}

export interface PersonFact {
  id: string;
  /** Open string, not a closed union: backend's fact_category vocabulary may grow. */
  category: string;
  text: string;
  confidence: number;
  createdAt: string;
  validFrom: string | null;
  validTo: string | null;
}

export interface PersonProfile {
  id: PersonId;
  name: string;
  aliases: string[];
  lastSeenAt: string | null;
  summary: string | null;
  facts: Page<PersonFact>;
}

export interface RenamePersonRequest {
  personId: PersonId;
  displayName: string;
}

export interface RenamePersonResult {
  /** The updated record, so callers can patch local list state without a refetch. */
  person: PersonListItem;
}
