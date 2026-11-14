export type ConversationId = string;
export type PersonId = string;

/**
 * Mobile's own vocabulary for a conversation/transcript's lifecycle, mapped
 * from the backend's AudioJob.status (pending|processing|complete|failed|
 * needs_review|no_speech) once that contract is agreed with #49. "committed"
 * means a backing Episode exists; the rest mirror an in-flight AudioJob.
 */
export type ProvisionalStatus = 'committed' | 'pending' | 'processing' | 'needs_review' | 'failed';

export interface ConversationParticipant {
  id: PersonId;
  name: string;
}

export interface ConversationListItem {
  id: ConversationId;
  startedAt: string;
  endedAt: string | null;
  summary: string;
  participants: ConversationParticipant[];
  provisionalStatus: ProvisionalStatus;
}

export interface TranscriptTurn {
  speakerPersonId: PersonId | null;
  speakerLabel: string;
  text: string;
  startMs: number | null;
  endMs: number | null;
  confidence: number | null;
}

export interface Transcript {
  turns: TranscriptTurn[];
  status: ProvisionalStatus;
}

export interface ConversationDetail extends ConversationListItem {
  transcript: Transcript;
}
