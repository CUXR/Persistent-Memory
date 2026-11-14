import type { ConversationDetail, ConversationListItem, ProvisionalStatus } from '../../domain';

function turn(speakerPersonId: string | null, speakerLabel: string, text: string, startMs: number, endMs: number) {
  return { speakerPersonId, speakerLabel, text, startMs, endMs, confidence: speakerPersonId ? 0.92 : null };
}

/** Full populated dataset, newest first. Includes committed and in-flight conversations
 * so "provisional transcripts stay distinguishable in recent history" has something to show. */
export const populatedConversations: ConversationListItem[] = Array.from({ length: 7 }, (_, index) => {
  const n = index + 1;
  const statuses: ProvisionalStatus[] = ['committed', 'committed', 'needs_review', 'committed', 'processing', 'committed', 'pending'];
  return {
    id: `conv-${n}`,
    startedAt: new Date(2026, 10, 14 - n, 9, 0).toISOString(),
    endedAt: statuses[index] === 'committed' ? new Date(2026, 10, 14 - n, 9, 15).toISOString() : null,
    summary: statuses[index] === 'committed' ? `Conversation ${n} summary` : 'Transcript pending review',
    participants: [{ id: 'person-1', name: 'Alex Rivera' }],
    provisionalStatus: statuses[index],
  };
});

const detailById: Record<string, ConversationDetail> = Object.fromEntries(
  populatedConversations.map((item) => [
    item.id,
    {
      ...item,
      transcript: {
        status: item.provisionalStatus,
        turns:
          item.provisionalStatus === 'committed'
            ? [
                turn('person-1', 'Alex Rivera', "Hey, did you get a chance to look at the draft?", 0, 3000),
                turn(null, 'You', 'Yeah, left a few comments.', 3200, 5000),
              ]
            : [turn(null, 'Unknown speaker', '(awaiting identity resolution)', 0, 1500)],
      },
    },
  ]),
);

export const populatedConversationDetails = detailById;
