import type { PersonFact, PersonListItem } from '../../domain';

export const populatedPeople: PersonListItem[] = [
  {
    id: 'person-1',
    name: 'Alex Rivera',
    aliases: ['Al'],
    lastSeenAt: new Date(2026, 10, 13, 9, 15).toISOString(),
    topFacts: ['Works at Cuxr', 'Likes climbing'],
    factCount: 2,
  },
  {
    id: 'person-2',
    name: 'Jordan Lee',
    aliases: [],
    lastSeenAt: new Date(2026, 10, 10, 18, 30).toISOString(),
    topFacts: ['Neighbor'],
    factCount: 1,
  },
];

const factsByPerson: Record<string, PersonFact[]> = {
  'person-1': [
    {
      id: 'fact-1',
      category: 'affiliation',
      text: 'Works at Cuxr',
      confidence: 0.95,
      createdAt: new Date(2026, 9, 1).toISOString(),
      validFrom: null,
      validTo: null,
    },
    {
      id: 'fact-2',
      category: 'hobby',
      text: 'Likes climbing',
      confidence: 0.8,
      createdAt: new Date(2026, 9, 3).toISOString(),
      validFrom: null,
      validTo: null,
    },
  ],
  'person-2': [
    {
      id: 'fact-3',
      category: 'affiliation',
      text: 'Neighbor',
      confidence: 0.7,
      createdAt: new Date(2026, 8, 20).toISOString(),
      validFrom: null,
      validTo: null,
    },
  ],
};

export const populatedPersonFacts = factsByPerson;
