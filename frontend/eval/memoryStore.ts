// Minimal in-memory store for evaluation purposes. In a real application, this would likely be replaced with a more robust solution.
// Given a query like "What do you know about John?", it tries to match the name in the query to one of the records and returns the associated items.

// Name associated with memory and the items related to that person. 
export type MemoryRecord = {
  person: string;
  items: string[];
};

//person resolved from the query. If the person does not exist then it will be null. The items associated with the resolved person. If no person is resolved, this will be an empty array.
export type RetrievalResult = {
  resolvedPerson: string | null;
  items: string[];
};
// Metrics report structure for evaluation results.
export type MetricsReport = {
  accuracy_person: number; // Proportion of cases where the correct person was identified.
  avg_items_returned: number; // Average number of items returned per query.
  pass_rate: number; // Proportion of cases that passed both person and items checks.
};

// Simple in-memory store implementation for evaluation. It matches the query against the person field in the memory records and returns the associated items if a match is found.
export class MemoryStore {
  constructor(private readonly memories: MemoryRecord[]) {}

  retrieve(query: string): RetrievalResult {
    const loweredQuery = query.toLowerCase();

    const match = this.memories.find((memory) =>
      loweredQuery.includes(memory.person.toLowerCase())
    );

    if (!match) {
      return {
        resolvedPerson: null,
        items: []
      };
    }

    return {
      resolvedPerson: match.person,
      items: match.items
    };
  }
}