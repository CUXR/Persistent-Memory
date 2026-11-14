/** Opaque cursor so an HttpAdapter can map it to whatever pagination scheme the backend agrees on. */
export interface PageRequest {
  cursor?: string | null;
  limit?: number;
}

export interface Page<T> {
  items: T[];
  /** null means there is no next page. */
  nextCursor: string | null;
  totalCount?: number;
}
