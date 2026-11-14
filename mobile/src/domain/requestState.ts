import type { AppError } from './error';

/**
 * Shared screen-data shape. "loading" is a timing state, not a fixture
 * scenario: it describes an in-flight promise identically for FixtureAdapter
 * and a future HttpAdapter, so fixtures only need to model empty/error/populated.
 */
export type RequestState<T> =
  | { status: 'idle' }
  | { status: 'loading'; previous?: T }
  | { status: 'error'; error: AppError; previous?: T }
  | { status: 'success'; data: T };
