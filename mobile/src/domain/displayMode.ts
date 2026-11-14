export type DisplayMode = 'none' | 'name' | 'name_and_facts';

export interface DisplayModeSetting {
  mode: DisplayMode;
  syncStatus: 'synced' | 'pending' | 'error';
  /** Local timestamp of the last change, for optimistic UI. */
  updatedAt: string;
}
