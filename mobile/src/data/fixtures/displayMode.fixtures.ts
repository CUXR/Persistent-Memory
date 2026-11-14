import type { DisplayModeSetting } from '../../domain';

export const initialDisplayMode: DisplayModeSetting = {
  mode: 'name',
  syncStatus: 'synced',
  updatedAt: new Date(2026, 10, 1).toISOString(),
};
