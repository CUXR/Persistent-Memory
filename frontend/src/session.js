export const DEV_USER_STORAGE_KEY = "persistent-memory-user-id";

export function getAuthenticatedUserId() {
  const globalUserId = globalThis?.__PERSISTENT_MEMORY_SESSION__?.userId;
  if (typeof globalUserId === "string" && globalUserId.trim()) {
    return globalUserId.trim();
  }

  try {
    const stored = globalThis?.localStorage?.getItem(DEV_USER_STORAGE_KEY);
    return stored?.trim() || null;
  } catch {
    return null;
  }
}

export function describeSessionSource() {
  if (globalThis?.__PERSISTENT_MEMORY_SESSION__?.userId) {
    return "Using authenticated session";
  }

  const localId = getAuthenticatedUserId();
  if (localId) {
    return `Local dev owner override: ${localId}`;
  }

  return "Using browser session or single-owner local fallback";
}
