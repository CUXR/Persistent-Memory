import assert from "node:assert/strict";
import test from "node:test";

import { DEV_USER_STORAGE_KEY, describeSessionSource, getAuthenticatedUserId } from "../src/session.js";

function withGlobals({ session, storage }, run) {
  const previousSession = globalThis.__PERSISTENT_MEMORY_SESSION__;
  const previousStorage = globalThis.localStorage;
  globalThis.__PERSISTENT_MEMORY_SESSION__ = session;
  globalThis.localStorage = storage;
  try {
    run();
  } finally {
    globalThis.__PERSISTENT_MEMORY_SESSION__ = previousSession;
    globalThis.localStorage = previousStorage;
  }
}

const fakeStorage = (values) => ({ getItem: (key) => (key in values ? values[key] : null) });

test("session prefers the injected authenticated session over local storage", () => {
  withGlobals(
    { session: { userId: "  session-user " }, storage: fakeStorage({ [DEV_USER_STORAGE_KEY]: "local-user" }) },
    () => {
      assert.equal(getAuthenticatedUserId(), "session-user");
      assert.equal(describeSessionSource(), "Using authenticated session");
    },
  );
});

test("session falls back to the local dev override in local storage", () => {
  withGlobals({ session: undefined, storage: fakeStorage({ [DEV_USER_STORAGE_KEY]: " local-user " }) }, () => {
    assert.equal(getAuthenticatedUserId(), "local-user");
    assert.equal(describeSessionSource(), "Local dev owner override: local-user");
  });
});

test("session reports the single-owner fallback when nothing is configured", () => {
  withGlobals({ session: undefined, storage: fakeStorage({}) }, () => {
    assert.equal(getAuthenticatedUserId(), null);
    assert.equal(describeSessionSource(), "Using browser session or single-owner local fallback");
  });

  const throwingStorage = {
    getItem() {
      throw new Error("storage disabled");
    },
  };
  withGlobals({ session: undefined, storage: throwingStorage }, () => {
    assert.equal(getAuthenticatedUserId(), null);
  });
});
