import assert from "node:assert/strict";
import test from "node:test";

import { createRouteLoaders, getErrorMessage, shouldSyncSearch } from "../src/loaders.js";

const EPISODE = "d5f514a5-8cd0-4c50-b41b-42135e847a55";
const EMILY = "3d0419c4-4dc1-4dea-b6dd-b2c251b32b44";
const JOHN = "fd7bb888-d393-445c-8ea4-2ff7cc226298";

function fakeApi(overrides = {}) {
  const calls = [];
  const record = (name, impl) => async (...args) => {
    calls.push([name, ...args]);
    return impl(...args);
  };
  const api = {
    listRecentConversations: record("listRecentConversations", async () => [{ id: EPISODE, participants: [] }]),
    getConversation: record("getConversation", async (id) => ({ id, transcript: "hi" })),
    listPeople: record("listPeople", async () => ({ items: [{ id: EMILY }], query: null, resolution: null })),
    getPersonProfile: record("getPersonProfile", async (id) => ({ person: { id } })),
    getPersonContext: record("getPersonContext", async (id, query) => ({ person_id: id, query })),
    ...Object.fromEntries(Object.entries(overrides).map(([name, impl]) => [name, record(name, impl)])),
  };
  return { api, calls };
}

test("conversations: a failing detail fetch keeps the list and reports detailError", async () => {
  const { api, calls } = fakeApi({
    getConversation: async () => {
      throw new Error("Conversation not found");
    },
  });
  const loaders = createRouteLoaders(api);

  const data = await loaders.conversations({ episodeId: EPISODE });

  assert.equal(data.items.length, 1);
  assert.equal(data.selectedConversation, null);
  assert.equal(data.detailError, "Conversation not found");
  assert.deepEqual(calls.map((c) => c[0]), ["listRecentConversations", "getConversation"]);
});

test("conversations: a failing list is page-fatal", async () => {
  const { api } = fakeApi({
    listRecentConversations: async () => {
      throw new Error("Authenticated user required");
    },
  });
  await assert.rejects(() => createRouteLoaders(api).conversations({ episodeId: null }), /Authenticated user required/);
});

test("conversations: malformed ids never reach the API", async () => {
  const { api, calls } = fakeApi();

  const data = await createRouteLoaders(api).conversations({ episodeId: "not-a-uuid" });

  assert.match(data.detailError, /not valid/);
  assert.deepEqual(calls.map((c) => c[0]), ["listRecentConversations"]);
});

test("people: explicit selection loads profile and context independently", async () => {
  const { api, calls } = fakeApi({
    getPersonContext: async () => {
      throw new Error("Retrieval unavailable");
    },
  });

  const data = await createRouteLoaders(api).people({ personId: EMILY, searchQuery: "", contextQuery: "hobbies" });

  assert.equal(data.selectedProfile.person.id, EMILY);
  assert.equal(data.detailError, null);
  assert.equal(data.selectedContext, null);
  assert.equal(data.contextError, "Retrieval unavailable");
  assert.equal(data.impliedSelection, false);
  assert.deepEqual(calls.map((c) => c[0]), ["listPeople", "getPersonProfile", "getPersonContext"]);
});

test("people: a resolver match implies the selection", async () => {
  const { api } = fakeApi({
    listPeople: async () => ({ items: [{ id: EMILY }], query: "Emily", resolution: { person_id: EMILY, is_ambiguous: false, candidates: [] } }),
  });

  const data = await createRouteLoaders(api).people({ personId: null, searchQuery: "Emily", contextQuery: "" });

  assert.equal(data.selectedProfile.person.id, EMILY);
  assert.equal(data.impliedSelection, true);
});

test("people: an explicit selection the search would re-resolve is still implied", async () => {
  const { api } = fakeApi({
    listPeople: async () => ({ items: [{ id: EMILY }], query: "Emily", resolution: { person_id: EMILY, is_ambiguous: false, candidates: [] } }),
  });
  const loaders = createRouteLoaders(api);

  const same = await loaders.people({ personId: EMILY, searchQuery: "Emily", contextQuery: "hobbies" });
  assert.equal(same.impliedSelection, true);

  const other = await loaders.people({ personId: JOHN, searchQuery: "Emily", contextQuery: "" });
  assert.equal(other.impliedSelection, false);
  assert.equal(other.selectedProfile.person.id, JOHN);
});

test("people: malformed ids and failed profiles keep the directory", async () => {
  const { api, calls } = fakeApi({
    getPersonProfile: async () => {
      throw new Error("Person not found");
    },
  });
  const loaders = createRouteLoaders(api);

  const bad = await loaders.people({ personId: "bad-id", searchQuery: "", contextQuery: "" });
  assert.equal(bad.items.length, 1);
  assert.match(bad.detailError, /not valid/);
  assert.deepEqual(calls.map((c) => c[0]), ["listPeople"]);

  const missing = await loaders.people({ personId: JOHN, searchQuery: "", contextQuery: "" });
  assert.equal(missing.items.length, 1);
  assert.equal(missing.selectedProfile, null);
  assert.equal(missing.detailError, "Person not found");
});

test("helpers: error messages and search sync", () => {
  assert.equal(getErrorMessage(new Error("boom")), "boom");
  assert.match(getErrorMessage("not an error"), /unexpected/);
  assert.equal(shouldSyncSearch("Em", null), true);
  assert.equal(shouldSyncSearch("Em", "Em"), false);
  assert.equal(shouldSyncSearch("", "Em"), true);
});
