import assert from "node:assert/strict";
import test from "node:test";

import { conversationsHref, isRecordId, parseRoute, peopleHref } from "../src/router.js";

test("parseRoute defaults to conversations for empty and unknown hashes", () => {
  for (const hash of ["", "#", "#/", "#/nope", "#/nope/123?search=x", undefined]) {
    const route = parseRoute(hash);
    assert.equal(route.key, "conversations", `hash ${JSON.stringify(hash)}`);
    assert.equal(route.episodeId, null);
    assert.equal(route.personId, null);
    assert.equal(route.searchQuery, "");
  }
});

test("parseRoute reads a selected conversation id", () => {
  assert.deepEqual(parseRoute("#/conversations"), {
    key: "conversations",
    personId: null,
    episodeId: null,
    searchQuery: "",
    contextQuery: "",
  });
  assert.equal(parseRoute("#/conversations/ep%201").episodeId, "ep 1");
});

test("parseRoute reads people ids, search, and ask queries", () => {
  assert.deepEqual(parseRoute("#/people/abc?search=Em&ask=What%20hobbies"), {
    key: "people",
    personId: "abc",
    episodeId: null,
    searchQuery: "Em",
    contextQuery: "What hobbies",
  });
  assert.deepEqual(parseRoute("#/people?search=John"), {
    key: "people",
    personId: null,
    episodeId: null,
    searchQuery: "John",
    contextQuery: "",
  });
});

test("parseRoute never throws on malformed percent-encoding", () => {
  assert.equal(parseRoute("#/people/%ZZ").personId, "%ZZ");
  assert.equal(parseRoute("#/conversations/abc%").episodeId, "abc%");
});

test("href builders encode ids and drop empty query parts", () => {
  assert.equal(peopleHref(), "#/people");
  assert.equal(peopleHref({ searchQuery: "" }), "#/people");
  assert.equal(peopleHref({ searchQuery: "Em Chen" }), "#/people?search=Em+Chen");
  assert.equal(
    peopleHref({ personId: "p/1", searchQuery: "Em", contextQuery: "hobbies?" }),
    "#/people/p%2F1?search=Em&ask=hobbies%3F",
  );
  assert.equal(conversationsHref(), "#/conversations");
  assert.equal(conversationsHref("e 1"), "#/conversations/e%201");
});

test("href builders round-trip through parseRoute", () => {
  const built = peopleHref({ personId: "p/1", searchQuery: "Em", contextQuery: "hobbies?" });
  assert.deepEqual(parseRoute(built), {
    key: "people",
    personId: "p/1",
    episodeId: null,
    searchQuery: "Em",
    contextQuery: "hobbies?",
  });
  assert.equal(parseRoute(conversationsHref("e 1")).episodeId, "e 1");
});

test("isRecordId accepts UUIDs only", () => {
  assert.equal(isRecordId("d5f514a5-8cd0-4c50-b41b-42135e847a55"), true);
  assert.equal(isRecordId("D5F514A5-8CD0-4C50-B41B-42135E847A55"), true);
  assert.equal(isRecordId("not-a-uuid"), false);
  assert.equal(isRecordId(""), false);
  assert.equal(isRecordId(null), false);
  assert.equal(isRecordId("d5f514a5-8cd0-4c50-b41b-42135e847a55/extra"), false);
});
