import assert from "node:assert/strict";
import test from "node:test";

import { renderConversationsState } from "../src/views/conversations.js";
import { renderPeopleState } from "../src/views/people.js";

test("conversation view renders ready state items", () => {
  const html = renderConversationsState({
    status: "ready",
    items: [
      {
        started_at: "2026-04-27T10:00:00+00:00",
        summary: "Project planning session.",
        participants: [{ id: "1", name: "Emily Chen" }],
      },
    ],
  });

  assert.match(html, /Project planning session\./);
  assert.match(html, /Emily Chen/);
});

test("conversation view renders selected conversation detail", () => {
  const html = renderConversationsState({
    status: "ready",
    items: [],
    selectedConversation: {
      id: "episode-1",
      started_at: "2026-04-27T10:00:00+00:00",
      summary: "Project planning session.",
      transcript: "Emily and John reviewed the roadmap in detail.",
      participants: [
        {
          id: "1",
          name: "Emily Chen",
          last_seen_at: "2026-04-27T10:00:00+00:00",
          aliases: ["Em"],
          top_facts: ["Leads the robotics club"],
          fact_count: 1,
          summary_count: 1,
          relationship_count: 1,
        },
      ],
    },
  });

  assert.match(html, /Project planning session\./);
  assert.match(html, /Emily and John reviewed the roadmap in detail\./);
  assert.match(html, /Open person/);
  assert.match(html, /Leads the robotics club/);
});

test("conversation view renders error state", () => {
  const html = renderConversationsState({
    status: "error",
    message: "Authenticated user required",
  });

  assert.match(html, /Could not load conversations/);
  assert.match(html, /Authenticated user required/);
});

test("people view renders empty and ready states", () => {
  const emptyHtml = renderPeopleState({
    status: "ready",
    directory: { items: [], query: null, resolution: null },
    items: [],
  });
  assert.match(emptyHtml, /No people found/);

  const readyHtml = renderPeopleState({
    status: "ready",
    directory: {
      items: [
        {
          id: "john-1",
          name: "John Rivera",
          last_seen_at: "2026-04-27T10:00:00+00:00",
          aliases: ["Johnny"],
          top_facts: ["Works on backend reliability"],
          fact_count: 1,
          summary_count: 0,
          relationship_count: 0,
        },
      ],
      query: "John",
      resolution: null,
    },
    items: [
      {
        id: "john-1",
        name: "John Rivera",
        last_seen_at: "2026-04-27T10:00:00+00:00",
        aliases: ["Johnny"],
        top_facts: ["Works on backend reliability"],
        fact_count: 1,
        summary_count: 0,
        relationship_count: 0,
      },
    ],
    searchQuery: "John",
  });

  assert.match(readyHtml, /John Rivera/);
  assert.match(readyHtml, /Johnny/);
  assert.match(readyHtml, /Works on backend reliability/);
  assert.match(readyHtml, /Search results for “John”/);
});

test("people view renders selected backend profile details", () => {
  const html = renderPeopleState({
    status: "ready",
    directory: {
      items: [
        {
          id: "john-1",
          name: "John Rivera",
          last_seen_at: "2026-04-27T10:00:00+00:00",
          aliases: ["Johnny"],
          top_facts: ["Works on backend reliability"],
          fact_count: 1,
          summary_count: 1,
          relationship_count: 1,
        },
      ],
      query: null,
      resolution: null,
    },
    items: [
      {
        id: "john-1",
        name: "John Rivera",
        last_seen_at: "2026-04-27T10:00:00+00:00",
        aliases: ["Johnny"],
        top_facts: ["Works on backend reliability"],
        fact_count: 1,
        summary_count: 1,
        relationship_count: 1,
      },
    ],
    selectedProfile: {
      person: {
        id: "john-1",
        name: "John Rivera",
        aliases: ["Johnny"],
      },
      profile: {
        facts: [{ fact_text: "Works on backend reliability" }],
        prefs: [{ pref_text: "Prefers quiet cafes" }],
        summaries: [{ summary_text: "Discussed the latest reliability review.", created_at: "2026-04-27T10:00:00+00:00" }],
        edges_from: [{ relation: "colleague", dst_name: "Emily Chen" }],
      },
    },
    contextQuery: "reliability",
    selectedContext: {
      facts: [{ fact_text: "Works on backend reliability" }],
      summaries: [],
      edges: [{ relation: "colleague", dst_name: "Emily Chen" }],
    },
    searchQuery: "",
  });

  assert.match(html, /Detailed memory context/);
  assert.match(html, /Prefers quiet cafes/);
  assert.match(html, /colleague: Emily Chen/);
  assert.match(html, /Showing the strongest stored matches/);
  assert.match(html, /Search memory/);
  assert.match(html, /What hobbies do they have/);
});

const johnCard = {
  id: "john-1",
  name: "John Rivera",
  last_seen_at: "2026-04-27T10:00:00+00:00",
  aliases: [],
  top_facts: [],
  fact_count: 0,
  summary_count: 0,
  relationship_count: 0,
};

test("both views render loading states", () => {
  assert.match(renderConversationsState({ status: "loading" }), /Loading conversations/);
  assert.match(renderPeopleState({ status: "loading" }), /Loading people/);
});

test("conversation view renders the empty state", () => {
  const html = renderConversationsState({ status: "ready", items: [], selectedConversation: null });
  assert.match(html, /No conversations yet/);
  assert.doesNotMatch(html, /conversation-card/);
});

test("people view renders the error state and the searched empty state", () => {
  assert.match(renderPeopleState({ status: "error", message: "Backend down" }), /Could not load people[\s\S]*Backend down/);

  const searched = renderPeopleState({
    status: "ready",
    directory: { items: [], query: "zzz", resolution: null },
    items: [],
    searchQuery: "zzz",
  });
  assert.match(searched, /No people matched that search yet\./);
});

test("a failed detail fetch keeps the list and shows an error in the detail pane", () => {
  const conversations = renderConversationsState({
    status: "ready",
    items: [
      {
        id: "ep-1",
        started_at: "2026-04-27T10:00:00+00:00",
        summary: "Still listed.",
        participants: [{ id: "1", name: "Emily Chen" }],
      },
    ],
    selectedConversation: null,
    detailError: "Conversation not found",
  });
  assert.match(conversations, /Still listed\./);
  assert.match(conversations, /Could not load this conversation[\s\S]*Conversation not found/);

  const people = renderPeopleState({
    status: "ready",
    directory: { items: [johnCard], query: null, resolution: null },
    items: [johnCard],
    selectedProfile: null,
    detailError: "Person not found",
    searchQuery: "",
  });
  assert.match(people, /John Rivera/);
  assert.match(people, /Could not load this person[\s\S]*Person not found/);
  assert.doesNotMatch(people, /Select a person/);
});

test("a failed memory search shows an error without hiding the profile", () => {
  const html = renderPeopleState({
    status: "ready",
    directory: { items: [johnCard], query: null, resolution: null },
    items: [johnCard],
    selectedProfile: {
      person: { id: "john-1", name: "John Rivera", aliases: [] },
      profile: { facts: [], prefs: [], summaries: [], edges_from: [] },
    },
    contextQuery: "hobbies",
    selectedContext: null,
    contextError: "Retrieval unavailable",
    searchQuery: "",
  });
  assert.match(html, /Detailed memory context/);
  assert.match(html, /Could not search this memory[\s\S]*Retrieval unavailable/);
});

test("resolver-implied selections close by clearing the search", () => {
  const html = renderPeopleState({
    status: "ready",
    directory: { items: [johnCard], query: "John", resolution: { person_id: "john-1", is_ambiguous: false, candidates: [] } },
    items: [johnCard],
    selectedProfile: {
      person: { id: "john-1", name: "John Rivera", aliases: [] },
      profile: { facts: [], prefs: [], summaries: [], edges_from: [] },
    },
    searchQuery: "John",
    impliedSelection: true,
  });
  assert.match(html, /Resolver match for “John”/);
  assert.match(html, /href="#\/people">Clear search<\/a>/);
  assert.doesNotMatch(html, /href="#\/people\?search=John">(Hide profile|Close)<\/a>/);
});

test("ambiguous resolver results render candidate cards with hints", () => {
  const html = renderPeopleState({
    status: "ready",
    directory: {
      items: [johnCard],
      query: "John",
      resolution: {
        person_id: null,
        is_ambiguous: true,
        candidates: [
          { person_id: "john-1", name: "John Rivera", hints: { affiliation: ["Works on backend reliability"] } },
          { person_id: "john-2", name: "John Park", hints: { hobby: [] } },
        ],
      },
    },
    items: [johnCard],
    searchQuery: "John",
  });
  assert.match(html, /Multiple people matched “John”/);
  assert.match(html, /2 candidates/);
  assert.match(html, /Works on backend reliability/);
  assert.match(html, /No disambiguation hints available/);
  assert.doesNotMatch(html, /undefined/);
});

test("views escape untrusted memory text", () => {
  const hostile = '<img src=x onerror="alert(1)">';
  const conversations = renderConversationsState({
    status: "ready",
    items: [
      {
        id: "ep-1",
        started_at: "2026-04-27T10:00:00+00:00",
        summary: hostile,
        participants: [{ id: "1", name: "<b>Em</b>" }],
      },
    ],
    selectedConversation: {
      id: "ep-1",
      started_at: "2026-04-27T10:00:00+00:00",
      summary: hostile,
      transcript: hostile,
      participants: [{ ...johnCard, name: "<b>Em</b>", aliases: [hostile], top_facts: [hostile] }],
    },
  });
  assert.doesNotMatch(conversations, /<img/);
  assert.doesNotMatch(conversations, /<b>Em<\/b>/);
  assert.match(conversations, /&lt;b&gt;Em&lt;\/b&gt;/);

  const people = renderPeopleState({
    status: "ready",
    directory: {
      items: [],
      query: hostile,
      resolution: {
        person_id: null,
        is_ambiguous: true,
        candidates: [{ person_id: "c-1", name: hostile, hints: { [hostile]: [hostile] } }],
      },
    },
    items: [{ ...johnCard, name: hostile, aliases: [hostile], top_facts: [hostile] }],
    selectedProfile: {
      person: { id: "john-1", name: hostile, aliases: [hostile] },
      profile: {
        facts: [{ fact_text: hostile }],
        prefs: [{ pref_text: hostile }],
        summaries: [{ summary_text: hostile, created_at: "2026-04-27T10:00:00+00:00" }],
        edges_from: [{ relation: hostile, dst_name: hostile }],
      },
    },
    contextQuery: hostile,
    selectedContext: { facts: [{ fact_text: hostile }], summaries: [], edges: [] },
    searchQuery: hostile,
  });
  assert.doesNotMatch(people, /<img/);
  assert.match(people, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
});

test("search form does not pin a resolver-implied selection", () => {
  const profile = {
    person: { id: "john-1", name: "John Rivera", aliases: [] },
    profile: { facts: [], prefs: [], summaries: [], edges_from: [] },
  };
  const implied = renderPeopleState({
    status: "ready",
    directory: { items: [johnCard], query: "John", resolution: { person_id: "john-1", is_ambiguous: false, candidates: [] } },
    items: [johnCard],
    selectedProfile: profile,
    contextQuery: "hobbies",
    searchQuery: "John",
    impliedSelection: true,
  });
  assert.match(implied, /data-selected-person-id=""/);
  assert.match(implied, /data-ask-query=""/);

  const explicit = renderPeopleState({
    status: "ready",
    directory: { items: [johnCard], query: null, resolution: null },
    items: [johnCard],
    selectedProfile: profile,
    contextQuery: "hobbies",
    searchQuery: "",
    impliedSelection: false,
  });
  assert.match(explicit, /data-selected-person-id="john-1"/);
  assert.match(explicit, /data-ask-query="hobbies"/);

  const failed = renderPeopleState({
    status: "ready",
    directory: { items: [johnCard], query: "John", resolution: { person_id: "john-1", is_ambiguous: false, candidates: [] } },
    items: [johnCard],
    selectedProfile: null,
    detailError: "Person not found",
    searchQuery: "John",
    impliedSelection: true,
  });
  assert.match(failed, /href="#\/people">Clear search<\/a>/);
});
