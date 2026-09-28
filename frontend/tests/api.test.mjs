import assert from "node:assert/strict";
import test from "node:test";

import { createApiClient } from "../src/api.js";

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function recordingClient(responder, overrides = {}) {
  const calls = [];
  const client = createApiClient({
    baseUrl: "http://localhost:8000",
    getUserId: () => "1234-user",
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      return responder(url, init);
    },
    ...overrides,
  });
  return { client, calls };
}

test("API client adds owner header and credentials for local dev", async () => {
  const { client, calls } = recordingClient(() => jsonResponse([]));

  await client.listPeople("Emily");

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://localhost:8000/people?query=Emily");
  assert.equal(calls[0].init.method, "GET");
  assert.equal(calls[0].init.credentials, "include");
  assert.equal(calls[0].init.headers["X-User-Id"], "1234-user");
  assert.equal(calls[0].init.headers.Accept, "application/json");
});

test("API client omits the owner header when no dev user id is configured", async () => {
  const { client, calls } = recordingClient(() => jsonResponse([]), { getUserId: () => null });

  await client.listRecentConversations();

  assert.equal("X-User-Id" in calls[0].init.headers, false);
});

test("API client exposes current-user, conversation, and people detail requests", async () => {
  const { client, calls } = recordingClient(() => jsonResponse({}));

  await client.getCurrentUser();
  await client.getConversation("episode/123");
  await client.getPersonProfile("person-123");
  await client.getPersonContext("person-123", "robotics club");
  await client.listPeople("   ");

  assert.deepEqual(
    calls.map((call) => call.url),
    [
      "http://localhost:8000/users/me",
      "http://localhost:8000/conversations/episode%2F123",
      "http://localhost:8000/people/person-123/profile",
      "http://localhost:8000/people/person-123/context?query=robotics+club",
      "http://localhost:8000/people",
    ],
  );
});

test("API client preserves a path prefix in the configured base URL", async () => {
  const { client, calls } = recordingClient(() => jsonResponse([]), { baseUrl: "https://example.test/api/" });

  await client.listRecentConversations();

  assert.equal(calls[0].url, "https://example.test/api/conversations/recent");
});

test("API client surfaces backend error details and the status code", async () => {
  const { client } = recordingClient(() => jsonResponse({ detail: "Authenticated user required" }, 401));

  await assert.rejects(
    () => client.listRecentConversations(),
    (error) => error.message === "Authenticated user required" && error.status === 401,
  );
});

test("API client describes FastAPI validation errors", async () => {
  const { client } = recordingClient(() =>
    jsonResponse(
      {
        detail: [
          { loc: ["query", "limit"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" },
          { loc: ["path", "episode_id"], msg: "Input should be a valid UUID", type: "uuid_parsing" },
        ],
      },
      422,
    ),
  );

  await assert.rejects(
    () => client.getConversation("not-a-uuid"),
    /Invalid request \(limit: Input should be greater than or equal to 1; episode_id: Input should be a valid UUID\)/,
  );
});

test("API client keeps the parameter name when it is literally 'query'", async () => {
  const { client } = recordingClient(() =>
    jsonResponse({ detail: [{ loc: ["query", "query"], msg: "String should have at most 200 characters", type: "string_too_long" }] }, 422),
  );

  await assert.rejects(
    () => client.listPeople("x".repeat(201)),
    /Invalid request \(query: String should have at most 200 characters\)/,
  );
});

test("API client falls back to plain-text bodies and generic messages", async () => {
  const plain = recordingClient(() => new Response("upstream unavailable", { status: 502 }));
  await assert.rejects(() => plain.client.listPeople(), /upstream unavailable/);

  const html = recordingClient(() => new Response("<html><body>Service Unavailable</body></html>", { status: 503 }));
  await assert.rejects(() => html.client.listPeople(), /Request failed with status 503/);

  const empty = recordingClient(() => new Response("", { status: 500 }));
  await assert.rejects(() => empty.client.listPeople(), /Request failed with status 500/);
});
