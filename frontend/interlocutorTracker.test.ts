import test from "node:test";
import assert from "node:assert/strict";

import { InterlocutorTracker, MockInterlocutorStream } from "./interlocutorTracker.ts";

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

test("loads context only when interlocutor identity changes", async () => {
  const calls: string[] = [];
  const events: string[] = [];

  const tracker = new InterlocutorTracker({
    pollIntervalMs: 30,
    debounceConsecutivePolls: 2,
    mockStream: new MockInterlocutorStream([
      { atMs: 0, personId: null },
      { atMs: 70, personId: "person_alice" },
      { atMs: 250, personId: "person_bob" },
    ]),
    getProfileContext: async (personId: string) => {
      calls.push(personId);
      return { personId };
    },
  });

  tracker.on("interlocutor_changed", (evt) => {
    events.push(`${evt.previous_person_id}->${evt.new_person_id}`);
  });

  await tracker.start();
  await wait(380);
  await tracker.stop();

  assert.equal(tracker.activeInterlocutorPersonId, "person_bob");
  assert.deepEqual(calls, ["person_alice", "person_bob"]);
  assert.deepEqual(events, ["null->person_alice", "person_alice->person_bob"]);
});

test("does not switch on a transient identity due to debounce", async () => {
  const calls: string[] = [];

  const tracker = new InterlocutorTracker({
    pollIntervalMs: 30,
    debounceConsecutivePolls: 2,
    mockStream: new MockInterlocutorStream([
      { atMs: 0, personId: null },
      { atMs: 70, personId: "person_alice" },
      { atMs: 100, personId: null },
    ]),
    getProfileContext: async (personId: string) => {
      calls.push(personId);
      return { personId };
    },
  });

  await tracker.start();
  await wait(240);
  await tracker.stop();

  assert.equal(tracker.activeInterlocutorPersonId, null);
  assert.equal(tracker.activeLevel1InterlocutorContext, null);
  assert.deepEqual(calls, []);
});

test("continues polling after a get_profile_context failure", async () => {
  let attempt = 0;
  const errors: string[] = [];

  const tracker = new InterlocutorTracker({
    pollIntervalMs: 30,
    debounceConsecutivePolls: 2,
    mockStream: new MockInterlocutorStream([
      { atMs: 0, personId: null },
      { atMs: 70, personId: "person_alice" },
    ]),
    getProfileContext: async () => {
      attempt += 1;
      if (attempt === 1) throw new Error("temporary profile fetch failure");
      return { ok: true };
    },
  });

  tracker.on("error", (err) => {
    errors.push(err.message);
  });

  await tracker.start();
  await wait(260);
  await tracker.stop();

  assert.ok(attempt >= 2);
  assert.equal(tracker.activeInterlocutorPersonId, "person_alice");
  assert.deepEqual(errors, ["temporary profile fetch failure"]);
});
