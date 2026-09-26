import assert from 'node:assert/strict';
import { once } from 'node:events';
import { randomUUID } from 'node:crypto';
import test from 'node:test';
import { InterlocutorTracker } from './interlocutorTracker.ts';

function trackerOptions(getCurrentPersonId, overrides = {}) {
  return {
    getCurrentPersonId,
    getProfileContext: async (id) => ({ person_id: id, facts: [] }),
    loadWearerState: () => ({ wearer_person_id: randomUUID() }),
    pollIntervalMs: 1,
    ...overrides,
  };
}

function nextEvent(tracker, event) {
  return once(tracker, event, { signal: AbortSignal.timeout(2000) });
}

test('an unknown face ID from the source becomes the audio target after debouncing', async () => {
  const personId = randomUUID();
  let polls = 0;
  const tracker = new InterlocutorTracker(trackerOptions(async () => {
    polls++;
    return personId;
  }));
  const changed = nextEvent(tracker, 'interlocutor_changed');
  try {
    await tracker.start();
    const [event] = await changed;
    assert.equal(event.new_person_id, personId);
    assert.equal(tracker.activeInterlocutorPersonId, personId);
    assert.equal(event.context.person_id, personId);
    assert.ok(polls >= 2);
  } finally {
    await tracker.stop();
  }
});

test('a one-poll debounce applies immediately and loss of a face clears the target', async () => {
  let personId = randomUUID();
  const tracker = new InterlocutorTracker(trackerOptions(() => personId, { debounceConsecutivePolls: 1 }));
  try {
    const first = nextEvent(tracker, 'interlocutor_changed');
    await tracker.start();
    assert.equal((await first)[0].new_person_id, personId);
    const cleared = nextEvent(tracker, 'interlocutor_changed');
    personId = null;
    assert.equal((await cleared)[0].new_person_id, null);
    assert.equal(tracker.activeLevel1InterlocutorContext, null);
  } finally {
    await tracker.stop();
  }
});

test('a profile failure does not leave a new person paired with old context', async () => {
  const tracker = new InterlocutorTracker(trackerOptions(() => randomUUID(), {
    debounceConsecutivePolls: 1,
    getProfileContext: async () => { throw new Error('backend unavailable'); },
  }));
  const failure = nextEvent(tracker, 'error');
  try {
    await tracker.start();
    assert.equal((await failure)[0].message, 'backend unavailable');
    assert.equal(tracker.activeInterlocutorPersonId, null);
    assert.equal(tracker.activeLevel1InterlocutorContext, null);
  } finally {
    await tracker.stop();
  }
});

test('recognition returning after stop cannot activate a person', async () => {
  let resolveRecognition;
  const tracker = new InterlocutorTracker(trackerOptions(() => new Promise((resolve) => {
    resolveRecognition = resolve;
  }), { debounceConsecutivePolls: 1 }));
  let switches = 0;
  tracker.on('interlocutor_changed', () => switches++);
  await tracker.start();
  await tracker.stop();
  resolveRecognition(randomUUID());
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(switches, 0);
  assert.equal(tracker.activeInterlocutorPersonId, null);
});
