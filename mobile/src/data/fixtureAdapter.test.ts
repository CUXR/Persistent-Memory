import { FixtureAdapter } from './fixtureAdapter';

describe('FixtureAdapter scenarios', () => {
  test('conversations: populated returns items, empty returns none, error rejects', async () => {
    const populated = new FixtureAdapter();
    const populatedPage = await populated.listConversations({});
    expect(populatedPage.items.length).toBeGreaterThan(0);

    const empty = new FixtureAdapter({ scenarios: { conversations: 'empty' } });
    const emptyPage = await empty.listConversations({});
    expect(emptyPage.items).toEqual([]);

    const errored = new FixtureAdapter({ scenarios: { conversations: 'error' } });
    await expect(errored.listConversations({})).rejects.toMatchObject({ retryable: true });
  });

  test('person profile: populated returns the record, empty returns a blank shape, error rejects', async () => {
    const populated = new FixtureAdapter();
    const profile = await populated.getPersonProfile('person-1');
    expect(profile.name).toBe('Alex Rivera');

    const empty = new FixtureAdapter({ scenarios: { personProfile: 'empty' } });
    const blank = await empty.getPersonProfile('person-1');
    expect(blank.facts.items).toEqual([]);

    const errored = new FixtureAdapter({ scenarios: { personProfile: 'error' } });
    await expect(errored.getPersonProfile('person-1')).rejects.toMatchObject({ retryable: true });
  });
});

describe('FixtureAdapter pagination', () => {
  test('walks conversation pages via nextCursor until exhausted', async () => {
    const adapter = new FixtureAdapter();
    const first = await adapter.listConversations({ limit: 3 });
    expect(first.items).toHaveLength(3);
    expect(first.nextCursor).not.toBeNull();

    const second = await adapter.listConversations({ limit: 3, cursor: first.nextCursor });
    expect(second.items).toHaveLength(3);
    expect(second.items[0]!.id).not.toBe(first.items[0]!.id);

    const third = await adapter.listConversations({ limit: 3, cursor: second.nextCursor });
    expect(third.nextCursor).toBeNull();
  });
});

describe('FixtureAdapter rename', () => {
  test('renamePerson updates subsequent listPeople results', async () => {
    const adapter = new FixtureAdapter();
    await adapter.renamePerson({ personId: 'person-1', displayName: 'Alexandra Rivera' });

    const page = await adapter.listPeople({});
    const renamed = page.items.find((person) => person.id === 'person-1');
    expect(renamed?.name).toBe('Alexandra Rivera');
  });
});

describe('FixtureAdapter display mode', () => {
  test('setDisplayMode settles to synced, and to error under the error scenario', async () => {
    const adapter = new FixtureAdapter();
    const result = await adapter.setDisplayMode('name_and_facts');
    expect(result.mode).toBe('name_and_facts');
    expect(result.syncStatus).toBe('synced');

    const errored = new FixtureAdapter({ scenarios: { displayMode: 'error' } });
    await expect(errored.setDisplayMode('none')).rejects.toMatchObject({ retryable: true });
  });
});
