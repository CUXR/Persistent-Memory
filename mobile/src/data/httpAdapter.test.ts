import { HttpAdapter } from './httpAdapter';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('HttpAdapter', () => {
  test('listConversations maps the wire shape (snake_case) to the domain shape', async () => {
    const fetchImpl = jest.fn(async (url: string) => {
      expect(url).toBe('http://api.test/conversations?limit=5');
      return jsonResponse({
        items: [
          {
            id: 'conv-1',
            started_at: '2026-11-01T09:00:00.000Z',
            ended_at: '2026-11-01T09:15:00.000Z',
            summary: 'Caught up on the draft',
            participants: [{ id: 'person-1', name: 'Alex Rivera' }],
            provisional_status: 'committed',
          },
        ],
        next_cursor: '5',
        total_count: 12,
      });
    }) as unknown as typeof fetch;

    const adapter = new HttpAdapter({ baseUrl: 'http://api.test', fetchImpl });
    const page = await adapter.listConversations({ limit: 5 });

    expect(page.nextCursor).toBe('5');
    expect(page.items[0]).toEqual({
      id: 'conv-1',
      startedAt: '2026-11-01T09:00:00.000Z',
      endedAt: '2026-11-01T09:15:00.000Z',
      summary: 'Caught up on the draft',
      participants: [{ id: 'person-1', name: 'Alex Rivera' }],
      provisionalStatus: 'committed',
    });
  });

  test('renamePerson sends display_name and maps the returned person', async () => {
    const fetchImpl = jest.fn(async (url: string, init?: RequestInit) => {
      expect(url).toBe('http://api.test/people/person-1');
      expect(init?.method).toBe('PATCH');
      expect(JSON.parse(init!.body as string)).toEqual({ display_name: 'Alexandra Rivera' });
      return jsonResponse({
        person: {
          id: 'person-1',
          name: 'Alexandra Rivera',
          aliases: [],
          last_seen_at: null,
          top_facts: [],
          fact_count: 0,
        },
      });
    }) as unknown as typeof fetch;

    const adapter = new HttpAdapter({ baseUrl: 'http://api.test', fetchImpl });
    const result = await adapter.renamePerson({ personId: 'person-1', displayName: 'Alexandra Rivera' });

    expect(result.person.name).toBe('Alexandra Rivera');
  });

  test('a non-2xx response rejects with an AppError, retryable only for 5xx', async () => {
    const serverError = jest.fn(
      async () => new Response(JSON.stringify({ detail: 'boom' }), { status: 503 }),
    ) as unknown as typeof fetch;
    const notFound = jest.fn(
      async () => new Response(JSON.stringify({ detail: 'Person not found' }), { status: 404 }),
    ) as unknown as typeof fetch;

    const adapterA = new HttpAdapter({ baseUrl: 'http://api.test', fetchImpl: serverError });
    await expect(adapterA.getDisplayMode()).rejects.toMatchObject({ message: 'boom', retryable: true });

    const adapterB = new HttpAdapter({ baseUrl: 'http://api.test', fetchImpl: notFound });
    await expect(adapterB.getPersonProfile('person-1')).rejects.toMatchObject({
      message: 'Person not found',
      retryable: false,
    });
  });
});
