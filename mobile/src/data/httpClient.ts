import type { AppError } from '../domain';

export interface HttpClientOptions {
  baseUrl: string;
  fetchImpl?: typeof fetch;
  /** Device/user identity isn't settled yet (#49) — placeholder header injection point. */
  getAuthHeaders?: () => Record<string, string>;
}

function joinUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}

/** FastAPI error bodies are either { detail: string } or { detail: [{ msg, loc }] }. */
async function describeErrorBody(response: Response): Promise<string> {
  let text = '';
  try {
    text = await response.text();
  } catch {
    text = '';
  }
  try {
    const body = JSON.parse(text) as { detail?: string | { msg?: string }[] };
    if (typeof body.detail === 'string' && body.detail) {
      return body.detail;
    }
    if (Array.isArray(body.detail)) {
      const messages = body.detail.map((entry) => entry?.msg).filter((msg): msg is string => Boolean(msg));
      if (messages.length) {
        return messages.join('; ');
      }
    }
  } catch {
    // Not JSON; fall through to the plain-text body.
  }
  return text.trim() || `Request failed with status ${response.status}`;
}

/** Thin fetch wrapper: builds the URL, attaches auth headers, and normalizes failures into AppError. */
export function createHttpClient(options: HttpClientOptions) {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error('A fetch implementation is required');
  }

  async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      Accept: 'application/json',
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.getAuthHeaders?.() ?? {}),
      ...((init.headers as Record<string, string>) ?? {}),
    };

    let response: Response;
    try {
      response = await fetchImpl(joinUrl(options.baseUrl, path), { ...init, headers });
    } catch (cause) {
      const error: AppError = { message: 'Network request failed', retryable: true, cause };
      throw error;
    }

    if (!response.ok) {
      const message = await describeErrorBody(response);
      const error: AppError = { message, retryable: response.status >= 500, cause: response.status };
      throw error;
    }

    return (await response.json()) as T;
  }

  return { request };
}
