import { getAuthenticatedUserId } from "./session.js";

function defaultBaseUrl() {
  return globalThis?.__PERSISTENT_MEMORY_API_BASE_URL__ || "http://localhost:8000";
}

function joinUrl(baseUrl, path) {
  // Plain string join so a base URL with a path prefix (e.g. https://host/api) is preserved.
  return `${String(baseUrl).replace(/\/+$/, "")}/${String(path).replace(/^\/+/, "")}`;
}

function describeValidationErrors(detail) {
  // FastAPI returns a list of {loc, msg} objects for 422 validation failures.
  const messages = detail
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return "";
      }
      const loc = Array.isArray(entry.loc) ? entry.loc : [];
      const parts = ["query", "path", "body", "header"].includes(loc[0]) ? loc.slice(1) : loc;
      const location = parts.join(".");
      return location ? `${location}: ${entry.msg}` : entry.msg || "";
    })
    .filter(Boolean);
  return messages.length ? `Invalid request (${messages.join("; ")})` : "";
}

async function parseError(response) {
  // Read the body exactly once; it may be JSON, plain text, or empty.
  let text = "";
  try {
    text = await response.text();
  } catch {
    text = "";
  }

  try {
    const body = JSON.parse(text);
    if (typeof body?.detail === "string" && body.detail) {
      return body.detail;
    }
    if (Array.isArray(body?.detail)) {
      const described = describeValidationErrors(body.detail);
      if (described) {
        return described;
      }
    }
  } catch {
    // Not JSON; fall through to the plain-text body.
  }

  const trimmed = text.trim();
  if (trimmed && !trimmed.startsWith("<")) {
    return trimmed;
  }
  return `Request failed with status ${response.status}`;
}

export function createApiClient({
  baseUrl = defaultBaseUrl(),
  fetchImpl = globalThis.fetch?.bind(globalThis),
  getUserId = getAuthenticatedUserId,
} = {}) {
  if (!fetchImpl) {
    throw new Error("A fetch implementation is required");
  }

  async function request(path) {
    const headers = {
      Accept: "application/json",
    };
    const userId = getUserId?.();
    if (userId) {
      headers["X-User-Id"] = userId;
    }

    const response = await fetchImpl(joinUrl(baseUrl, path), {
      method: "GET",
      credentials: "include",
      headers,
    });

    if (!response.ok) {
      const error = new Error(await parseError(response));
      error.status = response.status;
      throw error;
    }

    return response.json();
  }

  return {
    getCurrentUser() {
      return request("/users/me");
    },
    listRecentConversations() {
      return request("/conversations/recent");
    },
    getConversation(episodeId) {
      return request(`/conversations/${encodeURIComponent(episodeId)}`);
    },
    listPeople(query = "") {
      const params = new URLSearchParams();
      if (query.trim()) {
        params.set("query", query.trim());
      }
      const suffix = params.toString() ? `?${params.toString()}` : "";
      return request(`/people${suffix}`);
    },
    getPersonProfile(personId) {
      return request(`/people/${encodeURIComponent(personId)}/profile`);
    },
    getPersonContext(personId, query) {
      const params = new URLSearchParams({ query });
      return request(`/people/${encodeURIComponent(personId)}/context?${params.toString()}`);
    },
  };
}
