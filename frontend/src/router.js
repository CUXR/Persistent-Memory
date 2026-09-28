export const ROUTE_KEYS = ["conversations", "people"];
export const DEFAULT_ROUTE = "conversations";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Backend record ids are UUIDs; anything else can be rejected before a request is made. */
export function isRecordId(value) {
  return typeof value === "string" && UUID_PATTERN.test(value);
}

function safeDecode(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

/**
 * Parse a location hash such as ``#/people/<id>?search=Em&ask=hobbies`` into
 * a route descriptor. Unknown roots fall back to the conversations page.
 */
export function parseRoute(hash = "") {
  const trimmed = String(hash || "").replace(/^#\/?/, "");
  const [pathPart, queryPart = ""] = trimmed.split("?");
  const [root = "", maybeId] = pathPart.split("/");
  const params = new URLSearchParams(queryPart);
  const id = maybeId ? safeDecode(maybeId) : null;

  if (root === "people") {
    return {
      key: "people",
      personId: id,
      episodeId: null,
      searchQuery: params.get("search") || "",
      contextQuery: params.get("ask") || "",
    };
  }

  return {
    key: "conversations",
    personId: null,
    episodeId: root === "conversations" ? id : null,
    searchQuery: "",
    contextQuery: "",
  };
}

/** Build the hash for the people page, keeping only the non-empty query parts. */
export function peopleHref({ personId = null, searchQuery = "", contextQuery = "" } = {}) {
  const params = new URLSearchParams();
  if (searchQuery) {
    params.set("search", searchQuery);
  }
  if (contextQuery) {
    params.set("ask", contextQuery);
  }
  const path = personId ? `#/people/${encodeURIComponent(personId)}` : "#/people";
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return `${path}${suffix}`;
}

/** Build the hash for the conversations page, optionally opening one episode. */
export function conversationsHref(episodeId = null) {
  return episodeId ? `#/conversations/${encodeURIComponent(episodeId)}` : "#/conversations";
}
