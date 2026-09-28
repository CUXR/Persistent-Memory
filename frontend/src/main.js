import { createApiClient } from "./api.js";
import { createRouteLoaders, getErrorMessage, shouldSyncSearch } from "./loaders.js";
import { conversationsHref, parseRoute, peopleHref } from "./router.js";
import { describeSessionSource } from "./session.js";
import { renderConversationsState } from "./views/conversations.js";
import { renderPeopleState } from "./views/people.js";

const appRoot = document.querySelector("#app");
const apiClient = createApiClient();

let currentUserState = { status: "loading", user: null };
// Monotonic counter so a slow, superseded route load can never overwrite a newer one.
let renderSequence = 0;
let lastSyncedSearchQuery = null;
const loaders = createRouteLoaders(apiClient);

export const SEARCH_MAX_LENGTH = 200;
export const ASK_MAX_LENGTH = 400;

const routes = {
  conversations: {
    href: conversationsHref(),
    label: "Recent Conversations",
    title: "Recent Conversations",
    description: "Latest ingested episodes, grouped by when they happened and who took part.",
    renderer: renderConversationsState,
    load: loaders.conversations,
    render: (data) => renderConversationsState({ status: "ready", ...data }),
  },
  people: {
    href: peopleHref(),
    label: "People",
    title: "People",
    description: "People remembered by this account, with quick metadata for browsing.",
    renderer: renderPeopleState,
    load: loaders.people,
    render: (data) => renderPeopleState({ status: "ready", ...data }),
  },
};

function sessionChipText() {
  if (currentUserState.status === "ready" && currentUserState.user) {
    const user = currentUserState.user;
    const name = user.display_name || `${user.first_name} ${user.last_name}`.trim() || user.username;
    return `Signed in as ${name}`;
  }
  if (currentUserState.status === "error") {
    return `Session unresolved. ${describeSessionSource()}`;
  }
  return describeSessionSource();
}

function updateSessionChip() {
  const chip = appRoot.querySelector("#session-chip");
  if (chip) {
    // textContent, never innerHTML: the display name comes from the API.
    chip.textContent = sessionChipText();
  }
}

/** Build the static frame once; later renders only swap the parts that change. */
function ensureShell() {
  if (appRoot.querySelector("#route-content")) {
    return;
  }

  const navHtml = Object.entries(routes)
    .map(([key, item]) => `<a class="nav-link" data-route="${key}" href="${item.href}">${item.label}</a>`)
    .join("");

  appRoot.innerHTML = `
    <div class="shell">
      <div class="frame">
        <header class="masthead">
          <div class="brand-block">
            <p class="eyebrow">Authenticated memory browser</p>
            <h1 class="title">Persistent Memory</h1>
            <p class="subtitle">A quiet view into what this account has already stored: recent conversations, known people, and the summaries around them.</p>
          </div>
          <div class="masthead-side">
            <div id="session-chip" class="session-chip" aria-live="polite"></div>
            <form id="global-search-form" class="global-search-form" role="search">
              <input
                id="global-search-input"
                class="global-search-input"
                name="search"
                type="search"
                maxlength="${SEARCH_MAX_LENGTH}"
                aria-label="Find a person in memory"
                placeholder="Find a person in memory"
              />
              <button class="nav-link active" type="submit">Search</button>
            </form>
          </div>
        </header>
        <nav class="nav" aria-label="Primary">${navHtml}</nav>
        <main class="content">
          <section class="section-heading">
            <div>
              <h2 id="route-title" tabindex="-1"></h2>
              <p id="route-description"></p>
            </div>
            <div id="route-count" class="meta-note"></div>
          </section>
          <div id="route-content" class="route-content"></div>
        </main>
      </div>
    </div>
  `;

  bindGlobalSearchForm();
  updateSessionChip();
}

function renderShell(routeKey, contentHtml, countLabel = "") {
  ensureShell();
  const route = routes[routeKey];

  appRoot.querySelectorAll(".nav-link[data-route]").forEach((link) => {
    const isActive = link.getAttribute("data-route") === routeKey;
    link.classList.toggle("active", isActive);
    if (isActive) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  appRoot.querySelector("#route-title").textContent = route.title;
  appRoot.querySelector("#route-description").textContent = route.description;
  appRoot.querySelector("#route-count").textContent = countLabel;
  appRoot.querySelector("#route-content").innerHTML = contentHtml;
}

function syncGlobalSearch(searchQuery) {
  const input = appRoot.querySelector("#global-search-input");
  if (!(input instanceof HTMLInputElement)) {
    return;
  }
  // Only overwrite what the user typed when the route's search actually changed.
  if (shouldSyncSearch(searchQuery, lastSyncedSearchQuery)) {
    input.value = searchQuery || "";
    lastSyncedSearchQuery = searchQuery;
  }
}

function bindGlobalSearchForm() {
  const form = appRoot.querySelector("#global-search-form");
  const input = appRoot.querySelector("#global-search-input");
  if (!(form instanceof HTMLFormElement) || !(input instanceof HTMLInputElement)) {
    return;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    window.location.hash = peopleHref({ searchQuery: clip(input.value, SEARCH_MAX_LENGTH) });
  });
}

function clip(value, maxLength) {
  return String(value || "").trim().slice(0, maxLength);
}

/** Remember which element had focus so a re-render can hand it back. */
function captureFocus() {
  const active = document.activeElement;
  if (!(active instanceof HTMLElement) || !appRoot.contains(active)) {
    return null;
  }
  return { id: active.id || null, inRouteContent: Boolean(active.closest("#route-content")) };
}

function restoreFocus(snapshot) {
  if (!snapshot) {
    return;
  }
  const target = snapshot.id ? appRoot.querySelector(`#${CSS.escape(snapshot.id)}`) : null;
  if (target instanceof HTMLElement) {
    target.focus();
    if (target instanceof HTMLInputElement) {
      const end = target.value.length;
      target.setSelectionRange(end, end);
    }
    return;
  }
  if (snapshot.inRouteContent) {
    appRoot.querySelector("#route-title")?.focus();
  }
}

function bindPeopleForms() {
  const directoryForm = appRoot.querySelector("#directory-search-form");
  if (directoryForm) {
    directoryForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const input = appRoot.querySelector("#directory-search-input");
      if (!(input instanceof HTMLInputElement)) {
        return;
      }
      window.location.hash = peopleHref({
        personId: directoryForm.getAttribute("data-selected-person-id") || null,
        searchQuery: clip(input.value, SEARCH_MAX_LENGTH),
        contextQuery: directoryForm.getAttribute("data-ask-query") || "",
      });
    });
  }

  const queryForm = appRoot.querySelector("#person-query-form");
  if (queryForm) {
    queryForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const personId = queryForm.getAttribute("data-person-id");
      const input = appRoot.querySelector("#person-query-input");
      if (!personId || !(input instanceof HTMLInputElement)) {
        return;
      }
      window.location.hash = peopleHref({
        personId,
        searchQuery: queryForm.getAttribute("data-search-query") || "",
        contextQuery: clip(input.value, ASK_MAX_LENGTH),
      });
    });
  }

  appRoot.querySelectorAll("[data-ask-suggestion]").forEach((button) => {
    button.addEventListener("click", () => {
      const personId = button.getAttribute("data-person-id");
      const suggestion = button.getAttribute("data-ask-suggestion");
      if (!personId || !suggestion) {
        return;
      }
      window.location.hash = peopleHref({
        personId,
        searchQuery: button.getAttribute("data-search-query") || "",
        contextQuery: suggestion,
      });
    });
  });
}

function countLabel(count) {
  return count === 1 ? "1 item" : `${count} items`;
}

async function renderCurrentRoute() {
  const sequence = ++renderSequence;
  const params = parseRoute(window.location.hash);
  const route = routes[params.key];
  const focusSnapshot = captureFocus();

  renderShell(params.key, route.renderer({ status: "loading" }));
  syncGlobalSearch(params.searchQuery);

  try {
    const data = await route.load(params);
    if (sequence !== renderSequence) {
      return;
    }
    renderShell(params.key, route.render(data), countLabel(data.items.length));
    bindPeopleForms();
  } catch (error) {
    if (sequence !== renderSequence) {
      return;
    }
    renderShell(params.key, route.renderer({ status: "error", message: getErrorMessage(error) }));
  }
  restoreFocus(focusSnapshot);
}

async function loadCurrentUser() {
  try {
    currentUserState = { status: "ready", user: await apiClient.getCurrentUser() };
  } catch {
    currentUserState = { status: "error", user: null };
  }
  updateSessionChip();
}

if (!window.location.hash) {
  // replaceState does not fire hashchange, so the initial render below happens exactly once.
  window.history.replaceState(null, "", routes.conversations.href);
}

window.addEventListener("hashchange", () => {
  renderCurrentRoute();
});

loadCurrentUser();
renderCurrentRoute();
