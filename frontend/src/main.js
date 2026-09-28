import { createApiClient } from "./api.js";
import { conversationsHref, isRecordId, parseRoute, peopleHref } from "./router.js";
import { describeSessionSource } from "./session.js";
import { renderConversationsState } from "./views/conversations.js";
import { renderPeopleState } from "./views/people.js";

const appRoot = document.querySelector("#app");
const apiClient = createApiClient();

let currentUserState = { status: "loading", user: null };
// Monotonic counter so a slow, superseded route load can never overwrite a newer one.
let renderSequence = 0;
let lastSyncedSearchQuery = null;

function settle(promise) {
  return promise.then(
    (value) => ({ value, error: null }),
    (error) => ({ value: null, error: getErrorMessage(error) }),
  );
}

const NOTHING = Promise.resolve({ value: null, error: null });

function invalidId(label) {
  return Promise.resolve({ value: null, error: `${label} id in the address is not valid.` });
}

const routes = {
  conversations: {
    href: conversationsHref(),
    label: "Recent Conversations",
    title: "Recent Conversations",
    description: "Latest ingested episodes, grouped by when they happened and who took part.",
    renderer: renderConversationsState,
    async load({ episodeId }) {
      // The list drives the page state; a failing detail fetch only affects the detail pane.
      let detailPromise = NOTHING;
      if (episodeId) {
        detailPromise = isRecordId(episodeId) ? settle(apiClient.getConversation(episodeId)) : invalidId("The conversation");
      }
      const [items, detail] = await Promise.all([apiClient.listRecentConversations(), detailPromise]);
      return { items, selectedConversation: detail.value, detailError: detail.error };
    },
    render: (data) => renderConversationsState({ status: "ready", ...data }),
  },
  people: {
    href: peopleHref(),
    label: "People",
    title: "People",
    description: "People remembered by this account, with quick metadata for browsing.",
    renderer: renderPeopleState,
    async load({ personId, searchQuery, contextQuery }) {
      const directory = await apiClient.listPeople(searchQuery);
      if (personId && !isRecordId(personId)) {
        const invalid = await invalidId("The person");
        return {
          directory,
          items: directory.items,
          selectedProfile: null,
          detailError: invalid.error,
          selectedContext: null,
          contextError: null,
          contextQuery,
          searchQuery,
          impliedSelection: false,
        };
      }
      const impliedPersonId = !personId && directory.resolution?.person_id ? directory.resolution.person_id : null;
      const selectedPersonId = personId || impliedPersonId;
      const [profile, context] = await Promise.all([
        selectedPersonId ? settle(apiClient.getPersonProfile(selectedPersonId)) : NOTHING,
        selectedPersonId && contextQuery ? settle(apiClient.getPersonContext(selectedPersonId, contextQuery)) : NOTHING,
      ]);
      return {
        directory,
        items: directory.items,
        selectedProfile: profile.value,
        detailError: profile.error,
        selectedContext: context.value,
        contextError: context.error,
        contextQuery,
        searchQuery,
        impliedSelection: Boolean(impliedPersonId),
      };
    },
    render: (data) => renderPeopleState({ status: "ready", ...data }),
  },
};

function getErrorMessage(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Something unexpected happened while loading this page.";
}

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
              <h2 id="route-title"></h2>
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
  if (searchQuery !== lastSyncedSearchQuery) {
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
    window.location.hash = peopleHref({ searchQuery: input.value.trim() });
  });
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
        searchQuery: input.value.trim(),
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
        contextQuery: input.value.trim(),
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
