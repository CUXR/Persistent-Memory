import { formatDateTime } from "../format.js";
import { peopleHref } from "../router.js";
import { escapeHtml, renderStatusCard } from "../view-utils.js";

const SUGGESTED_QUERIES = [
  "What hobbies do they have?",
  "What do we know about their relationships?",
  "What came up in recent conversations?",
];

function renderFacts(topFacts) {
  if (topFacts.length === 0) {
    return `<div class="meta-note">No fact snippets saved yet.</div>`;
  }

  return `
    <ul class="fact-list">
      ${topFacts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join("")}
    </ul>
  `;
}

function renderAliases(aliases) {
  if (aliases.length === 0) {
    return "";
  }

  return `
    <div class="pill-row">
      ${aliases.map((alias) => `<span class="pill">${escapeHtml(alias)}</span>`).join("")}
    </div>
  `;
}

function renderStats(person) {
  const stats = [
    `${person.fact_count} facts`,
    `${person.summary_count} summaries`,
    `${person.relationship_count} relationships`,
  ];

  return `
    <div class="stat-row">
      ${stats.map((stat) => `<span class="meta-pill">${escapeHtml(stat)}</span>`).join("")}
    </div>
  `;
}

function renderPersonCard(person, searchQuery, impliedSelection) {
  // A resolver-implied selection lives on the search route itself, so "hiding" it means clearing the search.
  const actionLabel = person.is_selected ? (impliedSelection ? "Clear search" : "Hide profile") : "View profile";
  const actionHref = person.is_selected
    ? (impliedSelection ? peopleHref() : peopleHref({ searchQuery }))
    : peopleHref({ personId: person.id, searchQuery });

  return `
    <article class="list-card person-card ${person.is_selected ? "person-card-selected" : ""}">
      <div class="list-row">
        <div>
          <h3>${escapeHtml(person.name)}</h3>
          <div class="meta-note">Last seen ${escapeHtml(formatDateTime(person.last_seen_at))}</div>
        </div>
        <a class="nav-link ${person.is_selected ? "active" : ""}" href="${actionHref}">${escapeHtml(actionLabel)}</a>
      </div>
      ${renderStats(person)}
      ${renderAliases(person.aliases)}
      ${renderFacts(person.top_facts)}
    </article>
  `;
}

function renderResolutionBanner(directory, searchQuery) {
  if (!searchQuery) {
    return `
      <section class="list-card">
        <div class="list-row">
          <div>
            <h3>Browse people</h3>
            <div class="meta-note">Search by name, alias, or remembered facts. Resolver-backed matches will show up here when the query points to a known person.</div>
          </div>
          <div class="meta-pill">${directory.items.length} people</div>
        </div>
      </section>
    `;
  }

  const resolution = directory.resolution;
  if (resolution?.is_ambiguous) {
    return `
      <section class="list-card highlight-card">
        <div class="list-row">
          <div>
            <h3>Multiple people matched “${escapeHtml(searchQuery)}”</h3>
            <div class="meta-note">The backend resolver found several candidates. Use the hints below to pick the right profile.</div>
          </div>
          <div class="meta-pill">${resolution.candidates.length} candidates</div>
        </div>
        <div class="candidate-grid">
          ${resolution.candidates.map((candidate) => renderCandidateCard(candidate, searchQuery)).join("")}
        </div>
      </section>
    `;
  }

  if (resolution?.person_id) {
    return `
      <section class="list-card highlight-card">
        <div class="list-row">
          <div>
            <h3>Resolver match for “${escapeHtml(searchQuery)}”</h3>
            <div class="meta-note">The existing backend resolver found a specific person for this query.</div>
          </div>
          <div class="meta-pill">${directory.items.length} result${directory.items.length === 1 ? "" : "s"}</div>
        </div>
      </section>
    `;
  }

  return `
    <section class="list-card">
      <div class="list-row">
        <div>
          <h3>Search results for “${escapeHtml(searchQuery)}”</h3>
          <div class="meta-note">These results are filtered from the real people directory using names, aliases, and remembered fact snippets.</div>
        </div>
        <div class="meta-pill">${directory.items.length} result${directory.items.length === 1 ? "" : "s"}</div>
      </div>
    </section>
  `;
}

function renderCandidateCard(candidate, searchQuery) {
  const hintLines = Object.entries(candidate.hints || {})
    .filter(([, values]) => Array.isArray(values) && values.length > 0)
    .map(([category, values]) => `
      <div class="candidate-hint">
        <strong>${escapeHtml(category.replaceAll("_", " "))}</strong>
        <div>${escapeHtml(values[0])}</div>
      </div>
    `)
    .join("");

  return `
    <article class="list-card candidate-card">
      <div class="list-row">
        <h3>${escapeHtml(candidate.name)}</h3>
        <a class="nav-link active" href="${peopleHref({ personId: candidate.person_id, searchQuery })}">Open</a>
      </div>
      ${hintLines || `<div class="meta-note">No disambiguation hints available.</div>`}
    </article>
  `;
}

function renderRetrievalResults(selectedContext, contextQuery, contextError) {
  if (contextError) {
    return renderStatusCard("Could not search this memory", contextError, "error");
  }

  if (!contextQuery) {
    return `
      <section class="list-card">
        <h3>Ready to search</h3>
        <div class="meta-note">Use the query box above to ask the real retrieval service for the most relevant facts, summaries, and relationships for this person.</div>
      </section>
    `;
  }

  const totalMatches = (selectedContext?.facts?.length || 0)
    + (selectedContext?.summaries?.length || 0)
    + (selectedContext?.edges?.length || 0);

  if (!selectedContext || totalMatches === 0) {
    return `
      <section class="list-card">
        <h3>Relevant memory</h3>
        <div class="meta-note">No relevant memories were returned for “${escapeHtml(contextQuery)}”. Try a more specific detail like a hobby, workplace, or relationship.</div>
      </section>
    `;
  }

  const sections = [
    renderProfileSection(
      "Relevant facts",
      selectedContext.facts,
      "No relevant facts matched this question.",
      (fact) => `<div class="summary">${escapeHtml(fact.fact_text)}</div>`,
    ),
    renderProfileSection(
      "Relevant summaries",
      selectedContext.summaries,
      "No relevant summaries matched this question.",
      (summary) => `
        <div class="summary">
          ${escapeHtml(summary.summary_text)}
          <div class="meta-note">${escapeHtml(formatDateTime(summary.episode_time_start || summary.created_at))}</div>
        </div>
      `,
    ),
    renderProfileSection(
      "Relevant relationships",
      selectedContext.edges,
      "No relevant relationships matched this question.",
      (edge) => `<div class="summary">${escapeHtml(`${edge.relation}: ${edge.dst_name}`)}</div>`,
    ),
  ];

  return `
    <section class="list-card highlight-card">
      <div class="list-row">
        <div>
          <h3>Relevant memory</h3>
          <div class="meta-note">Showing the strongest stored matches for “${escapeHtml(contextQuery)}”.</div>
        </div>
        <div class="meta-pill">${totalMatches} match${totalMatches === 1 ? "" : "es"}</div>
      </div>
    </section>
    ${sections.join("")}
  `;
}

function renderProfileSection(title, items, emptyLabel, renderItem) {
  if (items.length === 0) {
    return `
      <section class="list-card">
        <h3>${escapeHtml(title)}</h3>
        <div class="meta-note">${escapeHtml(emptyLabel)}</div>
      </section>
    `;
  }

  return `
    <section class="list-card">
      <h3>${escapeHtml(title)}</h3>
      <div class="list">
        ${items.map(renderItem).join("")}
      </div>
    </section>
  `;
}

function renderSelectedProfile(selectedProfile, searchQuery, impliedSelection) {
  if (!selectedProfile) {
    return "";
  }

  const { person, profile } = selectedProfile;
  const closeHref = impliedSelection ? peopleHref() : peopleHref({ searchQuery });

  return `
    <section class="list-card profile-hero">
      <div class="list-row">
        <div>
          <h3>${escapeHtml(person.name)}</h3>
          <div class="meta-note">Detailed memory context loaded from the existing backend profile store.</div>
        </div>
        <a class="nav-link active" href="${closeHref}">${impliedSelection ? "Clear search" : "Close"}</a>
      </div>
      ${renderAliases(person.aliases)}
      <div class="stat-row">
        <span class="meta-pill">${profile.facts.length} facts</span>
        <span class="meta-pill">${profile.prefs.length} preferences</span>
        <span class="meta-pill">${profile.summaries.length} summaries</span>
        <span class="meta-pill">${profile.edges_from.length} relationships</span>
      </div>
    </section>
    <section class="detail-grid">
      ${renderProfileSection(
        "Facts",
        profile.facts,
        "No saved facts yet.",
        (fact) => `<div class="summary">${escapeHtml(fact.fact_text)}</div>`,
      )}
      ${renderProfileSection(
        "Preferences",
        profile.prefs,
        "No saved preferences yet.",
        (pref) => `<div class="summary">${escapeHtml(pref.pref_text)}</div>`,
      )}
      ${renderProfileSection(
        "Summaries",
        profile.summaries,
        "No saved summaries yet.",
        (summary) => `
          <div class="summary">
            ${escapeHtml(summary.summary_text)}
            <div class="meta-note">${escapeHtml(formatDateTime(summary.episode_time_start || summary.created_at))}</div>
          </div>
        `,
      )}
      ${renderProfileSection(
        "Relationships",
        profile.edges_from,
        "No saved relationship edges yet.",
        (edge) => `<div class="summary">${escapeHtml(`${edge.relation}: ${edge.dst_name}`)}</div>`,
      )}
    </section>
  `;
}

function renderSearchSection(state) {
  // A resolver-implied selection must be re-derived from the next search, not pinned to the URL.
  const selectedPersonId = state.impliedSelection ? "" : state.selectedProfile?.person?.id || "";
  const askQuery = state.impliedSelection ? "" : state.contextQuery || "";
  return `
    <section class="list-card search-shell">
      <div class="list-row">
        <div>
          <h3>People search</h3>
          <div class="meta-note">This search is backed by the real directory endpoint and person resolver.</div>
        </div>
      </div>
      <form
        id="directory-search-form"
        class="query-form"
        data-selected-person-id="${escapeHtml(selectedPersonId)}"
        data-ask-query="${escapeHtml(askQuery)}"
      >
        <input
          id="directory-search-input"
          class="query-input"
          name="search"
          type="search"
          maxlength="200"
          aria-label="Search people by name, alias, or remembered detail"
          placeholder="Search by name, alias, or remembered detail"
          value="${escapeHtml(state.searchQuery || "")}"
        />
        <button class="nav-link active" type="submit">Search people</button>
        ${state.searchQuery ? `<a class="nav-link" href="${peopleHref()}">Clear</a>` : ""}
      </form>
    </section>
  `;
}

function renderRetrievalComposer(selectedProfile, contextQuery, searchQuery) {
  if (!selectedProfile) {
    return "";
  }

  return `
    <section class="list-card search-shell">
      <div class="list-row">
        <div>
          <h3>Ask this memory</h3>
          <div class="meta-note">This search uses the backend retrieval service over stored facts, summaries, and relationships.</div>
        </div>
      </div>
      <form
        id="person-query-form"
        class="query-form"
        data-person-id="${escapeHtml(selectedProfile.person.id)}"
        data-search-query="${escapeHtml(searchQuery || "")}"
      >
        <input
          id="person-query-input"
          class="query-input"
          name="query"
          type="search"
          maxlength="400"
          aria-label="Ask a question about this person"
          placeholder="What do we know about this person?"
          value="${escapeHtml(contextQuery || "")}"
        />
        <button class="nav-link active" type="submit">Search memory</button>
      </form>
      <div class="pill-row">
        ${SUGGESTED_QUERIES.map((suggestion) => `
          <button
            class="suggestion-pill"
            type="button"
            data-person-id="${escapeHtml(selectedProfile.person.id)}"
            data-search-query="${escapeHtml(searchQuery || "")}"
            data-ask-suggestion="${escapeHtml(suggestion)}"
          >
            ${escapeHtml(suggestion)}
          </button>
        `).join("")}
      </div>
    </section>
  `;
}

export function renderPeopleState(state) {
  if (state.status === "loading") {
    return renderStatusCard("Loading people", "Gathering the current memory directory.");
  }

  if (state.status === "error") {
    return renderStatusCard("Could not load people", state.message, "error");
  }

  const directory = state.directory || { items: state.items || [], query: null, resolution: null };
  const impliedSelection = Boolean(state.impliedSelection);
  const items = (state.items || []).map((person) => ({
    ...person,
    is_selected: state.selectedProfile?.person?.id === person.id,
  }));

  const shouldShowEmptyDirectory = items.length === 0 && !state.selectedProfile;

  let detailColumn;
  if (state.detailError) {
    detailColumn = `
      ${renderStatusCard("Could not load this person", state.detailError, "error")}
      <div class="empty-cta"><a class="nav-link" href="${impliedSelection ? peopleHref() : peopleHref({ searchQuery: state.searchQuery })}">${impliedSelection ? "Clear search" : "Back to the directory"}</a></div>
    `;
  } else if (state.selectedProfile) {
    detailColumn = `
      ${renderSelectedProfile(state.selectedProfile, state.searchQuery, impliedSelection)}
      ${renderRetrievalComposer(state.selectedProfile, state.contextQuery, state.searchQuery)}
      ${renderRetrievalResults(state.selectedContext, state.contextQuery, state.contextError)}
    `;
  } else {
    detailColumn = renderStatusCard(
      "Select a person",
      "Choose someone from the directory to inspect their stored profile and ask memory questions.",
    );
  }

  return `
    ${renderSearchSection(state)}
    ${renderResolutionBanner(directory, state.searchQuery)}
    ${shouldShowEmptyDirectory ? `
      ${renderStatusCard("No people found", state.searchQuery
        ? "No people matched that search yet."
        : "This account does not have any saved people yet.")}
      <div class="empty-cta">Ingest a conversation or run the seed script to populate the memory store.</div>
      ${state.detailError ? renderStatusCard("Could not load this person", state.detailError, "error") : ""}
    ` : `
      <section class="people-layout">
        <div class="people-list-column">
          <section class="list">${items.map((person) => renderPersonCard(person, state.searchQuery, impliedSelection)).join("")}</section>
        </div>
        <div class="people-detail-column">
          ${detailColumn}
        </div>
      </section>
    `}
  `;
}
