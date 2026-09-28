import { formatDateTime } from "../format.js";
import { escapeHtml, renderStatusCard } from "../view-utils.js";

function renderConversationCard(conversation) {
  const participantLinks = conversation.participants
    .map(
      (participant) => `
        <a class="pill" href="#/people/${encodeURIComponent(participant.id)}">${escapeHtml(participant.name)}</a>
      `,
    )
    .join("");

  return `
    <article class="list-card conversation-card">
      <div class="list-row">
        <div>
          <h3>${escapeHtml(
            conversation.participants.map((participant) => participant.name).join(", ") || "Untitled conversation",
          )}</h3>
          <p class="summary">${escapeHtml(conversation.summary || "No summary available yet.")}</p>
        </div>
        <a class="nav-link" href="#/conversations/${encodeURIComponent(conversation.id)}">Open</a>
      </div>
      <div class="meta-note">${escapeHtml(formatDateTime(conversation.started_at))}</div>
      <div class="pill-row">${participantLinks}</div>
    </article>
  `;
}

function renderParticipantCard(participant) {
  const topFacts = participant.top_facts.length
    ? `
      <ul class="fact-list">
        ${participant.top_facts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join("")}
      </ul>
    `
    : `<div class="meta-note">No top facts saved yet.</div>`;

  return `
    <article class="list-card participant-card">
      <div class="list-row">
        <div>
          <h3>${escapeHtml(participant.name)}</h3>
          <div class="meta-note">Last seen ${escapeHtml(formatDateTime(participant.last_seen_at))}</div>
        </div>
        <a class="nav-link active" href="#/people/${encodeURIComponent(participant.id)}">Open person</a>
      </div>
      <div class="stat-row">
        <span class="meta-pill">${participant.fact_count} facts</span>
        <span class="meta-pill">${participant.summary_count} summaries</span>
        <span class="meta-pill">${participant.relationship_count} relationships</span>
      </div>
      ${participant.aliases.length ? `
        <div class="pill-row">
          ${participant.aliases.map((alias) => `<span class="pill">${escapeHtml(alias)}</span>`).join("")}
        </div>
      ` : ""}
      ${topFacts}
    </article>
  `;
}

function renderSelectedConversation(selectedConversation, detailError) {
  if (detailError) {
    return `
      ${renderStatusCard("Could not load this conversation", detailError, "error")}
      <div class="empty-cta"><a class="nav-link" href="#/conversations">Back to recent conversations</a></div>
    `;
  }

  if (!selectedConversation) {
    return renderStatusCard(
      "Select a conversation",
      "Choose a recent episode to inspect its full summary, transcript, and linked participants.",
    );
  }

  return `
    <section class="list-card profile-hero">
      <div class="list-row">
        <div>
          <h3>${escapeHtml(selectedConversation.participants.map((participant) => participant.name).join(", ") || "Conversation detail")}</h3>
          <div class="meta-note">${escapeHtml(formatDateTime(selectedConversation.started_at))}</div>
        </div>
        <a class="nav-link active" href="#/conversations">Close</a>
      </div>
      <p class="summary">${escapeHtml(selectedConversation.summary || "No summary available yet.")}</p>
    </section>
    <section class="detail-grid">
      <section class="list-card">
        <h3>Transcript</h3>
        <div class="transcript-block">${escapeHtml(selectedConversation.transcript || "No transcript stored for this episode.")}</div>
      </section>
      <section class="list-card">
        <h3>Participants</h3>
        <div class="list">
          ${selectedConversation.participants.map(renderParticipantCard).join("")}
        </div>
      </section>
    </section>
  `;
}

export function renderConversationsState(state) {
  if (state.status === "loading") {
    return renderStatusCard("Loading conversations", "Pulling the latest memory episodes now.");
  }

  if (state.status === "error") {
    return renderStatusCard("Could not load conversations", state.message, "error");
  }

  const items = state.items || [];

  if (items.length === 0 && !state.selectedConversation) {
    return `
      ${renderStatusCard("No conversations yet", "As new episodes are ingested, they will show up here.")}
      <div class="empty-cta">Start with a seeded or ingested episode and refresh this page.</div>
      ${state.detailError ? renderStatusCard("Could not load this conversation", state.detailError, "error") : ""}
    `;
  }

  return `
    <section class="people-layout">
      <div class="people-list-column">
        <section class="list">${items.map(renderConversationCard).join("")}</section>
      </div>
      <div class="people-detail-column">
        ${renderSelectedConversation(state.selectedConversation, state.detailError)}
      </div>
    </section>
  `;
}
