export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function renderStatusCard(title, message, kind = "info") {
  return `
    <section class="status-card ${kind === "error" ? "error" : ""}">
      <strong>${escapeHtml(title)}</strong>
      <div>${escapeHtml(message)}</div>
    </section>
  `;
}
