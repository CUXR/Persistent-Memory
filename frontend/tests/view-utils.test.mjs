import assert from "node:assert/strict";
import test from "node:test";

import { escapeHtml, renderStatusCard } from "../src/view-utils.js";

test("escapeHtml neutralizes markup, quotes, and non-string values", () => {
  assert.equal(escapeHtml('<a href="x">&\'</a>'), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;");
  assert.equal(escapeHtml(42), "42");
  assert.equal(escapeHtml(null), "null");
  assert.equal(escapeHtml(""), "");
});

test("renderStatusCard escapes its content and marks errors", () => {
  const info = renderStatusCard("Title <b>", "Body & more");
  assert.match(info, /Title &lt;b&gt;/);
  assert.match(info, /Body &amp; more/);
  assert.doesNotMatch(info, /class="status-card error"/);

  const error = renderStatusCard("Oops", "<script>alert(1)</script>", "error");
  assert.match(error, /status-card error/);
  assert.doesNotMatch(error, /<script>/);
});
