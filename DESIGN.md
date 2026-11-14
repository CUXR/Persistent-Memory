---
name: Memory glasses HUD
description: Sparse binocular text over the natural view, optimized for a 640x480 per-eye display.
colors:
  background: { light: "#000000", dark: "#000000" }
  primary: { light: "#FFFFFF", dark: "#FFFFFF" }
  secondary: { light: "#B8C2CC", dark: "#B8C2CC" }
  accent: { light: "#8BE0D0", dark: "#8BE0D0" }
typography:
  family: sans
  small: "18sp"
  body: "28sp"
  title: "36sp"
spacing:
  gap: "16dp"
  horizontal: "40dp"
  vertical: "32dp"
---

The optical display uses black for unlit pixels; both ambient modes use the same palette. White text carries the content, teal marks status and pagination, secondary gray carries gesture hints. Render the same layout in both eyes, without depth offsets.

Keep text left-aligned, short, and inside optical margins. One title, at most three body lines, one pagination label, and one gesture hint. Body line spacing is 1.25. Use immediate state changes; no decorative motion or shadows.

Android resource colors and dimensions implement these tokens. This initial system applies to the Android HUD.

The camera page preserves this hierarchy, replacing body copy with permission, active, capture, saved, or error status. Text-only capture is the default. Holding on the camera page enables a fitted live image inside the body region, with small status text on a black backing below it. A TextureView supplies one eye; Mercury MirroringView draws the same content in the other. Hiding preview or pausing restores the original body typography. Camera actions run once outside paired rendering.

The briefing page uses one fact per screen and keeps the person’s name in the title. Long titles and facts are ellipsized at one and three lines respectively. Name-only mode keeps the title and controls. Hidden mode leaves the optical view clear. The microphone page uses the same status layout as text-only camera capture.
