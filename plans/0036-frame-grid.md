# 0036 — view: camera frames as a grid row per observation

Issue: #239.

## Problem

`view` reports render each embedded camera frame as a full-width block image.
The renderer's own structure makes the layout worse than it needs to be: in
`_render_frame_parts`, the `camera '<name>' (step N):` label line is just the
tail of the buffered text run, so each frame arrives as
`…text ending in label</div><img>` — one image per row, label/image
association implicit. On a 3-camera rig every observation costs three screens
of scrolling.

The rig works around it by post-patching rendered pages with an end-of-body
script that regroups the DOM — which visibly re-flows the page once the
multi-MB document finishes parsing. The renderer already holds the label
(`_FRAME_LABEL_RE` matched, name and step in hand) at the moment it embeds
the image; it can emit the right markup in the first place.

## Design

All changes in `_html.py`; no CLI surface, no schema, no new module.

### 1. Markup (`_render_frame_parts`)

When a frame embeds successfully, emit a captioned figure instead of
appending the label text + bare `<img>`:

```html
<figure class="frame-cell">
  <figcaption>camera 'top_cam' (step 0)</figcaption>
  <img class="frame" loading="lazy" alt="camera top_cam step 0" src="…">
</figure>
```

- The caption text is the matched label line minus its trailing colon,
  escaped, and **deleted from the buffered run by index**: `pending` grows to
  `(name, step, label_index)` where `label_index = len(buffered)` recorded
  when the label is appended, and a successful embed does
  `del buffered[label_index]`. The index cannot dangle: `pending` is always
  consumed (embed or overwrite) before any flush empties the buffer. This is
  the mechanism, not "pop the last line" — parts may intervene between label
  and placeholder (text, `[image]` placeholders), the embed still fires, and
  those parts must survive in the text run. When they do intervene, they
  render in the pre-figure text run, so their text appears before the figure
  whose caption is the (earlier) label — a deliberate, accepted reordering
  for a degenerate input.
- Two labels before one placeholder: the later label drives the embed (it
  overwrote `pending`, today's behavior) and becomes the caption; the
  earlier label line simply stays in the text run.
- **Empty runs are never emitted.** The pre-figure flush emits a
  `<div class="content">` only when the label-stripped join is non-empty —
  the canonical `[label, placeholder]` observation strips down to an empty
  buffer and must not produce `<div class="content"></div>` (the suite's
  existing invariant, `test_stored_frame_embeds_between_escaped_text_runs`).
  Same guard on the final flush when `embedded > 0`; the `embedded == 0`
  text-only path keeps today's semantics exactly.
- Consecutive figures join one `<div class="frame-row">`; the row closes
  when a non-empty text run is flushed between embeds. A run that strips or
  joins to empty (e.g. a stray `""` part) flushes nothing and does not close
  the row. Observations in separate messages always get separate rows —
  each `_render_frame_parts` call emits its own.
- A frame that fails to embed (budget, missing artifact) keeps today's
  behavior exactly: label and placeholder stay in the text run, no figure,
  no empty cell, no row. `--no-frames` (`frame_ctx is None`) never reaches
  this code and is untouched.

### 2. CSS

```css
.frame-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 6px 0; }
.frame-cell { margin: 0; min-width: 0; }
.frame-cell figcaption { font-size: 12px; color: var(--muted); margin-bottom: 2px; overflow-wrap: anywhere; }
.frame-cell img.frame { width: 100%; max-width: 448px; margin: 0; cursor: zoom-in; }
.frame-cell.wide { grid-column: 1 / -1; }
.frame-cell.wide img.frame { max-width: none; cursor: zoom-out; }
```

`auto-fit` + `minmax` handles any camera count and viewport: 3 cameras sit
3-up at the report's 1120px main width, degrading to fewer columns on
narrow screens. Stored frames are capped at 448px on their longest side
(`_FRAME_MAX_SIDE`), so `max-width: 448px` keeps a single-camera row at
today's natural-size look instead of upscaling a 448px image to the full
column. The `.wide` expand deliberately lifts that cap: an expanded cell is
an upscaled (soft) image, same trade the current full-width-on-request
behavior would make — bigger beats sharper when a human is squinting at a
gripper. The existing `img.frame` rule (natural size, `max-width: 100%`)
still governs images outside rows, e.g. wire-blob embeds.

### 3. Expand on click

Frames in a 3-up row are a third of their old size, so one interaction is
justified: clicking a frame toggles `wide` on its figure (`grid-column: 1 /
-1` — full row width, in place, no overlay). One small event-delegated
inline script in the document (the report currently ships zero JS; this adds
a 4-line click handler on `document`, no per-image listeners). The handler
matches `event.target.closest('.frame-cell')` and no-ops otherwise —
wire-blob images (which also carry `img.frame`) and arbitrary clicks are
untouched. The report stays fully readable with JS disabled (grid and
captions are pure CSS/markup; only the expand toggle needs JS).

## Non-goals

Lightbox/overlay zoom, keyboard navigation, per-camera column ordering,
frame diffing, and any change to `--no-frames`/budget semantics. The
directory index page (plan 0035) is untouched.

## Tests (tests/test_html_view.py)

- One observation with three frames renders one `.frame-row` containing
  three `figure.frame-cell`s, each figcaption carrying its camera label
  (colon stripped), images in part order; the label line no longer appears
  in the preceding `.content` text; `<div class="content"></div>` count is
  zero (the suite's existing invariant, asserted again here).
- Two observations (separate user messages) yield two separate rows.
- Intervening non-empty text between embeds splits rows; a stray `""` text
  part between embeds does not (one row, no empty div).
- Text between a label and its placeholder: embed still happens, the
  intervening text stays in the pre-figure run, the caption is the label.
- Two labels before one placeholder: later label captions the figure,
  earlier label line survives in the text run.
- Single-frame observation: row + one cell (no special case).
- Failed embed (budget truncation path) leaves label and placeholder in the
  text run and emits no figure/row — assert today's exact fallback; the
  existing `test_all_frame_misses_are_byte_identical_to_frames_off` must
  stay green unmodified (CSS/script ship unconditionally in both renders).
- Escaping: camera name containing `<b>` is escaped in both figcaption and
  alt.
- The click script (`closest('.frame-cell')` delegation) and
  `.frame-cell.wide` rule are present in the document.

## Rollout

Pure rendering change in one module. Changelog `### Added` (or `### Changed`
— it alters default report layout) under `[Unreleased]` with
`(plan 0036, #239)`. After the next release, the rig deletes its local
`ensure_camgrid` patch.
