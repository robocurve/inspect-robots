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
  escaped. It is **removed from the buffered text run** (it currently
  arrives as the run's last line): the label's whole content lives in the
  figcaption, never duplicated.
- Consecutive figures with no intervening text join one
  `<div class="frame-row">`; any non-empty text run in between closes the
  current row. One observation's cameras (label/placeholder pairs
  back-to-back in the message parts) therefore form exactly one row.
- A frame that fails to embed (budget, missing artifact) keeps today's
  behavior exactly: its label stays in the text run, no figure, no empty
  cell. `--no-frames` placeholders are untouched.
- The buffered-text flush logic (`embedded == 0 or buffered`) keeps its
  current semantics for text-only runs.

### 2. CSS

```css
.frame-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 6px 0; }
.frame-cell { margin: 0; min-width: 0; }
.frame-cell figcaption { font-size: 12px; color: var(--muted); margin-bottom: 2px; overflow-wrap: anywhere; }
.frame-cell img.frame { width: 100%; margin: 0; cursor: zoom-in; }
.frame-cell.wide { grid-column: 1 / -1; }
.frame-cell.wide img.frame { cursor: zoom-out; }
```

`auto-fit` + `minmax` handles any camera count and viewport: 3 cameras sit
side by side on a normal window, degrade to fewer columns on narrow screens,
and a single camera fills the row (matching today's full-width look). The
existing `img.frame` rule keeps `max-width: 100%` for any image outside a
row (none are emitted today, but the rule stays as a safety net).

### 3. Expand on click

Frames in a 3-up row are a third of their old size, so one interaction is
justified: clicking a frame toggles `wide` on its figure (`grid-column: 1 /
-1` — full row width, in place, no overlay). One small event-delegated
inline script in the document (the report currently ships zero JS; this adds
a 4-line click handler on `document`, no per-image listeners, so page weight
and parse cost stay negligible). No other behavior changes; the report stays
fully readable with JS disabled (grid and captions are pure CSS/markup).

## Non-goals

Lightbox/overlay zoom, keyboard navigation, per-camera column ordering,
frame diffing, and any change to `--no-frames`/budget semantics. The
directory index page (plan 0035) is untouched.

## Tests (tests/test_html_view.py)

- One observation with three frames renders one `.frame-row` containing
  three `figure.frame-cell`s, each figcaption carrying its camera label
  (colon stripped), images in part order; the label line no longer appears
  in the preceding `.content` text.
- Two observations in one transcript yield two separate rows.
- Intervening non-empty text between embeds splits rows.
- Single-frame observation: row + one cell (no special case).
- Failed embed (budget truncation path) leaves the label in the text run
  and emits no figure/row — assert today's exact fallback.
- Escaping: camera name containing `<b>` is escaped in both figcaption and
  alt.
- The click script and `.frame-cell.wide` rule are present in the document.

## Rollout

Pure rendering change in one module. Changelog `### Added` (or `### Changed`
— it alters default report layout) under `[Unreleased]` with
`(plan 0036, #239)`. After the next release, the rig deletes its local
`ensure_camgrid` patch.
