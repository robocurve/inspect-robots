# 0065 — Transcript rail: live turn highlight + click-to-seek on the composite run video

- **Status:** draft (R0)
- **Issue:** #352
- **Lineage:** successor to PR #272's headline feature (plan 0039's multicam
  player with synced transcript rail). #272 synchronized N per-camera panes
  with custom JS and disk-side `media/` symlinks; plans 0060/0063 have since
  rebuilt the report around a single embedded composite MP4, which makes the
  rail *simpler*: one `<video>` to observe and seek, no pane sync, no media
  directory. The pane player and symlink plumbing are superseded; this plan
  ports only the rail.

## Problem

The composite run video (plan 0063) and the turn-oriented transcript (plan
0060) sit on the same page but do not know about each other: playback gives
no cue which `step N` the arm is executing, and finding the video moment for
a turn means scrubbing by eye. #272 demoed the fix — the transcript as a
live rail: the active message highlights during playback and clicking a
message seeks the video — but its implementation targeted the pre-0060 page.

## Design

Two data attributes bridge Python (which knows the timeline) and one small
JS addition (which owns playback):

- The composite `<video>` gains `data-steps="0,1,2,…"` — the encoder's step
  timeline (sorted union, exactly the steps that produced composite frames,
  so video frame *k* shows step `steps[k]`) — and `data-fps="{fps:g}"`.
  Frame *k* plays at `t = k / fps`; both mappings are exact, no
  `control_hz` re-derivation in JS (the union timeline is not guaranteed
  contiguous, so arithmetic on step numbers would be wrong — index lookup
  is the only correct mapping).
- Each turn section with frame references gains `data-step="{N}"` (the same
  `turn.references[0].step` its `step N` header shows). Preamble turns and
  turns without references get no attribute and stay inert.

### Encoder: timeline joins the return value

`_encode_composite_mp4` returns `tuple[bytes, tuple[str, ...], tuple[int,
...]] | None` — bytes, surviving stream keys (plan 0063), and now the
**emitted-frame step timeline** (union steps minus skipped all-empty steps,
in yield order; built alongside the frames, so it is exactly index-aligned
with the encoded video). The monkeypatch seam and fakes in
`tests/test_html_view.py` widen to the 3-tuple.

### HTML (`_html.py`)

- `_render_trial_media` interpolates `data-steps`/`data-fps` into the video
  element (comma-joined ints; ~6 KB for a 1,200-step run — noise next to
  the MP4 payload). `data-fps` uses the same `{fps:g}` formatting as the
  ffmpeg argv.
- `_render_turn` adds `data-step="{N}"` to `<section class="turn">` when the
  turn has references.
- A **Follow** toggle button (`<button type="button" data-follow>Follow</button>`)
  joins the `run-media-head`, rendered only on the composite success path.

### JS (inside the existing per-`run-media` `forEach`)

All new behavior lives next to the plan-0063 `syncVideo` function and is a
no-op unless `block.querySelector('.video-panel video[data-steps]')` exists
(fallback/live pages unchanged):

- Parse `steps` once; let `turns` = `transcript.querySelectorAll
  ('section.turn[data-step]')` scoped to the enclosing `details.transcript`
  (multi-trial pages stay per-trial; a missing `transcript` renders the rail
  inert — single-file pages always have one).
- On `timeupdate` (and once per `toggle`-driven play): `index =
  min(floor(video.currentTime * fps), steps.length - 1)`; `step =
  steps[index]`; the active turn is the last turn whose `data-step <= step`
  (turns are in ascending step order — same source ordering as the
  timeline). Move the `active` class; when Follow is on, `scrollIntoView
  ({block: 'nearest'})`.
- Click on a turn's `.turn-step` header seeks: `video.currentTime = i /
  fps` where `i` is the first index with `steps[i] >= turn step` (last
  index when the turn's step is past the end); playback state is left
  as-is. Click wiring only attaches when the composite exists, so headers
  never look interactive on fallback pages.
- Follow button toggles an `active` class on itself; default off
  (auto-scroll is opt-in — #272's demo lesson).

### CSS

- `.turn.active { border-left: 3px solid var(--user); padding-left: 12px;
  margin-left: -15px; }` — highlight without layout shift (compensating
  negative margin).
- `.rail-seekable .turn-step { cursor: pointer; }` — the JS adds
  `rail-seekable` to the enclosing `details.transcript` when it wires the
  rail, so pointer affordance appears exactly when clicking works.
- `[data-follow]` reuses the `.camera-tab` pill styling (same border/active
  variables) so the head stays visually consistent.

## Tests

The rail itself is JS (uncovered by pytest, like the existing flipbook
script); Python emissions are what the gate covers:

`tests/test_video.py`:
- Composite return gains the timeline: identical-steps fixture asserts
  `steps == (0, 1)`-style exact tuples; the hold-last/union fixture asserts
  union order; the skip-step fixture asserts the skipped step is absent
  from the returned timeline (index alignment with emitted frames is the
  contract).
- Migrated wrapper tests re-pin the 3-tuple boundary.

`tests/test_html_view.py`:
- Eligible page: video carries `data-steps` matching the fake encoder's
  timeline and `data-fps`; `run-media-head` contains the Follow button;
  turn sections carry `data-step` matching their `step N` headers; preamble
  turn carries none.
- Fallback page: no `data-steps`, no Follow button, turns still carry
  `data-step` (attribute emission is video-independent).
- Fake composite encoders return 3-tuples; the throwing fake is unaffected.

## Docs & changelog

- CHANGELOG `[Unreleased]` → Added: transcript rail (live step highlight,
  click-to-seek, Follow) on composite-video pages, crediting PR #272 as the
  origin (plan 0065, issue #352).
- `docs/guide/cli.md` report section: one short paragraph after the
  run-video paragraph describing the rail.
- `src/inspect_robots/CLAUDE.md` `_html.py` row gains rail wording.
