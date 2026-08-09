# 0065 — Transcript rail: live turn highlight + click-to-seek on the composite run video

- **Status:** draft (R3 resolved)
- **Issue:** #352
- **Critique rounds:** R1: 3 substantive (turn steps are policy-authored and
  not guaranteed ascending — active-turn is now an order-independent argmax
  matching `_place_feedback`'s precedent, and "same source ordering as the
  timeline" was factually wrong; Follow scrolled every `timeupdate` — now
  gated on active-turn *change*, which also amortizes the search cost;
  the timeline construction is now pinned to append-at-yield + read-after-
  wrapper so a partial timeline can never escape) plus five folded nits
  (Follow reuses `class="camera-tab"`; every turn carries a transparent
  left border so activation never shifts layout and dividers stay uniform;
  seek lands mid-frame at `(i + 0.5)/fps`; initial active turn set at
  wire-up; skip-step wording in the intro) — all resolved below. R2:
  verified all R1 resolutions hold; 4 substantive (the argmax tie rule said
  first-in-document while the sorted-array recipe and the `_place_feedback`
  precedent both give last-in-document — corrected to last; the
  previous-position scan was direction-ambiguous while the looping video
  guarantees backward moves — now explicitly bidirectional, with an empty
  candidate set clearing the active class; two unlisted test re-pins — the
  composite-success `camera-tab`-absence assert and the same-step-feedback
  test's exact `<section class="turn">` split) plus three folded nits
  (fourth exact-tuple fixture named, `**Core:**` changelog prefix,
  data attributes placed after `preload="metadata"`) — all resolved below.
  R3: verified all R2 resolutions hold byte-for-byte (tie rule, loop
  premise, timeline construction, seam, full re-pin sweep, branch
  inventory, browser semantics); 1 substantive (the R2 replacement string
  `'data-camera-tab' not in document` is unconditionally false — the bare
  substring lives in the always-embedded flipbook JS; corrected to the
  attribute-with-value form `'data-camera-tab="'`) plus two folded nits
  (gutter appended after the `margin` shorthand; the literal `autoplay`
  banned from new JS/markup) — resolved below.
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
  timeline (sorted union minus skipped all-empty steps: exactly the steps
  that produced composite frames, so video frame *k* shows step `steps[k]`)
  — and `data-fps="{fps:g}"`.
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
**emitted-frame step timeline**. Construction is pinned: the timeline is a
list closed over by the `composite_frames` generator, appended **immediately
before each `yield`**, and read **only after `_temporary_mp4` returns
non-`None`** — the wrapper's `_encode_arrays` loop exhausts the generator on
every success path and every partial-consumption path returns `None`, so a
partial timeline can never escape, and index-alignment with the encoded
frames holds by construction. (Do **not** precompute the timeline in a
pre-pass: it would double-`_normalize` every frame and can diverge from the
encode's mid-stream error cutoff.) The monkeypatch seam and fakes in
`tests/test_html_view.py` widen to the 3-tuple.

### HTML (`_html.py`)

- `_render_trial_media` interpolates `data-steps`/`data-fps` into the video
  element (comma-joined ints; ~6 KB for a 1,200-step run — noise next to
  the MP4 payload), **after** `preload="metadata"` so the existing
  attribute-prefix assertion survives unchanged. `data-fps` uses the same
  `{fps:g}` formatting as the ffmpeg argv.
- `_render_turn` adds `data-step="{N}"` to `<section class="turn">` when the
  turn has references.
- A **Follow** toggle button (`<button type="button" class="camera-tab"
  data-follow>Follow</button>`) joins the `run-media-head`, rendered only on
  the composite success path. Reusing `class="camera-tab"` gives it the pill
  styling for free and is safe: the tab-wiring JS selects by
  `[data-camera-tab]`, which Follow does not carry.

### JS (inside the existing per-`run-media` `forEach`)

All new behavior lives next to the plan-0063 `syncVideo` function and is a
no-op unless `block.querySelector('.video-panel video[data-steps]')` exists
(fallback/live pages unchanged):

- Parse `steps` once; let `turns` = `transcript.querySelectorAll
  ('section.turn[data-step]')` scoped to the enclosing `details.transcript`
  (multi-trial pages stay per-trial; a missing `transcript` renders the rail
  inert — single-file pages always have one).
- On `timeupdate` (and once at wire-up, so a highlight exists before the
  first event): `index = min(floor(video.currentTime * fps),
  steps.length - 1)`; `step = steps[index]`. The active turn is computed
  **order-independently**: among turns with `data-step <= step`, take the
  one with the greatest step (**last**-in-document on ties — matching
  `_place_feedback`'s `max(candidates, key=(step, index))`, pinned by the
  same-step-tail test) — turn steps are policy-authored transcript labels
  with no monotonicity guarantee, and `_place_feedback` already refuses the
  ascending assumption; the rail matches that precedent. (Binary search is
  valid only over `steps`, which the encoder genuinely emits ascending.)
  In practice: parse `[step, node]` pairs once at wire-up and sort by
  `(step, document position)`; "last sorted entry with step <= current" is
  the argmax, and the per-event work is a pointer walked **forward or
  backward** from the previous position — the video carries `loop`, so
  `currentTime` wraps to 0 every playthrough, and click-to-seek moves
  backward; a forward-only scan would stick after the first loop. When no
  entry has `step <= current` (possible at wrap or wire-up when another
  camera stored earlier steps than the first stepped turn), the candidate
  set is empty and the active class is **cleared**.
- The `active` class moves — and Follow's `scrollIntoView({block:
  'nearest'})` fires — **only when the active node changes**, never on
  every `timeupdate`: unconditional scrolling at 4–66 Hz would fight the
  user's own scrolling permanently (the composite plays and loops whenever
  its transcript is open), and change-gating also makes the per-event cost
  O(1) amortized.
- Click on a turn's `.turn-step` header seeks: `video.currentTime =
  (i + 0.5) / fps` where `i` is the first index with `steps[i] >= turn
  step` (last index when the turn's step is past the end) — the half-frame
  offset keeps boundary rounding from displaying frame `i - 1`. Playback
  state is left as-is (a paused video still fires `timeupdate` on seek, so
  the highlight follows). Click wiring only attaches when the composite
  exists, so headers never look interactive on fallback pages.
- Follow button toggles an `active` class on itself; default off
  (auto-scroll is opt-in — #272's demo lesson).

### CSS

- Every turn carries the gutter so activation never shifts layout **and**
  the `.turn + .turn` divider spans one uniform width: `.turn {
  border-left: 3px solid transparent; padding-left: 12px; margin-left:
  -15px; }` (appended to the existing `.turn` rule, **after** its
  `margin` shorthand so the shorthand cannot reset `margin-left`) with
  `.turn.active { border-left-color: var(--user); }`.
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
  contract); the black-prefill/all-empty-drop fixture's exact 2-tuple
  assert widens too.
- Migrated wrapper tests re-pin the 3-tuple boundary.

`tests/test_html_view.py`:
- Eligible page: video carries `data-steps` matching the fake encoder's
  timeline and `data-fps`; `run-media-head` contains the Follow button;
  turn sections carry `data-step` matching their `step N` headers; preamble
  turn carries none.
- Fallback page: no `data-steps`, no Follow button, turns still carry
  `data-step` (attribute emission is video-independent).
- Fake composite encoders return 3-tuples; the throwing fake is unaffected.
- Two existing assertions re-pin: the composite-success test's
  `'class="camera-tab' not in document` becomes
  `'data-camera-tab="' not in document` — the trailing quote is
  load-bearing: the bare substring `data-camera-tab` appears in the
  always-embedded flipbook JS (`querySelectorAll('[data-camera-tab]')`) on
  every page, so only the attribute-with-value form that
  `_render_flipbook_media` alone emits is assertable (the same
  quote-boundary reasoning that made the original `class="camera-tab` pin
  sound) — plus a Follow-present assert; and the same-step-feedback test's
  exact `document.split('<section class="turn">')` breaks once turns carry
  `data-step` — split on the unclosed prefix `'<section class="turn'`
  (or a regex) instead.
- New JS/markup must not contain the literal string `autoplay` (a
  document-wide absence pin exists), and nothing new may carry
  `data-camera-panel` or a bare `hidden` attribute inside run-media (regex
  pins).

## Docs & changelog

- CHANGELOG `[Unreleased]` → Added, with the conventional bold scope prefix
  (`**Core:**`): transcript rail (live step highlight, click-to-seek,
  Follow) on composite-video pages, crediting PR #272 as the origin
  (plan 0065, issue #352).
- `docs/guide/cli.md` report section: one short paragraph after the
  run-video paragraph describing the rail.
- `src/inspect_robots/CLAUDE.md` `_html.py` row gains rail wording.
