# 0090: Side-by-side cameras in the `view` flipbook fallback

Status: proposed. Issue #608.

## Problem

`inspect-robots view` shows each trial's run media in one of two ways:

- **Composite MP4** (the default when ffmpeg is available and the page's video
  budget allows): one `<video>` with every camera side by side.
- **Flipbook fallback** (`--no-video`, a video budget overflow, a failed
  encode, or no ffmpeg): a tab per camera and one `<img>` that steps through
  that camera's embedded frames. Only the active tab's camera is visible.

Published transcripts are rendered with `--no-video --frames-budget 12` to keep
pages small, so readers see the top camera only, while the composite MP4 and
the published per-run videos show left wrist, top and right wrist together.

Separately, the flipbook `<img>` is written without a `src`; until the script
sets one, browsers draw a broken-image icon and the alt text.

## Change

### Camera order (shared by both modes)

A new helper `_camera_order(cameras)` stable-sorts camera names into three
groups: names whose lowercased tokens (split on `_`, `-` and whitespace)
include `left` first, names including `right` last, everything else in
between; ties keep the incoming order. The middle group therefore holds the
centre camera whatever it is called (`top`, `cam_high`, `overhead`), so ALOHA
style `cam_left_wrist`/`cam_high`/`cam_right_wrist` reads left, high, right.
Token matching keeps names like `bright` or `leftover` in the middle group.
It is applied at two call
sites in `_render_trial_media`:

- Composite: the existing sort (first appearance in the transcript, then name)
  produces the incoming order; `_camera_order` re-sorts those names and
  `ordered_streams` is rebuilt from them, so `.camera-order` follows.
- Flipbook: `fallback_cameras` becomes `_camera_order` over the composite's
  incoming order followed by flipbook-only cameras, so both modes agree even
  when a camera only has stored streams.

For the usual bimanual rig both modes show left wrist, top, right wrist. This
also changes the left-to-right layout of newly encoded composite MP4s.

### Markup (`_render_flipbook_media`)

- No tabs. Every camera with at least two distinct source steps (the existing
  two-frame floor, now counting distinct steps so a camera whose placeholders
  repeat one cached step gets no frozen panel) renders in one `.flipbook-grid`
  (`repeat(auto-fit, minmax(180px, 1fr))`, so it wraps on narrow screens),
  each as a `.camera-panel.flipbook-panel` with `data-camera-panel`, its
  `.flipbook-player` image followed by a `.flipbook-loading` "Loading…" span,
  and a `.flipbook-label` (`data-panel-label`) that starts as
  `<camera> · step <first>` and that the script keeps at the step that panel
  is actually showing, so a camera the frame budget cut short is visibly stale.
- One shared `.flipbook-controls` row under the grid: play/pause, one scrubber
  (`aria-label="run step"`) and a step label. Its server-side `max` is the
  number of distinct steps across the shown cameras minus one; the script
  overwrites it from the frames it actually finds.
- The reason chip (`video budget`, `no ffmpeg`) is unchanged.
- CSS: remove only `.camera-tabs`; keep `.camera-tab` and `.camera-tab.active`,
  which the composite's Follow button uses.

### Script (`_FLIPBOOK_SCRIPT`)

- Remove the tab code and per-camera state. Per block with flipbook panels:
  collect each camera's source frames (`img.frame[data-trial][data-camera]`),
  dedupe by step (cached frames can repeat) and sort by step. A panel whose
  camera has no source frames is hidden. The timeline is the sorted union of
  the remaining cameras' steps. At timeline position `i`, each camera shows its
  latest frame with step <= `timeline[i]`, or its first frame if none yet, and
  its label shows that frame's step.
- One `setInterval` (250 ms, as today) advances `i` for every camera and wraps
  to 0 after the last step, as today. Play/pause and the scrubber (which
  pauses) drive that one timer. The first frame is shown on load even when the
  transcript is collapsed; the timer only runs while it is expanded.
- The `details.transcript` toggle listener keeps calling `syncVideo()` (the
  composite MP4 play/pause) and additionally starts or stops the flipbook
  timer when the block has a flipbook.
- State is one session-storage entry per trial, `{step, paused}`, keyed
  `inspect-robots:flipbook-run:<path>:<trial>` (a new prefix, so it can never
  collide with an old per-camera key). On load the position is the largest
  timeline index whose step is at or below the saved step, or index 0 when the
  saved step is below the first timeline step (a live page that dropped its
  earliest frames) or missing. Storing the step, not the index, keeps live
  pages (which reload every 2 s and drop their earliest frames as the run
  grows) from drifting. Old per-camera and tab keys are ignored.
- The composite-MP4 code path (rail seeking, Follow) is untouched apart from
  using `_camera_order`.

### Loading state (CSS)

- `.flipbook-player:not([src])` is `display: none`, so no broken-image icon.
  The adjacent `.flipbook-loading` span is shown (a centred "Loading…" box
  with a 4:3 aspect ratio) only while its player has no `src`
  (`.flipbook-player:not([src]) + .flipbook-loading`); once the script sets a
  frame it disappears. Plain sibling selectors, no `:has()`.
- Without JavaScript the flipbook never gets frames, so the flipbook block
  carries a `flipbook-media` class and a `<noscript><style>` rule in the head
  hides it entirely, rather than showing "Loading…" panels and dead controls. The page's per-turn frame strips still show
  every embedded frame. The player keeps no server-side `src`, so data URLs
  are not duplicated against the frame budget.

## Tests

`tests/test_html_view.py`:

- The fallback has no `data-camera-tab` buttons; every eligible camera's panel
  is present in left/top/right order and none is `hidden`; each has a
  "Loading…" span and a `.flipbook-label`.
- `_camera_order` puts `left` tokens first and `right` tokens last
  (case-insensitive, whole tokens only), keeps the incoming order otherwise,
  and orders `cam_left_wrist`, `cam_high`, `cam_right_wrist` correctly.
- A composite whose transcript order is `top, right_wrist, left_wrist` hands
  the encoder `left_wrist, top, right_wrist` and captions it that way.
- Exactly one `.flipbook-controls` row per trial, scrubber
  `aria-label="run step"` and `max` = distinct-step count minus one.
- A camera whose placeholders repeat one step gets no panel.
- `_FLIPBOOK_SCRIPT` no longer mentions `flipbook-tab` and uses the per-trial
  key; the `<noscript>` rule is in the head.
- Existing degrade tests (budget, encode failure, below-floor camera, no
  panel) and the Follow button markup keep passing.
- Manual check in a headless browser on a real `--no-video` transcript: three
  cameras side by side, advancing together, "Loading…" before frames, and no
  visible flicker across a live page's 2 s reload.

## Docs and changelog

- `docs/guide/cli.md` (the "per-camera tabs" sentence near line 738 and the
  "per-camera flipbook" mention near line 761) and `docs/guide/live-view.md`
  (the camera player paragraph near line 26 and "per-camera flipbook" near
  line 31): describe the side-by-side flipbook and that the position and pause
  state survive reloads. No em dashes.
- `changelog.d/608.changed.md`.

## Out of scope

- Re-rendering already published transcripts.
- Any change to the composite MP4 encoder itself (only the camera order it is
  given changes, via `_camera_order`).
