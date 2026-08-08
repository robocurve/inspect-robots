# 0063 — Composite side-by-side run video; "Raw transcript" rename

- **Status:** draft (R0)
- **Issue:** #347

## Problem

The report's "Run video" section (plan 0060) renders one MP4 per camera behind
`camera-tab` buttons: watching all three viewpoints of a moment means clicking
between tabs and losing the playhead each time (each `<video>` has its own).
The per-turn frame rows already show every camera side by side in one glance;
the run video should match. Native `<video>` gives a shared playhead for free
if the cameras are composed into **one** MP4 — no JS sync, no tabs.

Separately, the raw wire dropdown is labeled "LLM POV", which reads as jargon
in a report shared outside the team. It should say **"Raw transcript"**.

## Design

### One composite MP4 per trial

`_render_trial_media` stops encoding one MP4 per camera. When the page is
video-eligible (completed log, ffmpeg present, budget not yet truncated), it
encodes a **single MP4 whose frames are the per-step horizontal concatenation
of every stored camera stream**, in the same left-to-right order as the
per-turn frame rows. One `<video controls muted loop autoplay>` panel replaces
the camera tabs; a caption row under the "Run video" head lists the camera
names in composite order (`left · top · right`) so the video and the frame
rows read identically.

Camera order: first-appearance order of display cameras in
`frame_ctx.rendered` (exactly what `_render_turn_frames` produced), then any
stored streams never rendered in a turn, in their existing sorted-key order.

### Encoder: `_encode_composite_mp4` in `_video.py`

The existing pipe design is kept (rawvideo → ffmpeg stdin, stderr to temp
file, libx264 pinned). To share it between the per-camera path (still used by
the public `video` CLI via `encode_stream`) and the new composite path, the
core loop is refactored so frame *production* is separated from *piping*:

- `_encode_core(frames, ...)` keeps its exact public behavior (CLI contract
  unchanged, `StreamResult` unchanged).
- New `_encode_composite_mp4(ordered_streams, fps, ffmpeg) -> bytes | None`
  where `ordered_streams: Sequence[tuple[str, Sequence[tuple[int, Path]]]]`.
  Any failure degrades to `None` (caller falls back to flipbook tabs), same
  posture as `_encode_camera_mp4`.

Composite frame construction:

- **Timeline** = sorted union of step numbers across all streams. At each
  step, a camera that has a frame there contributes it; a camera that does
  not **holds its last frame**. Before a camera's first usable frame it
  contributes black (its own probed size). Peak memory grows from one frame
  to one held frame per camera plus the composite — three 224×224 frames in
  practice; still trivial.
- **Probe**: like `_encode_core`'s pre-spawn probe, scan each stream forward
  past empty frames to its first usable one to learn its shape. A stream
  with no usable frames is dropped from the composite. If *no* stream has a
  usable frame, return `None`.
- **Height alignment**: cameras may differ in resolution; each frame is
  padded with black rows (bottom pad) to the max probed height, then
  `np.hstack`. Width = sum of camera widths; the existing ffmpeg `pad` filter
  still handles odd totals.
- **Mid-stream shape change** in any camera is a `_FrameError`, same as
  today: the composite fails as a whole → `None` → flipbook fallback.
- **Empty frames mid-stream** (`_normalize` → `None`): treated as "no frame
  at this step" — hold last. If a step ends up with no camera contributing a
  *new* frame (only possible when every stream's entry at that step is
  empty), the step is skipped rather than emitting a duplicate composite.

### HTML changes (`_html.py`)

- Success path: one panel, no `camera-tabs`, no per-panel JS dependence:

  ```html
  <div class="run-media" data-trial="...">
    <div class="run-media-head">Run video</div>
    <div class="camera-order">left · top · right</div>
    <div class="camera-panel video-panel"><video controls muted loop autoplay
         src="data:video/mp4;base64,..."></video></div>
  </div>
  ```

- **Budget** (`_VideoBudget`, 30 MB base64 default, `--no-video` opt-out)
  is unchanged in mechanism: the single composite payload is charged against
  the limit; on overflow the budget goes sticky-truncated and this trial (and
  later trials) render the current per-camera flipbook tabs with the
  `video budget` chip. Because there is now one encode per trial instead of
  one per camera, the sticky "later cameras skip their encode" comment moves
  up a level: later *trials* skip the composite encode.
- **Fallback** (no ffmpeg, encode failure, budget, live/serve pages): the
  existing per-camera flipbook tabs render exactly as today — the tab/JS
  machinery is retained for these paths.
- **Cameras with rendered frames but no stored stream** cannot join the
  composite; when the composite succeeds they are dropped from the run-media
  block (their frames still appear in every turn row). When the composite is
  not produced, today's behavior (flipbook panels for them) is unchanged.
- `_warn_video_degrade` reasons gain `encode failed for <trial> composite`.

### "LLM POV" → "Raw transcript"

- `_render_pov`: summary text becomes `Raw transcript`; the CSS class
  `llm-pov` becomes `raw-transcript` (self-contained pages, no external
  consumers). `_STYLES` selector updated alongside.
- Docs (`docs/guide/cli.md`, `docs/guide/live-view.md`) and
  `src/inspect_robots/CLAUDE.md` module table updated to the new name.

### Divider demarcates steps, not the raw dropdown

Today the generic `details { border-top: 1px solid var(--line); ... }` rule
paints a horizontal line **above** the raw dropdown inside every turn, which
reads as a divider in the middle of a step. The divider belongs **between
steps**:

- `.raw-transcript` overrides the inherited rule (`border-top: 0;
  padding-top: 0;` with its existing margin retained), so no line renders
  above the dropdown. Other `details` uses (e.g. `.system-message`) keep the
  generic rule.
- `.turn + .turn { border-top: 1px solid var(--line); padding-top: 18px; }`
  draws one line between consecutive turns — after each turn's raw dropdown,
  before the next `step N` header — and none before the first turn or after
  the last (adjacent-sibling, so non-turn neighbors are unaffected).

## Tests

`tests/test_video.py`:
- Composite of two streams with identical steps → frames hstacked in given
  order (assert piped bytes via the existing fake-ffmpeg capture pattern).
- Step-union hold-last: stream A has steps {0,1,2}, B has {0,2} → at step 1,
  B's step-0 frame is repeated.
- Pre-first-frame black fill and no-usable-frames stream dropped; all-streams
  unusable → `None`.
- Height padding with mixed resolutions; width is the sum.
- Mid-stream shape change → `None`.
- Encoder failure / launch failure → `None` (mirrors existing
  `_encode_camera_mp4` failure tests).

`tests/test_html_view.py`:
- Video-eligible page renders exactly one `<video>` and no `camera-tab`
  buttons; caption row lists cameras in turn order.
- Budget overflow → sticky truncation → flipbook tabs with `video budget`
  chip (existing tests re-pinned to composite sizes).
- Encode failure → flipbook tabs (re-pin existing test).
- Stream never rendered in a turn still appears in the composite caption
  order (replaces the `data-camera-tab="unseen_camera"` assertion).
- `Raw transcript` label and `raw-transcript` class assertions replace the
  `llm-pov` ones (including the strip-helper regex at test line ~1142).

## Docs & changelog

- CHANGELOG `[Unreleased]` → Changed: composite run video (one shared
  playhead), LLM POV renamed to Raw transcript, with plan/issue links.
- `docs/guide/cli.md` run-video paragraph rewritten for the single composite
  video; `docs/guide/live-view.md` dropdown sentence renamed.
