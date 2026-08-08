# 0063 — Composite side-by-side run video; "Raw transcript" rename

- **Status:** draft (R4 resolved)
- **Issue:** #347
- **Critique rounds:** R1: 4 substantive (unscripted `autoplay loop` composite
  reversed 0060's pinned collapsed-trials-cost-nothing playback design; three
  missed doc sites still describing per-camera MP4s; coverage-gate gaps —
  unnamed skip-step test, unspecified probe-time read failure, sticky-budget
  tests need multi-trial restructuring not re-pinning, and a
  rendered-camera-without-stream guard would be an uncoverable branch;
  `.system-message` cited as a generic-border keeper when it already zeroes
  the border, and the rename touches two selectors, not one) — all resolved
  below. R2: verified all R1 resolutions hold against the code; 3 substantive
  (stale `autoplay` in the design intro contradicted the R1 playback
  resolution; `_encode_camera_mp4` orphaned with its five wrapper-failure
  tests uncovered — now explicitly deleted with the temp-file wrapper
  extracted/shared and tests migrated, plus the full monkeypatch-family
  retarget list; fourth doc site — the CLAUDE.md `_video.py` row's
  "report-only wrapper" sentence) — all resolved below. R3: verified all R2
  resolutions hold; 4 substantive (an all-empty stream dropped by the probe
  would still be named in the caption — encoder now returns surviving keys
  and the caption is survivor-driven; `activate(undefined)` on toggle would
  hide/pause any panel carrying `data-camera-panel`, so the composite
  panel's lack of that attribute is now pinned as load-bearing and the
  play/pause hook's single home specified; the call-time import seam the
  monkeypatch family relies on is now pinned; `.camera-order` had no CSS —
  muted small-caption rule named) plus four foldable nits (test-count
  wording, budget-test naming, `_VideoBudget` docstring, cli.md ~490
  rename sentence) — all resolved below. R4: verified all R3 resolutions
  hold; 3 substantive (survivor keys are `_safe` prefixes, not display
  names — the caption now maps through `display_names.get(key, key)` and a
  caption test uses a space-bearing name; the load-bearing absence of
  `data-camera-panel` had no named test — now asserted; the
  "wrapper(producer)" composition framing contradicted the tuple return —
  probe is now specified as a pre-wrapper branch and the migrated tests
  assert the tuple boundary) plus five foldable nits (dedup of migrated
  failure tests, fake-encoder tuple shape, cli.md line range and tail
  sentence, ordering-fixture requirement, storage-write exactness) — all
  resolved below.

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
per-turn frame rows. One `<video controls muted loop preload="metadata">`
panel (playback script-driven — see the HTML section) replaces the camera
tabs; a caption row under the "Run video" head lists the camera names in
composite order (`left · top · right`) so the video and the frame rows read
identically.

Camera order: first-appearance order of display cameras in
`frame_ctx.rendered` (exactly what `_render_turn_frames` produced), then any
stored streams never rendered in a turn, in their existing sorted-key order.
Rendered names are display names while stream keys are `_safe` filename
prefixes; the ordering is implemented as a *sort of the streams dict* by
`(first-appearance index of the stream's display name, stream key)` with
never-rendered streams getting an index past the end — an index lookup with a
default, **not** a "rendered camera missing its stream" guard branch: within
one `render_html` pass a rendered frame's `.npy` is the very file the stream
glob matches, so such a guard could never be covered and would fail the 100%
coverage gate.

### Encoder: `_encode_composite_mp4` in `_video.py`

The existing pipe design is kept (rawvideo → ffmpeg stdin, stderr to temp
file, libx264 pinned). To share it between the per-camera path (still used by
the public `video` CLI via `encode_stream`) and the new composite path, the
core loop is refactored so frame *production* is separated from *piping*:

- `_encode_core(frames, ...)` keeps its exact public behavior (CLI contract
  unchanged, `StreamResult` unchanged).
- New `_encode_composite_mp4(ordered_streams, fps, ffmpeg) ->
  tuple[bytes, tuple[str, ...]] | None` where `ordered_streams:
  Sequence[tuple[str, Sequence[tuple[int, Path]]]]`. The returned tuple
  carries the MP4 bytes **and the surviving stream keys in composite
  order** — an all-empty (warm-up) stream is dropped by the probe, and the
  caption must not name a camera that contributes nothing to the video, so
  the caption is built from the survivors, not the input. Survivor keys are
  `_safe` filename prefixes (not identity for real names — `_safe` swaps
  unsafe chars and appends a crc32 hash): the caller maps each survivor
  through the same `display_names.get(key, key)` lookup the tabs use today
  before joining the caption. Any failure degrades to `None` (caller falls
  back to flipbook tabs).
- The call-time `from inspect_robots._video import _encode_composite_mp4`
  **inside** `_render_trial_media` is retained (as with `_encode_camera_mp4`
  today): `inspect_robots._video._encode_composite_mp4` stays the patchable
  seam every retargeted monkeypatch test relies on; a top-of-module import
  would bind a direct reference in `_html` and defeat them all.
- **`_encode_camera_mp4` is deleted** — its only caller is the per-camera
  encode path this plan removes, and no fallback encodes per-camera MP4s.
  Its temp-file wrapper (mkstemp → encode → `read_bytes`, degrading every
  failure to `None` without leaking the temp file, `bytes | None` contract)
  is **extracted and shared**. `_encode_composite_mp4` first probes the
  streams (survivors + composite frame producer) — a probe `_FrameError` →
  `None` is its **own pre-wrapper branch**, covered by the named probe
  test — then calls wrapper(producer) for the bytes and returns
  `(bytes, survivors)`. The wrapper's four failure branches stay covered in
  one place: the four existing `_encode_camera_mp4` test functions covering
  five cases (`tests/test_video.py` ~464-521 — success, encode+launch
  failure in one test, mkstemp `OSError`, `read_bytes` `OSError`) migrate to
  `_encode_composite_mp4`, now asserting the tuple/`None` at its boundary
  (the success assertion becomes `(b"browser mp4", (<survivors>,))`).

Composite frame construction:

- **Timeline** = sorted union of step numbers across all streams. At each
  step, a camera that has a frame there contributes it; a camera that does
  not **holds its last frame**. Before a camera's first usable frame it
  contributes black (its own probed size). Peak memory grows from one frame
  to one held frame per camera plus the composite — three 224×224 frames in
  practice; still trivial.
- **Probe**: like `_encode_core`'s pre-spawn probe, scan each stream forward
  past empty frames to its first usable one to learn its shape. A stream
  whose frames are all *empty* (expected first-party warm-up data) is dropped
  from the composite. A `_FrameError` during the probe (truncated/corrupt
  file, wrong dtype — the same cases `_encode_core` errors on pre-spawn)
  **fails the whole composite** → `None` → flipbook fallback: corruption is
  a loud degrade, never a silently missing camera. If no stream has a usable
  frame, return `None`.
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

- Success path: one panel, no `camera-tabs`:

  ```html
  <div class="run-media" data-trial="...">
    <div class="run-media-head">Run video</div>
    <div class="camera-order">left · top · right</div>
    <!-- .camera-order gets a muted small-caption rule in _STYLES, matching
         the page's other captions:
         .camera-order { color: var(--muted); font-size: 12px;
                         margin: 4px 0 8px; } -->
    <div class="camera-panel video-panel"><video controls muted loop
         preload="metadata" src="data:video/mp4;base64,..."></video></div>
  </div>
  ```

- **Playback cost stays governed by transcript visibility** (0060's pinned
  R2: "a collapsed trial costs nothing"). The composite `<video>` carries
  **no `autoplay`** and `preload="metadata"`. Playback is wired inside the
  existing per-`run-media` `forEach` as **one** sync function: it queries
  `block.querySelector('.video-panel video')` and, from the enclosing
  `details.transcript` state, expanded → `void video.play().catch(() => {})`,
  collapsed → `pause()`; it is registered on the **same** `toggle`
  registration point the tab code uses today (one listener home) and called
  once at load. Single-transcript pages (which `render_html` opens by
  default, `open_transcript=transcript_count == 1`) therefore start playing
  exactly like today's first tab; multi-trial pages decode nothing until a
  trial is expanded.
- **The composite panel carries no `data-camera-panel` and no `hidden`
  handling — deliberately and load-bearing.** On a composite block the
  script's `tabs`/`panels` queries are empty, `active` is `undefined`, and
  `activate(undefined)` would hide and pause every `[data-camera-panel]`
  panel it found; with none present it is a no-op aside from a harmless
  sessionStorage tab-key write (the load-time read is membership-guarded).
  The tab/flipbook JS is otherwise retained unchanged for fallback pages.
- **Budget** (`_VideoBudget`, 30 MB base64 default, `--no-video` opt-out)
  is unchanged in mechanism: the single composite payload is charged against
  the limit; on overflow the budget goes sticky-truncated and this trial (and
  later trials) render the current per-camera flipbook tabs with the
  `video budget` chip. Because there is now one encode per trial instead of
  one per camera, the sticky "later cameras skip their encode" prose moves
  up a level — later *trials* skip the composite encode — in both the
  `_render_trial_media` comment and the `_VideoBudget` docstring (which says
  "camera" twice today).
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
  above the dropdown. The `details` elements that actually depend on the
  generic rule — `details.transcript`, `details.wire`, `details.wire-call` —
  keep it (`.system-message` already zeroes its border and is unaffected).
  The rename touches both existing selectors: `.llm-pov` and
  `.llm-pov .message`.
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
- Pre-first-frame black fill and all-empty stream dropped; every stream
  empty → `None`.
- Probe-time `_FrameError` (truncated first frame) → `None` (loud degrade,
  not a silently dropped camera).
- Skip-step branch: two streams each holding an *empty* frame at the same
  union step → that step emits no composite frame (no duplicate).
- Height padding with mixed resolutions; width is the sum.
- Mid-stream shape change → `None`.
- Wrapper failures (encode, launch, mkstemp, `read_bytes`) → `None`: these
  are the migrated `_encode_camera_mp4` tests, not new ones.

`tests/test_html_view.py`:
- Video-eligible page renders exactly one `<video>` and no `camera-tab`
  buttons; caption row lists cameras in turn order — the fixture's turn
  order must differ from sorted-key order (e.g. render `top` before
  `alpha`) so first-appearance ordering is actually pinned; at least one
  camera name is one `_safe` rewrites (a space — precedent:
  `test_space_in_camera_name_reconstructs_and_embeds`) so the caption is
  pinned to display names, not hash-suffixed keys; the video has **no
  `autoplay`** and `preload="metadata"` (playback is script-driven from the
  transcript toggle — pins 0060's collapsed-trials-cost-nothing rule); and
  the run-media block contains **no `data-camera-panel` attribute and no
  `hidden` panel** — pinning the load-bearing absence that keeps
  `activate(undefined)` away from the composite video.
- All-empty extra stream on disk → composite succeeds and the caption
  **omits** the dropped camera (survivor-driven caption).
- Sticky budget: the existing within-trial sticky tests
  (`test_mp4_budget_is_sticky_first_wins_and_skips_later_encodes`, the
  panel-less sticky half of the single-frame-camera test) are **restructured
  to multi-trial fixtures** — with one encode per trial, stickiness is only
  observable across trials: trial 1's composite overflows → truncates →
  trial 2 skips its encode (`encoder_calls` count pins the skip) and renders
  flipbook tabs with the `video budget` chip.
- Encode failure → flipbook tabs (re-pin
  `test_mp4_budget_and_failures_degrade_to_flipbook`; its budget half stays
  valid single-trial).
- Stream never rendered in a turn still appears in the composite caption
  order (replaces the `data-camera-tab="unseen_camera"` assertion). The
  whole `_encode_camera_mp4` monkeypatch family retargets
  `_encode_composite_mp4`:
  `test_mp4_tier_includes_unreferenced_camera_and_only_active_autoplays`
  (replaced wholesale — its two-payload/one-autoplay assertions are
  composite-order + no-autoplay assertions now),
  `test_suppressed_video_tiers_never_call_encoder`, and
  `test_video_budget_without_flipbook_source_omits_camera_panel`. Fake
  encoders now return `(b"...", ("cam", ...))` tuples (the throwing fake in
  the suppressed-tiers test is unaffected).
- `Raw transcript` label and `raw-transcript` class assertions replace the
  `llm-pov` ones (including the strip-helper regex at test line ~1142).

## Docs & changelog

- CHANGELOG `[Unreleased]` → Changed: composite run video (one shared
  playhead), LLM POV renamed to Raw transcript, step dividers, with
  plan/issue links.
- `docs/guide/cli.md`: the run-video paragraph (~494-500) rewritten for the
  single composite video, the budget paragraph (~515-521) — the unit of
  charge/degrade becomes the trial composite, not a camera, and its tail
  "re-renders them with MP4s" becomes "with the composite MP4" — **and**
  the "LLM POV" sentence in the first report paragraph (~490).
- `docs/guide/live-view.md`: the LLM POV sentence (~19) renamed, **and** the
  post-run upgrade paragraph (~28-31) that says a view pass "can replace
  each camera's flipbook with an embedded MP4" rewritten for the composite.
- `src/inspect_robots/CLAUDE.md`: the `_html.py` row ("budgeted
  completed-page MP4 embedding", "raw LLM POV dropdowns"), the `cli.py`
  row ("eligible embedded MP4s"), **and the `_video.py` row** (the
  "report-only wrapper" sentence — the report wrapper is now the composite
  encoder, and the report path degrades loudly as a whole video, not
  per-stream) updated to composite-video and Raw-transcript wording.
