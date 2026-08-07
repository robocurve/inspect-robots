# Report overhaul: turn-oriented transcript, LLM POV dropdown, and run video

> Status: PLANNED — issue #337 items 2-6 and 8 (rig-1 shakedown, 2026-08-07).
> Builds on plans 0055/0058. Closes #337 when merged.

## Problem

Reading a real rig report today (30-run shakedown, 2026-08-07) surfaced five
readability failures and one status bug:

1. A running log's scene card shows a "completed" badge mid-run (the
   in-progress slot's `"success"`-so-far status maps straight through
   `_STATUS_DISPLAY`).
2. Step numbers are glued to every image caption (`camera 'top_cam' (step
   38):`) instead of heading the turn.
3. Raw model wire text dominates: `state[joint_pos]: ...` dumps, verbatim
   `move_joints({"targets": {...}, "note": ...})` calls.
4. Operator feedback is duplicated in a standalone section at the top,
   stripped of the visual context it had inline.
5. There is no way to *watch* the run: the most intuitive artifact for an
   eval log is a video of what the robot did.

## Design

### 1. Turn-oriented transcript

The chat transcript is regrouped into **turns**. A turn starts at each
`user`-role message and collects the following `assistant` and `tool`
messages until the next user message (leading non-user messages form a
headerless preamble turn). Per turn, the default (human) layer renders, in
order:

- **`step N` mini-header** — N from the turn's frame references (the label
  step; present even when a frame lost the budget, since the label text
  remains). A turn with no frame reference gets a header from the
  `operator feedback (step N)` line when present, else no header.
- **frames** — captions reduced to the camera name (the step moved to the
  header).
- **operator/voice feedback lines** — the `operator feedback (step N): ...`
  text lines from the user message, rendered as a distinct chip/row inline
  (this replaces the removed top-level section as the only place feedback
  appears).
- **agent note** — already extracted by `_agent_notes` (note / done-summary /
  give-up-reason), promoted to the turn's headline text.
- **pretty tool call** — tool name plus arguments rendered as labeled chips:
  `targets` dicts become per-joint `name value` chips (values shown with 3
  decimals); scalar arguments become `key value` chips; `note` is excluded
  (it is the headline). Malformed/non-JSON arguments fall back to today's
  raw rendering.
- **LLM POV dropdown** — one collapsed native `<details class="llm-pov">`
  per turn holding the raw exchange verbatim: the user message's full text
  parts (state dumps included), the assistant's raw content and verbatim
  `name({...})` call text, and the raw `tool`-role result contents. No JS.
  Everything escaped exactly once at interpolation, as today.

The state dumps and raw calls therefore leave the default view entirely
(they live only inside the POV dropdown); nothing is dropped from the page.

Non-chat transcripts (the `_is_chat_transcript` false path) keep today's
JSON fallback rendering unchanged.

### 2. Scene badge fix for running logs

For a `status == "started"` log, a scene whose trial metadata's **last slot
carries the live marker** (`{"live": {...}}`, written by `LiveLogSink` on
every in-progress snapshot) renders its card badge as `running` (amber)
instead of mapping the slot's `"success"`-so-far through to `completed`.
Scenes without the marker in a started log are genuinely finished and keep
their real badge. Completed logs are untouched.

### 3. Remove the top-level Operator feedback section

Delete the `feedback_block` from `_scene_section`. The log schema and data
are untouched; the inline turn rendering (§1) is the single place feedback
appears.

### 4. Run video

Each trial's transcript section is headed by a **video block** (for the
common single-scene rig run this is the top of the report):

- **Flipbook tier (always, zero new bytes):** frame `<img>` tags gain
  `data-camera`/`data-step` attributes. A new inline `_FLIPBOOK_SCRIPT`
  (pattern: the existing `_FRAME_CLICK_SCRIPT`) scans the DOM per trial for
  embedded frames, groups by camera, and drives a player `<img>` by
  assigning `src` from the *existing* elements — the data URLs are never
  duplicated in the HTML, so page weight is unchanged. Controls:
  play/pause, camera tabs, a range-input scrubber labeled with the step
  number; autoplay on load, looping, ~4 fps. With fewer than 2 frames for
  every camera the block renders nothing.
- **MP4 tier (completed pages, ffmpeg present):** when
  `shutil.which("ffmpeg")` succeeds, the log status is not `"started"`, and
  the frames side-car directory resolves, the renderer stitches **all
  control steps** per (trial, camera) through the existing `_video.py`
  pipeline machinery (refactored to expose a reusable
  `encode_camera_mp4(...) -> bytes | None` used by both the `video` command
  and the renderer; same stderr-tempfile and per-stream failure-isolation
  discipline, plan 0016) and embeds `<video controls muted loop autoplay>`
  with a base64 `data:video/mp4` URL, one per camera tab, replacing the
  flipbook for that trial. Playback rate is real-time (frame rate =
  `control_hz` from the log's `embodiment_info`, floored to 1, capped 60).
- **Budgeting:** embedded MP4s are charged to a dedicated
  `_VIDEO_BUDGET_BYTES = 30_000_000` per page (not the frame budget — the
  video replaces the need for more stills, and H.264 across near-identical
  robot frames is far denser than PNG). Over budget → that camera degrades
  to the flipbook. `--no-video` on `view` skips the MP4 tier entirely
  (flipbook remains); `--no-frames` (no `_FrameContext`) disables both
  tiers.
- **Live pages always use the flipbook** — never invoke ffmpeg on the 2s
  tick. Because live pages already embed newest-first frames (plan 0058),
  the flipbook doubles as a turn-aligned scrubber of what the model saw.
- ffmpeg absent or any encode failure → silent per-camera degrade to the
  flipbook (stderr warning once per render pass, matching `video`'s
  failure-isolation posture).

### 5. Explicitly out of scope

- No GIFs (larger than the stills, 256 colors).
- No turn markers on the MP4 scrubber (native `<video>` has no tick API;
  revisit with a custom scrubber only if operators ask).
- No `inspect` (terminal) rendering changes.
- No summarize/_video CLI behavior changes beyond the shared encoder
  refactor.

## Testing (100% coverage, no ffmpeg or hardware in CI)

- **Turn grouping:** user/assistant/tool sequences, leading assistant
  preamble, missing roles, non-dict messages, non-chat fallback untouched
  (differential assertion against a golden snippet).
- **Headers/captions:** step from frame label; from operator-feedback line
  when frames absent; no header otherwise; captions are camera-name-only.
- **Pretty calls:** targets → joint chips (3-decimal formatting), scalars,
  note exclusion, malformed JSON falls back raw; done/give_up headline keys.
- **LLM POV:** state dumps and raw call text present inside `<details>`,
  absent outside; single-escape (adversarial `<script>` in state text and
  note text); tool-role results inside.
- **Badge:** started log with live marker → running badge on that scene
  only; finished scene in the same log keeps completed; completed log
  unchanged.
- **Operator section:** gone; inline feedback rows present; data untouched
  in the log JSON.
- **Flipbook:** data attributes on frame imgs; script tag present exactly
  once; no data-URL duplication (page contains each URL once — asserted by
  count); fewer-than-2-frames renders no block.
- **MP4 tier:** ffmpeg resolved via `shutil.which` monkeypatch; a fake
  encoder returning known bytes → `<video>` with data URL, per-camera tabs;
  encoder returning None → flipbook degrade; video budget exceeded →
  degrade; `--no-video`; started log never calls the encoder; real ffmpeg
  invocation stays covered by the existing `video`-command tests through
  the shared helper (its subprocess seam already has fakes).
- **CLI:** `--no-video` parsing and forwarding; serve tick unaffected
  (cadence tests untouched).
- Docs: `docs/guide/live-view.md` and the viewer section of
  `docs/guide/cli.md` (or nearest existing viewer doc) updated; CHANGELOG
  entry (`[plan 0059](plans/0059-report-turns-and-video.md)`,
  [#337](https://github.com/robocurve/inspect-robots/issues/337), Closes);
  module map rows (`_html.py`, `_video.py`, `cli.py`).

## Implementation order

1. Turn grouping + headers/captions + inline feedback + LLM POV (pure
   `_html.py`), with the badge fix and feedback-section removal.
2. Flipbook (data attributes + script + player block).
3. `_video.py` encoder refactor + MP4 tier + `--no-video` + budget.
4. Docs, CHANGELOG, module map.
