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
`user`-role message **whose content is list-shaped** (an observation
message); string-content user messages (agent nudges like "Respond with
exactly one tool call.", capx execution reports, take_pic frame deliveries)
render as ordinary visible messages *inside* the current turn and never
start one (R1: capx has no tool calls at all — its assistant prose IS the
content — and nudges would otherwise open junk turns). The **preamble
turn** is everything before the first list-shaped user message — including
string-content user messages like capx's "Goal: ..." opener (R2) — and is
always headerless. Per turn, the default (human) layer renders, in order:

- **`step N` mini-header** — N is the turn's **first** frame reference's
  step (R2: pinned for multi-step turns; the label step is present even
  when a frame lost the budget, since the label text remains — on live
  pages too). Turns without frame references are headerless — a
  fallback-matched feedback message never mints a header, and the preamble
  is always headerless (R2). No prompt-text parsing anywhere (R1).
- **frames** — captions reduced to the camera name (the step moved to the
  header). **Identity contract** (R1, load-bearing for the 0058 live
  cache): turns group the original message dicts and reference the
  original parts lists — never copied, filtered, or rebuilt — and frames
  render from the turn's `_FrameReference`s via `_frame_image` in document
  order, preserving budget-charge order on completed pages and the
  `reference.parts is parts` / `(trial_prefix, camera, step)` correlation
  on live pages.
- **operator/voice feedback chips** — sourced from the log's structured
  `scene.operator_messages[trial]` (R1: the prompt-text lines only exist
  for the agent policy and only for *delivered* messages — `/stop` notes
  and post-final-inference messages never reach a transcript, and non-chat
  policies have none). Each message is matched to the latest turn whose
  frame step is ≤ its `t`; ties across turns sharing a step resolve to the
  latest in document order. **A chat transcript with zero frame-step turns
  (capx: its camera labels carry no `(step N)`) routes the trial's
  messages to the residual block instead** — first-turn dumping would
  destroy all temporal placement (R2). A message whose `t` precedes every
  frame-step turn (frames first revealed late via take_pic) likewise goes
  to the residual (R3). The chip shows the text and its
  source label. The raw delivered lines inside the observation blob stay
  verbatim in the POV dropdown.
- **assistant text content** — stays visible in the default layer (R1:
  hiding it empties every capx turn and loses agent deliberation prose).
- **agent note** — already extracted by `_agent_notes` (note / done-summary
  / give-up-reason), promoted to the turn's headline text.
- **pretty tool calls** — one pretty block *per call* (assistant messages
  can carry several, e.g. motion + take_pic): tool name plus arguments as
  labeled chips — any dict-of-numbers argument (`targets`, `deltas`)
  becomes per-key `name value` chips at 4 decimals (R1: matches the state
  lines the model saw; 3 loses sub-mm cartesian deltas); scalars become
  `key value` chips; arrays (e.g. take_pic `cameras`) join as a comma
  chip; `note` is excluded (it is the headline) and `hindsight` renders
  note-style, never as a chip (R1: it is a required prose paragraph on
  done/give_up). Malformed/non-JSON arguments fall back to today's raw
  rendering.
- **LLM POV dropdown** — one collapsed native `<details class="llm-pov">`
  per turn holding the raw exchange verbatim: the user message's full text
  parts (state dumps and delivered feedback lines included), the verbatim
  `name({...})` call text, and the raw `tool`-role result contents. No JS.
  Everything escaped exactly once at interpolation, as today.

The state dumps and raw call text therefore leave the default view
entirely (they live only inside the POV dropdown); nothing is dropped from
the page.

Non-chat transcripts (the `_is_chat_transcript` false path) keep today's
JSON fallback rendering unchanged.

### 2. Scene badge fix for running logs

For a `status == "started"` log, a scene whose trial metadata's **last slot
carries the live marker** (`{"live": {...}}`, written by `LiveLogSink` on
every in-progress snapshot) renders its card badge as `running` (amber)
instead of mapping the slot's `"success"`-so-far through to `completed`.
Scenes without the marker in a started log are genuinely finished and keep
their real badge. Completed logs are untouched.

### 3. Demote the top-level Operator feedback section to a residual

The prominent `feedback_block` leaves `_scene_section`'s top. Because the
structured `operator_messages` now drive the inline chips (§1), the block
survives only as a **residual**: it renders (in its current list form,
below the transcripts and above the wire details rather than at the top;
R2) exactly the messages that could not be placed inline — i.e. trials with no chat transcript (non-chat
policies, transcript capture failures). When every message was placed
inline, the block renders nothing (R1: deleting it outright would make
feedback invisible for xpolicylab/scripted runs, and `/stop [note]`
end-of-episode notes are precisely the high-value annotations). The log
schema and data are untouched. `tests/test_eval_log.py`'s
`"Operator feedback" in document` assertion changes accordingly (R1).

### 4. Run video

Each trial's transcript section is headed by a **video block** (for the
common single-scene rig run this is the top of the report):

- **Flipbook tier (always, zero new bytes):** frame `<img>` tags gain
  `data-camera`/`data-step`/`data-trial` attributes. A new inline
  `_FLIPBOOK_SCRIPT` (pattern: the existing `_FRAME_CLICK_SCRIPT`) scans
  the DOM **by `[data-camera]` selector scoped to the trial's
  `data-trial`** (R1: `img.frame` alone would pull in wire blobs, which
  reuse the class), groups by camera, and drives a player `<img>` by
  assigning `src` from the *existing* elements — the data URLs are never
  duplicated in the HTML, so page weight is unchanged. The player img
  lives outside any `.frame-cell` so `_FRAME_CLICK_SCRIPT`'s wide-toggle
  never fires on player clicks (R1). Controls: play/pause, camera tabs, a
  range-input scrubber labeled with the step number; autoplay on load,
  looping, ~4 fps. Player position and pause state persist across the 2s
  live reload via `sessionStorage` keyed by
  (`location.pathname`, trial, camera) (R2: pathname is unique within one
  served directory; cross-directory same-port bleed is harmless pause
  state) — a served flipbook that restarts every refresh is unusable as a
  scrubber (R1). With fewer than 2 frames for every camera the block
  renders nothing.
- **Video block placement** (R2): inside each trial's existing
  `<details class="transcript">`, immediately after its `<summary>` — a
  collapsed trial costs nothing (no decode of invisible videos), and
  `_scene_section`'s surrounding order (feedback residual, transcripts,
  wires) is otherwise unchanged. **Only the active camera tab plays**:
  inactive tabs are paused with `preload="metadata"`, and the flipbook
  script owns tab switching for both tiers (R2: N cameras × M trials of
  perpetually looping hidden H.264 melts the viewing laptop; the no-JS
  constraint applies to the POV dropdown only).
- **MP4 tier (completed pages, ffmpeg present, never under `--serve`):**
  when `shutil.which("ffmpeg")` succeeds, the log status is not
  `"started"`, **and the render is not a serve pass** (R1: serve re-renders
  in the 2s tick loop — a run finishing mid-serve would run ffmpeg inside
  the tick and stall every live page past the 0055 cadence; serve pages
  get the flipbook, always), and the frames side-car directory resolves,
  the renderer stitches **all control steps** per (trial, camera) through
  the existing `_video.py` pipeline machinery. Layering pinned (R3, so the
  `video` command's output contract and existing tests stay untouched): a
  shared **private core** carrying the current pipeline (stderr-tempfile,
  per-stream failure isolation, plan 0016) whose result distinguishes
  launch failure from encode failure (private exception or result kind);
  `encode_stream` keeps its exact current public signature and behavior on
  top of it (writes `out_path`, raises `SystemExit` on launch failure —
  existing `video` tests and per-stream CLI output unchanged); new
  `encode_camera_mp4(...) -> bytes | None` calls the core with a temp
  output path, reads the bytes back, unlinks the temp file on success and
  failure alike, and returns `None` on any failure including `Popen`
  launch errors — it never raises `SystemExit` (R2: the directory pass
  catches only `Exception` per log, so the command posture would let one
  broken ffmpeg shim abort the entire `view` render). `-f mp4` cannot
  stream to a pipe, and faststart is irrelevant for a fully in-memory
  `data:` URL (R1). Streams are
  enumerated by globbing `f"{trial_prefix}_*.npy"` and parsing
  `camera_step` from the remainder after stripping the known trial prefix
  (R1: `discover_streams`' un-splittable key cannot give per-trial
  association, and transcript-derived cameras would miss unrevealed
  on-demand cameras). Embeds `<video controls muted loop>` with a base64
  `data:video/mp4` URL, one per camera tab, replacing the flipbook for
  that trial; the `autoplay` attribute lives only on the active tab's
  element (R3, consistent with active-tab-only playback). Playback is
  real-time (frame rate = `control_hz` via the existing `default_fps`
  guards).
  Accepted, documented cost: a first `view` of a large directory encodes
  every completed log's videos once; the mtime gate makes it a one-time
  cost per log. **Suppressed-tier pages stay upgradeable via a two-level
  stamp** (R2, mechanism corrected R3 — merely skipping the stamp cannot
  work: a fresh page's natural mtime is render wall time, *newer* than the
  log, so the `<` gate freezes it regardless): a full-tier render stamps
  the page with the source log's mtime `S`; a suppressed-tier render
  (serve pass or `--no-video`) stamps `S − 1ns`; and the gate compares the
  page's mtime against the stamp *this pass would write*. Serve ticks skip
  pages already at `S − 1ns` (no per-tick churn even with a live log
  present), while the next eligible plain `view` sees `S − 1ns < S`,
  re-renders with video, and stamps `S`. Pages rendered by pre-0059
  versions carry the full-tier stamp and need `--force` to gain video
  (documented; R3).
- **Budgeting:** embedded MP4s are charged to a dedicated
  `_VIDEO_BUDGET_BYTES = 30_000_000` per page of **encoded base64
  characters** (R1: matching the frame-budget precedent, which charges
  payload chars, not raw bytes), spent in document order, first-wins,
  like the frame budget (R2) — not the frame budget itself (the video
  replaces the need for more stills, and H.264 across near-identical
  robot frames is far denser than PNG). Over budget → that camera
  degrades to the flipbook, with a small header chip naming the degrade
  reason (budget vs no ffmpeg would otherwise be indistinguishable; R2). `--no-video` on `view` skips the MP4 tier entirely (flipbook
  remains); `--no-frames` (no `_FrameContext`) disables both tiers.
- **Live pages always use the flipbook** — never invoke ffmpeg on the 2s
  tick. Because live pages already embed newest-first frames (plan 0058),
  the flipbook there is a bounded window over recent turns (the full
  turn-aligned scrubber experience belongs to completed, unserved pages).
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

- **Turn grouping:** user/assistant/tool sequences; string-content user
  messages (nudges, capx reports) stay inside the current turn and render
  visibly; assistant prose visible in the default layer (capx transcript
  golden case); leading assistant preamble; missing roles; non-dict
  messages; non-chat fallback untouched (differential assertion against a
  golden snippet); frames on live pages still resolve through the 0058
  cache (identity contract regression test).
- **Headers/captions:** step from the turn's first frame label; headerless
  when the turn has no frame references, even when a feedback chip matched
  (R3: the R2 rule); captions are camera-name-only.
- **Feedback chips:** sourced from `scene.operator_messages`; `/stop`-style
  undelivered tail message appears inline (matched by `t`); multi-source
  labels; non-chat trial's messages land in the residual block; a
  zero-frame-step chat transcript (capx shape) routes to the residual too;
  fully placed messages leave the residual empty.
- **Pretty calls:** dict-of-numbers → per-key chips (4-decimal formatting),
  `deltas` as well as `targets`, scalars, array arguments (take_pic
  cameras), multiple calls per message, note exclusion, hindsight rendered
  note-style not chip, malformed JSON falls back raw.
- **LLM POV:** state dumps, delivered feedback lines, and raw call text
  present inside `<details>`, absent outside; single-escape (adversarial
  `<script>` in state text and note text); tool-role results inside.
- **Badge:** started log where the live-marker slot's scene shows running
  (asserted on the marker, not slot position; R1) while a finished scene
  in the same log keeps completed; completed log unchanged.
- **Operator section:** demoted to residual — absent when all messages
  placed inline; present listing only unplaced messages; data untouched in
  the log JSON (existing `test_eval_log.py` assertion updated).
- **Flipbook:** data attributes on frame imgs; script tag present exactly
  once; no data-URL duplication (page contains each URL once — asserted by
  count); fewer-than-2-frames renders no block.
- **MP4 tier:** ffmpeg resolved via `shutil.which` monkeypatch; a fake
  encoder returning known bytes → `<video>` with data URL, per-camera tabs;
  encoder returning None → flipbook degrade; video budget exceeded →
  degrade (budget counted in encoded chars); `--no-video`; started log
  never calls the encoder; **serve passes never call the encoder** even
  for completed logs; per-(trial, camera) stream enumeration incl. a
  camera never referenced in the transcript; real ffmpeg invocation stays
  covered by the existing `video`-command tests through the shared helper
  — its `_FakePopen` gains a writes-the-output-file behavior for the
  temp-file read-back path (R1: today it never writes `out_path`);
  `Popen` raising `OSError` under the renderer → flipbook degrade and
  `view` exits 0 (R2); the two-level stamp — suppressed-tier pages stamp
  `S − 1ns`, skip on serve ticks, upgrade and restamp `S` on the next
  eligible plain pass (R3); video block nests
  inside the trial details after its summary; only the active camera tab
  autoplays (inactive paused, `preload="metadata"`).
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
