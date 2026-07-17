# 0023 — `view` embeds the camera frames each agent turn saw

Issue: #141. Status: draft (revision 2).

## Problem

Agent transcripts are image-free by design (plan 0015: `_sanitize` replaces
every `image_url` part with the text `[image omitted: streamed camera
frame]` before persistence). The HTML viewer (plan 0022) therefore shows the
model's notes and tool calls but not what the model actually *saw*, which is
half of the "see what the model sees and decides" goal.

For `--store-frames` runs the pixels are already on disk, and the sanitized
transcript retains everything needed to find them:

- Observation content keeps the text part `camera {name!r}{suffix}:` where
  suffix is ` (step {n})` (agent plugin `_observation_content`). Through
  `eval()` the suffix is always present: the **rollout** injects
  `extra["env_step"] = t` unconditionally on the policy-facing observation
  (rollout.py; `types.py` reserves the key — embodiments must not set it).
  No-suffix labels occur only in pre-0018 logs or out-of-rollout policy use.
- `FrameStore` writes `{_safe(trial_id)}_{_safe(camera)}_{t:06d}.npy` with
  `trial_id = f"{scene.id}-e{epoch}"` (rollout.py). The same loop variable
  `t` and the same `obs` object flow to both the controller call and
  `_store_frames`, so `env_step == FrameStore.put`'s `t` **by
  construction**, for every embodiment.
- The frame for a labeled step *usually* exists — not always: frames are
  stored after `embodiment.step()` succeeds, so a trial that errors or is
  interrupted at step `t` has a label with no file, and camera-less
  observations store nothing. Both degrade (below).
- `SceneResult.policy_transcripts` is strictly parallel to `epochs`
  (lockstep append in eval.py; halt-only skips break the loop immediately),
  so the transcript's `enumerate` index *is* the epoch index — including
  across `None` entries.

Verified against a real yam run (log `adhoc_2764b484.json`): label
`camera 'top_cam' (step 25):` ↔ file `scene-0-e0_top_cam_000025.npy`,
224×224×3 uint8.

## Design

When the log's `stats.frames_dir` resolves to an existing directory, `view`
replaces each observation turn's `[image omitted: streamed camera frame]`
placeholder with the matching stored frame, inlined as a PNG data URL, up to
a page-size budget. Default ON; `--no-frames` opts out.

### The renderer restructuring this actually requires

Today `_chat_content` collapses a message's parts list into ONE
`"\n"`-joined string, and `_render_message` emits it as a single
`<div class="content">`. Part structure is gone before rendering — so this
plan adds a **part-level rendering path**, and that (not the encoder) is the
bulk of the diff:

- `_chat_content` itself is **unchanged** (cli.py imports it for the
  terminal `--transcript` path; the module map documents that ownership).
- `_render_message` gains a frame-aware branch used only when (a) a frame
  context is armed and (b) the message role is `user` and (c) its content
  is a parts list. All other messages, and all frames-off renders, use the
  existing collapsed path untouched — plan 0022's renderer tests pass with
  no edits.
- The part-level path groups parts into **runs**: iterate parts in order,
  appending each part's text form (text parts via
  `str(part.get("text", ""))`, non-text/non-dict parts as `[image]` —
  identical elementwise to `_chat_content`) to a run buffer. Only when a
  placeholder part is actually replaced by an image does the path flush the
  buffer as one `<div class="content">` (its lines `"\n"`-joined), emit the
  `<img>`, and start a new buffer. **Terminal flush rule (asymmetric, and
  the crux of byte-identity):** at end of parts, if zero embeds occurred,
  flush unconditionally — even an empty buffer — because the collapsed path
  renders `content: []` as an empty `<div class="content"></div>`
  (`_chat_content([])` returns `""`, not `None`); if at least one embed
  occurred, suppress empty trailing/adjacent runs (no empty divs beside an
  `<img>`). Consequence: **zero embeds produce one run, byte-identical
  content markup to the collapsed path** — a frames-armed page where every
  lookup degrades has exactly the same DOM as a frames-off page.

Context threading: `_scene_section` gains a `frame_ctx` parameter and
builds per-trial contexts; `_render_transcript` / `_render_chat_transcript`
/ `_render_message` pass through a frozen
`_FrameContext(frames_dir: Path, trial_prefix: str, budget: _FrameBudget)`
or `None` (all internal signatures; nothing exported). `trial_prefix =
_safe(f"{scene.scene_id}-e{epoch}")` with `epoch` the transcript's
enumerate index; `_safe` is imported from `inspect_robots.frames`. The
filename is reconstructed, never parsed — sidestepping the ambiguous
`trial_camera` split `_video.py` documents.

### Correlation contract (exact-match or degrade)

Within the part-level path, in part order:

1. A text part `re.fullmatch`-ing
   `camera '(?P<name>.*)' \(step (?P<step>\d+)\):` arms a pending
   (camera, step). `fullmatch`, not `match`+`$`: `$` would accept a
   trailing newline the writer never produces, and the contract is exact.
   The label is produced by `f"camera {name!r}..."`: a name containing
   only `'` gets double-quoted repr and misses the regex; a name containing
   backslashes, or both quote types (repr stays single-quoted with escapes),
   **matches** but captures repr-escaped text and degrades later at the
   file-existence check (different `_safe` CRC). Names that are repr-plain
   but filesystem-unsafe (spaces, slashes) reconstruct **correctly**, since
   the viewer applies the same `_safe` as the writer — that is the design's
   point and gets a positive test.
2. A text part exactly `[image omitted: streamed camera frame]` with a
   pending (camera, step) becomes
   `<img class="frame" loading="lazy" alt="camera {name} step {n}" src="data:image/png;base64,...">`
   iff `frames_dir / f"{trial_prefix}_{_safe(name)}_{step:06d}.npy"` exists,
   `np.load(..., allow_pickle=False)` succeeds (the default is already
   False; the explicit flag is documentation, not a behavior change —
   foreign frame files must never unpickle), the array is **already
   `uint8`** (no dtype coercion: a stored float frame coerced blind would
   embed as silently-black garbage, the exact failure `_video._normalize`
   documents; non-uint8 degrades), non-empty (`size > 0` — zero-size
   warm-up frames are documented first-party data in `_video` and must
   degrade, not embed as width-0 PNGs), of shape `(H, W)` /
   `(H, W, 1|3|4)`, and the page budget (below) is not exhausted. The
   pending state is consumed either way.
3. Any miss — no frames dir, regex miss, missing file (including the
   errored/interrupted final step), load error, wrong shape, budget
   exhausted — leaves the placeholder text in the run buffer exactly as
   today. No stderr output: absent pixels are not a failure of the page.

The camera-label text part always stays in the text flow (it is the image's
caption, immediately above it).

### Page budget (the honest big-page mitigation)

`loading="lazy"` defers nothing for data: URLs — document parse and memory
dominate — and the default 100-call budget × 3 cameras × multi-scene logs
can reach hundreds of MB unmitigated. So: a cumulative budget on encoded
PNG payload, default **50 MB**, tunable via `--frames-budget MB` (float; `0`
means unlimited). Once cumulative base64 payload would exceed the budget,
every further lookup degrades to the placeholder, and the page header's
meta row gains one visible chip: `frames truncated at {budget} MB
({embedded} embedded)`. Rendering therefore needs embed/truncation state;
that lives in the mutable `_FrameBudget` (embedded count, payload bytes,
truncated flag) carried by the contexts — render_html's return type stays
`str`; there is no result-object API (the previous draft's `RenderResult`
protected callers that do not exist — `render_html` is private-module and
only cli.py and tests import it).

### PNG encoding: new core module `src/inspect_robots/_pngenc.py`

Dependency-free (numpy + stdlib `zlib`/`struct`/`base64`):
`encode_png(arr) -> bytes`, `png_data_url(arr) -> str`, mirroring the agent
plugin's `_png.py` but strict-uint8 (frames are stored uint8; floats are
rejected — the plugin's float scaling serves live observations, not stored
frames). The plugin keeps its own copy: it supports older cores, and
coupling it to a same-release core for 40 lines is worse than the
duplication.

Frames whose longest side exceeds 448 px are integer-stride subsampled
(`arr[::k, ::k]`, `k = ceil(longest/448)`) before encoding; stored yam
frames are 224 px and embed as-is.

### CSS

`_STYLES` gains an `img.frame` rule: `display: block; max-width: 100%;
height: auto; margin: 6px 0; border: 1px solid var(--line);
border-radius: 6px;` so frames never overflow the narrow-viewport `main`.

### API and CLI surface

- `render_html(log, *, title, frames_dir: Path | None = None,
  frames_budget_bytes: int = 50_000_000) -> str`. Default `None` embeds
  nothing (plan 0022 tests untouched).
- `_cmd_view` resolves `log.stats.frames_dir` via the existing
  `resolve_frames_dir(frames_dir, log_path)` from `_video` (lazily imported
  in the same style as its three existing CLI call sites: the run-summary
  hint, `inspect`, and `_cmd_video`), passing `None` for `--no-frames`, a
  `frames_dir` of `None`, or an unresolvable dir.
- New `view` flags: `--no-frames` ("render placeholders instead of
  embedding stored camera frames") and `--frames-budget MB` (default 50,
  0 = unlimited; decimal megabytes, `bytes = int(mb * 1_000_000)`;
  negative values are a `SystemExit` in the style of `video --fps`; the
  truncation chip formats the budget with `:g`). `--no-frames` is a plain
  `store_true` flag, not the repo's `BooleanOptionalAction` pair idiom:
  the default is on, only the opt-out needs a name, and
  `--frames/--no-frames` would suggest a `--frames` value argument that
  does not exist.
- The `wrote OUT.html` line gains a size suffix whenever the written
  document exceeds 1 MB: `wrote OUT.html (12.3 MB)` — measured from the
  document itself, no renderer side channel. Stdout mode embeds
  identically (no wrote line, as before).

### Security / robustness

- PNG bytes come from the decoded array; base64 output is ours. Camera
  names appear escaped (`_escape`) in the `alt` attribute; no log-derived
  text enters `src`.
- The regex/lookup runs only on string parts inside list-shaped content of
  `user`-role messages of chat-shaped transcripts, only when frames are
  armed; adversarial content elsewhere keeps its plan-0022 behavior.

## Compatibility

- Logs without stored frames, moved logs whose frames dir no longer
  resolves, pre-0018 transcripts without step labels, non-agent
  transcripts, `--no-frames`: byte-identical placeholder rendering.
- No schema change, no new dependency, `__all__` untouched.
- Core version: minor bump on release.

## Tests

`tests/test_pngenc.py`:
- PNG signature and IHDR dims for `(H, W)`, `(H, W, 1)`, `(H, W, 3)`,
  `(H, W, 4)`; non-uint8 rejected; data-url prefix.

`tests/test_html_view.py` (renderer, `tmp_path` frames):
- Happy path: label + placeholder + matching `.npy` → one `<img` with
  `loading="lazy"`, base64 PNG src, escaped alt; caption text still in the
  preceding content run; placeholder text gone. Viewer-level accept
  branches for each allowed shape: `(H, W)`, `(H, W, 1)`, `(H, W, 3)`,
  `(H, W, 4)` (the branch-coverage gate counts these in the viewer's
  whitelist, not just in `_pngenc`).
- Byte-identity: frames-armed render where every lookup misses equals the
  frames-off render of the same log, exactly — and the fixture log's parts
  inventory must include the fragile edges: a `content: []` message, a text
  part with no `"text"` key, a non-str `"text"` value, and a non-dict part.
- Degrade branches, each its own case: missing file (including an
  errored-trial final step: label `(step t)` present, file absent); label
  without step suffix; camera name containing only `'` (regex miss); name
  with backslash (regex hit, file miss); positive control: name with a
  space reconstructs and embeds; placeholder with no pending label; label
  with no following placeholder; corrupt file; pickled `.npy` (object-array
  `np.save`) degrades instead of raising; wrong channel count `(H, W, 2)`;
  non-uint8 dtype file; zero-size array (`(0, W, 3)`).
- Multi-camera turn: three label+placeholder pairs embed in order.
- Multi-trial: `policy_transcripts = (None, chat)` — the chat transcript at
  index 1 looks up `-e1` files (pins enumerate-index-as-epoch).
- Oversize frame (1000 px) stride-subsamples: decoded IHDR dims ≤ 448.
- Budget: tiny `frames_budget_bytes` embeds the first frame, degrades the
  rest, and renders the truncation chip with the embedded count;
  `0` = unlimited embeds everything; a non-truncated render contains
  **zero** truncation chips (exact-count convention, like the suite's
  escaping tests).
- Non-chat transcripts and assistant/tool messages never embed.

CLI (`tests/test_registry_cli.py`):
- End-to-end synthetic log + frames dir resolved via the fallback candidate
  (`<log-dir>/frames/<stamp>`): embeds; `--no-frames` renders placeholders;
  recorded-but-unresolvable frames dir renders placeholders (no error);
  `--frames-budget` forwards; `wrote` size suffix appears over 1 MB and is
  absent under it; `-o -` embeds.

Gates: 100% core coverage, `mypy --strict` (src+tests), ruff incl. D1.

## Docs

- `docs/guide/cli.md` `## inspect-robots view`: frames embed for
  `--store-frames` runs, `--no-frames` and `--frames-budget` flags, page
  size implication.
- `docs/guide/logging-and-rerun.md`: the passage that tells programmatic
  consumers to use `StepRecord.image_refs`/`FrameRef.path` "instead of
  assembling the path from the transcript label" must be reconciled: it
  stays the right advice for code, and gains a sentence that
  `inspect-robots view` now performs exactly this join internally with an
  exact-match-or-degrade contract. The adjacent "step join key" prose gets
  the same mention.
- README's view mention gains "including the camera frames the model saw
  (for --store-frames runs)".
- CHANGELOG entry under core Added.
- Module map: `_pngenc.py` row; update the `_html.py` **and** `cli.py`
  rows (the `view` description).

## Rollout

Single PR (`Closes #141`), core only. Release the next core minor after
merge, then upgrade the rig test dir per standing practice.
