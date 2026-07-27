# 0028 — Agent: render per-camera depth into observation messages

Closes robocurve/inspect-robots#190.

## Problem

Embodiments can publish per-camera metric depth into `observation.extra`
(robocurve/inspect-robots-yam#70/#71, merged 2026-07-27): `{cam}_depth` is an
H×W float32 array of metres *or* a zero-arg callable returning one, aligned
to and same-resolution as that camera's entry in `observation.images`;
`{cam}_intrinsics` is a 3×3 K. The agent policy surfaces none of it. Both
render paths — the `images="always"` observation message and the
`images="on_demand"` `take_pic` reveal — go through `_image_parts`
(`policy.py:727-744`), which iterates `observation.images` only; the sole
`extra` key the policy reads anywhere is `env_step`. An LLM driving a
depth-capable rig still sees RGB alone and recovers scale by parallax
(measured on yam_arms: ~15 LLM calls per trial of manual calibration, 0/5
`stack_the_bowls` successes before the camera-side fix).

## Design

One seam, both modes: `_image_parts` gains depth parts per rendered camera.
`take_pic` inherits the behavior because its reveal path calls the same
helper with `reveal=<captured names>` (`policy.py:617,723`).

For each camera `name` in the rendered set, when
`observation.extra[f"{name}_depth"]` exists and the policy's `depth` knob is
`"render"` (default):

1. **Resolve.** If the value is callable, call it under `try/except
   Exception`; a failing thunk must not kill the observation message — emit
   the text line `depth 'left_cam' unavailable: {exc}` and no image. A
   non-callable value is used as-is. `np.asarray(value, dtype=np.float64)`
   must be 2-D; anything else gets the same unavailable line (reason
   "expected an HxW array").
2. **Normalize.** Valid pixels are finite and > 0. If fewer than 1% of
   pixels are valid, emit `depth 'left_cam': no usable depth this frame
   ({pct}% valid)` and no image. Otherwise the render window is the 2nd–98th
   percentile of valid values (degenerate window lo == hi widens to
   lo, lo+1e-3).
3. **Render.** Grayscale, near = bright: valid pixels map linearly from the
   window to 255 (at lo) down to 64 (at hi), clipped; invalid pixels are 0
   (pure black, distinguishable from the dimmest valid 64). uint8 H×W×3 (the
   existing `png_data_url` / `encode_png` in `_png.py` accepts the same
   shapes as RGB frames). No colormap dependency; a perceptual colormap is
   YAGNI while the metric anchors carry the absolute information.
4. **Anchor the scale in text.** The colormap discards absolute scale, which
   is the entire point of depth, so the label line carries it:

   ```
   depth 'left_cam' (step 3): bright 0.09 m -> dim 1.41 m (2nd-98th pctl), 87% valid, center 0.31 m:
   ```

   followed by the image part. "center" is the depth at (H//2, W//2) when
   that pixel is valid, omitted otherwise. Numbers use two decimals (metres).
5. **Ordering.** Depth parts follow immediately after that camera's RGB
   parts, inside the same loop, so RGB and depth for one camera are adjacent.

### Label shape is constrained by the HTML viewer

`src/inspect_robots/_html.py:25` rejoins transcript image slots to saved
frames with `_FRAME_LABEL_RE = camera '(?P<name>.*)' \(step (?P<step>\d+)\):`.
Depth labels MUST NOT match it: depth renders have no saved `.npy` frame
(core saves `observation.images` only), and a matching label would make the
viewer look for one. The format above starts with `depth '` — it cannot
match. A test pins this: `_FRAME_LABEL_RE.search(depth_label) is None` using
the real regex imported from `inspect_robots._html`.

In transcripts the depth image part becomes the standard
`[image omitted: streamed camera frame]` placeholder (`policy.py:111`), so
saved transcripts keep the metric text line — the valuable part — and drop
the pixels, exactly like RGB. The HTML viewer shows the depth text with no
image tile. Acceptable; saving depth renders to the frames dir is core-side
work and out of scope (noted in #190 as a non-goal).

### Configuration

New policy config field, following the existing style
(`policy.py:136` block): `depth: str = "render"`, accepting `"render"` |
`"off"`. `"off"` restores today's output byte-for-byte. Unknown values fail
at construction alongside the existing `images` validation. Rationale: depth
rendering costs one image per depth camera per observation — for
LLM-call-budgeted trials the operator needs a kill-switch without
reconfiguring the embodiment.

No system-prompt changes: the labels are self-describing, and the
embodiment's own `info.docs` already explains depth (aimed at code-level
consumers; the rendered parts are what this policy's LLM consumes).

### Failure and absence semantics (exhaustive)

- No `{name}_depth` key → exactly today's parts for that camera.
- `depth="off"` → exactly today's parts, even when keys exist.
- Thunk raises → text line, no image, message otherwise intact.
- Non-2-D / non-numeric value → text line, no image.
- <1% valid pixels → text line, no image.
- `{cam}_intrinsics` is never read (non-goal; tool layers consume K).

## Implementation tasks

Ordered; the plugin gates are `uv run --no-sync ruff check
plugins/inspect-robots-agent`, `ruff format --check`, `mypy --config-file
plugins/inspect-robots-agent/pyproject.toml plugins/.../src/...` (strict),
and `python -m pytest plugins/inspect-robots-agent/tests -q`; the core suite
(`uv run pytest`) must stay green (787 passed baseline).

1. **`_depth.py`** (new module in the plugin, beside `_png.py`):
   `depth_parts(name: str, value: Any, step_label: str | None) ->
   list[dict[str, Any]]` implementing resolve/normalize/render/anchor above,
   returning `[{"type":"text",...}, {"type":"image_url",...}]`, a single
   text-only list for the unavailable/degenerate cases, or `[]` never (an
   absent key is the caller's case). Unit tests: happy path (label text
   format pinned, image is a data URL, near pixel brighter than far pixel,
   invalid pixel black), thunk called exactly once, failing thunk, non-2-D
   value, <1% valid, degenerate window, center-invalid omits "center", the
   `_FRAME_LABEL_RE` non-match pin.
2. **Wire into `_image_parts` + config knob**: `_image_parts` gains
   `extra: Mapping[str, Any]` and `depth_mode: str` parameters (both callers
   pass `observation.extra` and the config value); per camera, append
   `depth_parts(...)` when the key exists and mode is `"render"`. Config
   field + validation + docstring. e2e tests (existing MockTransport style,
   `test_policy_e2e.py`): observation message contains RGB and depth parts
   for a camera with a depth thunk in `extra`; `take_pic` reveal includes
   depth for the captured camera; `depth="off"` and no-key cases match the
   pre-change message shape exactly; a failing thunk still produces a valid
   observation message with the unavailable line.
3. **Docs**: plugin README section (what renders, the label format, the
   `depth` knob, the transcript/HTML-viewer caveat); CHANGELOG if the plugin
   keeps one (check; the core repo convention applies otherwise).

## Out of scope

Consuming `{cam}_intrinsics`; `/act` wire changes; embedding depth into
`observation.images`; saving depth renders into the frames dir (core);
colormaps beyond grayscale; per-camera enable lists.
