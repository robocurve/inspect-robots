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

**Resolution happens once, at observation receipt — never at render time.**
The embodiment contract is resolve-on-receipt, and in `images="on_demand"`
an immediate `take_pic` reveal renders at `policy.py:613-619` after one or
more LLM round-trips inside the same `act()` (the `while True` at :471) —
tens of seconds after the observation arrived. A thunk resolved that late
returns depth captured *now*, silently misaligned with the frozen RGB shown
beside it. So: at `act()` entry, when the `depth` knob is `"render"`, every
`observation.extra[f"{name}_depth"]` for `name` in `observation.images` is
resolved into a per-observation cache
`dict[str, npt.NDArray[np.float64] | str]` (bare `np.ndarray` trips the
plugin's mypy `disallow_any_generics`; the file already uses
`npt.NDArray[np.float64]`) — array on success, human-readable failure text
on any error — and both render paths consume only the cache. Each thunk is called exactly once per
observation (pinned by an e2e test across an `on_demand` `act()` with a
late `take_pic`).

For each camera `name` in the rendered set with a cache entry:

1. **Resolve (at receipt, into the cache).** If the value is callable, call
   it; coerce with `np.asarray(value, dtype=np.float64)` and require 2-D —
   the call *and* the coercion sit in one `try/except Exception` (asarray
   itself raises on non-numeric values), and any failure stores the text
   `depth 'left_cam' unavailable: {reason}` in the cache instead of an
   array. A failing thunk must never kill the observation message: a
   text-only cache entry renders as that line with no image.
2. **Normalize.** Valid pixels are finite and > 0. If fewer than 1% of
   pixels are valid, emit `depth 'left_cam': no usable depth this frame
   ({pct}% valid)` and no image. Otherwise the render window is the 2nd–98th
   percentile of valid values (degenerate window lo == hi widens to
   lo, lo+1e-3).
3. **Render.** Grayscale, near = bright: valid pixels map linearly from the
   window to 255 (at lo) down to 64 (at hi), clipped; invalid pixels are 0
   (pure black, distinguishable from the dimmest valid 64). Emit **2-D
   uint8 H×W** — `encode_png` (`_png.py:34-35`) handles `ndim == 2` as true
   grayscale (colour type 0), identical appearance at a third of the raw
   bytes of H×W×3, which matters for the LLM-payload cost the `depth` knob
   exists to control. No colormap dependency; a perceptual colormap is
   YAGNI while the metric anchors carry the absolute information.
4. **Anchor the scale in text.** The colormap discards absolute scale, which
   is the entire point of depth, so the label line carries it:

   ```
   depth 'left_cam' (step 3): bright 0.09 m -> dim 1.41 m (2nd-98th pctl), 87% valid, center 0.31 m:
   ```

   followed by the image part. "center" is the depth at (H//2, W//2) when
   that pixel is valid, omitted otherwise. Metres use two decimals; the
   valid percentage is an integer. When the step label is empty
   (`_step_label` returns `""`, `policy.py:702-705`), the `(step N)` suffix
   is omitted exactly as the RGB label omits it (`policy.py:734-735`):
   `depth 'left_cam': bright 0.09 m -> dim 1.41 m ...`.
5. **Ordering.** Depth parts follow immediately after that camera's RGB
   parts, inside the same loop, so RGB and depth for one camera are adjacent.

### Label shape is constrained by the HTML viewer

`src/inspect_robots/_html.py:25` rejoins transcript image slots to saved
frames with `_FRAME_LABEL_RE = camera '(?P<name>.*)' \(step
(?P<step>\d{1,12})\):` (the viewer applies it with `fullmatch`,
`_html.py:357`). Depth labels MUST NOT match it: depth renders have no
saved `.npy` frame (core saves `observation.images` only), and a matching
label would make the viewer look for one. The format above starts with
`depth '` — it cannot match. A test pins this:
`_FRAME_LABEL_RE.search(depth_label) is None` using the real regex imported
from `inspect_robots._html` (`search` is strictly stronger than the
viewer's `fullmatch`).

In transcripts the depth image part becomes the standard
`[image omitted: streamed camera frame]` placeholder (`policy.py:111`), so
saved transcripts keep the metric text line — the valuable part — and drop
the pixels, exactly like RGB. Known cosmetic artifact, accepted: because
the depth label never matches the join regex, the viewer renders that
placeholder text verbatim under each depth label (`_html.py:360-371`)
instead of an image tile. Saving depth renders to the frames dir is
core-side work and out of scope (noted in #190 as a non-goal).

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

1. **`_depth.py`** (new module in the plugin, beside `_png.py`). Three
   functions, final signatures (resolution does NOT live in `depth_parts`):
   - `_render_depth(depth: npt.NDArray[np.float64]) ->
     npt.NDArray[np.uint8]` — the normalize+grayscale step alone, 2-D uint8
     out. Exists so pixel-level tests need no PNG decoder (the repo is
     encoder-only by design). uint8 input to `encode_png` also bypasses its
     float-normalization branch, which is load-bearing.
   - `depth_parts(name: str, entry: npt.NDArray[np.float64] | str,
     step_label: str) -> list[dict[str, Any]]` — a str entry (failure text
     from resolution) renders as that text line with no image; an array
     entry produces `[{"type":"text",...}, {"type":"image_url",...}]` via
     `_render_depth` + `png_data_url`, or the text-only degenerate cases
     (<1% valid). Never `[]` (an absent key is the caller's case).
   - `resolve_depth(observation: Observation) ->
     dict[str, npt.NDArray[np.float64] | str]` — the per-observation cache
     builder from Design (call thunk + coerce under one try/except).
   Unit tests: `_render_depth` — near pixel brighter than far, invalid
   pixel exactly 0, valid floor 64, degenerate window; `resolve_depth` —
   thunk called exactly once, failing thunk → text, non-2-D → text,
   non-numeric → text, plain array passthrough; `depth_parts` — label text
   format pinned (with and without step label), image is a PNG data URL,
   str-entry passthrough, <1% valid, center-invalid omits "center", the
   `_FRAME_LABEL_RE` non-match pin.
2. **Resolution cache, wiring, config knob.** Exact signatures (the cache
   replaces any `extra`/mode threading — `_image_parts` already receives
   `observation`, and mode is decided at resolution time):
   - `resolve_depth` (task 1) is called once at `act()` entry
     (`policy.py:443` area) when the knob is `"render"`, else `{}`.
   - `_image_parts(observation, *, reveal: tuple[str, ...] | None = None,
     depth: Mapping[str, Any] | None = None)` — `None` (the default)
     renders exactly today's output, so the four existing direct-caller
     tests (`test_policy_e2e.py:395,412,421,437`) pass unchanged.
   - `_observation_content(..., depth: Mapping[str, Any] | None = None)`
     threaded from `act()`; same default.
   Per camera, append `depth_parts(name, cache_entry, step_label)` when the
   cache has an entry. Config knob follows the existing `images` pattern at
   all four touch points: `__init__` parameter (`policy.py:159-174`),
   validation beside the `images` check (`:194-198`), inclusion in the
   `AgentPolicyConfig(...)` construction (`:317-330`), and a stored
   `self._depth` read by `act()`. Docstring. e2e tests
   (existing MockTransport style, `test_policy_e2e.py`): observation
   message contains RGB and depth parts for a camera with a depth thunk in
   `extra`; `take_pic` reveal includes depth for the captured camera **and
   the thunk was called exactly once for that observation despite the late
   reveal**; `depth="off"` and no-key cases match the pre-change message
   shape exactly; a failing thunk still produces a valid observation
   message with the unavailable line.
3. **Docs**: plugin README section (what renders, the label format, the
   `depth` knob, the transcript/HTML-viewer caveat); CHANGELOG if the plugin
   keeps one (check; the core repo convention applies otherwise).

## Wire clients — verified, no work

All three wire formats consume the canonical chat parts `_image_parts`
emits: Chat passes through; Responses translates `text`/`image_url`
(`_responses.py:152-158`); Anthropic requires PNG data URLs
(`_anthropic.py:213-235`), which `png_data_url` output satisfies. capx's
`_observation_content` is an independent copy, not an import — no API
break, and capx does **not** gain depth from this change (it consumes
metric depth directly through its grasp tooling instead).

## Out of scope

Consuming `{cam}_intrinsics`; `/act` wire changes; embedding depth into
`observation.images`; saving depth renders into the frames dir (core);
colormaps beyond grayscale; per-camera enable lists; depth in the capx
plugin's independent observation renderer.

## Revision note

Plan rev 2, after fresh-context critique: thunk resolution moved from
render time to a once-per-observation cache at `act()` entry (a late
`take_pic` reveal would otherwise resolve depth seconds after the RGB it
sits beside — silently misaligned data on the resolve-on-receipt
contract); exact signatures for the threading through
`_observation_content` specified with back-compatible defaults; asarray
coercion moved inside the failure `try/except` (it raises on non-numeric
input); 2-D grayscale PNG instead of H×W×3 (a third of the payload);
regex quote corrected and the viewer's placeholder-text artifact
documented as accepted.

Plan rev 3, after second critique (no design flaws found): the task-4
amendment folded into task 1 so the ordered task list states each signature
exactly once; `_render_depth` split out so pixel-level tests need no PNG
decoder (the repo is encoder-only); cache type pinned to
`npt.NDArray[np.float64] | str` for the plugin's strict mypy; empty-step
label and integer-percent formats specified; the config knob's four
constructor touch points named.
