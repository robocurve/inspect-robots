# 0036 — Depth observability parity: rerun live view + frame store

**Issue:** #236 · **Branch:** `feat/depth-observability` · **Status:** draft (pre-critique)

## Problem

Depth-to-LLM shipped in the agent plugin (plan 0028): embodiments supply
per-camera metric depth via `observation.extra[f"{cam}_depth"]` (2-D float
array or zero-arg callable, per plan 0021), and the policy renders it into
observation messages. But depth is invisible everywhere else:

- The **rerun live viewer** logs only RGB (`RerunSink._emit` →
  `rr.Image` per camera); a depth run cannot be monitored live.
- The **frame store** persists only `observation.images`; depth arrays are
  dropped, so post-hoc analysis and `inspect-robots video` have no depth
  record. Wire captures (rendered PNGs in request bodies) are currently the
  only persisted depth artifact.

Plan 0028 explicitly deferred frame-store depth; plan 0021 flagged the
follow-up ("strip declared bulk `extra` keys in `_store_frames`"). This plan
picks up both. Transcripts are a **non-goal**: they are image-free by design
for RGB and depth alike (wire capture is the exact-payload record, #207).

## Design

### D1. Resolve depth once per step (rollout-owned)

Hazard (from plan 0028): a depth *callable* resolved independently by the
policy (at `act()` entry), the rerun sink, and the frame store returns three
different frames for one timestep — the viewer would show a different moment
than the LLM saw.

Change: the rollout resolves depth **once**, at observation assembly (control
thread, before `policy.act`), mirroring the agent plugin's `resolve_depth`
contract: iterate `observation.images` keys, look up `f"{name}_depth"` in
`extra`, call if callable, `np.asarray(..., float)`, require `ndim == 2`,
convert failures to human-readable strings instead of raising. The resolved
mapping `dict[str, NDArray | str]` is:

- substituted back into the policy-facing observation's `extra` (arrays
  replace callables — the agent plugin's `resolve_depth` already passes
  arrays through unchanged, so no plugin change is needed);
- handed to the rerun sink and the frame store (below);
- **stripped from the persisted record**: after `_store_frames`, the stored
  observation's `extra` drops `{cam}_depth` keys (both callable and resolved
  forms), fixing the plan-0021 memory-bloat follow-up in the same motion.

Resolution is gated: if no consumer needs depth (no rerun sink, frames off,
and the policy will see the raw extra anyway), the rollout leaves `extra`
untouched — behavior identical to today. Simplest sufficient gate: resolve
when a rerun sink is active or the frame store is on; otherwise pass through.

### D2. Rerun: metric depth stream per camera

In `RerunSink`:

- `_StepPayload` gains `depth: Mapping[str, NDArray] | None`, snapshotted on
  the control thread in `log_step` (copy via `np.array`, same rationale as
  images: no live buffer crosses to the worker thread). String (failure)
  entries are skipped — rerun gets pixels or nothing.
- `_emit` logs each entry as `rr.DepthImage(depth, meter=1.0)` at entity
  path `{pre}/camera/{cam}/depth` (hierarchical child of the existing RGB
  entity, so the viewer groups them and hover shows meters). Guarded for SDK
  absence exactly like `_image`/`_scalar` (attribute lookup, degrade to
  no-op) — the fake-`rerun` unit suite plus the dedicated `test-rerun` CI job
  cover both paths.
- Backpressure: the `_image_watermark` drop branch in `_enqueue` drops
  `depth` alongside `images` and counts it in `_dropped_frames`; depth must
  not ride below the watermark for free.
- No `.compress()` on depth (JPEG is wrong for float; DepthImage handles its
  own encoding).

### D3. Frame store: `depth/` subdirectory, dtype-preserving refs

Naming constraint (scout): every frames-dir reader globs `*.npy`
non-recursively and re-parses names with `^(.+)_(\d{6,})\.npy$`, so a
sibling file `..._{cam}_depth_NNNNNN.npy` becomes a phantom video stream
that then hard-fails on dtype. A **subdirectory** sidesteps every reader at
once:

```
logs/frames/<run>/<trial>_<cam>_000042.npy          # RGB, unchanged
logs/frames/<run>/depth/<trial>_<cam>_000042.npy    # float32 metric depth
```

- `FrameStore.put_depth(trial_id, t, camera, depth) -> DepthFrameRef` writes
  `float32` (halves size vs float64; precision is far beyond sensor noise).
- `DepthFrameRef.load()` preserves dtype — the existing `FrameRef.load()`
  uint8 coercion would silently destroy depth, so depth gets its own ref
  type rather than a flag on the old one.
- `_store_frames` writes depth when the store is on and resolved depth
  arrays exist; the `not obs.images` short-circuit is left as-is (depth is
  keyed off RGB camera names, matching the agent plugin's convention — a
  depth-only camera is out of scope, as it is for the LLM path today).
- `StepRecord` gains `depth_refs` alongside `image_refs` (same shape,
  `Mapping[str, DepthFrameRef]`).
- Reuses the existing `store_frames` config flag — no new knob; depth
  presence in the observation is the opt-in (only RGB-D profiles have it).

### D4. Explicit non-changes

- `inspect-robots video`: no depth rendering (subdir keeps it blind to
  depth); a depth-video mode is a possible follow-up, not this plan.
- HTML viewer: unchanged; the plan preserves plan 0028's invariant that
  depth labels never match `_FRAME_LABEL_RE`.
- Transcripts: remain image-free; no embedded imagery.
- Agent plugin: unchanged (its `resolve_depth` already accepts pre-resolved
  arrays).
- Embodiments: unchanged (the `{cam}_depth` extra convention is untouched).

## Testing

- **Core (100% coverage gate applies):**
  - rollout: callable resolved exactly once per step (counting callable);
    resolution failures become strings and don't raise; persisted record has
    depth stripped; pass-through when no consumer is active.
  - frames: `put_depth` roundtrip preserves dtype/values; subdir layout;
    `_safe` collision behavior matches RGB path.
  - rerun sink (fake module): depth snapshotted, emitted to the right entity
    path, dropped+counted under watermark, no-op when archetype missing.
  - video: a frames dir containing `depth/` produces identical
    `discover_streams` output to one without (regression pin).
- **Real-SDK:** extend `test_rerun_sink.py` cases exercised by the
  `test-rerun` CI job with a DepthImage assertion.

## Rollout

Pure-additive core change → minor version bump. Changelog entry under
Unreleased. No config migration; existing RGB-only rigs see zero behavior
change (no `{cam}_depth` keys → every new path is a no-op).

## Open questions (for critique)

1. Should the resolve-once gate also cover "policy consumes depth but no
   sink/store" to guarantee LLM/viewer alignment even when only one consumer
   exists? (Cheap to include; slight behavior change for policies relying on
   late resolution.)
2. `meter=1.0` assumes embodiments supply meters — true for yam
   (librealsense aligned depth in meters). Worth asserting/documenting in
   the extra-key convention?
3. float32 vs float16 for stored depth (float16 quarters size, ~3 mm quant
   error at 4 m — likely fine, but float32 is the conservative default).
