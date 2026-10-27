# 0087 — `inspect-robots-openvla`: base OpenVLA as a policy plugin

## 1. Goal

Run base [OpenVLA](https://github.com/openvla/openvla) 7B from Inspect Robots with
no model server and no second environment:

```bash
inspect-robots list policies          # -> includes "openvla_local"
inspect-robots run --task my-task --policy openvla_local --embodiment my-robot \
    -P unnorm_key=bridge_orig -P camera=agentview
```

## 2. Why a new plugin instead of an existing one

- `inspect-robots-xpolicylab` already serves OpenVLA-OFT, but only through the
  XPolicyLab websocket server; that requires a separate process, GPU box and
  port, and it does not run base OpenVLA checkpoints.
- `inspect-robots-widowx` registers a policy named `openvla`, but it is a REST
  client for OpenVLA's first-party `/act` server and is bundled with that rig,
  not installable on its own.
- Hence this plugin loads the HF checkpoint in-process, and takes the entry
  point name **`openvla_local`** so the two distributions never collide in the
  `inspect_robots.policies` group.

## 3. The one big decision: in-process weights, stub-testable

`OpenVLAPolicy` owns a `AutoModelForVision2Seq` + `PrismaticProcessor` pair loaded
lazily on the first `act()`. Construction, `info`, `reset()` and
`check_compatibility` never import `torch`, which is what lets the whole suite run
in CI with neither a GPU nor the 14 GB checkpoint.

Both collaborators are injectable (`model=`, `processor=`), so tests drive the
real mapping code with a duck-typed stub that records its calls:

| adapter call | stub |
|---|---|
| `processor(prompt, image)` | records prompt and image, returns a dict subclass with `.to()` |
| `inputs.to(device, dtype=...)` | records the cast, returns itself |
| `model.predict_action(**inputs)` | records kwargs, returns a 7-D `numpy` array |

## 4. Action contract

| dim | meaning |
|---|---|
| 0–2 | end-effector position delta (m, base frame) |
| 3–5 | end-effector rotation delta, Euler XYZ (rad) |
| 6 | gripper, binary: `+1` closed / `-1` open |

`PolicyInfo.action_space` publishes `control_mode="eef_delta_pose"`,
`rotation_repr="euler_xyz"`, `gripper="binary"`, `frame="base"`. Base OpenVLA
predicts BridgeData deltas, so rotation stays Euler XYZ rather than being
misdeclared as axis-angle; an axis-angle embodiment fails compatibility instead
of misreading the values. Upstream emits
the RLDS gripper convention (`0` closed, `1` open), flipped once at the adapter
boundary. An 8-D OpenVLA-OFT checkpoint is rejected with a message pointing at a
dedicated adapter rather than silently truncated.

Instructions follow `observation.instruction` when the observation supplies one,
falling back to the instruction bound by `reset(scene)`; that keeps a
mid-rollout goal change working without a reset.

## 5. Image pipeline

`RGB` → (optional) center crop → one bilinear resample to 224 px → processor
rescales and normalises. The crop keeps `sqrt(scale)` of each side (upstream's
box formula) and targets 224 px in a single resample, which is what upstream's
`crop_and_resize` does.

## 6. Tests and CI

39 tests, 100% line+branch coverage, `mypy --strict`, `ruff` — all without
`torch`. Covered: prompt construction, per-step re-observation, instruction
rebinding on `reset()`, a mid-rollout goal change carried by
`observation.instruction`, gripper mapping both directions, non-gripper dims
passing through untouched, missing/empty/8-D/non-finite actions, missing camera,
the `[openvla]`-extra error path, closed-policy calls, and semantic mismatches
(rotation and position-only embodiments) that must raise `CompatibilityError`.

`.github/workflows/ci.yml` gains a `plugin-openvla` job cloned from
`plugin-xpolicylab`, wired into `ci-ok.needs`; `.github/workflows/release.yml`
gains `publish-openvla` (maintainer must create the `pypi-openvla` environment).

## 7. Risks

- **7B inference is slow on CPU.** Not a correctness issue; the contract layer is
  what this plugin is for. A GPU is only needed for a real rollout.
- **Checkpoint drift.** `predict_action` is upstream API; a breaking change there
  surfaces as a missing-extra/API error, and the stub keeps our side tested.

## 8. Non-goals

Batching, chunked/parallel decoding (that is OpenVLA-OFT), fine-tuning, and
quantized loaders; each is a separate adapter or a parameter on this one.
