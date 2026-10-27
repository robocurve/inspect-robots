# inspect-robots-openvla

An [Inspect Robots](https://github.com/robocurve/inspect-robots) `Policy`
adapter for **base OpenVLA** (`openvla/openvla-7b`).

Base OpenVLA is the 7-D end-effector action head — this adapter does **not**
cover OpenVLA-OFT (8-D, continuous action chunks), which needs its own adapter.

## Install

```bash
pip install "inspect-robots-openvla[openvla]"
```

The core install (no `[openvla]` extra) is numpy + Pillow only, so
`inspect-robots list policies` and `check_compatibility` work on a machine that
never downloads the 7B checkpoint.

## Use

```bash
inspect-robots run --task my-task --policy openvla_local --embodiment my-robot \
    -P unnorm_key=bridge_orig -P camera=agentview
```

```python
from inspect_robots import eval
from inspect_robots_openvla import OpenVLAPolicy

with OpenVLAPolicy(unnorm_key="bridge_orig") as policy:
    eval("my-task", policy, "my-robot")
```

## Action contract

| dim | meaning |
|---|---|
| 0–2 | end-effector position delta (m, base frame) |
| 3–5 | end-effector rotation delta, Euler XYZ `(roll, pitch, yaw)` (rad) |
| 6 | gripper, binary: `+1` closed / `-1` open |

Upstream OpenVLA emits the RLDS gripper convention (`0` closed, `1` open); the
adapter flips it once at the boundary, so embodiments receive the robosuite-style
`±1` value directly. `PolicyInfo.action_space` publishes
`control_mode="eef_delta_pose"`, `rotation_repr="euler_xyz"`,
`gripper="binary"`, `frame="base"`, which is what `check_compatibility` checks.
Base OpenVLA predicts BridgeData-style deltas, so rotation values are passed
through as Euler angles: an axis-angle embodiment is rejected rather than
silently misread. Non-finite model output raises `PolicyError` before the
gripper is thresholded.

## Instructions

`act()` prompts the model with `observation.instruction` when the observation
carries one, so a scene whose goal changes mid-rollout is followed; the
instruction bound by `reset(scene)` is the fallback for datasets that set it
once.

## Image pipeline

Matches upstream `get_vla_action()`: RGB frame → optional training-time center
crop (`scale=0.9` ⇒ each side keeps `sqrt(0.9)` of its length, then one bilinear
resample straight to 224×224, as upstream's `crop_and_resize` does) → Hugging
Face processor (rescale and normalise). Pass `center_crop=True`
when the checkpoint was trained with that augmentation, e.g. the LIBERO
fine-tunes.

`unnorm_key` must name the normalization statistics of the dataset the
checkpoint was trained on (e.g. `bridge_orig`, `libero_spatial`); it is required
for meaningful action scales.

## Testing without a GPU

```bash
uv sync --all-packages --extra dev
uv run pytest plugins/inspect-robots-openvla/tests
```

The tests inject a stub processor/model and assert the prompt text, the 7-D
flattening, the gripper flip, and the fail-fast paths. No torch, no checkpoint,
no simulator.
