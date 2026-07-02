# inspect-robots-isaacsim

An [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (Isaac Sim) **embodiment**
plugin for [Inspect Robots](https://github.com/robocurve/inspect-robots) — the "Inspect AI
for robotics".

Inspect Robots factors a robotics eval into two swappable inputs: a `Policy` (the VLA
"brain") and an `Embodiment` (the "body + world"). This package supplies the
second one, backed by a real Isaac Lab physics simulation, so you can run any
compatible VLA against your Isaac Sim setup.

The default profile is a **7-DoF Franka Panda under joint-position control** with
a binary gripper (action dimension `= num_arm_joints + 1 = 8`).

## Install

This package installs and registers on any machine, but `reset()`/`step()` need
a working **Isaac Lab** environment (NVIDIA Omniverse + GPU) — Isaac Sim is not a
PyPI dependency and is imported lazily.

```bash
# inside the conda/venv that already has isaaclab + isaacsim available
pip install inspect-robots inspect-robots-isaacsim
```

## Use it

The embodiment is discovered through the `inspect_robots.embodiments` entry point, so
it appears in the CLI without any import:

```bash
inspect-robots list embodiments          # -> includes "isaacsim"

inspect-robots run \
  --task my-benchmark \
  --policy my-vla \
  --embodiment isaacsim \
  -E task_id=Isaac-Lift-Cube-Franka-v0 \
  -E headless=true
```

Or programmatically:

```python
from inspect_robots import eval

logs = eval("my-benchmark", "my-vla", "isaacsim")
print(logs[0].results.metrics)
```

Constructing directly (e.g. for a non-Franka arm or extra cameras):

```python
from inspect_robots_isaacsim import IsaacSimEmbodiment

emb = IsaacSimEmbodiment(
    task_id="Isaac-Open-Drawer-Franka-v0",
    num_arm_joints=7,
    cameras=[("base_rgb", 224, 224), ("wrist_rgb", 224, 224)],
    control_hz=30.0,
    headless=True,
)
```

## Compatibility

Inspect Robots fail-fast-checks the `(policy, embodiment)` pair before any rollout. To
run against this embodiment your policy must emit **8-D `joint_pos` actions**
(`control_mode="joint_pos"`, `gripper="binary"`) and require only observation
keys this task provides. The mock `cubepick` policies (2-D `eef_delta_pos`) are
intentionally **incompatible** — bring a Franka-trained VLA.

### Mapping your task

Isaac Lab tasks vary in their observation-dict layout and success signal. The
constructor exposes hooks so you don't edit the adapter:

| Argument | Purpose |
|---|---|
| `obs_group` | top-level obs-dict group to read (default `"policy"`) |
| `image_keys` / `state_keys` | map Inspect Robots keys → your task's raw dict keys |
| `success_info_key` | where the task reports success in `info` (default `"success"`) |
| `num_arm_joints` | arm DoF; action dim is this `+ 1` for the gripper |

## Memory & GPU hygiene

Long unattended evals should not creep in RAM or VRAM. This adapter is built to
hold nothing per step, but a few usage rules keep a full run leak-free:

- **Free the simulator when done.** `eval()` does *not* close the embodiment for
  you, and an open `SimulationApp` holds GPU memory until the process exits. Use
  the embodiment as a context manager (or `close()` in a `finally`):

  ```python
  with IsaacSimEmbodiment() as emb:
      eval("my-bench", "my-vla", emb)
  # GPU + sim torn down here
  ```

  `close()` is idempotent and safe to call before launch or twice.

- **One simulator per process.** Isaac Sim is a hard process singleton; if you
  construct several embodiments the adapter reuses the one live `SimulationApp`
  rather than launching (and leaking) a second.

- **Stream frames to disk for long episodes.** Inspect Robots keeps each step's
  observation — including camera frames — in the per-trial record. Pass
  `store_frames=True` to `eval()` so frames go to disk side-cars instead of RAM.
  (Per-trial records are released after each scene is scored, so there is no
  cross-trial accumulation regardless.)

- **Skip rendering you don't need.** If your scorer is state/oracle based, build
  the embodiment with `cameras=()` to avoid allocating images at all.

The adapter never stores tensors on `self`, and copies observations off Isaac's
reused buffers (`.astype`), so a step loop holds no growing references — a
`tracemalloc` regression test asserts flat RAM over thousands of steps.

## Develop / test

The test suite runs **without Isaac** (it checks spaces, semantics, protocol
conformance, compatibility, and registry wiring):

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT.
