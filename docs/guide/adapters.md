# Authoring an embodiment adapter

Every embodiment adapter (a package registering an `inspect_robots.embodiments`
entry point) declares its spaces through `EmbodimentInfo`. Two consumers make
those declarations load-bearing:

- The CLI's default safety guardrails derive a bounds clamp and a per-step
  delta limit from the action space.
- The LLM agent policy (`inspect-robots-agent`) builds its whole tool surface
  from the spaces at bind time: tool schemas from the bounds and labels, the
  motion strategy from the control mode, the proprioceptive reference from
  the `StateSpec`. It also injects `EmbodimentInfo.docs` (free-form operating
  notes: joint layout, positive directions, gripper polarity) into its system
  prompt, so an adapter that ships good notes measurably helps LLM policies.

An adapter with missing or dishonest declarations silently degrades both.
This page is the contract; the conformance kit makes most of it mechanical.

## Declare runtime requirements

Declare imports that cannot be expressed by package metadata on the registered
component factory:

```python
RUNTIME_REQUIREMENTS = {
    "i2rt": 'uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"',
    "cv2": 'uv pip install "inspect-robots-yam[cameras]"',
}
```

The setup wizard checks the configured policy and embodiment when they are
registered, while `doctor` checks its selected embodiment before construction. Both use
`importlib.util.find_spec` without importing the declared top-level modules and
print each remediation command verbatim when a module is missing.

## Declare device slots

Declare device-shaped constructor arguments on the registered embodiment
factory. Import [`DeviceSlot`](/api/#inspect_robots.conformance.DeviceSlot)
from the `inspect_robots.conformance` submodule:

```python
from inspect_robots.conformance import DeviceSlot

DEVICE_SLOTS = (
    DeviceSlot(
        arg="left_channel",
        kind="can",
        label="left arm CAN channel",
        group="arms",
    ),
    DeviceSlot(
        arg="right_channel",
        kind="can",
        label="right arm CAN channel",
        group="arms",
    ),
    DeviceSlot(
        arg="top_cam_device",
        kind="v4l2",
        label="top camera",
        group="cameras",
    ),
    DeviceSlot(
        arg="left_cam_device",
        kind="v4l2",
        label="left camera",
        group="cameras",
    ),
    DeviceSlot(
        arg="right_cam_device",
        kind="v4l2",
        label="right camera",
        group="cameras",
    ),
)
```

These declarations also feed a run-time advisory device claim. If two evals
name the same device, the second fails at startup instead of double-driving the
hardware. The claim is `flock`-based and vanishes with the process. Claims are
per-user, so they do not guard against two different users driving one rig.
Camera and serial paths count as the same device after resolving symlinks,
while CAN interface names are compared verbatim, so `can0` in one config and a
udev-pinned alias for the same adapter in another do not collide. On hosts
without `XDG_RUNTIME_DIR`, the fallback lock directory lives under the
world-writable temporary directory, and the guard refuses lock directories it
does not own. The guard is a safety net against your own concurrent evals, not
a security boundary.

The recognized kinds are `v4l2` for stable camera paths, `can` for SocketCAN
interface names, and `serial` for absolute `/dev/serial/by-id` paths. The setup
wizard probes and interviews slots in declaration order, then writes each
selection to its `arg` key under `[embodiment.args]`. Slots with the same
non-`None` `group` are all-or-none. Ungrouped slots remain independent.

## Declare option slots

Declare boolean behavior toggles on the registered embodiment factory. Import
[`OptionSlot`](/api/#inspect_robots.conformance.OptionSlot) from the
`inspect_robots.conformance` submodule:

```python
from inspect_robots.conformance import OptionSlot

OPTION_SLOTS = (
    OptionSlot(
        arg="auto_start",
        label="Skip the operator start prompts (auto_start)",
    ),
)
```

Each slot is one yes/no question in the setup wizard. Its `arg` is the
`[embodiment.args]` key written as `true` or `false`. On re-runs, the carried
config value is the suggested answer. A declaration is skipped when its `arg`
collides with a device slot, a camera key, or an earlier option declaration.

## Declare number slots

Declare finite numeric settings on the registered embodiment factory. Import
[`NumberSlot`](/api/#inspect_robots.conformance.NumberSlot) from the
`inspect_robots.conformance` submodule:

```python
from inspect_robots.conformance import NumberSlot

NUMBER_SLOTS = (
    NumberSlot(
        arg="motor_temp_limit",
        label="Motor temperature limit (degrees C)",
        default=70,
        minimum=1,
        allow_none=True,
    ),
)
```

Each slot writes one `[embodiment.args]` value. Bounds are inclusive, and an
omitted bound leaves that side unbounded. Set `allow_none=True` to accept
`none` or `null` as a disabled value. On re-runs, a valid carried config value
is the suggestion; otherwise the declared default is used. Number slots are
asked after option slots. A declaration is skipped when its `arg` collides
with a device slot, a camera key, an option slot, or an earlier number slot.

Importing `NumberSlot` requires the inspect-robots core release that provides
it. On an older core, that import fails while the plugin entry point loads and
drops the whole plugin, rather than silently ignoring only `NUMBER_SLOTS`.
Plugins that adopt number slots must raise their minimum inspect-robots
version accordingly.

## The conformance kit

Add one test built on
[`assert_embodiment_conformant`](/api/#inspect_robots.conformance.assert_embodiment_conformant)
to your adapter repo:

```python
from inspect_robots.conformance import assert_embodiment_conformant

def test_embodiment_is_conformant() -> None:
    assert_embodiment_conformant(MyEmbodiment().info)
```

Users can audit an installed adapter the same way:

```bash
inspect-robots doctor --embodiment my_arms
```

Both run the same checks, and neither touches hardware (keep your
constructor hardware-free; connect in `reset()`).

### What the kit checks (errors)

| Code | Requirement |
|------|-------------|
| `semantics` | The action `Box` carries `ActionSemantics`. Guardrails and the agent cannot tell absolute targets from displacements without it. |
| `bounds` | Finite `low`/`high` on every dim. Without them the bounds clamp is skipped and no default delta limit can be derived. |
| `dim_labels` | Every dim is named (`("left_j0", ..., "right_gripper")`), uniquely. The agent moves joints by these names. |
| `state_alignment` | Absolute-target modes (`joint_pos`, `eef_abs_pose`) declare exactly one `StateSpec` field with `shape == (action_dim,)`: the proprioceptive reference the agent interpolates from. |
| `guardrails` | `DeltaLimitApprover(action_space)` constructs. This catches absolute pose modes (`eef_abs_pose`) whose rotation representation cannot be clamped per dimension (`quat_*`, `axis_angle`, `euler_xyz`; use `none` or `rot6d`), and displacement pose modes (`eef_delta_pose`) whose rotation representation's identity is not the zero vector (`quat_*`, `rot6d`; use `none`, `axis_angle`, or `euler_xyz`). |

### Warnings

- `control_hz` undeclared: agent motion durations fall back to 10 Hz step
  counting.
- Zero-width dims (`low == high`): nothing can be commanded there.

## What the kit cannot check

Conformance proves your adapter is guardrail-ready and agent-ready. It cannot
prove the declarations are honest. Verify these by hand:

1. **Control mode matches `step()` behavior.** If `step()` adds the action to
   the current position, declare `joint_delta` (or `eef_delta_*`), never
   `joint_pos`. Misdeclaring sends the delta limiter and the agent's motion
   layer down the absolute branch: "hold still" becomes "move by the current
   pose".
2. **Displacement bounds are per-step-sized.** In delta modes the declared
   box is the per-step displacement limit, not the absolute joint limits.
   Reusing absolute limits derives uselessly large swing limits, and an
   asymmetric absolute box (like a `[0, 1]` gripper) clamps one direction of
   motion to zero. Keep an absolute-limit clamp on the summed command inside
   the embodiment as a backstop.
3. **Policy and embodiment declare in lockstep.** If a config flag changes
   the control mode, both sides must build their semantics from the same
   config. A mismatch is a hard compatibility error (good: it fails before
   motion).
4. **Hold behavior between chunks.** Slow policies (VLA servers, LLM agents)
   leave seconds between action chunks. Verify on the rig that motors hold
   position in your configured mode before any unattended run, and keep a
   hand on the e-stop the first time.

## Conventions worth copying

The reference adapters (`inspect-robots-yam`, `plugins/inspect-robots-isaacsim`)
share a shape worth reusing: hardware access behind injected seams (driver
factory, camera reader, clock) so the full suite runs in CI; scalar-only
constructor kwargs so `-E key=value` works from the CLI; a hard safety clamp
inside `step()` independent of any approver; 100% coverage with the real
drivers behind `pragma: no cover` seams; and a preflight or `doctor` run
documented as the first step on new hardware.
