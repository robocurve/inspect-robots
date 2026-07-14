# Authoring an embodiment adapter

Every embodiment adapter (a package registering an `inspect_robots.embodiments`
entry point) declares its spaces through `EmbodimentInfo`. Two consumers make
those declarations load-bearing:

- The CLI's default safety guardrails derive a bounds clamp and a per-step
  delta limit from the action space.
- The LLM agent policy (`inspect-robots-agent`) builds its whole tool surface
  from the spaces at bind time: tool schemas from the bounds and labels, the
  motion strategy from the control mode, the proprioceptive reference from
  the `StateSpec`.

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

The setup wizard checks registered policies and embodiments, while `doctor`
checks its selected embodiment before construction. Both use
`importlib.util.find_spec` without importing the declared top-level modules and
print each remediation command verbatim when a module is missing.

## The conformance kit

Add one test to your adapter repo:

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
| `guardrails` | `DeltaLimitApprover(action_space)` constructs. This catches pose modes with rotation representations that cannot be clamped per dimension (`quat_*`, `axis_angle`, `euler_xyz`; use `none` or `rot6d`). |

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
