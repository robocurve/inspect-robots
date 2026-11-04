# 0087: Motor temperature line, `sleep` tool and wall clock for the agent policy

Status: implemented. Companion to inspect-robots-yam plan 0035.

## Problem

Long trials on real arms can end with outcome `overheat` when a motor reaches
the embodiment's thermal limit. The yam guardrail parks the arms safely, but
the policy had no way to see the temperature rising, no way to wait, and no
sense of wall-clock time. A typical failure: an agent holds one arm at full
reach for dozens of calls while working on a stiff part, and the trial ends
on the thermal limit rather than on the task.

## Change

### Temperature line in every observation

Embodiments may put `extra["motor_temps"]` on an observation with the same
shape as `motion_report`: a mapping whose `lines` are plain-language strings
for the policy (the yam plugin writes the hottest motor per arm against the
limit, plus a warning line inside the warning margin telling the policy to
move the arm back to its start pose and sleep). The agent renders those
lines after the motion report, tolerating absent or malformed data exactly
as `_motion_report_lines` does.

### `sleep` tool

Offered on every control surface. Arguments: `seconds` (at most 180) and the
usual `note`. The toolset holds the current setpoint for that long in one
stationary chunk (the last commanded target on absolute surfaces, so a
gripper squeeze stays squeezed; zero deltas on displacement surfaces).
Nothing moves and no LLM call is spent. The next turn's observation says how
long it slept and, when the embodiment reports temperatures, the hottest
motors before and after.

No parking logic lives in the harness. Moving the arm to a resting pose is
the model's own `move_to`, guided by the embodiment docs (the start pose it
saw at the first observation is the lowest-load pose). This is deliberate:
a primitive the model composes, not a policy baked into the tool.

### Wall clock

Every observation carries `Wall clock: N s since the trial began.` after the
step budget, from a policy-side monotonic clock set at `reset()`. The step
budget measures robot motion; the wall clock lets the model pace LLM time
and sleeps against real time.

## Out of scope

Automatic cooling and per-motor currents. The system prompt stays as is;
the tool description and the embodiment docs carry the guidance.
