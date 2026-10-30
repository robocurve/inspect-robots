# Policies and embodiments

Both are runtime-checkable Protocols: implement the methods on any class (no
inheritance required), or subclass the convenience base classes.

## A policy (VLA)

A [`Policy`](../../src/inspect_robots/policy.py) maps an observation to an
[`ActionChunk`](../../src/inspect_robots/types.py). It declares a
[`PolicyInfo`](../../src/inspect_robots/policy.py) (the action space it emits and the
observations it requires) used for compatibility checking.

```python
import numpy as np
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation


class MyVLA:
    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="my-vla",
            action_space=Box(
                shape=(7,),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pose", rotation_repr="euler_xyz", gripper="continuous"
                ),
            ),
            observation_space=ObservationSpace(
                state_keys=frozenset({"eef_pose", "gripper"}),
            ),
        )
        self.config = PolicyConfig(action_horizon=16)

    def reset(self, scene: Scene) -> None:
        ...  # clear any per-episode state

    def act(self, observation: Observation) -> ActionChunk:
        # The policy owns model-specific preprocessing (resize/normalize/history).
        chunk = my_model_infer(observation)        # -> (H, 7) array
        actions = [Action(data=a) for a in chunk]
        return ActionChunk(actions=actions, inference_latency_s=...)
```

The policy owns model-specific spatial preprocessing; the embodiment emits raw
frames. Temporal concerns (history, smoothing, ensembling) live in an
[ActionManager](../guide/concepts.md).

### Ending a trial

A policy's action space includes terminal actions: choices that end the trial,
such as declaring the task done or giving up. Declare the kinds the policy can
emit in `PolicyInfo.terminal_actions`, then return a chunk whose `terminal` is a
[`Termination`](../../src/inspect_robots/types.py) with that kind and the
policy's stated reason:

```python
from inspect_robots.types import DONE, GIVE_UP, Termination

self.info = PolicyInfo(name="my-vla", action_space=..., terminal_actions=frozenset({DONE, GIVE_UP}))

# Finish the last motion, then end the trial.
return ActionChunk(actions=actions, terminal=Termination(DONE, "all three bowls are stacked"))

# End the trial without moving.
return ActionChunk(actions=[], terminal=Termination(GIVE_UP, "the lid is out of reach"))
```

The trial ends after the last action the action manager executes from that
chunk. The kind becomes the record's termination reason and the reason its
termination detail; a `policy_terminal` event marks the step. The embodiment's
own termination (a success sensor, a fault) still wins when both happen on the
same step, and ending a trial is never a claim of success: scoring stays with
the task's scorers. A kind the policy did not declare fails the trial with a
`PolicyError`, so a typo cannot pass for a refusal.

## An embodiment (robot or sim)

An [`Embodiment`](../../src/inspect_robots/embodiment.py) produces observations and executes
actions. It declares an [`EmbodimentInfo`](../../src/inspect_robots/embodiment.py) with its
spaces, native control rate, and opt-in capability flags.

```python
from inspect_robots.embodiment import EmbodimentInfo, PRIVILEGED_SUCCESS, SEEDABLE
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box, CameraSpec, ObservationSpace
from inspect_robots.types import Action, Observation, StepResult


class MyArm:
    def __init__(self) -> None:
        self.info = EmbodimentInfo(
            name="my-arm",
            action_space=Box(shape=(7,), semantics=...),
            observation_space=ObservationSpace(
                cameras=(CameraSpec("base_rgb", 224, 224), CameraSpec("wrist_rgb", 224, 224)),
                state_keys=frozenset({"eef_pose", "gripper"}),
            ),
            control_hz=20.0,
            is_simulated=False,
            capabilities=frozenset({SEEDABLE}),  # real arms rarely have PRIVILEGED_SUCCESS
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        # On real hardware this may drive to home and block on operator confirmation.
        ...

    def step(self, action: Action) -> StepResult:
        # The rollout applies no wall-clock pacing of its own. If this embodiment
        # needs real-time cadence, pace it here and declare "self_paced".
        ...

    def close(self) -> None:
        ...
```

When a human operator ends an episode without giving a verdict (an end-episode
keypress), terminate with `termination_reason=inspect_robots.OPERATOR_END`
(`"operator_end"`). Attended runs (interactive terminal, no `--no-prompt`) then
get the operator prompt — `did the robot succeed? [y/n/partial/skip]` plus an
optional grader note — for exactly those trials, on registered tasks and ad-hoc
runs alike. Do **not** ask your own verdict prompt in the embodiment and
terminate with a definitive `"success"`/`"failure"`: that suppresses the
framework prompt and locks the run out of partial/skip verdicts and notes.

Lifecycle: [`eval`](../../src/inspect_robots/eval.py) closes what it resolves. An
embodiment looked up by registry name is closed when the run finishes (even
on a halt). If you construct the embodiment object yourself, you own it: call
`close()` yourself when you are done.

### The evaluation interface

`reset()` and `step()` are the embodiment's policy interface. Between trials,
[`eval`](../../src/inspect_robots/eval.py) also drives an evaluation interface:
four optional methods, each used only when the embodiment defines it.

| Method | Called | Effect |
|---|---|---|
| `health() -> HealthReport` | Before each trial | A failed check stops the run before anything moves. |
| `prepare_trial(scene) -> Readiness` | Before each trial, after `health()` | Not ready stops the run, with the stated reason as its error. |
| `home()` | Before each trial's `reset()` | Moves to the start pose. Skip it if `reset()` already homes. |
| `park()` | After each trial, once it is scored | Moves to a safe rest pose and holds it, so the scene can be reset by hand. |

```python
from inspect_robots import HealthCheck, HealthReport, Readiness


class MyArm:
    ...

    def health(self) -> HealthReport:
        return HealthReport(
            checks=(
                HealthCheck("motors", ok=self._bus_alive(), detail="CAN bus down"),
                HealthCheck("wrist camera", ok=self._frame_age_s() < 1.0, detail="stale frames"),
            )
        )

    def prepare_trial(self, scene: Scene) -> Readiness:
        if scene.setup == "open_drawer" and not self._has_drawer:
            return Readiness(ready=False, reason="this rig has no drawer")
        return Readiness(ready=True)

    def park(self) -> None:
        self._ramp_to(self._rest_pose)  # hold the pose; close() releases torque
```

[`HealthCheck`](../../src/inspect_robots/embodiment.py),
[`HealthReport`](../../src/inspect_robots/embodiment.py) and
[`Readiness`](../../src/inspect_robots/embodiment.py) are exported from
`inspect_robots`. Any exception from `health()`, `prepare_trial()` or `home()`
also stops the run before the trial starts. `park()` runs after every trial
whatever its outcome, except a run cancelled with Ctrl-C, so it must be safe to
call after a fault; it may decline by returning without moving. A failed park
after a clean trial stops the run. The log lists the methods an embodiment
offers under `embodiment_info["evaluation_interface"]`.

Set `EmbodimentInfo.morphology` to name the physical platform. Two embodiments
on the same platform must offer different policy interfaces (for example joint
targets and end-effector targets on the same arms). The conformance kit checks
this; see [Authoring an embodiment adapter](adapters.md#the-conformance-kit).

## Real-robot vs simulator

The interfaces assume real-robot reality: no guaranteed privileged success,
human-in-the-loop reset, wall-clock control. Simulators opt into more via
`capabilities` (`SEEDABLE`, `AUTO_RESET`, `PRIVILEGED_SUCCESS`, `RENDERABLE`, …).
A sim may put privileged success into `StepResult.info` for a scorer to read; a
real robot typically relies on an operator verdict
([`operator_scorer`](../../src/inspect_robots/scorer.py)) or a learned classifier.

## Compatibility

If the policy's action dimension/semantics or required observations don't match
the embodiment, [`eval`](../../src/inspect_robots/eval.py) raises a
[`CompatibilityError`](../../src/inspect_robots/errors.py) before any rollout. Use `remap=` to
alias differing camera/state key names:

```python
eval(task, MyVLA(), MyArm(), remap={"base_rgb": "camera_0"})
```
