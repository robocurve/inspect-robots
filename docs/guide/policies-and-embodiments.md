# Policies and embodiments

Both are runtime-checkable Protocols — implement the methods on any class (no
inheritance required), or subclass the convenience base classes.

## A policy (VLA)

A [`Policy`][inspect_robots.policy.Policy] maps an observation to an
[`ActionChunk`][inspect_robots.types.ActionChunk]. It declares a
[`PolicyInfo`][inspect_robots.policy.PolicyInfo] (the action space it emits and the
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
                    control_mode="eef_delta_pose", rotation_repr="rot6d", gripper="continuous"
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
frames. Temporal concerns (history, smoothing, ensembling) live in a
[Controller](../guide/concepts.md).

## An embodiment (robot or sim)

An [`Embodiment`][inspect_robots.embodiment.Embodiment] produces observations and executes
actions. It declares an [`EmbodimentInfo`][inspect_robots.embodiment.EmbodimentInfo] with its
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
        # Returns as soon as the command is issued; the framework paces the loop
        # unless this embodiment declares the "self_paced" capability.
        ...

    def close(self) -> None:
        ...
```

## Real-robot vs simulator

The interfaces assume **real-robot reality**: no guaranteed privileged success,
human-in-the-loop reset, wall-clock control. Simulators opt into more via
`capabilities` (`SEEDABLE`, `AUTO_RESET`, `PRIVILEGED_SUCCESS`, `RENDERABLE`, …).
A sim may put privileged success into `StepResult.info` for a scorer to read; a
real robot typically relies on an operator verdict
([`operator_scorer`][inspect_robots.scorer.operator_scorer]) or a learned classifier.

## Compatibility

If the policy's action dimension/semantics or required observations don't match
the embodiment, [`eval`][inspect_robots.eval.eval] raises a
[`CompatibilityError`][inspect_robots.errors.CompatibilityError] before any rollout. Use `remap=` to
alias differing camera/state key names:

```python
eval(task, MyVLA(), MyArm(), remap={"base_rgb": "camera_0"})
```
