"""Policy-facing rollout observations carry a reserved environment step."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from inspect_robots.approver import AutoApprover
from inspect_robots.controller import DefaultController
from inspect_robots.logging.sink import LogSink, NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.rollout import TrialRecord, rollout
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

_SCENE = Scene(id="step-probe", instruction="hold still")
_ACTION_SPACE = Box(
    shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world")
)


class _ProbePolicy:
    """Record every policy-facing observation and emit one action per call."""

    def __init__(self) -> None:
        self.info = PolicyInfo(name="step-probe", action_space=_ACTION_SPACE)
        self.config = PolicyConfig(action_horizon=1)
        self.observations: list[Observation] = []

    def reset(self, scene: Scene) -> None:
        """Clear observations before the trial."""
        self.observations.clear()

    def act(self, observation: Observation) -> ActionChunk:
        """Capture the observation and return a single hold-still action."""
        self.observations.append(observation)
        return ActionChunk(actions=[Action(data=np.zeros(2))])


class _ObservationSink(NullSink):
    """Capture observations delivered to ``log_step``."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Record the logged observation."""
        self.observations.append(observation)


class _ExtraEmbodiment(CubePickEmbodiment):
    """Attach colliding and unrelated extras to every produced observation."""

    def __init__(self) -> None:
        super().__init__()
        self.observations: list[Observation] = []

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Return and record a reset observation carrying embodiment extras."""
        self.observations.clear()
        observation = replace(
            super().reset(scene, seed=seed),
            extra={"env_step": "theirs", "unrelated": 1},
        )
        self.observations.append(observation)
        return observation

    def step(self, action: Action) -> StepResult:
        """Return and record a step observation carrying embodiment extras."""
        result = super().step(action)
        observation = replace(
            result.observation,
            extra={"env_step": "theirs", "unrelated": 1},
        )
        self.observations.append(observation)
        return replace(result, observation=observation)


def _run(
    policy: _ProbePolicy,
    embodiment: CubePickEmbodiment,
    *,
    max_steps: int,
    sink: LogSink | None = None,
) -> TrialRecord:
    return rollout(
        policy,
        embodiment,
        _SCENE,
        max_steps=max_steps,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=sink or NullSink(),
    )


def test_rollout_injects_step_only_into_policy_observations() -> None:
    policy = _ProbePolicy()
    sink = _ObservationSink()

    record = _run(policy, CubePickEmbodiment(), max_steps=4, sink=sink)

    assert [observation.extra["env_step"] for observation in policy.observations] == [0, 1, 2, 3]
    assert len(record.steps) == len(sink.observations) == 4
    assert all("env_step" not in step.observation.extra for step in record.steps)
    assert all("env_step" not in observation.extra for observation in sink.observations)


def test_rollout_step_merge_preserves_extras_and_overwrites_collision() -> None:
    policy = _ProbePolicy()

    _run(policy, _ExtraEmbodiment(), max_steps=3)

    assert [observation.extra["env_step"] for observation in policy.observations] == [0, 1, 2]
    assert all(observation.extra["unrelated"] == 1 for observation in policy.observations)


def test_rollout_step_injection_shares_the_images_mapping() -> None:
    policy = _ProbePolicy()
    embodiment = _ExtraEmbodiment()

    _run(policy, embodiment, max_steps=3)

    assert len(policy.observations) == 3
    assert all(
        policy_observation.images is embodiment_observation.images
        for policy_observation, embodiment_observation in zip(
            policy.observations, embodiment.observations[:3], strict=True
        )
    )
