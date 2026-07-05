"""Tests for core types and spaces: immutability, validation, semantics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, ActionChunk, Observation, StepResult


def test_core_types_are_frozen() -> None:
    obs = Observation(instruction="go")
    act = Action(data=np.zeros(2))
    chunk = ActionChunk(actions=[act])
    step = StepResult(observation=obs)
    for frozen in (obs, act, chunk, step):
        with pytest.raises(dataclasses.FrozenInstanceError):
            frozen.foo = 1  # type: ignore[union-attr]


def test_action_chunk_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        ActionChunk(actions=[])


def test_action_chunk_len() -> None:
    a = Action(data=np.zeros(2))
    assert len(ActionChunk(actions=[a, a, a])) == 3


def test_box_dim_and_bounds_validation() -> None:
    box = Box(shape=(6,), low=np.full(6, -1.0), high=np.full(6, 1.0))
    assert box.dim == 6
    with pytest.raises(ValueError, match="shape"):
        Box(shape=(6,), low=np.zeros(3))


def test_box_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="low must be elementwise"):
        Box(shape=(2,), low=np.array([0.0, 1.0]), high=np.array([1.0, 0.5]))


def test_action_semantics_defaults() -> None:
    sem = ActionSemantics(control_mode="eef_delta_pose")
    assert sem.rotation_repr == "none"
    assert sem.gripper == "none"
    assert sem.frame == "base"


def test_observation_space_derives_state_keys_from_spec() -> None:
    spec = StateSpec(
        fields=(
            StateField(key="joint_pos", shape=(7,), unit="rad"),
            StateField(key="gripper", shape=(1,), unit="normalized"),
        )
    )
    space = ObservationSpace(
        cameras=(CameraSpec(name="wrist", height=224, width=224),),
        state=spec,
    )
    assert space.state_keys == {"joint_pos", "gripper"}
    assert space.camera_names == {"wrist"}


def test_observation_space_rejects_inconsistent_state_keys() -> None:
    spec = StateSpec(fields=(StateField(key="joint_pos", shape=(7,)),))
    # Consistent duplication is allowed...
    ObservationSpace(state_keys=frozenset({"joint_pos"}), state=spec)
    # ...but a silent disagreement is not.
    with pytest.raises(ValueError, match="inconsistent"):
        ObservationSpace(state_keys=frozenset({"eef_pos"}), state=spec)


def test_task_validation_and_scorer_names() -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Epochs, Task

    scene = Scene(id="s", instruction="x")
    with pytest.raises(ConfigError, match="max_steps"):
        Task(name="t", scenes=[scene], scorer="success_at_end", max_steps=0)
    with pytest.raises(ConfigError, match="Epochs count"):
        Task(name="t", scenes=[scene], scorer="success_at_end", max_steps=5, epochs=0)
    with pytest.raises(ConfigError, match="Epochs count"):
        Epochs(count=0)

    # A scorer registry name resolves to one scorer, never to a sequence of
    # one-character "scorers" (str is a Sequence).
    task = Task(name="t", scenes=[scene], scorer="success_at_end", max_steps=5)
    (scorer,) = task.scorers
    assert scorer.name == "success_at_end"

    # Sequences may mix objects and names.
    from inspect_robots.scorer import episode_length

    mixed = Task(name="t", scenes=[scene], scorer=[episode_length(), "success_at_end"], max_steps=5)
    assert [s.name for s in mixed.scorers] == ["episode_length", "success_at_end"]
