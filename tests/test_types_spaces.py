"""Tests for core types and spaces: immutability, validation, semantics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from roboinspect.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from roboinspect.types import Action, ActionChunk, Observation, StepResult


def test_core_types_are_frozen() -> None:
    obs = Observation(instruction="go")
    act = Action(data=np.zeros(2))
    chunk = ActionChunk(actions=[act])
    step = StepResult(observation=obs)
    for frozen in (obs, act, chunk, step):
        with pytest.raises(dataclasses.FrozenInstanceError):
            frozen.foo = 1  # type: ignore[attr-defined]


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
