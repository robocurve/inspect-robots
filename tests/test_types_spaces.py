"""Tests for core types and spaces: immutability, validation, semantics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.spaces import (
    ABSOLUTE_CONTROL_MODES,
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, ActionChunk, Observation, StepResult


def test_embodiment_info_docs_preserve_frozen_value_semantics() -> None:
    action_space = Box(shape=(1,))
    observation_space = ObservationSpace()
    info = EmbodimentInfo(
        name="test",
        action_space=action_space,
        observation_space=observation_space,
    )
    same = EmbodimentInfo(
        name="test",
        action_space=action_space,
        observation_space=observation_space,
    )

    assert info.docs is None
    assert info == same
    assert hash(info) == hash(same)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.docs = "changed"  # type: ignore[misc]


def test_cubepick_publishes_operating_notes() -> None:
    docs = CubePickEmbodiment().info.docs
    assert docs is not None
    assert docs.strip()


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
    assert sem.max_step is None
    assert {"joint_pos", "eef_abs_pose"} == ABSOLUTE_CONTROL_MODES


def test_action_semantics_joint_delta_and_dim_labels() -> None:
    sem = ActionSemantics(control_mode="joint_delta", dim_labels=("left_j0", "left_gripper"))
    assert sem.dim_labels == ("left_j0", "left_gripper")
    # Default stays None so unlabeled spaces keep working.
    assert ActionSemantics(control_mode="joint_pos").dim_labels is None


def test_box_validates_dim_labels_length() -> None:
    labeled = ActionSemantics(control_mode="joint_pos", dim_labels=("a", "b", "c"))
    with pytest.raises(ValueError, match="dim_labels"):
        Box(shape=(2,), semantics=labeled)
    assert Box(shape=(3,), semantics=labeled).dim == 3
    # No labels: any dim is fine.
    Box(shape=(2,), semantics=ActionSemantics(control_mode="joint_pos"))


def test_action_semantics_max_step_round_trips_with_value_semantics() -> None:
    sem = ActionSemantics(
        control_mode="joint_pos",
        dim_labels=("joint", "gripper"),
        max_step=(None, 0.1),
    )
    same = ActionSemantics(
        control_mode="joint_pos",
        dim_labels=("joint", "gripper"),
        max_step=(None, 0.1),
    )
    different = dataclasses.replace(sem, max_step=(None, 0.2))
    box = Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=sem,
    )

    assert box.semantics is sem
    assert sem.max_step == (None, 0.1)
    assert sem == same
    assert hash(sem) == hash(same)
    assert sem != different


def test_box_validates_max_step_length() -> None:
    semantics = ActionSemantics(control_mode="joint_pos", max_step=(0.1,))
    with pytest.raises(ValueError, match=r"max_step has 1 entries.*2 dimensions"):
        Box(shape=(2,), semantics=semantics)
    assert Box(shape=(1,), semantics=semantics).dim == 1


@pytest.mark.parametrize("entry", [0.0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_action_semantics_rejects_invalid_max_step_entries(entry: float) -> None:
    with pytest.raises(ValueError, match="max_step entries must be finite and > 0"):
        ActionSemantics(control_mode="joint_pos", max_step=(entry,))


def test_action_semantics_allows_none_max_step_entries() -> None:
    assert ActionSemantics(control_mode="joint_pos", max_step=(None, 0.1)).max_step == (
        None,
        0.1,
    )
    assert ActionSemantics(control_mode="joint_delta", max_step=(None,)).max_step == (None,)


def test_action_semantics_rejects_declared_step_for_non_absolute_mode() -> None:
    with pytest.raises(
        ValueError,
        match=r"displacement boxes already declare per-step limits via low/high",
    ):
        ActionSemantics(control_mode="joint_delta", max_step=(None, 0.1))


def test_box_rejects_max_step_on_pinned_dimension() -> None:
    semantics = ActionSemantics(control_mode="joint_pos", max_step=(None, 0.1))
    with pytest.raises(ValueError, match=r"pinned dimension.*low == high"):
        Box(
            shape=(2,),
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 0.0]),
            semantics=semantics,
        )


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


def test_task_envelope_is_a_frozen_view_of_the_horizon() -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task, TaskEnvelope

    scene = Scene(id="s", instruction="x")
    task = Task(name="t", scenes=[scene], scorer="success_at_end", max_steps=80)
    assert task.envelope == TaskEnvelope(name="t", max_steps=80)
    assert task.resolve_envelope(None) == task.envelope
    with pytest.raises(AttributeError):
        task.envelope.max_steps = 81  # type: ignore[misc]

    seconds_task = Task(name="timed", scenes=[scene], scorer="success_at_end", max_seconds=1.01)
    assert seconds_task.resolve_envelope(10.0) == TaskEnvelope(name="timed", max_steps=11)
    with pytest.raises(ConfigError, match="requires an embodiment control_hz"):
        _ = seconds_task.envelope


@pytest.mark.parametrize("max_steps", [True, 0, -1])
def test_task_rejects_invalid_steps_horizon(max_steps: int) -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task

    with pytest.raises(ConfigError, match="max_steps must be >= 1"):
        Task(
            name="t",
            scenes=[Scene(id="s", instruction="x")],
            scorer="success_at_end",
            max_steps=max_steps,
        )


@pytest.mark.parametrize("max_seconds", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_task_rejects_invalid_seconds_horizon(max_seconds: float) -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task

    with pytest.raises(ConfigError, match="max_seconds must be finite and > 0"):
        Task(
            name="timed",
            scenes=[Scene(id="s", instruction="x")],
            scorer="success_at_end",
            max_seconds=max_seconds,
        )


@pytest.mark.parametrize("control_hz", [None, True, 0.0, -1.0, float("nan"), float("inf")])
def test_seconds_horizon_rejects_invalid_control_rate(control_hz: float | None) -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task

    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="x")],
        scorer="success_at_end",
        max_seconds=120.0,
    )
    with pytest.raises(ConfigError, match="control_hz"):
        task.resolve_envelope(control_hz)


def test_seconds_horizon_rejects_nonfinite_resolved_budget() -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task

    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="x")],
        scorer="success_at_end",
        max_seconds=1e308,
    )
    with pytest.raises(ConfigError, match="finite step budget"):
        task.resolve_envelope(1e308)


def test_seconds_horizon_underflow_still_resolves_to_one_step() -> None:
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task, TaskEnvelope

    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="x")],
        scorer="success_at_end",
        max_seconds=5e-324,
    )
    assert task.resolve_envelope(0.5) == TaskEnvelope(name="timed", max_steps=1)


def test_task_validation_and_scorer_names() -> None:
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene
    from inspect_robots.task import Epochs, Task

    scene = Scene(id="s", instruction="x")
    with pytest.raises(ConfigError, match="exactly one"):
        Task(name="t", scenes=[scene], scorer="success_at_end")
    with pytest.raises(ConfigError, match="exactly one"):
        Task(
            name="t",
            scenes=[scene],
            scorer="success_at_end",
            max_steps=5,
            max_seconds=1.0,
        )
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


def test_operator_end_constant_is_public_vocabulary() -> None:
    import inspect_robots
    from inspect_robots.types import OPERATOR_END

    assert OPERATOR_END == "operator_end"
    assert inspect_robots.OPERATOR_END is OPERATOR_END
