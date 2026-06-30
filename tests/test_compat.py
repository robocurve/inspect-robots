"""Compatibility checking: spaces, semantics, key remap, scene realizability."""

from __future__ import annotations

import numpy as np
import pytest

from roboinspect import eval
from roboinspect.compat import assert_compatible, check_compatibility
from roboinspect.embodiment import EmbodimentInfo
from roboinspect.errors import CompatibilityError
from roboinspect.mock import CubePickEmbodiment, ScriptedPolicy
from roboinspect.mock.policies import _ACTION_SPACE
from roboinspect.policy import PolicyConfig, PolicyInfo
from roboinspect.scene import Scene, Target
from roboinspect.scorer import success_at_end
from roboinspect.spaces import ActionSemantics, Box, ObservationSpace
from roboinspect.task import Task
from roboinspect.types import Action, ActionChunk, Observation


class _StubPolicy:
    """A configurable policy for compatibility tests."""

    def __init__(self, info: PolicyInfo):
        self.info = info
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(self.info.action_space.dim))])


def test_matching_pair_is_compatible() -> None:
    report = check_compatibility(ScriptedPolicy(), CubePickEmbodiment())
    assert report.ok
    assert report.errors == []


def test_action_dim_mismatch_is_error() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="wide",
            action_space=Box(
                shape=(7,),
                semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
            ),
        )
    )
    report = check_compatibility(policy, CubePickEmbodiment())
    assert not report.ok
    assert any(i.code == "action_dim" for i in report.errors)


def test_control_mode_mismatch_is_error() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="joints",
            action_space=Box(
                shape=(2,),
                semantics=ActionSemantics(control_mode="joint_pos"),
            ),
        )
    )
    report = check_compatibility(policy, CubePickEmbodiment())
    assert any(i.code == "control_mode" for i in report.errors)


def test_missing_required_state_is_error() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="needs-force",
            action_space=_ACTION_SPACE,
            observation_space=ObservationSpace(state_keys=frozenset({"force_torque"})),
        )
    )
    report = check_compatibility(policy, CubePickEmbodiment())
    assert any(i.code == "missing_state" for i in report.errors)


def test_state_remap_resolves_mismatch() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="aliased",
            action_space=_ACTION_SPACE,
            observation_space=ObservationSpace(state_keys=frozenset({"ee"})),
        )
    )
    report = check_compatibility(policy, CubePickEmbodiment(), remap={"ee": "eef_pos"})
    assert report.ok


def test_scene_target_realizability() -> None:
    embodiment = CubePickEmbodiment()
    # Embodiment declares it only supports a "reach" target kind.
    embodiment.info = EmbodimentInfo(
        name=embodiment.info.name,
        action_space=embodiment.info.action_space,
        observation_space=embodiment.info.observation_space,
        control_hz=embodiment.info.control_hz,
        is_simulated=True,
        supported_target_kinds=frozenset({"reach"}),
    )
    task = Task(
        name="t",
        scenes=[Scene(id="s", instruction="x", target=Target(kind="pour"))],
        scorer=success_at_end(),
        max_steps=10,
    )
    report = check_compatibility(ScriptedPolicy(), embodiment, task)
    assert any(i.code == "scene_target" for i in report.errors)


def test_assert_compatible_raises() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="wide",
            action_space=Box(
                shape=(7,),
                semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
            ),
        )
    )
    with pytest.raises(CompatibilityError, match="action_dim"):
        assert_compatible(policy, CubePickEmbodiment())


def test_eval_fails_fast_on_incompatible(tmp_path: object) -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="joints",
            action_space=Box(shape=(2,), semantics=ActionSemantics(control_mode="joint_pos")),
        )
    )
    task = Task(
        name="t",
        scenes=[Scene(id="s", instruction="x")],
        scorer=success_at_end(),
        max_steps=10,
    )
    with pytest.raises(CompatibilityError):
        eval(task, policy, CubePickEmbodiment(), log_dir=str(tmp_path))
