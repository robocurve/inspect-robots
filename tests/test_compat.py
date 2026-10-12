"""Compatibility checking: spaces, semantics, key remap, scene realizability."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.compat import assert_compatible, check_compatibility
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import CompatibilityError
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.mock.policies import _ACTION_SPACE
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation


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


def test_action_shape_mismatch_with_equal_dim_is_error() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="matrix-actions",
            action_space=Box(
                shape=(2, 3),
                semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
            ),
        )
    )
    embodiment = CubePickEmbodiment()
    embodiment.info = replace(
        embodiment.info,
        action_space=Box(
            shape=(6,),
            semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
        ),
    )
    report = check_compatibility(policy, embodiment)
    assert not report.ok
    assert any(i.code == "action_shape" for i in report.errors)


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


def test_embodiment_only_max_step_declaration_is_compatible() -> None:
    policy = _StubPolicy(
        PolicyInfo(
            name="absolute-policy",
            action_space=Box(
                shape=(2,),
                semantics=ActionSemantics(control_mode="joint_pos"),
            ),
        )
    )
    embodiment = CubePickEmbodiment()
    embodiment.info = replace(
        embodiment.info,
        action_space=Box(
            shape=(2,),
            low=np.zeros(2),
            high=np.ones(2),
            semantics=ActionSemantics(control_mode="joint_pos", max_step=(None, 0.1)),
        ),
    )

    assert check_compatibility(policy, embodiment).ok


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


def test_seconds_task_with_valid_control_rate_is_compatible() -> None:
    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="x")],
        scorer=success_at_end(),
        max_seconds=120.0,
    )
    report = check_compatibility(ScriptedPolicy(), CubePickEmbodiment(), task)
    assert not any(i.code == "task_horizon_control_rate" for i in report.errors)


@pytest.mark.parametrize("control_hz", [None, True, 0.0, -1.0, float("nan"), float("inf")])
def test_seconds_task_requires_positive_finite_control_rate(
    control_hz: float | None,
) -> None:
    embodiment = CubePickEmbodiment()
    embodiment.info = replace(embodiment.info, control_hz=control_hz)
    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="x")],
        scorer=success_at_end(),
        max_seconds=120.0,
    )
    report = check_compatibility(ScriptedPolicy(), embodiment, task)
    issue = next(i for i in report.errors if i.code == "task_horizon_control_rate")
    assert "finite positive embodiment rate" in issue.message


def test_seconds_task_rejects_nonfinite_resolved_step_budget() -> None:
    embodiment = CubePickEmbodiment()
    embodiment.info = replace(embodiment.info, control_hz=1e308)
    task = Task(
        name="timed",
        scenes=[Scene(id="s", instruction="x")],
        scorer=success_at_end(),
        max_seconds=1e308,
    )
    report = check_compatibility(ScriptedPolicy(), embodiment, task)
    issue = next(i for i in report.errors if i.code == "task_horizon_control_rate")
    assert "finite step budget" in issue.message


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


def test_action_semantics_unknown_and_mismatches() -> None:
    # 1. Action semantics missing on policy
    pol_no_sem = _StubPolicy(PolicyInfo(name="no-sem", action_space=Box(shape=(3,))))
    emb = CubePickEmbodiment()
    emb.info = replace(
        emb.info,
        action_space=Box(shape=(3,), semantics=ActionSemantics(control_mode="eef_delta_pos")),
    )
    rep = check_compatibility(pol_no_sem, emb)
    assert rep.ok
    assert any(w.code == "action_semantics_unknown" for w in rep.warnings)

    # 2. Rotation representation mismatch
    pol_rot = _StubPolicy(
        PolicyInfo(
            name="rot-quat",
            action_space=Box(
                shape=(3,),
                semantics=ActionSemantics(control_mode="eef_delta_pos", rotation_repr="quat_xyzw"),
            ),
        )
    )
    rep_rot = check_compatibility(pol_rot, emb)
    assert not rep_rot.ok
    assert any(e.code == "rotation_repr" for e in rep_rot.errors)

    # 3. Gripper and frame warnings
    pol_grip = _StubPolicy(
        PolicyInfo(
            name="grip-binary",
            action_space=Box(
                shape=(3,),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pos",
                    rotation_repr="none",
                    gripper="binary",
                    frame="camera",
                ),
            ),
        )
    )
    emb_grip = CubePickEmbodiment()
    emb_grip.info = replace(
        emb_grip.info,
        action_space=Box(
            shape=(3,),
            semantics=ActionSemantics(
                control_mode="eef_delta_pos",
                rotation_repr="none",
                gripper="none",
                frame="base",
            ),
        ),
    )
    rep_grip = check_compatibility(pol_grip, emb_grip)
    assert rep_grip.ok
    assert any(w.code == "gripper" for w in rep_grip.warnings)
    assert any(w.code == "frame" for w in rep_grip.warnings)


def test_control_rate_warning() -> None:
    pol = _StubPolicy(PolicyInfo(name="p", action_space=_ACTION_SPACE, control_hz=20.0))
    emb = CubePickEmbodiment()
    emb.info = replace(emb.info, control_hz=10.0)
    rep = check_compatibility(pol, emb)
    assert rep.ok
    assert any(w.code == "control_rate" for w in rep.warnings)


def test_scene_setup_unsupported_error() -> None:
    emb = CubePickEmbodiment()
    emb.info = replace(emb.info, supported_setups=frozenset({"tabletop"}))
    task = Task(
        name="overhead-task",
        scenes=[Scene(id="s1", instruction="do it", setup="conveyor")],
        scorer=success_at_end(),
        max_steps=5,
    )
    rep = check_compatibility(ScriptedPolicy(), emb, task)
    assert not rep.ok
    assert any(e.code == "scene_setup" for e in rep.errors)


def test_assert_compatible_returns_report_when_ok() -> None:
    rep = assert_compatible(ScriptedPolicy(), CubePickEmbodiment())
    assert rep.ok
    rep.raise_for_errors()
