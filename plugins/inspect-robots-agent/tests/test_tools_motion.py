"""Tool surface + control-mode-aware motion synthesis (plan 0008 §4b/§4c)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Observation
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import ToolsetError, build_toolset

_ARM_LABELS = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*[f"j{i}" for i in range(6)], "gripper")
)


def _bimanual_space() -> Box:
    return Box(
        shape=(14,),
        low=np.array([-np.pi] * 6 + [0.0] + [-np.pi] * 6 + [0.0]),
        high=np.array([np.pi] * 6 + [1.0] + [np.pi] * 6 + [1.0]),
        semantics=ActionSemantics("joint_pos", dim_labels=_ARM_LABELS),
    )


def _bimanual_obs_space() -> ObservationSpace:
    return ObservationSpace(state=StateSpec(fields=(StateField(key="joint_pos", shape=(14,)),)))


def _delta_space() -> Box:
    return Box(
        shape=(2,),
        low=np.array([-0.1, -0.1]),
        high=np.array([0.1, 0.1]),
        semantics=ActionSemantics("eef_delta_pos", frame="world"),
    )


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=json.dumps(arguments))


def _obs(state: dict[str, np.ndarray] | None = None) -> Observation:
    return Observation(state=state or {"joint_pos": np.zeros(14)})


# --- bind-time validation ------------------------------------------------------


def test_build_refuses_unsupported_configurations() -> None:
    no_sem = Box(shape=(2,), low=np.zeros(2), high=np.ones(2))
    with pytest.raises(ToolsetError, match="semantics"):
        build_toolset(no_sem, _bimanual_obs_space(), control_hz=10.0)

    vel = Box(shape=(2,), semantics=ActionSemantics("joint_vel"))
    with pytest.raises(ToolsetError, match="joint_vel"):
        build_toolset(vel, _bimanual_obs_space(), control_hz=10.0)

    quat = Box(shape=(7,), semantics=ActionSemantics("eef_abs_pose", rotation_repr="quat_wxyz"))
    with pytest.raises(ToolsetError, match="rotation_repr"):
        build_toolset(quat, _bimanual_obs_space(), control_hz=10.0)


def test_absolute_mode_requires_exactly_one_aligned_state_field() -> None:
    space = _bimanual_space()
    with pytest.raises(ToolsetError, match="StateSpec"):
        build_toolset(space, ObservationSpace(), control_hz=10.0)  # no StateSpec at all

    none_match = ObservationSpace(state=StateSpec(fields=(StateField(key="eef", shape=(3,)),)))
    with pytest.raises(ToolsetError, match="exactly one"):
        build_toolset(space, none_match, control_hz=10.0)

    two_match = ObservationSpace(
        state=StateSpec(fields=(StateField(key="a", shape=(14,)), StateField(key="b", shape=(14,))))
    )
    with pytest.raises(ToolsetError, match="exactly one"):
        build_toolset(space, two_match, control_hz=10.0)


# --- schemas -------------------------------------------------------------------


def test_schemas_match_control_mode_and_embed_labels() -> None:
    absolute = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    names = [t["function"]["name"] for t in absolute.schemas()]
    assert names == ["move_joints", "done", "give_up"]
    move_schema = json.dumps(absolute.schemas()[0])
    assert "left_j0" in move_schema and "right_gripper" in move_schema

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    names = [t["function"]["name"] for t in displacement.schemas()]
    assert names == ["move_by", "done", "give_up"]


# --- absolute-mode synthesis ---------------------------------------------------


def test_move_joints_interpolates_named_partial_targets() -> None:
    toolset = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"right_j2": 0.4, "right_gripper": 1.0}, duration_s=1.0),
        _obs(),
    )
    assert result.error is None and result.chunk is not None
    chunk = result.chunk
    assert len(chunk.actions) == 10
    assert chunk.control_hz == 10.0
    final = np.asarray(chunk.actions[-1].data)
    assert final[9] == pytest.approx(0.4)  # right_j2 is dim 7 + 2
    assert final[13] == pytest.approx(1.0)  # right_gripper
    assert np.allclose(np.delete(final, [9, 13]), 0.0)  # unnamed dims hold
    mid = np.asarray(chunk.actions[4].data)
    assert mid[9] == pytest.approx(0.2)  # linear halfway


def test_move_joints_index_fallback_without_labels() -> None:
    space = Box(
        shape=(2,), low=np.zeros(2), high=np.ones(2), semantics=ActionSemantics("joint_pos")
    )
    obs_space = ObservationSpace(state=StateSpec(fields=(StateField(key="q", shape=(2,)),)))
    toolset = build_toolset(space, obs_space, control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"1": 0.6}, duration_s=0.5),
        Observation(state={"q": np.array([0.2, 0.2])}),
    )
    assert result.error is None and result.chunk is not None
    final = np.asarray(result.chunk.actions[-1].data)
    assert final[0] == pytest.approx(0.2) and final[1] == pytest.approx(0.6)


# --- displacement-mode synthesis ------------------------------------------------


def test_move_by_splits_displacement_across_steps() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 0.1}, duration_s=0.5), _obs({}))
    assert result.error is None and result.chunk is not None
    actions = [np.asarray(a.data) for a in result.chunk.actions]
    assert len(actions) == 5
    for a in actions:
        assert a[0] == pytest.approx(0.02) and a[1] == 0.0


# --- done / give_up -------------------------------------------------------------


def test_done_emits_hold_still_with_request_stop() -> None:
    toolset = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    state = np.full(14, 0.3)
    result = toolset.execute(_call("done", summary="fork placed"), _obs({"joint_pos": state}))
    assert result.error is None and result.chunk is not None
    (action,) = result.chunk.actions
    assert np.allclose(action.data, state)  # absolute hold = repeat current state
    assert action.meta["request_stop"] is True
    assert action.meta["stop_reason"] == "done"

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = displacement.execute(_call("give_up", reason="cannot see the cube"), _obs({}))
    assert result.chunk is not None
    (action,) = result.chunk.actions
    assert np.allclose(action.data, 0.0)  # displacement hold = zeros
    assert action.meta["stop_reason"] == "give_up"


# --- structured tool errors (fed back to the LLM, never exceptions) -------------


def test_tool_errors_are_messages_not_exceptions() -> None:
    toolset = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    cases = [
        _call("move_joints", targets={"left_elbow": 0.1}, duration_s=1.0),  # unknown label
        _call("move_joints", targets={"left_j0": float("nan")}, duration_s=1.0),
        _call("move_joints", targets={"left_j0": "fast"}, duration_s=1.0),  # non-numeric
        _call("move_joints", targets={}, duration_s=1.0),  # nothing to do
        _call("move_joints", targets={"left_j0": 0.1}, duration_s=0.0),  # bad duration
        _call("move_joints", targets={"left_j0": 0.1}, duration_s=60.0),  # over the cap
        _call("nonexistent_tool", x=1),
        ToolCall(id="c", name="move_joints", arguments="{not json"),
    ]
    for call in cases:
        result = toolset.execute(call, _obs())
        assert result.chunk is None and result.error, f"expected error for {call}"
    # Errors name the offender so the model can self-correct.
    unknown = toolset.execute(cases[0], _obs())
    assert unknown.error is not None and "left_elbow" in unknown.error


def test_control_hz_none_falls_back_to_default() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=None)
    result = toolset.execute(_call("move_by", deltas={"0": 0.05}, duration_s=1.0), _obs({}))
    assert result.chunk is not None
    assert len(result.chunk.actions) == 10  # 10 Hz fallback
    assert result.chunk.control_hz is None  # defer to the embodiment's native rate
