"""Tool surface + speed-limited motion synthesis (plan 0014)."""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Observation
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import ToolResult, ToolsetError, build_toolset

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


def _absolute_space(
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    labels: tuple[str, ...] | None = ("joint",),
) -> Box:
    return Box(
        shape=(1,),
        low=np.array([-1.0]) if low is None else low,
        high=np.array([1.0]) if high is None else high,
        semantics=ActionSemantics("joint_pos", dim_labels=labels),
    )


def _absolute_obs_space(dim: int = 1, key: str = "q") -> ObservationSpace:
    return ObservationSpace(state=StateSpec(fields=(StateField(key=key, shape=(dim,)),)))


def _delta_space(
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> Box:
    return Box(
        shape=(2,),
        low=np.array([-0.1, -0.1]) if low is None else low,
        high=np.array([0.1, 0.1]) if high is None else high,
        semantics=ActionSemantics("eef_delta_pos", frame="world"),
    )


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=json.dumps(arguments))


def _obs(state: dict[str, np.ndarray] | None = None) -> Observation:
    return Observation(state={"q": np.array([0.0])} if state is None else state)


def _execute_absolute(
    target: float,
    current: float = 0.0,
    *,
    control_hz: float | None = 10.0,
    max_speed_frac: float = 0.5,
) -> ToolResult:
    toolset = build_toolset(
        _absolute_space(),
        _absolute_obs_space(),
        control_hz=control_hz,
        max_speed_frac=max_speed_frac,
    )
    return toolset.execute(
        _call("move_joints", targets={"joint": target}),
        _obs({"q": np.array([current])}),
    )


# --- bind-time validation ------------------------------------------------------


def test_build_refuses_unsupported_configurations() -> None:
    no_sem = Box(shape=(2,), low=np.zeros(2), high=np.ones(2))
    with pytest.raises(ToolsetError, match="semantics"):
        build_toolset(no_sem, _bimanual_obs_space(), control_hz=10.0)

    vel = Box(
        shape=(2,),
        low=-np.ones(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_vel"),
    )
    with pytest.raises(ToolsetError, match="joint_vel"):
        build_toolset(vel, _bimanual_obs_space(), control_hz=10.0)

    quat = Box(
        shape=(7,),
        low=-np.ones(7),
        high=np.ones(7),
        semantics=ActionSemantics("eef_abs_pose", rotation_repr="quat_wxyz"),
    )
    with pytest.raises(ToolsetError, match="rotation_repr"):
        build_toolset(quat, _bimanual_obs_space(), control_hz=10.0)


def test_absolute_mode_requires_exactly_one_aligned_state_field() -> None:
    space = _bimanual_space()
    with pytest.raises(ToolsetError, match="StateSpec"):
        build_toolset(space, ObservationSpace(), control_hz=10.0)

    none_match = ObservationSpace(state=StateSpec(fields=(StateField(key="eef", shape=(3,)),)))
    with pytest.raises(ToolsetError, match="exactly one"):
        build_toolset(space, none_match, control_hz=10.0)

    two_match = ObservationSpace(
        state=StateSpec(fields=(StateField(key="a", shape=(14,)), StateField(key="b", shape=(14,))))
    )
    with pytest.raises(ToolsetError, match="exactly one"):
        build_toolset(space, two_match, control_hz=10.0)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (None, np.array([1.0])),
        (np.array([-1.0]), None),
        (np.array([float("-inf")]), np.array([1.0])),
        (np.array([-1.0]), np.array([float("inf")])),
        (np.array([float("nan")]), np.array([1.0])),
    ],
)
def test_absolute_mode_requires_finite_bounds(
    low: np.ndarray | None, high: np.ndarray | None
) -> None:
    space = Box(
        shape=(1,),
        low=low,
        high=high,
        semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
    )
    with pytest.raises(ToolsetError, match="finite low and high bounds"):
        build_toolset(space, _absolute_obs_space(), control_hz=10.0)


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (None, np.array([0.1, 0.1]), "finite low and high bounds"),
        (np.array([-0.1, -0.1]), None, "finite low and high bounds"),
        (np.array([-np.inf, -0.1]), np.array([0.1, 0.1]), "finite low and high bounds"),
        (np.array([-0.1, -0.1]), np.array([0.1, np.inf]), "finite low and high bounds"),
        (np.array([0.01, -0.1]), np.array([0.1, 0.1]), "contain zero"),
        (np.array([-0.1, -0.1]), np.array([-0.01, 0.1]), "contain zero"),
    ],
)
def test_displacement_mode_requires_finite_zero_containing_bounds(
    low: np.ndarray | None, high: np.ndarray | None, message: str
) -> None:
    space = Box(
        shape=(2,),
        low=low,
        high=high,
        semantics=ActionSemantics("eef_delta_pos"),
    )
    with pytest.raises(ToolsetError, match=message):
        build_toolset(space, ObservationSpace(), control_hz=10.0)


def test_zero_width_and_zero_sided_bounds_bind() -> None:
    fixed = _absolute_space(low=np.array([0.3]), high=np.array([0.3]))
    build_toolset(fixed, _absolute_obs_space(), control_hz=10.0)

    build_toolset(
        _delta_space(low=np.array([0.0, -0.1]), high=np.array([0.1, 0.0])),
        ObservationSpace(),
        control_hz=10.0,
    )


@pytest.mark.parametrize("control_hz", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_declared_control_hz_must_be_finite_and_positive(control_hz: float) -> None:
    with pytest.raises(ToolsetError, match="control_hz must be finite and > 0"):
        build_toolset(_delta_space(), ObservationSpace(), control_hz=control_hz)


def test_control_hz_none_binds() -> None:
    build_toolset(_delta_space(), ObservationSpace(), control_hz=None)


def test_build_rejects_degenerate_derivations() -> None:
    with pytest.raises(ToolsetError, match="too large to derive a playout cap"):
        build_toolset(_delta_space(), ObservationSpace(), control_hz=1e308)

    with pytest.raises(ToolsetError, match="underflows to a zero per-step limit"):
        build_toolset(
            _absolute_space(),
            _absolute_obs_space(),
            control_hz=10.0,
            max_speed_frac=5e-324,
        )

    huge = _absolute_space(low=np.array([-1e308]), high=np.array([1e308]))
    with pytest.raises(ToolsetError, match=r"range .* overflows"):
        build_toolset(huge, _absolute_obs_space(), control_hz=10.0)

    # float32 bounds whose difference overflows only in the native dtype:
    # DeltaLimitApprover subtracts without promoting, so this must reject too.
    huge32 = _absolute_space(
        low=np.array([-3e38], dtype=np.float32), high=np.array([3e38], dtype=np.float32)
    )
    with pytest.raises(ToolsetError, match=r"range .* overflows"):
        build_toolset(huge32, _absolute_obs_space(), control_hz=10.0)

    offset = _absolute_space(low=np.array([1e16]), high=np.array([1e16 + 2.0]))
    with pytest.raises(ToolsetError, match="too coarse at this magnitude"):
        build_toolset(offset, _absolute_obs_space(), control_hz=10.0)

    subnormal_range = _absolute_space(low=np.array([0.0]), high=np.array([5e-324]))
    with pytest.raises(ToolsetError, match="too coarse at this magnitude"):
        build_toolset(subnormal_range, _absolute_obs_space(), control_hz=10.0)

    # frac/hz is nonzero but multiplying by a small range underflows the
    # derived limit to zero; the dimension must not be misreported as fixed.
    with pytest.raises(ToolsetError, match="underflows the derived per-step limit"):
        build_toolset(
            _absolute_space(low=np.array([0.0]), high=np.array([0.1])),
            _absolute_obs_space(),
            control_hz=1.0,
            max_speed_frac=5e-324,
        )

    matrix = Box(
        shape=(2, 2),
        low=np.zeros((2, 2)),
        high=np.ones((2, 2)),
        semantics=ActionSemantics("joint_pos"),
    )
    with pytest.raises(ToolsetError, match="only 1-D"):
        build_toolset(matrix, _absolute_obs_space(dim=4), control_hz=10.0)


def test_low_precision_bounds_never_outrun_native_backstop() -> None:
    # DeltaLimitApprover derives 0.05 * (high - low) in the box's dtype;
    # float16 rounds that to 0.0999755859375, below the float64 0.1. Every
    # emitted step must respect the *native* value or the backstop clamps.
    space = Box(
        shape=(1,),
        low=np.array([-1.0], dtype=np.float16),
        high=np.array([1.0], dtype=np.float16),
        semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
    )
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.9999}),
        _obs({"q": np.array([0.0])}),
    )
    assert result.chunk is not None
    chain = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
    store: dict[str, Any] = {}
    for action in result.chunk.actions:
        assert chain.review(action, store) is action


@pytest.mark.parametrize("max_speed_frac", [0.0, -0.1, float("inf"), float("nan")])
def test_build_rejects_invalid_max_speed_frac(max_speed_frac: float) -> None:
    with pytest.raises(ToolsetError, match="max_speed_frac must be finite and > 0"):
        build_toolset(
            _absolute_space(),
            _absolute_obs_space(),
            control_hz=10.0,
            max_speed_frac=max_speed_frac,
        )


# --- schemas -------------------------------------------------------------------


def test_schemas_match_control_mode_and_remove_duration() -> None:
    absolute = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    names = [t["function"]["name"] for t in absolute.schemas()]
    assert names == ["move_joints", "done", "give_up"]
    move_schema = json.dumps(absolute.schemas()[0])
    assert "left_j0" in move_schema and "right_gripper" in move_schema
    assert "duration_s" not in move_schema
    assert absolute.schemas()[0]["function"]["parameters"]["required"] == ["targets"]

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    names = [t["function"]["name"] for t in displacement.schemas()]
    assert names == ["move_by", "done", "give_up"]
    assert displacement.schemas()[0]["function"]["parameters"]["required"] == ["deltas"]


# --- absolute-mode synthesis ---------------------------------------------------


def test_move_joints_derives_steps_and_snaps_bit_exact_target() -> None:
    result = _execute_absolute(1.0, current=-0.5)
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 16
    assert result.chunk.actions[-1].data[0] == 1.0
    assert result.note == "executing move_joints over 16 steps (1.6s)"

    # Pins the snap itself: 0.3 is interior (clip cannot repair it) and plain
    # linspace arithmetic lands on 0.30000000000000004 instead.
    snapped = _execute_absolute(0.3, current=-0.1)
    assert snapped.chunk is not None
    assert snapped.chunk.actions[-1].data[0] == 0.3


def test_move_joints_clips_every_interpolant_into_box() -> None:
    space = _absolute_space(low=np.array([-1.0]), high=np.array([0.3]))
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.3}),
        _obs({"q": np.array([-0.1])}),
    )
    assert result.error is None and result.chunk is not None
    assert all(-1.0 <= float(action.data[0]) <= 0.3 for action in result.chunk.actions)


def test_move_joints_headroom_stays_within_default_backstop() -> None:
    current = -0.5
    result = _execute_absolute(1.0, current=current)
    assert result.chunk is not None
    emitted = [current, *(float(action.data[0]) for action in result.chunk.actions)]
    assert all(right - left <= 0.1 for left, right in pairwise(emitted))


@pytest.mark.parametrize(("target", "current"), [(0.01, 0.0), (0.4, 0.4)])
def test_move_joints_has_one_step_floor(target: float, current: float) -> None:
    result = _execute_absolute(target, current=current)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 1


def test_move_joints_preserves_unnamed_dimensions_and_index_labels() -> None:
    space = Box(
        shape=(2,),
        low=np.zeros(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_pos"),
    )
    obs_space = _absolute_obs_space(dim=2)
    toolset = build_toolset(space, obs_space, control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"1": 0.6}),
        _obs({"q": np.array([0.2, 0.2])}),
    )
    assert result.error is None and result.chunk is not None
    final = np.asarray(result.chunk.actions[-1].data)
    assert final[0] == 0.2 and final[1] == 0.6


def test_move_joints_rejects_out_of_bounds_but_accepts_bound() -> None:
    toolset = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    rejected = toolset.execute(
        _call("move_joints", targets={"joint": 1.01}),
        _obs(),
    )
    assert rejected.error == "target for joint is outside [-1.0, 1.0]"

    accepted = toolset.execute(
        _call("move_joints", targets={"joint": -1.0}),
        _obs(),
    )
    assert accepted.error is None and accepted.chunk is not None
    assert accepted.chunk.actions[-1].data[0] == -1.0


def test_zero_width_target_uses_bound_not_noisy_observation_and_no_steps() -> None:
    bound = 0.30000000000000004
    space = _absolute_space(low=np.array([bound]), high=np.array([bound]), labels=("fixed",))
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0)

    accepted = toolset.execute(
        _call("move_joints", targets={"fixed": bound}),
        _obs({"q": np.array([0.31])}),
    )
    assert accepted.error is None and accepted.chunk is not None
    assert len(accepted.chunk.actions) == 1
    assert accepted.chunk.actions[0].data[0] == bound

    rejected = toolset.execute(
        _call("move_joints", targets={"fixed": 0.3}),
        _obs({"q": np.array([bound])}),
    )
    assert rejected.error == "dimension fixed is fixed at 0.30000000000000004"


@pytest.mark.parametrize(("control_hz", "max_speed_frac"), [(5.0, 0.5), (10.0, 1.0)])
def test_move_joints_per_step_ceiling_matches_default_backstop(
    control_hz: float, max_speed_frac: float
) -> None:
    result = _execute_absolute(
        1.0,
        current=-1.0,
        control_hz=control_hz,
        max_speed_frac=max_speed_frac,
    )
    assert result.chunk is not None
    assert len(result.chunk.actions) == 21
    emitted = [-1.0, *(float(action.data[0]) for action in result.chunk.actions)]
    assert all(right - left <= 0.1 for left, right in pairwise(emitted))


def test_move_joints_honors_non_default_speed_fraction() -> None:
    result = _execute_absolute(0.5, max_speed_frac=0.25)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 11


@pytest.mark.parametrize("bad_state", [float("nan"), float("inf"), float("-inf")])
def test_move_joints_rejects_non_finite_observed_state(bad_state: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _execute_absolute(0.5, current=bad_state)


def test_broken_sensor_raises_even_with_malformed_arguments() -> None:
    toolset = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    # A malformed tool call must not mask a broken sensor behind a
    # correctable structured error.
    with pytest.raises(ValueError, match="non-finite"):
        toolset.execute(
            _call("move_joints", targets={"unknown_dim": 0.1}),
            _obs({"q": np.array([float("nan")])}),
        )


def test_move_joints_absurd_finite_state_returns_cap_error() -> None:
    result = _execute_absolute(0.0, current=1e308)
    assert result.chunk is None
    assert result.error is not None and "split the move into smaller motions" in result.error


def test_move_joints_over_cap_returns_structured_error() -> None:
    result = _execute_absolute(1.0, current=-1.0, max_speed_frac=0.01)
    assert result.chunk is None
    assert result.error is not None and "split the move into smaller motions" in result.error

    boundary = _execute_absolute(0.98, current=-1.0, max_speed_frac=0.1)
    assert boundary.error is None and boundary.chunk is not None
    assert len(boundary.chunk.actions) == 100


def test_absolute_chunks_pass_default_approvers_across_calls() -> None:
    space = _absolute_space()
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0)
    chain = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
    store: dict[str, Any] = {}

    first = toolset.execute(
        _call("move_joints", targets={"joint": 1.0}),
        _obs({"q": np.array([-0.5])}),
    )
    second = toolset.execute(
        _call("move_joints", targets={"joint": -1.0}),
        _obs({"q": np.array([1.0])}),
    )
    assert first.chunk is not None and second.chunk is not None
    for action in (*first.chunk.actions, *second.chunk.actions):
        assert chain.review(action, store) is action


# --- displacement-mode synthesis ----------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "delta", "expected_steps", "expected_value"),
    [
        (-0.1, 0.05, 0.2, 5, 0.04),
        (-0.1, 0.05, -0.2, 3, -0.2 / 3),
    ],
)
def test_move_by_splits_by_directional_box_side(
    low: float, high: float, delta: float, expected_steps: int, expected_value: float
) -> None:
    space = _delta_space(low=np.array([low, -0.1]), high=np.array([high, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": delta}), _obs({}))
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == expected_steps
    assert all(
        float(action.data[0]) == pytest.approx(expected_value) for action in result.chunk.actions
    )


def test_move_by_headroom_avoids_one_ulp_box_overrun() -> None:
    limit = 0.8137309994111953
    delta = 4.068654997055977
    space = _delta_space(high=np.array([limit, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": delta}), _obs({}))
    assert result.chunk is not None
    assert len(result.chunk.actions) == 6
    assert all(abs(float(action.data[0])) <= limit for action in result.chunk.actions)


def test_move_by_all_zero_is_one_step_hold() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 0.0, "1": 0.0}), _obs({}))
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 1
    assert np.array_equal(result.chunk.actions[0].data, np.zeros(2))


@pytest.mark.parametrize(
    ("low", "high", "delta"),
    [(0.0, 0.1, -0.01), (-0.1, 0.0, 0.01)],
)
def test_move_by_rejects_zero_bound_direction(low: float, high: float, delta: float) -> None:
    space = _delta_space(low=np.array([low, -0.1]), high=np.array([high, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": delta}), _obs({}))
    assert result.error == "dimension 0 cannot move in that direction"


def test_move_by_ignores_max_speed_frac() -> None:
    lengths = []
    for frac in (0.5, 0.1):
        toolset = build_toolset(
            _delta_space(), ObservationSpace(), control_hz=10.0, max_speed_frac=frac
        )
        result = toolset.execute(_call("move_by", deltas={"0": 0.2}), _obs({}))
        assert result.chunk is not None
        lengths.append(len(result.chunk.actions))
    assert lengths == [3, 3]


def test_move_by_over_cap_errors_and_boundary_succeeds() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    too_large = toolset.execute(_call("move_by", deltas={"0": 10.0}), _obs({}))
    assert too_large.chunk is None
    assert too_large.error is not None and "split the move into smaller motions" in too_large.error

    boundary = toolset.execute(_call("move_by", deltas={"0": 9.9}), _obs({}))
    assert boundary.error is None and boundary.chunk is not None
    assert len(boundary.chunk.actions) == 100


def test_huge_finite_move_by_returns_cap_error() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 1e308}), _obs({}))
    assert result.chunk is None
    assert result.error is not None and "split the move into smaller motions" in result.error


def test_subnormal_delta_underflow_is_a_structured_error() -> None:
    space = _delta_space(low=np.array([-5e-324, -0.1]), high=np.array([5e-324, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 5e-324}), _obs({}))
    assert result.chunk is None
    assert result.error is not None and "too small to split" in result.error


def test_arbitrary_precision_json_integer_is_a_structured_error() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    # 10**400 overflows float() and crashes np.isfinite; both int sizes must
    # come back as errors the LLM can correct, never exceptions.
    overflowing = toolset.execute(_call("move_by", deltas={"0": 10**400}), _obs({}))
    assert overflowing.chunk is None
    assert overflowing.error is not None and "must be a finite number" in overflowing.error

    representable = toolset.execute(_call("move_by", deltas={"0": 10**100}), _obs({}))
    assert representable.chunk is None
    assert representable.error is not None and "split the move" in representable.error


# --- done, notes, and structured errors ---------------------------------------


def test_done_and_give_up_emit_control_mode_hold() -> None:
    absolute = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    state = np.full(14, 0.3)
    result = absolute.execute(
        _call("done", summary="fork placed"),
        Observation(state={"joint_pos": state}),
    )
    assert result.error is None and result.chunk is not None
    (action,) = result.chunk.actions
    assert np.array_equal(action.data, state)
    assert action.meta["request_stop"] is True
    assert action.meta["stop_reason"] == "done"

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = displacement.execute(_call("give_up", reason="cannot see"), _obs({}))
    assert result.chunk is not None
    assert np.array_equal(result.chunk.actions[0].data, np.zeros(2))
    assert result.chunk.actions[0].meta["stop_reason"] == "give_up"


def test_tool_errors_are_messages_not_exceptions() -> None:
    toolset = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    cases = [
        _call("move_joints", targets={"left_elbow": 0.1}),
        _call("move_joints", targets={"left_j0": float("nan")}),
        _call("move_joints", targets={"left_j0": "fast"}),
        _call("move_joints", targets={}),
        _call("nonexistent_tool", x=1),
        ToolCall(id="c", name="move_joints", arguments="{not json"),
        ToolCall(id="c", name="move_joints", arguments="[]"),
    ]
    for call in cases:
        result = toolset.execute(call, Observation(state={"joint_pos": np.zeros(14)}))
        assert result.chunk is None and result.error, f"expected error for {call}"
    assert "left_elbow" in str(
        toolset.execute(cases[0], Observation(state={"joint_pos": np.zeros(14)})).error
    )


def test_stray_duration_key_is_ignored() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(
        _call("move_by", deltas={"0": 0.05}, duration_s=0.001),
        _obs({}),
    )
    assert result.error is None and result.chunk is not None


def test_control_hz_none_uses_fallback_without_seconds_note() -> None:
    result = _execute_absolute(1.0, control_hz=None)
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 11
    assert result.chunk.control_hz is None
    assert result.note == "executing move_joints over 11 steps"


def test_declared_rate_note_divides_steps_by_hz() -> None:
    result = _execute_absolute(0.5, control_hz=20.0)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 11
    assert result.note == "executing move_joints over 11 steps (0.6s)"
