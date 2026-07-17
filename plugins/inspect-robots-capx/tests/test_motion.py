"""Motion helpers preserve cursor semantics and the core delta backstop."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from inspect_robots.approver import DeltaLimitApprover
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Action
from inspect_robots_capx._motion import MotionQueue


def _space() -> Box:
    return Box(
        shape=(3,),
        low=np.array([-1.0, -2.0, 0.0]),
        high=np.array([1.0, 2.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=("shoulder", "elbow", "gripper"),
        ),
    )


def _motion(
    *, control_hz: float = 10.0, max_speed_frac: float = 0.1, open_high: bool = True
) -> MotionQueue:
    return MotionQueue(
        _space(),
        control_hz=control_hz,
        max_speed_frac=max_speed_frac,
        gripper_index=2,
        gripper_open_is_high=open_high,
    )


def test_speed_fraction_and_control_rate_set_per_step_interpolation() -> None:
    motion = _motion(control_hz=10.0, max_speed_frac=0.1)
    start = np.array([0.0, 0.0, 1.0])
    motion.begin_turn(start)

    motion.move_to_joints(np.array([0.2, -0.4]))
    chunk = motion.take_chunk()

    points = [start, *(action.data for action in chunk.actions)]
    deltas = np.abs(np.diff(np.stack(points), axis=0))
    assert len(chunk.actions) == 11
    assert np.all(deltas <= np.array([0.02, 0.04, 0.01]))
    assert np.array_equal(chunk.actions[-1].data, np.array([0.2, -0.4, 1.0]))
    assert chunk.control_hz == 10.0


def test_low_control_rate_never_exceeds_delta_limit_approver_backstop() -> None:
    space = _space()
    motion = MotionQueue(
        space,
        control_hz=1.0,
        max_speed_frac=0.5,
        gripper_index=2,
    )
    start = np.array([-1.0, -2.0, 1.0])
    motion.begin_turn(start)
    motion.move_to_joints(np.array([1.0, 2.0]))
    chunk = motion.take_chunk()

    approver = DeltaLimitApprover(space)
    store: dict[str, object] = {}
    approver.review(Action(data=start), store)
    for action in chunk.actions:
        assert approver.review(action, store) is action

    points = [start, *(action.data for action in chunk.actions)]
    per_step = np.abs(np.diff(np.stack(points), axis=0))
    native_backstop = 0.05 * (space.high - space.low)  # type: ignore[operator]
    assert np.all(per_step <= native_backstop)


def test_cursor_chains_across_arm_and_gripper_calls() -> None:
    motion = _motion()
    motion.begin_turn(np.array([0.0, 0.0, 1.0]))

    motion.move_to_joints(np.array([0.1, 0.2]))
    arm_endpoint = motion.cursor
    motion.close_gripper()
    closed_endpoint = motion.cursor
    motion.open_gripper()
    open_endpoint = motion.cursor
    chunk = motion.take_chunk(code="move_to_joints(...)")

    assert arm_endpoint is not None and np.array_equal(arm_endpoint, [0.1, 0.2, 1.0])
    assert closed_endpoint is not None and np.array_equal(closed_endpoint, [0.1, 0.2, 0.0])
    assert open_endpoint is not None and np.array_equal(open_endpoint, [0.1, 0.2, 1.0])
    assert np.array_equal(chunk.actions[-1].data, open_endpoint)
    assert chunk.actions[0].meta["code"] == "move_to_joints(...)"
    assert all(
        np.max(np.abs(right.data - left.data)) <= 0.04 for left, right in pairwise(chunk.actions)
    )


def test_hold_chunk_has_one_full_state_action_and_stop_metadata() -> None:
    motion = _motion()
    state = np.array([0.3, -0.2, 0.8])

    chunk = motion.hold_chunk(state, stop_reason="FINISH", inference_latency_s=0.25)

    assert len(chunk.actions) == 1
    assert np.array_equal(chunk.actions[0].data, state)
    assert chunk.actions[0].meta == {"request_stop": True, "stop_reason": "FINISH"}
    assert chunk.inference_latency_s == 0.25


def test_gripper_values_come_from_box_bounds_with_configurable_polarity() -> None:
    normal = _motion(open_high=True)
    inverted = _motion(open_high=False)

    assert normal.gripper_open_value == 1.0
    assert normal.gripper_closed_value == 0.0
    assert inverted.gripper_open_value == 0.0
    assert inverted.gripper_closed_value == 1.0


def test_offset_bounds_with_coarse_float_grid_are_rejected() -> None:
    space = Box(
        shape=(2,),
        low=np.array([1e16, 0.0]),
        high=np.array([1e16 + 2.0, 1.0]),
        semantics=ActionSemantics("joint_pos", gripper="continuous", dim_labels=("j0", "gripper")),
    )

    with pytest.raises(ValueError, match="too coarse"):
        MotionQueue(space, control_hz=10.0, max_speed_frac=0.1, gripper_index=1)
