import math

import numpy as np
import pytest

from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots_jev.menu import Move
from inspect_robots_jev.motion import MotionMapper

LABELS = tuple(
    f"{arm}_{p}"
    for arm in ("left", "right")
    for p in ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
)
ARM_LOW = (0.15, -0.25, 0.03, -math.pi, 0.0, 0.0, 0.0)
ARM_HIGH = (0.48, 0.25, 0.40, math.pi, 0.0, 0.0, 1.0)


def box(*, max_step: bool) -> Box:
    sem = ActionSemantics(
        control_mode="eef_abs_pose",
        gripper="continuous",
        frame="base",
        dim_labels=LABELS,
        max_step=((0.01, 0.01, 0.01, None, None, None, 0.2) * 2) if max_step else None,
    )
    return Box(low=np.array(ARM_LOW * 2), high=np.array(ARM_HIGH * 2), shape=(14,), semantics=sem)


def state() -> np.ndarray:
    s = np.zeros(14)
    s[[0, 7]] = 0.30
    s[[2, 9]] = 0.10
    s[[6, 13]] = 1.0
    return s


def test_axis_move_splits_by_declared_max_step() -> None:
    m = MotionMapper(box(max_step=True), control_hz=10.0)
    chunk = m.chunk(Move("left", "x", 0.02, None), state())
    assert len(chunk.actions) == 2 and chunk.control_hz == 10.0
    assert chunk.actions[-1].data[0] == pytest.approx(0.32)
    assert chunk.actions[0].data[0] == pytest.approx(0.31)
    others = [i for i in range(14) if i != 0]
    np.testing.assert_allclose(chunk.actions[-1].data[others], state()[others])


def test_small_move_is_one_action() -> None:
    m = MotionMapper(box(max_step=True), control_hz=None)
    chunk = m.chunk(Move("right", "z", -0.005, None), state())
    assert len(chunk.actions) == 1 and chunk.actions[0].data[9] == pytest.approx(0.095)


def test_move_is_clipped_to_bounds() -> None:
    m = MotionMapper(box(max_step=True), control_hz=10.0)
    s = state()
    s[0] = 0.47
    chunk = m.chunk(Move("left", "x", 0.05, None), s)
    assert chunk.actions[-1].data[0] == pytest.approx(0.48)
    assert len(chunk.actions) == 1


def test_gripper_close_splits_by_gripper_step() -> None:
    m = MotionMapper(box(max_step=True), control_hz=10.0)
    chunk = m.chunk(Move("left", None, 0.0, 0.0), state())
    assert len(chunk.actions) == 5 and chunk.actions[-1].data[6] == 0.0
    assert chunk.actions[0].data[6] == pytest.approx(0.8)


def test_hold_returns_current() -> None:
    m = MotionMapper(box(max_step=True), control_hz=10.0)
    chunk = m.chunk(Move("left", None, 0.0, None), state())
    assert len(chunk.actions) == 1
    np.testing.assert_allclose(chunk.actions[0].data, state())


def test_fallback_five_percent_limits_match_rig_defaults() -> None:
    m = MotionMapper(box(max_step=False), control_hz=10.0)
    assert m.step_limits[0] == pytest.approx(0.05 * 0.33)
    assert len(m.chunk(Move("left", "x", 0.05, None), state()).actions) == 4
    assert len(m.chunk(Move("left", None, 0.0, 0.0), state()).actions) == 20


def test_readers_and_bounds() -> None:
    m = MotionMapper(box(max_step=True), control_hz=10.0)
    assert m.gripper_position(state(), "left") == (0.30, 0.0, 0.10)
    assert m.gripper_position(state(), "right") == (0.30, 0.0, 0.10)
    assert m.gripper_opening(state(), "right") == 1.0
    assert m.index("right_gripper") == 13
    assert m.bounds_xyz("left") == ((0.15, -0.25, 0.03), (0.48, 0.25, 0.40))


def test_playout_cap() -> None:
    m = MotionMapper(box(max_step=True), control_hz=1.0, max_playout_s=1.0)
    with pytest.raises(ValueError, match="playout cap"):
        m.chunk(Move("left", "x", 0.05, None), state())


def test_rejects_box_without_labels_or_bounds() -> None:
    with pytest.raises(ValueError, match="dim_labels"):
        MotionMapper(Box(low=np.zeros(2), high=np.ones(2), shape=(2,)), control_hz=None)
    sem = ActionSemantics(control_mode="eef_abs_pose", dim_labels=("left_x", "left_y"))
    with pytest.raises(ValueError, match="left_z"):
        MotionMapper(
            Box(low=np.zeros(2), high=np.ones(2), shape=(2,), semantics=sem), control_hz=None
        )
