import json
import math
from pathlib import Path

import numpy as np
import pytest

from inspect_robots_jev.calibration import (
    Calibration,
    invert,
    pose_to_matrix,
    solve_calibration,
    table_tag_pose_from_corners,
    transform,
)


def rot_z(a: float) -> np.ndarray:
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def test_transform_invert_round_trip() -> None:
    T = pose_to_matrix(rot_z(0.7) @ np.diag([1, -1, -1]), [0.3, -0.2, 1.1])
    p = (0.11, 0.22, 0.33)
    q = transform(T, p)
    assert transform(invert(T), q) == pytest.approx(p)
    np.testing.assert_allclose(invert(T) @ T, np.eye(4), atol=1e-12)


def test_table_tag_pose_from_square_on_table() -> None:
    s = 0.04
    corners = [(0.26, 0.14, 0.0), (0.34, 0.14, 0.0), (0.34, 0.06, 0.0), (0.26, 0.06, 0.0)]
    T = table_tag_pose_from_corners(corners)
    assert T[:3, 3] == pytest.approx((0.30, 0.10, 0.0))
    # x along TL->TR (+x), y along TL->BL (-y in arm frame), z = x cross y (-z: into the table)
    np.testing.assert_allclose(T[:3, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 1], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-12)
    assert s > 0


def test_table_tag_pose_orthonormalises_noisy_corners() -> None:
    corners = [(0.26, 0.14, 0.001), (0.34, 0.141, 0.0), (0.341, 0.06, -0.001), (0.26, 0.059, 0.0)]
    T = table_tag_pose_from_corners(corners)
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_table_tag_pose_needs_four_corners() -> None:
    with pytest.raises(ValueError, match="four"):
        table_tag_pose_from_corners([(0, 0, 0), (1, 0, 0), (1, 1, 0)])


def test_solve_calibration_recovers_camera_pose() -> None:
    cam_in_left = pose_to_matrix(np.diag([1, -1, -1]), [0.30, 0.0, 1.2])  # camera looking down
    cam_in_right = pose_to_matrix(np.diag([1, -1, -1]), [0.30, 0.5, 1.2])
    tag_in_left = pose_to_matrix(np.diag([1, -1, -1]), [0.30, 0.10, 0.0])
    tag_in_camera = invert(cam_in_left) @ tag_in_left
    tag_in_right = cam_in_right @ tag_in_camera
    cal = solve_calibration(
        tag_in_camera=tag_in_camera, tag_in_arm={"left": tag_in_left, "right": tag_in_right}
    )
    np.testing.assert_allclose(cal.camera_to_arm["left"], cam_in_left, atol=1e-12)
    np.testing.assert_allclose(cal.camera_to_arm["right"], cam_in_right, atol=1e-12)
    # a point 0.7 m in front of the camera (its +z) lands 0.5 m above the left table plane
    assert cal.to_arm("left", (0.0, 0.0, 0.7)) == pytest.approx((0.30, 0.0, 0.5))


def test_save_load_round_trip(tmp_path: Path) -> None:
    cal = Calibration(
        camera_to_arm={
            "left": pose_to_matrix(rot_z(0.1), [1, 2, 3]),
            "right": pose_to_matrix(rot_z(-0.2), [4, 5, 6]),
        }
    )
    f = tmp_path / "cal.json"
    cal.save(f)
    back = Calibration.load(f)
    for arm in ("left", "right"):
        np.testing.assert_array_equal(back.camera_to_arm[arm], cal.camera_to_arm[arm])


def test_load_rejects_missing_arm_and_bad_shape(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"left": np.eye(4).tolist()}))
    with pytest.raises(ValueError, match="right"):
        Calibration.load(f)
    f.write_text(json.dumps({"left": np.eye(4).tolist(), "right": [[1, 2], [3, 4]]}))
    with pytest.raises(ValueError, match="4x4"):
        Calibration.load(f)
