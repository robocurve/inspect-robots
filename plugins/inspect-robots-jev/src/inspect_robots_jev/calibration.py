"""Camera-to-arm transforms from one fixed table tag.

The table tag's pose in each arm's frame comes from touching its four corners
with the gripper (see README). Seeing the same tag from the camera then gives
``camera_to_arm = tag_in_arm @ inv(tag_in_camera)``. Pure numpy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from inspect_robots_jev.world import ARMS, Arm, Vec3

Mat4 = npt.NDArray[np.float64]


def pose_to_matrix(rotation: npt.ArrayLike, translation: npt.ArrayLike) -> Mat4:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and a translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def invert(T: Mat4) -> Mat4:
    """Invert a rigid transform without a general matrix inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform(T: Mat4, p: Vec3) -> Vec3:
    """Apply a rigid transform to a point."""
    v = T @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (float(v[0]), float(v[1]), float(v[2]))


def table_tag_pose_from_corners(corners_arm: Sequence[Vec3]) -> Mat4:
    """Pose of the printed tag in an arm frame from its four touched corners.

    Corner order follows the *printed tag's* own orientation: top-left,
    top-right, bottom-right, bottom-left. The tag frame is AprilTag's: origin
    at the centre, x along top-left→top-right, y along top-left→bottom-left
    (down the tag), z = x cross y, which points into the tag.
    """
    if len(corners_arm) != 4:
        raise ValueError("need exactly four corners: TL, TR, BR, BL")
    c = np.asarray(corners_arm, dtype=np.float64)
    centre = c.mean(axis=0)
    x = (c[1] - c[0]) + (c[2] - c[3])
    y = (c[3] - c[0]) + (c[2] - c[1])
    x /= np.linalg.norm(x)
    y -= x * float(np.dot(y, x))
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return pose_to_matrix(np.column_stack([x, y, z]), centre)


@dataclass(frozen=True)
class Calibration:
    """Per-arm camera→arm-base transforms, saved as JSON matrices."""

    camera_to_arm: Mapping[Arm, Mat4]

    def to_arm(self, arm: Arm, p_camera: Vec3) -> Vec3:
        """Express a camera-frame point in ``arm``'s base frame."""
        return transform(self.camera_to_arm[arm], p_camera)

    def save(self, path: Path) -> None:
        """Write ``{"left": [[..]*4], "right": [[..]*4]}``."""
        payload = {arm: self.camera_to_arm[arm].tolist() for arm in self.camera_to_arm}
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        """Read a file written by ``save``; every arm must have a 4x4 matrix."""
        raw = json.loads(Path(path).read_text())
        mats: dict[Arm, Mat4] = {}
        for arm in ARMS:
            if arm not in raw:
                raise ValueError(f"calibration file lacks the {arm} arm")
            mat = np.asarray(raw[arm], dtype=np.float64)
            if mat.shape != (4, 4):
                raise ValueError(f"calibration for {arm} is not a 4x4 matrix")
            mats[arm] = mat
        return cls(camera_to_arm=mats)


def solve_calibration(*, tag_in_camera: Mat4, tag_in_arm: Mapping[Arm, Mat4]) -> Calibration:
    """Combine the tag seen from the camera with the tag touched by each arm."""
    inv_cam = invert(tag_in_camera)
    return Calibration(camera_to_arm={arm: tag_in_arm[arm] @ inv_cam for arm in tag_in_arm})
