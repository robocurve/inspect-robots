"""The bimanual YAM as Dream's YAM station describes it to Dream-1.

Inspect Robots' YAM embodiments share one 14-D layout: per arm ``[j0..j5, gripper]``, left then
right, joints in radians, grippers normalized (0 closed, 1 open). Dream-1 reads each hand's pose
beside the joints, so it is sent the station keys, from which the serve computes the poses in the
frame the model was trained in. Dream's own simulated-YAM example (``dream_examples/yam_mujoco``)
does the same.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

Vector = npt.NDArray[np.float64]

ARM_JOINTS = 6
ARM_DIM = ARM_JOINTS + 1
DIM = 2 * ARM_DIM
DIM_LABELS = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(ARM_JOINTS)), "gripper")
)
GRIPPERS = (ARM_JOINTS, ARM_DIM + ARM_JOINTS)
STATE_KEY = "joint_pos"
CAMERAS = {"top_cam": "top", "left_cam": "left", "right_cam": "right"}

FRAME_WIDTH, FRAME_HEIGHT = 320, 176
MAX_ROWS = 48
# Dream-1 rows are 30 Hz actions; Dream's YAM station plays them at 20 Hz, two thirds speed.
TRAINED_HZ = 30.0
STATION_HZ = 20.0
OBS_WINDOW = 9
OBS_FPS = 5.0

_ARMS = {"l": slice(0, ARM_DIM), "r": slice(ARM_DIM, DIM)}
_ROBOT = {
    "robot.arm": "yam",
    "robot.exec_head": "joint",
    "robot.max_joint_vel_rad_s": 0.0,
    "robot.gripper_open": 1.0,
}
_GRIPPER_SLACK = 0.05


def vector(value: Any) -> Vector:
    """A finite 14-D bimanual vector, or ValueError."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (DIM,) or not np.isfinite(array).all():
        raise ValueError(f"expected a finite {DIM}-D bimanual YAM vector, got {value!r}")
    return array


def jpeg(rgb: npt.NDArray[np.uint8]) -> str:
    """One camera frame as the station sends it: 320x176 JPEG, quality 85, base64 encoded."""
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).convert("RGB")
    if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
        image = image.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def frames(jpegs: Mapping[str, str]) -> dict[str, str]:
    """Key one base64 JPEG per embodiment camera under its station frame key."""
    missing = sorted(set(CAMERAS) - set(jpegs))
    if missing:
        raise ValueError(f"missing frames for cameras {missing}")
    return {f"image_window.{CAMERAS[camera]}": jpegs[camera] for camera in CAMERAS}


def observation(state: Vector, prompt: str) -> dict[str, Any]:
    """One tick's station keys: each arm's joints and gripper, the robot keys, and the prompt."""
    keys: dict[str, Any] = {"prompt": prompt, **_ROBOT}
    for side, arm in _ARMS.items():
        keys[f"state.arm_{side}.joint_pos"] = state[arm][:ARM_JOINTS].tolist()
        keys[f"state.arm_{side}.gripper_pos"] = state[arm][ARM_JOINTS:].tolist()
    return keys


def commanded(row: Vector) -> dict[str, Vector]:
    """The last row played, per chunk key, so Dream's seam join takes over where the arms head."""
    return {f"q_{side}_chunk": row[arm] for side, arm in _ARMS.items()}


def rows(reply: Mapping[str, Any]) -> Vector:
    """Dream's per-arm chunks as ``[rows, 14]`` targets; ValueError for a reply in another contract.

    Joints pass through: the embodiment and its approvers clamp them. A gripper within a small
    slack of 0..1 is clipped into it; one further out means the reply is in another convention.
    """
    arms = []
    for side in _ARMS:
        key = f"q_{side}_chunk"
        if key not in reply:
            raise ValueError(f"Dream reply has no {key!r}; keys: {sorted(reply)}")
        chunk = np.array(reply[key], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != ARM_DIM or not 1 <= len(chunk) <= MAX_ROWS:
            raise ValueError(f"{key} must be [1..{MAX_ROWS}, {ARM_DIM}], got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError(f"{key} has non-finite values")
        grip = chunk[:, ARM_JOINTS]
        if grip.min() < -_GRIPPER_SLACK or grip.max() > 1 + _GRIPPER_SLACK:
            raise ValueError(f"{key} gripper leaves 0..1: [{grip.min():.3f}, {grip.max():.3f}]")
        chunk[:, ARM_JOINTS] = grip.clip(0.0, 1.0)
        arms.append(chunk)
    left, right = arms
    if left.shape != right.shape:
        raise ValueError(f"left and right chunks differ: {left.shape} vs {right.shape}")
    return np.hstack(arms)
