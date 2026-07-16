"""Translate the supported ROS message dictionaries to and from NumPy values.

rosbridge performs ROS message conversion server-side. The adapter therefore
needs only the six standard message shapes used for state, images, poses, arm
commands, and gripper commands. ROS 1 and ROS 2 differ in type strings and in
the duration keys inside ``JointTrajectory``.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

MessageKind = Literal[
    "joint_state",
    "compressed_image",
    "pose_stamped",
    "joint_trajectory",
    "float64_multi_array",
    "float64",
]

_ROS1_TYPES: dict[MessageKind, str] = {
    "joint_state": "sensor_msgs/JointState",
    "compressed_image": "sensor_msgs/CompressedImage",
    "pose_stamped": "geometry_msgs/PoseStamped",
    "joint_trajectory": "trajectory_msgs/JointTrajectory",
    "float64_multi_array": "std_msgs/Float64MultiArray",
    "float64": "std_msgs/Float64",
}
_ROS2_TYPES: dict[MessageKind, str] = {
    "joint_state": "sensor_msgs/msg/JointState",
    "compressed_image": "sensor_msgs/msg/CompressedImage",
    "pose_stamped": "geometry_msgs/msg/PoseStamped",
    "joint_trajectory": "trajectory_msgs/msg/JointTrajectory",
    "float64_multi_array": "std_msgs/msg/Float64MultiArray",
    "float64": "std_msgs/msg/Float64",
}


def message_type(kind: MessageKind, ros_version: int) -> str:
    """Return the rosbridge type string for one supported message and ROS version."""
    if ros_version == 1:
        return _ROS1_TYPES[kind]
    if ros_version == 2:
        return _ROS2_TYPES[kind]
    raise ValueError(f"ros_version must be 1 or 2, got {ros_version!r}")


def parse_joint_state(
    msg: Mapping[str, Any], joint_names: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Return positions reordered into configured joint order.

    A configured joint absent from the message is an error that lists every
    available name, because many ROS drivers split arm and gripper state across
    publishers and that configuration must not fail silently.
    """
    raw_names = msg.get("name")
    raw_positions = msg.get("position")
    if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
        raise ValueError("JointState field 'name' must be an array of strings")
    if not isinstance(raw_positions, list):
        raise ValueError("JointState field 'position' must be an array")
    if len(raw_names) != len(raw_positions):
        raise ValueError(
            "JointState name and position arrays must have equal length; "
            f"got {len(raw_names)} names and {len(raw_positions)} positions"
        )
    try:
        positions = np.asarray(raw_positions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("JointState field 'position' must contain numeric values") from exc
    by_name = dict(zip(raw_names, positions, strict=True))
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise ValueError(
            f"JointState is missing configured joints {missing}; available names: {sorted(by_name)}"
        )
    return np.asarray([by_name[name] for name in joint_names], dtype=np.float64)


def parse_compressed_image(msg: Mapping[str, Any]) -> npt.NDArray[np.uint8]:
    """Decode base64 JPEG or PNG bytes into a copied ``(H, W, 3)`` RGB uint8 array."""
    data = msg.get("data")
    if not isinstance(data, str):
        raise ValueError("CompressedImage field 'data' must be a base64 string")
    try:
        encoded = base64.b64decode(data, validate=True)
        with Image.open(BytesIO(encoded)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        image_format = msg.get("format", "unknown")
        raise ValueError(
            f"could not decode CompressedImage format {image_format!r} as JPEG or PNG: {exc}"
        ) from exc


def parse_pose_stamped(msg: Mapping[str, Any]) -> npt.NDArray[np.float64]:
    """Convert PoseStamped xyz plus xyzw orientation to ``[x, y, z, qw, qx, qy, qz]``."""
    try:
        pose = msg["pose"]
        position = pose["position"]
        orientation = pose["orientation"]
        values = (
            position["x"],
            position["y"],
            position["z"],
            orientation["w"],
            orientation["x"],
            orientation["y"],
            orientation["z"],
        )
        return np.asarray(values, dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "PoseStamped must contain numeric pose.position.{x,y,z} and "
            "pose.orientation.{x,y,z,w} fields"
        ) from exc


def build_joint_trajectory(
    joint_names: Sequence[str],
    positions: npt.ArrayLike,
    *,
    period_s: float,
    ros_version: int,
) -> dict[str, Any]:
    """Build a one-point joint trajectory whose duration equals the control period."""
    if not np.isfinite(period_s) or period_s < 0:
        raise ValueError(f"period_s must be finite and non-negative, got {period_s!r}")
    position_array = np.asarray(positions, dtype=np.float64).reshape(-1)
    if len(joint_names) != position_array.size:
        raise ValueError(
            f"joint trajectory has {len(joint_names)} names but {position_array.size} positions"
        )
    whole_seconds = int(period_s)
    nanoseconds = round((period_s - whole_seconds) * 1_000_000_000)
    if nanoseconds == 1_000_000_000:
        whole_seconds += 1
        nanoseconds = 0
    if ros_version == 1:
        duration = {"secs": whole_seconds, "nsecs": nanoseconds}
    elif ros_version == 2:
        duration = {"sec": whole_seconds, "nanosec": nanoseconds}
    else:
        raise ValueError(f"ros_version must be 1 or 2, got {ros_version!r}")
    return {
        "joint_names": list(joint_names),
        "points": [
            {
                "positions": position_array.tolist(),
                "time_from_start": duration,
            }
        ],
    }


def build_float64_multi_array(values: npt.ArrayLike) -> dict[str, Any]:
    """Build ``Float64MultiArray`` with a flat numeric ``data`` field."""
    return {"data": np.asarray(values, dtype=np.float64).reshape(-1).tolist()}


def build_gripper_command(value: float, command_type: str) -> dict[str, Any]:
    """Build a raw scalar or one-element array gripper command without normalization."""
    if command_type == "float64":
        return {"data": float(value)}
    if command_type == "float64_multi_array":
        return build_float64_multi_array([value])
    raise ValueError(
        f"gripper command_type must be 'float64' or 'float64_multi_array', got {command_type!r}"
    )
