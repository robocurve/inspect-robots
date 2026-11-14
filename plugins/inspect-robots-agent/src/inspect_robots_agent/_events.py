"""Causal observation triggers and image fidelity for the agent policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.types import ImageArray, Observation

_WATCHDOG_INTERVAL = 4
_WRIST_EDGE = 112
_WRIST_CAMERAS = frozenset({"left_cam", "right_cam"})
_GRIPPER_MOTION = 0.12


class ObservationEvents:
    """Track one trial's normalized gripper readback and validated move targets.

    Fallback is full fidelity when the adapter's state mapping cannot identify
    the gripper coordinates. The threshold matches a fraction of a normalized
    stroke regardless of which direction closes the gripper.
    """

    def __init__(self, state_labels: tuple[str, tuple[str, ...]] | None) -> None:
        self._state_key, labels = state_labels or ("", ())
        self._state_width = len(labels)
        self._grippers = [i for i, label in enumerate(labels) if "gripper" in label.lower()]
        self._previous: npt.NDArray[np.float64] | None = None
        self._target: npt.NDArray[np.float64] | None = None
        self._observations = 0

    def observe(
        self,
        observation: Observation,
        *,
        previous_target: npt.NDArray[np.float64] | None = None,
        recovery: bool = False,
    ) -> str | None:
        """Return why this observation needs high fidelity, or None on a routine turn.

        Pass the preceding move's validated absolute target if available. Begin
        with the previous readback, then compare succeeding targets to the last
        command so a held grasp does not count as a new closure on each turn.
        """
        index = self._observations
        self._observations += 1
        previous = self._previous
        current = self._gripper_values(observation.state.get(self._state_key))
        self._previous = current
        new_target = self._gripper_values(previous_target)
        old_target = self._target if self._target is not None else previous
        if new_target is not None:
            self._target = new_target
        if index == 0:
            return "initial"
        if recovery or observation.extra.get("replan") or observation.extra.get("approvals"):
            return "recovery"
        if (
            current is None
            or previous is None
            or (previous_target is not None and new_target is None)
        ):
            return "missing_gripper_state"
        if new_target is not None and old_target is not None and _moved(new_target, old_target):
            return "gripper_command"
        if _moved(current, previous):
            return "gripper_motion"
        if index % _WATCHDOG_INTERVAL == 0:
            return "watchdog"
        return None

    def _gripper_values(
        self, values: npt.NDArray[np.floating[Any]] | None
    ) -> npt.NDArray[np.float64] | None:
        if values is None or not self._grippers or values.shape != (self._state_width,):
            return None
        grippers = np.asarray(values[self._grippers], dtype=np.float64)
        # Allow small readback overshoot on adapters using a normalized stroke.
        if not bool(np.all((grippers >= -0.05) & (grippers <= 1.05))):
            return None
        return grippers


def _moved(current: npt.NDArray[np.float64], previous: npt.NDArray[np.float64]) -> bool:
    return bool(np.any(np.abs(current - previous) >= _GRIPPER_MOTION))


def routine_camera_observation(observation: Observation) -> Observation:
    """Return a wire-only view with smaller wrist images and unchanged camera names."""
    reduced = {
        name: _wrist_thumbnail(image)
        for name, image in observation.images.items()
        if name in _WRIST_CAMERAS
    }
    if any(image is not observation.images[name] for name, image in reduced.items()):
        return replace(observation, images={**observation.images, **reduced})
    return observation


def _wrist_thumbnail(image: ImageArray) -> ImageArray:
    height, width = image.shape[:2]
    edge = max(height, width)
    if edge <= _WRIST_EDGE or min(height, width) == 0:
        return image
    new_height = max(1, round(height * _WRIST_EDGE / edge))
    new_width = max(1, round(width * _WRIST_EDGE / edge))
    reduced = _area_average(_area_average(image, new_height), new_width, axis=1)
    return np.rint(reduced).astype(np.uint8)


def _area_average(
    image: npt.NDArray[np.uint8] | npt.NDArray[np.float64],
    length: int,
    *,
    axis: int = 0,
) -> npt.NDArray[np.float64]:
    """Average each output pixel's source footprint, including fractional edges."""
    rows = np.moveaxis(image, axis, 0)
    source_length = rows.shape[0]
    edges = np.linspace(0, source_length, length + 1)
    starts = np.minimum(edges.astype(np.intp), source_length - 1)
    fractions = (edges - starts).reshape((-1,) + (1,) * (rows.ndim - 1))
    prefix = np.cumsum(rows, axis=0, dtype=np.float64)
    preceding = np.zeros_like(prefix[starts])
    has_preceding = starts > 0
    preceding[has_preceding] = prefix[starts[has_preceding] - 1]
    integral = preceding + fractions * rows[starts]
    return np.moveaxis(np.diff(integral, axis=0) * (length / source_length), 0, axis)
