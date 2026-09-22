"""Turn a menu pick into a bounded absolute Cartesian ``ActionChunk``.

The mapper never emits a per-tick step larger than the action space allows:
``ActionSemantics.max_step`` when declared, else 5 % of each dimension's
range, which is the same default ``DeltaLimitApprover`` derives, so nothing
the mapper produces is ever clamped by the guardrails.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk
from inspect_robots_jev.menu import Move
from inspect_robots_jev.world import ARMS, AXES, Arm, Vec3

_FALLBACK_HZ = 10.0
_RANGE_FRACTION = 0.05


class MotionMapper:
    """Address an absolute Cartesian action box by ``{arm}_{x|y|z|gripper}`` labels."""

    def __init__(
        self, action_space: Box, *, control_hz: float | None, max_playout_s: float = 10.0
    ) -> None:
        semantics = action_space.semantics
        labels = semantics.dim_labels if semantics is not None else None
        if labels is None or action_space.low is None or action_space.high is None:
            raise ValueError("jev needs a bounded action box with dim_labels")
        self._index = {label: i for i, label in enumerate(labels)}
        for arm in ARMS:
            for part in (*AXES, "gripper"):
                if f"{arm}_{part}" not in self._index:
                    raise ValueError(f"action box lacks the {arm}_{part} dimension")
        self.low = np.asarray(action_space.low, dtype=np.float64)
        self.high = np.asarray(action_space.high, dtype=np.float64)
        declared = semantics.max_step if semantics is not None else None
        limits = _RANGE_FRACTION * (self.high - self.low)
        if declared is not None:
            for i, entry in enumerate(declared):
                if entry is not None:
                    limits[i] = float(entry)
        self.step_limits = limits
        self.control_hz = control_hz
        self._max_actions = math.ceil(max_playout_s * (control_hz or _FALLBACK_HZ))

    def index(self, label: str) -> int:
        """Return the action index of a labelled dimension."""
        return self._index[label]

    def bounds_xyz(self, arm: Arm) -> tuple[Vec3, Vec3]:
        """Return ``(low, high)`` of the arm's x, y, z dimensions."""
        idx = [self._index[f"{arm}_{axis}"] for axis in AXES]
        low = tuple(float(self.low[i]) for i in idx)
        high = tuple(float(self.high[i]) for i in idx)
        return (low[0], low[1], low[2]), (high[0], high[1], high[2])

    def gripper_position(self, eef_state: npt.NDArray[np.floating[Any]], arm: Arm) -> Vec3:
        """Read the arm's grasp point from a state vector laid out like the action box."""
        x, y, z = (float(eef_state[self._index[f"{arm}_{axis}"]]) for axis in AXES)
        return (x, y, z)

    def gripper_opening(self, eef_state: npt.NDArray[np.floating[Any]], arm: Arm) -> float:
        """Read the arm's gripper opening (0 closed, 1 open) from a state vector."""
        return float(eef_state[self._index[f"{arm}_gripper"]])

    def chunk(self, move: Move, current: npt.NDArray[np.floating[Any]]) -> ActionChunk:
        """Interpolate from ``current`` to the move's target in guardrail-sized steps."""
        start = np.clip(np.asarray(current, dtype=np.float64), self.low, self.high)
        target = start.copy()
        if move.axis is not None:
            i = self._index[f"{move.arm}_{move.axis}"]
            target[i] = start[i] + move.delta_m
        elif move.gripper is not None:
            i = self._index[f"{move.arm}_gripper"]
            target[i] = move.gripper
        else:
            return ActionChunk(actions=[Action(data=start)], control_hz=self.control_hz)
        target = np.clip(target, self.low, self.high)
        distance = abs(float(target[i] - start[i]))
        steps = max(1, math.ceil(distance / float(self.step_limits[i]) - 1e-9))
        if steps > self._max_actions:
            raise ValueError(
                f"move of {distance:.4f} on {move.arm}_{move.axis or 'gripper'} needs {steps} "
                f"ticks, above the {self._max_actions}-tick playout cap"
            )
        fractions = np.linspace(1.0 / steps, 1.0, steps)
        actions = [Action(data=start + (target - start) * f) for f in fractions[:-1]]
        actions.append(Action(data=target))
        return ActionChunk(actions=actions, control_hz=self.control_hz)
