"""Joint-space motion synthesis for model-generated policy code.

The per-step ceiling reproduces ``DeltaLimitApprover``'s native-dtype
``0.05 * (high - low)`` arithmetic. The speed fraction may make the ceiling
tighter, but never looser, so actions within one returned chunk are not
silently rewritten by the CLI's default delta backstop.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk

_BACKSTOP_STEP_FRAC = 0.05
_RELATIVE_HEADROOM = 1e-6


class MotionQueue:
    """Queue interpolated full-config targets around a turn-local state cursor."""

    def __init__(
        self,
        action_space: Box,
        *,
        control_hz: float,
        max_speed_frac: float,
        gripper_index: int,
        gripper_open_is_high: bool = True,
    ) -> None:
        if len(action_space.shape) != 1:
            raise ValueError(f"motion needs a 1-D action box, got {action_space.shape}")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
            raise ValueError("max_speed_frac must be finite and > 0")
        if not 0 <= gripper_index < action_space.dim:
            raise ValueError(f"gripper index {gripper_index} is outside {action_space.dim}-D box")
        low = action_space.low
        high = action_space.high
        if low is None or high is None:
            raise ValueError("motion needs finite low and high action bounds")
        if not bool(np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("motion needs finite low and high action bounds")

        low64 = np.asarray(low, dtype=np.float64)
        high64 = np.asarray(high, dtype=np.float64)
        with np.errstate(over="ignore"):
            native_range = np.asarray(high - low, dtype=np.float64)
            native_backstop = np.asarray(_BACKSTOP_STEP_FRAC * (high - low), dtype=np.float64)
        float64_range = high64 - low64
        if not bool(np.all(np.isfinite(native_range)) and np.all(np.isfinite(float64_range))):
            raise ValueError("action-space range (high - low) overflows; bounds are too large")
        movable = high64 > low64
        # Interpolants snap to the float grid at the bounds' magnitude; if that
        # grid is coarse relative to the backstop (offset boxes like
        # [1e16, 1e16 + 2]), emitted steps can exceed the approver's budget.
        spacing = np.spacing(np.maximum(np.abs(low64), np.abs(high64)))
        if bool(np.any(movable & (spacing > 5e-7 * native_backstop))):
            raise ValueError(
                "bounds are too coarse at this magnitude for speed-limited "
                "interpolation (float spacing exceeds the per-step budget)"
            )
        step_frac = min(max_speed_frac / control_hz, _BACKSTOP_STEP_FRAC)
        step_limits = np.minimum(step_frac * float64_range, native_backstop)
        if bool(np.any(movable & (step_limits <= 0))):
            raise ValueError("speed fraction underflows the per-step limit for a movable dimension")

        self.control_hz = control_hz
        self._dim = action_space.dim
        self._gripper_index = gripper_index
        self._arm_indices = tuple(index for index in range(self._dim) if index != gripper_index)
        self._low = low64
        self._high = high64
        self._step_limits = step_limits
        high_value = float(high64[gripper_index])
        low_value = float(low64[gripper_index])
        self.gripper_open_value = high_value if gripper_open_is_high else low_value
        self.gripper_closed_value = low_value if gripper_open_is_high else high_value
        self._cursor: npt.NDArray[np.float64] | None = None
        self._actions: list[Action] = []

    @property
    def arm_dof(self) -> int:
        """Return the joint count expected by ``move_to_joints`` and Pyroki."""
        return len(self._arm_indices)

    @property
    def cursor(self) -> npt.NDArray[np.float64] | None:
        """Return a defensive copy of the current queued endpoint, if seeded."""
        return None if self._cursor is None else self._cursor.copy()

    def reset(self) -> None:
        """Clear queued actions and the observation-derived cursor."""
        self._cursor = None
        self._actions.clear()

    def begin_turn(self, state: npt.NDArray[Any]) -> None:
        """Discard old queue state and seed this turn from observed proprioception."""
        cursor = np.asarray(state, dtype=np.float64).reshape(-1)
        if cursor.shape != (self._dim,):
            raise ValueError(
                f"proprioceptive reference has shape {cursor.shape}, expected ({self._dim},)"
            )
        if not bool(np.all(np.isfinite(cursor))):
            raise ValueError("proprioceptive reference contains a non-finite value")
        self._cursor = cursor.copy()
        self._actions.clear()

    def move_to_joints(self, joints: npt.NDArray[Any]) -> None:
        """Queue an arm target while preserving the cursor's gripper value."""
        cursor = self._require_cursor()
        arm = np.asarray(joints, dtype=np.float64).reshape(-1)
        if arm.shape != (self.arm_dof,):
            raise ValueError(
                f"move_to_joints expects {self.arm_dof} arm joints, got shape {arm.shape}"
            )
        if not bool(np.all(np.isfinite(arm))):
            raise ValueError("move_to_joints received a non-finite joint target")
        target = cursor.copy()
        target[list(self._arm_indices)] = arm
        self._queue_target(target)

    def open_gripper(self) -> None:
        """Queue a speed-limited ramp to the configured open box bound."""
        self._queue_gripper(self.gripper_open_value)

    def close_gripper(self) -> None:
        """Queue a speed-limited ramp to the configured closed box bound."""
        self._queue_gripper(self.gripper_closed_value)

    def has_actions(self) -> bool:
        """Report whether model code queued any targets in the current turn."""
        return bool(self._actions)

    def take_chunk(
        self,
        *,
        inference_latency_s: float | None = None,
        code: str | None = None,
    ) -> ActionChunk:
        """Drain the non-empty queue into a chunk and annotate its first action."""
        if not self._actions:
            raise ValueError("motion queue is empty")
        actions = self._actions
        self._actions = []
        if code is not None:
            actions[0] = replace(actions[0], meta={**dict(actions[0].meta), "code": code})
        return ActionChunk(
            actions=actions,
            control_hz=self.control_hz,
            inference_latency_s=inference_latency_s,
        )

    def hold_chunk(
        self,
        state: npt.NDArray[Any],
        *,
        stop_reason: str,
        inference_latency_s: float | None = None,
    ) -> ActionChunk:
        """Return the required one-action stop chunk at the observed full config."""
        value = np.asarray(state, dtype=np.float64).reshape(-1)
        if value.shape != (self._dim,):
            raise ValueError(f"hold state has shape {value.shape}, expected ({self._dim},)")
        action = Action(
            data=value.copy(),
            meta={"request_stop": True, "stop_reason": stop_reason},
        )
        return ActionChunk(
            actions=[action],
            control_hz=self.control_hz,
            inference_latency_s=inference_latency_s,
        )

    def _require_cursor(self) -> npt.NDArray[np.float64]:
        if self._cursor is None:
            raise RuntimeError("motion cursor is unset; provide an observation before moving")
        return self._cursor

    def _queue_gripper(self, value: float) -> None:
        target = self._require_cursor().copy()
        target[self._gripper_index] = value
        self._queue_target(target)

    def _queue_target(self, target: npt.NDArray[np.float64]) -> None:
        current = self._require_cursor()
        delta = target - current
        ratios: list[np.float64] = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index, distance in enumerate(np.abs(delta)):
                limit = self._step_limits[index]
                if distance == 0:
                    continue
                if limit <= 0:
                    raise ValueError(f"action dimension {index} is fixed and cannot move")
                ratios.append(np.divide(distance, limit))
            ratio = max(ratios, default=np.float64(0.0))
            headed_ratio = np.divide(ratio, 1.0 - _RELATIVE_HEADROOM)
        if not np.isfinite(headed_ratio):
            raise ValueError("motion distance is too large to interpolate safely")
        steps = max(1, math.ceil(float(headed_ratio)))
        for fraction in np.linspace(1.0 / steps, 1.0, steps):
            self._actions.append(Action(data=current + delta * fraction))
        self._actions[-1] = Action(data=target.copy())
        self._cursor = target.copy()
