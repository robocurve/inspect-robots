"""The agent's speed-limited tool surface over a bound action space.

Tools are generated from the embodiment's spaces, so the plugin never needs
embodiment-specific motion knowledge. Absolute targets are interpolated with a
per-step limit of ``min(max_speed_frac / hz, 0.05)`` times each dimension's
range. Displacements are split by the available box side, which the plugin
treats as the embodiment author's per-action limit.

Tool mistakes come back as structured error strings the LLM can correct.
Broken observations still raise, and unsupported configurations fail at build
(bind) time with a clear message.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.spaces import CANONICAL_STATE_UNITS, Box, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation

_ABSOLUTE_MODES = frozenset({"joint_pos", "eef_abs_pose"})
_DISPLACEMENT_MODES = frozenset({"eef_delta_pos", "eef_delta_pose", "joint_delta"})
_POSE_MODES = frozenset({"eef_abs_pose", "eef_delta_pose"})
_SAFE_ROT = frozenset({"none", "rot6d"})

_FALLBACK_HZ = 10.0
_MAX_DURATION_S = 10.0
_BACKSTOP_STEP_FRAC = 0.05
_RELATIVE_HEADROOM = 1e-6


class ToolsetError(Exception):
    """An action space / observation space this toolset cannot drive."""


@dataclass(frozen=True)
class ToolResult:
    """One executed tool call: an action chunk to play, or an error for the LLM.

    Exactly one of ``chunk``/``error`` is set. ``note`` is the human/LLM-facing
    confirmation text for successful calls.
    """

    chunk: ActionChunk | None = None
    error: str | None = None
    note: str = ""


class Toolset:
    """Schemas and execution for one embodiment, built via ``build_toolset``."""

    def __init__(
        self,
        *,
        absolute: bool,
        labels: tuple[str, ...],
        state_key: str | None,
        state_labels: tuple[str, tuple[str, ...]] | None,
        control_hz: float | None,
        bounds_text: str,
        low: npt.NDArray[np.float64],
        high: npt.NDArray[np.float64],
        step_limits: npt.NDArray[np.float64],
        pose: bool = False,
    ):
        self._absolute = absolute
        self._pose = pose
        self._labels = labels
        self._index_by_label = {label: i for i, label in enumerate(labels)}
        self._state_key = state_key
        self._state_labels = state_labels
        self._hz = control_hz
        self._resolved_hz = control_hz if control_hz is not None else _FALLBACK_HZ
        self._max_steps = math.ceil(_MAX_DURATION_S * self._resolved_hz)
        if pose:
            # Cartesian surfaces get a Cartesian name; the LLM should think in
            # workspace terms, not joints.
            self._move_tool = "move_to" if absolute else "move_by"
        else:
            self._move_tool = "move_joints" if absolute else "move_by"
        self._bounds_text = bounds_text
        self._low = low
        self._high = high
        self._step_limits = step_limits
        if absolute:
            self._positive_limits = np.zeros_like(high)
            self._negative_limits = np.zeros_like(low)
        else:
            self._positive_limits = high
            self._negative_limits = np.abs(low)

    def state_labels(self) -> tuple[str, tuple[str, ...]] | None:
        """Return the state field and per-element labels selected at build time."""
        return self._state_labels

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for this embodiment."""
        if self._absolute and self._pose:
            move_description = (
                "Move to absolute Cartesian end-effector targets (meters for "
                "positions, radians for rotations, per the dimension labels). "
                "The motion is a straight line interpolated at a fixed safe "
                "speed and the result reports its step count. Unnamed "
                "dimensions hold their current value. Coordinates are absolute "
                "in the embodiment's declared frame; on multi-arm embodiments "
                "each arm uses its own base frame and axes may differ between "
                "arms depending on mounting. Rotation dimensions are absolute "
                "targets measured relative to the trial's start orientation "
                "(0 means the start orientation) and interpolate linearly "
                "without wrapping, so prefer intermediate values for large "
                "rotations. " + self._bounds_text
            )
            values_key = "targets"
        elif self._absolute:
            move_description = (
                "Move to absolute joint/dimension targets. The motion is smoothly "
                "interpolated at a fixed safe speed and the result reports its step "
                "count. Unnamed dimensions hold their current value. " + self._bounds_text
            )
            values_key = "targets"
        else:
            move_description = (
                "Move BY the given displacement per dimension. The motion is split "
                "into steps that fit the per-action bounds and the result reports its "
                "step count. Unnamed dimensions do not move. " + self._bounds_text
            )
            values_key = "deltas"
        move = {
            "type": "function",
            "function": {
                "name": self._move_tool,
                "description": move_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        values_key: {
                            "type": "object",
                            "description": (
                                "Map of dimension name to value. Valid names: "
                                + ", ".join(self._labels)
                            ),
                        },
                    },
                    "required": [values_key],
                },
            },
        }
        done = {
            "type": "function",
            "function": {
                "name": "done",
                "description": (
                    "Declare the task finished. The trial ends; a scorer judges success."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
        }
        give_up = {
            "type": "function",
            "function": {
                "name": "give_up",
                "description": "Stop trying; the task cannot be completed. The trial ends.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            },
        }
        return [move, done, give_up]

    def execute(self, call: Any, observation: Observation) -> ToolResult:
        """Turn one tool call into an action chunk or an error string for the LLM."""
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, ValueError):
            return ToolResult(error=f"arguments for {call.name} are not valid JSON")
        if not isinstance(arguments, dict):
            return ToolResult(error=f"arguments for {call.name} must be a JSON object")
        if call.name in ("done", "give_up"):
            return self._stop(call.name, arguments, observation)
        if call.name != self._move_tool:
            return ToolResult(
                error=f"unknown tool {call.name!r}; available: {self._move_tool}, done, give_up"
            )
        return self._move(arguments, observation)

    def _current_state(self, observation: Observation) -> npt.NDArray[np.float64]:
        if self._state_key is None:
            return np.zeros(len(self._labels))
        return np.asarray(observation.state[self._state_key], dtype=np.float64)

    def _stop(self, name: str, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        data = self._current_state(observation)
        detail = str(arguments.get("summary") or arguments.get("reason") or "")
        action = Action(
            data=data,
            meta={"request_stop": True, "stop_reason": name, "stop_detail": detail},
        )
        return ToolResult(
            chunk=ActionChunk(actions=[action], control_hz=self._hz),
            note=f"{name}: {detail}",
        )

    def _move(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        current: npt.NDArray[np.float64] | None = None
        if self._absolute:
            # A broken sensor must end the trial before any argument
            # validation: a malformed tool call must not mask it behind a
            # correctable structured error.
            current = self._current_state(observation)
            if not bool(np.all(np.isfinite(current))):
                raise ValueError("proprioceptive reference contains a non-finite value")

        values_key = "targets" if self._absolute else "deltas"
        values = arguments.get(values_key)
        if not isinstance(values, dict) or not values:
            return ToolResult(error=f"{values_key} must be a non-empty object of name: value")

        vector = np.zeros(len(self._labels))
        named_indices: list[int] = []
        for label, raw in values.items():
            index = self._index_by_label.get(str(label))
            if index is None:
                return ToolResult(
                    error=f"unknown dimension {label!r}; valid names: {', '.join(self._labels)}"
                )
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            try:
                # Arbitrary-precision JSON integers overflow float() (and crash
                # np.isfinite outright); both must stay structured errors.
                coerced = float(raw)
            except OverflowError:
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            if not np.isfinite(coerced):
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            vector[index] = coerced
            named_indices.append(index)

        if self._absolute:
            assert current is not None
            return self._move_absolute(values, vector, named_indices, current)
        return self._move_displacement(vector, named_indices)

    def _move_absolute(
        self,
        values: dict[Any, Any],
        vector: npt.NDArray[np.float64],
        named_indices: list[int],
        current: npt.NDArray[np.float64],
    ) -> ToolResult:
        target = current.copy()
        for label, index in zip(values, named_indices, strict=True):
            value = vector[index]
            if self._step_limits[index] == 0:
                if value != self._low[index]:
                    return ToolResult(
                        error=f"dimension {label} is fixed at {float(self._low[index])!r}"
                    )
            elif value < self._low[index] or value > self._high[index]:
                return ToolResult(
                    error=(
                        f"target for {label} is outside "
                        f"[{float(self._low[index])!r}, {float(self._high[index])!r}]"
                    )
                )
            target[index] = value

        ratios: list[np.float64] = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index in named_indices:
                distance = np.abs(np.subtract(target[index], current[index]))
                limit = self._step_limits[index]
                if distance > 0 and limit > 0:
                    ratios.append(np.divide(distance, limit))
            ratio = max(ratios, default=np.float64(0.0))
            headed_ratio = np.divide(ratio, 1.0 - _RELATIVE_HEADROOM)
        if headed_ratio > self._max_steps:
            return self._cap_error()
        steps = max(1, math.ceil(float(headed_ratio)))

        fractions = np.linspace(1.0 / steps, 1.0, steps)
        actions = [
            Action(data=np.clip(current + (target - current) * fraction, self._low, self._high))
            for fraction in fractions
        ]
        actions[-1] = Action(data=np.clip(target.copy(), self._low, self._high))
        return self._success(actions, steps)

    def _move_displacement(
        self,
        vector: npt.NDArray[np.float64],
        named_indices: list[int],
    ) -> ToolResult:
        ratios: list[np.float64] = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index in named_indices:
                value = vector[index]
                if value == 0:
                    continue
                limit = self._positive_limits[index] if value > 0 else self._negative_limits[index]
                if limit == 0:
                    return ToolResult(
                        error=f"dimension {self._labels[index]} cannot move in that direction"
                    )
                ratios.append(np.divide(np.abs(value), limit))
            ratio = max(ratios, default=np.float64(0.0))
            headed_ratio = np.divide(ratio, 1.0 - _RELATIVE_HEADROOM)
        if headed_ratio > self._max_steps:
            return self._cap_error()
        steps = max(1, math.ceil(float(headed_ratio)))
        per_step = vector / steps
        for index in named_indices:
            # Subnormal deltas can underflow the split to exactly zero; a
            # success note over a zero-motion chunk would be a silent lie.
            if vector[index] != 0 and per_step[index] == 0:
                return ToolResult(
                    error=(
                        f"delta for {self._labels[index]} is too small to split "
                        "into executable steps"
                    )
                )
        actions = [Action(data=per_step.copy()) for _ in range(steps)]
        return self._success(actions, steps)

    def _cap_error(self) -> ToolResult:
        return ToolResult(
            error=(
                f"requested motion exceeds the {_MAX_DURATION_S:g}s playout cap; "
                "split the move into smaller motions"
            )
        )

    def _success(self, actions: list[Action], steps: int) -> ToolResult:
        note = f"executing {self._move_tool} over {steps} steps"
        if self._hz is not None:
            note += f" ({steps / self._hz:.1f}s)"
        return ToolResult(
            chunk=ActionChunk(actions=actions, control_hz=self._hz),
            note=note,
        )


def build_toolset(
    action_space: Box,
    observation_space: ObservationSpace,
    control_hz: float | None,
    max_speed_frac: float = 0.1,
) -> Toolset:
    """Validate an embodiment's spaces and build its agent-facing tools.

    Raises [`ToolsetError`][inspect_robots_agent._tools.ToolsetError] for
    configurations the motion layer cannot drive, before a trial begins.
    """
    semantics = action_space.semantics
    if semantics is None:
        raise ToolsetError(
            "the embodiment's action space declares no semantics; the agent cannot "
            "tell absolute targets from displacements"
        )
    mode = semantics.control_mode
    if mode in _POSE_MODES and semantics.rotation_repr not in _SAFE_ROT:
        raise ToolsetError(
            f"rotation_repr {semantics.rotation_repr!r} cannot be driven per-dimension; "
            f"only {sorted(_SAFE_ROT)} are supported"
        )
    if mode not in _ABSOLUTE_MODES | _DISPLACEMENT_MODES:
        raise ToolsetError(f"control_mode {mode!r} is not supported by the agent policy yet")
    if control_hz is not None and (not np.isfinite(control_hz) or control_hz <= 0):
        raise ToolsetError("control_hz must be finite and > 0 when declared")
    if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
        raise ToolsetError("max_speed_frac must be finite and > 0")
    resolved_hz = control_hz if control_hz is not None else _FALLBACK_HZ
    if not math.isfinite(_MAX_DURATION_S * resolved_hz):
        raise ToolsetError(f"control_hz {control_hz!r} is too large to derive a playout cap")
    if max_speed_frac / resolved_hz == 0.0:
        raise ToolsetError(
            f"max_speed_frac {max_speed_frac!r} underflows to a zero per-step "
            f"limit at control_hz {resolved_hz!r}"
        )
    if len(action_space.shape) != 1:
        raise ToolsetError(
            f"only 1-D (vector) action spaces are supported, got shape {action_space.shape}"
        )

    absolute = mode in _ABSOLUTE_MODES
    dim = action_space.dim
    state_key: str | None = None
    state_spec = observation_space.state
    if absolute:
        if state_spec is None:
            raise ToolsetError(
                "absolute-target control needs a StateSpec on the embodiment's "
                "observation space to locate the proprioceptive reference"
            )
        matching = [field.key for field in state_spec.fields if field.shape == (dim,)]
        if len(matching) != 1:
            raise ToolsetError(
                f"absolute-target control needs exactly one state field with shape "
                f"({dim},); found {matching or 'none'}"
            )
        state_key = matching[0]

    low, high = action_space.low, action_space.high
    if (
        low is None
        or high is None
        or not bool(np.all(np.isfinite(low)) and np.all(np.isfinite(high)))
    ):
        mode_name = "absolute-target" if absolute else "displacement"
        raise ToolsetError(f"{mode_name} control needs finite low and high bounds")
    low64 = np.asarray(low, dtype=np.float64)
    high64 = np.asarray(high, dtype=np.float64)
    # The range must be finite in the box's native dtype, not just in float64:
    # DeltaLimitApprover subtracts without promoting, so float32 bounds like
    # [-3e38, 3e38] overflow for it even though the float64 difference is fine.
    with np.errstate(over="ignore"):
        native_range = np.asarray(high - low, dtype=np.float64)
    if not bool(np.all(np.isfinite(native_range)) and np.all(np.isfinite(high64 - low64))):
        raise ToolsetError("action-space range (high - low) overflows; bounds are too large")
    if not absolute and (bool(np.any(low64 > 0)) or bool(np.any(high64 < 0))):
        raise ToolsetError("displacement control bounds must contain zero in every dimension")
    # DeltaLimitApprover derives its default limit in the box's native dtype;
    # reproduce that arithmetic exactly so our ceiling can never exceed it.
    native_backstop = np.asarray(_BACKSTOP_STEP_FRAC * (high - low), dtype=np.float64)
    step_limits = np.zeros_like(high64)
    if absolute:
        # Interpolants snap to the float grid at the bounds' magnitude. If that
        # grid is coarse relative to the backstop (offset boxes like
        # [1e16, 1e16 + 2], or ranges whose 5% underflows to zero), emitted
        # steps jump by more than the backstop allows and motions silently
        # truncate — the exact failure this plugin exists to remove.
        spacing = np.spacing(np.maximum(np.abs(low64), np.abs(high64)))
        movable = high64 > low64
        if bool(np.any(movable & (spacing > 5e-7 * native_backstop))):
            raise ToolsetError(
                "bounds are too coarse at this magnitude for speed-limited "
                "interpolation (float spacing exceeds the per-step budget)"
            )
        resolved = control_hz if control_hz is not None else _FALLBACK_HZ
        step_frac = min(max_speed_frac / resolved, _BACKSTOP_STEP_FRAC)
        step_limits = np.minimum(step_frac * (high64 - low64), native_backstop)
        # The frac/hz quotient can be nonzero yet still underflow to a zero
        # limit once multiplied by a small range; a movable dimension with a
        # zero limit would be misreported as fixed.
        if bool(np.any(movable & (step_limits == 0))):
            raise ToolsetError(
                f"max_speed_frac {max_speed_frac!r} underflows the derived "
                "per-step limit for a movable dimension; increase it"
            )

    authored_labels = semantics.dim_labels
    labels = authored_labels or tuple(str(i) for i in range(dim))
    state_labels: tuple[str, tuple[str, ...]] | None = None
    if authored_labels is not None:
        if state_key is not None:
            state_labels = (state_key, authored_labels)
        elif state_spec is not None:
            candidates = [
                field.key
                for field in state_spec.fields
                if field.shape == (dim,) and field.key in CANONICAL_STATE_UNITS
            ]
            if len(candidates) == 1:
                state_labels = (candidates[0], authored_labels)
    pairs = ", ".join(
        f"{label}: [{lo:.4g}, {hi:.4g}]"
        for label, lo, hi in zip(labels, low64.tolist(), high64.tolist(), strict=False)
    )
    bounds_text = f"Per-dimension bounds: {pairs}."

    return Toolset(
        absolute=absolute,
        labels=labels,
        state_key=state_key,
        state_labels=state_labels,
        control_hz=control_hz,
        bounds_text=bounds_text,
        low=low64,
        high=high64,
        step_limits=step_limits,
        pose=mode in _POSE_MODES,
    )
