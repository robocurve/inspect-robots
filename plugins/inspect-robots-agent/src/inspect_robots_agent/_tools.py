"""The agent's tool surface over a bound action space (plan 0008 §4b/§4c).

Tools are generated from the embodiment's spaces — the plugin never knows
what a "YAM" or an "arm" is. The control mode picks the motion tool:

- Absolute-target modes (``joint_pos``, ``eef_abs_pose``): ``move_joints``
  with named partial targets, linearly interpolated from the current
  observed state; hold-still repeats the current state.
- Displacement modes (``eef_delta_pos``, ``eef_delta_pose``,
  ``joint_delta``): ``move_by`` splits the requested displacement evenly
  across the chunk's steps; hold-still is all zeros.

Tool mistakes (unknown label, non-finite value, bad duration, malformed
JSON) come back as structured error strings the LLM sees and can correct —
never exceptions. Unsupported configurations fail at build (= ``bind``)
time with a clear message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.spaces import Box, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation

_ABSOLUTE_MODES = frozenset({"joint_pos", "eef_abs_pose"})
_DISPLACEMENT_MODES = frozenset({"eef_delta_pos", "eef_delta_pose", "joint_delta"})
_POSE_MODES = frozenset({"eef_abs_pose", "eef_delta_pose"})
_SAFE_ROT = frozenset({"none", "rot6d"})

_FALLBACK_HZ = 10.0
_MAX_DURATION_S = 10.0


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
    """Schemas + execution for one bound embodiment. Built via ``build_toolset``."""

    def __init__(
        self,
        *,
        absolute: bool,
        labels: tuple[str, ...],
        state_key: str | None,
        control_hz: float | None,
        bounds_text: str,
    ):
        self._absolute = absolute
        self._labels = labels
        self._index_by_label = {label: i for i, label in enumerate(labels)}
        self._state_key = state_key
        self._hz = control_hz
        self._move_tool = "move_joints" if absolute else "move_by"
        self._bounds_text = bounds_text

    # -- schemas ---------------------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-format tool definitions for this embodiment."""
        if self._absolute:
            move_description = (
                "Move to absolute joint/dimension targets, smoothly interpolated "
                "from the current pose over duration_s. Unnamed dimensions hold "
                "their current value. " + self._bounds_text
            )
            values_key = "targets"
        else:
            move_description = (
                "Move BY the given displacement per dimension, split evenly over "
                "duration_s. Unnamed dimensions do not move. " + self._bounds_text
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
                        "duration_s": {
                            "type": "number",
                            "description": (
                                f"Motion duration in seconds (0 < d <= {_MAX_DURATION_S})"
                            ),
                        },
                    },
                    "required": [values_key, "duration_s"],
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

    # -- execution ---------------------------------------------------------------

    def execute(self, call: Any, observation: Observation) -> ToolResult:
        """Turn one tool call into an ActionChunk, or an error string for the LLM."""
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
        # Hold still per control mode: repeat the pose (absolute) or move by
        # nothing (displacement), flagged for rollout's policy-stop channel.
        data = self._current_state(observation) if self._absolute else np.zeros(len(self._labels))
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
        values_key = "targets" if self._absolute else "deltas"
        values = arguments.get(values_key)
        duration = arguments.get("duration_s")
        if not isinstance(values, dict) or not values:
            return ToolResult(error=f"{values_key} must be a non-empty object of name: value")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            return ToolResult(error="duration_s must be a number")
        if not 0 < float(duration) <= _MAX_DURATION_S:
            return ToolResult(
                error=f"duration_s must be in (0, {_MAX_DURATION_S}] seconds, got {duration}"
            )

        vector = np.zeros(len(self._labels))
        for label, raw in values.items():
            index = self._index_by_label.get(str(label))
            if index is None:
                return ToolResult(
                    error=f"unknown dimension {label!r}; valid names: {', '.join(self._labels)}"
                )
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not np.isfinite(raw):
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            vector[index] = float(raw)

        hz = self._hz if self._hz is not None else _FALLBACK_HZ
        steps = max(1, round(float(duration) * hz))
        if self._absolute:
            current = self._current_state(observation)
            target = current.copy()
            for label in values:
                target[self._index_by_label[str(label)]] = vector[self._index_by_label[str(label)]]
            fractions = np.linspace(1.0 / steps, 1.0, steps)
            actions = [Action(data=current + (target - current) * f) for f in fractions]
        else:
            per_step = vector / steps
            actions = [Action(data=per_step.copy()) for _ in range(steps)]
        return ToolResult(
            chunk=ActionChunk(actions=actions, control_hz=self._hz),
            note=f"executing {self._move_tool} over {steps} steps ({duration}s)",
        )


def build_toolset(
    action_space: Box, observation_space: ObservationSpace, control_hz: float | None
) -> Toolset:
    """Validate a (action space, observation space) pairing and build its tools.

    Raises [`ToolsetError`][inspect_robots_agent._tools.ToolsetError] with a
    clear message for configurations the motion layer cannot drive — at bind
    time, never mid-trial.
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

    absolute = mode in _ABSOLUTE_MODES
    dim = action_space.dim
    state_key: str | None = None
    if absolute:
        spec = observation_space.state
        if spec is None:
            raise ToolsetError(
                "absolute-target control needs a StateSpec on the embodiment's "
                "observation space to locate the proprioceptive reference"
            )
        matching = [f.key for f in spec.fields if f.shape == (dim,)]
        if len(matching) != 1:
            raise ToolsetError(
                f"absolute-target control needs exactly one state field with shape "
                f"({dim},); found {matching or 'none'}"
            )
        state_key = matching[0]

    labels = semantics.dim_labels or tuple(str(i) for i in range(dim))
    low, high = action_space.low, action_space.high
    if low is not None and high is not None:
        pairs = ", ".join(
            f"{label}: [{lo:.4g}, {hi:.4g}]"
            for label, lo, hi in zip(labels, low.tolist(), high.tolist(), strict=False)
        )
        bounds_text = f"Per-dimension bounds: {pairs}."
    else:
        bounds_text = "The action space declares no bounds; move conservatively."

    return Toolset(
        absolute=absolute,
        labels=labels,
        state_key=state_key,
        control_hz=control_hz,
        bounds_text=bounds_text,
    )
