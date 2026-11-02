"""Tool libraries: what a language-model policy can call to act, and to end a trial.

The paper's policy adapter gives a language model two libraries of tools: control
tools, each of which yields a sequence of actions, and termination tools (done,
give up with a stated reason) that end the trial. A [`Tool`][inspect_robots.tools.Tool]
pairs a JSON tool schema with the function that runs it, and a
[`ToolLibrary`][inspect_robots.tools.ToolLibrary] is an ordered set of tools that
validates a call and turns it into a [`ToolResult`][inspect_robots.tools.ToolResult]:
an action chunk, or an error string the model can correct.

[`control_tools`][inspect_robots.tools.control_tools] builds the speed-limited
motion tools from an embodiment's spaces, so no embodiment-specific motion
knowledge is needed. Absolute targets are interpolated with a declared
per-dimension limit where available and a range-derived backstop otherwise.
Dimensions a call leaves unnamed keep the last target commanded in the trial (the
measured state before the first move), so tracking sag never becomes the next
command. Displacements are split by the available box side, which is treated as
the embodiment author's per-action limit.
[`termination_tools`][inspect_robots.tools.termination_tools] builds ``done`` and
``give_up``, which hold the current setpoint and end the trial with a
[`Termination`][inspect_robots.types.Termination].

Broken observations still raise, and unsupported configurations fail at build
time with a [`ToolError`][inspect_robots.tools.ToolError].
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.spaces import (
    ABSOLUTE_CONTROL_MODES,
    CANONICAL_STATE_UNITS,
    Box,
    ObservationSpace,
    reference_state_fields,
)
from inspect_robots.types import Action, ActionChunk, Observation, Termination

_DISPLACEMENT_MODES = frozenset({"eef_delta_pos", "eef_delta_pose", "joint_delta"})
_POSE_MODES = frozenset({"eef_abs_pose", "eef_delta_pose"})
_SAFE_ROT = frozenset({"none", "rot6d"})

_FALLBACK_HZ = 10.0
_MAX_DURATION_S = 10.0
_BACKSTOP_STEP_FRAC = 0.05
_DEFAULT_SPEED_FRAC = 0.1
_RELATIVE_HEADROOM = 1e-6
#: The longest a single `sleep` may hold, in seconds.
_SLEEP_MAX_S = 180.0
#: Readback-minus-setpoint offsets below this are measurement noise, not listed.
_SETPOINT_OFFSET_MIN = 5e-3
_HINDSIGHT_DESCRIPTION = (
    "What do you know now that you wish you had known at the start of this episode? "
    "Concrete, transferable facts about this rig, task, or embodiment (camera mounting "
    "and extrinsics, table and base geometry, gripper axis and offsets, controller "
    "behavior, metric scale), written as advice to a future agent attempting the same "
    "task. Say 'none' if nothing qualifies."
)

#: A motion hook over the exact clipped absolute waypoints for one move call.
#:
#: The input is a read-only stacked float64 copy with shape ``(steps, dim)``.
#: Its first row is the first commanded waypoint and its last row is the final
#: target. Return ``None`` to allow the motion or a non-empty reason to reject
#: it. Exceptions propagate unchanged.
PreCheck = Callable[[npt.NDArray[np.float64]], str | None]


CONTROL = "control"
TERMINATION = "termination"


class ToolError(Exception):
    """An action space / observation space these tools cannot drive."""


@dataclass(frozen=True, eq=False)
class ToolResult:
    """The policy-relevant effects of one validated tool call.

    ``note`` is the result confirmation text for a successful call, distinct
    from the call's required ``note`` argument. ``capture`` encodes a
    well-formed capture request: ``None`` means cameras were omitted and the
    policy should resolve every available camera, ``()`` means the observation
    had no images to show, and a non-empty tuple names the requested cameras.
    Motion targets are carried separately so their measured residual can be
    reported after playout.
    """

    chunk: ActionChunk | None = None
    error: str | None = None
    note: str = ""
    capture: tuple[str, ...] | None = None
    target: npt.NDArray[np.float64] | None = None
    #: Seconds a validated ``sleep`` holds the current setpoint (its chunk is in ``chunk``).
    sleep: float | None = None


#: Runs one tool call: its decoded arguments object and the current observation.
ToolRun = Callable[[dict[str, Any], Observation], ToolResult]


@dataclass(frozen=True)
class Tool:
    """One tool a language-model policy can call.

    ``schema`` is the tool definition in the OpenAI function format
    (``{"type": "function", "function": {"name", "description", "parameters"}}``);
    wire formats convert it for other providers. ``run`` receives the decoded
    arguments object and the current observation. ``kind`` is ``"control"`` or
    ``"termination"``. Libraries list tools by ascending ``order``, then in the
    order they were added, so a model always sees the same tool list.
    """

    name: str
    schema: Mapping[str, Any]
    run: ToolRun
    kind: str = CONTROL
    order: int = 100


class ToolLibrary:
    """An ordered set of uniquely named tools that turns calls into results."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        ordered = sorted(tools, key=lambda tool: tool.order)
        names = [tool.name for tool in ordered]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ToolError(f"duplicate tool name(s): {', '.join(duplicates)}")
        self._tools = tuple(ordered)
        self._by_name = {tool.name: tool for tool in ordered}

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __add__(self, other: ToolLibrary) -> ToolLibrary:
        return ToolLibrary((*self._tools, *other._tools))

    @property
    def names(self) -> tuple[str, ...]:
        """Tool names in listing order."""
        return tuple(tool.name for tool in self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """The tool definitions, in listing order, as fresh dictionaries."""
        return [json.loads(json.dumps(tool.schema)) for tool in self._tools]

    def execute(self, name: str, arguments: str, observation: Observation) -> ToolResult:
        """Decode ``arguments`` (a JSON object) and run tool ``name`` on ``observation``.

        A malformed call or an unknown tool returns a ``ToolResult`` whose
        ``error`` the model can read and correct.
        """
        try:
            decoded = json.loads(arguments)
        except (TypeError, ValueError):
            return ToolResult(error=f"arguments for {name} are not valid JSON")
        if not isinstance(decoded, dict):
            return ToolResult(error=f"arguments for {name} must be a JSON object")
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult(error=f"unknown tool {name!r}; available: {', '.join(self.names)}")
        return tool.run(decoded, observation)


class ControlTools:
    """The motion tools for one embodiment and the trial state they share.

    Build with [`control_tools`][inspect_robots.tools.control_tools]. ``library``
    holds the tools (``move_*``, ``sleep`` and, for on-demand images,
    ``take_pic``); the methods expose what an adapter reports around them: the
    held setpoint, bounds, pinned dimensions and the residual after a move.
    """

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
        images: str = "always",
        cameras: tuple[str, ...] = (),
        pre_check: PreCheck | None = None,
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
        self._images = images
        self._cameras = cameras
        self._pre_check = pre_check
        #: The clipped target of the trial's last successful absolute move;
        #: ``None`` before the first one and after ``start_trial()``.
        self._held_target: npt.NDArray[np.float64] | None = None
        if absolute:
            self._positive_limits = np.zeros_like(high)
            self._negative_limits = np.zeros_like(low)
        else:
            self._positive_limits = high
            self._negative_limits = np.abs(low)

    def state_labels(self) -> tuple[str, tuple[str, ...]] | None:
        """Return the state field and per-element labels selected at build time."""
        return self._state_labels

    def setpoint_lines(self, observation: Observation) -> list[str]:
        """Render the held setpoint under the readback, and the dimensions where they differ.

        Absolute modes only: displacements have no setpoint. Before the first
        move of a trial the line says so, since that move starts from the
        readback. The offset line lists ``readback - setpoint`` per dimension,
        omitting magnitudes under ``_SETPOINT_OFFSET_MIN`` as noise; it is
        skipped when the readback cannot be compared (absent, misshapen, or
        non-finite).
        """
        if not self._absolute:
            return []
        held = self._held_target
        if held is None:
            return ["setpoint: none yet; the first move starts from the readback."]
        rounded = np.round(held, 4).tolist()
        pairs = " ".join(
            f"{label}={value}" for label, value in zip(self._labels, rounded, strict=True)
        )
        lines = [f"setpoint: {pairs}"]
        raw_state = observation.state.get(self._state_key) if self._state_key else None
        if raw_state is None:
            return lines
        state = np.asarray(raw_state, dtype=np.float64)
        if state.shape != held.shape or not bool(np.all(np.isfinite(state))):
            return lines
        offsets = state - held
        notable = [
            f"{label}={offset:+.4f}"
            for label, offset in zip(self._labels, offsets.tolist(), strict=True)
            if abs(offset) >= _SETPOINT_OFFSET_MIN
        ]
        if notable:
            lines.append("readback minus setpoint: " + " ".join(notable))
        else:
            lines.append(f"readback minus setpoint: none over {_SETPOINT_OFFSET_MIN:g}.")
        return lines

    def start_trial(self) -> None:
        """Forget the held target so the next move starts from the measured state.

        Call this at every trial reset: the previous trial's last command says
        nothing about where a freshly reset embodiment is.
        """
        self._held_target = None

    @property
    def resolved_hz(self) -> float:
        """The control rate motions are timed at (the fallback when the embodiment names none)."""
        return self._resolved_hz

    def hold_steps(self, seconds: float) -> int:
        """Steps a stationary hold of ``seconds`` takes at the resolved control rate, at least 1."""
        return max(1, round(seconds * self._resolved_hz))

    def hold_chunk(self, target: npt.NDArray[np.float64], steps: int) -> ActionChunk:
        """A chunk that re-sends ``target`` for ``steps`` steps: a ``sleep``."""
        actions = [Action(data=target.copy()) for _ in range(steps)]
        actions[-1] = Action(data=target.copy(), meta={"chunk_final": True})
        return ActionChunk(actions=actions, control_hz=self._hz)

    @property
    def bounds_text(self) -> str:
        """Formatted per-dimension bounds text."""
        return self._bounds_text

    @property
    def pinned_labels(self) -> tuple[str, ...]:
        """Dimensions fixed/pinned where movement is prohibited."""
        if self._absolute:
            return tuple(
                label
                for label, limit in zip(self._labels, self._step_limits, strict=True)
                if limit == 0
            )
        return tuple(
            label
            for label, pos, neg in zip(
                self._labels, self._positive_limits, self._negative_limits, strict=True
            )
            if pos == 0 and neg == 0
        )

    @property
    def library(self) -> ToolLibrary:
        """The control tools: the move tool, ``sleep``, and ``take_pic`` for on-demand images."""
        tools = [
            Tool(self._move_tool, self._move_schema(), self._move, CONTROL, order=0),
            Tool("sleep", self._sleep_schema(), self._sleep, CONTROL, order=20),
        ]
        if self._images == "on_demand":
            tools.append(
                Tool("take_pic", self._take_pic_schema(), self._take_pic, CONTROL, order=30)
            )
        return ToolLibrary(tools)

    @property
    def control_hz(self) -> float | None:
        """The declared control rate stamped on every chunk (``None`` if undeclared)."""
        return self._hz

    def hold_target(self, observation: Observation) -> npt.NDArray[np.float64]:
        """The setpoint the controller is tracking: the last absolute target, else the readback."""
        held = self._held_target
        return self._current_state(observation) if held is None else held.copy()

    def _move_schema(self) -> dict[str, Any]:
        if self._absolute and self._pose:
            move_description = (
                "Move to absolute Cartesian end-effector targets (meters for "
                "positions, radians for rotations, per the dimension labels). "
                "The motion is a straight line interpolated at a fixed safe "
                "speed and the result reports its step count. Unnamed "
                "dimensions keep their last commanded target (their measured "
                "value before the first move of a trial), so a gripper squeeze "
                "or a rotation stays commanded without naming it again. "
                "Coordinates are absolute "
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
                "count. Unnamed dimensions keep their last commanded target (their "
                "measured value before the first move of a trial). " + self._bounds_text
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
                        "note": {
                            "type": "string",
                            "description": (
                                # An operator-facing status caption. Wording that asks the model
                                # to explain why it chose the motion reads as a request for its
                                # hidden reasoning, which some models refuse on every call.
                                "One or two plain sentences for the human operator watching "
                                "the robot: describe the scene as it currently looks (images, "
                                "if any, and state) and state the goal of this motion. Shown "
                                "live and in the saved transcript."
                            ),
                        },
                    },
                    "required": [values_key, "note"],
                },
            },
        }
        return move

    def _take_pic_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "take_pic",
                "description": (
                    "Capture the current frame from one or more cameras. "
                    "Omit cameras to capture every available camera. "
                    "Declared cameras: " + ", ".join(self._cameras) + "."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cameras": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Camera names to capture. Valid declared names: "
                                + ", ".join(self._cameras)
                            ),
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "What you need to inspect and why seeing a camera "
                                "frame is necessary before the next decision. The "
                                "user reads these notes live and in the saved "
                                "transcript."
                            ),
                        },
                    },
                    "required": ["note"],
                },
            },
        }

    def _sleep_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "sleep",
                "description": (
                    "Hold every setpoint exactly where it is for `seconds` (at most "
                    f"{_SLEEP_MAX_S:g}) and do nothing else. Sleeping spends environment "
                    "steps and wall-clock time, not LLM calls. Use it to let hot motors "
                    "cool after moving the arm to a resting pose, or to let the scene "
                    "settle."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "number", "description": "How long to hold."},
                        "note": {
                            "type": "string",
                            "description": (
                                "What you observe right now and why waiting is worth the "
                                "time. Written for the user, in one or two plain sentences."
                            ),
                        },
                    },
                    "required": ["seconds", "note"],
                },
            },
        }

    def residual(
        self, target: npt.NDArray[np.float64], observation: Observation
    ) -> tuple[str, float] | None:
        """Return the largest finite target offset, or ``None`` when it cannot be measured."""
        if self._state_key is None:
            return None
        raw_state = observation.state.get(self._state_key)
        if raw_state is None:
            return None
        try:
            state = np.asarray(raw_state, dtype=np.float64)
        except Exception:  # pragma: no cover
            # Delivery must survive third-party state containers whose array
            # coercion raises something less conventional than ValueError.
            return None
        if state.shape != (len(self._labels),):
            return None
        try:
            difference = np.abs(np.subtract(target, state))
        except Exception:  # pragma: no cover
            return None
        if difference.shape != state.shape or not bool(np.all(np.isfinite(difference))):
            return None
        index = int(np.argmax(difference))
        return self._labels[index], float(difference[index])

    def _current_state(self, observation: Observation) -> npt.NDArray[np.float64]:
        if self._state_key is None:
            return np.zeros(len(self._labels))
        return np.asarray(observation.state[self._state_key], dtype=np.float64)

    def _take_pic(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        call_note = arguments.get("note")
        if not isinstance(call_note, str) or not call_note.strip():
            return ToolResult(
                error="note is required: describe what you observe and why you chose this capture"
            )

        requested = arguments.get("cameras")
        if "cameras" not in arguments:
            if not observation.images:
                return ToolResult(capture=())
            return ToolResult(capture=None)
        if (
            not isinstance(requested, list)
            or not requested
            or any(not isinstance(name, str) for name in requested)
        ):
            return ToolResult(error="cameras must be a non-empty list of strings when provided")
        if not observation.images:
            return ToolResult(capture=())
        present = tuple(observation.images)
        unknown = [name for name in requested if name not in observation.images]
        if unknown:
            return ToolResult(
                error=(
                    f"unknown or unavailable camera {unknown[0]!r}; "
                    f"available now: {', '.join(repr(name) for name in present)}"
                )
            )
        return ToolResult(capture=tuple(requested))

    def _sleep(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        """Validate a ``sleep`` and build the stationary chunk that holds the current setpoint.

        Absolute surfaces hold the last commanded target (the measured state
        before any move), so a gripper squeeze stays squeezed; displacement
        surfaces hold with zero deltas. Nothing moves, so the playout cap for
        motions does not apply.
        """
        call_note = arguments.get("note")
        if not isinstance(call_note, str) or not call_note.strip():
            return ToolResult(error="note is required: describe what you observe and why you wait")
        seconds = arguments.get("seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            return ToolResult(error="seconds must be a positive number")
        seconds = min(float(seconds), _SLEEP_MAX_S)
        if self._absolute:
            held = self._held_target
            if held is None:
                current = self._current_state(observation)
                if not bool(np.all(np.isfinite(current))):
                    raise ValueError("proprioceptive reference contains a non-finite value")
                held = np.clip(current, self._low, self._high)
                self._held_target = held.copy()
            target = held.copy()
        else:
            target = np.zeros(len(self._labels))
        steps = self.hold_steps(seconds)
        return ToolResult(
            chunk=self.hold_chunk(target, steps),
            note=f"sleeping {seconds:g} s ({steps} steps) holding the current setpoint",
            sleep=seconds,
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

        call_note = arguments.get("note")
        if not isinstance(call_note, str) or not call_note.strip():
            return ToolResult(
                error=("note is required: describe what you observe and why you chose this motion")
            )

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
        held = self._held_target
        target = (current if held is None else held).copy()
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

        ratios: list[float] = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index in named_indices:
                distance = np.abs(np.subtract(target[index], current[index]))
                limit = self._step_limits[index]
                if distance > 0 and limit > 0:
                    ratios.append(float(np.divide(distance, limit)))
            ratio = max(ratios, default=0.0)
            headed_ratio = np.divide(ratio, 1.0 - _RELATIVE_HEADROOM)
        if headed_ratio > self._max_steps:
            return self._cap_error()
        steps = max(1, math.ceil(float(headed_ratio)))

        # Named dimensions interpolate from where the arm is. Unnamed ones are
        # already commanded at the held target (the controller keeps tracking
        # the previous setpoint between calls), so they sit there from the
        # first waypoint: interpolating them from the measured state would
        # briefly command the tracking sag, e.g. loosen a gripper squeeze.
        start = target.copy()
        start[named_indices] = current[named_indices]
        fractions = np.linspace(1.0 / steps, 1.0, steps)
        actions = [
            Action(data=np.clip(start + (target - start) * fraction, self._low, self._high))
            for fraction in fractions
        ]
        actions[-1] = Action(data=np.clip(target.copy(), self._low, self._high))
        clipped_target = np.clip(target.copy(), self._low, self._high)
        if self._pre_check is not None:
            waypoints: npt.NDArray[np.float64] = np.asarray(
                np.stack([action.data for action in actions]),
                dtype=np.float64,
            )
            waypoints.flags.writeable = False
            if message := self._pre_check(waypoints):
                return ToolResult(error="pre-check rejected this motion: " + message)
        self._held_target = clipped_target.copy()
        return self._success(actions, steps, target=clipped_target)

    def _move_displacement(
        self,
        vector: npt.NDArray[np.float64],
        named_indices: list[int],
    ) -> ToolResult:
        ratios: list[float] = []
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
                ratios.append(float(np.divide(np.abs(value), limit)))
            ratio = max(ratios, default=0.0)
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

    def _success(
        self,
        actions: list[Action],
        steps: int,
        *,
        target: npt.NDArray[np.float64] | None = None,
    ) -> ToolResult:
        # Both callers build at least one step, so there is always a last action.
        last = actions[-1]
        actions[-1] = Action(data=last.data, meta={**last.meta, "chunk_final": True})
        note = f"executing {self._move_tool} over {steps} steps"
        if self._hz is not None:
            note += f" ({steps / self._hz:.1f}s)"
        return ToolResult(
            chunk=ActionChunk(actions=actions, control_hz=self._hz),
            note=note,
            target=target,
        )


def control_tools(
    action_space: Box,
    observation_space: ObservationSpace,
    control_hz: float | None,
    max_speed_frac: float = 0.1,
    images: str = "always",
    pre_check: PreCheck | None = None,
) -> ControlTools:
    """Validate an embodiment's spaces and build its control tools.

    ``max_speed_frac`` caps motion speed as a fraction of each dimension's range
    per second. ``images="on_demand"`` adds ``take_pic``. ``pre_check`` vets
    each absolute move's waypoints before it is accepted. Raises
    [`ToolError`][inspect_robots.tools.ToolError] for configurations the motion
    layer cannot drive, before a trial begins.
    """
    if images == "on_demand" and not observation_space.cameras:
        raise ToolError(
            "images='on_demand' needs at least one camera declared by the embodiment.\n"
            "fix: drop -P images=on_demand"
        )
    semantics = action_space.semantics
    if semantics is None:
        raise ToolError(
            "the embodiment's action space declares no semantics; the agent cannot "
            "tell absolute targets from displacements"
        )
    mode = semantics.control_mode
    if mode in _POSE_MODES and semantics.rotation_repr not in _SAFE_ROT:
        raise ToolError(
            f"rotation_repr {semantics.rotation_repr!r} cannot be driven per-dimension; "
            f"only {sorted(_SAFE_ROT)} are supported"
        )
    if mode not in ABSOLUTE_CONTROL_MODES | _DISPLACEMENT_MODES:
        raise ToolError(f"control_mode {mode!r} is not supported by the agent policy yet")
    if pre_check is not None and mode in _DISPLACEMENT_MODES:
        raise ToolError(
            "pre_check cannot be used with displacement control modes: they emit "
            "per-step delta vectors that a configuration checker would misread "
            "(plan 0031)"
        )
    if control_hz is not None and (not np.isfinite(control_hz) or control_hz <= 0):
        raise ToolError("control_hz must be finite and > 0 when declared")
    if not np.isfinite(max_speed_frac) or max_speed_frac <= 0:
        raise ToolError("max_speed_frac must be finite and > 0")
    resolved_hz = control_hz if control_hz is not None else _FALLBACK_HZ
    if not math.isfinite(_MAX_DURATION_S * resolved_hz):
        raise ToolError(f"control_hz {control_hz!r} is too large to derive a playout cap")
    if mode in _DISPLACEMENT_MODES and max_speed_frac / resolved_hz == 0.0:
        raise ToolError(
            f"max_speed_frac {max_speed_frac!r} underflows to a zero per-step "
            f"limit at control_hz {resolved_hz!r}"
        )
    if len(action_space.shape) != 1:
        raise ToolError(
            f"only 1-D (vector) action spaces are supported, got shape {action_space.shape}"
        )

    absolute = mode in ABSOLUTE_CONTROL_MODES
    dim = action_space.dim
    state_key: str | None = None
    state_spec = observation_space.state
    if absolute:
        if state_spec is None:
            raise ToolError(
                "absolute-target control needs a StateSpec on the embodiment's "
                "observation space to locate the proprioceptive reference"
            )
        matching = reference_state_fields(state_spec, dim, mode)
        if len(matching) != 1:
            raise ToolError(
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
        raise ToolError(f"{mode_name} control needs finite low and high bounds")
    low64 = np.asarray(low, dtype=np.float64)
    high64 = np.asarray(high, dtype=np.float64)
    # The range must be finite in the box's native dtype, not just in float64:
    # DeltaLimitGuard subtracts without promoting, so float32 bounds like
    # [-3e38, 3e38] overflow for it even though the float64 difference is fine.
    with np.errstate(over="ignore"):
        native_range = np.asarray(high - low, dtype=np.float64)
    if not bool(np.all(np.isfinite(native_range)) and np.all(np.isfinite(high64 - low64))):
        raise ToolError("action-space range (high - low) overflows; bounds are too large")
    if not absolute and (bool(np.any(low64 > 0)) or bool(np.any(high64 < 0))):
        raise ToolError("displacement control bounds must contain zero in every dimension")
    # DeltaLimitGuard derives undeclared dimensions in the box's native
    # dtype; reproduce that arithmetic exactly before substituting declarations.
    native_backstop = np.asarray(_BACKSTOP_STEP_FRAC * (high - low), dtype=np.float64)
    step_limits = np.zeros_like(high64)
    if absolute:
        declared = semantics.max_step or (None,) * dim
        declared_mask = np.fromiter(
            (entry is not None for entry in declared), dtype=np.bool_, count=dim
        )
        effective_budget = native_backstop.copy()
        for index, entry in enumerate(declared):
            if entry is not None:
                effective_budget[index] = entry
        # Interpolants snap to the float grid at the bounds' magnitude. If that
        # grid is coarse relative to the effective per-dimension budget (offset
        # boxes like [1e16, 1e16 + 2], or ranges whose 5% underflows to zero),
        # emitted steps jump by more than the budget allows and motions silently
        # truncate — the exact failure this plugin exists to remove.
        spacing = np.spacing(np.maximum(np.abs(low64), np.abs(high64)))
        movable = high64 > low64
        if bool(np.any(movable & (spacing > 5e-7 * effective_budget))):
            raise ToolError(
                "bounds are too coarse at this magnitude for speed-limited "
                "interpolation (float spacing exceeds the per-step budget)"
            )
        resolved = control_hz if control_hz is not None else _FALLBACK_HZ
        step_frac = min(max_speed_frac / resolved, _BACKSTOP_STEP_FRAC)
        step_limits = np.minimum(step_frac * (high64 - low64), native_backstop)
        declared_scale = min(max_speed_frac / _DEFAULT_SPEED_FRAC, 1.0)
        step_limits[declared_mask] = effective_budget[declared_mask] * declared_scale
        # The frac/hz quotient can be nonzero yet still underflow to a zero
        # limit once multiplied by a small range or declared max_step; a
        # movable dimension with a zero limit would be misreported as fixed.
        if bool(np.any(movable & (step_limits == 0))):
            raise ToolError(
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

    return ControlTools(
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
        images=images,
        cameras=tuple(camera.name for camera in observation_space.cameras),
        pre_check=pre_check,
    )


def termination_tools(control: ControlTools) -> ToolLibrary:
    """``done`` and ``give_up``: end the trial, holding the setpoint ``control`` is tracking.

    Ending a trial must not move anything, so each re-sends the setpoint the
    controller already tracks (a gripper squeeze included) rather than the
    measured state, which lags it by the tracking sag. The call's ``summary``
    or ``reason`` becomes the stated reason of the
    [`Termination`][inspect_robots.types.Termination]; a ``hindsight`` other
    than ``"none"`` is kept in the action's ``meta["stop_hindsight"]``.
    """

    def stop(name: str) -> ToolRun:
        def run(arguments: dict[str, Any], observation: Observation) -> ToolResult:
            detail = str(arguments.get("summary") or arguments.get("reason") or "")
            meta: dict[str, Any] = {}
            hindsight = arguments.get("hindsight")
            # The schema's "say 'none'" escape hatch must not pollute harvested
            # metadata with literal sentinels.
            if (
                isinstance(hindsight, str)
                and hindsight.strip()
                and hindsight.strip().lower() != "none"
            ):
                meta["stop_hindsight"] = hindsight.strip()
            action = Action(data=control.hold_target(observation), meta=meta)
            return ToolResult(
                chunk=ActionChunk(
                    actions=[action],
                    control_hz=control.control_hz,
                    terminal=Termination(name, detail),
                ),
                note=f"{name}: {detail}",
            )

        return run

    done = {
        "type": "function",
        "function": {
            "name": "done",
            "description": ("Declare the task finished. The trial ends; a scorer judges success."),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "hindsight": {
                        "type": "string",
                        "description": _HINDSIGHT_DESCRIPTION,
                    },
                },
                "required": ["summary", "hindsight"],
            },
        },
    }
    give_up = {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": (
                "Stop trying; the task cannot be completed. The trial ends. "
                "If a workspace limit is what stops you, say so in your reason: "
                "operators can widen limits between trials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "hindsight": {
                        "type": "string",
                        "description": _HINDSIGHT_DESCRIPTION,
                    },
                },
                "required": ["reason", "hindsight"],
            },
        },
    }
    return ToolLibrary(
        [
            Tool("done", done, stop("done"), TERMINATION, order=10),
            Tool("give_up", give_up, stop("give_up"), TERMINATION, order=11),
        ]
    )
