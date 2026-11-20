"""IR+: the controller a coding agent writes for itself, offered as the agent's tools.

Driving YAM arms directly through Codex on Robocurve's rigs, GPT-6.2 Astra wrote
a small controller of its own in almost every trial (159 trials, 503 Python files
on 2026-11-19), a little differently each time. The same functions recur, with
settings that cluster tightly:

- ``go`` / ``move``: move to a target over a duration, easing in and out
  (smoothstep in 131 of 157 definitions), never faster than a speed limit,
  default 1.3 s;
- ``grip``: set the gripper opening and wait about 0.5 s;
- ``park``: return to the pose the arm started in (about 2.5 s);
- ``state`` / ``status``: print the joints, gripper and end-effector pose;
- ``snap`` / ``capture``: grab the cameras' current images;
- waits between steps.

What the agent did with them mattered as much as the functions: one turn ran a
whole sequence (approach, descend, grip, lift, look) and looked at the cameras
wherever in that sequence it chose. ``toolset="plus"`` offers those functions as
tools on any absolute action space, lets one turn hold several calls, run in
order (see ``LLMAgentPolicy``), and offers ``sequence``, one call holding several
steps, which is how the agents' scripts batched them.

The motion tools are built on the core move tool (`inspect_robots.tools`), so its
validation, bounds, per-dimension speed limits, pinned dimensions, held setpoint
and pre-check apply unchanged; only the timing differs. The waypoints follow the
same straight line in the action space, eased with smoothstep over
``max(seconds, the speed-limited duration x 1.5)`` (smoothstep's peak speed is
1.5 times its average, so the eased move never exceeds the limit the linear one
keeps). Where the action space is Cartesian, the embodiment's own IK follows that
path, as it does for ``move_to``; the agents' controllers instead solved IK once
at the goal and interpolated the joints, which needs a kinematic model of the arm
and is not done here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.policy_adapter import state_lines
from inspect_robots.tools import CONTROL, ControlTools, Tool, ToolLibrary, ToolResult
from inspect_robots.types import Action, ActionChunk, Observation

#: A move's duration when the call gives none (the agents' controllers' median).
DEFAULT_SECONDS = 1.3
#: How long ``grip`` takes to change the opening and let it settle.
GRIP_SECONDS = 0.5
#: How long ``park`` takes to return to the trial's starting pose.
PARK_SECONDS = 2.5
#: The longest one move may take, as for the core move tool.
MAX_SECONDS = 10.0
#: Smoothstep's peak speed over its average: eased moves stretch the linear step count by this.
EASE_PEAK = 1.5
#: The longest ``wait`` may hold, as for the core ``sleep``.
WAIT_MAX_SECONDS = 180.0
#: The most steps one ``sequence`` call may hold.
SEQUENCE_MAX_STEPS = 20
#: Tool names, in listing order (the motion tool is ``go`` on Cartesian spaces).
TOOL_NAMES = ("go", "move_joints", "grip", "park", "state", "look", "wait", "sequence")

_NOTE = {
    "type": "string",
    "description": (
        "Optional: what you observe and why you chose this call, in a sentence. "
        "The user reads these notes live and in the saved transcript."
    ),
}
_FALLBACK_NOTE = "(no note given)"


def smoothstep(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Ease in and out: 0 at 0, 1 at 1, zero slope at both ends."""
    return u * u * (3.0 - 2.0 * u)


class PlusTools:
    """IR+'s control tools for one bound embodiment, sharing ``control``'s held setpoint.

    ``labels``, ``low`` and ``high`` describe the action space; ``pose`` says it
    is Cartesian (the motion tool is then ``go``, else ``move_joints``). Raises
    ``ValueError`` for a displacement action space, which has no target to go to.
    """

    def __init__(
        self,
        control: ControlTools,
        labels: tuple[str, ...],
        low: npt.NDArray[np.float64],
        high: npt.NDArray[np.float64],
        *,
        pose: bool,
    ) -> None:
        base = [name for name in control.library.names if name in ("move_to", "move_joints")]
        if not base:
            raise ValueError(
                "toolset=plus needs an absolute action space (joint or end-effector "
                "targets); this embodiment takes displacements"
            )
        self._control = control
        self._base_move = base[0]
        self._labels = labels
        self._low = np.asarray(low, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)
        self._move_name = "go" if pose else "move_joints"
        self._grippers = tuple(label for label in labels if "gripper" in label.lower())
        prefixes = tuple(dict.fromkeys(label.split("_", 1)[0] for label in labels))
        self._arms = prefixes if len(prefixes) > 1 and all("_" in label for label in labels) else ()
        self._start: npt.NDArray[np.float64] | None = None

    # -- trial state -------------------------------------------------------------------

    def start_trial(self) -> None:
        """Forget the previous trial's starting pose; the next observation sets it."""
        self._start = None

    def observe(self, observation: Observation) -> None:
        """Remember the trial's starting pose (the setpoint before any move) for ``park``."""
        if self._start is None:
            self._start = self._control.hold_target(observation).copy()

    # -- the library ------------------------------------------------------------------

    @property
    def library(self) -> ToolLibrary:
        """The motion tool, ``grip`` (given a gripper), ``park``, ``state``, ``look``, ``wait``."""
        tools = [Tool(self._move_name, self._move_schema(), self._go, CONTROL, order=0)]
        if self._grippers:
            tools.append(Tool("grip", self._grip_schema(), self._grip, CONTROL, order=1))
        tools += [
            Tool("park", self._park_schema(), self._park, CONTROL, order=2),
            Tool("state", self._state_schema(), self._state, CONTROL, order=3),
            Tool("look", self._look_schema(), self._look, CONTROL, order=4),
            Tool("wait", self._wait_schema(), self._wait, CONTROL, order=5),
            Tool("sequence", self._sequence_schema(), self._sequence, CONTROL, order=6),
        ]
        return ToolLibrary(tools)

    @property
    def step_tools(self) -> tuple[str, ...]:
        """The tools a ``sequence`` step may name, in listing order."""
        names = [self._move_name, "grip", "park", "state", "look", "wait"]
        return tuple(name for name in names if name != "grip" or self._grippers)

    def sequence_steps(self, arguments: Mapping[str, Any]) -> list[tuple[str, str]] | str:
        """A ``sequence`` call's steps as ``(tool, JSON arguments)``, or an error to report.

        A step is ``{"tool": name, "args": {...}}``; a step without ``args``
        takes its other keys as the arguments (``{"tool": "grip", "opening": 0}``).
        The steps' own arguments are checked when each runs, as separate calls are.
        """
        steps = arguments.get("steps")
        if not isinstance(steps, list) or not steps:
            return "steps must be a non-empty list of {tool, args} objects"
        if len(steps) > SEQUENCE_MAX_STEPS:
            return f"a sequence holds at most {SEQUENCE_MAX_STEPS} steps; got {len(steps)}"
        allowed = self.step_tools
        out: list[tuple[str, str]] = []
        for number, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                return f"step {number} must be an object with `tool` and `args`"
            tool = step.get("tool")
            if tool not in allowed:
                return f"step {number}: tool must be one of: {', '.join(allowed)}; got {tool!r}"
            args = (
                step["args"] if "args" in step else {k: v for k, v in step.items() if k != "tool"}
            )
            if not isinstance(args, dict):
                return f"step {number}: args must be an object"
            out.append((tool, json.dumps(args)))
        return out

    # -- runs ---------------------------------------------------------------------------

    def _go(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        targets = arguments.get("targets")
        if not isinstance(targets, dict) or not targets:
            return ToolResult(error="targets must be a non-empty object of name: value")
        return self._move(targets, arguments.get("seconds"), arguments.get("note"), observation)

    def _grip(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        gripper = arguments.get("gripper", self._grippers[0] if len(self._grippers) == 1 else None)
        if gripper not in self._grippers:
            return ToolResult(error=f"gripper must be one of: {', '.join(self._grippers)}")
        opening = arguments.get("opening")
        return self._move(
            {gripper: opening}, GRIP_SECONDS, arguments.get("note"), observation, name="grip"
        )

    def _park(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        self.observe(observation)
        assert self._start is not None
        arm = arguments.get("arm")
        if arm is not None and arm not in self._arms:
            return ToolResult(error=f"arm must be one of: {', '.join(self._arms)}")
        pinned = set(self._control.pinned_labels)
        targets = {
            label: float(self._start[index])
            for index, label in enumerate(self._labels)
            if label not in pinned and (arm is None or label.startswith(f"{arm}_"))
        }
        return self._move(targets, PARK_SECONDS, arguments.get("note"), observation, name="park")

    def _state(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        lines = state_lines(observation, self._control.state_labels())
        lines += self._control.setpoint_lines(observation)
        return ToolResult(note="\n".join(lines) or "no state is available in this observation")

    def _look(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        requested = arguments.get("cameras")
        if requested is None:
            return ToolResult(capture=tuple(observation.images))
        if (
            not isinstance(requested, list)
            or not requested
            or any(not isinstance(name, str) for name in requested)
        ):
            return ToolResult(error="cameras must be a non-empty list of camera names when given")
        unknown = [name for name in requested if name not in observation.images]
        if unknown:
            present = ", ".join(repr(name) for name in observation.images) or "none"
            return ToolResult(error=f"unknown camera {unknown[0]!r}; available now: {present}")
        return ToolResult(capture=tuple(requested))

    def _wait(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        sleep = {"seconds": arguments.get("seconds"), "note": _note(arguments.get("note"))}
        result = self._control.library.execute("sleep", json.dumps(sleep), observation)
        if result.chunk is None:
            return result
        assert result.sleep is not None
        return ToolResult(
            chunk=result.chunk,
            note=f"waiting {result.sleep:g} s ({len(result.chunk)} steps) holding the setpoint",
            sleep=result.sleep,
        )

    def _sequence(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        # The agent policy expands a sequence into its steps before anything runs.
        steps = self.sequence_steps(arguments)
        error = steps if isinstance(steps, str) else "sequence runs only in the agent policy"
        return ToolResult(error=error)

    def _move(
        self,
        targets: Mapping[str, object],
        seconds: object,
        note: object,
        observation: Observation,
        *,
        name: str | None = None,
    ) -> ToolResult:
        """Validate with the core move tool, then re-time its straight line with smoothstep."""
        hz = self._control.resolved_hz
        cap = math.ceil(MAX_SECONDS * hz)
        if seconds is None:
            duration = DEFAULT_SECONDS
        elif (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            return ToolResult(error="seconds must be a positive number of seconds when given")
        elif seconds > MAX_SECONDS:
            return ToolResult(
                error=f"seconds {seconds:g} is over the {MAX_SECONDS:g} s limit for one move"
            )
        else:
            duration = float(seconds)
        held = self._control.hold_target(observation)
        start = held.copy()
        labels = self._control.state_labels()
        if labels is not None:
            readback = np.asarray(observation.state[labels[0]], dtype=np.float64)
            for label in targets:
                if label in self._labels and readback.shape == start.shape:
                    index = self._labels.index(label)
                    start[index] = readback[index]  # named dimensions move from where they are
        base = self._control.library.execute(
            self._base_move,
            json.dumps({"targets": dict(targets), "note": _note(note)}),
            observation,
        )
        if base.error is not None or base.chunk is None or base.target is None:
            return ToolResult(error=_renamed(base.error, self._base_move, self._move_name))
        linear_steps = len(base.chunk)
        steps = max(math.ceil(EASE_PEAK * linear_steps), round(duration * hz), 1)
        if steps > cap:
            # An eased move this long would pass the cap; play the speed-limited straight line.
            chunk, eased = base.chunk, False
        else:
            target = base.target
            fractions = smoothstep(np.arange(1, steps + 1, dtype=np.float64) / steps)
            actions = [
                Action(data=np.clip(start + (target - start) * f, self._low, self._high))
                for f in fractions
            ]
            actions[-1] = Action(data=target.copy(), meta={"chunk_final": True})
            chunk = ActionChunk(actions=actions, control_hz=self._control.control_hz)
            self._control.track(chunk, observation)
            eased = True
        played = len(chunk)
        note_text = f"executing {name or self._move_name} over {played} steps ({played / hz:.1f}s)"
        if not eased:
            note_text += "; linear, since easing it would pass the 10 s limit"
        return ToolResult(chunk=chunk, note=note_text, target=base.target)

    # -- schemas ------------------------------------------------------------------------

    def _function(
        self, name: str, description: str, properties: dict[str, Any], required: list[str]
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {**properties, "note": _NOTE},
                    "required": required,
                },
            },
        }

    def _seconds(self, default: float) -> dict[str, Any]:
        return {
            "type": "number",
            "description": (
                f"How long the move takes (default {default:g} s, at most {MAX_SECONDS:g} s). "
                "A move is never faster than the embodiment's speed limit allows."
            ),
        }

    def _move_schema(self) -> dict[str, Any]:
        kind = (
            "end-effector targets (meters for positions, radians for rotations, per the "
            "dimension labels)"
            if self._move_name == "go"
            else "joint targets (per the dimension labels)"
        )
        return self._function(
            self._move_name,
            f"Move to absolute {kind}, easing in and out over `seconds`. Dimensions you "
            "leave out keep their last commanded target. Returns when the move is "
            "issued; with other calls in the same turn, the next one starts after it "
            "has played. Embodiment bounds: " + self._control.bounds_text,
            {
                "targets": {
                    "type": "object",
                    "description": "Target values by dimension name: " + ", ".join(self._labels),
                    "additionalProperties": {"type": "number"},
                },
                "seconds": self._seconds(DEFAULT_SECONDS),
            },
            ["targets"],
        )

    def _grip_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "opening": {
                "type": "number",
                "description": "The gripper target, in its dimension's units.",
            }
        }
        required = ["opening"]
        if len(self._grippers) > 1:
            properties["gripper"] = {"type": "string", "enum": list(self._grippers)}
            required.insert(0, "gripper")
        return self._function(
            "grip",
            f"Set a gripper's opening over {GRIP_SECONDS:g} s; everything else holds.",
            properties,
            required,
        )

    def _park_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        description = f"Return to the pose this trial started in, over {PARK_SECONDS:g} s"
        if self._arms:
            properties["arm"] = {
                "type": "string",
                "enum": list(self._arms),
                "description": "Only this arm's dimensions; every dimension if omitted.",
            }
            description += " (one arm, or every dimension)"
        return self._function("park", description + ".", properties, [])

    def _state_schema(self) -> dict[str, Any]:
        return self._function(
            "state",
            "Read the state now (proprioception and the held setpoint), between moves in "
            "the same turn.",
            {},
            [],
        )

    def _look_schema(self) -> dict[str, Any]:
        return self._function(
            "look",
            "See the cameras now. Their images follow this turn's results, labelled with "
            "where in the turn they were taken. Every camera if `cameras` is omitted.",
            {"cameras": {"type": "array", "items": {"type": "string"}}},
            [],
        )

    def _sequence_schema(self) -> dict[str, Any]:
        allowed = list(self.step_tools)
        return self._function(
            "sequence",
            "Run several steps in one call, in order, exactly as the same calls made one "
            "after another would run: each motion plays out before the next step starts, "
            "`state` and `look` report what is true at their place, and a failing step "
            "skips the rest. Use it for a whole routine (approach, descend, grip, lift, "
            "look) instead of one step per turn. Returns one result listing each step's. "
            f"At most {SEQUENCE_MAX_STEPS} steps.",
            {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "enum": allowed},
                            "args": {
                                "type": "object",
                                "description": "The arguments that tool takes on its own.",
                            },
                        },
                        "required": ["tool"],
                    },
                }
            },
            ["steps"],
        )

    def _wait_schema(self) -> dict[str, Any]:
        return self._function(
            "wait",
            f"Hold every setpoint for `seconds` (at most {WAIT_MAX_SECONDS:g}).",
            {"seconds": {"type": "number"}},
            ["seconds"],
        )


def _note(note: object) -> str:
    return note if isinstance(note, str) and note.strip() else _FALLBACK_NOTE


def _renamed(error: str | None, base: str, name: str) -> str:
    text = error or "the move could not be built"
    return text.replace(base, name) if base != name else text
