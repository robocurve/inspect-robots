"""IR+: the controller a coding agent writes for itself, offered as the agent's tools.

Driving YAM arms directly through Codex on Robocurve's rigs, GPT-6.2 Astra wrote
a small controller of its own in almost every trial (159 trials, 503 Python files
on 2026-11-19), a little differently each time. The same functions recur, with
settings that cluster tightly, and IR+ offers their consensus as tools:

| Tool | In the trials (trials defining it) |
|---|---|
| ``go`` | go 91, goto 7, pos 6, xyz 10; with ik/solve 150, down/orient/rot/ori 85 |
| ``move_joints`` | move 96, joint(s) 46, moveq 11 |
| ``grip`` | grip 140 |
| ``park`` | park 111, home 25, rest 8 |
| ``state`` | state 89, status 34, stat 11, report 10 |
| ``look`` | snap 79, capture 27, frame 17, grab 17, shot 7 |
| ``wait`` | time.sleep in every trial |

and ``sequence``, several of them in one call, as one script ran them.

How they move, as the controllers did (medians over the definitions):

- **IK once, at the goal:** least squares over the arm's joints on the arm's
  kinematic model, residual = [position error (m), 0.22 x rotation error
  (rotation vector, rad)], within the joint limits, seeded from the measured
  joints, then the trial's start pose, then the model's own seeds; accepted
  within 6 mm and 0.05 rad, else nothing moves and the tool says how close the
  best solution came.
- **Joints, eased:** from the measured joints straight to the solution, all
  joints together, eased with smoothstep (3u^2 - 2u^3) over
  max(seconds, the largest joint change / 0.95 rad/s), default 1.3 s, then
  0.2 s to settle. The controllers streamed this at 40-60 Hz; here it plays at
  the embodiment's control rate.
- **Orientation:** yaw, tilt and roll in degrees. All zero points the tool
  straight down, turned as the arm model's ``down`` says (else as the arm faces
  at the trial's start); yaw turns it
  about the vertical, tilt leans its tip forward (+x), roll leans it toward the
  arm's right (-y): R = Rz(yaw) Ry(-tilt) Rx(-roll) R_down. Or a full 3x3
  rotation matrix, as some controllers took.
- **Gripper:** set within the move, or by ``grip`` with 0.5 s to settle.
- **Park:** the joints and gripper the trial started with, over 2.5 s.

Each move's result is the pose it reached, read back from the measured joints
(the controllers printed it after every move).

IR+ needs a joint-position action space and the arms' kinematic model: a factory
(``kinematics``) that maps the embodiment to one ``ArmModel`` per arm, keyed by
the arms' label prefix (``left_j0`` .. ``left_gripper`` is arm ``left``).
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.tools import CONTROL, ControlTools, Tool, ToolLibrary, ToolResult
from inspect_robots.types import Action, ActionChunk, Observation

Vec = npt.NDArray[np.float64]

#: A move's duration when the call gives none (the controllers' median).
DEFAULT_SECONDS = 1.3
#: The fastest any joint moves on average over a move, rad/s.
JOINT_SPEED = 0.95
#: Held still after every move so the arm settles before the next step.
SETTLE_SECONDS = 0.2
#: How long ``grip`` takes, at least.
GRIP_SECONDS = 0.5
#: How long ``park`` takes, at least.
PARK_SECONDS = 2.5
#: The longest one move may take.
MAX_MOVE_SECONDS = 20.0
#: The longest ``wait`` may hold.
WAIT_MAX_SECONDS = 180.0
#: Smoothstep's peak speed over its average: an eased gripper change needs this many more steps.
EASE_PEAK = 1.5
#: IK: the rotation error's weight against the position error (m per rad).
ROT_WEIGHT = 0.22
#: IK: a solution counts within this position (m) and rotation (rad) error.
POS_TOL, ROT_TOL = 0.006, 0.05
#: The most steps one ``sequence`` call may hold.
SEQUENCE_MAX_STEPS = 20
#: Tool names, in listing order.
TOOL_NAMES = ("go", "move_joints", "grip", "park", "state", "look", "wait", "sequence")
#: The tools a ``sequence`` step may name.
STEP_TOOLS = TOOL_NAMES[:-1]

_NOTE = {"type": "string", "description": "optional: what you see and why you chose this"}
_GRIPPER = {
    "type": "number",
    "description": "optional: the gripper's opening during the move, 0 closed .. 1 open",
}


@runtime_checkable
class ArmModel(Protocol):
    """One arm's kinematics in its own base frame (+x forward, +y left, +z up).

    ``fk`` maps the arm's joint angles (radians, in action-label order) to the
    grasp point's position (m) and the tool's 3x3 rotation, whose third column
    points along the tool toward the grasp point. ``joint_low`` and
    ``joint_high`` are the joints' limits. A model may also offer
    ``ik_seeds(position)``, extra joint vectors to start IK from, and ``down``,
    the 3x3 tool rotation that orientation zero means (the tool pointing straight
    down, turned as the controllers' ``down()`` was); without it, zero is the
    tool pointing down turned as it faces at the trial's start.
    """

    joint_low: Vec
    joint_high: Vec

    def fk(self, q: Vec) -> tuple[Vec, Vec]:
        """Grasp-point position and tool rotation for joint angles ``q``."""
        ...


#: ``kinematics(embodiment_info) -> {arm prefix: ArmModel}``.
KinematicsFactory = Callable[[EmbodimentInfo], Mapping[str, ArmModel]]


def load_kinematics(spec: str | KinematicsFactory) -> KinematicsFactory:
    """Resolve ``-P kinematics=module:attr`` to the factory it names (a callable passes through)."""
    if callable(spec):
        return spec
    module, _, attr = str(spec).partition(":")
    if not module or not attr:
        raise ValueError(f"kinematics must be 'module:factory', got {spec!r}")
    factory = getattr(importlib.import_module(module), attr)
    if not callable(factory):
        raise ValueError(f"kinematics {spec!r} is not callable")
    return factory  # type: ignore[no-any-return]


def smoothstep(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Ease in and out: 0 at 0, 1 at 1, zero slope at both ends."""
    return u * u * (3.0 - 2.0 * u)


@dataclass
class _Arm:
    name: str
    model: ArmModel
    joints: list[int]  # action indices of the arm's joints, in the model's order
    gripper: int | None
    low: Vec  # joint limits: the model's within the action space's
    high: Vec
    down: Vec | None = None  # orientation zero: the tool pointing straight down


class PlusTools:
    """IR+'s tools for one bound embodiment, sharing ``control``'s held setpoint.

    Raises ``ValueError`` when the arms in ``models`` don't match the action
    space's labels. Each move registers a report, read back from the
    observation after it has played (``take_report``).
    """

    def __init__(
        self,
        control: ControlTools,
        labels: tuple[str, ...],
        low: Vec,
        high: Vec,
        models: Mapping[str, ArmModel],
        *,
        max_step: Sequence[float | None] | None = None,
    ) -> None:
        try:
            from scipy.optimize import least_squares  # type: ignore[import-untyped]
            from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - scipy ships with the plus extra
            raise ValueError(
                "toolset=plus needs scipy for its IK: pip install 'inspect-robots-agent[plus]'"
            ) from exc
        self._least_squares = least_squares
        self._rotation = Rotation
        self._control = control
        self._labels = labels
        self._low = np.asarray(low, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)
        self._max_step = tuple(max_step) if max_step is not None else (None,) * len(labels)
        if not models:
            raise ValueError("the kinematics factory returned no arms")
        self._arms: dict[str, _Arm] = {}
        for name, model in models.items():
            mine = [i for i, label in enumerate(labels) if label.startswith(f"{name}_")]
            grippers = [i for i in mine if "gripper" in labels[i].lower()]
            joints = [i for i in mine if i not in grippers]
            jlow = np.asarray(model.joint_low, dtype=np.float64)
            jhigh = np.asarray(model.joint_high, dtype=np.float64)
            if not joints or len(joints) != len(jlow):
                raise ValueError(
                    f"arm {name!r}: the action space has {len(joints)} joint dimensions "
                    f"labelled {name}_*, its model {len(jlow)} joints"
                )
            self._arms[name] = _Arm(
                name=name,
                model=model,
                joints=joints,
                gripper=grippers[0] if grippers else None,
                low=np.maximum(jlow, self._low[joints]),
                high=np.minimum(jhigh, self._high[joints]),
            )
        self._start: Vec | None = None
        self._report: Callable[[Observation], str] | None = None

    # -- trial state -------------------------------------------------------------------

    def start_trial(self) -> None:
        """Forget the previous trial's start pose; the next observation sets it."""
        self._start = None
        self._report = None
        for arm in self._arms.values():
            arm.down = None

    def observe(self, observation: Observation) -> None:
        """Remember the trial's start pose (for ``park`` and the orientation's zero)."""
        if self._start is not None:
            return
        self._start = self._readback(observation)
        for arm in self._arms.values():
            down = getattr(arm.model, "down", None)
            if down is None:
                _, rotation = arm.model.fk(self._start[arm.joints])
                down = _down_like(np.asarray(rotation, dtype=np.float64))
            arm.down = np.asarray(down, dtype=np.float64)

    def take_report(self) -> Callable[[Observation], str] | None:
        """The last move's report, once: call it with the observation after the move."""
        report, self._report = self._report, None
        return report

    # -- the library ------------------------------------------------------------------

    @property
    def library(self) -> ToolLibrary:
        """Every IR+ tool (``grip`` only given grippers), in ``TOOL_NAMES`` order."""
        tools = [
            Tool("go", self._go_schema(), self._go, CONTROL, order=0),
            Tool("move_joints", self._move_joints_schema(), self._move_joints, CONTROL, order=1),
        ]
        if self._grippers:
            tools.append(Tool("grip", self._grip_schema(), self._grip, CONTROL, order=2))
        tools += [
            Tool("park", self._park_schema(), self._park, CONTROL, order=3),
            Tool("state", self._state_schema(), self._state, CONTROL, order=4),
            Tool("look", self._look_schema(), self._look, CONTROL, order=5),
            Tool("wait", self._wait_schema(), self._wait, CONTROL, order=6),
            Tool("sequence", self._sequence_schema(), self._sequence, CONTROL, order=7),
        ]
        return ToolLibrary(tools)

    @property
    def step_tools(self) -> tuple[str, ...]:
        """The tools a ``sequence`` step may name, in listing order."""
        return tuple(name for name in STEP_TOOLS if name != "grip" or self._grippers)

    @property
    def _grippers(self) -> bool:
        return any(arm.gripper is not None for arm in self._arms.values())

    def sequence_steps(self, arguments: Mapping[str, Any]) -> list[tuple[str, str]] | str:
        """A ``sequence`` call's steps as ``(tool, JSON arguments)``, or an error to report.

        A step is ``{"tool": name, "args": {...}}``; a step without ``args``
        takes its other keys as the arguments (``{"tool": "park", "arm": "left"}``).
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

    def _arm(self, arguments: Mapping[str, Any]) -> _Arm | str:
        name = arguments.get("arm")
        if name not in self._arms:
            return f"arm must be one of: {', '.join(self._arms)}"
        return self._arms[name]

    def _go(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        arm = self._arm(arguments)
        if isinstance(arm, str):
            return ToolResult(error=arm)
        self.observe(observation)
        try:
            position = np.array([_number(arguments, key) for key in ("x", "y", "z")])
            rotation = self._target_rotation(arm, arguments)
        except ValueError as exc:
            return ToolResult(error=str(exc))
        readback = self._readback(observation)
        solved = self._ik(arm, position, rotation, readback[arm.joints])
        if isinstance(solved, str):
            return ToolResult(error=solved)
        result = self._joint_move(
            arm, solved, arguments.get("gripper"), arguments.get("seconds"), observation
        )
        if result.chunk is not None:
            took = self._took(result)
            self._report = lambda obs: (
                f"{arm.name} arm moved over {took:.1f}s; now {self._pose(arm, obs)} "
                f"({self._offset_mm(arm, obs, position):.0f} mm from the target)"
            )
        return result

    def _move_joints(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        arm = self._arm(arguments)
        if isinstance(arm, str):
            return ToolResult(error=arm)
        self.observe(observation)
        joints = arguments.get("joints")
        count = len(arm.joints)
        if (
            not isinstance(joints, list)
            or len(joints) != count
            or not all(_finite(value) for value in joints)
        ):
            return ToolResult(
                error=f"joints must be {count} angles (rad), in the arm's joint order"
            )
        target = np.asarray(joints, dtype=np.float64)
        outside = [i for i in range(count) if not arm.low[i] <= target[i] <= arm.high[i]]
        if outside:
            i = outside[0]
            return ToolResult(
                error=f"joint {i} target {target[i]:+.3f} is outside its limits "
                f"[{arm.low[i]:+.3f}, {arm.high[i]:+.3f}]; nothing moved"
            )
        result = self._joint_move(
            arm, target, arguments.get("gripper"), arguments.get("seconds"), observation
        )
        if result.chunk is not None:
            took = self._took(result)
            self._report = lambda obs: (
                f"{arm.name} arm moved over {took:.1f}s; now joints "
                f"{_q(self._readback(obs)[arm.joints])}; {self._pose(arm, obs)}"
            )
        return result

    def _grip(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        arm = self._arm(arguments)
        if isinstance(arm, str):
            return ToolResult(error=arm)
        if arm.gripper is None:
            return ToolResult(error=f"the {arm.name} arm has no gripper")
        opening = arguments.get("opening")
        if not _finite(opening):
            return ToolResult(error="opening must be a number, 0 closed .. 1 open")
        index = arm.gripper
        # As the controllers' grip: the last command with the gripper changed, eased like any move.
        held = self._control.hold_target(observation)
        result = self._joint_move(
            arm, held[arm.joints], opening, GRIP_SECONDS, observation, joints_from=held
        )
        if result.chunk is not None:
            target = float(result.chunk.actions[-1].data[index])
            took = self._took(result)
            self._report = lambda obs: (
                f"{arm.name} gripper commanded {target:.2f}; reads "
                f"{self._readback(obs)[index]:.3f} after {took:.1f}s"
            )
        return result

    def _park(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        self.observe(observation)
        assert self._start is not None
        arm = self._arm(arguments)
        if isinstance(arm, str):
            return ToolResult(error=arm)
        gripper = None if arm.gripper is None else float(self._start[arm.gripper])
        result = self._joint_move(arm, self._start[arm.joints], gripper, PARK_SECONDS, observation)
        if result.chunk is not None:
            took = self._took(result)
            self._report = lambda obs: (
                f"{arm.name} arm parked over {took:.1f}s at its start pose; {self._pose(arm, obs)}"
            )
        return result

    def _state(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        name = arguments.get("arm")
        if name is not None and name not in self._arms:
            return ToolResult(error=f"arm must be one of: {', '.join(self._arms)}")
        self.observe(observation)
        readback = self._readback(observation)
        effort = observation.state.get("joint_eff")
        lines = []
        for arm in self._arms.values() if name is None else [self._arms[name]]:
            line = f"{arm.name}: joints {_q(readback[arm.joints])}; {self._pose(arm, observation)}"
            if effort is not None and np.asarray(effort).shape == readback.shape:
                values = np.asarray(effort, dtype=np.float64)[arm.joints]
                line += "; joint effort [" + ", ".join(f"{v:+.2f}" for v in values) + "] N·m"
            lines.append(line)
        return ToolResult(note="\n".join(lines))

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
        seconds = arguments.get("seconds")
        sleep = {"seconds": seconds, "note": "wait"}
        result = self._control.library.execute("sleep", json.dumps(sleep), observation)
        if result.chunk is None:
            return result
        assert result.sleep is not None
        took = result.sleep
        self._report = lambda obs: f"waited {took:.1f}s"
        return ToolResult(chunk=result.chunk, note=f"waiting {took:g} s", sleep=took)

    def _sequence(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        # The agent policy expands a sequence into its steps before anything runs.
        steps = self.sequence_steps(arguments)
        error = steps if isinstance(steps, str) else "sequence runs only in the agent policy"
        return ToolResult(error=error)

    # -- motion -------------------------------------------------------------------------

    def _joint_move(
        self,
        arm: _Arm,
        target: Vec,
        gripper: object,
        seconds: object,
        observation: Observation,
        *,
        joints_from: Vec | None = None,
    ) -> ToolResult:
        """Ease the arm's joints (and gripper) to ``target``, then settle.

        The joints start where they are measured (``go``, ``move_joints``, ``park``) or
        from ``joints_from`` (``grip``: the last command, so a sagging arm is not lowered).
        """
        if seconds is None:
            duration = DEFAULT_SECONDS
        elif not _finite(seconds) or float(seconds) <= 0:  # type: ignore[arg-type]
            return ToolResult(error="seconds must be a positive number when given")
        else:
            duration = float(seconds)  # type: ignore[arg-type]
        if gripper is not None and not _finite(gripper):
            return ToolResult(error="gripper must be a number, 0 closed .. 1 open")
        hz = self._control.resolved_hz
        held = self._control.hold_target(observation).copy()
        start = held.copy()
        source = self._readback(observation) if joints_from is None else joints_from
        start[arm.joints] = source[arm.joints]
        end = start.copy()
        end[arm.joints] = target
        if arm.gripper is not None and gripper is not None:
            index = arm.gripper
            end[index] = float(np.clip(float(gripper), self._low[index], self._high[index]))  # type: ignore[arg-type]
        biggest = float(np.max(np.abs(end[arm.joints] - start[arm.joints])))
        duration = max(duration, biggest / JOINT_SPEED)
        for index, limit in enumerate(self._max_step):
            if limit:
                duration = max(duration, EASE_PEAK * abs(end[index] - start[index]) / limit / hz)
        if duration > MAX_MOVE_SECONDS:
            return ToolResult(
                error=f"that move would take {duration:.1f}s, over the {MAX_MOVE_SECONDS:g}s "
                "limit for one move; nothing moved"
            )
        steps = max(1, round(duration * hz))
        fractions = smoothstep(np.arange(1, steps + 1, dtype=np.float64) / steps)
        actions = [Action(data=start + (end - start) * f) for f in fractions]
        actions += [Action(data=end.copy()) for _ in range(round(SETTLE_SECONDS * hz))]
        chunk = self._chunk(actions, observation)
        return ToolResult(chunk=chunk, note=f"moving the {arm.name} arm over {steps / hz:.1f}s")

    def _chunk(self, actions: list[Action], observation: Observation) -> ActionChunk:
        clipped = [Action(data=np.clip(a.data, self._low, self._high)) for a in actions]
        clipped[-1] = Action(data=clipped[-1].data, meta={"chunk_final": True})
        chunk = ActionChunk(actions=clipped, control_hz=self._control.control_hz)
        self._control.track(chunk, observation)
        return chunk

    def _took(self, result: ToolResult) -> float:
        assert result.chunk is not None
        settle = round(SETTLE_SECONDS * self._control.resolved_hz)
        return (len(result.chunk) - settle) / self._control.resolved_hz

    def _ik(self, arm: _Arm, position: Vec, rotation: Vec, current: Vec) -> Vec | str:
        """Joints putting the grasp point at ``position`` with ``rotation``, or why not."""
        rot = self._rotation

        def residual(q: Vec) -> Vec:
            p, r = arm.model.fk(q)
            rotvec = rot.from_matrix(rotation.T @ np.asarray(r)).as_rotvec()
            return np.asarray(
                np.r_[np.asarray(p) - position, ROT_WEIGHT * rotvec], dtype=np.float64
            )

        seeds = [current]
        if self._start is not None:
            seeds.append(self._start[arm.joints])
        extra = getattr(arm.model, "ik_seeds", None)
        if callable(extra):
            seeds += [np.asarray(s, dtype=np.float64) for s in extra(position)]
        lo, hi = arm.low, arm.high
        best: tuple[Vec, float, float] | None = None
        for seed in seeds:
            x0 = np.clip(seed, lo + 1e-6, hi - 1e-6)
            sol = self._least_squares(
                residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=150
            )
            r = residual(sol.x)
            pos_err = float(np.linalg.norm(r[:3]))
            rot_err = float(np.linalg.norm(r[3:]) / ROT_WEIGHT)
            if best is None or pos_err + 0.1 * rot_err < best[1] + 0.1 * best[2]:
                best = (np.asarray(sol.x, dtype=np.float64), pos_err, rot_err)
            if pos_err <= POS_TOL and rot_err <= ROT_TOL:
                return np.asarray(sol.x, dtype=np.float64)
        assert best is not None
        reached, _ = arm.model.fk(best[0])
        return (
            f"unreachable; the closest solution puts the grasp point {best[1] * 1000:.0f} mm "
            f"and {math.degrees(best[2]):.0f} deg off, at {_xyz(np.asarray(reached))}; "
            "nothing moved"
        )

    # -- orientation and readback -----------------------------------------------------

    def _target_rotation(self, arm: _Arm, arguments: Mapping[str, Any]) -> Vec:
        given = arguments.get("rotation")
        if given is not None:
            try:
                matrix = np.asarray(given, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("rotation must be a 3x3 matrix (rows)") from exc
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError("rotation must be a 3x3 matrix (rows)")
            u, _, vt = np.linalg.svd(matrix)  # the nearest rotation
            fixed = u @ vt
            if np.linalg.det(fixed) < 0:
                raise ValueError("rotation must be a rotation matrix (determinant +1)")
            return np.asarray(fixed, dtype=np.float64)
        yaw, tilt, roll = (_number(arguments, key, 0.0) for key in ("yaw", "tilt", "roll"))
        return self._orientation(arm, yaw, tilt, roll)

    def _orientation(self, arm: _Arm, yaw: float, tilt: float, roll: float) -> Vec:
        assert arm.down is not None, "observe() sets the orientation's zero"
        turn = self._rotation.from_euler("ZYX", [yaw, -tilt, -roll], degrees=True).as_matrix()
        return np.asarray(turn @ arm.down, dtype=np.float64)

    def _angles(self, arm: _Arm, rotation: Vec) -> tuple[float, float, float]:
        assert arm.down is not None
        yaw, tilt, roll = self._rotation.from_matrix(rotation @ arm.down.T).as_euler(
            "ZYX", degrees=True
        )
        return float(yaw), float(-tilt), float(-roll)

    def _readback(self, observation: Observation) -> Vec:
        """The measured action-space vector (joints and grippers), else the held setpoint."""
        labels = self._control.state_labels()
        held = self._control.hold_target(observation)
        if labels is not None and labels[0] in observation.state:
            value = np.asarray(observation.state[labels[0]], dtype=np.float64)
            if value.shape == held.shape:
                return value
        return np.asarray(held, dtype=np.float64).copy()

    def _pose(self, arm: _Arm, observation: Observation) -> str:
        readback = self._readback(observation)
        position, rotation = arm.model.fk(readback[arm.joints])
        angles = self._angles(arm, np.asarray(rotation, dtype=np.float64))
        yaw, tilt, roll = (round(value, 1) + 0.0 for value in angles)  # no "-0.0"
        text = (
            f"grasp point {_xyz(np.asarray(position))} m, yaw {yaw:+.1f}, tilt {tilt:+.1f}, "
            f"roll {roll:+.1f} deg"
        )
        if arm.gripper is not None:
            text += f"; gripper {readback[arm.gripper]:.3f}"
        return text

    def _offset_mm(self, arm: _Arm, observation: Observation, target: Vec) -> float:
        position, _ = arm.model.fk(self._readback(observation)[arm.joints])
        return float(np.linalg.norm(np.asarray(position) - target)) * 1000.0

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

    @property
    def _arm_property(self) -> dict[str, Any]:
        return {"type": "string", "enum": list(self._arms), "description": "which arm"}

    @property
    def _seconds(self) -> dict[str, Any]:
        return {
            "type": "number",
            "description": (
                f"how long the move takes (default {DEFAULT_SECONDS:g}s; never faster than "
                f"{JOINT_SPEED:g} rad/s for the joint that moves most)"
            ),
        }

    def _go_schema(self) -> dict[str, Any]:
        number = {"type": "number"}
        row = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
        return self._function(
            "go",
            "Move one arm so its grasp point (between the fingertips) is at x, y, z (metres, "
            "in that arm's own base frame: +x forward, +y left, +z up) with the given tool "
            "orientation. Orientation in degrees: all zero points the tool straight down, "
            "turned as the arm faces at its home pose; yaw turns it about the vertical "
            "(positive: counterclockwise seen from above), tilt leans its tip forward (+x), "
            "roll leans it to the side (positive toward the arm's right, -y). Exactly, "
            "R = Rz(yaw) Ry(-tilt) Rx(-roll) R_down; or give `rotation`, a full 3x3 matrix "
            "(rows, the tool's axes in the base frame, the third along the tool toward the "
            "grasp point) instead. Inverse kinematics on the arm's model finds the joints "
            f"(within {POS_TOL * 1000:.0f} mm and {math.degrees(ROT_TOL):.0f} deg, or nothing "
            "moves); the joints then move there together, easing in and out, and settle "
            f"{SETTLE_SECONDS:g}s. Returns the pose reached.",
            {
                "arm": self._arm_property,
                "x": number,
                "y": number,
                "z": number,
                "yaw": number,
                "tilt": number,
                "roll": number,
                "rotation": {"type": "array", "items": row, "minItems": 3, "maxItems": 3},
                "gripper": _GRIPPER,
                "seconds": self._seconds,
            },
            ["arm", "x", "y", "z"],
        )

    def _move_joints_schema(self) -> dict[str, Any]:
        count = len(next(iter(self._arms.values())).joints)
        return self._function(
            "move_joints",
            f"Move one arm's {count} joints (radians, in the arm's joint order) to these angles "
            "together, easing in and out. Returns the joints and pose reached.",
            {
                "arm": self._arm_property,
                "joints": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": count,
                    "maxItems": count,
                },
                "gripper": _GRIPPER,
                "seconds": self._seconds,
            },
            ["arm", "joints"],
        )

    def _grip_schema(self) -> dict[str, Any]:
        return self._function(
            "grip",
            f"Set one gripper's opening (0 closed .. 1 open) and wait at least {GRIP_SECONDS:g}s; "
            "returns the reading then.",
            {"arm": self._arm_property, "opening": {"type": "number"}},
            ["arm", "opening"],
        )

    def _park_schema(self) -> dict[str, Any]:
        return self._function(
            "park",
            "Move one arm back to the joints and gripper opening it started the trial with "
            f"(at least {PARK_SECONDS:g}s).",
            {"arm": self._arm_property},
            ["arm"],
        )

    def _state_schema(self) -> dict[str, Any]:
        return self._function(
            "state",
            "Read the arms now: joint angles, grasp-point position and orientation (as go "
            "takes them), gripper reading, joint effort. One arm, or every arm if none is given.",
            {"arm": self._arm_property},
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

    def _wait_schema(self) -> dict[str, Any]:
        return self._function(
            "wait",
            f"Hold still for this many seconds (up to {WAIT_MAX_SECONDS:g}).",
            {"seconds": {"type": "number"}},
            ["seconds"],
        )

    def _sequence_schema(self) -> dict[str, Any]:
        return self._function(
            "sequence",
            "Run several steps in one call, in order, exactly as the same calls made one "
            "after another would run: each move plays out before the next step starts, "
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
                            "tool": {"type": "string", "enum": list(self.step_tools)},
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


def _down_like(rotation: Vec) -> Vec:
    """The tool pointing straight down, turned about the vertical as ``rotation`` is."""
    heading = rotation[:, 0].copy()
    heading[2] = 0.0
    if np.linalg.norm(heading) < 1e-6:  # the tool's x axis is vertical: use its y axis
        heading = np.cross(rotation[:, 1], [0.0, 0.0, 1.0])
        heading[2] = 0.0
    x = heading / np.linalg.norm(heading)
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(arguments: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = arguments.get(key, default)
    if value is None:
        raise ValueError(f"{key} is required")
    if not _finite(value):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _q(values: Vec) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in values) + "]"


def _xyz(position: Vec) -> str:
    return f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})"
