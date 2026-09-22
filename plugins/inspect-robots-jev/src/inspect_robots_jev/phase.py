"""Deterministic phase machine for the cube-into-bowl task.

Code owns task progress (rule R4): Jev picks small moves inside a phase, the
machine decides from tag poses and gripper opening when a phase is finished.
Every rule is a tolerance test on numbers the perceiver already produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from inspect_robots_jev.menu import MenuKind
from inspect_robots_jev.world import ARMS, Arm, Vec3, WorldState

PhaseName = Literal[
    "approach",
    "descend",
    "grasp",
    "reopen",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
    "verify",
    "done",
]
#: Which arm/target/menu each phase uses. Target is the object the menu talks about.
_TARGET: dict[PhaseName, str] = {
    "approach": "cube",
    "descend": "cube",
    "grasp": "cube",
    "reopen": "cube",
    "lift": "cube",
    "carry": "bowl",
    "lower": "bowl",
    "release": "bowl",
    "retreat": "bowl",
    "verify": "cube",
    "done": "bowl",
}
_MENU: dict[PhaseName, MenuKind] = {
    "approach": "xyz",
    "descend": "z",
    "grasp": "grip_close",
    "reopen": "grip_open",
    "lift": "z",
    "carry": "xyz",
    "lower": "z",
    "release": "grip_open",
    "retreat": "z",
    "verify": "z",
    "done": "done",
}


@dataclass(frozen=True)
class TaskConfig:
    """Object names and tolerances for cube-into-bowl, metres unless noted."""

    cube: str = "cube"
    bowl: str = "bowl"
    hover_m: float = 0.05
    xy_tol_m: float = 0.01
    z_tol_m: float = 0.01
    lift_m: float = 0.08
    bowl_depth_m: float = 0.04
    closed_on_object: float = 0.35
    closed_on_air: float = 0.02
    open_threshold: float = 0.8
    cube_half_m: float = 0.0127
    #: Decisions to wait in ``verify`` for the released cube's tag to be re-detected
    #: before giving up and ending the trial (the scorer then fails it honestly).
    verify_decisions: int = 8
    #: ``descend``/``lower`` count as arrived when the gripper's measured height has
    #: not dropped by more than ``z_tol_m`` over this many consecutive decisions:
    #: the fingers (or the carried cube) are resting on something.
    contact_decisions: int = 3


@dataclass(frozen=True)
class Phase:
    """The current stage: who moves, toward what, with which menu, to which point."""

    name: PhaseName
    arm: Arm
    target: str
    menu_kind: MenuKind
    goal: Vec3 | None

    @property
    def positional(self) -> bool:
        """True when Jev is steering toward ``goal`` rather than operating the gripper."""
        return self.goal is not None


def _add(p: Vec3, dz: float) -> Vec3:
    return (p[0], p[1], p[2] + dz)


def _clamp(p: Vec3, bounds: tuple[Vec3, Vec3] | None) -> Vec3:
    if bounds is None:
        return p
    low, high = bounds
    return (
        min(max(p[0], low[0]), high[0]),
        min(max(p[1], low[1]), high[1]),
        min(max(p[2], low[2]), high[2]),
    )


class PhaseMachine:
    """Track the cube-into-bowl stage and advance it on tolerance rules.

    ``bounds`` is ``(low_xyz, high_xyz)`` of the arm's workspace; every
    positional goal is clamped into it before the tolerance test, so a goal
    below the configured z floor (YAM default 0.03 m) cannot deadlock a phase.
    """

    def __init__(self, config: TaskConfig, *, bounds: tuple[Vec3, Vec3] | None = None) -> None:
        self._cfg = config
        self._bounds = bounds
        self._name: PhaseName = "approach"
        self._arm: Arm = "left"
        self._grasp_point: Vec3 | None = None
        self._phase: Phase | None = None
        self._z_trace: list[tuple[float, float]] = []  # (measured z, commanded descent)
        self._verify_count = 0

    @property
    def phase(self) -> Phase:
        """The phase computed by the last ``reset``/``advance`` call."""
        if self._phase is None:
            raise RuntimeError("PhaseMachine.reset() has not been called")
        return self._phase

    def reset(self, world: WorldState) -> Phase:
        """Start over, choosing the arm whose base (frame origin) is nearer the cube."""
        cube = world.objects[self._cfg.cube]
        self._arm = min(ARMS, key=lambda arm: math.hypot(*cube.in_frame[arm]))
        self._name = "approach"
        self._grasp_point = None
        self._z_trace = []
        self._verify_count = 0
        self._phase = self._build(world)
        return self._phase

    def advance(self, world: WorldState, *, descended_m: float = 1.0) -> Phase:
        """Apply at most one transition for this world snapshot and return the phase.

        ``descended_m`` is how far the previous decision commanded the gripper
        DOWN (0 for holds, gripper moves, sideways or upward picks). The
        contact rule counts only those decisions and fires only when the
        commanded descent over its window exceeded ``z_tol_m`` while the
        measured height barely changed, so tiny picks that the arm under-
        travels never masquerade as "resting on something".
        """
        cfg = self._cfg
        arm = self._arm
        grip = world.grippers[arm]
        current = self._build(world)
        goal = current.goal
        before = self._name
        blocked = self._track_contact(grip.position[2], descended_m)
        if self._name == "approach" and goal is not None and self._near(grip.position, goal):
            self._name = "descend"
        elif (
            self._name == "descend"
            and goal is not None
            and (self._near_z(grip.position, goal) or blocked)
        ):
            self._name = "grasp"
        elif self._name == "grasp":
            if grip.opening <= cfg.closed_on_air:
                self._name = "reopen"  # closed on nothing: open again before retrying
            elif grip.opening <= cfg.closed_on_object:
                self._grasp_point = grip.position
                self._name = "lift"
        elif self._name == "reopen" and grip.opening >= cfg.open_threshold:
            self._name = "approach"
        elif (
            self._name == "lift" and goal is not None and grip.position[2] >= goal[2] - cfg.z_tol_m
        ):
            self._name = "carry"
        elif self._name == "carry" and goal is not None and self._near_xy(grip.position, goal):
            self._name = "lower"
        elif (
            self._name == "lower"
            and goal is not None
            and (self._near_z(grip.position, goal) or blocked)
        ):
            self._name = "release"
        elif self._name == "release" and grip.opening >= cfg.open_threshold:
            self._name = "retreat"
        elif (
            self._name == "retreat"
            and goal is not None
            and grip.position[2] >= goal[2] - cfg.z_tol_m
        ):
            self._name = "verify"
        elif self._name == "verify":
            # The trial may only end once the released cube has been SEEN again
            # (not inferred from the gripper), or the wait budget is spent.
            cube = world.objects.get(cfg.cube)
            seen = (
                cube is not None and not cube.from_gripper and world.steps_since_seen(cfg.cube) == 0
            )
            self._verify_count += 1
            if seen or self._verify_count > cfg.verify_decisions:
                self._name = "done"
        if self._name != before:
            self._z_trace = []
        self._phase = self._build(world)
        return self._phase

    def _track_contact(self, z: float, descended_m: float) -> bool:
        """Record the gripper height after each commanded descent; True on contact.

        Contact means the last ``contact_decisions + 1`` recorded heights span
        no more than ``z_tol_m`` although the DOWN commands that produced them
        asked for more than ``z_tol_m`` of travel in total.
        """
        if self._name not in ("descend", "lower"):
            self._z_trace = []
            return False
        if descended_m <= 0.0:
            return False
        self._z_trace.append((z, descended_m))
        n = self._cfg.contact_decisions + 1
        if len(self._z_trace) < n:
            return False
        window = self._z_trace[-n:]
        heights = [h for h, _ in window]
        commanded = sum(d for _, d in window[1:])
        return (max(heights) - min(heights)) <= self._cfg.z_tol_m < commanded

    # -- helpers -----------------------------------------------------------

    def _target_name(self) -> str:
        return self._cfg.cube if _TARGET[self._name] == "cube" else self._cfg.bowl

    def _goal(self, world: WorldState) -> Vec3 | None:
        cfg = self._cfg
        arm = self._arm
        name = self._name
        if _MENU[name] in ("grip_close", "grip_open", "done"):
            return None
        # verify talks about the cube but rises above the bowl, where the arm is
        anchor = cfg.bowl if name == "verify" else self._target_name()
        target = world.objects[anchor].in_frame[arm]
        if name == "approach":
            goal = _add(target, cfg.hover_m)
        elif name == "descend":
            goal = target
        elif name == "lift":
            base = self._grasp_point if self._grasp_point is not None else target
            goal = _add(base, cfg.lift_m)
        elif name == "carry":
            goal = _add(target, cfg.hover_m + cfg.cube_half_m)
        elif name == "lower":
            goal = _add(target, cfg.cube_half_m - cfg.bowl_depth_m)
        elif name == "retreat":
            goal = _add(target, cfg.lift_m)
        else:  # verify: keep rising so the gripper clears the camera's view of the cube
            goal = _add(target, cfg.lift_m + cfg.hover_m)
        return _clamp(goal, self._bounds)

    def _build(self, world: WorldState) -> Phase:
        return Phase(
            name=self._name,
            arm=self._arm,
            target=self._target_name(),
            menu_kind=_MENU[self._name],
            goal=self._goal(world),
        )

    def _near_xy(self, p: Vec3, goal: Vec3) -> bool:
        tol = self._cfg.xy_tol_m
        return abs(p[0] - goal[0]) <= tol and abs(p[1] - goal[1]) <= tol

    def _near_z(self, p: Vec3, goal: Vec3) -> bool:
        return abs(p[2] - goal[2]) <= self._cfg.z_tol_m

    def _near(self, p: Vec3, goal: Vec3) -> bool:
        return self._near_xy(p, goal) and self._near_z(p, goal)
