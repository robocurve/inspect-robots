"""World model shared by the perceiver, phase machine, serializer, and scorer.

Positions are metres in an arm's base frame (YAM ``eef_pos`` convention:
+x forward, +y left, +z up). The wording helpers turn geometry into the
directional phrases the spike showed Jev reads correctly; do not change the
words without rerunning ``examples/jev_textsim.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Vec3 = tuple[float, float, float]
Arm = Literal["left", "right"]
ARMS: tuple[Arm, Arm] = ("left", "right")
AXES: tuple[str, str, str] = ("x", "y", "z")
#: Move words per (axis, sign).
DIRECTION: dict[tuple[str, int], str] = {
    ("x", 1): "FORWARD",
    ("x", -1): "BACK",
    ("y", 1): "LEFT",
    ("y", -1): "RIGHT",
    ("z", 1): "UP",
    ("z", -1): "DOWN",
}
#: Where-the-target-is words per (axis, sign); z differs from the move words.
_WHERE: dict[tuple[str, int], str] = {**DIRECTION, ("z", 1): "ABOVE", ("z", -1): "BELOW"}
#: Rounding step for every distance Jev reads: 0.5 cm hides tag jitter.
ROUND_M = 0.005


def round_half_cm(metres: float) -> float:
    """Round to the nearest 0.5 cm so tag jitter never changes the wording."""
    return round(metres / ROUND_M) * ROUND_M


def _cm_text(metres: float) -> str:
    return f"{abs(metres) * 100.0:g} cm"


def describe_offset(target: Vec3, origin: Vec3) -> list[str]:
    """Describe where ``target`` is relative to ``origin``, one phrase per axis."""
    words: list[str] = []
    for axis, t, o in zip(AXES, target, origin, strict=True):
        d = round_half_cm(t - o)
        if d == 0.0:
            words.append(f"aligned on {axis}")
        else:
            words.append(f"{_cm_text(d)} {_WHERE[(axis, 1 if d > 0 else -1)]}")
    return words


def distance(a: Vec3, b: Vec3) -> float:
    """Euclidean distance in metres."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


@dataclass(frozen=True)
class ObjectView:
    """One tagged object's centre in every arm frame and the decision step it was last seen.

    Steps are policy decisions (one per ``act`` call), not control ticks.
    """

    name: str
    in_frame: Mapping[Arm, Vec3]
    last_seen_step: int
    #: True when the pose is the gripper's pose plus a grasp offset (object held
    #: and its tags hidden), not a tag detection. ``last_seen_step`` is still
    #: refreshed so stale checks stay quiet while carrying.
    from_gripper: bool = False


@dataclass(frozen=True)
class GripperView:
    """One gripper's grasp point in its own base frame and its opening (0 closed, 1 open)."""

    position: Vec3
    opening: float


@dataclass(frozen=True)
class WorldState:
    """Everything the phase machine and serializer may look at for one step."""

    step: int
    objects: Mapping[str, ObjectView]
    grippers: Mapping[Arm, GripperView]

    def offset(self, arm: Arm, obj: str) -> Vec3:
        """Return ``object - gripper`` in ``arm``'s frame."""
        target = self.objects[obj].in_frame[arm]
        origin = self.grippers[arm].position
        return (target[0] - origin[0], target[1] - origin[1], target[2] - origin[2])

    def steps_since_seen(self, obj: str) -> int:
        """Steps elapsed since the object's tag was last detected."""
        return self.step - self.objects[obj].last_seen_step
