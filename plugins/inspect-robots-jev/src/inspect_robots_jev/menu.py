"""Build the Choice options Jev picks from, and parse its pick back into a move.

Every option's description carries a when-to-use hint; the spike showed that
without hints Jev picks the largest step regardless of distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from inspect_robots_jev.world import AXES, DIRECTION, Arm

MenuKind = Literal["xyz", "z", "grip_close", "grip_open", "done"]
#: The option id that means "do nothing this turn"; present in every menu.
HOLD = "hold"
_SIGN = {"plus": 1, "minus": -1}


@dataclass(frozen=True)
class Move:
    """A parsed option: one axis displacement, a gripper target, or hold."""

    arm: Arm
    axis: str | None
    delta_m: float
    gripper: float | None

    @property
    def is_hold(self) -> bool:
        """True when the option asks for no motion at all."""
        return self.axis is None and self.gripper is None


def _hints(target: str, steps_cm: Sequence[float]) -> dict[float, str]:
    steps = sorted(steps_cm)
    hints: dict[float, str] = {}
    for i, s in enumerate(steps):
        if len(steps) == 1:
            hints[s] = f"the only step size; use it whenever the {target} is not yet reached"
        elif i == 0:
            hints[s] = f"use when the {target} is less than {steps[1]:g} cm away in that direction"
        elif i == len(steps) - 1:
            hints[s] = f"use only when the {target} is more than {s:g} cm away in that direction"
        else:
            hints[s] = (
                f"use when the {target} is {s:g} to {steps[i + 1]:g} cm away in that direction"
            )
    return hints


def build_menu(
    arm: Arm, kind: MenuKind, target_name: str, steps_cm: Sequence[float]
) -> dict[str, str]:
    """Return ``{option_id: description}`` for one phase of the task."""
    menu: dict[str, str] = {}
    if kind in ("xyz", "z"):
        hints = _hints(target_name, steps_cm)
        axes: tuple[str, ...] = AXES if kind == "xyz" else ("z",)
        for axis in axes:
            for s in sorted(steps_cm):
                for word, sign in _SIGN.items():
                    menu[f"{arm}_{axis}_{word}_{s:g}cm"] = (
                        f"move the {arm} gripper {DIRECTION[(axis, sign)]} by {s:g} cm; {hints[s]}"
                    )
        menu[HOLD] = (
            f"do not move this turn (only if the {arm} gripper is already at the {target_name})"
        )
    elif kind == "grip_close":
        menu[f"{arm}_close"] = f"close the {arm} gripper on the {target_name}"
        menu[HOLD] = "do not close yet"
    elif kind == "grip_open":
        menu[f"{arm}_open"] = f"open the {arm} gripper to release into the {target_name}"
        menu[HOLD] = "do not open yet"
    else:
        menu[HOLD] = "the task is complete; do nothing"
    return menu


def parse_option(option_id: str) -> Move:
    """Invert ``build_menu``'s id grammar; raise ``ValueError`` on anything else."""
    if option_id == HOLD:
        return Move("left", None, 0.0, None)
    parts = option_id.split("_")
    if len(parts) == 2 and parts[0] in ("left", "right") and parts[1] in ("close", "open"):
        return Move(cast(Arm, parts[0]), None, 0.0, 0.0 if parts[1] == "close" else 1.0)
    if (
        len(parts) == 4
        and parts[0] in ("left", "right")
        and parts[1] in AXES
        and parts[2] in _SIGN
        and parts[3].endswith("cm")
    ):
        try:
            cm = float(parts[3][:-2])
        except ValueError as err:
            raise ValueError(f"unrecognised option id {option_id!r}") from err
        return Move(cast(Arm, parts[0]), parts[1], _SIGN[parts[2]] * cm / 100.0, None)
    raise ValueError(f"unrecognised option id {option_id!r}")
