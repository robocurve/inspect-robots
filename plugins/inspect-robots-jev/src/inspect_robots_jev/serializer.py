"""Turn the curated world state into the text Jev reads (rules R1 and R2).

Only the current target's geometry appears, described relative to the active
gripper in directional words; every other object is a bare name. The exact
wording is pinned by tests because the spike showed it decides success.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inspect_robots_jev.phase import Phase
from inspect_robots_jev.world import WorldState, describe_offset, distance

_TASK: dict[str, str] = {
    "approach": (
        "Move the {arm} gripper to the target point 5 cm above the {target}. "
        "Move toward that point, never away."
    ),
    "descend": "Lower the {arm} gripper straight down onto the {target}.",
    "grasp": "Close the {arm} gripper on the {target}.",
    "reopen": "The {arm} gripper closed on nothing. Open it again before retrying the {target}.",
    "lift": "Raise the {arm} gripper straight up, carrying the {target}.",
    "carry": (
        "Move the {arm} gripper to the target point above the {target}, carrying the cube. "
        "Move toward that point, never away."
    ),
    "lower": "Lower the {arm} gripper into the {target}.",
    "release": "Open the {arm} gripper to drop the cube into the {target}.",
    "retreat": "Raise the {arm} gripper straight up, away from the {target}.",
    "verify": ("Raise the {arm} gripper straight up until the camera can see the {target} again."),
    "done": "The task is complete.",
}
_STEER = (
    "Which move takes the {arm} gripper toward the target point? Move in the direction "
    "the target is in, along the axis with the largest remaining gap, with a step no "
    "larger than that gap."
)


def gripper_word(
    opening: float, *, open_threshold: float = 0.8, closed_threshold: float = 0.35
) -> str:
    """Name the gripper state Jev reads: open, closed, or partly closed.

    The thresholds default to ``TaskConfig``'s values; the policy passes its
    own so the words and the phase machine's grasp test never drift apart.
    """
    if opening >= open_threshold:
        return "open"
    if opening <= closed_threshold:
        return "closed"
    return "partly closed"


def build_state(
    world: WorldState,
    phase: Phase,
    history: Sequence[str],
    *,
    history_len: int = 5,
    open_threshold: float = 0.8,
    closed_threshold: float = 0.35,
) -> dict[str, Any]:
    """Return Jev's ``state`` object for this step, keys in reading order.

    Positional phases describe the *target point* (for example 5 cm above the
    cube), not the object itself, because that point is what Jev steers to.
    """
    arm = phase.arm
    grip = world.grippers[arm]
    state: dict[str, Any] = {"task": _TASK[phase.name].format(arm=arm, target=phase.target)}
    if phase.goal is not None:
        state[f"target_point_relative_to_{arm}_gripper"] = describe_offset(
            phase.goal, grip.position
        )
        state["straight_line_distance_cm"] = round(distance(phase.goal, grip.position) * 100, 1)
    state[f"{arm}_gripper"] = gripper_word(
        grip.opening, open_threshold=open_threshold, closed_threshold=closed_threshold
    )
    state["other_objects_on_table_ignore_them"] = sorted(
        name for name in world.objects if name != phase.target
    )
    state["recent_moves_oldest_first"] = list(history[-history_len:]) if history_len else []
    if phase.target in world.objects:
        unseen = world.steps_since_seen(phase.target)
        if unseen > 0:
            state["note"] = (
                f"the {phase.target} was last seen {unseen} decisions ago; assume it has not moved"
            )
    return state


def instructions_for(phase: Phase) -> str:
    """Return the Choice ``instructions`` text for this phase."""
    if phase.goal is not None:
        return _STEER.format(arm=phase.arm)
    if phase.name == "done":
        return "Pick hold."
    return f"Should the {phase.arm} gripper act now?"
