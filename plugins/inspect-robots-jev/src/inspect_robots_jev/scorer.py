"""Pose-based success for cube-into-bowl, read from the last recorded action's meta.

``eval()`` scores a trial before ``policy.on_trial_end`` runs, so the scorer
reads ``record.steps[-1].action.meta["jev"]`` (approvers preserve meta) rather
than ``record.metadata``. A cube that was never re-detected after release is
not credited: its last pose would be the in-gripper estimate. ``seen_ago``
and ``max_unseen`` count policy decisions, not control ticks.
"""

from __future__ import annotations

import math
from typing import Any

from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Target
from inspect_robots.scorer import Score, Scorer


class _CubeInBowl:
    name = "jev_cube_in_bowl"

    def __init__(
        self, cube: str, bowl: str, radius_m: float, max_above_m: float, max_unseen: int
    ) -> None:
        self._cube = cube
        self._bowl = bowl
        self._radius = radius_m
        self._max_above = max_above_m
        self._max_unseen = max_unseen

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        if not record.steps:
            return Score(value=False, explanation="no jev world state recorded")
        meta: Any = record.steps[-1].action.meta.get("jev")
        if not isinstance(meta, dict):
            return Score(value=False, explanation="no jev world state recorded")
        world = meta.get("world", {})
        cube, bowl = world.get(self._cube), world.get(self._bowl)
        if cube is None or bowl is None:
            return Score(value=False, explanation=f"{self._cube} or {self._bowl} never observed")
        if meta.get("held") or cube.get("from_gripper"):
            return Score(value=False, explanation=f"{self._cube} still held at the end")
        seen_ago = int(cube.get("seen_ago", 0))
        if seen_ago > self._max_unseen:
            return Score(
                value=False,
                explanation=(f"{self._cube} not observed after release ({seen_ago} decisions ago)"),
            )
        arm = "left" if "left" in cube and "left" in bowl else "right"
        if arm not in cube or arm not in bowl:
            return Score(value=False, explanation="cube and bowl share no arm frame")
        c, b = cube[arm], bowl[arm]
        horizontal = math.hypot(c[0] - b[0], c[1] - b[1])
        above = c[2] - b[2]
        inside = horizontal <= self._radius and above <= self._max_above
        return Score(
            value=inside,
            explanation=(
                f"horizontal {horizontal * 100:.1f} cm, {above * 100:+.1f} cm above bowl tag"
            ),
            metadata={"horizontal_m": horizontal, "above_m": above, "arm": arm},
        )


def cube_in_bowl(
    *,
    cube: str = "cube",
    bowl: str = "bowl",
    radius_m: float = 0.04,
    max_above_m: float = 0.03,
    max_unseen: int = 3,
) -> Scorer:
    """Success when the cube's tag ends within ``radius_m`` of the bowl centre and not above it."""
    return _CubeInBowl(cube, bowl, radius_m, max_above_m, max_unseen)
