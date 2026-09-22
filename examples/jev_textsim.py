"""Live wording-regression probe for the ``jev`` policy (NOT run in CI).

A point gripper per arm in a 40 x 40 x 30 cm box, one target cube and two
distractors, the plugin's own ``describe_offset`` wording and ``build_menu``
options, and the real Jev over OpenRouter. Reproduces the 2026-09-19 spike
(30/30 with target-only geometry). Run it before changing any wording in
``inspect_robots_jev.world`` or ``inspect_robots_jev.menu``.

Requires ``OPENROUTER_API_KEY``; about 350 calls, well under one cent.
Exit code 1 when fewer than 27 of 30 episodes reach the target.

    uv run --no-sync python examples/jev_textsim.py [episodes]
"""

from __future__ import annotations

import random
import sys
from concurrent.futures import ThreadPoolExecutor

from inspect_robots_jev._decisions import DecisionsClient
from inspect_robots_jev.menu import HOLD, build_menu, parse_option
from inspect_robots_jev.world import ARMS, AXES, Arm, Vec3, describe_offset, distance

BOX = {"x": (-0.20, 0.20), "y": (-0.20, 0.20), "z": (0.0, 0.30)}
STEPS_CM = (0.5, 2.0, 5.0)
SUCCESS_M = 0.0075
MAX_MOVES = 80


def _rand(rng: random.Random) -> Vec3:
    return tuple(round(rng.uniform(*BOX[a]) * 200) / 200 for a in AXES)  # type: ignore[return-value]


def episode(seed: int, client: DecisionsClient) -> tuple[bool, int]:
    rng = random.Random(seed)
    grips: dict[Arm, list[float]] = {arm: list(_rand(rng)) for arm in ARMS}
    target = _rand(rng)
    distractors = ["blue cube", "green cube"]
    arm: Arm = min(ARMS, key=lambda a: distance(tuple(grips[a]), target))  # type: ignore[arg-type]
    history: list[str] = []
    for i in range(MAX_MOVES):
        g = tuple(grips[arm])
        if distance(g, target) <= SUCCESS_M:  # type: ignore[arg-type]
            return True, i
        state = {
            "task": f"Move the {arm} gripper onto the red cube. Move toward it, never away.",
            f"target_point_relative_to_{arm}_gripper": describe_offset(target, g),  # type: ignore[arg-type]
            "straight_line_distance_cm": round(distance(g, target) * 100, 1),  # type: ignore[arg-type]
            "other_objects_on_table_ignore_them": distractors,
            "recent_moves_oldest_first": history[-5:],
        }
        menu = build_menu(arm, "xyz", "red cube", STEPS_CM)
        answer = client.choose(
            state=state,
            instructions=(
                f"Which move takes the {arm} gripper toward the target point? Move in the "
                "direction the target is in, along the axis with the largest remaining gap, "
                "with a step no larger than that gap."
            ),
            criteria=menu,
        )
        history.append(menu[answer.choice].split(";")[0])
        if answer.choice == HOLD:
            continue
        move = parse_option(answer.choice)
        if move.axis is not None:
            k = AXES.index(move.axis)
            lo, hi = BOX[move.axis]
            grips[arm][k] = min(max(grips[arm][k] + move.delta_m, lo), hi)
    return False, MAX_MOVES


def main() -> int:
    """Run the probe and print a one-line verdict."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    client = DecisionsClient()
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda s: episode(s, client), range(n)))
    wins = sum(1 for ok, _ in results if ok)
    steps = [k for ok, k in results if ok]
    mean = sum(steps) / len(steps) if steps else float("nan")
    print(f"{wins}/{n} episodes reached the target; mean moves on success {mean:.1f}")
    return 0 if wins >= 0.9 * n else 1


if __name__ == "__main__":
    sys.exit(main())
