"""RoboInspect quickstart: evaluate a policy on an embodiment over a benchmark.

Run with:  uv run python examples/quickstart.py

This uses the dependency-free CubePick mock world, so it needs no hardware or
simulator. Swap ``ScriptedPolicy``/``CubePickEmbodiment`` for any registered
plugin policy/embodiment to evaluate a real VLA on a real or simulated robot.
"""

from __future__ import annotations

from roboinspect import eval
from roboinspect.mock import CubePickEmbodiment, ScriptedPolicy
from roboinspect.scene import Scene
from roboinspect.scorer import episode_length, success_at_end
from roboinspect.task import Epochs, Task


def main() -> None:
    # A benchmark = a dataset of scenes (initial conditions) + scorer(s).
    task = Task(
        name="cubepick-reach",
        scenes=[
            Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(5)
        ],
        scorer=[success_at_end(), episode_length()],
        max_steps=80,
        epochs=Epochs(count=2, reducer="mean"),
    )

    # Two swappable inputs: the policy (VLA) and the embodiment (robot/sim).
    (log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir="logs")

    print(f"status:  {log.status}")
    print(f"scenes:  {log.results.total_scenes}   trials: {log.results.total_trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}")


if __name__ == "__main__":
    main()
