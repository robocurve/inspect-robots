"""Register the in-tree builtin components with the registry.

Imported lazily by [`inspect_robots.registry`][inspect_robots.registry] on first
lookup, so importing ``inspect_robots`` stays cheap and free of import cycles.
"""

from __future__ import annotations

from inspect_robots.logging import JsonLogSink, RerunSink
from inspect_robots.mock import CubePickEmbodiment, NoopPolicy, RandomPolicy, ScriptedPolicy
from inspect_robots.registry import embodiment, policy, scorer, sink, task
from inspect_robots.scene import Scene
from inspect_robots.scorer import (
    episode_length,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    success_at_end,
)
from inspect_robots.task import Task

# Embodiments
embodiment("cubepick")(CubePickEmbodiment)

# Policies
policy("scripted")(ScriptedPolicy)
policy("random")(RandomPolicy)
policy("noop")(NoopPolicy)

# Scorers
scorer("success_at_end")(success_at_end)
scorer("episode_length")(episode_length)
scorer("min_distance_to_goal")(min_distance_to_goal)
scorer("reached_goal_state")(reached_goal_state)
scorer("operator")(operator_scorer)

# Sinks
sink("json")(JsonLogSink)
sink("rerun")(RerunSink)


@task("cubepick-reach")
def _cubepick_reach(num_scenes: int = 4, max_steps: int = 80) -> Task:
    """A simple reach benchmark over a handful of seeded cube layouts."""
    return Task(
        name="cubepick-reach",
        scenes=[
            Scene(id=f"scene-{i}", instruction="reach the cube", init_seed=i)
            for i in range(num_scenes)
        ],
        scorer=success_at_end(),
        max_steps=max_steps,
    )
