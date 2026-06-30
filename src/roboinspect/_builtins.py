"""Register the in-tree builtin components with the registry.

Imported lazily by [`roboinspect.registry`][roboinspect.registry] on first lookup, so importing
``roboinspect`` stays cheap and free of import cycles.
"""

from __future__ import annotations

from roboinspect.logging import JsonLogSink, RerunSink
from roboinspect.mock import CubePickEmbodiment, NoopPolicy, RandomPolicy, ScriptedPolicy
from roboinspect.registry import embodiment, policy, scorer, sink, task
from roboinspect.scene import Scene
from roboinspect.scorer import (
    episode_length,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    success_at_end,
)
from roboinspect.task import Task

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
