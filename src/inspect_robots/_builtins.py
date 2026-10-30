"""Register the in-tree builtin components with the registry.

Imported lazily by [`inspect_robots.registry`][inspect_robots.registry] on first
lookup, so importing ``inspect_robots`` stays cheap and free of import cycles.
"""

from __future__ import annotations

from inspect_robots.grader import operator_grader, vlm_grader
from inspect_robots.logging import JsonLogSink, LiveLogSink, RerunSink
from inspect_robots.mock import CubePickEmbodiment, NoopPolicy, RandomPolicy, ScriptedPolicy
from inspect_robots.registry import (
    analyzer,
    embodiment,
    grader,
    guard,
    policy,
    scorer,
    sink,
    task,
)
from inspect_robots.scene import Scene
from inspect_robots.scorer import (
    episode_length,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    success_at_end,
)
from inspect_robots.spaces import Box
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

# Graders
grader("operator")(operator_grader)
grader("vlm")(vlm_grader)

# Sinks
sink("json")(JsonLogSink)
sink("live-json")(LiveLogSink)
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


@analyzer("scores")
def _scores_analyzer() -> object:
    """Stored versus re-scored metrics for each log, with the number of edits."""
    from inspect_robots.analyzer import _ScoresAnalyzer

    return _ScoresAnalyzer()


@analyzer("terminations")
def _terminations_analyzer() -> object:
    """Termination reasons counted across logs, with each stated reason."""
    from inspect_robots.analyzer import _TerminationsAnalyzer

    return _TerminationsAnalyzer()


@analyzer("digest")
def _digest_analyzer() -> object:
    """The offline run digest that ``summarize`` produces without a model."""
    from inspect_robots.analyzer import _DigestAnalyzer

    return _DigestAnalyzer()


@analyzer("html")
def _html_analyzer() -> object:
    """The HTML viewer page for each log."""
    from inspect_robots.analyzer import _HtmlAnalyzer

    return _HtmlAnalyzer()


@guard("clamp")
def _clamp_guard(action_space: Box) -> object:
    """Clamp every action to the action space's bounds."""
    from inspect_robots.guard import ClampGuard

    return ClampGuard(action_space)


@guard("delta-limit")
def _delta_limit_guard(action_space: Box, max_delta: float | None = None) -> object:
    """Cap each step's change; ``max_delta`` in the space's native units (default: derived)."""
    from inspect_robots.guard import DeltaLimitGuard

    return DeltaLimitGuard(action_space, max_delta=max_delta)
