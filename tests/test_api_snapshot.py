"""Guard the public API surface against accidental growth/shrinkage.

If you intend to change the public API, update ``EXPECTED`` here and note it in
the changelog. Everything not in ``inspect_robots.__all__`` (or prefixed ``_``) is
private and carries no stability guarantee.
"""

from __future__ import annotations

import inspect_robots

EXPECTED = {
    # evaluation + logs
    "eval",
    "eval_set",
    "read_eval_log",
    "EvalLog",
    "EvalResults",
    "EvalSpec",
    "EvalStats",
    "SceneResult",
    "TrialRecord",
    # tasks, scenes, scoring
    "Task",
    "Epochs",
    "Scene",
    "Target",
    "Scorer",
    "Score",
    "success_at_end",
    "episode_length",
    "min_distance_to_goal",
    "reached_goal_state",
    "operator_scorer",
    # the two swappable inputs
    "Policy",
    "PolicyBase",
    "PolicyInfo",
    "PolicyConfig",
    "Embodiment",
    "EmbodimentBase",
    "EmbodimentInfo",
    # types & spaces
    "Observation",
    "Action",
    "ActionChunk",
    "StepResult",
    "Box",
    "ActionSemantics",
    "ObservationSpace",
    "CameraSpec",
    "StateField",
    "StateSpec",
    # registry decorators + resolution
    "task",
    "policy",
    "embodiment",
    "scorer",
    "sink",
    "registered",
    "resolve",
    # meta
    "__version__",
}


def test_public_api_snapshot() -> None:
    assert set(inspect_robots.__all__) == EXPECTED
    # __all__ must have no duplicates.
    assert len(inspect_robots.__all__) == len(set(inspect_robots.__all__))


def test_public_names_are_importable() -> None:
    for name in inspect_robots.__all__:
        assert hasattr(inspect_robots, name), f"{name} declared in __all__ but missing"
