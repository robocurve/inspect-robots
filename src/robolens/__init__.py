"""RoboLens — the Inspect AI for robotics.

An evaluation framework for VLA (vision-language-action) models. A benchmark is
defined once as a :class:`~robolens.task.Task` and run against any compatible
pairing of a :class:`~robolens.policy.Policy` (the VLA) and an
:class:`~robolens.embodiment.Embodiment` (a real robot or simulator).

The public API is everything exported here via ``__all__``. Anything not listed
(or prefixed with ``_``) is private and carries no stability guarantee.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("robolens")
except PackageNotFoundError:  # pragma: no cover - only hit in a non-installed tree
    __version__ = "0.0.0+unknown"

from robolens.eval import eval
from robolens.log import (
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
    read_eval_log,
)

__all__ = [
    "EvalLog",
    "EvalResults",
    "EvalSpec",
    "EvalStats",
    "SceneResult",
    "__version__",
    "eval",
    "read_eval_log",
]
