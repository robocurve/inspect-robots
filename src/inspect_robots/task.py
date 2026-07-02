"""The Task — an embodiment-agnostic benchmark definition.

Mirrors Inspect AI's ``Task = dataset + scorer + epochs/reducer``, adapted for
robotics: the dataset is a sequence of [`Scene`][inspect_robots.scene.Scene] initial
conditions and the rollout horizon (``max_steps``) and control rate live here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from inspect_robots.scene import Scene
from inspect_robots.scorer import Scorer


@dataclass(frozen=True)
class Epochs:
    """Repeat count plus the reducer used to combine per-epoch scores.

    Mirrors Inspect's ``Epochs(count, reducer)``; reducer is a registered name
    (default ``"mean"``).
    """

    count: int = 1
    reducer: str = "mean"


@dataclass
class Task:
    """A benchmark: scenes + scorer(s) + horizon, independent of any embodiment."""

    name: str
    scenes: Sequence[Scene]
    scorer: Scorer | Sequence[Scorer]
    max_steps: int
    epochs: int | Epochs = 1
    control_hz: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scorers(self) -> list[Scorer]:
        if isinstance(self.scorer, Sequence):
            return list(self.scorer)
        return [self.scorer]

    @property
    def epoch_spec(self) -> Epochs:
        return self.epochs if isinstance(self.epochs, Epochs) else Epochs(count=self.epochs)
