"""The Task — an embodiment-agnostic benchmark definition.

Mirrors Inspect AI's ``Task = dataset + scorer + epochs/reducer``, adapted for
robotics: the dataset is a sequence of [`Scene`][inspect_robots.scene.Scene] initial
conditions and the rollout horizon (``max_steps``) lives here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from inspect_robots.errors import ConfigError
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

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ConfigError(f"Epochs count must be >= 1, got {self.count}")


@dataclass
class Task:
    """A benchmark: scenes + scorer(s) + horizon, independent of any embodiment.

    ``scorer`` accepts scorer objects or **registry names** (e.g.
    ``scorer="success_at_end"``), or a sequence mixing both.
    """

    name: str
    scenes: Sequence[Scene]
    scorer: Scorer | str | Sequence[Scorer | str]
    max_steps: int
    epochs: int | Epochs = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigError(f"Task {self.name!r}: max_steps must be >= 1, got {self.max_steps}")
        _ = self.epoch_spec  # validates an int epochs count via Epochs

    @property
    def scorers(self) -> list[Scorer]:
        """Resolve registry names while preserving the declared scorer order."""
        # A str IS a Sequence: treat it as a single registry name, never as a
        # sequence of one-character "scorers".
        if isinstance(self.scorer, str) or not isinstance(self.scorer, Sequence):
            raw: list[Scorer | str] = [self.scorer]
        else:
            raw = list(self.scorer)
        out: list[Scorer] = []
        for entry in raw:
            if isinstance(entry, str):
                from inspect_robots.registry import resolve

                out.append(cast(Scorer, resolve("scorer", entry)))
            else:
                out.append(entry)
        return out

    @property
    def epoch_spec(self) -> Epochs:
        """Normalize an integer epoch count to an ``Epochs`` specification."""
        return self.epochs if isinstance(self.epochs, Epochs) else Epochs(count=self.epochs)
