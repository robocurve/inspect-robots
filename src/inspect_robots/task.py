"""The Task — an embodiment-agnostic benchmark definition.

Mirrors Inspect AI's ``Task = dataset + scorer + epochs/reducer``, adapted for
robotics: the dataset is a sequence of [`Scene`][inspect_robots.scene.Scene] initial
conditions and the rollout horizon (``max_steps``) lives here.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class TaskEnvelope:
    """Identity and rollout limits of a task, safe to hand to adapters.

    This is what ``eval()`` passes to an embodiment's optional
    ``bind_task(envelope)`` hook: enough for the adapter to display or
    pre-allocate for the run (e.g. an operator countdown against
    ``max_steps``), and nothing that would let it second-guess scoring or the
    dataset. Deliberately carries no control rate — the rollout enforces no
    wall-clock rate of its own (R1, revised); a self-paced embodiment owns its
    own cadence.
    """

    name: str
    max_steps: int


@dataclass
class Task:
    """A benchmark: scenes + scorer(s) + horizon, independent of any embodiment.

    ``scorer`` accepts scorer objects or **registry names** (e.g.
    ``scorer="success_at_end"``), or a sequence mixing both.

    Declare exactly one rollout horizon. ``max_steps`` is already resolved;
    ``max_seconds`` is keyword-only and is resolved against the paired
    embodiment's positive finite ``control_hz`` by ``eval()``.
    """

    name: str
    scenes: Sequence[Scene]
    scorer: Scorer | str | Sequence[Scorer | str]
    max_steps: int | None = None
    epochs: int | Epochs = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_seconds: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if (self.max_steps is None) == (self.max_seconds is None):
            raise ConfigError(
                f"Task {self.name!r}: declare exactly one of max_steps or max_seconds"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ConfigError(f"Task {self.name!r}: max_steps must be >= 1, got {self.max_steps}")
        if self.max_seconds is not None and (
            isinstance(self.max_seconds, bool)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ConfigError(
                f"Task {self.name!r}: max_seconds must be finite and > 0, got {self.max_seconds!r}"
            )
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

    @property
    def envelope(self) -> TaskEnvelope:
        """Return the already-resolved adapter view of a step-based task.

        Seconds-based tasks require an embodiment rate and must use
        [`resolve_envelope`][inspect_robots.task.Task.resolve_envelope].
        """
        return self.resolve_envelope(None)

    def resolve_envelope(self, control_hz: float | None) -> TaskEnvelope:
        """Resolve the adapter-safe task view against an embodiment rate.

        Compatibility checking normally rejects an invalid rate before this
        method is called. The validation here is a defensive public-API guard
        for callers resolving envelopes directly.
        """
        if self.max_steps is not None:
            return TaskEnvelope(name=self.name, max_steps=self.max_steps)

        if (
            control_hz is None
            or isinstance(control_hz, bool)
            or not math.isfinite(control_hz)
            or control_hz <= 0
        ):
            raise ConfigError(
                f"Task {self.name!r}: max_seconds requires an embodiment control_hz "
                f"that is finite and > 0, got {control_hz!r}"
            )

        assert self.max_seconds is not None  # exactly-one invariant from __post_init__
        raw_steps = self.max_seconds * control_hz
        if not math.isfinite(raw_steps):
            raise ConfigError(
                f"Task {self.name!r}: max_seconds={self.max_seconds!r} at "
                f"control_hz={control_hz!r} does not yield a finite step budget"
            )
        # Both factors are positive, so the mathematical ceiling is at least
        # one even if their binary-float product underflows to 0.0.
        return TaskEnvelope(name=self.name, max_steps=max(1, math.ceil(raw_steps)))
