"""Scoring: Scores, the Scorer protocol, epoch reducers, and builtin scorers.

Mirrors Inspect AI's ``@scorer``/reducer split. A scorer maps a recorded
trajectory (+ the scene's ``Target``) to a :class:`Score`; an epoch *reducer*
collapses the per-epoch scores of one scene into a single score before metrics
aggregate across scenes.

Scorers consume the *recorded* trajectory (not a live environment), so scoring is
reproducible from a saved log.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean as _mean
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from robolens.scene import Target

if TYPE_CHECKING:
    from robolens.rollout import TrialRecord

ScoreValue = bool | int | float | str
Reducer = Callable[[Sequence["Score"]], "Score"]


@dataclass(frozen=True)
class Score:
    """The outcome a scorer assigns to one trajectory."""

    value: ScoreValue
    explanation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def value_to_float(value: ScoreValue) -> float:
    """Coerce a score value to a float for metric aggregation."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return 0.0


@runtime_checkable
class Scorer(Protocol):
    """Maps a recorded trajectory + scene target to a :class:`Score`."""

    @property
    def name(self) -> str: ...

    def __call__(self, record: TrialRecord, target: Target | None) -> Score: ...


# --------------------------------------------------------------------------- #
# Epoch reducers: list[Score] -> Score  (namespaced separately from metrics)
# --------------------------------------------------------------------------- #
def reduce_mean(scores: Sequence[Score]) -> Score:
    return Score(value=_mean(value_to_float(s.value) for s in scores))


def reduce_max(scores: Sequence[Score]) -> Score:
    return Score(value=max(value_to_float(s.value) for s in scores))


def reduce_mode(scores: Sequence[Score]) -> Score:
    floats = [value_to_float(s.value) for s in scores]
    # majority value; ties resolved toward the larger value for determinism
    return Score(value=max(set(floats), key=lambda v: (floats.count(v), v)))


_REDUCERS: dict[str, Reducer] = {
    "mean": reduce_mean,
    "max": reduce_max,
    "mode": reduce_mode,
}


def get_reducer(name: str) -> Reducer:
    try:
        return _REDUCERS[name]
    except KeyError:
        raise ValueError(f"unknown epoch reducer {name!r}; known: {sorted(_REDUCERS)}") from None


def reduce_scores(name: str, scores: Sequence[Score]) -> Score:
    return get_reducer(name)(scores)


# --------------------------------------------------------------------------- #
# Builtin scorers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SuccessAtEnd:
    name: str = "success_at_end"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        last = record.steps[-1] if record.steps else None
        success = bool(
            last is not None
            and last.result.terminated
            and last.result.termination_reason == "success"
        )
        return Score(
            value=success,
            explanation="reached success termination" if success else "did not succeed",
        )


def success_at_end() -> Scorer:
    """Score 1.0 iff the episode terminated with reason ``"success"``."""
    return _SuccessAtEnd()


@dataclass(frozen=True)
class _EpisodeLength:
    name: str = "episode_length"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        return Score(value=len(record.steps))


def episode_length() -> Scorer:
    """Score = number of environment steps taken."""
    return _EpisodeLength()
