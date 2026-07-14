"""Scoring: Scores, the Scorer protocol, epoch reducers, and builtin scorers.

Mirrors Inspect AI's ``@scorer``/reducer split. A scorer maps a recorded
trajectory (+ the scene's ``Target``) to a [`Score`][inspect_robots.scorer.Score]; an epoch
*reducer* collapses the per-epoch scores of one scene into a single score before metrics
aggregate across scenes.

Scorers consume the *recorded* trajectory (not a live environment), so scoring is
reproducible from a saved log.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb
from statistics import mean as _mean
from statistics import median as _median
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from inspect_robots.scene import Target

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord

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
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return 0.0


@runtime_checkable
class Scorer(Protocol):
    """Maps a recorded trajectory + scene target to a [`Score`][inspect_robots.scorer.Score]."""

    @property
    def name(self) -> str:
        """Stable identifier used as the score key in evaluation results."""
        ...

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        """Assign one score from the recorded trajectory and optional target."""
        ...


# --------------------------------------------------------------------------- #
# Epoch reducers: list[Score] -> Score  (namespaced separately from metrics)
# --------------------------------------------------------------------------- #
def _numeric(value: ScoreValue) -> float:
    """Strictly coerce a value to a number for numeric reduction.

    Unlike [`value_to_float`][inspect_robots.scorer.value_to_float] (which is lenient for metric
    aggregation), this
    raises on a non-numeric string rather than silently coercing it to 0.0 — so a
    ``mean`` over categorical scores fails loudly instead of lying.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        raise TypeError(
            f"cannot numerically reduce non-numeric score value {value!r}; "
            "use a categorical reducer such as 'mode'"
        ) from None


def reduce_mean(scores: Sequence[Score]) -> Score:
    """Collapse numeric epoch values with the arithmetic mean."""
    return Score(value=_mean(_numeric(s.value) for s in scores))


def reduce_median(scores: Sequence[Score]) -> Score:
    """Collapse numeric epoch values with the median."""
    return Score(value=_median(_numeric(s.value) for s in scores))


def reduce_max(scores: Sequence[Score]) -> Score:
    """Keep the largest numeric epoch value."""
    return Score(value=max(_numeric(s.value) for s in scores))


def reduce_min(scores: Sequence[Score]) -> Score:
    """Keep the smallest numeric epoch value."""
    return Score(value=min(_numeric(s.value) for s in scores))


def reduce_mode(scores: Sequence[Score]) -> Score:
    """Most common raw value (works for categorical scores). Deterministic."""
    values = [s.value for s in scores]
    counts = Counter(values)
    best = max(values, key=lambda v: (counts[v], str(v)))
    return Score(value=best)


def pass_at_k(k: int) -> Reducer:
    """Unbiased pass@k estimator over the epoch scores (success = value >= 0.5)."""
    if k < 1:
        raise ValueError("k must be >= 1")

    def reducer(scores: Sequence[Score]) -> Score:
        n = len(scores)
        c = sum(1 for s in scores if _numeric(s.value) >= 0.5)
        if k > n:
            raise ValueError(f"pass_at_{k} needs at least {k} epochs, got {n}")
        # 1 - C(n-c, k) / C(n, k): probability >=1 of k draws is correct.
        value = 1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)
        return Score(value=value)

    return reducer


_REDUCERS: dict[str, Reducer] = {
    "mean": reduce_mean,
    "median": reduce_median,
    "max": reduce_max,
    "min": reduce_min,
    "mode": reduce_mode,
}


def get_reducer(name: str) -> Reducer:
    """Resolve a builtin reducer or parse a dynamic ``pass_at_<k>`` name."""
    if name in _REDUCERS:
        return _REDUCERS[name]
    if name.startswith("pass_at_"):
        try:
            return pass_at_k(int(name[len("pass_at_") :]))
        except ValueError as exc:
            raise ValueError(f"invalid pass@k reducer {name!r}: {exc}") from None
    raise ValueError(f"unknown epoch reducer {name!r}; known: {sorted(_REDUCERS)} or 'pass_at_<k>'")


def reduce_scores(name: str, scores: Sequence[Score]) -> Score:
    """Apply the named epoch reducer to one scene's scores."""
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


def _distances(record: TrialRecord) -> list[float]:
    return [float(s.result.info["distance"]) for s in record.steps if "distance" in s.result.info]


@dataclass(frozen=True)
class _MinDistanceToGoal:
    name: str = "min_distance_to_goal"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        dists = _distances(record)
        if not dists:
            return Score(value=float("inf"), explanation="no distance signal recorded")
        return Score(value=min(dists))


def min_distance_to_goal() -> Scorer:
    """Score = the closest the effector got to the goal (lower is better)."""
    return _MinDistanceToGoal()


@dataclass(frozen=True)
class _ReachedGoalState:
    threshold: float
    name: str = "reached_goal_state"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        dists = _distances(record)
        reached = bool(dists) and min(dists) <= self.threshold
        return Score(value=reached, explanation=f"min_distance <= {self.threshold}")


def reached_goal_state(threshold: float = 0.05) -> Scorer:
    """Success iff the effector came within ``threshold`` of the goal."""
    return _ReachedGoalState(threshold=threshold)


# Recognized affirmative operator verdicts (case-insensitive).
_OPERATOR_SUCCESS = frozenset({"success", "pass", "yes", "y", "1", "true"})


@dataclass(frozen=True)
class _OperatorScorer:
    name: str = "operator"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        # R6: the human verdict is captured once during rollout and recorded;
        # this scorer only READS it, so scoring stays reproducible from a log.
        verdict = record.operator_judgement
        if verdict is None:
            return Score(value=False, explanation="no operator judgement recorded")
        success = verdict.strip().lower() in _OPERATOR_SUCCESS
        return Score(value=success, explanation=f"operator verdict: {verdict!r}")


def operator_scorer() -> Scorer:
    """Score from the human operator's recorded success judgement (R6)."""
    return _OperatorScorer()


class VLMScorer:
    """Reserved interface (R10): score from a VLM classifier over final frames.

    Implemented in a later milestone; instantiating and calling it raises so the
    contract is visible but no half-baked behavior ships.
    """

    name = "vlm"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        """Fail explicitly because VLM scoring is reserved but not implemented."""
        raise NotImplementedError(
            "VLMScorer is a reserved interface; not implemented in this release"
        )
