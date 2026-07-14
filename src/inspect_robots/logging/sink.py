"""The LogSink protocol and a no-op base implementation.

A sink observes a run's lifecycle. The rollout engine and ``eval()`` call these
hooks in a fixed order: ``on_eval_start`` → (per trial: ``on_trial_start`` →
``log_step``* → ``on_trial_end``) → ``on_eval_end``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog, EvalSpec
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.types import Action, Observation, StepResult


@runtime_checkable
class LogSink(Protocol):
    """Observes the lifecycle of an evaluation run."""

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Receive immutable run identity before any trial hooks."""
        ...

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Mark the scene and epoch whose steps will follow."""
        ...

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Observe one completed control transition in step order."""
        ...

    def on_trial_end(self, record: TrialRecord) -> None:
        """Receive the complete trajectory after its final step."""
        ...

    def on_eval_end(self, log: EvalLog) -> None:
        """Receive the immutable aggregate after all trials finish."""
        ...


class NullSink:
    """A sink that does nothing — a convenient base for partial implementations."""

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Provide the optional no-op for sinks that need no run setup."""
        return None

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Provide the optional no-op for sinks that need no per-trial setup."""
        return None

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Provide the optional no-op for sinks that do not consume live transitions."""
        return None

    def on_trial_end(self, record: TrialRecord) -> None:
        """Provide the optional no-op for sinks that do not consume completed trajectories."""
        return None

    def on_eval_end(self, log: EvalLog) -> None:
        """Provide the optional no-op for sinks that need no finalization."""
        return None
