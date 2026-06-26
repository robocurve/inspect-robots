"""The LogSink protocol and a no-op base implementation.

A sink observes a run's lifecycle. The rollout engine and ``eval()`` call these
hooks in a fixed order: ``on_eval_start`` → (per trial: ``on_trial_start`` →
``log_step``* → ``on_trial_end``) → ``on_eval_end``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from robolens.log import EvalLog, EvalSpec
    from robolens.rollout import TrialRecord
    from robolens.types import Action, Observation, StepResult


@runtime_checkable
class LogSink(Protocol):
    """Observes the lifecycle of an evaluation run."""

    def on_eval_start(self, spec: EvalSpec) -> None: ...

    def on_trial_start(self, scene_id: str, epoch: int) -> None: ...

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None: ...

    def on_trial_end(self, record: TrialRecord) -> None: ...

    def on_eval_end(self, log: EvalLog) -> None: ...


class NullSink:
    """A sink that does nothing — a convenient base for partial implementations."""

    def on_eval_start(self, spec: EvalSpec) -> None:
        return None

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        return None

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        return None

    def on_trial_end(self, record: TrialRecord) -> None:
        return None

    def on_eval_end(self, log: EvalLog) -> None:
        return None
