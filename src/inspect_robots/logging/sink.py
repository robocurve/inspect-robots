"""The LogSink protocol, its optional extension, and a no-op base implementation.

A sink observes a run's lifecycle. The rollout engine and ``eval()`` call these
hooks in a fixed order: ``on_eval_start`` → (per trial: ``on_trial_start`` →
``log_step``* → ``on_trial_end``) → ``on_eval_end``.

Sinks may additionally define the duck-typed
``log_policy_messages(t, messages)`` extension. The rollout calls it at most
once per control step and only when the policy performed an inference. Policy
implementations are expected to supply plain-JSON-type messages shaped like
``TrialRecord.policy_transcript`` entries, but core does not enforce or
normalize that shape on this live path, so sinks must render defensively.
Sinks must not mutate the supplied messages. The extension deliberately stays
off both ``LogSink`` and ``NullSink``: it must not change structural protocol
conformance or advertise a no-op that makes policies build transcript deltas.

A second duck-typed extension, ``bind_spaces(action_space, observation_space)``,
is called by ``eval()`` exactly once per run, before ``on_eval_start``, with the
resolved embodiment ``Box`` and ``ObservationSpace``. A sink that defines a
callable ``bind_spaces`` MUST accept this two-argument signature; like
``log_policy_messages``, the name is claimed by the eval loop.

A third duck-typed extension, ``bind_frames_dir(frames_dir)``, is called by
``eval()`` exactly once per run, after ``bind_spaces`` and before
``on_eval_start``. The argument is the run's stored-frame directory as a string,
or ``None`` when frame storage is off. A callable with this claimed name MUST
accept that one-argument signature. It deliberately remains off ``LogSink`` and
``NullSink`` like the other optional extensions.

A fourth duck-typed extension, ``on_eval_error(error)``, is called by ``eval()``
when an exception escapes while or after ``on_eval_start``. It gives sinks whose
startup was attempted one best-effort cleanup point while preserving the original
exception. A callable with this claimed name MUST accept the escaping
``BaseException``. It remains off ``LogSink`` so existing sinks keep their
structural conformance unchanged.
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
        """Receive the recorded trajectory (partial on error) after the trial's last step."""
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

    def on_eval_error(self, error: BaseException) -> None:
        """Provide the optional no-op for sinks that need abort cleanup."""
        return None
