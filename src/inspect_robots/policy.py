"""The Policy (VLA) interface — one of Inspect Robots's two swappable inputs.

A [`Policy`][inspect_robots.policy.Policy] is the "brain": given an
[`Observation`][inspect_robots.types.Observation]
(plus the scene's instruction), it returns an
[`ActionChunk`][inspect_robots.types.ActionChunk] to be executed open-loop.

The public contract is a runtime-checkable [`Policy`][inspect_robots.policy.Policy] ``Protocol`` so
callers
can wrap existing models without inheriting. [`PolicyBase`][inspect_robots.policy.PolicyBase] is an
optional
convenience ABC with sane defaults.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from inspect_robots.scene import Scene
from inspect_robots.spaces import Box, ObservationSpace
from inspect_robots.types import ActionChunk, Observation

if TYPE_CHECKING:
    from inspect_robots.embodiment import EmbodimentInfo
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.task import TaskEnvelope


@dataclass(frozen=True)
class PolicyConfig:
    """Inference-time configuration, recorded in the eval log.

    The VLA analog of Inspect's ``GenerateConfig``: action-chunk handling and
    sampling knobs that affect reproducibility.
    """

    action_horizon: int = 1
    replan_interval: int | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if (
            not isinstance(self.action_horizon, int)
            or isinstance(self.action_horizon, bool)
            or self.action_horizon < 1
        ):
            raise ValueError(f"action_horizon must be an integer >= 1, got {self.action_horizon!r}")
        if self.replan_interval is not None and (
            not isinstance(self.replan_interval, int)
            or isinstance(self.replan_interval, bool)
            or self.replan_interval < 1
        ):
            raise ValueError(
                f"replan_interval must be an integer >= 1 or None, got {self.replan_interval!r}"
            )
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0.0
        ):
            raise ValueError(
                f"temperature must be a finite number >= 0.0 or None, got {self.temperature!r}"
            )


@dataclass(frozen=True)
class PolicyInfo:
    """Static description of a policy used for compatibility checking + logging."""

    name: str
    action_space: Box
    observation_space: ObservationSpace = field(default_factory=ObservationSpace)
    # Desired control rate (Hz), if the policy was trained for a specific one.
    control_hz: float | None = None
    # Policy checkpoint hash, revision, or identifier.
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Validate policy metadata and spaces."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"PolicyInfo name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.action_space, Box):
            raise TypeError(f"action_space must be a Box, got {type(self.action_space).__name__}")
        if not isinstance(self.observation_space, ObservationSpace):
            raise TypeError(
                f"observation_space must be an ObservationSpace, "
                f"got {type(self.observation_space).__name__}"
            )
        if self.control_hz is not None and (
            isinstance(self.control_hz, bool)
            or not isinstance(self.control_hz, (int, float))
            or not math.isfinite(self.control_hz)
            or self.control_hz <= 0.0
        ):
            raise ValueError(
                f"control_hz must be a positive finite number or None, got {self.control_hz!r}"
            )
        if self.checkpoint is not None and not isinstance(self.checkpoint, str):
            raise TypeError(
                f"checkpoint must be a string or None, got {type(self.checkpoint).__name__}"
            )


@runtime_checkable
class Policy(Protocol):
    """The VLA contract.

    Policies may additionally define six optional hooks, none part of this
    Protocol so existing policies stay conformant. ``bind(embodiment_info)``
    lets embodiment-adaptive policies adopt the embodiment's spaces; ``eval()``
    calls it after resolving both components and before compatibility checking.
    ``bind_task(envelope)`` lets horizon-aware policies learn the task identity
    and step budget before rollouts start; ``eval()`` calls it before the first
    rollout, but it never fires on a direct ``rollout()`` call, so policies must
    keep a fallback.
    ``on_trial_start(scene_id, epoch, log_dir, run_id)`` runs immediately before
    each trial's rollout. It is a policy lifecycle hook, distinct from the sink
    bus hook of the same name, which runs first and takes only scene id and epoch.
    ``on_trial_end(record, log_dir, run_id)`` runs when a trial finishes (including
    errored and cancelled trials, except trials whose ``on_trial_start`` raised,
    which never reached ``reset()``), before sinks see the record. Mutations to
    ``record.metadata`` land in the log, and exceptions degrade the run to
    status="error" rather than crashing the overall evaluation.
    ``transcript()`` returns a small JSON-serializable audit record for the
    current trial, such as an LLM conversation. The framework calls it once per
    trial at trial end after a successful ``reset()``, including errored trials.
    It must be idempotent and safe between resets, must not mutate policy state,
    and its return value must not alias live state. Camera images must not be
    embedded because frame sidecars already persist them. Collection runs on
    the rollout thread and is best-effort: the framework normalizes and bounds
    the result, and a raising or misbehaving hook cannot change trial outcome.
    ``transcript_delta()`` returns plain-JSON-type messages appended since its
    previous call, or since ``reset()`` on the first call, and returns ``None``
    or an empty list when nothing is new. Implementations must sanitize only
    the new slice in O(new messages), including eliding image bytes before the
    result reaches visualization sinks, and ``reset()`` must rewind the cursor.
    ``PolicyBase`` ships defaults for ``bind()``, ``bind_task()``, and
    ``transcript()`` but deliberately has no ``transcript_delta()`` default:
    policies must opt in so every inference does not pay for a no-op hook call.
    """

    info: PolicyInfo
    config: PolicyConfig

    def reset(self, scene: Scene) -> None:
        """Begin a scene with any policy-local state cleared or initialized."""
        ...

    def act(self, observation: Observation) -> ActionChunk:
        """Infer a non-empty open-loop action chunk from the latest observation."""
        ...


class PolicyBase(ABC):
    """Optional base class providing defaults; inherit only for the helpers."""

    info: PolicyInfo
    config: PolicyConfig = PolicyConfig()

    def bind(self, embodiment_info: EmbodimentInfo) -> None:  # noqa: B027 - no-op default
        """Default: fixed-space policies ignore the embodiment they run on."""

    def bind_task(self, envelope: TaskEnvelope) -> None:  # noqa: B027 - no-op default
        """Default: horizon-unaware policies ignore the task envelope."""

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:  # noqa: B027
        """Optional: Hook called by eval() immediately before each trial's rollout.

        This policy hook receives log context and is distinct from the sink bus
        hook of the same name, which runs first with only scene id and epoch.
        """

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:  # noqa: B027
        """Optional: Hook called by eval() when a trial completes.

        Called for errored and cancelled trials too, except trials whose
        ``on_trial_start`` raised, which never reached ``reset()``, before sinks
        see the record. Mutations to record.metadata land in the log, and
        exceptions degrade the run to status="error" rather than crashing the
        overall evaluation.
        """

    def reset(self, scene: Scene) -> None:  # noqa: B027 - intentional no-op default
        """Default: stateless policies need no per-scene reset."""

    def transcript(self) -> Any | None:
        """Return a JSON-serializable audit record for the current trial, or None."""
        return None

    @abstractmethod
    def act(self, observation: Observation) -> ActionChunk:
        """Infer a non-empty open-loop action chunk from the latest observation."""
        ...
