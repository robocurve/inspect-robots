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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from inspect_robots.scene import Scene
from inspect_robots.spaces import Box, ObservationSpace
from inspect_robots.types import ActionChunk, Observation

if TYPE_CHECKING:
    from inspect_robots.embodiment import EmbodimentInfo
    from inspect_robots.rollout import TrialRecord


@dataclass(frozen=True)
class PolicyConfig:
    """Inference-time configuration, recorded in the eval log.

    The VLA analog of Inspect's ``GenerateConfig``: action-chunk handling and
    sampling knobs that affect reproducibility.
    """

    action_horizon: int = 1
    replan_interval: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class PolicyInfo:
    """Static description of a policy used for compatibility checking + logging."""

    name: str
    action_space: Box
    observation_space: ObservationSpace = field(default_factory=ObservationSpace)
    # Desired control rate (Hz), if the policy was trained for a specific one.
    control_hz: float | None = None


@runtime_checkable
class Policy(Protocol):
    """The VLA contract.

    Policies may additionally define three optional hooks, none part of this
    Protocol so existing policies stay conformant. ``bind(embodiment_info)``
    lets embodiment-adaptive policies adopt the embodiment's spaces; ``eval()``
    calls it after resolving both components and before compatibility checking.
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
    ``PolicyBase`` ships defaults for ``bind()`` and ``transcript()`` but
    deliberately has no ``transcript_delta()`` default: policies must opt in so
    every inference does not pay for a no-op hook call.
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

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:  # noqa: B027
        """Optional: hook called by eval() when a trial completes.

        Allows the policy to persist artifacts.
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
