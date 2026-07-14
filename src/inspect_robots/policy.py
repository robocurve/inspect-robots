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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

    Policies may additionally define an **optional** ``bind(embodiment_info)``
    hook (not part of this Protocol, so existing policies stay conformant):
    ``eval()`` calls it after resolving both components and before the
    compatibility check, letting embodiment-adaptive policies — e.g. an LLM
    agent that drives whatever it is paired with — adopt the embodiment's
    spaces. ``PolicyBase`` ships a no-op default.
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

    @abstractmethod
    def act(self, observation: Observation) -> ActionChunk:
        """Infer a non-empty open-loop action chunk from the latest observation."""
        ...
