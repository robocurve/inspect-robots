"""The Policy (VLA) interface — one of RoboLens's two swappable inputs.

A :class:`Policy` is the "brain": given an :class:`~robolens.types.Observation`
(plus the scene's instruction), it returns an
:class:`~robolens.types.ActionChunk` to be executed open-loop.

The public contract is a runtime-checkable :class:`Policy` ``Protocol`` so callers
can wrap existing models without inheriting. :class:`PolicyBase` is an optional
convenience ABC with sane defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from robolens.scene import Scene
from robolens.spaces import Box, ObservationSpace
from robolens.types import ActionChunk, Observation


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


@runtime_checkable
class Policy(Protocol):
    """The VLA contract."""

    info: PolicyInfo
    config: PolicyConfig

    def reset(self, scene: Scene) -> None: ...

    def act(self, observation: Observation) -> ActionChunk: ...


class PolicyBase(ABC):
    """Optional base class providing defaults; inherit only for the helpers."""

    info: PolicyInfo
    config: PolicyConfig = PolicyConfig()

    def reset(self, scene: Scene) -> None:  # noqa: B027 - intentional no-op default
        """Default: stateless policies need no per-scene reset."""

    @abstractmethod
    def act(self, observation: Observation) -> ActionChunk: ...
