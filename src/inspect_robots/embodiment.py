"""The Embodiment interface — Inspect Robots's second swappable input.

An [`Embodiment`][inspect_robots.embodiment.Embodiment] is the "body + world": a real robot or a
simulator. It
produces observations, executes actions, and owns the action/observation spaces,
the native control rate, and reset/safety machinery.

Designed around real-robot reality: ``reset`` may drive to a home pose and block
on human confirmation; there is no guaranteed privileged success oracle.
Simulators are a stricter special case that opt into extra ``capabilities``.

Per R1 (see the design doc): ``step()`` returns as soon as the command is issued
and does NOT block for the control period — the framework owns pacing — unless
the embodiment declares the ``"self_paced"`` capability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from inspect_robots.scene import Scene
from inspect_robots.spaces import Box, ObservationSpace
from inspect_robots.types import Action, Observation, StepResult

if TYPE_CHECKING:
    from inspect_robots.task import TaskEnvelope

# Opt-in capability flags an embodiment may advertise.
Capability = str
SEEDABLE: Capability = "seedable"
RESETTABLE: Capability = "resettable"
AUTO_RESET: Capability = "auto_reset"
PRIVILEGED_SUCCESS: Capability = "privileged_success"
RENDERABLE: Capability = "renderable"
SELF_PACED: Capability = "self_paced"


@dataclass(frozen=True)
class EmbodimentInfo:
    """Static description of an embodiment for compatibility checking + logging.

    ``docs`` contains free-form markdown operating notes for policies that can
    read text, such as joint layout and positive directions, zero-pose
    geometry, gripper polarity, frame conventions, and workspace hints. Keep
    it concise because consumers inject it into system prompts verbatim.
    ``None`` means no notes are offered; consumers must treat empty and
    whitespace-only strings the same as absence.
    """

    name: str
    action_space: Box
    observation_space: ObservationSpace
    control_hz: float | None = None
    is_simulated: bool = False
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    # Setup-hook names and target kinds this embodiment can realize (for R7
    # scene-realizability checks). Empty means "unconstrained" for the tracer.
    supported_setups: frozenset[str] = field(default_factory=frozenset)
    supported_target_kinds: frozenset[str] = field(default_factory=frozenset)
    docs: str | None = None


@runtime_checkable
class Embodiment(Protocol):
    """The robot/simulator contract.

    Embodiments may additionally define an **optional** ``bind_task(envelope)``
    hook (not part of this Protocol, so existing embodiments stay conformant):
    ``eval()`` calls it with the task's
    [`TaskEnvelope`][inspect_robots.task.TaskEnvelope] before the
    compatibility check, letting adapters learn the rollout horizon — e.g. an
    operator countdown showing elapsed/total. The hook is optional *input*,
    not a guarantee: it never fires on direct ``rollout()`` calls or on older
    cores, so adapters must keep a graceful fallback. It fires once per
    ``eval()``, which can be several times over an embodiment's lifetime;
    each call replaces the previous envelope. ``EmbodimentBase`` ships a
    no-op default.
    """

    info: EmbodimentInfo

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Prepare a scene and return its initial observation, using ``seed`` if supported."""
        ...

    def step(self, action: Action) -> StepResult:
        """Issue one action without waiting for the control period unless ``self_paced``."""
        ...

    def close(self) -> None:
        """Release hardware, simulator, or transport resources held by the adapter."""
        ...


class EmbodimentBase(ABC):
    """Optional base class with a no-op ``close``; inherit for the convenience."""

    info: EmbodimentInfo

    @abstractmethod
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Prepare a scene and return its initial observation, using ``seed`` if supported."""
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Issue one action without waiting for the control period unless ``self_paced``."""
        ...

    def bind_task(self, envelope: TaskEnvelope) -> None:  # noqa: B027 - no-op default
        """Default: embodiments with nothing to display or pre-allocate ignore it."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default
        """Default: nothing to release."""
