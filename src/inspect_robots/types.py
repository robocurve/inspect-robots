"""Core observation/action data types exchanged between policy and embodiment.

These are the wire format of a rollout. They are deliberately small, immutable,
and NumPy-native. Arrays are raw (the policy owns model-specific preprocessing);
images are ``(H, W, C)`` ``uint8``.

The dataclasses set ``eq=False`` because they carry NumPy arrays, whose
element-wise ``==`` does not yield a single bool — identity/round-trip semantics
are what callers actually need here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]
StateArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, eq=False)
class Observation:
    """A single multi-modal observation produced by an embodiment.

    ``images`` are keyed by camera name; ``state`` holds proprioception keyed by a
    controlled vocabulary (e.g. ``"eef_pos"``, ``"gripper"``). ``instruction`` is
    the language goal for this step (usually constant across an episode, but may
    change for long-horizon tasks).
    """

    images: Mapping[str, ImageArray] = field(default_factory=dict)
    state: Mapping[str, StateArray] = field(default_factory=dict)
    instruction: str | None = None
    image_times: Mapping[str, float] = field(default_factory=dict)
    state_time: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Action:
    """A single action to apply to an embodiment.

    Semantics (control mode, rotation representation, gripper kind, frame) live on
    the action *space*, not on every action instance —
    see [`inspect_robots.spaces`][inspect_robots.spaces].
    """

    data: StateArray
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class ActionChunk:
    """A horizon of actions predicted by one policy inference.

    Modern VLAs (π0, ACT, diffusion policies) predict ``H`` future actions that
    are executed open-loop because inference is slower than the control rate.
    ``H == 1`` is the degenerate "reactive policy" case. ``control_hz`` is the
    rate the chunk was intended to be played at (``None`` defers to the
    embodiment's native rate); ``inference_latency_s``, when measured, is logged.
    """

    actions: Sequence[Action]
    control_hz: float | None = None
    inference_latency_s: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.actions) == 0:
            raise ValueError("ActionChunk must contain at least one action")

    def __len__(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, eq=False)
class StepResult:
    """The outcome of applying one action to an embodiment.

    ``terminated`` means the task ended (success or hard failure);
    ``termination_reason`` disambiguates (e.g. ``"success"``, ``"collision"``,
    ``"fault"``, ``"out_of_bounds"``). ``truncated`` means a time/horizon cutoff.
    A simulator may expose privileged success via ``info``.
    """

    observation: Observation
    reward: float | None = None
    terminated: bool = False
    termination_reason: str | None = None
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
