"""Action/observation spaces and action *semantics*.

Spaces describe the *shape* of actions and observations; :class:`ActionSemantics`
describes what an action *means* (control mode, rotation representation, gripper
kind, reference frame). Semantics are what make compatibility checking real (a
7-DoF VLA vs a 6-DoF arm; delta vs absolute poses) and make temporal ensembling
correct.

This module ships a minimal-but-functional core for the tracer slice; richer
validation and the full :class:`StateSpec` vocabulary are layered on in a later
step without changing these signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

ControlMode = Literal[
    "joint_pos",
    "joint_vel",
    "eef_delta_pose",
    "eef_abs_pose",
    "eef_delta_pos",
]
RotationRepr = Literal[
    "none",
    "quat_wxyz",
    "quat_xyzw",
    "rot6d",
    "axis_angle",
    "euler_xyz",
]
GripperKind = Literal["none", "continuous", "binary"]
Frame = Literal["base", "world", "camera"]


@dataclass(frozen=True)
class ActionSemantics:
    """What an action vector *means*. Attached to an action :class:`Box`."""

    control_mode: ControlMode
    rotation_repr: RotationRepr = "none"
    gripper: GripperKind = "none"
    frame: Frame = "base"


@dataclass(frozen=True, eq=False)
class Box:
    """A continuous box-shaped space. Optional ``low``/``high`` bounds and, for
    action spaces, :class:`ActionSemantics`."""

    shape: tuple[int, ...]
    low: npt.NDArray[np.floating[Any]] | None = None
    high: npt.NDArray[np.floating[Any]] | None = None
    semantics: ActionSemantics | None = None

    @property
    def dim(self) -> int:
        out = 1
        for n in self.shape:
            out *= n
        return out


@dataclass(frozen=True)
class CameraSpec:
    """An image stream an embodiment provides or a policy requires."""

    name: str
    height: int
    width: int
    channels: int = 3


@dataclass(frozen=True)
class ObservationSpace:
    """The observations an embodiment provides / a policy requires.

    ``state_keys`` are the proprioception keys (controlled vocabulary). A later
    step replaces the bare set with a richer ``StateSpec`` (units, dtypes).
    """

    cameras: tuple[CameraSpec, ...] = ()
    state_keys: frozenset[str] = field(default_factory=frozenset)
