"""Action/observation spaces and action *semantics*.

Spaces describe the *shape* of actions and observations;
[`ActionSemantics`][inspect_robots.spaces.ActionSemantics]
describes what an action *means* (control mode, rotation representation, gripper
kind, reference frame). Semantics are what make compatibility checking real (a
7-DoF VLA vs a 6-DoF arm; delta vs absolute poses) and make temporal ensembling
correct.

This module ships a minimal-but-functional core for the tracer slice; richer
validation and the full [`StateSpec`][inspect_robots.spaces.StateSpec] vocabulary are
layered on in a later step without changing these signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

ControlMode = Literal[
    "joint_pos",
    "joint_vel",
    "joint_delta",
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
    """What an action vector *means*. Attached to an action [`Box`][inspect_robots.spaces.Box].

    ``dim_labels`` optionally names each action dimension (e.g.
    ``("left_j0", ..., "right_gripper")`` for a bimanual arm) so tooling —
    LLM agent policies, logging, visualization — can address dimensions by
    name. Length is validated against the owning box in ``Box.__post_init__``
    (the semantics/shape pairing is only visible there).
    """

    control_mode: ControlMode
    rotation_repr: RotationRepr = "none"
    gripper: GripperKind = "none"
    frame: Frame = "base"
    dim_labels: tuple[str, ...] | None = None


@dataclass(frozen=True, eq=False)
class Box:
    """A continuous box-shaped space. Optional ``low``/``high`` bounds and, for
    action spaces, [`ActionSemantics`][inspect_robots.spaces.ActionSemantics]."""

    shape: tuple[int, ...]
    low: npt.NDArray[np.floating[Any]] | None = None
    high: npt.NDArray[np.floating[Any]] | None = None
    semantics: ActionSemantics | None = None

    def __post_init__(self) -> None:
        for name, bound in (("low", self.low), ("high", self.high)):
            if bound is not None and tuple(bound.shape) != self.shape:
                raise ValueError(
                    f"Box {name} shape {tuple(bound.shape)} != space shape {self.shape}"
                )
        if self.low is not None and self.high is not None and bool(np.any(self.low > self.high)):
            raise ValueError("Box low must be elementwise <= high")
        labels = self.semantics.dim_labels if self.semantics is not None else None
        if labels is not None and len(labels) != self.dim:
            raise ValueError(
                f"ActionSemantics.dim_labels has {len(labels)} entries but the box "
                f"has {self.dim} dimensions"
            )

    @property
    def dim(self) -> int:
        """Return the flattened element count implied by ``shape``."""
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


# Canonical proprioception keys and their conventional units. Adapters are
# encouraged to use these names/units so cross-embodiment compatibility checks on
# state are meaningful (loosely aligned with LeRobot dataset conventions).
CANONICAL_STATE_UNITS: dict[str, str] = {
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "eef_pos": "m",
    "eef_pose": "m+quat",
    "eef_quat": "unit_quat",
    "gripper": "normalized",  # 0 (open) .. 1 (closed)
    "gripper_width": "m",
}


@dataclass(frozen=True)
class StateField:
    """One proprioception field: its key, shape, unit, and dtype."""

    key: str
    shape: tuple[int, ...]
    unit: str = ""
    dtype: str = "float64"


@dataclass(frozen=True)
class StateSpec:
    """A richer description of an embodiment's proprioception than a bare key set."""

    fields: tuple[StateField, ...] = ()

    @property
    def keys(self) -> frozenset[str]:
        """Names used to align rich and compatibility-level state declarations."""
        return frozenset(f.key for f in self.fields)


@dataclass(frozen=True)
class ObservationSpace:
    """The observations an embodiment provides / a policy requires.

    ``state_keys`` is the compatibility-relevant set of proprioception keys.
    ``state`` optionally carries the richer [`StateSpec`][inspect_robots.spaces.StateSpec]
    (shapes/units).
    """

    cameras: tuple[CameraSpec, ...] = ()
    state_keys: frozenset[str] = field(default_factory=frozenset)
    state: StateSpec | None = None

    def __post_init__(self) -> None:
        # If a rich StateSpec is given, keep state_keys consistent with it.
        if self.state is not None:
            if not self.state_keys:
                object.__setattr__(self, "state_keys", self.state.keys)
            elif self.state_keys != self.state.keys:
                raise ValueError(
                    f"ObservationSpace state_keys {sorted(self.state_keys)} inconsistent "
                    f"with StateSpec keys {sorted(self.state.keys)}; provide one or make "
                    "them agree"
                )

    @property
    def camera_names(self) -> frozenset[str]:
        """Camera name set used during compatibility checks."""
        return frozenset(c.name for c in self.cameras)
