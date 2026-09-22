"""A toy Cartesian rig for end-to-end tests: two arms, a cube, a bowl, a top camera.

No physics: the gripper teleports to each commanded pose, closes to 0.3 when
it grips within 1.5 cm of the cube, the cube rides with a closed gripper, and
the "camera" is a fake detector that reports tags at the true object poses.
``step()`` never terminates, so a trial can only end through the policy's
``request_stop`` or the task horizon.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Action, Observation, StepResult

LABELS = tuple(
    f"{arm}_{p}"
    for arm in ("left", "right")
    for p in ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
)
ARM_LOW = (0.15, -0.25, 0.0, -math.pi, 0.0, 0.0, 0.0)
ARM_HIGH = (0.48, 0.25, 0.40, math.pi, 0.0, 0.0, 1.0)
BOX = Box(
    low=np.array(ARM_LOW * 2),
    high=np.array(ARM_HIGH * 2),
    shape=(14,),
    semantics=ActionSemantics(
        control_mode="eef_abs_pose",
        gripper="continuous",
        frame="base",
        dim_labels=LABELS,
        max_step=(0.02, 0.02, 0.02, None, None, None, 0.25) * 2,
    ),
)
#: Right arm base sits 0.5 m to the left (+y) of the left arm base; camera frame == left frame.
RIGHT_OFFSET = np.array([0.0, -0.5, 0.0])
K = np.array([[900.0, 0, 320], [0, 900.0, 240], [0, 0, 1]], dtype=np.float32)
CUBE_HALF = 0.0127
GRIP_RADIUS = 0.015


class Det:
    def __init__(self, tag_id: int, centre_cam: npt.NDArray[np.float64], offset_z: float) -> None:
        self.tag_id = tag_id
        self.pose_R = np.eye(3)
        # the detector reports the TAG pose; the object centre is offset_z further along tag z
        self.pose_t = (centre_cam - np.array([0.0, 0.0, offset_z])).reshape(3, 1)
        self.center = np.array([10.0, 10.0])


class FakeRig:
    """Minimal ``Embodiment`` for the jev end-to-end test."""

    def __init__(self, *, occlude: bool = False, floor_z: float | None = None) -> None:
        # occlude: a top-down camera cannot see a tag under the gripper. The cube's
        # tag is hidden when the gripper is within 2 cm xy and less than 6 cm above
        # it; the bowl's tag while the gripper is within 4 cm xy and under 10 cm.
        # floor_z: over the bowl, the fingertips (or the carried cube) stop at this
        # height: a bowl shallower than the phase machine's bowl_depth_m, so a
        # commanded descent never reaches its computed goal.
        self.occlude = occlude
        self.floor_z = floor_z
        self.info = EmbodimentInfo(
            name="fake-rig",
            action_space=BOX,
            observation_space=ObservationSpace(),
            control_hz=10.0,
        )
        self.cube = np.array([0.30, 0.10, CUBE_HALF])
        self.bowl = np.array([0.35, -0.10, 0.0])
        self.eef = np.zeros(14)
        self.commanded: list[np.ndarray] = []
        self.holding = False

    def reset(self, scene: Scene, seed: int | None = None) -> Observation:
        self.eef = np.zeros(14)
        self.eef[0:3] = (0.20, 0.0, 0.20)
        self.eef[6] = 1.0
        self.eef[7:10] = (0.20, 0.0, 0.20)
        self.eef[13] = 1.0
        self.cube = np.array([0.30, 0.10, CUBE_HALF])
        self.holding = False
        self.commanded = []
        return self.observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        target = np.clip(np.asarray(action.data, dtype=np.float64), BOX.low, BOX.high)
        self.commanded.append(target.copy())
        prev_open = self.eef[6]
        self.eef = target.copy()
        if self.floor_z is not None and np.hypot(*(self.eef[:2] - self.bowl[:2])) <= 0.04:
            self.eef[2] = max(self.eef[2], self.floor_z)  # the arm stops where it touches
        grip = self.eef[0:3]
        if self.holding:
            self.cube = grip.copy()
            self.eef[6] = max(self.eef[6], 0.3)  # the jaws stop on the cube
            if self.eef[6] >= 0.8:
                self.holding = False
                self.cube[2] = (
                    max(self.bowl[2] + CUBE_HALF, 0.0) if self._over_bowl() else CUBE_HALF
                )
        elif self.eef[6] < prev_open and np.linalg.norm(grip - self.cube) <= GRIP_RADIUS:
            self.holding = True
            self.eef[6] = 0.3  # jaws stop on the cube
        elif self.eef[6] < 0.8 and not self.holding and self.eef[6] < prev_open:
            self.eef[6] = 0.0  # closed on air
        return StepResult(observation=self.observe(None), terminated=False)

    def _over_bowl(self) -> bool:
        return bool(np.hypot(*(self.cube[:2] - self.bowl[:2])) <= 0.04)

    def _hidden(self, obj: np.ndarray, xy_m: float, above_m: float) -> bool:
        if not self.occlude:
            return False
        grip = self.eef[0:3]
        return bool(np.hypot(*(grip[:2] - obj[:2])) <= xy_m and 0.0 <= grip[2] - obj[2] < above_m)

    def detector(self, gray: Any, params: Any, size: float) -> list[Det]:
        dets: list[Det] = []
        if abs(size - 0.02) < 1e-9 and not self.holding and not self._hidden(self.cube, 0.02, 0.06):
            dets.append(Det(0, self.cube.copy(), CUBE_HALF))  # top face tag
        if abs(size - 0.04) < 1e-9 and not self._hidden(self.bowl, 0.04, 0.10):
            dets.append(Det(14, self.bowl.copy(), 0.0))
        if abs(size - 0.08) < 1e-9:
            dets.append(Det(20, np.array([0.30, 0.20, 0.0]), 0.0))
        return dets

    def observe(self, instruction: str | None) -> Observation:
        return Observation(
            images={"top_cam": np.zeros((32, 32, 3), dtype=np.uint8)},
            state={"eef_state": self.eef.copy()},
            instruction=instruction,
            extra={"top_cam_intrinsics": K},
        )

    def close(self) -> None:
        return None
