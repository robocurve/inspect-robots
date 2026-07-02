"""CubePick — a deterministic 2D toy world for exercising the full stack.

A point end-effector in the unit square must reach a cube. The action is a 2D
end-effector position delta. Success is declared (and exposed as privileged
``info["success"]``) when the effector is within ``goal_radius`` of the cube.
Fully deterministic given a seed; no third-party dependencies.
"""

from __future__ import annotations

import numpy as np

from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace
from inspect_robots.types import Action, Observation, StepResult

_IMG = 32  # rendered camera resolution (square)


class CubePickEmbodiment:
    """A 2D reach-the-cube simulator."""

    def __init__(
        self,
        *,
        max_step: float = 0.1,
        goal_radius: float = 0.05,
        start: tuple[float, float] = (0.1, 0.1),
    ):
        from inspect_robots.embodiment import (
            AUTO_RESET,
            PRIVILEGED_SUCCESS,
            RENDERABLE,
            RESETTABLE,
            SEEDABLE,
            EmbodimentInfo,
        )

        self.max_step = max_step
        self.goal_radius = goal_radius
        self.start = np.asarray(start, dtype=np.float64)
        self.num_steps = 0

        self._eef = self.start.copy()
        self._cube = np.array([0.8, 0.8], dtype=np.float64)

        self.info = EmbodimentInfo(
            name="cubepick",
            action_space=Box(
                shape=(2,),
                low=np.array([-max_step, -max_step]),
                high=np.array([max_step, max_step]),
                semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="top", height=_IMG, width=_IMG, channels=3),),
                state_keys=frozenset({"eef_pos", "cube_pos"}),
            ),
            control_hz=10.0,
            is_simulated=True,
            capabilities=frozenset(
                {SEEDABLE, RESETTABLE, AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE}
            ),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        rng = np.random.RandomState(seed if seed is not None else 0)
        # Place the cube reachably in the far quadrant; deterministic per seed.
        self._cube = rng.uniform(0.6, 0.9, size=2)
        self._eef = self.start.copy()
        self.num_steps = 0
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        self.num_steps += 1
        delta = np.clip(np.asarray(action.data, dtype=np.float64), -self.max_step, self.max_step)
        self._eef = np.clip(self._eef + delta, 0.0, 1.0)
        dist = float(np.linalg.norm(self._eef - self._cube))
        success = dist <= self.goal_radius
        return StepResult(
            observation=self._observe(None),
            reward=-dist,
            terminated=success,
            termination_reason="success" if success else None,
            truncated=False,
            info={"success": success, "distance": dist},
        )

    def close(self) -> None:
        return None

    def _observe(self, instruction: str | None) -> Observation:
        return Observation(
            images={"top": self._render()},
            state={
                "eef_pos": self._eef.astype(np.float64),
                "cube_pos": self._cube.astype(np.float64),
            },
            instruction=instruction,
        )

    def _render(self) -> np.ndarray:
        img = np.zeros((_IMG, _IMG, 3), dtype=np.uint8)
        cy, cx = (np.clip(self._cube, 0, 1) * (_IMG - 1)).astype(int)
        ey, ex = (np.clip(self._eef, 0, 1) * (_IMG - 1)).astype(int)
        img[cx, cy] = (0, 200, 0)  # cube = green
        img[ex, ey] = (200, 0, 0)  # effector = red
        return img
