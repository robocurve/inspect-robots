"""Mock policies for the CubePick world.

- [`ScriptedPolicy`][roboinspect.mock.policies.ScriptedPolicy] — a deterministic oracle that
walks the effector to the
  cube. It predicts a full action *chunk* by simulating its own future motion, so
  the chunk is a genuine open-loop trajectory (``H > 1``).
- [`RandomPolicy`][roboinspect.mock.policies.RandomPolicy] — emits random deltas; mostly fails.
- [`NoopPolicy`][roboinspect.mock.policies.NoopPolicy] — emits zero actions; never succeeds.
"""

from __future__ import annotations

import numpy as np

from roboinspect.policy import PolicyConfig, PolicyInfo
from roboinspect.scene import Scene
from roboinspect.spaces import ActionSemantics, Box, ObservationSpace
from roboinspect.types import Action, ActionChunk, Observation

_ACTION_SPACE = Box(
    shape=(2,),
    semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
)
# The scripted oracle reads the effector and cube positions from proprioception.
_SCRIPTED_OBS = ObservationSpace(state_keys=frozenset({"eef_pos", "cube_pos"}))


class ScriptedPolicy:
    """Deterministic oracle: walk straight to the cube, in chunks."""

    def __init__(self, *, chunk_size: int = 4, max_step: float = 0.1):
        self.chunk_size = chunk_size
        self.max_step = max_step
        self.num_inferences = 0
        self.info = PolicyInfo(
            name="scripted", action_space=_ACTION_SPACE, observation_space=_SCRIPTED_OBS
        )
        self.config = PolicyConfig(action_horizon=chunk_size)

    def reset(self, scene: Scene) -> None:
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        eef = np.asarray(observation.state["eef_pos"], dtype=np.float64).copy()
        cube = np.asarray(observation.state["cube_pos"], dtype=np.float64)
        actions: list[Action] = []
        # Simulate our own motion forward to build a coherent open-loop trajectory.
        for _ in range(self.chunk_size):
            delta = cube - eef
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                step = np.zeros(2, dtype=np.float64)
            else:
                step = delta / dist * min(self.max_step, dist)
            actions.append(Action(data=step))
            eef = eef + step
        return ActionChunk(actions=actions)


class RandomPolicy:
    """Emit random small deltas. Deterministic given the construction seed."""

    def __init__(self, *, chunk_size: int = 4, max_step: float = 0.1, seed: int = 0):
        self.chunk_size = chunk_size
        self.max_step = max_step
        self.num_inferences = 0
        self._rng = np.random.RandomState(seed)
        self._base_seed = seed
        self._reset_count = 0
        self.info = PolicyInfo(name="random", action_space=_ACTION_SPACE)
        self.config = PolicyConfig(action_horizon=chunk_size)

    def reset(self, scene: Scene) -> None:
        # Re-seed per scene for reproducibility while still varying across scenes.
        self._rng = np.random.RandomState(self._base_seed + self._reset_count)
        self._reset_count += 1
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        actions = [
            Action(data=self._rng.uniform(-self.max_step, self.max_step, size=2))
            for _ in range(self.chunk_size)
        ]
        return ActionChunk(actions=actions)


class NoopPolicy:
    """Emit zero actions; never moves."""

    def __init__(self, *, chunk_size: int = 1):
        self.chunk_size = chunk_size
        self.num_inferences = 0
        self.info = PolicyInfo(name="noop", action_space=_ACTION_SPACE)
        self.config = PolicyConfig(action_horizon=chunk_size)

    def reset(self, scene: Scene) -> None:
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        return ActionChunk(actions=[Action(data=np.zeros(2)) for _ in range(self.chunk_size)])
