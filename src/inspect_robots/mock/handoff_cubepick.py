"""CubePick for handoff trials: a world an outside process can command directly.

The released world is one JSON file, ``cubepick_state.json``, standing in for a
robot's bus: while the embodiment has released it, whoever writes that file
commands the robot. [`CubePickDriver`][inspect_robots.mock.handoff_cubepick.CubePickDriver]
is the motor-level interface over it (bounded moves, the touch sensor), the
same code the embodiment itself would run.

The task this world supports is "touch the cube": touching leaves a mark that
outlasts parking, as moved objects do on a real robot, so a trial can be graded
after the effector has gone back to its park position.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from inspect_robots.mock.cubepick import CubePickEmbodiment
from inspect_robots.rollout import Record
from inspect_robots.scene import Scene
from inspect_robots.transcript import operator_event
from inspect_robots.types import Observation

STATE_FILE = "cubepick_state.json"
#: How close to the park position counts as parked.
PARK_TOLERANCE = 1e-6


class CubePickDriver:
    """Direct control of a released CubePick world through its state file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def state(self) -> dict[str, Any]:
        """The world as it is now: ``eef``, ``cube``, ``park``, ``touched`` and limits."""
        state: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return state

    def _write(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.path)

    def move(self, dx: float, dy: float) -> dict[str, Any]:
        """Move the effector by one bounded delta and return the new state.

        Each component is limited to ``max_step`` and the effector stays in the
        unit square. Coming within ``goal_radius`` of the cube touches it.
        """
        state = self.state()
        limit = float(state["max_step"])
        delta = np.clip(np.array([dx, dy], dtype=np.float64), -limit, limit)
        eef = np.clip(np.asarray(state["eef"], dtype=np.float64) + delta, 0.0, 1.0)
        cube = np.asarray(state["cube"], dtype=np.float64)
        state["eef"] = [float(eef[0]), float(eef[1])]
        if float(np.linalg.norm(eef - cube)) <= float(state["goal_radius"]):
            state["touched"] = True
        self._write(state)
        return state

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        """Move in bounded steps until the effector is at ``(x, y)`` or cannot get closer."""
        state = self.state()
        while True:
            eef = state["eef"]
            dx, dy = x - eef[0], y - eef[1]
            if abs(dx) <= PARK_TOLERANCE and abs(dy) <= PARK_TOLERANCE:
                return state
            state = self.move(dx, dy)
            if state["eef"] == eef:
                return state

    def park(self) -> dict[str, Any]:
        """Move the effector to its park position."""
        park = self.state()["park"]
        return self.move_to(park[0], park[1])


class HandoffCubePick(CubePickEmbodiment):
    """CubePick with the ``release``/``retake`` hooks a handoff trial needs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._touched = False
        self._driver: CubePickDriver | None = None
        self.info = replace(
            self.info,
            name="cubepick-handoff",
            docs=(
                "The red point is the end effector and the cube is green in the top "
                "camera, blue once touched; +x points right and +y points down in "
                "the image, and both positions stay within the unit square."
            ),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Place a seeded, untouched cube and put the effector at its park position."""
        self._touched = False
        self._driver = None
        return super().reset(scene, seed=seed)

    def release(self, workdir: str) -> str:
        """Write the world to its state file and return how to command it."""
        path = Path(workdir) / STATE_FILE
        driver = CubePickDriver(path)
        driver._write(
            {
                "eef": [float(v) for v in self._eef],
                "cube": [float(v) for v in self._cube],
                "park": [float(v) for v in self.start],
                "touched": self._touched,
                "max_step": self.max_step,
                "goal_radius": self.goal_radius,
            }
        )
        self._driver = driver
        return (
            f"The robot is a point effector in a 2D unit square. Its bus is the file "
            f"`{path}`; whatever writes that file commands the robot.\n\n"
            f"A driver is installed for `{sys.executable}`:\n\n"
            "```python\n"
            "from inspect_robots.mock.handoff_cubepick import CubePickDriver\n"
            f'robot = CubePickDriver("{path}")\n'
            "robot.state()        # eef, cube, park, touched, max_step, goal_radius\n"
            "robot.move(dx, dy)   # one move, each component limited to max_step\n"
            "robot.move_to(x, y)  # repeated moves to a position\n"
            "robot.park()         # back to the park position\n"
            "```\n\n"
            "The cube is touched once the effector comes within `goal_radius` of it, "
            "and stays touched."
        )

    def retake(self) -> Observation:
        """Read the world back from its state file and return what it looks like now."""
        if self._driver is None:
            raise RuntimeError("retake() without a release()")
        state = self._driver.state()
        self._driver = None
        self._eef = np.asarray(state["eef"], dtype=np.float64)
        self._touched = bool(state["touched"])
        return self._observe(None)

    def is_parked(self) -> bool:
        """Whether the effector is at its park position."""
        return bool(np.all(np.abs(self._eef - self.start) <= PARK_TOLERANCE))

    def park(self) -> None:
        """Put the effector at its park position."""
        self._eef = self.start.copy()

    def _observe(self, instruction: str | None) -> Observation:
        observation = super()._observe(instruction)
        return replace(
            observation,
            state={**observation.state, "cube_touched": np.array([float(self._touched)])},
        )

    def _render(self) -> np.ndarray:
        img = super()._render()
        if self._touched:
            img[(img == (0, 200, 0)).all(axis=-1)] = (0, 0, 200)
        return img


class CubeTouchedGrader:
    """Judge a CubePick trial from the final observation: was the cube touched?"""

    name = "cubepick-touched"

    def grade(self, record: Record, scene: Scene) -> None:
        """Record ``success`` or ``failure``; leave the trial unjudged without a final state."""
        final = record.parked_observation or record.unstepped_observation
        if final is None or "cube_touched" not in final.state:
            return
        verdict = "success" if float(final.state["cube_touched"][0]) > 0 else "failure"
        record.operator_judgement = verdict
        record.events.append(operator_event(t=len(record.steps), verdict=verdict, source="state"))
