"""In-process execution boundary for model-generated Python code.

Model output executes with the evaluator's process privileges. This module is
an integration surface, not a security sandbox; untrusted models require an
external container or equivalent isolation.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.types import Observation
from inspect_robots_capx._motion import MotionQueue
from inspect_robots_capx._servers import CapxServerClients


@dataclass(frozen=True)
class ExecutionResult:
    """Captured feedback from one model-code turn without propagating its exception."""

    stdout: str
    stderr: str
    raised: bool


class _TurnView(dict[str, Any]):
    """Turn-scoped ``obs`` mapping that resolves zero-arg callable values on access.

    Embodiments may provide bulk ``observation.extra`` entries (depth,
    intrinsics, extrinsics) as zero-argument callables to keep trial records
    small; resolving them here keeps the documented ``obs[...]`` idioms
    working unchanged for both forms.
    """

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        return value() if callable(value) else value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the resolved value for ``key``, or ``default`` when absent.

        Membership decides absence, so a ``KeyError`` raised inside an
        embodiment-provided thunk propagates instead of masquerading as a
        missing key.
        """
        if key not in self:
            return default
        return self[key]


class CodeSandbox:
    """Persistent per-trial Python namespace with observation-bound robot helpers."""

    def __init__(
        self,
        *,
        servers: CapxServerClients,
        motion: MotionQueue,
        camera: str,
        state_key: str,
        depth_key: str = "depth",
        intrinsics_key: str = "intrinsics",
        extrinsics_key: str = "extrinsics",
    ) -> None:
        self._servers = servers
        self._motion = motion
        self._camera = camera
        self._state_key = state_key
        self._depth_key = depth_key
        self._intrinsics_key = intrinsics_key
        self._extrinsics_key = extrinsics_key
        self._observation: Observation | None = None
        self._namespace: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Start a fresh trial namespace with only the documented helpers bound."""
        self._observation = None
        self._motion.reset()
        self._namespace = {
            "__builtins__": __builtins__,
            "segment": self._segment,
            "plan_grasp": self._plan_grasp,
            "solve_ik": self._solve_ik,
            "move_to_joints": self._motion.move_to_joints,
            "open_gripper": self._motion.open_gripper,
            "close_gripper": self._motion.close_gripper,
        }

    def set_observation(self, observation: Observation) -> None:
        """Expose one turn's observation and reseed motion from its full state field."""
        try:
            state = observation.state[self._state_key]
        except KeyError as exc:
            raise ValueError(
                f"observation.state is missing bound proprioceptive key {self._state_key!r}"
            ) from exc
        self._observation = observation
        self._motion.begin_turn(state)
        obs = _TurnView(
            {
                "images": observation.images,
                "state": observation.state,
            }
        )
        obs.update(observation.extra)
        self._namespace["obs"] = obs

    def execute(self, code: str) -> ExecutionResult:
        """Execute one turn and return stdout, stderr, and whether it raised."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        raised = False
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(code, self._namespace, self._namespace)
            except Exception:
                raised = True
                traceback.print_exc()
        return ExecutionResult(stdout=stdout.getvalue(), stderr=stderr.getvalue(), raised=raised)

    def _require_observation(self) -> Observation:
        if self._observation is None:
            raise RuntimeError("no observation is bound for this code turn")
        return self._observation

    def _extra_array(self, key: str) -> npt.NDArray[Any]:
        observation = self._require_observation()
        if key not in observation.extra:
            raise KeyError(
                f"observation.extra is missing {key!r}; embodiments must provide "
                f"extra[{key!r}] as an array or a zero-argument callable returning one"
            )
        value = observation.extra[key]
        if callable(value):
            value = value()
        return np.asarray(value)

    def _segment(self, text: str) -> list[dict[str, Any]]:
        observation = self._require_observation()
        try:
            image = observation.images[self._camera]
        except KeyError as exc:
            raise KeyError(f"observation has no configured camera {self._camera!r}") from exc
        return self._servers.sam3.segment(image, text)

    def _plan_grasp(self, mask: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        return self._servers.graspnet.plan(
            self._extra_array(self._depth_key),
            self._extra_array(self._intrinsics_key),
            np.asarray(mask),
        )

    def _solve_ik(
        self,
        position: npt.NDArray[Any],
        quaternion_wxyz: npt.NDArray[Any],
    ) -> npt.NDArray[np.float64]:
        return self._servers.pyroki.solve_ik(
            position,
            quaternion_wxyz,
            self._motion.arm_dof,
        )
