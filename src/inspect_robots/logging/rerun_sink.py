"""Optional Rerun visualization sink.

Logs camera images, proprioception, action vectors, and success markers to a
`Rerun <https://github.com/rerun-io/rerun>`_ recording. ``rerun-sdk`` is imported
lazily *inside* methods so the core package never depends on it; if it is not
installed, the sink warns once and becomes a no-op (so unattended runs and the
core-only import gate are unaffected).

Each trial's entities are namespaced under ``trial/<scene_id>/e<epoch>`` so
successive trials never overwrite one another on the shared step timeline.

Install with ``pip install "inspect-robots[rerun]"``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog, EvalSpec
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.types import Action, Observation, StepResult


class RerunSink:
    """Stream a rollout to a Rerun recording (``.rrd``) or a live viewer."""

    def __init__(
        self,
        recording_path: str | None = None,
        *,
        application_id: str = "inspect_robots",
        spawn: bool = False,
    ):
        self.recording_path = recording_path
        self.application_id = application_id
        self.spawn = spawn
        self._rr: Any | None = None
        self._warned = False
        self._prefix = "trial"

    def _ensure_rerun(self) -> Any | None:
        if self._rr is not None:
            return self._rr
        try:
            import rerun as rr
        except ImportError:
            if not self._warned:
                warnings.warn(
                    "rerun-sdk is not installed; RerunSink is a no-op. "
                    'Install with: pip install "inspect-robots[rerun]"',
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return None
        self._rr = rr
        return rr

    @property
    def available(self) -> bool:
        return self._ensure_rerun() is not None

    @staticmethod
    def _set_step(rr: Any, t: int) -> None:
        if hasattr(rr, "set_time"):  # rerun-sdk >= 0.23
            rr.set_time("step", sequence=t)
        else:  # older SDKs
            rr.set_time_sequence("step", t)

    @staticmethod
    def _scalar(rr: Any, value: float) -> Any:
        scalars = getattr(rr, "Scalars", None)  # rerun-sdk >= 0.23
        if scalars is not None:
            return scalars(value)
        return rr.Scalar(value)  # older SDKs

    def on_eval_start(self, spec: EvalSpec) -> None:
        rr = self._ensure_rerun()
        if rr is None:
            return
        rr.init(self.application_id, spawn=self.spawn)
        if self.recording_path is not None:
            rr.save(self.recording_path)

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        # Namespace this trial's entities so trials never overwrite each other.
        self._prefix = f"trial/{scene_id}/e{epoch}"

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        rr = self._ensure_rerun()
        if rr is None:
            return
        self._set_step(rr, t)
        pre = self._prefix
        for cam, image in observation.images.items():
            rr.log(f"{pre}/camera/{cam}", rr.Image(image))
        for key, value in observation.state.items():
            for i, scalar in enumerate(np.atleast_1d(np.asarray(value, dtype=np.float64))):
                rr.log(f"{pre}/state/{key}/{i}", self._scalar(rr, float(scalar)))
        for i, scalar in enumerate(np.atleast_1d(np.asarray(action.data, dtype=np.float64))):
            rr.log(f"{pre}/action/{i}", self._scalar(rr, float(scalar)))
        if result.reward is not None:
            rr.log(f"{pre}/reward", self._scalar(rr, float(result.reward)))
        if result.terminated:
            rr.log(
                f"{pre}/event/terminated",
                rr.TextLog(result.termination_reason or "terminated"),
            )

    def on_trial_end(self, record: TrialRecord) -> None:
        return None

    def on_eval_end(self, log: EvalLog) -> None:
        return None
