"""Optional Rerun visualization sink.

Logs camera images, proprioception, action vectors, and success markers to a
`Rerun <https://github.com/rerun-io/rerun>`_ recording. ``rerun-sdk`` is imported
lazily *inside* methods so the core package never depends on it; if it is not
installed, the sink warns once and becomes a no-op (so unattended runs and the
core-only import gate are unaffected).

Install with ``pip install "roboinspect[rerun]"``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from roboinspect.log import EvalLog, EvalSpec
    from roboinspect.rollout import TrialRecord
    from roboinspect.types import Action, Observation, StepResult

_WARNED = False


class RerunSink:
    """Stream a rollout to a Rerun recording (``.rrd``) or a live viewer."""

    def __init__(
        self,
        recording_path: str | None = None,
        *,
        application_id: str = "roboinspect",
        spawn: bool = False,
    ):
        self.recording_path = recording_path
        self.application_id = application_id
        self.spawn = spawn
        self._rr: Any | None = None
        self._t = 0

    def _ensure_rerun(self) -> Any | None:
        global _WARNED
        if self._rr is not None:
            return self._rr
        try:
            import rerun as rr
        except ImportError:
            if not _WARNED:
                warnings.warn(
                    "rerun-sdk is not installed; RerunSink is a no-op. "
                    'Install with: pip install "roboinspect[rerun]"',
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED = True
            return None
        self._rr = rr
        return rr

    @property
    def available(self) -> bool:
        return self._ensure_rerun() is not None

    def on_eval_start(self, spec: EvalSpec) -> None:
        rr = self._ensure_rerun()
        if rr is None:
            return
        rr.init(self.application_id, spawn=self.spawn)
        if self.recording_path is not None:
            rr.save(self.recording_path)

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        self._t = 0

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        rr = self._ensure_rerun()
        if rr is None:
            return
        rr.set_time_sequence("step", t)
        for cam, image in observation.images.items():
            rr.log(f"camera/{cam}", rr.Image(image))
        for key, value in observation.state.items():
            for i, scalar in enumerate(np.atleast_1d(np.asarray(value, dtype=np.float64))):
                rr.log(f"state/{key}/{i}", rr.Scalar(float(scalar)))
        for i, scalar in enumerate(np.atleast_1d(np.asarray(action.data, dtype=np.float64))):
            rr.log(f"action/{i}", rr.Scalar(float(scalar)))
        if result.reward is not None:
            rr.log("reward", rr.Scalar(float(result.reward)))
        if result.terminated:
            rr.log("event/terminated", rr.TextLog(result.termination_reason or "terminated"))

    def on_trial_end(self, record: TrialRecord) -> None:
        return None

    def on_eval_end(self, log: EvalLog) -> None:
        return None
