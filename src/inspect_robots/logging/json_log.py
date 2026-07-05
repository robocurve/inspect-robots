"""The canonical JSON eval-log sink.

Writes the immutable [`EvalLog`][inspect_robots.log.EvalLog] to ``log_dir`` once the run
finishes. The write is atomic (temp file + ``os.replace``) so an interrupted
overnight run never leaves a half-written log.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog, EvalSpec
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.types import Action, Observation, StepResult

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "eval"


class JsonLogSink:
    """Persist the final [`EvalLog`][inspect_robots.log.EvalLog] as JSON.

    Per-step data lives in the ``TrialRecord``/``FrameStore``, not here; this
    sink only writes the final log (``path`` holds where it landed).
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.path: Path | None = None

    def on_eval_start(self, spec: EvalSpec) -> None:
        return None

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        return None

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        return None

    def on_trial_end(self, record: TrialRecord) -> None:
        return None

    def on_eval_end(self, log: EvalLog) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_slug(log.eval.task)}_{uuid.uuid4().hex[:8]}.json"
        self.path = self.log_dir / filename
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(log.to_dict(), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
