"""The canonical JSON eval-log sink.

Writes the immutable [`EvalLog`][inspect_robots.log.EvalLog] to ``log_dir`` once the run
finishes. The write is atomic (temp file + ``os.replace``) so an interrupted
overnight run never leaves a half-written log.

The file is strict RFC 8259 JSON: non-finite floats (``nan``, ``±inf``, e.g. a
``min_distance_to_goal`` score when no distance was ever recorded) are mapped
to ``null`` before serialization, so any conforming parser (``jq``, browsers,
non-Python tooling) can read the log. ``allow_nan=False`` is kept on the
``json.dump`` call as a regression backstop: if a non-finite value ever slips
past the sanitizer, writing fails loudly instead of emitting ``Infinity``/
``NaN`` literals.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog, EvalSpec
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.types import Action, Observation, StepResult

_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Leaves room for the filename's "_" + 8 hex chars + ".json.tmp" suffix inside a
# 255-byte name, with headroom for filesystems that allow less.
_SLUG_MAX = 200


def _slug(name: str) -> str:
    # Capped so the derived filename stays inside the 255-byte limit even for a
    # long task name: on_eval_end runs after every trial, so an ENAMETOOLONG
    # there would lose the whole finished run. The name itself is preserved in
    # the log body, and the filename's uuid suffix keeps distinct runs distinct
    # even when two long names truncate to the same slug.
    return _SLUG_RE.sub("-", name.lower()).strip("-")[:_SLUG_MAX].strip("-") or "eval"


def _sanitize(obj: object) -> object:
    """Recursively map non-finite floats to ``None`` (JSON ``null``).

    ``json.dump`` would happily emit the non-standard ``Infinity``/``NaN``
    literals for them (``default=`` never fires for floats), which RFC 8259
    parsers reject.
    """
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, dict):
        return {key: _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    return obj


class JsonLogSink:
    """Persist the final [`EvalLog`][inspect_robots.log.EvalLog] as JSON.

    Per-step data lives in the ``TrialRecord``/``FrameStore``, not here; this
    sink only writes the final log (``path`` holds where it landed).
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.path: Path | None = None

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Defer output until the final immutable log is available."""
        return None

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Ignore trial starts because the final log carries scene and epoch results."""
        return None

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Ignore live transitions because the final log is the canonical payload."""
        return None

    def on_trial_end(self, record: TrialRecord) -> None:
        """Ignore individual trajectories because their results arrive in the final log."""
        return None

    def on_eval_end(self, log: EvalLog) -> None:
        """Atomically serialize the final log and expose its path."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_slug(log.eval.task)}_{uuid.uuid4().hex[:8]}.json"
        self.path = self.log_dir / filename
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(_sanitize(log.to_dict()), fh, indent=2, sort_keys=True, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
