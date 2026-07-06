"""FrameStore — rollout-owned streaming of camera frames to disk (R5).

A long multi-camera episode would exhaust memory if every frame were retained in
the [`TrialRecord`][inspect_robots.rollout.TrialRecord]. Instead the rollout streams frames to
disk through a [`FrameStore`][inspect_robots.frames.FrameStore] and keeps only lightweight
[`FrameRef`][inspect_robots.frames.FrameRef]
handles. This is owned by the rollout, NOT by any log sink, so trajectories are
recorded (and scorable) independent of which optional sinks are enabled.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str) -> str:
    """Make ``name`` filesystem-safe without introducing collisions.

    Unsafe characters become ``-``; when anything was replaced, a short hash of
    the original is appended so e.g. ``a/b`` and ``a-b`` stay distinct.
    """
    safe = _SAFE_RE.sub("-", name)
    if safe != name:
        safe = f"{safe}-{zlib.crc32(name.encode()) & 0xFFFFFFFF:08x}"
    return safe


@dataclass(frozen=True)
class FrameRef:
    """A handle to a camera frame stored on disk."""

    camera: str
    t: int
    path: str

    def load(self) -> npt.NDArray[np.uint8]:
        return np.asarray(np.load(self.path), dtype=np.uint8)


class FrameStore:
    """Persist frames as ``.npy`` files under ``root`` and hand back refs."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def put(self, trial_id: str, t: int, camera: str, image: npt.NDArray[np.uint8]) -> FrameRef:
        path = self.root / f"{_safe(trial_id)}_{_safe(camera)}_{t:06d}.npy"
        np.save(path, image)
        self.count += 1
        return FrameRef(camera=camera, t=t, path=str(path))
