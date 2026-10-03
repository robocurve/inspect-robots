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
from urllib.parse import unquote_to_bytes

import numpy as np
import numpy.typing as npt

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FRAME_NAME_RE = re.compile(r"~f1~([^~]*)~([^~]*)_(-?[0-9]+)\.npy")


def _encode_component(name: str) -> str:
    """Encode UTF-8 without delimiter or case-insensitive filesystem aliases."""
    return "".join(
        chr(byte) if byte in b"abcdefghijklmnopqrstuvwxyz0123456789._-" else f"%{byte:02X}"
        for byte in name.encode("utf-8")
    )


def _frame_filename(trial_id: str, camera: str, t: int) -> str:
    """Map the complete frame identity to a reversible, flat, versioned name."""
    return f"~f1~{_encode_component(trial_id)}~{_encode_component(camera)}_{t:06d}.npy"


def _parse_frame_filename(name: str) -> tuple[str, str, int] | None:
    """Decode only canonical versioned filenames; never guess legacy boundaries."""
    match = _FRAME_NAME_RE.fullmatch(name)
    if match is None:
        return None
    try:
        trial_id, camera = (unquote_to_bytes(part).decode("utf-8") for part in match.group(1, 2))
        t = int(match.group(3))
    except ValueError:
        return None
    if _frame_filename(trial_id, camera, t) != name:
        return None
    return trial_id, camera, t


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
        """Load the referenced array from disk as ``uint8``."""
        return np.asarray(np.load(self.path), dtype=np.uint8)


class FrameStore:
    """Persist frames as ``.npy`` files under ``root`` and hand back refs."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def put(self, trial_id: str, t: int, camera: str, image: npt.NDArray[np.uint8]) -> FrameRef:
        """Persist one camera frame and return its lightweight reference."""
        path = self.root / _frame_filename(trial_id, camera, t)
        np.save(path, image)
        self.count += 1
        return FrameRef(camera=camera, t=t, path=str(path))
