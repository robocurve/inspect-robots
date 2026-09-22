"""AprilTag detections in the top camera -> ``WorldState`` in each arm's frame.

Tag frame convention (AprilTag 3, as ``pupil-apriltags`` returns it): x to
the tag's right, y down the tag, z into the tag away from the camera, origin
at the tag centre, ``pose_t`` in metres given ``camera_params`` and the black
square's edge as ``tag_size``. An object's centre is ``offset_m`` in that
frame, so a face tag on a 25.4 mm cube carries ``(0, 0, +0.0127)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots.types import Observation
from inspect_robots_jev.calibration import Calibration, pose_to_matrix, transform
from inspect_robots_jev.world import ARMS, Arm, GripperView, ObjectView, Vec3, WorldState


@dataclass(frozen=True)
class TagSpec:
    """One printed tag: its object, black-square edge, and the object centre in tag frame."""

    tag_id: int
    object: str
    size_m: float
    offset_m: Vec3


@dataclass(frozen=True)
class TagLayout:
    """Every tag on the table, keyed by id, plus which one is the calibration reference."""

    tags: Mapping[int, TagSpec]
    table_tag_id: int

    @classmethod
    def load(cls, path: Path) -> TagLayout:
        """Read ``{"table_tag_id": 20, "tags": [{"id", "object", "size_m", "offset_m"}, ...]}``."""
        raw = json.loads(Path(path).read_text())
        tags: dict[int, TagSpec] = {}
        for entry in raw["tags"]:
            off = entry.get("offset_m", [0.0, 0.0, 0.0])
            tags[int(entry["id"])] = TagSpec(
                tag_id=int(entry["id"]),
                object=str(entry["object"]),
                size_m=float(entry["size_m"]),
                offset_m=(float(off[0]), float(off[1]), float(off[2])),
            )
        table = int(raw["table_tag_id"])
        if table not in tags:
            raise ValueError(f"table_tag_id {table} is not listed under tags")
        return cls(tags=tags, table_tag_id=table)

    def sizes(self) -> dict[float, list[int]]:
        """Group tag ids by black-square size (the detector takes one size per call)."""
        out: dict[float, list[int]] = {}
        for spec in self.tags.values():
            out.setdefault(spec.size_m, []).append(spec.tag_id)
        return out


class Detection(Protocol):
    """The fields of ``pupil_apriltags.Detection`` the perceiver reads."""

    tag_id: int
    center: Any
    pose_R: Any
    pose_t: Any


#: (gray uint8 HxW, (fx, fy, cx, cy), tag_size_m) -> detections with pose.
Detector = Callable[
    [npt.NDArray[np.uint8], tuple[float, float, float, float], float], Sequence[Any]
]


def apriltag_detector(**kwargs: Any) -> Detector:
    """Build one real ``pupil_apriltags`` 36h11 detector and return it as a ``Detector``.

    ``quad_decimate`` defaults to 1.0 (full resolution) because the cube tags
    are only a few dozen pixels wide; pass a larger value to trade accuracy for
    speed on big tags.
    """
    kwargs.setdefault("quad_decimate", 1.0)
    try:
        import pupil_apriltags
    except ImportError as err:  # pragma: no cover - exercised via monkeypatched sys.modules
        raise ImportError(
            "pupil-apriltags is required for tag perception: pip install pupil-apriltags"
        ) from err
    detector = pupil_apriltags.Detector(families="tag36h11", **kwargs)

    def detect(
        gray: npt.NDArray[np.uint8], camera_params: tuple[float, float, float, float], size: float
    ) -> Sequence[Any]:
        result: Sequence[Any] = detector.detect(
            gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=size
        )
        return result

    return detect


def to_gray(image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Collapse HxWx3 to the C-contiguous 2-D uint8 array the detector requires."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    out: npt.NDArray[np.uint8] = np.ascontiguousarray(arr.astype(np.uint8))
    return out


#: Depth arrays are metres (YAM serves float32 metres); anything beyond this is a
#: unit mistake (e.g. raw millimetres) and is ignored rather than scaling objects 1000x.
_MAX_DEPTH_M = 10.0


def _depth_at(depth: Any, centre: Any, window: int) -> float | None:
    arr = np.asarray(depth, dtype=np.float64)
    cx, cy = round(float(centre[0])), round(float(centre[1]))
    half = window // 2
    patch = arr[max(0, cy - half) : cy + half + 1, max(0, cx - half) : cx + half + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class Perceiver:
    """Keep a per-object memory of tag poses and express them in every arm's frame."""

    def __init__(
        self,
        *,
        layout: TagLayout,
        camera: str,
        calibration: Calibration,
        detector: Detector | None = None,
        depth_window_px: int = 5,
        grasp_offset_m: Vec3 = (0.0, 0.0, 0.0),
    ) -> None:
        self._layout = layout
        self._camera = camera
        self._cal = calibration
        self._detector = detector
        self._window = depth_window_px
        self._grasp_offset = grasp_offset_m
        self._memory: dict[str, ObjectView] = {}

    def reset(self) -> None:
        """Forget every remembered object pose (call at trial start)."""
        self._memory = {}

    def _detect_all(self, observation: Observation) -> Sequence[Any]:
        if self._detector is None:
            self._detector = apriltag_detector()
        gray = to_gray(observation.images[self._camera])
        K = np.asarray(observation.extra[f"{self._camera}_intrinsics"], dtype=np.float64)
        params = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        found: list[Any] = []
        for size, ids in self._layout.sizes().items():
            found.extend(d for d in self._detector(gray, params, size) if d.tag_id in ids)
        return found

    def update(
        self,
        observation: Observation,
        step: int,
        *,
        grippers: Mapping[Arm, GripperView],
        held: tuple[Arm, str] | None = None,
    ) -> WorldState:
        """Detect tags, refresh remembered objects, and return this step's world.

        ``step`` is the policy's decision counter (one per ``act``), which is
        what ``ObjectView.last_seen_step`` and every stale budget count in.
        """
        depth = observation.extra.get(f"{self._camera}_depth")
        if callable(depth):
            depth = depth()  # resolve the YAM lazy thunk once per update, not per detection
        sums: dict[str, dict[Arm, np.ndarray]] = {}
        counts: dict[str, int] = {}
        for det in self._detect_all(observation):
            spec = self._layout.tags[int(det.tag_id)]
            t = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
            if depth is not None:
                z = _depth_at(depth, det.center, self._window)
                if z is not None and t[2] > 0 and z < _MAX_DEPTH_M:
                    t = t * (z / t[2])
            T = pose_to_matrix(det.pose_R, t)
            centre_cam = transform(T, spec.offset_m)
            per_arm = sums.setdefault(spec.object, {arm: np.zeros(3) for arm in ARMS})
            for arm in ARMS:
                per_arm[arm] += np.asarray(self._cal.to_arm(arm, centre_cam))
            counts[spec.object] = counts.get(spec.object, 0) + 1
        for name, per_arm in sums.items():
            n = counts[name]
            in_frame: dict[Arm, Vec3] = {}
            for arm in ARMS:
                v = per_arm[arm] / n
                in_frame[arm] = (float(v[0]), float(v[1]), float(v[2]))
            self._memory[name] = ObjectView(name, in_frame, last_seen_step=step)
        if held is not None:
            arm, name = held
            g = grippers[arm].position
            o = self._grasp_offset
            own: Vec3 = (g[0] + o[0], g[1] + o[1], g[2] + o[2])
            prev = self._memory.get(name)
            in_frame = dict(prev.in_frame) if prev is not None else {}
            in_frame[arm] = own
            self._memory[name] = ObjectView(name, in_frame, last_seen_step=step, from_gripper=True)
        return WorldState(step=step, objects=dict(self._memory), grippers=dict(grippers))
