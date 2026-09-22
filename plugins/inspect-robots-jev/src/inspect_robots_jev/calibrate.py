"""``python -m inspect_robots_jev.calibrate``: solve camera→arm transforms once per rig.

Inputs: a JSON of the table tag's four corners touched by each gripper
(printed tag's top-left, top-right, bottom-right, bottom-left, in metres in
that arm's frame), one top-camera frame saved as ``.npy`` (HxWx3 uint8), the
matching 3x3 intrinsics as ``.npy``, and the tag layout. Output: the
calibration JSON the ``jev`` policy loads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from inspect_robots.types import Observation
from inspect_robots_jev.calibration import (
    pose_to_matrix,
    solve_calibration,
    table_tag_pose_from_corners,
)
from inspect_robots_jev.perceiver import Detector, TagLayout, apriltag_detector, to_gray
from inspect_robots_jev.world import ARMS, Vec3


def _corners(raw: Sequence[Sequence[float]]) -> list[Vec3]:
    return [(float(c[0]), float(c[1]), float(c[2])) for c in raw]


def main(argv: Sequence[str] | None = None, *, detector: Detector | None = None) -> int:
    """Parse arguments, detect the table tag, write the calibration file; 0 on success."""
    parser = argparse.ArgumentParser(prog="inspect_robots_jev.calibrate", description=__doc__)
    parser.add_argument(
        "--corners", required=True, type=Path, help="JSON {left: [[x,y,z]*4], right: [...]}"
    )
    parser.add_argument(
        "--frame", required=True, type=Path, help=".npy HxWx3 uint8 top-camera frame"
    )
    parser.add_argument("--intrinsics", required=True, type=Path, help=".npy 3x3 camera matrix")
    parser.add_argument("--tags", required=True, type=Path, help="tag layout JSON")
    parser.add_argument("--out", required=True, type=Path, help="where to write calibration.json")
    args = parser.parse_args(argv)

    layout = TagLayout.load(args.tags)
    spec = layout.tags[layout.table_tag_id]
    corners = json.loads(args.corners.read_text())
    tag_in_arm = {}
    for arm in ARMS:
        if arm not in corners:
            print(f"corners file lacks the {arm} arm", file=sys.stderr)
            return 2
        tag_in_arm[arm] = table_tag_pose_from_corners(_corners(corners[arm]))

    frame = np.load(args.frame)
    K = np.load(args.intrinsics)
    params = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    detect = detector if detector is not None else apriltag_detector()
    found = [d for d in detect(to_gray(frame), params, spec.size_m) if d.tag_id == spec.tag_id]
    if not found:
        print(f"table tag {spec.tag_id} not found in {args.frame}", file=sys.stderr)
        return 2
    det = found[0]
    tag_in_camera = pose_to_matrix(det.pose_R, np.asarray(det.pose_t, dtype=np.float64).reshape(3))
    cal = solve_calibration(tag_in_camera=tag_in_camera, tag_in_arm=tag_in_arm)
    cal.save(args.out)
    for arm in ARMS:
        cam = cal.camera_to_arm[arm][:3, 3]
        print(f"{arm}: camera at x={cam[0]:.3f} y={cam[1]:.3f} z={cam[2]:.3f} m in the {arm} frame")
    print(f"wrote {args.out}")
    return 0


def observation_to_npy(observation: Observation, camera: str, out_dir: Path) -> tuple[Path, Path]:
    """Save one observation's frame and intrinsics as the two ``.npy`` files ``main`` reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = out_dir / f"{camera}_frame.npy"
    intr = out_dir / f"{camera}_intrinsics.npy"
    np.save(frame, np.asarray(observation.images[camera]))
    np.save(intr, np.asarray(observation.extra[f"{camera}_intrinsics"]))
    return frame, intr


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
