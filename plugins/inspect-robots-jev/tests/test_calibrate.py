import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots.types import Observation
from inspect_robots_jev import calibrate
from inspect_robots_jev.calibration import Calibration

FIXTURE = Path(__file__).parent / "fixtures" / "tags_cube_bowl.json"


class Det:
    def __init__(self, tag_id: int, t: tuple[float, float, float]) -> None:
        self.tag_id = tag_id
        self.pose_R = np.eye(
            3
        )  # tag flat on the table, aligned with the image axes, seen from above
        self.pose_t = np.array(t).reshape(3, 1)
        self.center = np.array([5.0, 5.0])


def write_inputs(tmp_path: Path) -> dict[str, Path]:
    corners = {
        "left": [[0.26, 0.24, 0.0], [0.34, 0.24, 0.0], [0.34, 0.16, 0.0], [0.26, 0.16, 0.0]],
        "right": [[0.26, -0.26, 0.0], [0.34, -0.26, 0.0], [0.34, -0.34, 0.0], [0.26, -0.34, 0.0]],
    }
    paths = {
        "corners": tmp_path / "corners.json",
        "frame": tmp_path / "frame.npy",
        "intr": tmp_path / "K.npy",
        "out": tmp_path / "cal.json",
    }
    paths["corners"].write_text(json.dumps(corners))
    np.save(paths["frame"], np.zeros((10, 10, 3), dtype=np.uint8))
    np.save(paths["intr"], np.array([[900.0, 0, 5], [0, 900.0, 5], [0, 0, 1]]))
    return paths


def argv(p: dict[str, Path]) -> list[str]:
    return [
        "--corners",
        str(p["corners"]),
        "--frame",
        str(p["frame"]),
        "--intrinsics",
        str(p["intr"]),
        "--tags",
        str(FIXTURE),
        "--out",
        str(p["out"]),
    ]


def test_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = write_inputs(tmp_path)
    seen: list[Any] = []

    def detector(gray: Any, params: Any, size: float) -> list[Det]:
        seen.append((gray.shape, params, size))
        return [Det(3, (0.0, 0.0, 0.5)), Det(20, (0.0, 0.0, 1.2))]

    assert calibrate.main(argv(p), detector=detector) == 0
    assert seen == [((10, 10), (900.0, 900.0, 5.0, 5.0), 0.08)]
    cal = Calibration.load(p["out"])
    # the tag sits at (0.30, 0.20, 0) in the left frame, 1.2 m below a downward-looking camera
    assert cal.to_arm("left", (0.0, 0.0, 1.2)) == pytest.approx((0.30, 0.20, 0.0))
    assert cal.to_arm("left", (0.0, 0.0, 0.7)) == pytest.approx((0.30, 0.20, 0.5))
    assert cal.to_arm("right", (0.0, 0.0, 1.2)) == pytest.approx((0.30, -0.30, 0.0))
    out = capsys.readouterr().out
    assert "left: camera at" in out and "wrote" in out


def test_missing_table_tag_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = write_inputs(tmp_path)
    assert calibrate.main(argv(p), detector=lambda g, k, s: [Det(3, (0, 0, 0.5))]) == 2
    assert "table tag 20 not found" in capsys.readouterr().err
    assert not p["out"].exists()


def test_missing_arm_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = write_inputs(tmp_path)
    p["corners"].write_text(json.dumps({"left": [[0, 0, 0]] * 4}))
    assert calibrate.main(argv(p), detector=lambda g, k, s: []) == 2
    assert "lacks the right arm" in capsys.readouterr().err


def test_default_detector_is_constructed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_inputs(tmp_path)
    monkeypatch.setattr(
        calibrate, "apriltag_detector", lambda: lambda g, k, s: [Det(20, (0, 0, 1.0))]
    )
    assert calibrate.main(argv(p)) == 0


def test_observation_to_npy(tmp_path: Path) -> None:
    obs = Observation(
        images={"top_cam": np.ones((4, 4, 3), dtype=np.uint8)},
        extra={"top_cam_intrinsics": np.eye(3, dtype=np.float32)},
    )
    frame, intr = calibrate.observation_to_npy(obs, "top_cam", tmp_path / "cap")
    assert np.load(frame).shape == (4, 4, 3) and np.load(intr).shape == (3, 3)
