import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots.types import Observation
from inspect_robots_jev import perceiver as perceiver_mod
from inspect_robots_jev.calibration import Calibration, pose_to_matrix
from inspect_robots_jev.perceiver import Perceiver, TagLayout, TagSpec, apriltag_detector, to_gray
from inspect_robots_jev.world import Arm, GripperView

FIXTURE = Path(__file__).parent / "fixtures" / "tags_cube_bowl.json"
K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]], dtype=np.float32)


class Det:
    def __init__(
        self, tag_id: int, t: tuple[float, float, float], center: tuple[float, float] = (10, 10)
    ) -> None:
        self.tag_id = tag_id
        self.pose_R = np.eye(3)
        self.pose_t = np.array(t).reshape(3, 1)
        self.center = np.array(center, dtype=float)


class FakeDetector:
    def __init__(self, dets: list[Det]) -> None:
        self.dets = dets
        self.calls: list[tuple[tuple[int, ...], str, tuple[float, float, float, float], float]] = []

    def __call__(
        self, gray: Any, params: tuple[float, float, float, float], size: float
    ) -> list[Det]:
        self.calls.append((gray.shape, str(gray.dtype), params, size))
        return list(self.dets)  # every size call returns everything; the perceiver filters by id


def identity_cal() -> Calibration:
    # camera frame == left frame; right frame is the left frame shifted 0.5 m in -y
    return Calibration(
        camera_to_arm={"left": np.eye(4), "right": pose_to_matrix(np.eye(3), [0.0, -0.5, 0.0])}
    )


def obs(depth: Any = None, image: Any = None) -> Observation:
    extra: dict[str, Any] = {"top_cam_intrinsics": K}
    if depth is not None:
        extra["top_cam_depth"] = depth
    img = image if image is not None else np.full((8, 8, 3), 200, dtype=np.uint8)
    return Observation(images={"top_cam": img}, state={}, extra=extra)


GRIPS: dict[Arm, GripperView] = {
    "left": GripperView((0.2, 0.0, 0.2), 1.0),
    "right": GripperView((0.2, 0.5, 0.2), 1.0),
}


def test_layout_load_and_sizes() -> None:
    layout = TagLayout.load(FIXTURE)
    assert layout.table_tag_id == 20 and layout.tags[3].object == "cube"
    assert layout.tags[0].offset_m == (0.0, 0.0, 0.0127)
    assert layout.sizes() == {0.02: [0, 1, 2, 3, 4], 0.04: [14], 0.08: [20]}


def test_layout_rejects_unlisted_table_tag(tmp_path: Path) -> None:
    f = tmp_path / "t.json"
    f.write_text('{"table_tag_id": 9, "tags": [{"id": 0, "object": "cube", "size_m": 0.02}]}')
    with pytest.raises(ValueError, match="table_tag_id 9"):
        TagLayout.load(f)


def test_offset_sign_pins_tag_frame_convention() -> None:
    det = FakeDetector([Det(0, (0.0, 0.0, 0.7))])
    p = Perceiver(
        layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=identity_cal(), detector=det
    )
    world = p.update(obs(), 0, grippers=GRIPS)
    assert world.objects["cube"].in_frame["left"] == pytest.approx((0.0, 0.0, 0.7127))
    assert world.objects["cube"].in_frame["right"] == pytest.approx((0.0, -0.5, 0.7127))
    assert world.objects["cube"].last_seen_step == 0
    assert world.objects["cube"].from_gripper is False
    # one detector call per distinct tag size, uint8 2-D input, K unpacked
    assert len(det.calls) == 3
    assert det.calls[0][0] == (8, 8) and det.calls[0][1] == "uint8"
    assert det.calls[0][2] == (900.0, 900.0, 640.0, 360.0)
    assert sorted(c[3] for c in det.calls) == [0.02, 0.04, 0.08]


def test_two_tags_on_one_object_average() -> None:
    det = FakeDetector([Det(0, (0.10, 0.0, 0.7)), Det(1, (0.12, 0.0, 0.7))])
    p = Perceiver(
        layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=identity_cal(), detector=det
    )
    world = p.update(obs(), 0, grippers=GRIPS)
    assert world.objects["cube"].in_frame["left"] == pytest.approx((0.11, 0.0, 0.7127))


def test_depth_refines_range_and_ignores_invalid() -> None:
    depth = np.full((20, 20), 0.9, dtype=np.float32)
    det = FakeDetector([Det(14, (0.1, 0.2, 0.6), center=(10, 10))])
    p = Perceiver(
        layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=identity_cal(), detector=det
    )
    world = p.update(obs(depth=depth), 0, grippers=GRIPS)
    assert world.objects["bowl"].in_frame["left"] == pytest.approx((0.15, 0.30, 0.9))
    # zero-arg callable depth with only invalid pixels falls back to the tag range
    bad = np.zeros((20, 20), dtype=np.float32)
    world = p.update(obs(depth=lambda: bad), 1, grippers=GRIPS)
    assert world.objects["bowl"].in_frame["left"] == pytest.approx((0.1, 0.2, 0.6))


def test_missing_object_keeps_memory_and_counts_unseen() -> None:
    det = FakeDetector([Det(0, (0.0, 0.0, 0.7))])
    p = Perceiver(
        layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=identity_cal(), detector=det
    )
    p.update(obs(), 0, grippers=GRIPS)
    det.dets = []
    world = p.update(obs(), 2, grippers=GRIPS)
    assert world.objects["cube"].in_frame["left"] == pytest.approx((0.0, 0.0, 0.7127))
    assert world.steps_since_seen("cube") == 2
    p.reset()
    assert p.update(obs(), 3, grippers=GRIPS).objects == {}


def test_held_object_follows_gripper() -> None:
    det = FakeDetector([Det(0, (0.0, 0.0, 0.7))])
    p = Perceiver(
        layout=TagLayout.load(FIXTURE),
        camera="top_cam",
        calibration=identity_cal(),
        detector=det,
        grasp_offset_m=(0.0, 0.0, -0.01),
    )
    p.update(obs(), 0, grippers=GRIPS)
    det.dets = []
    world = p.update(obs(), 1, grippers=GRIPS, held=("left", "cube"))
    assert world.objects["cube"].from_gripper is True
    assert world.objects["cube"].in_frame["left"] == pytest.approx((0.2, 0.0, 0.19))
    assert world.objects["cube"].in_frame["right"] == pytest.approx((0.0, -0.5, 0.7127))
    assert world.steps_since_seen("cube") == 0
    # a never-seen held object is created from the gripper alone
    p.reset()
    world = p.update(obs(), 2, grippers=GRIPS, held=("right", "cube"))
    assert world.objects["cube"].in_frame == {"right": (0.2, 0.5, 0.19)}


def test_to_gray_handles_2d_and_3d() -> None:
    g = to_gray(np.full((4, 4, 3), 100, dtype=np.uint8))
    assert g.shape == (4, 4) and g.dtype == np.uint8 and g.flags["C_CONTIGUOUS"]
    assert to_gray(np.ones((4, 4), dtype=np.float32) * 7).tolist()[0][0] == 7


def test_apriltag_detector_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []
    seen: list[Any] = []

    class StubDetector:
        def __init__(self, **kw: Any) -> None:
            built.append(kw)

        def detect(self, img: Any, **kw: Any) -> list[str]:
            seen.append((img.dtype, img.ndim, img.flags["C_CONTIGUOUS"], kw))
            return ["d"]

    monkeypatch.setitem(
        sys.modules, "pupil_apriltags", types.SimpleNamespace(Detector=StubDetector)
    )
    det = apriltag_detector()
    gray = to_gray(np.zeros((6, 6, 3), dtype=np.uint8))
    assert det(gray, (1.0, 2.0, 3.0, 4.0), 0.02) == ["d"]
    det(gray, (1.0, 2.0, 3.0, 4.0), 0.04)
    assert built == [{"families": "tag36h11", "quad_decimate": 1.0}]
    assert seen[0] == (
        np.uint8,
        2,
        True,
        {"estimate_tag_pose": True, "camera_params": (1.0, 2.0, 3.0, 4.0), "tag_size": 0.02},
    )


def test_perceiver_builds_real_detector_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def build() -> FakeDetector:
        calls.append(1)
        return FakeDetector([])

    monkeypatch.setattr(perceiver_mod, "apriltag_detector", build)
    p = Perceiver(layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=identity_cal())
    p.update(obs(), 0, grippers=GRIPS)
    p.update(obs(), 1, grippers=GRIPS)
    assert calls == [1]


def test_tagspec_is_plain_data() -> None:
    assert TagSpec(1, "x", 0.02, (0.0, 0.0, 0.0)).object == "x"


def test_depth_thunk_resolved_once_and_implausible_depth_ignored() -> None:
    calls: list[int] = []

    def thunk() -> Any:
        calls.append(1)
        return np.full((20, 20), 900.0, dtype=np.float32)  # millimetres by mistake

    det = FakeDetector([Det(0, (0.0, 0.0, 0.7)), Det(1, (0.0, 0.0, 0.7)), Det(14, (0.1, 0.1, 0.6))])
    p = Perceiver(
        layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=identity_cal(), detector=det
    )
    world = p.update(obs(depth=thunk), 0, grippers=GRIPS)
    assert calls == [1]
    assert world.objects["bowl"].in_frame["left"] == pytest.approx((0.1, 0.1, 0.6))
