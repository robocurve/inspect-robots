from __future__ import annotations

import base64
import io
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _dream_fakes import FRAMES, LEFT, RIGHT, STATE, reply
from PIL import Image

from inspect_robots_dream import _station


def decode(jpeg: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(jpeg)))


def test_layout_matches_the_yam_embodiments() -> None:
    assert _station.DIM_LABELS == (
        "left_j0",
        "left_j1",
        "left_j2",
        "left_j3",
        "left_j4",
        "left_j5",
        "left_gripper",
        "right_j0",
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_gripper",
    )
    assert _station.GRIPPERS == (6, 13)


def test_vector_accepts_only_finite_14d_values() -> None:
    assert _station.vector(STATE.tolist()).tolist() == STATE.tolist()
    for bad in (STATE[:13], np.full(14, np.nan), None):
        with pytest.raises(ValueError, match="14-D"):
            _station.vector(bad)


def test_jpeg_resizes_to_the_station_frame() -> None:
    rgb = np.zeros((360, 640, 3), np.uint8)
    rgb[:, :320] = (255, 0, 0)
    frame = decode(_station.jpeg(rgb))
    assert (frame.format, frame.size) == ("JPEG", (320, 176))
    pixels = np.asarray(frame)
    assert pixels[88, 40, 0] > 200 > pixels[88, 40, 1] and pixels[88, 280, 0] < 50


def test_jpeg_keeps_station_sized_frames_and_drops_alpha() -> None:
    assert decode(_station.jpeg(np.zeros((176, 320, 3), np.uint8))).size == (320, 176)
    assert decode(_station.jpeg(np.zeros((176, 320, 4), np.uint8))).mode == "RGB"


def test_frames_use_station_keys() -> None:
    assert _station.frames(FRAMES) == {
        "image_window.top": "T",
        "image_window.left": "L",
        "image_window.right": "R",
    }
    with pytest.raises(ValueError, match="right_cam"):
        _station.frames({"top_cam": "T", "left_cam": "L"})


def test_observation_splits_arms_and_grippers() -> None:
    assert _station.observation(STATE, "put the red cube on the plate") == {
        "prompt": "put the red cube on the plate",
        "robot.arm": "yam",
        "robot.exec_head": "joint",
        "robot.max_joint_vel_rad_s": 0.0,
        "robot.gripper_open": 1.0,
        "state.arm_l.joint_pos": LEFT[:6].tolist(),
        "state.arm_l.gripper_pos": LEFT[6:].tolist(),
        "state.arm_r.joint_pos": RIGHT[:6].tolist(),
        "state.arm_r.gripper_pos": RIGHT[6:].tolist(),
    }


def test_commanded_names_each_arms_row() -> None:
    last = _station.commanded(STATE)
    assert last.keys() == {"q_l_chunk", "q_r_chunk"}
    assert last["q_l_chunk"].tolist() == LEFT.tolist()
    assert last["q_r_chunk"].tolist() == RIGHT.tolist()


def test_rows_pack_left_then_right() -> None:
    rows = _station.rows(reply())
    assert rows.shape == (48, 14)
    assert rows[0].tolist() == STATE.tolist()


def test_rows_clip_grippers_within_slack() -> None:
    answer = reply(rows=2)
    answer["q_l_chunk"][0, 6] = 1.03
    answer["q_r_chunk"][1, 6] = -0.02
    rows = _station.rows(answer)
    assert (rows[0, 6], rows[1, 13]) == (1.0, 0.0)


def _set(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    return lambda answer: answer.__setitem__(key, value)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda answer: answer.pop("q_r_chunk"), "no 'q_r_chunk'"),
        (_set("q_l_chunk", np.zeros((48, 6))), r"\[1\.\.48, 7\]"),
        (_set("q_l_chunk", np.zeros((49, 7))), r"\[1\.\.48, 7\]"),
        (_set("q_l_chunk", np.zeros(7)), r"\[1\.\.48, 7\]"),
        (_set("q_l_chunk", np.full((48, 7), np.inf)), "non-finite"),
        (_set("q_r_chunk", np.tile([0.0] * 6 + [40.0], (48, 1))), "gripper leaves 0..1"),
        (_set("q_r_chunk", np.tile(RIGHT, (12, 1))), "differ"),
    ],
)
def test_rows_reject_replies_in_another_contract(
    change: Callable[[dict[str, Any]], None], match: str
) -> None:
    answer = reply()
    change(answer)
    with pytest.raises(ValueError, match=match):
        _station.rows(answer)
