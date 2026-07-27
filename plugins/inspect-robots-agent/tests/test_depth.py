from __future__ import annotations

import numpy as np

from inspect_robots._html import _FRAME_LABEL_RE
from inspect_robots.types import Observation
from inspect_robots_agent._depth import _render_depth, depth_parts, resolve_depth


def test_render_depth_near_pixel_is_brighter_than_far_pixel() -> None:
    rendered = _render_depth(np.array([[0.1, 1.0]], dtype=np.float64))

    assert rendered[0, 0] > rendered[0, 1]


def test_render_depth_invalid_pixel_is_exactly_zero() -> None:
    rendered = _render_depth(np.array([[0.0, np.nan, np.inf, 1.0]], dtype=np.float64))

    assert rendered[0, :3].tolist() == [0, 0, 0]


def test_render_depth_valid_floor_is_64() -> None:
    depth = np.arange(1.0, 52.0, dtype=np.float64).reshape(3, 17)
    rendered = _render_depth(depth)

    assert rendered[2, 15] == 64


def test_render_depth_degenerate_window_renders() -> None:
    rendered = _render_depth(np.full((2, 3), 0.5, dtype=np.float64))

    assert rendered.shape == (2, 3)
    assert rendered.dtype == np.uint8
    assert np.all(rendered > 0)


def test_resolve_depth_calls_thunk_exactly_once() -> None:
    calls = 0

    def depth_thunk() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones((2, 2), dtype=np.float64)

    resolved = resolve_depth(
        Observation(
            images={"left_cam": np.zeros((2, 2, 3), dtype=np.uint8)},
            extra={"left_cam_depth": depth_thunk},
        )
    )

    assert calls == 1
    assert isinstance(resolved["left_cam"], np.ndarray)


def test_resolve_depth_failing_thunk_becomes_text() -> None:
    def depth_thunk() -> np.ndarray:
        raise RuntimeError("camera offline")

    resolved = resolve_depth(
        Observation(
            images={"left_cam": np.zeros((2, 2, 3), dtype=np.uint8)},
            extra={"left_cam_depth": depth_thunk},
        )
    )

    assert resolved == {"left_cam": "depth 'left_cam' unavailable: camera offline"}


def test_resolve_depth_non_2d_becomes_text() -> None:
    resolved = resolve_depth(
        Observation(
            images={"left_cam": np.zeros((2, 2, 3), dtype=np.uint8)},
            extra={"left_cam_depth": np.ones(3, dtype=np.float64)},
        )
    )

    assert resolved == {
        "left_cam": "depth 'left_cam' unavailable: expected a 2-D array, got shape (3,)"
    }


def test_resolve_depth_non_numeric_becomes_text() -> None:
    resolved = resolve_depth(
        Observation(
            images={"left_cam": np.zeros((2, 2, 3), dtype=np.uint8)},
            extra={"left_cam_depth": [["near", "far"]]},
        )
    )

    entry = resolved["left_cam"]
    assert isinstance(entry, str)
    assert entry.startswith("depth 'left_cam' unavailable:")


def test_resolve_depth_plain_array_passthrough() -> None:
    depth = np.ones((2, 2), dtype=np.float64)
    resolved = resolve_depth(
        Observation(
            images={"left_cam": np.zeros((2, 2, 3), dtype=np.uint8)},
            extra={"left_cam_depth": depth},
        )
    )

    assert resolved["left_cam"] is depth


def test_depth_parts_label_with_step_is_exact() -> None:
    depth = np.concatenate(
        (
            np.full(3, 0.09),
            np.full(81, 0.31),
            np.full(3, 1.41),
            np.zeros(13),
        )
    ).reshape(10, 10)

    parts = depth_parts("left_cam", depth, "step 3")

    assert parts[0] == {
        "type": "text",
        "text": (
            "depth 'left_cam' (step 3): bright 0.09 m -> dim 1.41 m "
            "(2nd-98th pctl), 87% valid, center 0.31 m:"
        ),
    }


def test_depth_parts_label_without_step_is_exact() -> None:
    depth = np.concatenate(
        (
            np.full(3, 0.09),
            np.full(81, 0.31),
            np.full(3, 1.41),
            np.zeros(13),
        )
    ).reshape(10, 10)

    parts = depth_parts("left_cam", depth, "")

    assert parts[0] == {
        "type": "text",
        "text": (
            "depth 'left_cam': bright 0.09 m -> dim 1.41 m "
            "(2nd-98th pctl), 87% valid, center 0.31 m:"
        ),
    }


def test_depth_parts_image_is_png_data_url() -> None:
    parts = depth_parts("left_cam", np.ones((2, 2), dtype=np.float64), "step 3")

    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_depth_parts_string_entry_is_text_only() -> None:
    text = "depth 'left_cam' unavailable: camera offline"

    assert depth_parts("left_cam", text, "step 3") == [{"type": "text", "text": text}]


def test_depth_parts_less_than_one_percent_valid_is_text_only() -> None:
    parts = depth_parts("left_cam", np.zeros((10, 10), dtype=np.float64), "step 3")

    assert parts == [
        {
            "type": "text",
            "text": "depth 'left_cam': no usable depth this frame (0% valid)",
        }
    ]


def test_depth_parts_omits_center_when_center_pixel_is_invalid() -> None:
    depth = np.ones((10, 10), dtype=np.float64)
    depth[5, 5] = 0.0

    parts = depth_parts("left_cam", depth, "step 3")

    assert "center" not in parts[0]["text"]


def test_depth_label_does_not_match_frame_label_regex() -> None:
    parts = depth_parts("left_cam", np.ones((2, 2), dtype=np.float64), "step 3")
    depth_label = parts[0]["text"]

    assert _FRAME_LABEL_RE.search(depth_label) is None
