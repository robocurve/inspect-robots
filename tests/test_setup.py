"""Pure helpers for setup-wizard camera discovery and config rendering."""

from __future__ import annotations

from pathlib import Path

from inspect_robots._setup import (
    SUGGESTED,
    _read_raw_config,
    _render_config,
    _scan_cameras,
)


def test_scan_cameras_prefers_sorted_absolute_index0_entries(tmp_path: Path) -> None:
    v4l_dir = tmp_path / "by-id"
    v4l_dir.mkdir()
    for name in (
        "usb-camera-b-video-index1",
        "usb-camera-b-video-index0",
        "usb-camera-a-video-index0",
    ):
        (v4l_dir / name).touch()

    devices = _scan_cameras(v4l_dir)

    assert devices == [
        str(v4l_dir / "usb-camera-a-video-index0"),
        str(v4l_dir / "usb-camera-b-video-index0"),
    ]
    assert all(Path(device).is_absolute() for device in devices)


def test_scan_cameras_falls_back_to_all_sorted_entries(tmp_path: Path) -> None:
    v4l_dir = tmp_path / "by-path"
    v4l_dir.mkdir()
    for name in ("camera-z", "camera-a-video-index1", "camera-m"):
        (v4l_dir / name).touch()

    assert _scan_cameras(v4l_dir) == [
        str(v4l_dir / "camera-a-video-index1"),
        str(v4l_dir / "camera-m"),
        str(v4l_dir / "camera-z"),
    ]


def test_scan_cameras_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert _scan_cameras(tmp_path / "missing") == []


def test_read_raw_config_preserves_percent_and_literal_tilde(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_text(
        "[policy.args]\nlabel = 100% ready\ncheckpoint = ~/models/policy.pt\n",
        encoding="utf-8",
    )

    result = _read_raw_config(path)

    assert not isinstance(result, str)
    assert result["policy.args"]["label"] == "100% ready"
    assert result["policy.args"]["checkpoint"].startswith("~")


def test_read_raw_config_returns_parse_error_text(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_text("key = value without a section\n", encoding="utf-8")

    result = _read_raw_config(path)

    assert isinstance(result, str)
    assert "File contains no section headers" in result


def test_read_raw_config_returns_unicode_decode_error_text(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_bytes(b"[defaults]\npolicy = \xff\n")

    result = _read_raw_config(path)

    assert isinstance(result, str)
    assert "utf-8" in result


def test_render_config_matches_readme_quickstart_block() -> None:
    rendered = _render_config(
        dict(SUGGESTED),
        {
            "top_cam_device": "/dev/v4l/by-id/YOUR-TOP-CAM",
            "left_cam_device": "/dev/v4l/by-id/YOUR-LEFT-CAM",
            "right_cam_device": "/dev/v4l/by-id/YOUR-RIGHT-CAM",
        },
        {},
    )

    assert rendered == (
        "[defaults]\n"
        "policy = molmoact2        # from the inspect-robots-yam plugin\n"
        "embodiment = yam_arms     # same plugin; cameras configured below\n"
        "scorer = success_at_end\n"
        "max_steps = 1200          # 120 s at 10 Hz\n"
        "rerun = true              # live viewer of cameras/state/actions each run\n"
        "store_frames = true       # save each run's camera frames under logs/frames/\n"
        "\n"
        "[embodiment.args]\n"
        "top_cam_device = /dev/v4l/by-id/YOUR-TOP-CAM\n"
        "left_cam_device = /dev/v4l/by-id/YOUR-LEFT-CAM\n"
        "right_cam_device = /dev/v4l/by-id/YOUR-RIGHT-CAM\n"
    )


def test_render_config_omits_skipped_keys_and_empty_sections() -> None:
    rendered = _render_config(
        {"policy": "custom-policy"},
        {},
        {"embodiment.args": {}, "policy.args": {}},
    )

    assert "policy = custom-policy" in rendered
    for key in SUGGESTED.keys() - {"policy"}:
        assert f"{key} =" not in rendered
    assert "[embodiment.args]" not in rendered
    assert "[policy.args]" not in rendered
    assert rendered.endswith("\n")


def test_render_config_omits_empty_defaults_section() -> None:
    rendered = _render_config({}, {"top_cam_device": "/dev/top"}, {})

    assert "[defaults]" not in rendered
    assert rendered == "[embodiment.args]\ntop_cam_device = /dev/top\n"


def test_render_config_carries_unmanaged_content_without_managed_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.ini"
    path.write_text(
        "[defaults]\n"
        "policy = old-policy\n"
        "sim_embodiment = cubepick\n"
        "\n"
        "[embodiment.args]\n"
        "top_cam_device = /dev/old-top\n"
        "left_channel = can2\n"
        "\n"
        "[policy.args]\n"
        "checkpoint = ~/models/policy.pt\n",
        encoding="utf-8",
    )
    carried = _read_raw_config(path)
    assert not isinstance(carried, str)

    rendered = _render_config(
        {"policy": "new-policy", "embodiment": "yam_arms"},
        {
            "top_cam_device": "/dev/new-top",
            "left_cam_device": "/dev/new-left",
            "right_cam_device": "/dev/new-right",
        },
        carried,
    )

    assert "sim_embodiment = cubepick" in rendered
    assert "left_channel = can2" in rendered
    assert "[policy.args]\ncheckpoint = ~/models/policy.pt" in rendered
    assert rendered.count("policy = new-policy") == 1
    assert "old-policy" not in rendered
    assert rendered.count("top_cam_device = /dev/new-top") == 1
    assert "/dev/old-top" not in rendered
    assert rendered.index("[defaults]") < rendered.index("[embodiment.args]")
    assert rendered.index("[embodiment.args]") < rendered.index("[policy.args]")
