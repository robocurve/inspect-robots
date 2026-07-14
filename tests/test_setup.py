"""Pure helpers for setup-wizard camera discovery and config rendering."""

from __future__ import annotations

import errno
import io
import os
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from inspect_robots._setup import (
    _VIDIOC_ENUM_FMT,
    _VIDIOC_QUERYCAP,
    SUGGESTED,
    _can_serial,
    _identify_by_replug,
    _read_raw_config,
    _render_config,
    _scan_cameras,
    _scan_can,
    _scan_serial,
    _suggest_can_pinning,
    _v4l2_color_capture,
    run_setup,
)
from inspect_robots.conformance import DeviceSlot


def _scripted_input(
    responses: list[str | BaseException],
) -> tuple[Callable[[str], str], list[str]]:
    pending = responses.copy()
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        response = pending.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    return input_fn, prompts


def _config_path(tmp_path: Path) -> Path:
    return tmp_path / "inspect-robots" / "config.ini"


def _make_devices(directory: Path, count: int = 3) -> list[str]:
    directory.mkdir(parents=True)
    devices: list[str] = []
    for number in range(1, count + 1):
        path = directory / f"camera-{number}-video-index0"
        path.touch()
        devices.append(str(path))
    return devices


def _make_can_interfaces(sysfs_net: Path, *names: str) -> None:
    for name in names:
        interface = sysfs_net / name
        interface.mkdir(parents=True)
        (interface / "type").write_text("280\n", encoding="utf-8")


def _attach_can_adapter(
    tmp_path: Path,
    sysfs_net: Path,
    ifname: str,
    serial: str | None,
    *,
    usb: bool = True,
) -> None:
    # "usb1" matches the numbered bus segments real sysfs uses (never bare "usb").
    root = tmp_path / ("usb1" if usb else "platform") / ifname / "adapter"
    root.mkdir(parents=True)
    if serial is not None:
        (root / "serial").write_text(serial + "\n", encoding="utf-8")
    device = root / "net-device"
    device.mkdir()
    try:
        (sysfs_net / ifname / "device").symlink_to(device, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


def _register_device_slots(
    monkeypatch: pytest.MonkeyPatch,
    slots: tuple[DeviceSlot, ...],
    name: str = "slot-body",
) -> None:
    class _Factory:
        DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = slots

    monkeypatch.setattr(
        "inspect_robots.registry.registered",
        lambda kind: {name: _Factory} if kind == "embodiment" else {},
    )


def _slot_defaults(name: str = "slot-body") -> list[str]:
    return ["", name, "", "", "", ""]


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspect_robots.registry.registered", lambda _kind: {})


def test_scan_cameras_prefers_color_capture_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v4l_dir = tmp_path / "by-id"
    v4l_dir.mkdir()
    # RealSense-style layout: index0 is the depth node, index2 the IR pair,
    # index4 the color stream — the name says nothing about capturability.
    verdicts = {
        "usb-realsense-video-index0": False,
        "usb-realsense-video-index2": False,
        "usb-realsense-video-index4": True,
        "usb-webcam-video-index0": True,
        "usb-webcam-video-index1": None,
    }
    for name in verdicts:
        (v4l_dir / name).touch()
    monkeypatch.setattr(
        "inspect_robots._setup._v4l2_color_capture",
        lambda path: verdicts[Path(path).name],
    )

    devices = _scan_cameras(v4l_dir)

    assert devices == [
        str(v4l_dir / "usb-realsense-video-index4"),
        str(v4l_dir / "usb-webcam-video-index0"),
    ]
    assert all(Path(device).is_absolute() for device in devices)


def test_scan_cameras_lists_all_entries_when_probe_is_inconclusive(tmp_path: Path) -> None:
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
        str(v4l_dir / "usb-camera-b-video-index1"),
    ]


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


_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_META_CAPTURE = 0x00800000
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def _fake_v4l2_ioctl(
    capabilities: int, device_caps: int, fourccs: list[bytes]
) -> Callable[[int, int, bytearray], int]:
    """Simulate the kernel's VIDIOC_QUERYCAP / VIDIOC_ENUM_FMT ioctl ABI.

    Buffer offsets follow ``struct v4l2_capability`` (capabilities at byte 84,
    device_caps at 88) and ``struct v4l2_fmtdesc`` (index/type at 0/4,
    pixelformat at 44) — fixed kernel layouts, not implementation choices.
    """

    def ioctl(fd: int, request: int, buf: bytearray) -> int:
        if request == _VIDIOC_QUERYCAP:
            struct.pack_into("=II", buf, 84, capabilities, device_caps)
        elif request == _VIDIOC_ENUM_FMT:
            index, buf_type = struct.unpack_from("=II", buf, 0)
            assert buf_type == 1  # V4L2_BUF_TYPE_VIDEO_CAPTURE
            if index >= len(fourccs):
                raise OSError(errno.EINVAL, "format enumeration exhausted")
            buf[44:48] = fourccs[index]
        else:
            raise AssertionError(f"unexpected ioctl request {request:#x}")
        return 0

    return ioctl


def _endless_v4l2_ioctl() -> Callable[[int, int, bytearray], int]:
    """Simulate a misbehaving driver whose format enumeration never ends."""

    def ioctl(fd: int, request: int, buf: bytearray) -> int:
        if request == _VIDIOC_QUERYCAP:
            struct.pack_into("=II", buf, 84, _V4L2_CAP_DEVICE_CAPS, _V4L2_CAP_VIDEO_CAPTURE)
        else:
            buf[44:48] = b"Z16 "
        return 0

    return ioctl


def _probe_target(tmp_path: Path) -> Path:
    node = tmp_path / "cam"
    node.touch()
    return node


_needs_fcntl = pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")


@_needs_fcntl
def test_v4l2_color_capture_true_for_color_capable_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fcntl.ioctl",
        _fake_v4l2_ioctl(
            _V4L2_CAP_DEVICE_CAPS,
            _V4L2_CAP_VIDEO_CAPTURE,
            [b"GREY", b"YUYV"],
        ),
    )

    assert _v4l2_color_capture(_probe_target(tmp_path)) is True


@_needs_fcntl
def test_v4l2_color_capture_true_for_bayer_only_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Raw Bayer nodes are color cameras OpenCV can debayer; hiding them in a
    # mixed rig would silently drop a working camera from the listing.
    monkeypatch.setattr(
        "fcntl.ioctl",
        _fake_v4l2_ioctl(
            _V4L2_CAP_DEVICE_CAPS,
            _V4L2_CAP_VIDEO_CAPTURE,
            [b"RGGB"],
        ),
    )

    assert _v4l2_color_capture(_probe_target(tmp_path)) is True


@_needs_fcntl
def test_v4l2_color_capture_false_for_depth_only_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No DEVICE_CAPS flag: the probe must fall back to `capabilities`.
    monkeypatch.setattr(
        "fcntl.ioctl",
        _fake_v4l2_ioctl(_V4L2_CAP_VIDEO_CAPTURE, 0, [b"Z16 "]),
    )

    assert _v4l2_color_capture(_probe_target(tmp_path)) is False


@_needs_fcntl
def test_v4l2_color_capture_false_for_endless_format_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A driver that never terminates VIDIOC_ENUM_FMT must yield False, not a
    # hung wizard.
    monkeypatch.setattr("fcntl.ioctl", _endless_v4l2_ioctl())

    assert _v4l2_color_capture(_probe_target(tmp_path)) is False


@_needs_fcntl
def test_v4l2_color_capture_false_for_metadata_node_without_enumerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Color fourccs are on offer, but a node without VIDEO_CAPTURE must be
    # rejected on capabilities alone — enumerating it would report True.
    monkeypatch.setattr(
        "fcntl.ioctl",
        _fake_v4l2_ioctl(
            _V4L2_CAP_DEVICE_CAPS | _V4L2_CAP_META_CAPTURE,
            _V4L2_CAP_META_CAPTURE,
            [b"YUYV"],
        ),
    )

    assert _v4l2_color_capture(_probe_target(tmp_path)) is False


def test_v4l2_color_capture_none_for_unopenable_path(tmp_path: Path) -> None:
    assert _v4l2_color_capture(tmp_path / "missing") is None


def test_v4l2_color_capture_none_for_non_v4l2_file(tmp_path: Path) -> None:
    # A regular file opens fine but rejects V4L2 ioctls (ENOTTY).
    assert _v4l2_color_capture(_probe_target(tmp_path)) is None


def test_v4l2_color_capture_none_without_fcntl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "fcntl", None)

    assert _v4l2_color_capture(_probe_target(tmp_path)) is None


def test_scan_can_filters_type_280_sorts_and_skips_unreadable(tmp_path: Path) -> None:
    sysfs_net = tmp_path / "net"
    for ifname, interface_type in (("can9", "280\n"), ("eth0", "1\n"), ("can1", "280")):
        interface = sysfs_net / ifname
        interface.mkdir(parents=True)
        (interface / "type").write_text(interface_type, encoding="utf-8")
    (sysfs_net / "broken").mkdir()

    assert _scan_can(sysfs_net) == ["can1", "can9"]


def test_scan_can_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert _scan_can(tmp_path / "missing") == []


def test_scan_can_skips_undecodable_type_file(tmp_path: Path) -> None:
    interface = tmp_path / "net" / "can0"
    interface.mkdir(parents=True)
    (interface / "type").write_bytes(b"\xff")

    assert _scan_can(tmp_path / "net") == []


def test_scan_serial_lists_sorted_absolute_paths(tmp_path: Path) -> None:
    serial_by_id = tmp_path / "serial-by-id"
    serial_by_id.mkdir()
    for name in ("usb-controller-z", "usb-controller-a"):
        (serial_by_id / name).touch()

    assert _scan_serial(serial_by_id) == [
        str(serial_by_id / "usb-controller-a"),
        str(serial_by_id / "usb-controller-z"),
    ]


def test_scan_serial_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert _scan_serial(tmp_path / "missing") == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_can_serial_reads_through_device_symlink(tmp_path: Path) -> None:
    sysfs_net = tmp_path / "net"
    interface = sysfs_net / "can0"
    interface.mkdir(parents=True)
    adapter = tmp_path / "usb3" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "serial").write_text(" 3B004B\n", encoding="utf-8")
    device = adapter / "net-device"
    device.mkdir()
    try:
        (interface / "device").symlink_to(device, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    assert _can_serial(sysfs_net, "can0") == "3B004B"


def test_can_serial_missing_serial_returns_none(tmp_path: Path) -> None:
    assert _can_serial(tmp_path / "net", "can0") is None


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


def test_render_config_long_commented_value_round_trips(tmp_path: Path) -> None:
    policy = "my_custom_policy_v2"
    path = tmp_path / "config.ini"
    path.write_text(_render_config({"policy": policy}, {}, {}), encoding="utf-8")

    carried = _read_raw_config(path)

    assert not isinstance(carried, str)
    assert carried["defaults"]["policy"] == policy


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


def test_run_setup_defaults_and_numbered_cameras_write_golden_config(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "", "1", "2", "3"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path), "DISPLAY": ":0"},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "missing-by-path",
    )

    path = _config_path(tmp_path)
    assert result == 0
    assert path.read_text(encoding="utf-8") == _render_config(
        dict(SUGGESTED),
        {
            "top_cam_device": devices[0],
            "left_cam_device": devices[1],
            "right_cam_device": devices[2],
        },
        {},
    )
    output = out.getvalue()
    assert f"Found 3 camera device(s) under {by_id}:" in output
    assert f"  1. {Path(devices[0]).name}" in output
    assert f"Wrote {path}" in output
    assert 'Next: inspect-robots "place the fork on the plate"' in output


def test_run_setup_headless_defaults_rerun_false_and_explains(tmp_path: Path) -> None:
    scripted_input, prompts = _scripted_input([""] * 7)
    out = io.StringIO()
    note = (
        "no display detected (SSH?): the rerun viewer cannot open here; "
        "use --rerun-connect to stream to a viewer on another machine; "
        "frames still record with store_frames"
    )

    def input_fn(prompt: str) -> str:
        if prompt.startswith("live rerun viewer"):
            assert note in out.getvalue().splitlines()
        return scripted_input(prompt)

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "live rerun viewer [false]" in prompts[4]
    assert out.getvalue().splitlines().count(note) == 1
    assert "rerun = false" in _config_path(tmp_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("display_variable", ["DISPLAY", "WAYLAND_DISPLAY"])
def test_run_setup_with_display_defaults_rerun_true_without_note(
    tmp_path: Path,
    display_variable: str,
) -> None:
    input_fn, prompts = _scripted_input([""] * 7)
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path), display_variable: ":0"},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "live rerun viewer [true]" in prompts[4]
    assert "no display detected (SSH?)" not in out.getvalue()
    assert "rerun = true" in _config_path(tmp_path).read_text(encoding="utf-8")


def test_run_setup_headless_existing_rerun_true_wins_and_note_is_printed(
    tmp_path: Path,
) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text("[defaults]\nrerun = true\n", encoding="utf-8")
    input_fn, prompts = _scripted_input([""] * 7)
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "live rerun viewer [true]" in prompts[4]
    assert "no display detected (SSH?)" in out.getvalue()
    assert "rerun = true" in path.read_text(encoding="utf-8")


def test_run_setup_strips_typed_overrides_and_whitespace_uses_default(tmp_path: Path) -> None:
    input_fn, _ = _scripted_input(
        ["  my-policy  ", "  my-body  ", "   ", " 42 ", " false ", " false ", ""]
    )

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert "policy = my-policy" in text
    assert "embodiment = my-body" in text
    assert "scorer = success_at_end" in text
    assert "max_steps = 42" in text
    assert "rerun = false" in text
    assert "store_frames = false" in text
    assert "[embodiment.args]" not in text


def test_run_setup_reprompts_invalid_typed_values(tmp_path: Path) -> None:
    input_fn, prompts = _scripted_input(
        ["", "", "", "abc", "0", "7", "maybe", "false", "1", "true", ""]
    )
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path), "DISPLAY": ":0"},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert sum(prompt.startswith("max steps ") for prompt in prompts) == 3
    assert sum(prompt.startswith("live rerun viewer ") for prompt in prompts) == 2
    assert sum(prompt.startswith("store camera frames ") for prompt in prompts) == 2
    assert out.getvalue().count("must be an integer >= 1") == 2
    assert out.getvalue().count("must be true or false") == 2


def test_run_setup_noninteractive_raises() -> None:
    input_fn, _ = _scripted_input([])

    with pytest.raises(
        SystemExit,
        match=r"^setup is interactive; see the README for manual config$",
    ):
        run_setup({}, input_fn=input_fn, out=io.StringIO(), interactive=False)


def test_run_setup_without_config_home_raises() -> None:
    input_fn, _ = _scripted_input([])

    with pytest.raises(
        SystemExit,
        match=r"^cannot locate a config home: set \$XDG_CONFIG_HOME or \$HOME$",
    ):
        run_setup({}, input_fn=input_fn, out=io.StringIO(), interactive=True)


@pytest.mark.parametrize("interruption", [EOFError(), KeyboardInterrupt()])
def test_run_setup_interruption_aborts_without_writing(
    tmp_path: Path, interruption: BaseException
) -> None:
    input_fn, _ = _scripted_input(["", interruption])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 1
    assert "setup aborted; nothing written" in out.getvalue()
    assert not _config_path(tmp_path).exists()
    assert not _config_path(tmp_path).with_name("config.ini.bak").exists()


def test_run_setup_existing_valid_config_supplies_prompt_defaults_and_backup(
    tmp_path: Path,
) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    old = (
        "[defaults]\n"
        "policy = old-policy\n"
        "embodiment = old-body\n"
        "scorer = old-scorer\n"
        "max_steps = 88\n"
        "rerun = false\n"
        "store_frames = false\n"
    )
    path.write_text(old, encoding="utf-8")
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", ""])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    for expected in ("old-policy", "old-body", "old-scorer", "88", "false"):
        assert any(f"[{expected}]" in prompt for prompt in prompts)
    assert "policy = old-policy" in path.read_text(encoding="utf-8")
    assert path.with_name("config.ini.bak").read_text(encoding="utf-8") == old


@pytest.mark.parametrize("answer", ["y", ""])
def test_run_setup_repairs_malformed_config_and_backs_it_up(tmp_path: Path, answer: str) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    broken = "key = value without a section\n"
    path.write_text(broken, encoding="utf-8")
    input_fn, prompts = _scripted_input([answer, "", "", "", "", "", "", ""])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path), "DISPLAY": ":0"},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "File contains no section headers" in out.getvalue()
    assert "Back up the broken file and start fresh? [Y/n]" in prompts[0]
    assert path.read_text(encoding="utf-8") == _render_config(dict(SUGGESTED), {}, {})
    assert path.with_name("config.ini.bak").read_text(encoding="utf-8") == broken


def test_run_setup_declines_malformed_config_repair(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    broken = "not an ini file\n"
    path.write_text(broken, encoding="utf-8")
    input_fn, _ = _scripted_input(["n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
    )

    assert result == 1
    assert "File contains no section headers" in out.getvalue()
    assert "setup aborted; nothing written" in out.getvalue()
    assert path.read_text(encoding="utf-8") == broken
    assert not path.with_name("config.ini.bak").exists()


def test_run_setup_ignores_only_invalid_existing_prompt_values(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\n"
        "policy = kept-policy\n"
        "max_steps = abc\n"
        "rerun = perhaps\n"
        "store_frames = false\n",
        encoding="utf-8",
    )
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", ""])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "ignoring invalid max_steps 'abc' from config.ini" in out.getvalue()
    assert "ignoring invalid rerun 'perhaps' from config.ini" in out.getvalue()
    assert any("policy [kept-policy]" in prompt for prompt in prompts)
    assert any("max steps [1200]" in prompt for prompt in prompts)
    assert any("live rerun viewer [false]" in prompt for prompt in prompts)
    assert any("store camera frames [false]" in prompt for prompt in prompts)


def test_run_setup_warns_only_for_unregistered_policy_and_embodiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inspect_robots.registry.registered",
        lambda kind: {"known-policy" if kind == "policy" else "known-body": object()},
    )
    input_fn, _ = _scripted_input(
        ["missing-policy", "missing-body", "missing-scorer", "", "", "", ""]
    )
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    warning = (
        "is not registered here — install its plugin, e.g. `uv pip install inspect-robots-yam`"
    )
    assert f"'missing-policy' {warning}" in out.getvalue()
    assert f"'missing-body' {warning}" in out.getvalue()
    assert "missing-scorer' is not registered" not in out.getvalue()


def test_run_setup_registered_names_do_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inspect_robots.registry.registered",
        lambda kind: {"known-policy": object()} if kind == "policy" else {"known-body": object()},
    )
    input_fn, _ = _scripted_input(["known-policy", "known-body", "", "", "", "", ""])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "is not registered here" not in out.getvalue()


def test_run_setup_carries_unmanaged_defaults_and_embodiment_args(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\npolicy = old\nsim_embodiment = cubepick\n"
        "\n[embodiment.args]\nleft_channel = can2\ntop_cam_device = /dev/old\n",
        encoding="utf-8",
    )
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = path.read_text(encoding="utf-8")
    assert result == 0
    assert "sim_embodiment = cubepick" in text
    assert "left_channel = can2" in text
    assert "top_cam_device = /dev/old" in text


def test_run_setup_carries_multiline_values_in_all_sections(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\n"
        "notes = defaults line 1\n"
        "    defaults line 2\n"
        "\n"
        "[embodiment.args]\n"
        "calibration = args line 1\n"
        "    args line 2\n"
        "\n"
        "[policy.args]\n"
        "notes = policy line 1\n"
        "    policy line 2\n",
        encoding="utf-8",
    )
    original = _read_raw_config(path)
    assert not isinstance(original, str)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    rewritten = _read_raw_config(path)
    assert result == 0
    assert not isinstance(rewritten, str)
    for section, key in (
        ("defaults", "notes"),
        ("embodiment.args", "calibration"),
        ("policy.args", "notes"),
    ):
        original_lines = [line.lstrip() for line in original[section][key].splitlines()]
        rewritten_lines = [line.lstrip() for line in rewritten[section][key].splitlines()]
        assert rewritten_lines == original_lines


def test_run_setup_declining_cameras_preserves_existing_assignments(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    camera_lines = [
        "top_cam_device = /dev/old-top",
        "left_cam_device = /dev/old-left",
        "right_cam_device = /dev/old-right",
    ]
    path.write_text("[embodiment.args]\n" + "\n".join(camera_lines) + "\n", encoding="utf-8")
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = path.read_text(encoding="utf-8")
    assert result == 0
    assert all(line in text.splitlines() for line in camera_lines)


def test_run_setup_camera_choices_manual_paths_skip_and_invalid_entries(
    tmp_path: Path,
) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    missing = "/another-machine/top-camera"
    input_fn, prompts = _scripted_input(
        [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "garbage",
            "9",
            "relative/path",
            missing,
            devices[1],
            "3",
        ]
    )
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert f"top_cam_device = {missing}" in text
    assert f"left_cam_device = {devices[1]}" in text
    assert f"right_cam_device = {devices[2]}" in text
    assert f"warning: {missing} does not exist here " in out.getvalue()
    assert f"warning: {devices[1]} does not exist here" not in out.getvalue()
    assert (
        out.getvalue().count("enter a device number, absolute path, 'u' to identify, or 's'") == 3
    )
    assert sum(prompt.startswith("top camera") for prompt in prompts) == 4


def test_run_setup_skipping_every_camera_writes_no_camera_keys(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    _make_devices(by_id)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "", "s", "s", "s"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "_cam_device" not in _config_path(tmp_path).read_text(encoding="utf-8")


def test_run_setup_camera_offer_defaults_yes_when_devices_found(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    _make_devices(by_id)
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "", "s", "s", "s"])

    run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    assert "[Y/n]" in prompts[6]
    assert any(prompt.startswith("top camera") for prompt in prompts)


def test_run_setup_camera_offer_defaults_yes_for_existing_camera_keys(
    tmp_path: Path,
) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text("[embodiment.args]\ntop_cam_device = /dev/old\n", encoding="utf-8")
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "", "s", "s", "s"])

    run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert "[Y/n]" in prompts[6]
    assert "_cam_device" not in path.read_text(encoding="utf-8")


def test_run_setup_camera_offer_defaults_no_without_devices_or_config(tmp_path: Path) -> None:
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", ""])
    out = io.StringIO()

    run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert "[y/N]" in prompts[6]
    assert "no /dev/v4l devices found" not in out.getvalue()


def test_run_setup_partial_cameras_can_go_back_and_assign_all(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    input_fn, prompts = _scripted_input(
        ["", "", "", "", "", "", "", "1", "s", "s", "", "1", "2", "3"]
    )
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert "yam_arms needs all three cameras or none" in out.getvalue()
    assert sum(prompt.startswith("top camera") for prompt in prompts) == 2
    assert all(
        f"{role}_cam_device = {device}" in text
        for role, device in zip(("top", "left", "right"), devices, strict=True)
    )


def test_run_setup_partial_cameras_can_write_none(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    _make_devices(by_id)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "", "1", "s", "s", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "yam_arms needs all three cameras or none" in out.getvalue()
    assert "_cam_device" not in _config_path(tmp_path).read_text(encoding="utf-8")


def test_run_setup_partial_cameras_write_none_drops_existing_assignments(
    tmp_path: Path,
) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[embodiment.args]\n"
        "top_cam_device = /dev/old-top\n"
        "left_cam_device = /dev/old-left\n"
        "right_cam_device = /dev/old-right\n",
        encoding="utf-8",
    )
    by_id = tmp_path / "by-id"
    _make_devices(by_id)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "", "1", "s", "s", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "_cam_device" not in path.read_text(encoding="utf-8")


def test_run_setup_falls_back_to_by_path_devices(tmp_path: Path) -> None:
    by_path = tmp_path / "by-path"
    devices = _make_devices(by_path)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "", "1", "2", "3"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=by_path,
    )

    assert result == 0
    assert f"Found 3 camera device(s) under {by_path}:" in out.getvalue()
    assert f"top_cam_device = {devices[0]}" in _config_path(tmp_path).read_text(encoding="utf-8")


def test_run_setup_without_detected_devices_accepts_manual_paths(tmp_path: Path) -> None:
    manual = ["/remote/top", "/remote/left", "/remote/right"]
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "y", *manual])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert "no /dev/v4l devices found (not Linux, or no cameras attached)" in out.getvalue()
    assert all(
        f"{role}_cam_device = {path}" in text
        for role, path in zip(("top", "left", "right"), manual, strict=True)
    )


def test_run_setup_yes_no_prompts_reprompt_invalid_answers(tmp_path: Path) -> None:
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "perhaps", "yes", "s", "s", "s"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert sum("Configure cameras?" in prompt for prompt in prompts) == 2
    assert "please answer yes or no" in out.getvalue()


def test_identify_by_replug_finds_disappeared_then_restored_device() -> None:
    devices = ["/dev/camera-top", "/dev/camera-left", "/dev/camera-right"]
    scans = iter(
        [
            [devices[0], devices[2]],
            devices,
        ]
    )
    input_fn, prompts = _scripted_input(["", ""])
    out = io.StringIO()

    identified = _identify_by_replug(
        "left",
        devices,
        input_fn=input_fn,
        out=out,
        rescan=lambda: next(scans),
    )

    assert identified == devices[1]
    assert prompts == [
        "Unplug the left camera now, then press Enter...",
        "Plug it back in, then press Enter...",
    ]
    assert "That was: camera-left" in out.getvalue()


@pytest.mark.parametrize(
    ("noun", "retry_noun", "label", "expected"),
    [
        ("CAN interface", "CAN interface", "left CAN channel", "no CAN interface disappeared"),
        ("serial device", "serial device", "arm serial port", "no serial device disappeared"),
    ],
)
def test_identify_by_replug_parameterizes_non_camera_nouns(
    noun: str,
    retry_noun: str,
    label: str,
    expected: str,
) -> None:
    input_fn, prompts = _scripted_input([""])
    out = io.StringIO()

    identified = _identify_by_replug(
        label,
        ["device0"],
        input_fn=input_fn,
        out=out,
        rescan=lambda: ["device0"],
        noun=noun,
        retry_noun=retry_noun,
        unplug_label=label,
    )

    assert identified is None
    assert prompts == [f"Unplug the {label} now, then press Enter..."]
    assert expected in out.getvalue()


@pytest.mark.parametrize("detected_on_retry", [True, False])
def test_identify_by_replug_retries_replug_scan_once(
    detected_on_retry: bool,
) -> None:
    devices = ["/dev/camera-top", "/dev/camera-left"]
    without_top = [devices[1]]
    scans = iter(
        [
            without_top,
            without_top,
            devices if detected_on_retry else without_top,
        ]
    )
    input_fn, prompts = _scripted_input(["", "", ""])
    out = io.StringIO()

    identified = _identify_by_replug(
        "top",
        devices,
        input_fn=input_fn,
        out=out,
        rescan=lambda: next(scans),
    )

    assert identified == devices[0]
    assert prompts[-1] == "camera-top was not detected; press Enter to rescan..."
    warning = "warning: camera-top was still not detected; keeping the assignment"
    assert (warning in out.getvalue()) is not detected_on_retry


def test_run_setup_identifies_camera_by_real_directory_replug(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    pending = ["", "", "", "", "", "", "", "u", "", "", "1", "3"]
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if prompt.startswith("Unplug the top camera"):
            Path(devices[1]).unlink()
        elif prompt.startswith("Plug it back in"):
            Path(devices[1]).touch()
        return pending.pop(0)

    out = io.StringIO()
    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert f"top_cam_device = {devices[1]}" in text
    assert f"left_cam_device = {devices[0]}" in text
    assert f"right_cam_device = {devices[2]}" in text
    assert f"That was: {Path(devices[1]).name}" in out.getvalue()
    assert Path(devices[1]).exists()
    assert "Plug it back in, then press Enter..." in prompts


@pytest.mark.parametrize("missing_count", [0, 2])
def test_run_setup_ambiguous_unplug_diff_explains_and_reprompts_role(
    tmp_path: Path,
    missing_count: int,
) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    pending = ["", "", "", "", "", "", "", "u", "", "1", "2", "3"]
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if prompt.startswith("Unplug the top camera"):
            for device in devices[:missing_count]:
                Path(device).unlink()
        return pending.pop(0)

    out = io.StringIO()
    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert sum(prompt.startswith("top camera") for prompt in prompts) == 2
    explanation = (
        "no camera device disappeared" if missing_count == 0 else "2 camera devices disappeared"
    )
    assert explanation in out.getvalue()


def test_run_setup_path_toggle_is_accepted_when_not_advertised(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    by_path = tmp_path / "by-path"
    by_id_devices = _make_devices(by_id)
    _make_devices(by_path)
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "", "p", "p", "1", "2", "3"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=by_path,
    )

    output = out.getvalue()
    role_prompts = [prompt for prompt in prompts if " camera — " in prompt]
    assert result == 0
    assert output.count(f"Found 3 camera device(s) under {by_id}:") == 2
    assert output.count(f"Found 3 camera device(s) under {by_path}:") == 1
    assert all("'p'" not in prompt for prompt in role_prompts)
    assert "only 3 by-id entries" not in output
    assert f"top_cam_device = {by_id_devices[0]}" in _config_path(tmp_path).read_text(
        encoding="utf-8"
    )


def test_run_setup_advertises_by_path_when_by_id_entries_collide(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    by_path = tmp_path / "by-path"
    _make_devices(by_id, count=1)
    by_path_devices = _make_devices(by_path)
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "", "p", "1", "2", "3"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=by_path,
    )

    explanation = (
        "only 1 by-id entries for 3 detected cameras — identical cameras without serials "
        "collide there; by-path names are stable per physical USB port"
    )
    role_prompts = [prompt for prompt in prompts if " camera — " in prompt]
    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert out.getvalue().count(explanation) == 1
    assert all("'p' to switch listing" in prompt for prompt in role_prompts)
    assert f"Found 3 camera device(s) under {by_path}:" in out.getvalue()
    assert all(device in text for device in by_path_devices)


def test_run_setup_declining_duplicate_device_reprompts_role(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "", "1", "1", "n", "2", "3"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert f"warning: {devices[0]} is already assigned to the top camera" in out.getvalue()
    assert any(
        "Use " in prompt and "for both top and left cameras? [y/N]" in prompt for prompt in prompts
    )
    assert sum(prompt.startswith("left camera") for prompt in prompts) == 2
    assert f"left_cam_device = {devices[1]}" in text


def test_run_setup_accepting_duplicate_device_assigns_both_roles(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "", "1", "1", "y", "2"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert f"top_cam_device = {devices[0]}" in text
    assert f"left_cam_device = {devices[0]}" in text
    assert f"right_cam_device = {devices[1]}" in text


def test_run_setup_enter_accepts_detected_current_camera_defaults(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id)
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[embodiment.args]\n"
        + "\n".join(
            f"{role}_cam_device = {device}"
            for role, device in zip(("top", "left", "right"), devices, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    input_fn, prompts = _scripted_input([""] * 10)

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    text = path.read_text(encoding="utf-8")
    role_prompts = [prompt for prompt in prompts if " camera — " in prompt]
    assert result == 0
    assert all(
        f"[{device} (current)]" in prompt
        for device, prompt in zip(devices, role_prompts, strict=True)
    )
    assert all(device in text for device in devices)


def test_run_setup_marks_undetected_current_camera_defaults(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    _make_devices(by_id)
    current_devices = ["/remote/top", "/remote/left", "/remote/right"]
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[embodiment.args]\n"
        + "\n".join(
            f"{role}_cam_device = {device}"
            for role, device in zip(("top", "left", "right"), current_devices, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    input_fn, prompts = _scripted_input([""] * 10)

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
    )

    role_prompts = [prompt for prompt in prompts if " camera — " in prompt]
    text = path.read_text(encoding="utf-8")
    assert result == 0
    assert all("(current, not detected)" in prompt for prompt in role_prompts)
    assert all(device in text for device in current_devices)


def test_render_config_comment_at_exact_boundary_never_glues(tmp_path: Path) -> None:
    policy = "policy-with-17chr"  # "policy = " + 17 chars == 26, the pad width
    assert len(f"policy = {policy}") == 26
    path = tmp_path / "config.ini"
    path.write_text(_render_config({"policy": policy}, {}, {}), encoding="utf-8")

    carried = _read_raw_config(path)

    assert not isinstance(carried, str)
    assert carried["defaults"]["policy"] == policy
    assert f"policy = {policy}  # " in path.read_text(encoding="utf-8")


def test_run_setup_multiline_prompted_default_still_parses(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\nscorer = line one\n    line two\n",
        encoding="utf-8",
    )
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    rewritten = _read_raw_config(path)
    assert result == 0
    assert not isinstance(rewritten, str)
    lines = [line.lstrip() for line in rewritten["defaults"]["scorer"].splitlines()]
    assert lines == ["line one", "line two"]


def test_run_setup_multiline_current_camera_still_parses(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\npolicy = molmoact2\n\n"
        "[embodiment.args]\n"
        "top_cam_device = /dev/one\n    /dev/one-continued\n"
        "left_cam_device = /dev/two\n"
        "right_cam_device = /dev/three\n",
        encoding="utf-8",
    )
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    rewritten = _read_raw_config(path)
    assert result == 0
    assert not isinstance(rewritten, str)
    lines = [line.lstrip() for line in rewritten["embodiment.args"]["top_cam_device"].splitlines()]
    assert lines == ["/dev/one", "/dev/one-continued"]


def test_run_setup_prints_intro_header_and_enter_hint(tmp_path: Path) -> None:
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = out.getvalue()
    assert result == 0
    assert f"inspect-robots setup — writes {_config_path(tmp_path)}" in text
    assert "press Enter to accept it, or type a replacement" in text
    assert "Found an existing config" not in text


def test_run_setup_intro_names_existing_config_and_backup(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text("[defaults]\npolicy = configured\n", encoding="utf-8")
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = out.getvalue()
    assert result == 0
    assert "Found an existing config; its values are the suggestions" in text
    assert "config.ini.bak" in text


class _TtyStringIO(io.StringIO):
    """StringIO that claims to be a terminal, to exercise the ANSI branch."""

    def isatty(self) -> bool:
        return True


def test_run_setup_paints_output_on_a_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "n"])
    out = _TtyStringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = out.getvalue()
    assert result == 0
    assert "\x1b[1minspect-robots setup\x1b[0m" in text
    assert f"\x1b[32mWrote {_config_path(tmp_path)}\x1b[0m" in text
    assert "\x1b[33m" in text  # headless note and unregistered warnings
    assert any("[\x1b[36mmolmoact2\x1b[0m]" in prompt for prompt in prompts)
    written = _config_path(tmp_path).read_text(encoding="utf-8")
    assert "\x1b[" not in written


def test_run_setup_honors_no_color_on_a_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "n"])
    out = _TtyStringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "\x1b[" not in out.getvalue()
    assert all("\x1b[" not in prompt for prompt in prompts)


def test_run_setup_repeats_plugin_reminder_after_writing(tmp_path: Path) -> None:
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = out.getvalue()
    assert result == 0
    reminder = (
        "reminder: policy 'molmoact2' and embodiment 'yam_arms' not registered "
        "here; install the plugin (e.g. `uv pip install inspect-robots-yam`) "
        "before your first run"
    )
    assert reminder in text
    assert text.index("Wrote ") < text.index("reminder: ")
    assert "runtime dependenc" not in text


def test_run_setup_reports_one_missing_runtime_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _PolicyFactory:
        pass

    class _EmbodimentFactory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "definitely_missing_xyz": "pip install thing"
        }

    def fake_registered(kind: str) -> dict[str, object]:
        if kind == "policy":
            return {"runtime-policy": _PolicyFactory}
        return {"runtime-body": _EmbodimentFactory}

    monkeypatch.setattr("inspect_robots.registry.registered", fake_registered)
    input_fn, _ = _scripted_input(["runtime-policy", "runtime-body", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = out.getvalue()
    assert result == 0
    assert "setup complete, but 1 runtime dependency is missing:" in text
    assert "  ✗ definitely_missing_xyz (runtime-body) → pip install thing" in text
    assert text.index("Wrote ") < text.index("1 runtime dependency is missing:")
    assert text.index("1 runtime dependency is missing:") < text.index("Next: ")


def test_run_setup_reports_multiple_missing_runtime_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _PolicyFactory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "definitely_missing_xyz_policy": "install policy dependency"
        }

    class _EmbodimentFactory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "definitely_missing_xyz_body": "install embodiment dependency"
        }

    def fake_registered(kind: str) -> dict[str, object]:
        if kind == "policy":
            return {"runtime-policy": _PolicyFactory}
        return {"runtime-body": _EmbodimentFactory}

    monkeypatch.setattr("inspect_robots.registry.registered", fake_registered)
    input_fn, _ = _scripted_input(["runtime-policy", "runtime-body", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    text = out.getvalue()
    assert result == 0
    assert "setup complete, but 2 runtime dependencies are missing:" in text
    assert "definitely_missing_xyz_policy (runtime-policy)" in text
    assert "definitely_missing_xyz_body (runtime-body)" in text


def test_run_setup_omits_runtime_checklist_when_requirements_are_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _PolicyFactory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {"os": "install os"}

    class _EmbodimentFactory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {"os": "install os"}

    def fake_registered(kind: str) -> dict[str, object]:
        if kind == "policy":
            return {"runtime-policy": _PolicyFactory}
        return {"runtime-body": _EmbodimentFactory}

    monkeypatch.setattr("inspect_robots.registry.registered", fake_registered)
    input_fn, _ = _scripted_input(["runtime-policy", "runtime-body", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "runtime dependenc" not in out.getvalue()


def test_run_setup_no_reminder_when_components_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inspect_robots.registry.registered",
        lambda kind: {"molmoact2": object(), "yam_arms": object()},
    )
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert "reminder:" not in out.getvalue()


def test_run_setup_reminder_names_only_the_missing_component(tmp_path: Path) -> None:
    input_fn, _ = _scripted_input(["", "", "", "", "", "", "n"])
    out = io.StringIO()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "inspect_robots.registry.registered",
            lambda kind: {"molmoact2": object()},
        )
        result = run_setup(
            {"XDG_CONFIG_HOME": str(tmp_path)},
            input_fn=input_fn,
            out=out,
            interactive=True,
            by_id_dir=tmp_path / "none-id",
            by_path_dir=tmp_path / "none-path",
        )

    text = out.getvalue()
    assert result == 0
    assert "reminder: embodiment 'yam_arms' not registered here" in text
    assert "policy 'molmoact2'" not in text.split("Wrote ")[1]


def test_run_setup_registered_device_slots_use_labels_and_can_number_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left arm CAN channel"),),
    )
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can_left")
    input_fn, prompts = _scripted_input([*_slot_defaults(), "", "1"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        sysfs_net=sysfs_net,
        serial_by_id_dir=tmp_path / "serial",
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert "Found 1 CAN interface(s) under /sys/class/net:" in out.getvalue()
    assert "  1. can_left" in out.getvalue()
    assert any(
        prompt == "left arm CAN channel — number, 'u' to identify by unplugging, 's' to skip: "
        for prompt in prompts
    )
    assert "left_channel = can_left" in text


@pytest.mark.parametrize("registered_without_slots", [False, True])
def test_run_setup_falls_back_to_cameras_without_registered_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_without_slots: bool,
) -> None:
    if registered_without_slots:

        class _Factory:
            pass

        monkeypatch.setattr(
            "inspect_robots.registry.registered",
            lambda kind: {"slot-body": _Factory} if kind == "embodiment" else {},
        )
    input_fn, prompts = _scripted_input([*_slot_defaults(), ""])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        by_id_dir=tmp_path / "none-id",
        by_path_dir=tmp_path / "none-path",
    )

    assert result == 0
    assert any(prompt.startswith("Configure cameras?") for prompt in prompts)
    assert not any(prompt.startswith("Configure devices?") for prompt in prompts)


def test_run_setup_device_gate_defaults_no_without_probe_or_existing_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left arm CAN channel"),),
    )
    input_fn, prompts = _scripted_input([*_slot_defaults(), ""])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        sysfs_net=tmp_path / "none-net",
    )

    assert result == 0
    assert "Configure devices? [y/N] " in prompts
    assert not any(prompt.startswith("left arm CAN channel") for prompt in prompts)


def test_run_setup_serial_slots_support_number_and_manual_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (
            DeviceSlot("controller_port", "serial", "controller serial port"),
            DeviceSlot("debug_port", "serial", "debug serial port"),
        ),
    )
    serial_by_id = tmp_path / "serial-by-id"
    serial_by_id.mkdir()
    listed = serial_by_id / "usb-controller"
    listed.touch()
    manual = "/remote/serial-controller"
    input_fn, _ = _scripted_input([*_slot_defaults(), "", "1", manual])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        serial_by_id_dir=serial_by_id,
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert out.getvalue().count("Found 1 serial device(s) under /dev/serial/by-id:") == 1
    assert f"controller_port = {listed}" in text
    assert f"debug_port = {manual}" in text
    assert f"warning: {manual} does not exist here" in out.getvalue()


def test_run_setup_slot_duplicate_guard_is_same_kind_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (
            DeviceSlot("top_cam", "v4l2", "top camera"),
            DeviceSlot("side_cam", "v4l2", "side camera"),
            DeviceSlot("controller_port", "serial", "controller serial port"),
        ),
    )
    by_id = tmp_path / "by-id"
    devices = _make_devices(by_id, count=2)
    serial_by_id = tmp_path / "serial"
    serial_by_id.mkdir()
    (serial_by_id / "listed").touch()
    input_fn, prompts = _scripted_input(
        [*_slot_defaults(), "", devices[0], devices[0], "n", devices[1], devices[0]]
    )
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=tmp_path / "none-path",
        serial_by_id_dir=serial_by_id,
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert f"warning: {devices[0]} is already assigned to top camera" in out.getvalue()
    assert (
        sum(
            "Use " in prompt and "for both top camera and side camera?" in prompt
            for prompt in prompts
        )
        == 1
    )
    assert f"top_cam = {devices[0]}" in text
    assert f"side_cam = {devices[1]}" in text
    assert f"controller_port = {devices[0]}" in text


def test_run_setup_can_manual_entry_rejects_paths_and_warns_when_unlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left arm CAN channel"),),
    )
    input_fn, prompts = _scripted_input([*_slot_defaults(), "y", "/dev/can0", "can_remote"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        sysfs_net=tmp_path / "none-net",
    )

    assert result == 0
    assert sum(prompt.startswith("left arm CAN channel") for prompt in prompts) == 2
    assert "enter an interface number, bare interface name" in out.getvalue()
    assert "warning: can_remote is not in the scanned CAN interfaces" in out.getvalue()
    assert "left_channel = can_remote" in _config_path(tmp_path).read_text(encoding="utf-8")


def test_run_setup_can_manual_entry_accepts_a_listed_name_without_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left CAN channel"),),
    )
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0")
    input_fn, _ = _scripted_input([*_slot_defaults(), "", "can0"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        sysfs_net=sysfs_net,
    )

    assert result == 0
    assert "not in the scanned CAN interfaces" not in out.getvalue()
    assert "left_channel = can0" in _config_path(tmp_path).read_text(encoding="utf-8")


def test_run_setup_can_slot_identifies_by_unplug_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left arm CAN channel"),),
    )
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0")
    pending = [*_slot_defaults(), "", "u", "", ""]
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if prompt.startswith("Unplug the left arm CAN channel"):
            (sysfs_net / "can0" / "type").write_text("1\n", encoding="utf-8")
        elif prompt.startswith("Plug it back in"):
            (sysfs_net / "can0" / "type").write_text("280\n", encoding="utf-8")
        return pending.pop(0)

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        sysfs_net=sysfs_net,
    )

    assert result == 0
    assert "Unplug the left arm CAN channel now, then press Enter..." in prompts
    assert "left_channel = can0" in _config_path(tmp_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("go_back", [False, True])
def test_run_setup_device_group_all_or_none_keeps_ungrouped_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    go_back: bool,
) -> None:
    _register_device_slots(
        monkeypatch,
        (
            DeviceSlot("left_channel", "can", "left CAN channel", "arms"),
            DeviceSlot("right_channel", "can", "right CAN channel", "arms"),
            DeviceSlot("controller_port", "serial", "controller serial port"),
        ),
    )
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0", "can1")
    serial_by_id = tmp_path / "serial"
    serial_by_id.mkdir()
    serial = serial_by_id / "controller"
    serial.touch()
    if not go_back:
        path = _config_path(tmp_path)
        path.parent.mkdir()
        path.write_text(
            "[defaults]\nembodiment = slot-body\n\n"
            "[embodiment.args]\nleft_channel = old-left\nright_channel = old-right\n",
            encoding="utf-8",
        )
    responses: list[str | BaseException] = [
        *_slot_defaults(),
        "",
        "1",
        "s",
        "1",
        "" if go_back else "n",
    ]
    if go_back:
        responses += ["1", "2"]
    input_fn, prompts = _scripted_input(responses)
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        sysfs_net=sysfs_net,
        serial_by_id_dir=serial_by_id,
    )

    text = _config_path(tmp_path).read_text(encoding="utf-8")
    assert result == 0
    assert (
        "slot-body needs all arms slots or none; writing none unless you go back" in out.getvalue()
    )
    assert f"controller_port = {serial}" in text
    assert sum(prompt.startswith("controller serial port") for prompt in prompts) == 1
    assert ("left_channel = can0" in text) is go_back
    assert ("right_channel = can1" in text) is go_back
    assert "old-left" not in text
    assert "old-right" not in text


def test_run_setup_declining_devices_preserves_slot_args_and_unmanaged_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left CAN channel"),),
    )
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\nembodiment = slot-body\n\n"
        "[embodiment.args]\nleft_channel = can9\ncalibration = preserved\n",
        encoding="utf-8",
    )
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "n"])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        sysfs_net=tmp_path / "none-net",
    )

    text = path.read_text(encoding="utf-8")
    assert result == 0
    assert "Configure devices? [Y/n] " in prompts
    assert "left_channel = can9" in text
    assert "calibration = preserved" in text
    assert not any(prompt.startswith("left CAN channel") for prompt in prompts)


def test_run_setup_slot_enter_accepts_current_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("left_channel", "can", "left CAN channel"),),
    )
    path = _config_path(tmp_path)
    path.parent.mkdir()
    path.write_text(
        "[defaults]\nembodiment = slot-body\n\n[embodiment.args]\nleft_channel = can0\n",
        encoding="utf-8",
    )
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0")
    input_fn, prompts = _scripted_input(["", "", "", "", "", "", "", ""])

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=io.StringIO(),
        interactive=True,
        sysfs_net=sysfs_net,
    )

    assert result == 0
    assert any("[can0 (current)]" in prompt for prompt in prompts)
    assert "left_channel = can0" in path.read_text(encoding="utf-8")


def test_run_setup_v4l2_slot_can_switch_to_by_path_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_device_slots(
        monkeypatch,
        (DeviceSlot("camera_device", "v4l2", "inspection camera"),),
    )
    by_id = tmp_path / "by-id"
    by_path = tmp_path / "by-path"
    _make_devices(by_id, count=1)
    by_path_devices = _make_devices(by_path, count=2)
    input_fn, prompts = _scripted_input([*_slot_defaults(), "", "p", "2"])
    out = io.StringIO()

    result = run_setup(
        {"XDG_CONFIG_HOME": str(tmp_path)},
        input_fn=input_fn,
        out=out,
        interactive=True,
        by_id_dir=by_id,
        by_path_dir=by_path,
    )

    assert result == 0
    assert "only 1 by-id entries for 2 detected cameras" in out.getvalue()
    assert any("'s' to skip, 'p' to switch listing" in prompt for prompt in prompts)
    assert f"camera_device = {by_path_devices[1]}" in _config_path(tmp_path).read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_suggest_can_pinning_prints_exact_rules_for_distinct_usb_serials(
    tmp_path: Path,
) -> None:
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0", "can1")
    _attach_can_adapter(tmp_path, sysfs_net, "can0", "3B004B")
    _attach_can_adapter(tmp_path, sysfs_net, "can1", "3B004C")
    slots = (
        DeviceSlot("can_left_channel", "can", "left CAN channel"),
        DeviceSlot("right_bus", "can", "right CAN bus"),
    )
    out = io.StringIO()

    _suggest_can_pinning(
        sysfs_net,
        slots,
        {"can_left_channel": "can0", "right_bus": "can1"},
        out=out,
    )

    assert out.getvalue() == (
        "these CAN interfaces have order-dependent names; a replug can swap them.\n"
        "pin them by adapter serial (paste into /etc/udev/rules.d/70-can-names.rules,\n"
        "then replug or reboot), and re-run setup to record the pinned names:\n"
        '  SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="3B004B", NAME="can_left"\n'
        '  SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="3B004C", NAME="can_right"\n'
    )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_suggest_can_pinning_includes_unassigned_order_dependent_interface(
    tmp_path: Path,
) -> None:
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0", "can1")
    _attach_can_adapter(tmp_path, sysfs_net, "can0", "LEFT")
    _attach_can_adapter(tmp_path, sysfs_net, "can1", "SPARE")
    slots = (DeviceSlot("left_channel", "can", "left CAN channel"),)
    out = io.StringIO()

    _suggest_can_pinning(sysfs_net, slots, {"left_channel": "can0"}, out=out)

    text = out.getvalue()
    assert 'ATTRS{serial}=="LEFT", NAME="can_left"' in text
    assert 'ATTRS{serial}=="SPARE", NAME="can_can1"' in text


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
@pytest.mark.parametrize("fallback_reason", ["derived", "scanned", "long"])
def test_suggest_can_pinning_name_conflict_or_length_falls_back_as_a_set(
    tmp_path: Path,
    fallback_reason: str,
) -> None:
    sysfs_net = tmp_path / "net"
    names = ("can0", "can1", "can_left") if fallback_reason == "scanned" else ("can0", "can1")
    _make_can_interfaces(sysfs_net, *names)
    _attach_can_adapter(tmp_path, sysfs_net, "can0", "ONE")
    _attach_can_adapter(tmp_path, sysfs_net, "can1", "TWO")
    slots: tuple[DeviceSlot, ...]
    assignments: dict[str, str]
    if fallback_reason == "derived":
        slots = (
            DeviceSlot("left_channel", "can", "left CAN channel"),
            DeviceSlot("left_bus", "can", "duplicate stable name"),
        )
        assignments = {"left_channel": "can0", "left_bus": "can1"}
    elif fallback_reason == "scanned":
        slots = (DeviceSlot("left_channel", "can", "left CAN channel"),)
        assignments = {"left_channel": "can0"}
    else:
        slots = (DeviceSlot("extraordinarily_long_channel", "can", "long CAN channel"),)
        assignments = {"extraordinarily_long_channel": "can0"}
    out = io.StringIO()

    _suggest_can_pinning(sysfs_net, slots, assignments, out=out)

    text = out.getvalue()
    assert 'ATTRS{serial}=="ONE", NAME="can_a"' in text
    assert 'ATTRS{serial}=="TWO", NAME="can_b"' in text
    assert 'NAME="can_can1"' not in text


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
@pytest.mark.parametrize("serials", [("SAME", "SAME"), ("ONLY", None)])
def test_suggest_can_pinning_bad_serials_print_warning_without_rules(
    tmp_path: Path,
    serials: tuple[str, str | None],
) -> None:
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0", "can1")
    _attach_can_adapter(tmp_path, sysfs_net, "can0", serials[0])
    _attach_can_adapter(tmp_path, sysfs_net, "can1", serials[1])
    slots = (DeviceSlot("left_channel", "can", "left CAN channel"),)
    out = io.StringIO()

    _suggest_can_pinning(sysfs_net, slots, {"left_channel": "can0"}, out=out)

    assert out.getvalue() == (
        "these CAN interfaces have order-dependent names; a replug can swap them.\n"
    )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_suggest_can_pinning_non_usb_interfaces_are_completely_silent(
    tmp_path: Path,
) -> None:
    sysfs_net = tmp_path / "net"
    _make_can_interfaces(sysfs_net, "can0", "can1")
    _attach_can_adapter(tmp_path, sysfs_net, "can0", "ONE", usb=False)
    _attach_can_adapter(tmp_path, sysfs_net, "can1", "TWO", usb=False)
    slots = (DeviceSlot("left_channel", "can", "left CAN channel"),)
    out = io.StringIO()

    _suggest_can_pinning(sysfs_net, slots, {"left_channel": "can0"}, out=out)

    assert out.getvalue() == ""


def test_suggest_can_pinning_pinned_names_or_no_assigned_kernel_name_are_silent(
    tmp_path: Path,
) -> None:
    pinned_net = tmp_path / "pinned-net"
    _make_can_interfaces(pinned_net, "can_left", "can_right")
    slots = (DeviceSlot("left_channel", "can", "left CAN channel"),)
    pinned_out = io.StringIO()

    _suggest_can_pinning(pinned_net, slots, {"left_channel": "can_left"}, out=pinned_out)

    order_net = tmp_path / "order-net"
    _make_can_interfaces(order_net, "can0", "can1")
    unassigned_out = io.StringIO()
    _suggest_can_pinning(order_net, slots, {"left_channel": "can9"}, out=unassigned_out)
    assert pinned_out.getvalue() == ""
    assert unassigned_out.getvalue() == ""
