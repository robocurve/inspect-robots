"""Tests for frame-to-video export (plan 0016): _video.py and the video CLI."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import inspect_robots.cli as cli
from inspect_robots._video import (
    StreamResult,
    count_frames,
    default_fps,
    discover_streams,
    encode_stream,
    frames_dir_candidates,
    resolve_frames_dir,
)
from inspect_robots.cli import main
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult

# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #

# Non-square on purpose: a swapped -s WxH cannot pass with H != W.
_H, _W = 4, 6


def _rgb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(_H, _W, 3), dtype=np.uint8)


def _write_frames(root: Path, prefix: str, arrays: list[np.ndarray]) -> list[tuple[int, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for t, arr in enumerate(arrays):
        path = root / f"{prefix}_{t:06d}.npy"
        np.save(path, arr)
        out.append((t, path))
    return out


def _truncate(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _frames_log(frames_dir: str | None, control_hz: object = 10.0) -> EvalLog:
    return EvalLog(
        version=1,
        status="success",
        eval=EvalSpec(
            task="adhoc",
            policy="p",
            embodiment="e",
            created="x",
            inspect_robots_version="0",
            embodiment_info={"control_hz": control_hz},
        ),
        results=EvalResults(total_scenes=1, total_trials=1, metrics={}),
        stats=EvalStats(
            started_at="a",
            completed_at="b",
            duration_s=0.0,
            total_steps=1,
            frames_dir=frames_dir,
        ),
        samples=(SceneResult(scene_id="s0", status="success", epochs=({},)),),
    )


def _write_log(tmp_path: Path, log: EvalLog, name: str = "run.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")
    return path


class _FakeStdin:
    """Pipe stand-in that records bytes and can break on write or close."""

    def __init__(self, fail_on_write_after: int | None, fail_at_close: bool) -> None:
        self.piped = bytearray()
        self.writes = 0
        self._fail_on_write_after = fail_on_write_after
        self._fail_at_close = fail_at_close

    def write(self, data: bytes) -> int:
        if self._fail_on_write_after is not None and self.writes >= self._fail_on_write_after:
            raise BrokenPipeError
        self.writes += 1
        self.piped.extend(data)
        return len(data)

    def close(self) -> None:
        if self._fail_at_close:
            raise BrokenPipeError


class _FakePopen:
    """Records argv, consumes stdin, and writes stderr through the real fd."""

    calls: ClassVar[list[_FakePopen]] = []
    returncode_next = 0
    stderr_text_next = ""
    fail_on_write_after: ClassVar[int | None] = None
    fail_at_close = False

    def __init__(self, argv: list[str], stdin: Any, stdout: Any, stderr: int) -> None:
        self.argv = argv
        self.stdout = stdout
        self._stderr_fd = stderr
        self.stdin = _FakeStdin(_FakePopen.fail_on_write_after, _FakePopen.fail_at_close)
        self.killed = False
        _FakePopen.calls.append(self)

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        # Like a real child: stderr text lands in the file the parent passed.
        if _FakePopen.stderr_text_next:
            os.write(self._stderr_fd, _FakePopen.stderr_text_next.encode())
        return 1 if self.killed else _FakePopen.returncode_next


@pytest.fixture()
def fake_popen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_FakePopen]:
    _FakePopen.calls = []
    _FakePopen.returncode_next = 0
    _FakePopen.stderr_text_next = ""
    _FakePopen.fail_on_write_after = None
    _FakePopen.fail_at_close = False
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    # Keep ffmpeg's stderr temp files inside tmp_path so leaks are assertable.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return _FakePopen


def _no_temp_leak(tmp_path: Path) -> bool:
    return not list(tmp_path.glob("*.ffmpeg.log"))


# --------------------------------------------------------------------------- #
# Discovery, grouping, resolution, fps
# --------------------------------------------------------------------------- #


def test_discovery_groups_sorts_and_scopes_to_npy(tmp_path: Path) -> None:
    _write_frames(tmp_path, "b_cam", [_rgb(0)])
    # Written out of order; also a 7-digit step to pin numeric sorting.
    np.save(tmp_path / "a_left_cam_000010.npy", _rgb(1))
    np.save(tmp_path / "a_left_cam_000002.npy", _rgb(2))
    np.save(tmp_path / "a_left_cam_1000000.npy", _rgb(3))
    (tmp_path / "notes.npy").write_bytes(b"not a frame")
    (tmp_path / "a_left_cam.mp4").write_bytes(b"previous output")

    streams, strays = discover_streams(tmp_path)

    assert list(streams) == ["a_left_cam", "b_cam"]
    assert [t for t, _ in streams["a_left_cam"]] == [2, 10, 1000000]
    assert [p.name for p in strays] == ["notes.npy"]
    assert count_frames(tmp_path) == 4


def test_frames_dir_resolution_as_is_fallback_miss_and_backslashes(tmp_path: Path) -> None:
    log_path = tmp_path / "out" / "logs" / "run.json"
    stamp_dir = tmp_path / "out" / "logs" / "frames" / "stamp"
    stamp_dir.mkdir(parents=True)

    # As-is hit (relative to CWD it's absolute here).
    assert resolve_frames_dir(str(stamp_dir), log_path) == stamp_dir
    # Fallback hit: the stored string doesn't exist from this CWD, but the
    # log's parent is the log dir by construction.
    assert resolve_frames_dir("elsewhere/logs/frames/stamp", log_path) == stamp_dir
    # Double miss.
    assert resolve_frames_dir("elsewhere/logs/frames/other", log_path) is None
    # A Windows-written log stores backslashes; POSIX Path.name would return
    # the whole string and miss the fallback.
    assert resolve_frames_dir(r"logs\frames\stamp", log_path) == stamp_dir
    _, fallback = frames_dir_candidates(r"logs\frames\stamp", log_path)
    assert fallback == stamp_dir


@pytest.mark.parametrize(
    ("control_hz", "expected"),
    [
        (10.0, (10.0, "control_hz from log")),
        (12.5, (12.5, "control_hz from log")),
        (None, (10.0, "default")),
        ("fast", (10.0, "default")),
        (True, (10.0, "default")),
        (0, (10.0, "default")),
        (-5, (10.0, "default")),
        (float("inf"), (10.0, "default")),
    ],
)
def test_default_fps_guards(control_hz: object, expected: tuple[float, str]) -> None:
    info = {} if control_hz is None else {"control_hz": control_hz}
    assert default_fps(info) == expected


# --------------------------------------------------------------------------- #
# encode_stream
# --------------------------------------------------------------------------- #


def test_encode_pins_argv_and_pipes_exact_bytes(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0), _rgb(1)])
    out = tmp_path / "s.mp4"

    result = encode_stream(frames, out, 12.5, "/fake/ffmpeg")

    assert result == StreamResult(piped=2, skipped_empty=0, error=None)
    (proc,) = fake_popen.calls
    argv = proc.argv
    assert argv[0] == "/fake/ffmpeg"
    assert argv[-1] == str(out)
    # The full pinned encode decision: a refactor must not silently drop it.
    for flag, value in [
        ("-s", f"{_W}x{_H}"),
        ("-r", "12.5"),
        ("-loglevel", "error"),
        ("-f", "rawvideo"),
        ("-c:v", "libx264"),
        ("-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert argv[argv.index("-pix_fmt") + 1] == "rgb24"  # input pix_fmt
    assert argv[-3:-1] == ["-pix_fmt", "yuv420p"]  # output pix_fmt
    assert proc.stdout is subprocess.DEVNULL
    expected = np.load(frames[0][1]).tobytes() + np.load(frames[1][1]).tobytes()
    assert bytes(proc.stdin.piped) == expected
    assert _no_temp_leak(tmp_path)


def test_encode_expands_grayscale_and_drops_alpha(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    gray2d = np.arange(_H * _W, dtype=np.uint8).reshape(_H, _W)
    gray3d = gray2d[..., None]
    rgba = np.dstack([_rgb(0), np.full((_H, _W), 7, dtype=np.uint8)])
    frames = _write_frames(tmp_path / "f", "s", [gray2d, gray3d, rgba])

    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")

    assert result.error is None
    assert result.piped == 3
    (proc,) = fake_popen.calls
    gray_rgb = np.stack([gray2d] * 3, axis=-1).tobytes()
    assert bytes(proc.stdin.piped) == gray_rgb + gray_rgb + _rgb(0).tobytes()


def test_encode_skips_empty_frames_and_probes_forward(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    # Empty warm-up frames are first-party data (isaacsim passes them
    # through); dtype is meaningless when empty, so float64 empties skip too.
    empty = np.empty((0, 0, 3), dtype=np.float64)
    frames = _write_frames(tmp_path / "f", "s", [empty, _rgb(0), empty])

    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")

    assert result == StreamResult(piped=1, skipped_empty=2, error=None)
    (proc,) = fake_popen.calls
    assert proc.argv[proc.argv.index("-s") + 1] == f"{_W}x{_H}"


def test_encode_all_empty_stream_fails_without_spawning(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    empty = np.empty((0,), dtype=np.uint8)
    frames = _write_frames(tmp_path / "f", "s", [empty, empty])
    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert result == StreamResult(piped=0, skipped_empty=2, error="no usable frames")
    assert fake_popen.calls == []


def test_encode_truncated_first_frame_fails_pre_spawn(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0), _rgb(1)])
    _truncate(frames[0][1])
    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert result.piped == 0
    assert result.error is not None and "unreadable frame s_000000.npy" in result.error
    assert fake_popen.calls == []
    assert _no_temp_leak(tmp_path)


def test_encode_truncated_mid_stream_kills_and_unlinks(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0), _rgb(1)])
    _truncate(frames[1][1])
    out = tmp_path / "s.mp4"
    out.write_bytes(b"partial")  # what a real ffmpeg would have created

    result = encode_stream(frames, out, 10.0, "/fake/ffmpeg")

    assert result.error is not None and "s_000001.npy" in result.error
    (proc,) = fake_popen.calls
    assert proc.killed
    assert not out.exists()
    assert _no_temp_leak(tmp_path)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((_H, _W, 2), dtype=np.uint8),  # unsupported channel count
        np.zeros((_H, _W, 3), dtype=np.float64),  # non-uint8 dtype
    ],
)
def test_encode_rejects_unsupported_frames_mid_stream(
    tmp_path: Path, fake_popen: type[_FakePopen], bad: np.ndarray
) -> None:
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0), bad])
    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert result.error is not None and "unsupported" in result.error
    (proc,) = fake_popen.calls
    assert proc.killed


def test_encode_shape_change_mid_stream_names_the_file(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    small = np.zeros((2, 3, 3), dtype=np.uint8)
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0), small])
    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert result.error is not None
    assert "shape changed" in result.error and "s_000001.npy" in result.error


def test_encode_nonzero_exit_reports_stderr_tail(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    fake_popen.returncode_next = 1
    fake_popen.stderr_text_next = "Unknown encoder 'libx264'\n"
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0)])
    out = tmp_path / "s.mp4"
    out.write_bytes(b"partial")

    result = encode_stream(frames, out, 10.0, "/fake/ffmpeg")

    assert result.error == "Unknown encoder 'libx264'"
    assert not out.exists()
    assert _no_temp_leak(tmp_path)


def test_encode_nonzero_exit_with_silent_stderr_reports_code(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    # A SIGKILLed ffmpeg (OOM killer) writes nothing: never a blank reason.
    fake_popen.returncode_next = -9
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0)])
    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert result.error == "ffmpeg exited with code -9"


def test_encode_broken_pipe_on_write_reports_stderr_tail(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    fake_popen.fail_on_write_after = 1
    fake_popen.returncode_next = 1
    fake_popen.stderr_text_next = "No space left on device\n"
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0), _rgb(1)])

    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")

    assert result.error == "No space left on device"
    assert result.piped == 1


def test_encode_broken_pipe_at_close_is_caught(
    tmp_path: Path, fake_popen: type[_FakePopen]
) -> None:
    # BufferedWriter can surface the broken pipe only at close/flush.
    fake_popen.fail_at_close = True
    fake_popen.returncode_next = 1
    fake_popen.stderr_text_next = "moov atom write failed\n"
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0)])
    result = encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert result.error == "moov atom write failed"


def test_encode_popen_raising_is_a_hard_exit_without_temp_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("bad shebang")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frames = _write_frames(tmp_path / "f", "s", [_rgb(0)])

    with pytest.raises(SystemExit, match="could not launch ffmpeg"):
        encode_stream(frames, tmp_path / "s.mp4", 10.0, "/fake/ffmpeg")
    assert _no_temp_leak(tmp_path)


# --------------------------------------------------------------------------- #
# The video subcommand
# --------------------------------------------------------------------------- #


def _which_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/ffmpeg")


def test_video_end_to_end_writes_streams_and_summary(
    tmp_path: Path,
    fake_popen: type[_FakePopen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _which_fake(monkeypatch)
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "scene-0-e0_left_cam", [_rgb(0), _rgb(1)])
    _write_frames(frames_root, "scene-0-e0_right_cam", [_rgb(2)])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))

    assert main(["video", str(log_path)]) == 0

    out = capsys.readouterr().out
    assert "fps: 10 (control_hz from log)" in out
    assert f"wrote {frames_root / 'scene-0-e0_left_cam.mp4'} (2 frames)" in out
    assert f"wrote {frames_root / 'scene-0-e0_right_cam.mp4'} (1 frames)" in out
    assert "wrote 2/2 streams" in out
    assert len(fake_popen.calls) == 2


def test_video_failure_isolation_and_exit_code(
    tmp_path: Path,
    fake_popen: type[_FakePopen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Loop-continue and exit aggregation are behaviors branch coverage alone
    # cannot force: the first stream fails, the second must still encode.
    _which_fake(monkeypatch)
    frames_root = tmp_path / "frames"
    bad = _write_frames(frames_root, "a_cam", [_rgb(0)])
    _write_frames(frames_root, "b_cam", [_rgb(1)])
    _truncate(bad[0][1])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))

    assert main(["video", str(log_path)]) == 1

    captured = capsys.readouterr()
    assert "failed: a_cam: unreadable frame" in captured.err
    assert f"wrote {frames_root / 'b_cam.mp4'} (1 frames)" in captured.out
    assert "wrote 1/2 streams, 1 failed" in captured.out


def test_video_warns_about_stray_npy_and_empty_skips_on_stderr(
    tmp_path: Path,
    fake_popen: type[_FakePopen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _which_fake(monkeypatch)
    frames_root = tmp_path / "frames"
    empty = np.empty((0,), dtype=np.uint8)
    _write_frames(frames_root, "cam", [empty, _rgb(0)])
    np.save(frames_root / "notes.npy", _rgb(1))
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))

    assert main(["video", str(log_path)]) == 0

    err = capsys.readouterr().err
    assert "warning: skipping notes.npy" in err
    assert "warning: cam: skipped 1 empty frames" in err


def test_video_requires_stored_frames(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_path = _write_log(tmp_path, _frames_log(None))
    with pytest.raises(SystemExit, match="no stored frames"):
        main(["video", str(log_path)])


def test_video_unresolvable_frames_dir_lists_both_candidates(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path, _frames_log("gone/frames/stamp"))
    with pytest.raises(SystemExit, match=r"tried .*gone/frames/stamp.*frames/stamp"):
        main(["video", str(log_path)])


def test_video_empty_frames_dir_errors(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    np.save(frames_root / "notes.npy", _rgb(0))  # stray only: still "no frames"
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))
    with pytest.raises(SystemExit, match="no frames found"):
        main(["video", str(log_path)])


def test_video_fps_override_and_validation(
    tmp_path: Path,
    fake_popen: type[_FakePopen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _which_fake(monkeypatch)
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))

    assert main(["video", str(log_path), "--fps", "12.5"]) == 0
    assert "fps: 12.5 (--fps)" in capsys.readouterr().out
    (proc,) = fake_popen.calls
    assert proc.argv[proc.argv.index("-r") + 1] == "12.5"

    for bad in ["0", "-1", "inf", "nan"]:
        with pytest.raises(SystemExit, match="positive finite"):
            main(["video", str(log_path), "--fps", bad])


def test_video_hand_edited_infinite_control_hz_falls_back(
    tmp_path: Path,
    fake_popen: type[_FakePopen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # json.dumps happily emits the Infinity literal and json.load accepts it;
    # the sink's sanitizer only protects logs it wrote.
    _which_fake(monkeypatch)
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root), control_hz=float("inf")))

    assert main(["video", str(log_path)]) == 0
    assert "fps: 10 (default)" in capsys.readouterr().out


def test_video_ffmpeg_path_validation(
    tmp_path: Path, fake_popen: type[_FakePopen], monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))

    with pytest.raises(SystemExit, match="not an executable file"):
        main(["video", str(log_path), "--ffmpeg", str(tmp_path / "missing")])
    # A directory passes os.access(..., X_OK); isfile must reject it.
    with pytest.raises(SystemExit, match="not an executable file"):
        main(["video", str(log_path), "--ffmpeg", str(tmp_path)])
    # Present but not executable: monkeypatched os.access, not chmod, which
    # is meaningless on Windows (X_OK is true for any existing file there).
    stub = tmp_path / "ffmpeg-stub"
    stub.write_text("#!/bin/sh\n")
    with pytest.MonkeyPatch.context() as access_patch:
        access_patch.setattr(os, "access", lambda path, mode: False)
        with pytest.raises(SystemExit, match="not an executable file"):
            main(["video", str(log_path), "--ffmpeg", str(stub)])

    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    assert main(["video", str(log_path), "--ffmpeg", str(stub)]) == 0
    (proc,) = _FakePopen.calls
    assert proc.argv[0] == str(stub)


def test_video_missing_ffmpeg_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="ffmpeg not found on PATH"):
        main(["video", str(log_path)])


def test_video_out_dir_created_and_file_collision_rejected(
    tmp_path: Path,
    fake_popen: type[_FakePopen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _which_fake(monkeypatch)
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))

    out_dir = tmp_path / "videos" / "nested"
    assert main(["video", str(log_path), "--out", str(out_dir)]) == 0
    assert f"wrote {out_dir / 'cam.mp4'}" in capsys.readouterr().out

    collision = tmp_path / "afile"
    collision.write_text("x")
    with pytest.raises(SystemExit, match="not a directory"):
        main(["video", str(log_path), "--out", str(collision)])


# --------------------------------------------------------------------------- #
# inspect and run-summary integration
# --------------------------------------------------------------------------- #


def test_inspect_shows_frames_line_and_video_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0), _rgb(1)])
    log_path = _write_log(tmp_path, _frames_log("frames"))

    assert main(["inspect", str(log_path)]) == 0

    out = capsys.readouterr().out
    assert "frames:      frames (2 frames)\n" in out
    assert f"hint: render videos with: inspect-robots video {log_path}" in out
    assert out.index("scenes:") < out.index("frames:") < out.index("metrics:")


def test_inspect_frames_line_uses_resolved_fallback_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The stored (run-CWD-relative) string does not resolve from here, but
    # the log-adjacent fallback does — print the path that actually works.
    frames_root = tmp_path / "frames" / "stamp"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log_path = _write_log(tmp_path, _frames_log("elsewhere/frames/stamp"))
    assert main(["inspect", str(log_path)]) == 0
    assert f"frames:      {frames_root} (1 frames)" in capsys.readouterr().out


def test_inspect_frames_dir_unresolvable_notes_and_skips_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = _write_log(tmp_path, _frames_log("gone/frames/stamp"))
    assert main(["inspect", str(log_path)]) == 0
    out = capsys.readouterr().out
    assert "frames:      gone/frames/stamp (not found from this directory)" in out
    assert "hint: render videos" not in out


def test_inspect_zero_frames_suppresses_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A camera-less --store-frames run records a frames_dir but writes no
    # frames; the hint must not point at a command that would error.
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    log_path = _write_log(tmp_path, _frames_log(str(frames_root)))
    assert main(["inspect", str(log_path)]) == 0
    out = capsys.readouterr().out
    assert f"frames:      {frames_root} (0 frames)" in out
    assert "hint: render videos" not in out


def test_inspect_without_frames_dir_prints_no_frames_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = _write_log(tmp_path, _frames_log(None))
    assert main(["inspect", str(log_path)]) == 0
    assert "frames:" not in capsys.readouterr().out


def test_run_summary_video_hint_gated_on_frames_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_root = tmp_path / "frames"
    _write_frames(frames_root, "cam", [_rgb(0)])
    log = _frames_log(str(frames_root))
    cli._print_run_summary(log, str(tmp_path / "run.json"), is_adhoc=True)
    assert "hint: render videos with: inspect-robots video" in capsys.readouterr().out

    empty_root = tmp_path / "empty-frames"
    empty_root.mkdir()
    cli._print_run_summary(_frames_log(str(empty_root)), str(tmp_path / "run.json"), is_adhoc=True)
    assert "hint: render videos" not in capsys.readouterr().out
