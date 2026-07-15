"""Render stored camera frames to per-stream MP4 videos via the ffmpeg binary.

`FrameStore` persists raw ``.npy`` arrays; this module reunites them with a
log and pipes them one frame at a time to an external ``ffmpeg`` process, so
the core gains video export with no new Python dependencies and peak memory
of a single frame. Design and failure taxonomy: plans/0016-frame-video-export.md.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

# Everything before the trailing _NNNNNN step token FrameStore appended.
# Trial ids and camera names may both contain "_", so the (trial, camera)
# split is ambiguous — the whole prefix never needs splitting. \d{6,} keeps
# t >= 10**6 (overflowing the 06d pad) grouping and sorting correctly.
_FRAME_RE = re.compile(r"^(.+)_(\d{6,})\.npy$")

_STDERR_TAIL_LINES = 5
DEFAULT_FPS = 10.0


class _FrameError(Exception):
    """A single frame file could not be loaded or is not renderable."""


@dataclass(frozen=True)
class StreamResult:
    """Outcome of encoding one (trial, camera) stream."""

    piped: int
    skipped_empty: int
    error: str | None


def frames_dir_candidates(frames_dir: str, log_path: Path) -> tuple[Path, Path]:
    """The two places a log's frames directory can be, in resolution order.

    ``frames_dir`` is stored as configured at run time (typically relative to
    the run's CWD). First candidate: the string as-is. Second: derived from
    the log's location — the log lives directly in ``<log-dir>/`` and frames
    in ``<log-dir>/frames/<stamp>``, so the log's parent is the log dir at
    any ``--log-dir`` depth. A log written on Windows stores backslashes,
    which POSIX ``Path.name`` would not split.
    """
    stamp = PureWindowsPath(frames_dir).name if "\\" in frames_dir else Path(frames_dir).name
    return Path(frames_dir), log_path.parent / "frames" / stamp


def resolve_frames_dir(frames_dir: str, log_path: Path) -> Path | None:
    """Resolve a stored frames-directory string to an existing directory.

    Returns ``None`` when neither candidate exists (moved machine, different
    CWD) — callers decide whether that is an error (``video``) or a note
    (``inspect``).
    """
    for candidate in frames_dir_candidates(frames_dir, log_path):
        if candidate.is_dir():
            return candidate
    return None


def discover_streams(root: Path) -> tuple[dict[str, list[tuple[int, Path]]], list[Path]]:
    """Group ``root``'s frame files into per-stream step-ordered lists.

    Returns ``(streams, strays)``: streams keyed by filename prefix in sorted
    order (deterministic output; ``glob`` order is OS-dependent), each a list
    of ``(step, path)`` sorted numerically by step, plus the ``.npy`` files
    that do not match the FrameStore pattern. Non-``.npy`` entries (such as
    this module's own ``.mp4`` outputs on a second run) are ignored entirely.
    """
    streams: dict[str, list[tuple[int, Path]]] = {}
    strays: list[Path] = []
    for path in sorted(root.glob("*.npy")):
        match = _FRAME_RE.match(path.name)
        if match is None:
            strays.append(path)
            continue
        streams.setdefault(match.group(1), []).append((int(match.group(2)), path))
    for frames in streams.values():
        frames.sort()
    return dict(sorted(streams.items())), strays


def count_frames(root: Path) -> int:
    """How many pattern-matching frame files ``root`` holds.

    The same enumeration ``video`` errors on when zero, so hint gates and the
    error gate can never disagree.
    """
    streams, _ = discover_streams(root)
    return sum(len(frames) for frames in streams.values())


def default_fps(embodiment_info: Mapping[str, Any]) -> tuple[float, str]:
    """The playback rate a log implies, with its source for display.

    Uses the embodiment's nominal ``control_hz`` (frames are stored once per
    rollout step, so it is the best proxy the log offers). The guards mirror
    ``_print_step_limit_notice`` — numeric, not bool, > 0 — plus finite:
    plain ``json.load`` accepts the ``Infinity`` literal, and a hand-edited
    ``control_hz`` must fall back rather than become ``-r inf``.
    """
    rate = embodiment_info.get("control_hz")
    if (
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and rate > 0
        and math.isfinite(rate)
    ):
        return float(rate), "control_hz from log"
    return DEFAULT_FPS, "default"


def _normalize(path: Path) -> npt.NDArray[np.uint8] | None:
    """Load one frame and normalize it to contiguous ``(H, W, 3)`` uint8.

    ``None`` means an empty array: expected first-party data (the isaacsim
    adapter passes render warm-up frames through empty), skipped rather than
    failed. Dtype is strictly uint8 — deliberately diverging from
    ``FrameRef.load``'s coercion, which would turn 0-1 floats into silently
    black video. Any load failure (a truncated file from an interrupted run
    is the expected corrupt artifact here) or unsupported shape raises
    ``_FrameError`` naming the file.
    """
    try:
        array = np.load(path)
    except Exception as exc:
        # Broad on purpose: chopped files raise ValueError, EOFError, or
        # OSError depending on where the truncation landed.
        raise _FrameError(f"unreadable frame {path.name}: {exc}") from exc
    if array.size == 0:
        return None
    if array.dtype != np.uint8:
        raise _FrameError(f"unsupported dtype {array.dtype} in {path.name} (frames are uint8)")
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    elif not (array.ndim == 3 and array.shape[2] == 3):
        raise _FrameError(f"unsupported shape {array.shape} in {path.name}")
    return np.ascontiguousarray(array)


def _ffmpeg_argv(ffmpeg: str, width: int, height: int, fps: float, out_path: Path) -> list[str]:
    """The pinned encode command for one rawvideo stream.

    ``libx264`` is pinned rather than trusting the build's mp4 default: an
    LGPL ffmpeg without it silently falls back to mpeg4, which browsers
    won't play — pinning turns that into a loud per-stream failure. The pad
    filter keeps odd dimensions legal for yuv420p.
    """
    return [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]


def _stderr_tail(fd: int, name: str) -> str:
    """Read the last lines of ffmpeg's stderr temp file and unlink it.

    Reading only after the child exits guarantees the tail is complete and
    stays clear of Windows sharing quirks; the parent's descriptor and the
    read handle are both closed before the unlink (Windows again).
    """
    os.close(fd)
    try:
        with open(name, encoding="utf-8", errors="replace") as reopened:
            lines = reopened.read().strip().splitlines()
    finally:
        os.unlink(name)
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def encode_stream(
    frames: Sequence[tuple[int, Path]], out_path: Path, fps: float, ffmpeg: str
) -> StreamResult:
    """Pipe one stream's frames through ffmpeg into ``out_path``.

    Frames are loaded one at a time, so peak memory is a single frame. On any
    failure the partial output is unlinked (``missing_ok`` — ffmpeg may die
    before creating it) after the process is dead, and ``error`` carries
    either the offending file or ffmpeg's stderr tail. ``Popen`` itself
    failing is a hard ``SystemExit``: unlike per-stream failures it would
    repeat identically for every stream.
    """
    skipped = 0
    index = 0
    first: npt.NDArray[np.uint8] | None = None
    try:
        # Pre-spawn probe: the argv needs -s WxH, so scan forward past empty
        # frames to the first usable one before any process exists. Failures
        # here have no process to kill and nothing to clean.
        while index < len(frames):
            candidate = _normalize(frames[index][1])
            index += 1
            if candidate is None:
                skipped += 1
                continue
            first = candidate
            break
    except _FrameError as exc:
        return StreamResult(piped=0, skipped_empty=skipped, error=str(exc))
    if first is None:
        return StreamResult(piped=0, skipped_empty=skipped, error="no usable frames")

    height, width = first.shape[0], first.shape[1]
    # stderr goes to a temp file, never a pipe: unread PIPE buffers fill and
    # deadlock ffmpeg against our stdin writes on exactly the long episodes
    # this tool exists for.
    stderr_fd, stderr_name = tempfile.mkstemp(suffix=".ffmpeg.log")
    try:
        proc = subprocess.Popen(
            _ffmpeg_argv(ffmpeg, width, height, fps, out_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fd,
        )
    except OSError as exc:
        os.close(stderr_fd)
        os.unlink(stderr_name)
        raise SystemExit(f"could not launch ffmpeg ({ffmpeg}): {exc}") from exc

    piped = 0
    error: str | None = None
    broken_pipe = False
    stdin = cast("IO[bytes]", proc.stdin)
    try:
        try:
            stdin.write(first.tobytes())
            piped += 1
            for _step, path in frames[index:]:
                frame = _normalize(path)
                if frame is None:
                    skipped += 1
                    continue
                if frame.shape != first.shape:
                    raise _FrameError(
                        f"frame shape changed from {first.shape} to {frame.shape} at {path.name}"
                    )
                stdin.write(frame.tobytes())
                piped += 1
        except _FrameError as exc:
            error = str(exc)
            proc.kill()
        except OSError:
            # ffmpeg died mid-stream; its stderr tail is the real complaint.
            broken_pipe = True
        try:
            # A buffered tail can surface the broken pipe at close, not
            # write — including after our own kill() above.
            stdin.close()
        except OSError:
            broken_pipe = True
        returncode = proc.wait()
    finally:
        # Also reached when an unanticipated exception escapes (Ctrl-C
        # mid-pipe, MemoryError): the temp file is unlinked on every path.
        tail = _stderr_tail(stderr_fd, stderr_name)
    if error is None and (broken_pipe or returncode != 0):
        error = tail or f"ffmpeg exited with code {returncode}"
    if error is not None:
        out_path.unlink(missing_ok=True)
    return StreamResult(piped=piped, skipped_empty=skipped, error=error)
