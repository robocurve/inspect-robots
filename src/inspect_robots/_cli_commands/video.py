"""The ``inspect-robots video`` command: render a log's stored camera frames to one
MP4 per (trial, camera) stream via the ffmpeg binary.."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path


def _cmd_video(args: argparse.Namespace) -> int:
    """Render a log's stored frames to one MP4 per (trial, camera) stream.

    Per-stream failures are isolated: each is reported on stderr, the
    remaining streams still encode, and the exit code is 1 if any failed.
    Results (``wrote ...`` lines, the fps note, the final summary) go to
    stdout; warnings and failure reports go to stderr.
    """
    from inspect_robots import read_eval_log
    from inspect_robots._video import (
        default_fps,
        discover_streams,
        encode_stream,
        frames_dir_candidates,
        resolve_frames_dir,
    )

    log = read_eval_log(args.log)
    frames_dir = log.stats.frames_dir
    if frames_dir is None:
        raise SystemExit("this log has no stored frames (re-run with --store-frames)")
    log_path = Path(args.log)
    root = resolve_frames_dir(frames_dir, log_path)
    if root is None:
        as_is, fallback = frames_dir_candidates(frames_dir, log_path)
        raise SystemExit(f"frames directory not found; tried {as_is} and {fallback}")

    streams, strays = discover_streams(root)
    for stray in strays:
        print(
            f"warning: skipping {stray.name}: does not match the frame filename pattern",
            file=sys.stderr,
        )
    if not streams:
        raise SystemExit(f"no frames found in {root}")

    if args.fps is not None:
        if not (math.isfinite(args.fps) and args.fps > 0):
            raise SystemExit("--fps must be a positive finite number")
        fps, fps_source = args.fps, "--fps"
    else:
        fps, fps_source = default_fps(log.eval.embodiment_info)

    if args.ffmpeg is not None:
        if not (os.path.isfile(args.ffmpeg) and os.access(args.ffmpeg, os.X_OK)):
            raise SystemExit(f"--ffmpeg {args.ffmpeg} is not an executable file")
        ffmpeg = args.ffmpeg
    else:
        which = shutil.which("ffmpeg")
        if which is None:
            raise SystemExit(
                "ffmpeg not found on PATH; install it (e.g. apt install ffmpeg) "
                "or pass --ffmpeg PATH"
            )
        ffmpeg = which

    out_dir = root if args.out is None else Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"--out {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    # All validation is done: only now may result-looking stdout appear.
    print(f"fps: {fps:g} ({fps_source})")
    failed = 0
    for prefix, frames in streams.items():
        out_path = out_dir / f"{prefix}.mp4"
        result = encode_stream(frames, out_path, fps, ffmpeg)
        if result.skipped_empty:
            plural = "frame" if result.skipped_empty == 1 else "frames"
            print(
                f"warning: {prefix}: skipped {result.skipped_empty} empty {plural}",
                file=sys.stderr,
            )
        if result.error is not None:
            failed += 1
            print(f"failed: {prefix}: {result.error}", file=sys.stderr)
        else:
            plural = "frame" if result.piped == 1 else "frames"
            print(f"wrote {out_path} ({result.piped} {plural})")
    total = len(streams)
    summary = f"wrote {total - failed}/{total} streams"
    if failed:
        summary += f", {failed} failed"
    print(summary)
    return 1 if failed else 0
