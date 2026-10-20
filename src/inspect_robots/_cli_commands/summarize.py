"""The ``inspect-robots summarize`` command: distill a saved eval log into a
learnings file.."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import cast


def _cmd_summarize(args: argparse.Namespace) -> int:
    """Distill a saved log and atomically write its markdown learnings artifact."""
    from inspect_robots._summarize import summarize
    from inspect_robots.errors import ConfigError

    stdout_mode = args.out == "-"
    log_path = Path(args.log)
    out_path = (
        None
        if stdout_mode
        else (
            log_path.parent / "learnings" / f"{log_path.stem}.md"
            if args.out is None
            else Path(args.out)
        )
    )
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out {out_path} is a directory; pass a Markdown file path")
    if out_path is not None and out_path.resolve() == log_path.resolve():
        # The one data-loss path in this command: rendering over the log it reads.
        raise SystemExit(f"--out {out_path} would overwrite the input log; pass a different path")

    try:
        document = summarize(
            log_path,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if stdout_mode:
        sys.stdout.write(document)
        return 0

    file_path = cast(Path, out_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(document)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, file_path)
    print(f"wrote {file_path}")
    return 0
