"""The ``robolens`` command-line interface.

In M0 this is a minimal entry point that reports the version. Subcommands
(``list``, ``run``) are added once the registry and ``eval()`` land.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from robolens import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robolens",
        description="RoboLens — the Inspect AI for robotics.",
    )
    parser.add_argument("--version", action="version", version=f"robolens {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    # With no subcommand yet, show help so the CLI is self-documenting.
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
