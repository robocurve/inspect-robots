"""The ``inspect-robots setup`` command: the interactive first-run wizard.."""

from __future__ import annotations

import os
import sys


def _cmd_setup() -> int:
    from inspect_robots._setup import run_setup

    return run_setup(
        os.environ,
        input_fn=input,
        out=sys.stdout,
        interactive=sys.stdin.isatty(),
    )
