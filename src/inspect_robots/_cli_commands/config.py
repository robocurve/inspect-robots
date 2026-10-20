"""The ``inspect-robots config`` command: view or set user defaults.."""

from __future__ import annotations

import argparse
import os

from inspect_robots.defaults import _set_default, load_defaults


def _cmd_config(args: argparse.Namespace) -> int:
    if args.config_command == "set":
        path = _set_default(os.environ, args.key, args.value)
        print(f"wrote {args.key} = {args.value} to {path}")
        return 0
    defaults = load_defaults(os.environ)
    rows: list[tuple[str, object, str | None]] = [
        ("policy", defaults.policy, defaults.policy_source),
        ("embodiment", defaults.embodiment, defaults.embodiment_source),
        ("sim_embodiment", defaults.sim_embodiment, defaults.sim_embodiment_source),
        ("scorer", defaults.scorer, None),
        ("grader", defaults.grader, None),
        ("max_steps", defaults.max_steps, None),
        ("store_frames", defaults.store_frames, None),
        ("rerun", defaults.rerun, None),
        ("rerun_save", defaults.rerun_save, None),
        ("rerun_port", defaults.rerun_port, None),
    ]
    for key, value, source in rows:
        shown = "(unset)" if value is None else value
        suffix = f"  ({source})" if source else ""
        print(f"{key}: {shown}{suffix}")
    return 0
