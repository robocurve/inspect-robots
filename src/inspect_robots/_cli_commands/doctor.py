"""The ``inspect-robots doctor`` command: preflight an installed embodiment adapter.."""

from __future__ import annotations

import argparse
import os

from inspect_robots.defaults import load_defaults

from ._shared import _parse_kvs
from .components import _config_args, _pick_component, _resolve_or_exit


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Preflight runtime requirements and conformance for an installed adapter.

    Purely declarative — the embodiment is constructed (adapters keep
    constructors hardware-free by convention) but never reset or stepped.
    """
    from inspect_robots.conformance import (
        check_device_slots,
        check_embodiment,
        missing_runtime_requirements,
    )
    from inspect_robots.registry import registered

    defaults = load_defaults(os.environ)
    name, source = _pick_component(
        "embodiment", args.embodiment, defaults.embodiment, defaults.embodiment_source
    )
    config_kvs = _config_args(
        "embodiment", name, defaults.embodiment_args_owner, defaults.embodiment_args
    )
    kvs = {**config_kvs, **_parse_kvs(args.embodiment_args)}
    print(f"embodiment: {name} ({source})")
    factory = registered("embodiment").get(name)
    missing = missing_runtime_requirements(factory)
    for module, remedy in missing.items():
        print(f"  [error] runtime-requirement: {module} missing → {remedy}")
    # Check before construction so constructor failures do not hide device findings.
    device_issues = check_device_slots(factory, kvs)
    for issue in device_issues:
        print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    embodiment = _resolve_or_exit("embodiment", name, **kvs)
    try:
        report = check_embodiment(embodiment.info)
    finally:
        embodiment.close()
    print(report.summary())
    if not report.ok:
        print("see the adapter authoring guide: docs/guide/adapters.md")
    return 1 if not report.ok or missing or device_issues else 0
