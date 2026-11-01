"""The ``inspect-robots list`` command: print registered components by kind.."""

from __future__ import annotations

from ._shared import _KIND_BY_PLURAL


def _cmd_list(what: str | None) -> int:
    from inspect_robots.registry import registered

    plurals = [what] if what else sorted(_KIND_BY_PLURAL)
    for plural in plurals:
        kind = _KIND_BY_PLURAL[plural]
        names = sorted(registered(kind))
        print(f"{plural}:")
        for name in names:
            print(f"  - {name}")
        if not names:
            print("  (none)")
    return 0
