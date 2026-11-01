"""Constants and helpers shared across ``inspect-robots`` CLI commands: ANSI
styling, outcome phrasing, component-kind vocabularies, and ``k=v`` parsing.."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from inspect_robots.defaults import _ENV_EMBODIMENT, _ENV_POLICY, _parse_value
from inspect_robots.types import OPERATOR_END


def _styled(text: str, code: str) -> str:
    """Wrap ``text`` in an ANSI style when stdout is an interactive terminal.

    Plain text is returned when piped/redirected or when ``NO_COLOR`` is set,
    so scripts and CI logs never see escape codes.
    """
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _format_metric(value: float | int | None) -> str:
    """Format numeric metric value or return 'n/a' when None."""
    return "n/a" if value is None else f"{value:.4g}"


_BOLD = "1"


_BOLD_BRIGHT_MAGENTA = "1;95"


_DIM = "2"


_CYAN = "36"


_GREEN = "32"


_RED = "31"


_YELLOW = "33"


_OUTCOME_PHRASES = {
    "success": "succeeded",
    "failure": "failed",
    "max_steps": "hit step limit",
    OPERATOR_END: "ended by operator",
    "give_up": "gave up",
    "done": "reported done",
    "policy_stop": "stopped by policy",
    "truncated": "truncated",
}


_KIND_BY_PLURAL = {
    "tasks": "task",
    "policies": "policy",
    "embodiments": "embodiment",
    "scorers": "scorer",
    "graders": "grader",
    "sinks": "sink",
    "operator_inputs": "operator_input",
}


_PLURAL_BY_KIND = {kind: plural for plural, kind in _KIND_BY_PLURAL.items()}


_SUBCOMMANDS = (
    "list",
    "run",
    "eval-set",
    "inspect",
    "summarize",
    "view",
    "video",
    "config",
    "setup",
    "doctor",
)


_ENV_BY_KIND = {"policy": _ENV_POLICY, "embodiment": _ENV_EMBODIMENT}


DEFAULT_RERUN_CONNECT_URL = "rerun+http://127.0.0.1:9876/proxy"


_LIVE_FRAMES_BUDGET_MB = 8.0


def _parse_kvs(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key] = _parse_value(value)
    return out
