"""Unit tests for `_print_environment_diagnostics`."""

from __future__ import annotations

import io

import inspect_robots
from inspect_robots.cli import _print_environment_diagnostics


def test_diagnostics_emits_header_line(capsys) -> None:
    sink = io.StringIO()
    _print_environment_diagnostics(sink)
    out = sink.getvalue()
    assert "Environment diagnostics:" in out
    assert "python:" in out
    assert "platform:" in out
    assert "inspect-robots:" in out
    assert inspect_robots.__version__ in out


def test_diagnostics_handles_missing_optional_extras(capsys) -> None:
    """Even when one optional extra is missing, the diagnostics block is printed."""
    sink = io.StringIO()
    _print_environment_diagnostics(sink)
    out = sink.getvalue()
    # Must produce the section even if pyfakefs (test-only) is not installed
    # in the user's environment; we still see the "not installed" line.
    assert "pyfakefs" in out
