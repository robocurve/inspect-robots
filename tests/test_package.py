"""Smoke tests for the package: it imports, has a version, exposes a CLI."""

from __future__ import annotations

import inspect_robots


def test_has_version() -> None:
    assert isinstance(inspect_robots.__version__, str)
    assert inspect_robots.__version__


def test_public_api_is_fenced() -> None:
    # Everything reachable as public must be declared in __all__ (no accidental
    # surface growth). This guard tightens as the API grows.
    assert "__version__" in inspect_robots.__all__


def test_cli_runs() -> None:
    from inspect_robots.cli import main

    assert main([]) == 0
