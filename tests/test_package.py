"""Smoke tests for the package: it imports, has a version, exposes a CLI."""

from __future__ import annotations

import robolens


def test_has_version() -> None:
    assert isinstance(robolens.__version__, str)
    assert robolens.__version__


def test_public_api_is_fenced() -> None:
    # Everything reachable as public must be declared in __all__ (no accidental
    # surface growth). This guard tightens as the API grows.
    assert "__version__" in robolens.__all__


def test_cli_runs() -> None:
    from robolens.cli import main

    assert main([]) == 0
