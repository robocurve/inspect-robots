"""Smoke tests for the package: it imports, has a version, exposes a CLI."""

from __future__ import annotations

import roboinspect


def test_has_version() -> None:
    assert isinstance(roboinspect.__version__, str)
    assert roboinspect.__version__


def test_public_api_is_fenced() -> None:
    # Everything reachable as public must be declared in __all__ (no accidental
    # surface growth). This guard tightens as the API grows.
    assert "__version__" in roboinspect.__all__


def test_cli_runs() -> None:
    from roboinspect.cli import main

    assert main([]) == 0
