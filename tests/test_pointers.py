"""Tests for guarded portable-log pointer resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from inspect_robots._pointers import resolve_log_pointer


def test_resolve_log_pointer_accepts_only_paths_beneath_log_directory(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.json"

    assert resolve_log_pointer(log_path, "wire/run/trial/calls.jsonl") == (
        tmp_path / "wire/run/trial/calls.jsonl"
    )
    assert resolve_log_pointer(log_path, "") is None
    assert resolve_log_pointer(log_path, None) is None
    assert resolve_log_pointer(log_path, "../outside.jsonl") is None
    assert resolve_log_pointer(log_path, tmp_path / "sidecar.jsonl") is None


def test_resolve_log_pointer_degrades_resolution_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_resolve(self: Path) -> Path:
        del self
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    assert resolve_log_pointer(tmp_path / "run.json", "wire/calls.jsonl") is None
