"""Tests for guarded portable-log pointer resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from inspect_robots._pointers import read_jsonl_prefix, resolve_log_pointer


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


def test_read_jsonl_prefix_keeps_rows_before_first_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text('{"call": 0}\n{"call": 1}\n{"torn', encoding="utf-8")
    assert read_jsonl_prefix(path) == [{"call": 0}, {"call": 1}]


def test_read_jsonl_prefix_stops_at_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text('{"call": 0}\n[]\n{"call": 2}\n', encoding="utf-8")
    assert read_jsonl_prefix(path) == [{"call": 0}]


def test_read_jsonl_prefix_none_for_empty_missing_or_undecodable(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert read_jsonl_prefix(empty) is None
    assert read_jsonl_prefix(tmp_path / "absent.jsonl") is None
    torn_first = tmp_path / "torn.jsonl"
    torn_first.write_bytes(b"\xff\xfe")
    assert read_jsonl_prefix(torn_first) is None


def test_read_jsonl_prefix_keeps_prefix_before_torn_multibyte_tail(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b'{"call": 0}\n{"note": "\xe2\x82')
    assert read_jsonl_prefix(path) == [{"call": 0}]
