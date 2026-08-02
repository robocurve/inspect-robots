"""Replay-grade wire capture sink."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any, TextIO, cast

import pytest

import inspect_robots_agent._capture as capture_module
from inspect_robots_agent._capture import WireCapture

_PNG_BYTES = b"\x89PNG\r\n\x1a\ncapture-test"
_PAYLOAD = base64.b64encode(_PNG_BYTES).decode("ascii")
_DATA_URL = f"data:image/png;base64,{_PAYLOAD}"


def _record(
    capture: WireCapture,
    *,
    attempt: int = 0,
    request: dict[str, Any] | None = None,
    status: int | None = 200,
    response_text: str | None = '{"ok":true}',
    error: str | None = None,
) -> None:
    capture.record(
        attempt=attempt,
        endpoint="/chat/completions",
        request={} if request is None else request,
        status=status,
        response_text=response_text,
        error=error,
        t_start=123.5,
        duration_s=0.25,
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert isinstance(row, dict)
        rows.append(cast(dict[str, Any], row))
    return rows


def test_extracts_all_wire_image_shapes_without_mutating_request(tmp_path: Path) -> None:
    request: dict[str, Any] = {
        "messages": [
            {
                "type": "image_url",
                "image_url": {
                    "url": _DATA_URL,
                    "cache_control": {"scope": "inside"},
                    "detail": "high",
                },
                "cache_control": {"type": "ephemeral"},
                "vendor": "chat",
            },
            {
                "type": "input_image",
                "image_url": _DATA_URL,
                "detail": "auto",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _PAYLOAD,
                    "vendor": "anthropic-source",
                },
                "cache_control": {"type": "ephemeral"},
                "vendor": "anthropic",
            },
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": _PAYLOAD,
                    "vendor": "gemini-source",
                },
                "vendor": "gemini-live",
            },
        ]
    }
    original = json.loads(json.dumps(request))
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")

    _record(capture, request=request)

    digest = hashlib.sha256(_PNG_BYTES).hexdigest()
    sentinel = f"$blob:{digest}"
    row = _rows(tmp_path / "wire/run-1/scene-e0/calls.jsonl")[0]
    parts = row["request"]["messages"]
    assert parts[0] == {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{sentinel}",
            "cache_control": {"scope": "inside"},
            "detail": "high",
        },
        "cache_control": {"type": "ephemeral"},
        "vendor": "chat",
    }
    assert parts[1] == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{sentinel}",
        "detail": "auto",
    }
    assert parts[2] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": sentinel,
            "vendor": "anthropic-source",
        },
        "cache_control": {"type": "ephemeral"},
        "vendor": "anthropic",
    }
    assert parts[3] == {
        "inlineData": {
            "mimeType": "image/png",
            "data": sentinel,
            "vendor": "gemini-source",
        },
        "vendor": "gemini-live",
    }
    assert request == original
    assert re.fullmatch(r"\$blob:[0-9a-f]{64}", sentinel)
    assert (tmp_path / f"wire/run-1/blobs/{digest}.png").read_bytes() == _PNG_BYTES


def test_blob_deduplication_spans_records_and_trials(tmp_path: Path) -> None:
    capture = WireCapture()
    request: dict[str, Any] = {"part": {"type": "input_image", "image_url": _DATA_URL}}
    capture.begin_trial(str(tmp_path), "run-1", "first-e0")
    _record(capture, request=request)
    _record(capture, request=request)
    capture.end_trial()
    capture.begin_trial(str(tmp_path), "run-1", "second-e0")
    _record(capture, request=request)
    capture.end_trial()

    assert len(list((tmp_path / "wire/run-1/blobs").glob("*.png"))) == 1
    assert len(_rows(tmp_path / "wire/run-1/first-e0/calls.jsonl")) == 2
    assert len(_rows(tmp_path / "wire/run-1/second-e0/calls.jsonl")) == 1


def test_row_schema_and_response_normalization(tmp_path: Path) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    _record(capture, status=503, response_text='{"message":"busy"}')
    _record(
        capture,
        status=None,
        response_text=None,
        error="connection reset",
    )
    long_text = "not-json-" + "x" * 2100
    _record(capture, response_text=long_text)
    array_text = json.dumps(["y" * 2100])
    _record(capture, response_text=array_text)
    _record(capture, response_text=None)
    path = tmp_path / "wire/run-1/scene-e0/calls.jsonl"
    rows = _rows(path)

    assert set(rows[0]) == {
        "call",
        "attempt",
        "endpoint",
        "t",
        "duration_s",
        "request",
        "status",
        "response",
    }
    assert rows[0] == {
        "call": 0,
        "attempt": 0,
        "endpoint": "/chat/completions",
        "t": 123.5,
        "duration_s": 0.25,
        "request": {},
        "status": 503,
        "response": {"message": "busy"},
    }
    assert rows[1]["status"] is None
    assert rows[1]["response"] is None
    assert rows[1]["error"] == "connection reset"
    assert rows[2]["response"] == long_text[:2000]
    assert rows[3]["response"] == array_text[:2000]
    assert rows[4]["response"] is None


def test_call_counter_retries_and_trial_reset(tmp_path: Path) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "first-e0")
    for attempt in (0, 1, 0, 2, 0):
        _record(capture, attempt=attempt)
    assert [row["call"] for row in _rows(tmp_path / "wire/run-1/first-e0/calls.jsonl")] == [
        0,
        0,
        1,
        1,
        2,
    ]

    # Beginning a new trial also closes a still-open prior handle.
    capture.begin_trial(str(tmp_path), "run-1", "second-e0")
    _record(capture)
    pointer = capture.end_trial()
    assert pointer == "wire/run-1/second-e0/calls.jsonl"
    assert pointer is not None
    assert [row["call"] for row in _rows(tmp_path / pointer)] == [0]
    assert len(_rows(tmp_path / "wire/run-1/first-e0/calls.jsonl")) == 5


def test_lazy_creation_and_closed_sink_noops(tmp_path: Path) -> None:
    capture = WireCapture()
    _record(capture)
    assert not capture.began
    assert not (tmp_path / "wire").exists()

    capture.begin_trial(str(tmp_path), "run-1", "empty-e0")
    assert capture.began
    assert capture.end_trial() is None
    assert not (tmp_path / "wire").exists()

    _record(capture)
    assert not (tmp_path / "wire").exists()


def test_trial_id_is_sanitized_in_path_and_pointer(tmp_path: Path) -> None:
    trial_id = "a/b-e0"
    safe_id = f"a-b-e0-{zlib.crc32(trial_id.encode()) & 0xFFFFFFFF:08x}"
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", trial_id)
    _record(capture)

    pointer = capture.end_trial()

    assert pointer == f"wire/run-1/{safe_id}/calls.jsonl"
    assert pointer is not None
    assert (tmp_path / pointer).is_file()
    assert not (tmp_path / "wire/run-1/a/b-e0").exists()


def test_each_record_is_visible_before_end_trial(tmp_path: Path) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    path = tmp_path / "wire/run-1/scene-e0/calls.jsonl"

    _record(capture)
    assert len(_rows(path)) == 1
    _record(capture)
    assert len(_rows(path)) == 2
    capture.end_trial()
    _record(capture)
    assert len(_rows(path)) == 2
    assert capture.began


def test_record_failure_disables_trial_and_warns_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")

    _record(capture, request={"unserializable": object()})
    _record(capture)

    assert capture.end_trial() is None
    assert capsys.readouterr().err == (
        "[agent] wire capture disabled for this trial: "
        "Object of type object is not JSON serializable\n"
    )
    assert capsys.readouterr().err == ""
    assert not (tmp_path / "wire").exists()


def test_second_failure_in_dead_trial_stays_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    _record(capture)
    handle: TextIO = capture.__dict__["_handle"]

    class _BrokenWriteAndClose:
        def write(self, value: str) -> int:
            raise OSError("write failed")

        def close(self) -> None:
            handle.close()
            raise OSError("close failed")

    capture.__dict__["_handle"] = _BrokenWriteAndClose()
    _record(capture)
    _record(capture)

    assert capture.end_trial() == "wire/run-1/scene-e0/calls.jsonl"
    assert capsys.readouterr().err == (
        "[agent] wire capture disabled for this trial: write failed\n"
    )


def test_failing_stderr_diagnostics_never_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_print(*args: object, **kwargs: object) -> None:
        raise OSError("stderr unavailable")

    monkeypatch.setattr(capture_module, "print", bad_print, raising=False)

    never_begun = WireCapture()
    never_begun.warn_if_never_began()
    never_begun.warn_if_never_began()

    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    _record(capture, request={"unserializable": object()})
    _record(capture)
    assert capture.end_trial() is None


def test_begin_and_end_failures_never_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    begin_capture = WireCapture()

    def bad_safe(name: str) -> str:
        raise OSError(f"cannot sanitize {name}")

    monkeypatch.setattr(capture_module, "_safe", bad_safe)
    begin_capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    _record(begin_capture)
    assert begin_capture.began
    assert begin_capture.end_trial() is None
    assert capsys.readouterr().err == (
        "[agent] wire capture disabled for this trial: cannot sanitize scene-e0\n"
    )

    monkeypatch.undo()
    end_capture = WireCapture()
    end_capture.begin_trial(str(tmp_path), "run-1", "other-e0")
    _record(end_capture)
    handle: TextIO = end_capture.__dict__["_handle"]

    class _BrokenClose:
        def close(self) -> None:
            handle.close()
            raise OSError("close failed")

    end_capture.__dict__["_handle"] = _BrokenClose()
    assert end_capture.end_trial() == "wire/run-1/other-e0/calls.jsonl"
    assert capsys.readouterr().err == (
        "[agent] wire capture disabled for this trial: close failed\n"
    )


def test_version_skew_warning_is_once_per_never_begun_instance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = WireCapture()
    capture.warn_if_never_began()
    capture.warn_if_never_began()
    assert capsys.readouterr().err == (
        "[agent] wire capture inactive: core predates on_trial_start\n"
    )

    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    capture.end_trial()
    capture.warn_if_never_began()
    assert capsys.readouterr().err == ""


def test_keyboard_interrupt_in_record_propagates_after_disabling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    monkeypatch.setattr(
        "inspect_robots_agent._capture.copy.deepcopy",
        lambda value: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        _record(capture)
    monkeypatch.undo()
    _record(capture)
    assert capture.end_trial() is None
    assert "wire capture disabled" in capsys.readouterr().err


def test_blob_write_is_atomic_and_partial_writes_leave_no_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")

    def _partial_write(self: Path, data: bytes) -> int:
        self.write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _partial_write)
    _record(
        capture,
        request={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PAYLOAD}"},
                        }
                    ],
                }
            ]
        },
    )
    assert "wire capture disabled" in capsys.readouterr().err
    monkeypatch.undo()

    blob_dir = tmp_path / "wire/run-1/blobs"
    published = list(blob_dir.glob("*.png")) if blob_dir.exists() else []
    assert published == []

    capture.begin_trial(str(tmp_path), "run-1", "scene-e1")
    _record(
        capture,
        request={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PAYLOAD}"},
                        }
                    ],
                }
            ]
        },
    )
    (blob,) = (tmp_path / "wire/run-1/blobs").glob("*.png")
    assert not list((tmp_path / "wire/run-1/blobs").glob(".*.tmp"))
    assert hashlib.sha256(blob.read_bytes()).hexdigest() == blob.stem


class _InterruptingHandle:
    """Stand-in file handle whose close simulates a mid-close Ctrl-C."""

    def close(self) -> None:
        raise KeyboardInterrupt


def test_keyboard_interrupt_in_begin_and_end_trial_propagates(tmp_path: Path) -> None:
    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-1", "scene-e0")
    _record(capture)
    capture._handle = cast(TextIO, _InterruptingHandle())
    with pytest.raises(KeyboardInterrupt):
        capture.begin_trial(str(tmp_path), "run-1", "scene-e1")

    capture = WireCapture()
    capture.begin_trial(str(tmp_path), "run-2", "scene-e0")
    _record(capture)
    capture._handle = cast(TextIO, _InterruptingHandle())
    with pytest.raises(KeyboardInterrupt):
        capture.end_trial()
