"""Replay-grade, per-attempt LLM wire capture.

Capture is stored below the evaluation log directory as
``wire/<run_id>/<trial_id>/calls.jsonl`` with decoded image bytes deduplicated
at ``wire/<run_id>/blobs/<sha256>.png``. Each JSONL row contains ``call``,
``attempt``, ``endpoint``, ``t``, ``duration_s``, ``request``, ``status``, and
``response``; ``error`` is present only for failed attempts.
There is one row per attempt. ``call`` is the zero-based logical call shared by
its retries, and ``t`` is the attempt's caller-supplied start time. A response
that parses as a JSON object is stored as that object at any HTTP status;
non-object or invalid JSON is stored as its first 2000 text characters, and a
missing response is stored as null.

Inline PNG payloads in supported chat, Responses, Anthropic, and Gemini Live
image parts are replaced in a deep copy of the request by
``$blob:<sha256>``. Data-URL prefixes and all other part keys remain intact.
Readers find references by scanning string values for the unanchored pattern
``\\$blob:[0-9a-f]{64}`` and restore them by replacing the sentinel with base64
of the corresponding decoded PNG blob.

Capture is best-effort and cannot fail an evaluation. The first failure in a
trial emits one stderr warning, disables that trial's sink, and is never
re-raised. Trials and files are opened lazily, so closed or zero-record trials
produce no rows or metadata pointer.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Any, TextIO

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PNG_DATA_URL_PREFIX = "data:image/png;base64,"
_FAILURE_PREFIX = "[agent] wire capture disabled for this trial: "
_VERSION_SKEW_NOTE = "[agent] wire capture inactive: core predates on_trial_start"


def _safe(name: str) -> str:
    safe = _SAFE_RE.sub("-", name)
    if safe != name:
        safe = f"{safe}-{zlib.crc32(name.encode()) & 0xFFFFFFFF:08x}"
    return safe


class WireCapture:
    """Persist exact LLM request attempts without affecting policy execution."""

    def __init__(self) -> None:
        """Create a closed sink with no trial history."""
        self._began = False
        self._warned_version_skew = False
        self._target_dir: Path | None = None
        self._blob_dir: Path | None = None
        self._relative_path: str | None = None
        self._handle: TextIO | None = None
        self._rows_written = False
        self._call = -1
        self._dead = True
        self._warned_failure = False

    @property
    def began(self) -> bool:
        """Return whether ``begin_trial`` has ever run on this instance."""
        return self._began

    def begin_trial(self, log_dir: str, run_id: str, trial_id: str) -> None:
        """Open a logical trial while deferring all filesystem creation."""
        try:
            previous_handle = self._handle
            # Clear every destination before closing the prior trial so even a
            # failing close cannot send later rows to its file.
            self._target_dir = None
            self._blob_dir = None
            self._relative_path = None
            self._handle = None
            self._rows_written = False
            self._call = -1
            self._dead = False
            self._warned_failure = False
            self._began = True

            if previous_handle is not None:
                previous_handle.close()

            safe_trial_id = _safe(trial_id)
            run_dir = Path(log_dir) / "wire" / run_id
            self._target_dir = run_dir / safe_trial_id
            self._blob_dir = run_dir / "blobs"
            self._relative_path = (Path("wire") / run_id / safe_trial_id / "calls.jsonl").as_posix()
        except BaseException as exc:
            self._disable(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    def record(
        self,
        *,
        attempt: int,
        endpoint: str,
        request: dict[str, Any],
        status: int | None,
        response_text: str | None,
        error: str | None,
        t_start: float,
        duration_s: float,
    ) -> None:
        """Append and flush one provider attempt when a trial is open."""
        try:
            if self._dead or self._target_dir is None or self._blob_dir is None:
                return

            if attempt == 0:
                self._call += 1
            captured_request = copy.deepcopy(request)
            self._replace_blobs(captured_request)

            row: dict[str, Any] = {
                "call": self._call,
                "attempt": attempt,
                "endpoint": endpoint,
                "t": t_start,
                "duration_s": duration_s,
                "request": captured_request,
                "status": status,
                "response": self._response_value(response_text),
            }
            if error is not None:
                row["error"] = error
            line = json.dumps(row, separators=(",", ":")) + "\n"

            if self._handle is None:
                self._target_dir.mkdir(parents=True, exist_ok=True)
                self._handle = (self._target_dir / "calls.jsonl").open("a", encoding="utf-8")
            self._handle.write(line)
            self._handle.flush()
            self._rows_written = True
        except BaseException as exc:
            self._disable(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    def end_trial(self) -> str | None:
        """Close the trial and return its relative JSONL pointer when nonempty."""
        pointer = self._relative_path if self._rows_written else None
        try:
            handle = self._handle
            self._handle = None
            self._dead = True
            if handle is not None:
                handle.close()
            return pointer
        except BaseException as exc:
            self._disable(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return pointer

    def warn_if_never_began(self) -> None:
        """Warn once when an older core never invoked the policy start hook."""
        if self._began or self._warned_version_skew:
            return
        self._warned_version_skew = True
        try:
            print(_VERSION_SKEW_NOTE, file=sys.stderr)
        except BaseException:
            # Diagnostics are subordinate to the never-fail capture contract.
            return

    def _disable(self, exc: BaseException) -> None:
        self._dead = True
        if self._warned_failure:
            return
        self._warned_failure = True
        try:
            print(f"{_FAILURE_PREFIX}{exc}", file=sys.stderr)
        except BaseException:
            return

    def _replace_blobs(self, value: Any) -> None:
        if isinstance(value, dict):
            inline_data = value.get("inlineData")
            if isinstance(inline_data, dict):
                payload = inline_data.get("data")
                if isinstance(payload, str):
                    inline_data["data"] = self._store_blob(payload)
            part_type = value.get("type")
            if part_type == "image_url":
                image_url = value.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url.startswith(_PNG_DATA_URL_PREFIX):
                        image_url["url"] = self._replace_data_url(url)
            elif part_type == "input_image":
                url = value.get("image_url")
                if isinstance(url, str) and url.startswith(_PNG_DATA_URL_PREFIX):
                    value["image_url"] = self._replace_data_url(url)
            elif part_type == "image":
                source = value.get("source")
                if isinstance(source, dict) and source.get("type") == "base64":
                    payload = source.get("data")
                    if isinstance(payload, str):
                        source["data"] = self._store_blob(payload)
            for child in value.values():
                self._replace_blobs(child)
        elif isinstance(value, list):
            for child in value:
                self._replace_blobs(child)

    def _replace_data_url(self, value: str) -> str:
        payload = value[len(_PNG_DATA_URL_PREFIX) :]
        return _PNG_DATA_URL_PREFIX + self._store_blob(payload)

    def _store_blob(self, payload: str) -> str:
        # record() rejects closed trials before blob extraction can reach here.
        assert self._blob_dir is not None
        decoded = base64.b64decode(payload)
        digest = hashlib.sha256(decoded).hexdigest()
        path = self._blob_dir / f"{digest}.png"
        if not path.exists():
            self._blob_dir.mkdir(parents=True, exist_ok=True)
            # Atomic publish: a partial write must never sit under the
            # content hash, or dedup would pin the poisoned bytes run-long.
            # The deterministic temp name is safe only while trials run
            # sequentially in one process and run_ids are uuid-suffixed;
            # parallel trials would need per-writer temp names.
            tmp = self._blob_dir / f".{digest}.tmp"
            tmp.write_bytes(decoded)
            os.replace(tmp, path)
        return f"$blob:{digest}"

    @staticmethod
    def _response_value(response_text: str | None) -> Any:
        if response_text is None:
            return None
        try:
            parsed = json.loads(response_text)
        except (json.JSONDecodeError, RecursionError):
            return response_text[:2000]
        return parsed if isinstance(parsed, dict) else response_text[:2000]
