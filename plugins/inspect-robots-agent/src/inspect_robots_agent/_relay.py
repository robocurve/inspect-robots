"""Metadata-only Relay lifecycles; application execution always stays with the caller."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, TypeVar
from uuid import uuid4
from weakref import finalize

import nemo_relay as relay

from ._capture import _safe
from ._llm import AssistantMessage, ToolCall
from ._tools import ToolResult

_T = TypeVar("_T")
_Handle = TypeVar("_Handle", relay.LLMHandle, relay.ToolHandle)
_LOG = logging.getLogger(__name__)
_DRAIN_TIMEOUT_S = 1.0


class _Writer:
    """Keep queued callbacks and deferred cleanup attached to their original trial."""

    def __init__(self, path: Path, capture_id: str) -> None:
        self._file = path.open("w", encoding="utf-8")
        self._capture_id = capture_id
        self._lock = Lock()
        self._closed = False
        self.drained = Event()
        self.error: str | None = None
        self.events_written = 0

    def _failed(self, exc: Exception) -> None:
        if self.error is None:
            self.error = type(exc).__name__
            _LOG.warning("Relay writer incomplete (%s)", self.error)

    def write(self, event: relay.Event) -> None:
        """Write only this trial's native events; close on its terminal event."""
        metadata = event.metadata
        if not isinstance(metadata, dict) or metadata.get("capture_id") != self._capture_id:
            return
        with self._lock:
            if self._closed:
                return
            try:
                self._file.write(event.to_json() + "\n")
                self.events_written += 1
            except Exception as exc:
                self._failed(exc)
        if metadata.get("capture_terminal"):
            self.close()

    def close(self) -> None:
        """Finish file I/O on a background thread, never the evaluation thread."""
        with self._lock:
            if not self._closed:
                try:
                    self._file.close()
                except Exception as exc:
                    self._failed(exc)
                finally:
                    self._closed = True
                    self.drained.set()


def _abandon(name: str, writer: _Writer) -> None:
    # The core can skip on_trial_end if scoring is interrupted. This callback
    # must not retain RelayTrace, or the subscriber would keep its owner alive.
    try:
        relay.subscribers.deregister(name)
    finally:
        if not writer.drained.is_set():
            Thread(target=writer.close, daemon=True, name="relay-capture-close").start()


class RelayTrace:
    """Own one trial's stack and asynchronous ATOF writer at a time."""

    def __init__(self, model: str, wire: str) -> None:
        self._model = model
        self._wire = wire
        self._root: relay.ScopeHandle | None = None
        self._stack: relay.ScopeStack | None = None
        self._writer: _Writer | None = None
        self._finalizer: finalize[[str, _Writer], RelayTrace] | None = None
        self._metadata: dict[str, Any] = {}
        self._name = ""
        self._path: str | None = None
        self._error: str | None = None
        self._expected_events = 0

    def _failed(self, exc: Exception) -> None:
        # Exception messages may contain provider payloads or filesystem details.
        if self._error is None:
            self._error = type(exc).__name__
            _LOG.warning("Relay capture incomplete (%s)", self._error)

    @contextmanager
    def _capture_errors(self) -> Iterator[None]:
        """Suppress tracing failures without wrapping application execution."""
        try:
            yield
        except Exception as exc:
            self._failed(exc)

    def begin(self, log_dir: str, run_id: str, scene_id: str, epoch: int) -> None:
        """Start an isolated trial; a writer failure does not stop the policy."""
        if self._name:
            self.end("interrupted", None)
        self._error = None
        self._writer = None
        self._expected_events = 2  # trial start and end
        capture_id = uuid4().hex
        self._name = f"inspect-robots-{capture_id}"
        self._metadata = {
            "capture_id": capture_id,
            "run_id": run_id,
            "scene_id": scene_id,
            "epoch": epoch,
        }
        self._path = f"relay/{_safe(run_id)}/{_safe(scene_id)}-e{epoch}-{capture_id}/events.jsonl"

        with self._capture_errors():
            self._stack = relay.create_scope_stack()
            path = Path(log_dir) / self._path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = _Writer(path, capture_id)
            self._finalizer = finalize(self, _abandon, self._name, self._writer)
            relay.subscribers.register(self._name, self._writer.write)
            with relay.use_scope_stack(self._stack):
                self._root = relay.scope.push(
                    "inspect-robots.trial",
                    relay.ScopeType.Agent,
                    metadata=self._metadata,
                )

    def _observe(
        self,
        operation: Callable[[], _T],
        start: Callable[[], _Handle],
        finish: Callable[[_Handle, dict[str, Any], _T | None], None],
        metadata: dict[str, Any],
    ) -> _T:
        handle: _Handle | None = None
        if self._root is not None:
            with self._capture_errors():
                assert self._stack is not None
                with relay.use_scope_stack(self._stack):
                    handle = start()
        if handle is not None:
            self._expected_events += 2
        result: _T | None = None
        metadata["status"] = "success"
        try:
            result = operation()
            return result
        except BaseException as exc:
            metadata.update(status="error", error_type=type(exc).__name__)
            metadata["otel.status_code"] = "ERROR"
            raise
        finally:
            if handle is not None:
                with self._capture_errors():
                    assert self._stack is not None
                    with relay.use_scope_stack(self._stack):
                        finish(handle, metadata, result)

    def model_call(self, operation: Callable[[], AssistantMessage], index: int) -> AssistantMessage:
        """Trace one logical call, leaving provider retries in native wire capture."""
        metadata = {**self._metadata, "model_call_index": index, "wire": self._wire}

        def start() -> relay.LLMHandle:
            return relay.llm.call(
                "provider.complete",
                relay.LLMRequest({}, {}),
                handle=self._root,
                model_name=self._model,
                metadata=metadata,
            )

        def finish(
            handle: relay.LLMHandle, meta: dict[str, Any], result: AssistantMessage | None
        ) -> None:
            # Capture completion before queued publication, so the model end
            # cannot move past the next tool call when subscribers are slow.
            ended_at = datetime.now(timezone.utc)
            usage = result.usage if result is not None else None
            normalized = {
                target: usage[source]
                for source, target in (
                    ("input_tokens", "prompt_tokens"),
                    ("output_tokens", "completion_tokens"),
                    ("cache_read_input_tokens", "cache_read_tokens"),
                    ("cache_creation_input_tokens", "cache_write_tokens"),
                )
                if usage is not None and source in usage
            }
            annotation = relay.AnnotatedLLMResponse(model=self._model, usage=normalized or None)
            relay.llm.call_end(
                handle, {}, metadata=meta, annotated_response=annotation, timestamp=ended_at
            )

        return self._observe(operation, start, finish, metadata)

    def tool_call(
        self,
        operation: Callable[[], ToolResult],
        call: ToolCall,
        index: int | None,
    ) -> ToolResult:
        """Record dispatch acceptance, not physical motion completion."""
        metadata = {
            **self._metadata,
            "model_call_index": index,
            "source": "policy" if index is None else "model",
            "boundary": "tool_dispatch",
        }

        def start() -> relay.ToolHandle:
            return relay.tools.call(
                call.name,
                {},
                handle=self._root,
                tool_call_id=call.id,
                metadata=metadata,
            )

        def finish(
            handle: relay.ToolHandle, meta: dict[str, Any], result: ToolResult | None
        ) -> None:
            if result is not None:
                meta["status"] = "rejected" if result.error is not None else "accepted"
                meta["otel.status_code"] = "ERROR" if result.error is not None else "OK"
                meta["action_chunk_returned"] = result.chunk is not None
            relay.tools.call_end(handle, relay.ToolExecutionResult({}), metadata=meta)

        return self._observe(operation, start, finish, metadata)

    def end(self, status: str, termination_reason: str | None) -> dict[str, Any]:
        """Close and drain this trial only, then return a portable capture receipt."""

        with self._capture_errors():
            if self._root is not None:
                assert self._stack is not None
                with relay.use_scope_stack(self._stack):
                    relay.scope.pop(
                        self._root,
                        metadata={
                            **self._metadata,
                            "status": status,
                            "otel.status_code": "OK" if status == "success" else "ERROR",
                            "termination_reason": termination_reason,
                            "capture_terminal": True,
                        },
                    )
            elif self._writer is not None:
                Thread(target=self._writer.close, daemon=True, name="relay-capture-close").start()
        if self._writer is not None:
            writer = self._writer
            if not writer.drained.wait(_DRAIN_TIMEOUT_S):
                self._failed(TimeoutError())
            with self._capture_errors():
                _abandon(self._name, writer)
            self._error = self._error or writer.error
            # SDK sanitizers can clear correlation metadata asynchronously.
            # A terminal event alone does not establish that every span arrived.
            if writer.events_written != self._expected_events:
                self._failed(RuntimeError("Relay event count mismatch"))
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        self._root = None
        self._name = ""
        receipt: dict[str, Any] = {
            "status": "error" if self._error else "captured",
            "path": self._path,
        }
        if self._writer is not None:
            receipt.update(
                events_expected=self._expected_events, events_written=self._writer.events_written
            )
        if self._error:
            receipt["error_type"] = self._error
        return receipt
