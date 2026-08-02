"""An in-process v1beta BidiGenerateContent server for Live wire tests.

The server acknowledges setup, records every decoded client envelope by
connection, and emits queued response batches whenever a generation trigger
arrives. Tests may also provide a callback to close a connection or inject
messages at any exact inbound boundary.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import Server, ServerConnection, serve

MessageHook = Callable[["StubBidiServer", ServerConnection, int, dict[str, Any]], bool]


class StubBidiServer:
    """Serve a steerable Gemini Live JSON endpoint on a free local port."""

    def __init__(
        self,
        responses: list[list[dict[str, Any]]] | None = None,
        *,
        hook: MessageHook | None = None,
    ) -> None:
        self.connections: list[list[dict[str, Any]]] = []
        self.closed_connections: list[int] = []
        self._responses = list(responses or [])
        self._hook = hook
        self._lock = threading.Lock()
        self._stopped = False
        self._server: Server = serve(self._handler, "127.0.0.1", 0, max_size=None)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """Return the free-port websocket URL accepted by the policy seam."""
        port = self._server.socket.getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return all inbound envelopes in connection and arrival order."""
        with self._lock:
            return [message for connection in self.connections for message in connection]

    def wait_for_connections(self, count: int, timeout_s: float = 2.0) -> None:
        """Wait until at least ``count`` websocket sessions have opened."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.connections) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"expected {count} connections, saw {len(self.connections)}")

    def wait_for_closed(self, count: int, timeout_s: float = 2.0) -> None:
        """Wait until at least ``count`` handlers have observed socket closure."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.closed_connections) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(
            f"expected {count} closed connections, saw {len(self.closed_connections)}"
        )

    def stop(self) -> None:
        """Stop accepting connections and join the background server thread."""
        if self._stopped:
            return
        self._stopped = True
        self._server.shutdown()
        self._thread.join(timeout=5)

    def send(self, ws: ServerConnection, *messages: dict[str, Any]) -> bool:
        """Return false if the peer closes before the full batch is sent."""
        try:
            for message in messages:
                ws.send(json.dumps(message))
        except ConnectionClosed:
            return False
        return True

    def _handler(self, ws: ServerConnection) -> None:
        with self._lock:
            connection_index = len(self.connections)
            self.connections.append([])
        try:
            for raw in ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                message = json.loads(raw)
                if not isinstance(message, dict):
                    continue
                with self._lock:
                    self.connections[connection_index].append(message)
                if self._hook is not None and self._hook(self, ws, connection_index, message):
                    # Every terminating failure hook closes the connection.
                    # Returning here prevents already-buffered client frames
                    # from consuming a response meant for the recovery socket.
                    return
                if "setup" in message:
                    if not self.send(ws, {"setupComplete": {}}):
                        return
                elif _is_generation_trigger(message):
                    batch: list[dict[str, Any]] | None = None
                    with self._lock:
                        if self._responses:
                            batch = self._responses[0]
                    if batch is not None:
                        if not self.send(ws, *batch):
                            return
                        with self._lock:
                            if self._responses and self._responses[0] is batch:
                                self._responses.pop(0)
        finally:
            with self._lock:
                self.closed_connections.append(connection_index)


def _is_generation_trigger(message: dict[str, Any]) -> bool:
    """Return whether one client envelope asks the server to generate."""
    if "toolResponse" in message:
        return True
    content = message.get("clientContent")
    return isinstance(content, dict) and content.get("turnComplete") is True


def tool_call(
    call_id: str = "fc_1",
    name: str = "done",
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one exact v1beta toolCall server envelope."""
    return {
        "toolCall": {
            "functionCalls": [
                {
                    "name": name,
                    "args": {"summary": "ok"} if args is None else args,
                    "id": call_id,
                }
            ]
        }
    }


def completed_text(text: str) -> list[dict[str, Any]]:
    """Build a model text turn followed by its v1beta completion envelope."""
    return [
        {"serverContent": {"modelTurn": {"parts": [{"text": text}]}}},
        {"serverContent": {"turnComplete": True}},
    ]
