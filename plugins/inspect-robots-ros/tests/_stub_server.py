"""In-process rosbridge-shaped websocket server for adapter tests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

from websockets.sync.server import Server, ServerConnection, serve


class StubRosbridgeServer:
    """Serve the rosbridge operations used by the plugin on a free local port."""

    def __init__(self) -> None:
        self.ops: list[dict[str, Any]] = []
        self.deferred_services: set[str] = set()
        self.service_results: dict[str, tuple[bool, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._connections: set[ServerConnection] = set()
        self._subscriptions: dict[ServerConnection, dict[str, dict[str, Any]]] = {}
        self._server: Server = serve(self._handler, "127.0.0.1", 0, max_size=None)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """The websocket URL bound to the server's ephemeral port."""
        port = self._server.socket.getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    @property
    def subscriptions(self) -> dict[str, dict[str, Any]]:
        """Current subscriptions keyed by rosbridge subscription ID."""
        with self._lock:
            return {
                subscription_id: dict(operation)
                for by_id in self._subscriptions.values()
                for subscription_id, operation in by_id.items()
            }

    def ops_of(self, op: str) -> list[dict[str, Any]]:
        """Return a snapshot of received operations matching ``op``."""
        with self._lock:
            return [dict(operation) for operation in self.ops if operation.get("op") == op]

    def wait_for(
        self, predicate: Callable[[list[dict[str, Any]]], bool], *, timeout_s: float = 2.0
    ) -> None:
        """Wait until a predicate over the recorded operations becomes true."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                snapshot = [dict(operation) for operation in self.ops]
            if predicate(snapshot):
                return
            time.sleep(0.005)
        raise TimeoutError("stub rosbridge did not receive the expected operation")

    def publish(self, topic: str, msg: Mapping[str, Any]) -> None:
        """Send a topic message only to connections subscribed to that topic."""
        frame = {"op": "publish", "topic": topic, "msg": dict(msg)}
        with self._lock:
            targets = [
                connection
                for connection, by_id in self._subscriptions.items()
                if any(operation.get("topic") == topic for operation in by_id.values())
            ]
        self._send(targets, frame)

    def send_status(self, level: str, message: str, *, request_id: str | None = None) -> None:
        """Send a rosbridge status notification to every live client."""
        frame: dict[str, Any] = {"op": "status", "level": level, "msg": message}
        if request_id is not None:
            frame["id"] = request_id
        self._send(self._connected_targets(), frame)

    def send_service_response(
        self,
        request_id: str,
        *,
        result: bool = True,
        values: Mapping[str, Any] | None = None,
    ) -> None:
        """Send an explicitly correlated service response to every live client."""
        frame = {
            "op": "service_response",
            "id": request_id,
            "values": dict(values or {}),
            "result": result,
        }
        self._send(self._connected_targets(), frame)

    def drop_connections(self) -> None:
        """Close every client socket to exercise receive-thread death latching."""
        for connection in self._connected_targets():
            with suppress(Exception):
                connection.close()

    def stop(self) -> None:
        """Shut down the websocket server and join its serving thread."""
        with self._lock:
            targets = list(self._connections)
        for connection in targets:
            with suppress(Exception):
                connection.close()
        self._server.shutdown()
        self._thread.join(timeout=5)

    def _handler(self, ws: ServerConnection) -> None:
        with self._lock:
            self._connections.add(ws)
            self._subscriptions[ws] = {}
        try:
            for raw in ws:
                if not isinstance(raw, (str, bytes)):
                    continue
                operation = json.loads(raw)
                if not isinstance(operation, dict):
                    continue
                with self._lock:
                    self.ops.append(operation)
                op = operation.get("op")
                if op == "subscribe":
                    subscription_id = operation.get("id")
                    if isinstance(subscription_id, str):
                        with self._lock:
                            self._subscriptions[ws][subscription_id] = operation
                elif op == "unsubscribe":
                    subscription_id = operation.get("id")
                    if isinstance(subscription_id, str):
                        with self._lock:
                            self._subscriptions[ws].pop(subscription_id, None)
                elif op == "call_service":
                    self._handle_service(ws, operation)
        finally:
            with self._lock:
                self._connections.discard(ws)
                self._subscriptions.pop(ws, None)

    def _handle_service(self, ws: ServerConnection, operation: dict[str, Any]) -> None:
        service = operation.get("service")
        request_id = operation.get("id")
        if not isinstance(service, str) or not isinstance(request_id, str):
            return
        if service in self.deferred_services:
            return
        result, values = self.service_results.get(service, (True, {"ok": True}))
        self._send(
            [ws],
            {
                "op": "service_response",
                "id": request_id,
                "values": values,
                "result": result,
            },
        )

    def _connected_targets(self) -> list[ServerConnection]:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                targets = list(self._connections)
            if targets:
                return targets
            time.sleep(0.005)
        raise TimeoutError("stub rosbridge has no connected client")

    @staticmethod
    def _send(targets: list[ServerConnection], frame: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(frame), separators=(",", ":"))
        for connection in targets:
            with suppress(Exception):
                connection.send(encoded)
