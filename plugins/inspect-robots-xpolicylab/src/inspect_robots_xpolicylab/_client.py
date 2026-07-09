"""A synchronous client for the XPolicyLab policy-server websocket protocol.

Deliberately simpler than upstream's asyncio ``PolicyEvalClient``: Inspect
Robots's rollout loop is synchronous with one in-flight request at a time, so
this client blocks on ``websockets.sync`` and does request/response matching
by ``message_id``. Reconnection is not transparent — a dead socket marks the
client disconnected and raises; the adapter reconnects on next use (replaying
the ``hello`` handshake via :meth:`connect`).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as ws_connect

from inspect_robots_xpolicylab._protocol import (
    REQUEST_RESPONSE_PAIRS,
    Frame,
    MessageType,
    WsError,
    decode_frame,
    encode_frame,
)

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

logger = logging.getLogger(__name__)

# Connection-level failures that mean "the socket is gone" (vs. protocol errors).
_CONNECTION_ERRORS = (WebSocketException, OSError, EOFError)


class PolicyClient:
    """Blocking request/response client for one XPolicyLab policy server."""

    def __init__(
        self,
        url: str,
        evaluation_id: str,
        *,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 120.0,
        connect_attempts: int = 10,
        connect_retry_delay_s: float = 5.0,
    ) -> None:
        self.url = url
        self.evaluation_id = evaluation_id
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s
        self.connect_attempts = max(1, connect_attempts)
        self.connect_retry_delay_s = connect_retry_delay_s
        self._ws: ClientConnection | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def connect(self) -> None:
        """Connect and perform the ``hello`` handshake; no-op when connected.

        Retries with a delay: policy cold-start (model weights loading) can
        take minutes before the server port is ready.
        """
        if self._ws is not None:
            return
        last_err: Exception | None = None
        for attempt in range(1, self.connect_attempts + 1):
            try:
                self._ws = ws_connect(self.url, max_size=None, open_timeout=self.connect_timeout_s)
                break
            except Exception as exc:  # noqa: BLE001 - retry any connect failure
                last_err = exc
                logger.debug(
                    "connect attempt %s/%s to %s failed: %s",
                    attempt,
                    self.connect_attempts,
                    self.url,
                    exc,
                )
                if attempt < self.connect_attempts:
                    time.sleep(self.connect_retry_delay_s)
        if self._ws is None:
            raise ConnectionError(
                f"could not connect to XPolicyLab policy server at {self.url} after "
                f"{self.connect_attempts} attempts ({last_err}). Start one from your "
                "XPolicyLab checkout, e.g.: cd XPolicyLab/policy/<POLICY> && "
                "bash setup_eval_policy_server.sh ... <port> 0.0.0.0"
            ) from last_err
        self.request(MessageType.HELLO, {})

    def request(
        self,
        message_type: MessageType,
        payload: dict[str, Any],
        *,
        trial_id: str | None = None,
        action_case_id: str | None = None,
        step: int = 0,
    ) -> Frame:
        """Send one request frame and block for its paired response.

        Server ``error`` frames raise :class:`WsError` with the server's code;
        connection loss marks the client disconnected and raises
        ``ConnectionError``. ``close`` has no paired response — the request
        frame itself is returned.
        """
        ws = self._ws
        if ws is None:
            raise ConnectionError(f"not connected to {self.url}; call connect() first")
        frame = Frame(
            message_type=message_type,
            request_id=str(uuid4()),
            evaluation_id=self.evaluation_id,
            action_case_id=action_case_id,
            trial_id=trial_id,
            step=step,
            payload=payload,
        )
        expected = REQUEST_RESPONSE_PAIRS.get(message_type)
        if expected is None and message_type is not MessageType.CLOSE:
            raise ValueError(f"no response pairing for {message_type.value}")
        try:
            ws.send(encode_frame(frame))
            if expected is None:
                return frame
            deadline = time.monotonic() + self.request_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                raw = ws.recv(timeout=remaining)
                if not isinstance(raw, (bytes, bytearray)):
                    continue  # protocol is binary-only; ignore stray text frames
                try:
                    reply = decode_frame(bytes(raw))
                except WsError as exc:
                    logger.error("invalid frame from %s: %s", self.url, exc)
                    continue
                if reply.request_id != frame.request_id:
                    continue  # a response to some other (abandoned) request
                if reply.message_type is MessageType.ERROR:
                    err = reply.payload
                    raise WsError(
                        str(err.get("code", "internal")),
                        str(err.get("message", "policy server error")),
                        details=dict(err.get("details") or {}),
                    )
                if reply.message_type is not expected:
                    raise WsError(
                        "invalid_frame",
                        f"expected {expected.value}, got {reply.message_type.value}",
                    )
                return reply
        except TimeoutError as exc:
            raise WsError(
                "timeout",
                f"timed out after {self.request_timeout_s:g}s waiting for "
                f"{expected.value if expected else message_type.value} from {self.url}",
            ) from exc
        except _CONNECTION_ERRORS as exc:
            self._abandon()
            raise ConnectionError(
                f"connection to XPolicyLab policy server {self.url} lost during "
                f"{message_type.value}: {exc}"
            ) from exc

    def reset(self, trial_id: str) -> Frame:
        return self.request(MessageType.RESET, {"trial_id": trial_id}, trial_id=trial_id)

    def infer(self, observation: dict[str, Any], *, trial_id: str, step: int) -> Frame:
        return self.request(
            MessageType.INFER, {"observation": observation}, trial_id=trial_id, step=step
        )

    def trial_end(self, trial_id: str) -> Frame:
        return self.request(MessageType.TRIAL_END, {"trial_id": trial_id}, trial_id=trial_id)

    def close(self) -> None:
        """Send protocol ``close`` (best-effort) and drop the socket; idempotent."""
        ws = self._ws
        if ws is None:
            return
        try:
            self.request(MessageType.CLOSE, {"reason": "client closed"})
        except Exception:  # noqa: BLE001 - close must never raise
            logger.debug("close frame to %s failed", self.url, exc_info=True)
        try:
            ws.close()
        except Exception:  # noqa: BLE001 - close must never raise
            logger.debug("socket close to %s failed", self.url, exc_info=True)
        finally:
            self._ws = None

    def _abandon(self) -> None:
        """Drop a dead socket without protocol goodbyes."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - already dead
                pass
