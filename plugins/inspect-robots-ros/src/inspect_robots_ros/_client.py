"""Threaded synchronous client for the rosbridge v2 websocket protocol.

One receive thread owns ``ClientConnection.recv``. Rollout-facing calls only
send, which matches websockets 16's documented concurrency contract: concurrent
send and receive are supported, while concurrent receives raise
``ConcurrencyError``. Topic traffic is reduced to a latest-value slot carrying
the monotonic receive stamp and a per-topic sequence number.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from websockets.sync.client import connect as ws_connect

from inspect_robots_ros._protocol import (
    PublishedMessage,
    RosbridgeError,
    ServiceResponse,
    StatusMessage,
    advertise,
    call_service,
    decode_message,
    encode_message,
    parse_incoming,
    publish,
    subscribe,
    unadvertise,
    unsubscribe,
)

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection


@dataclass(frozen=True)
class TopicSample:
    """The latest message for a topic with receive time and monotonic sequence."""

    msg: dict[str, Any]
    stamp: float
    seq: int


@dataclass
class _PendingService:
    response: ServiceResponse | None = None


class RosbridgeClient:
    """Own one rosbridge socket, one receiver, and latest-value topic caches."""

    def __init__(
        self,
        url: str,
        *,
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._ws: ClientConnection | None = None
        self._receiver_thread: threading.Thread | None = None
        self._topics: dict[str, TopicSample] = {}
        self._pending_services: dict[str, _PendingService] = {}
        self._subscriptions: dict[str, str] = {}
        self._advertisements: set[str] = set()
        self._request_counter = 0
        self._latched_error: Exception | None = None
        self._closing = False
        self._closed = False

    @property
    def connected(self) -> bool:
        """Whether a socket has been established and not explicitly closed."""
        with self._lock:
            return self._ws is not None and not self._closed

    @property
    def receiver_alive(self) -> bool:
        """Whether the sole receive thread is still running."""
        with self._lock:
            thread = self._receiver_thread
        return thread is not None and thread.is_alive()

    @property
    def latched_error(self) -> Exception | None:
        """The first asynchronous protocol or connection failure, if any."""
        with self._lock:
            return self._latched_error

    def connect(self) -> None:
        """Connect once and start the sole receive thread; never reconnect a dead client."""
        with self._lock:
            self._raise_latched_locked()
            if self._closed:
                raise RuntimeError("RosbridgeClient is closed")
            if self._ws is not None:
                return
        try:
            ws = ws_connect(self.url, max_size=None, open_timeout=self.connect_timeout_s)
        except Exception as exc:
            raise ConnectionError(f"could not connect to rosbridge at {self.url}: {exc}") from exc
        with self._lock:
            self._ws = ws
            thread = threading.Thread(
                target=self._receive_loop,
                name="inspect-robots-ros-receiver",
                daemon=True,
            )
            self._receiver_thread = thread
        thread.start()

    def advertise(self, topic: str, *, message_type: str) -> None:
        """Advertise a command topic and remember it for best-effort close cleanup."""
        self._send(advertise(topic, message_type=message_type))
        with self._lock:
            self._advertisements.add(topic)

    def unadvertise(self, topic: str) -> None:
        """Remove one advertisement and its close-cleanup record."""
        self._send(unadvertise(topic))
        with self._lock:
            self._advertisements.discard(topic)

    def subscribe(
        self,
        topic: str,
        *,
        subscription_id: str,
        message_type: str,
        throttle_rate: int,
        queue_length: int = 1,
        compression: str = "none",
    ) -> None:
        """Create one explicit-id subscription and track it for exact removal."""
        operation = subscribe(
            topic,
            subscription_id=subscription_id,
            message_type=message_type,
            throttle_rate=throttle_rate,
            queue_length=queue_length,
            compression=compression,
        )
        self._send(operation)
        with self._lock:
            self._subscriptions[subscription_id] = topic

    def unsubscribe(self, topic: str, *, subscription_id: str) -> None:
        """Remove exactly one id-keyed subscription and its cleanup record."""
        self._send(unsubscribe(topic, subscription_id=subscription_id))
        with self._lock:
            self._subscriptions.pop(subscription_id, None)

    def publish(self, topic: str, msg: Mapping[str, Any]) -> None:
        """Publish one ROS message, surfacing any previously latched async error first."""
        self._send(publish(topic, msg))

    def call_service(self, service: str, args: Mapping[str, Any] | None = None) -> ServiceResponse:
        """Call a service and wait for the response carrying the matching request ID."""
        with self._lock:
            self._check_ready_locked()
            self._request_counter += 1
            request_id = f"inspect-robots-service-{self._request_counter}"
            pending = _PendingService()
            self._pending_services[request_id] = pending
        try:
            self._send(call_service(service, request_id=request_id, args=args))
            deadline = self._clock() + self.request_timeout_s
            while True:
                with self._lock:
                    self._raise_latched_locked()
                    if self._closed:
                        raise ConnectionError(f"rosbridge client for {self.url} closed")
                    response = pending.response
                if response is not None:
                    if not response.result:
                        raise RosbridgeError(
                            "service_failed",
                            f"service {service!r} returned result=false; "
                            f"values={response.values!r}",
                        )
                    return response
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise RosbridgeError(
                        "timeout",
                        f"timed out after {self.request_timeout_s:g}s waiting for "
                        f"service {service!r} from {self.url}",
                    )
                self._sleep(min(0.01, remaining))
        finally:
            with self._lock:
                self._pending_services.pop(request_id, None)

    def latest(self, topic: str) -> TopicSample | None:
        """Read a topic's latest slot after surfacing any latched failure."""
        with self._lock:
            self._check_ready_locked()
            return self._topics.get(topic)

    def sequence(self, topic: str) -> int:
        """Read the current per-topic receive sequence, or zero before the first message."""
        sample = self.latest(topic)
        return sample.seq if sample is not None else 0

    def wait_for_sample(self, topic: str, *, after_seq: int = 0, timeout_s: float) -> TopicSample:
        """Wait until a topic slot has a sequence strictly newer than ``after_seq``."""
        deadline = self._clock() + timeout_s
        while True:
            sample = self.latest(topic)
            if sample is not None and sample.seq > after_seq:
                return sample
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out after {timeout_s:g}s waiting for a new message on {topic!r}"
                )
            self._sleep(min(0.01, remaining))

    def close(self) -> None:
        """Best-effort unsubscribe/unadvertise, close, and join; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            ws = self._ws
            subscriptions = tuple(self._subscriptions.items())
            advertisements = tuple(self._advertisements)
            thread = self._receiver_thread
        if ws is not None:
            for subscription_id, topic in subscriptions:
                self._send_best_effort(ws, unsubscribe(topic, subscription_id=subscription_id))
            for topic in advertisements:
                self._send_best_effort(ws, unadvertise(topic))
            with suppress(Exception):
                ws.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(max(self.connect_timeout_s, 1.0), 5.0))
        with self._lock:
            self._ws = None
            self._subscriptions.clear()
            self._advertisements.clear()
            self._closed = True

    def _send(self, operation: Mapping[str, Any]) -> None:
        with self._lock:
            self._check_ready_locked()
            ws = self._ws
            assert ws is not None
        try:
            ws.send(encode_message(operation))
        except Exception as exc:
            error = ConnectionError(
                f"connection to rosbridge at {self.url} lost while sending: {exc}"
            )
            self._latch(error)
            raise error from exc

    @staticmethod
    def _send_best_effort(ws: ClientConnection, operation: Mapping[str, Any]) -> None:
        with suppress(Exception):
            ws.send(encode_message(operation))

    def _receive_loop(self) -> None:
        with self._lock:
            ws = self._ws
        assert ws is not None
        try:
            while True:
                raw = ws.recv()
                if not isinstance(raw, (str, bytes)):
                    raise RosbridgeError(
                        "invalid_frame",
                        f"rosbridge sent unsupported frame type {type(raw).__name__}",
                    )
                incoming = parse_incoming(decode_message(raw))
                if isinstance(incoming, PublishedMessage):
                    with self._lock:
                        previous = self._topics.get(incoming.topic)
                        seq = previous.seq + 1 if previous is not None else 1
                        self._topics[incoming.topic] = TopicSample(
                            msg=incoming.msg,
                            stamp=self._clock(),
                            seq=seq,
                        )
                elif isinstance(incoming, ServiceResponse):
                    with self._lock:
                        pending = self._pending_services.get(incoming.request_id)
                        if pending is not None:
                            pending.response = incoming
                elif isinstance(incoming, StatusMessage):
                    status_error = incoming.as_error()
                    if status_error is not None:
                        self._latch(status_error)
                        return
        except Exception as exc:
            with self._lock:
                closing = self._closing
            if not closing:
                if isinstance(exc, RosbridgeError):
                    receive_error: Exception = exc
                else:
                    receive_error = ConnectionError(
                        f"connection to rosbridge at {self.url} lost while receiving: {exc}"
                    )
                self._latch(receive_error)

    def _latch(self, error: Exception) -> None:
        with self._lock:
            if self._latched_error is None:
                self._latched_error = error

    def _check_ready_locked(self) -> None:
        self._raise_latched_locked()
        if self._closed:
            raise RuntimeError("RosbridgeClient is closed")
        if self._ws is None:
            raise ConnectionError(f"not connected to rosbridge at {self.url}; call connect() first")

    def _raise_latched_locked(self) -> None:
        if self._latched_error is not None:
            raise self._latched_error
