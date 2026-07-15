"""Build and parse the rosbridge v2 JSON operations used by the adapter.

The transport carries one JSON object per websocket text frame. Outbound
builders return ordinary dictionaries so their exact wire shapes stay visible
and testable. Incoming operations are parsed into frozen dataclasses before the
threaded client handles them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]


class RosbridgeError(Exception):
    """A rosbridge protocol failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PublishedMessage:
    """One topic message delivered by a rosbridge ``publish`` operation."""

    topic: str
    msg: JsonObject


@dataclass(frozen=True)
class ServiceResponse:
    """A service reply correlated to a prior ``call_service`` operation."""

    request_id: str
    values: JsonObject
    result: bool


@dataclass(frozen=True)
class StatusMessage:
    """A rosbridge status notification, including asynchronous publish errors."""

    level: str
    message: str
    request_id: str | None = None

    def as_error(self) -> RosbridgeError | None:
        """Return a latchable error only when rosbridge marks this status as an error."""
        if self.level != "error":
            return None
        return RosbridgeError("status_error", self.message)


IncomingMessage = PublishedMessage | ServiceResponse | StatusMessage


def subscribe(
    topic: str,
    *,
    subscription_id: str,
    message_type: str,
    throttle_rate: int,
    queue_length: int = 1,
    compression: str = "none",
) -> JsonObject:
    """Build an explicit-id subscription with latest-value queue semantics."""
    return {
        "op": "subscribe",
        "id": subscription_id,
        "topic": topic,
        "type": message_type,
        "throttle_rate": throttle_rate,
        "queue_length": queue_length,
        "compression": compression,
    }


def unsubscribe(topic: str, *, subscription_id: str) -> JsonObject:
    """Build removal of exactly one id-keyed subscription."""
    return {"op": "unsubscribe", "id": subscription_id, "topic": topic}


def advertise(topic: str, *, message_type: str) -> JsonObject:
    """Build an explicit topic advertisement with its ROS message type."""
    return {"op": "advertise", "topic": topic, "type": message_type}


def unadvertise(topic: str) -> JsonObject:
    """Build removal of a topic advertisement owned by this client."""
    return {"op": "unadvertise", "topic": topic}


def publish(topic: str, msg: Mapping[str, Any]) -> JsonObject:
    """Build one topic publication without altering the ROS message dictionary."""
    return {"op": "publish", "topic": topic, "msg": dict(msg)}


def call_service(
    service: str, *, request_id: str, args: Mapping[str, Any] | None = None
) -> JsonObject:
    """Build an id-correlated service call, using an empty argument object by default."""
    return {
        "op": "call_service",
        "id": request_id,
        "service": service,
        "args": dict(args or {}),
    }


def encode_message(message: Mapping[str, Any]) -> str:
    """Encode one operation as a compact websocket JSON text frame."""
    try:
        return json.dumps(dict(message), separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RosbridgeError("invalid_frame", f"could not encode JSON operation: {exc}") from exc


def decode_message(raw: str | bytes) -> JsonObject:
    """Decode and validate one websocket JSON object."""
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RosbridgeError("invalid_frame", f"could not decode JSON operation: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RosbridgeError("invalid_frame", "rosbridge frame must be a JSON object")
    if not isinstance(decoded.get("op"), str):
        raise RosbridgeError("invalid_frame", "rosbridge frame is missing string field 'op'")
    return decoded


def parse_incoming(message: Mapping[str, Any]) -> IncomingMessage | None:
    """Parse the incoming operation kinds consumed by the client.

    Operations outside ``publish``, ``service_response``, and ``status`` are
    ignored so additions to rosbridge do not break the receive loop.
    """
    op = message.get("op")
    if op == "publish":
        topic = _required_str(message, "topic")
        msg = _required_object(message, "msg")
        return PublishedMessage(topic=topic, msg=msg)
    if op == "service_response":
        request_id = _required_str(message, "id")
        values = _required_object(message, "values")
        result = message.get("result")
        if not isinstance(result, bool):
            raise RosbridgeError(
                "invalid_frame", "service_response field 'result' must be a boolean"
            )
        return ServiceResponse(request_id=request_id, values=values, result=result)
    if op == "status":
        level = _required_str(message, "level")
        status_text = _required_str(message, "msg")
        status_request_id = message.get("id")
        if status_request_id is not None and not isinstance(status_request_id, str):
            raise RosbridgeError("invalid_frame", "status field 'id' must be a string")
        return StatusMessage(level=level, message=status_text, request_id=status_request_id)
    return None


def _required_str(message: Mapping[str, Any], key: str) -> str:
    value = message.get(key)
    if not isinstance(value, str):
        raise RosbridgeError("invalid_frame", f"rosbridge frame field {key!r} must be a string")
    return value


def _required_object(message: Mapping[str, Any], key: str) -> JsonObject:
    value = message.get(key)
    if not isinstance(value, Mapping):
        raise RosbridgeError(
            "invalid_frame", f"rosbridge frame field {key!r} must be a JSON object"
        )
    return dict(value)
