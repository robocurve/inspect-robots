"""XPolicyLab websocket wire protocol: envelope, message types, msgpack codec.

This module mirrors ``client_server/ws/protocol/`` in the upstream repo
(https://github.com/XPolicyLab/XPolicyLab) so the adapter can talk to any
XPolicyLab policy server without depending on the git-only ``xpolicylab``
package. Validated against upstream commit
``fe71eb54675cef495fea817a637386a4f4529153``; the protocol carries no version
field, so keep this file small and diffable against upstream.

Wire format: one websocket **binary** message per frame, msgpack-encoded with
``msgpack_numpy`` for arrays. The envelope carries ``message_type``,
``message_id`` (called ``request_id`` in-memory), ``evaluation_id``, optional
``action_case_id``/``trial_id``/``repeat_index``, a per-trial ``step`` counter,
``sent_at``, and a ``payload`` map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np


class MessageType(str, Enum):
    """Frame kinds; request/response pairs are in ``REQUEST_RESPONSE_PAIRS``."""

    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    PREPARE_CASE = "prepare_case"
    PREPARE_CASE_ACK = "prepare_case_ack"
    RESET = "reset"
    RESET_RESULT = "reset_result"
    INFER = "infer"
    INFER_RESULT = "infer_result"
    TRIAL_END = "trial_end"
    TRIAL_END_ACK = "trial_end_ack"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    CLOSE = "close"
    ERROR = "error"


REQUEST_RESPONSE_PAIRS: dict[MessageType, MessageType] = {
    MessageType.HELLO: MessageType.HELLO_ACK,
    MessageType.PREPARE_CASE: MessageType.PREPARE_CASE_ACK,
    MessageType.RESET: MessageType.RESET_RESULT,
    MessageType.INFER: MessageType.INFER_RESULT,
    MessageType.TRIAL_END: MessageType.TRIAL_END_ACK,
    MessageType.HEARTBEAT: MessageType.HEARTBEAT_ACK,
}

# Error codes a server may send in an ``error`` frame (upstream ``ErrorCode``),
# plus the codes this client raises locally ("timeout", "invalid_frame").
KNOWN_ERROR_CODES = frozenset(
    {
        "invalid_frame",
        "unknown_message_type",
        "timeout",
        "infer_failed",
        "reset_failed",
        "internal",
    }
)


class WsError(Exception):
    """A protocol-level failure: server ``error`` frame, timeout, or bad frame."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Frame:
    """One websocket message (request or response)."""

    message_type: MessageType
    request_id: str
    evaluation_id: str
    action_case_id: str | None = None
    trial_id: str | None = None
    repeat_index: int | None = None
    step: int = 0
    sent_at: str = field(default_factory=_utc_now_iso)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """The msgpack-able wire dict (``request_id`` travels as ``message_id``)."""
        return {
            "message_type": self.message_type.value,
            "message_id": self.request_id,
            "evaluation_id": self.evaluation_id,
            "action_case_id": self.action_case_id,
            "trial_id": self.trial_id,
            "repeat_index": self.repeat_index,
            "step": self.step,
            "sent_at": self.sent_at,
            "payload": self.payload,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Frame:
        message_type_value = data.get("message_type")
        if message_type_value is None:
            raise WsError("invalid_frame", "missing message_type")
        try:
            message_type = MessageType(message_type_value)
        except ValueError as exc:
            raise WsError("invalid_frame", f"unknown message_type: {message_type_value!r}") from exc

        message_id = data.get("message_id")
        if message_id is None:
            raise WsError("invalid_frame", "missing message_id")
        evaluation_id = data.get("evaluation_id")
        if evaluation_id is None:
            raise WsError("invalid_frame", "missing evaluation_id")

        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            raise WsError("invalid_frame", "payload must be a map")

        return cls(
            message_type=message_type,
            request_id=str(message_id),
            evaluation_id=str(evaluation_id),
            action_case_id=data.get("action_case_id"),
            trial_id=data.get("trial_id"),
            repeat_index=data.get("repeat_index"),
            step=int(data.get("step", 0)),
            sent_at=str(data.get("sent_at") or _utc_now_iso()),
            payload=dict(payload),
        )


def _encode_hook(obj: Any) -> Any:
    # Same restriction as upstream: object-dtype arrays cannot round-trip.
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind == "O":
        raise WsError("invalid_frame", "object dtype numpy arrays are not supported")
    return msgpack_numpy.encode(obj)


def _decode_hook(obj: dict[Any, Any]) -> Any:
    decoded = msgpack_numpy.decode(obj)
    if isinstance(decoded, np.ndarray) and decoded.dtype.kind == "O":
        raise ValueError("object dtype numpy arrays are not supported")
    return decoded


def encode_frame(frame: Frame | dict[str, Any]) -> bytes:
    """Encode a frame (or a pre-built wire dict) to msgpack bytes."""
    wire = frame.to_wire() if isinstance(frame, Frame) else dict(frame)
    try:
        packed: bytes = msgpack.packb(wire, default=_encode_hook, use_bin_type=True)
    except WsError:
        raise
    except Exception as exc:
        raise WsError("invalid_frame", f"msgpack encode failed: {exc}") from exc
    return packed


def decode_wire(data: bytes) -> dict[str, Any]:
    """Decode msgpack bytes to a wire dict (arrays restored via msgpack_numpy)."""
    try:
        obj = msgpack.unpackb(data, raw=False, object_hook=_decode_hook)
    except Exception as exc:
        raise WsError("invalid_frame", f"msgpack decode failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise WsError("invalid_frame", "frame must be a msgpack map")
    return obj


def decode_frame(data: bytes) -> Frame:
    """Decode msgpack bytes straight to a validated :class:`Frame`."""
    try:
        return Frame.from_wire(decode_wire(data))
    except WsError:
        raise
    except (TypeError, ValueError) as exc:
        raise WsError("invalid_frame", f"invalid frame envelope: {exc}") from exc
