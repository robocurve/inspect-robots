"""An in-process, XPolicyLab-shaped stub policy server for the plugin tests.

Implements the websocket protocol the way upstream's ``PolicyServer`` does —
hello/reset/infer/trial_end/close over msgpack binary frames — and returns
``demo_policy``-shaped action chunks with deterministic values, so tests can
assert on exact flattening order. Special instructions steer failure modes:
``"error"`` → server error frame, ``"hang"`` → no reply, ``"drop"`` → the
server closes the connection.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from websockets.sync.server import Server, ServerConnection, serve

from inspect_robots_xpolicylab._protocol import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
)


class StubPolicyServer:
    """Serves one XPolicyLab policy-server protocol endpoint on a free port."""

    def __init__(
        self,
        *,
        action_type: str = "joint",
        arms: int = 1,
        arm_dim: int = 7,
        ee_dim: int = 1,
        horizon: int = 2,
        latency_ms: float | None = 12.5,
    ) -> None:
        self.action_type = action_type
        self.arms = arms
        self.arm_dim = arm_dim
        self.ee_dim = ee_dim
        self.horizon = horizon
        self.latency_ms = latency_ms
        self.frames: list[Frame] = []
        self._server: Server = serve(self._handler, "127.0.0.1", 0, max_size=None)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        port = self._server.socket.getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    def frames_of(self, message_type: MessageType) -> list[Frame]:
        return [f for f in self.frames if f.message_type is message_type]

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #

    def _handler(self, ws: ServerConnection) -> None:
        for raw in ws:
            if not isinstance(raw, (bytes, bytearray)):
                continue
            frame = decode_frame(bytes(raw))
            self.frames.append(frame)
            if frame.message_type is MessageType.HELLO:
                self._reply(ws, frame, MessageType.HELLO_ACK, {"ok": True, "server": "stub"})
            elif frame.message_type is MessageType.RESET:
                self._reply(ws, frame, MessageType.RESET_RESULT, {"ok": True})
            elif frame.message_type is MessageType.INFER:
                observation = frame.payload.get("observation") or {}
                instruction = observation.get("instruction")
                if instruction == "error":
                    self._reply(
                        ws,
                        frame,
                        MessageType.ERROR,
                        {"code": "infer_failed", "message": "boom", "details": {"who": "stub"}},
                    )
                elif instruction == "hang":
                    continue
                elif instruction == "drop":
                    ws.close()
                    return
                else:
                    payload: dict[str, Any] = {"actions": self._make_actions()}
                    if self.latency_ms is not None:
                        payload["latency_ms"] = self.latency_ms
                    self._reply(ws, frame, MessageType.INFER_RESULT, payload)
            elif frame.message_type is MessageType.TRIAL_END:
                self._reply(ws, frame, MessageType.TRIAL_END_ACK, {"ok": True})
            elif frame.message_type is MessageType.CLOSE:
                ws.close()
                return

    def _reply(
        self,
        ws: ServerConnection,
        request: Frame,
        message_type: MessageType,
        payload: dict[str, Any],
    ) -> None:
        ws.send(
            encode_frame(
                Frame(
                    message_type=message_type,
                    request_id=request.request_id,
                    evaluation_id=request.evaluation_id,
                    action_case_id=request.action_case_id,
                    trial_id=request.trial_id,
                    step=request.step,
                    payload=payload,
                )
            )
        )

    def _make_actions(self) -> list[dict[str, Any]]:
        """``horizon`` demo_policy-shaped steps with per-step-distinct values.

        Step ``i`` puts ``i`` in every arm slot and ``100 + i`` in every
        end-effector slot, so tests can assert exact flattening order.
        """
        arm_key = "arm_joint_state" if self.action_type == "joint" else "ee_pose"
        arm_n = self.arm_dim if self.action_type == "joint" else 7
        prefixes = ("",) if self.arms == 1 else ("left_", "right_")
        steps: list[dict[str, Any]] = []
        for i in range(self.horizon):
            step: dict[str, Any] = {}
            for prefix in prefixes:
                step[prefix + arm_key] = np.full(arm_n, float(i), dtype=np.float32)
                if self.ee_dim > 0:
                    step[prefix + "ee_joint_state"] = np.full(
                        self.ee_dim, 100.0 + i, dtype=np.float32
                    )
            steps.append(step)
        return steps
