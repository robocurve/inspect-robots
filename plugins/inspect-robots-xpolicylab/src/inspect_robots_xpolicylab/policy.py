"""The ``Policy`` adapter that drives an XPolicyLab policy server.

This is the "brain" half of an Inspect Robots eval, backed by any of
XPolicyLab's 40+ served policies. It conforms to :class:`inspect_robots.Policy`
(the runtime-checkable protocol), so once installed it can be paired with any
compatible :class:`inspect_robots.Embodiment` and run through
``inspect_robots.eval``.

Design mirrors the isaacsim plugin's laziness: constructing the adapter and
reading ``.info`` never touch the network (so ``inspect-robots list policies``
and fail-fast compatibility checks work with no server running); the websocket
connects on first ``reset()``/``act()``. If the socket dies mid-eval, the next
call reconnects once (replaying the ``hello`` handshake) before failing — a
dead server costs one errored trial, not every remaining one.

Observation/action mapping (XPolicyLab Observation Data Format v1.0):

- ``Observation.images[name]`` → ``vision/<slot>/color`` per the ``cameras``
  map; images pass through as ``(H, W, 3)`` uint8 RGB (both sides agree).
- ``Observation.state[key]`` → ``state/<xpl_key>`` per ``state_map``; mapped
  keys absent from the observation are skipped (all v1.0 state fields are
  optional).
- Reply ``actions`` (one dict per future control step) flatten to one vector
  per step in a fixed order — for each arm (left then right): arm key then
  end-effector key — forming the returned ``ActionChunk``.
"""

from __future__ import annotations

import atexit
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any
from uuid import uuid4

import numpy as np

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    CameraSpec,
    Observation,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots_xpolicylab._client import PolicyClient
from inspect_robots_xpolicylab._protocol import WsError

_EE_POSE_DIM = 7  # [x, y, z, qw, qx, qy, qz]

# XPolicyLab state key -> Inspect Robots canonical state key (see
# `inspect_robots.spaces.CANONICAL_STATE_UNITS`). Single-arm family; dual-arm
# setups pass their own map with `left_*`/`right_*` keys.
_DEFAULT_STATE_MAP: dict[str, str] = {
    "arm_joint_state": "joint_pos",
    "ee_joint_state": "gripper",
    "ee_pose": "eef_pose",
}


def _parse_str_mapping(value: str, arg: str) -> dict[str, str]:
    """Parse the compact ``"key:value,key:value"`` form used by CLI ``-P`` args."""
    out: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition(":")
        if not sep or not key.strip() or not val.strip():
            raise ValueError(
                f"{arg} entry {item!r} is not 'key:value'; expected e.g. "
                "'cam_head:base_rgb,cam_wrist:wrist_rgb'"
            )
        out[key.strip()] = val.strip()
    if not out:
        raise ValueError(f"{arg} string form parsed to an empty mapping: {value!r}")
    return out


def _as_mapping(value: Mapping[str, str] | str, arg: str) -> dict[str, str]:
    if isinstance(value, str):
        return _parse_str_mapping(value, arg)
    return dict(value)


def _as_str_tuple(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


class XPolicyLabPolicy:
    """An Inspect Robots ``Policy`` served by an XPolicyLab policy server."""

    def __init__(
        self,
        *,
        url: str = "ws://localhost:19000",
        action_type: str = "joint",
        arms: int = 1,
        arm_dim: int = 7,
        ee_dim: int = 1,
        cameras: Mapping[str, str] | str | None = None,
        state_map: Mapping[str, str] | str | None = None,
        required_state_keys: Sequence[str] | str | None = None,
        action_keys: Sequence[str] | str | None = None,
        action_dim: int | None = None,
        camera_height: int | None = None,
        camera_width: int | None = None,
        control_hz: float | None = None,
        name: str = "xpolicylab",
        evaluation_id: str | None = None,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 120.0,
        connect_attempts: int = 10,
        connect_retry_delay_s: float = 5.0,
    ) -> None:
        if action_type not in ("joint", "ee"):
            raise ValueError(f"action_type must be 'joint' or 'ee', got {action_type!r}")
        if arms not in (1, 2):
            raise ValueError(f"arms must be 1 or 2, got {arms!r}")
        if arm_dim < 1:
            raise ValueError(f"arm_dim must be >= 1, got {arm_dim}")
        if ee_dim < 0:
            raise ValueError(f"ee_dim must be >= 0, got {ee_dim}")
        if (camera_height is None) != (camera_width is None):
            raise ValueError("camera_height and camera_width must be given together")

        self.url = url
        self.action_type = action_type
        self._cameras = _as_mapping(
            cameras if cameras is not None else {"cam_head": "cam_head"}, "cameras"
        )
        self._state_map = _as_mapping(
            state_map if state_map is not None else _DEFAULT_STATE_MAP, "state_map"
        )
        if required_state_keys is not None:
            required = frozenset(_as_str_tuple(required_state_keys))
        elif action_type == "joint":
            required = frozenset({"joint_pos", "gripper"})
        else:
            required = frozenset()
        self._required_state_keys = required

        if action_keys is not None:
            self._action_keys = _as_str_tuple(action_keys)
            if action_dim is None:
                raise ValueError("action_dim is required when action_keys is given")
            dim = action_dim
        else:
            per_arm = (arm_dim if action_type == "joint" else _EE_POSE_DIM) + ee_dim
            dim = arms * per_arm
            arm_key = "arm_joint_state" if action_type == "joint" else "ee_pose"
            prefixes = ("",) if arms == 1 else ("left_", "right_")
            keys: list[str] = []
            for prefix in prefixes:
                keys.append(prefix + arm_key)
                if ee_dim > 0:
                    keys.append(prefix + "ee_joint_state")
            self._action_keys = tuple(keys)

        semantics = ActionSemantics(
            control_mode="joint_pos" if action_type == "joint" else "eef_abs_pose",
            rotation_repr="none" if action_type == "joint" else "quat_wxyz",
            gripper="continuous" if ee_dim > 0 else "none",
        )
        camera_specs: tuple[CameraSpec, ...] = ()
        if camera_height is not None and camera_width is not None:
            camera_specs = tuple(
                CameraSpec(name=cam, height=camera_height, width=camera_width)
                for cam in self._cameras.values()
            )
        self.info = PolicyInfo(
            name=name,
            action_space=Box(shape=(dim,), semantics=semantics),
            observation_space=ObservationSpace(
                cameras=camera_specs, state_keys=self._required_state_keys
            ),
            control_hz=control_hz,
        )
        self.config = PolicyConfig()

        self._client = PolicyClient(
            url,
            evaluation_id or f"inspect-robots-{uuid4()}",
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
            connect_attempts=connect_attempts,
            connect_retry_delay_s=connect_retry_delay_s,
        )
        self._instruction: str | None = None
        self._open_trial_id: str | None = None
        self._trial_count = 0
        self._step = 0
        self._closed = False
        # `eval()` closes embodiments it resolves, not policies; the atexit hook
        # is the safety net for registry-resolved CLI runs. Unregistered on
        # explicit close() so instances don't accumulate across a test suite.
        atexit.register(self._atexit_close)

    # ------------------------------------------------------------------ #
    # Policy protocol
    # ------------------------------------------------------------------ #

    def reset(self, scene: Scene) -> None:
        """End any open trial server-side, then start a fresh one for ``scene``."""
        if self._closed:
            raise RuntimeError("XPolicyLabPolicy is closed")
        client = self._ensure_connected()
        if self._open_trial_id is not None:
            open_trial, self._open_trial_id = self._open_trial_id, None
            client.trial_end(open_trial)
        self._instruction = scene.instruction
        self._trial_count += 1
        trial_id = f"{scene.id}-{self._trial_count}"
        self._step = 0
        client.reset(trial_id)
        self._open_trial_id = trial_id

    def act(self, observation: Observation) -> ActionChunk:
        """One inference round-trip: observation dict out, action chunk back."""
        if self._closed:
            raise RuntimeError("XPolicyLabPolicy is closed")
        if self._open_trial_id is None:
            raise RuntimeError("act() called before reset(); call reset(scene) first")
        client = self._ensure_connected()
        obs_dict = self._to_xpolicylab_observation(observation)
        start = time.perf_counter()
        reply = client.infer(obs_dict, trial_id=self._open_trial_id, step=self._step)
        wall_s = time.perf_counter() - start
        self._step += 1

        raw_actions = reply.payload.get("actions")
        if not isinstance(raw_actions, Sequence) or len(raw_actions) == 0:
            raise WsError(
                "infer_failed",
                f"policy server returned no actions (trial {self._open_trial_id}, "
                f"step {self._step - 1}); got {raw_actions!r}",
            )
        actions = [Action(data=self._flatten_action(step)) for step in raw_actions]
        meta: dict[str, Any] = {}
        if "latency_ms" in reply.payload:
            meta["server_latency_ms"] = float(reply.payload["latency_ms"])
        return ActionChunk(
            actions=actions,
            control_hz=self.info.control_hz,
            inference_latency_s=wall_s,
            meta=meta,
        )

    def close(self) -> None:
        """End any open trial, say goodbye, drop the socket; idempotent."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_close)
        if self._client.connected and self._open_trial_id is not None:
            with suppress(Exception):
                self._client.trial_end(self._open_trial_id)
        self._open_trial_id = None
        self._client.close()

    def __enter__(self) -> XPolicyLabPolicy:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _atexit_close(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_connected(self) -> PolicyClient:
        """Connect lazily; after a socket drop, reconnect (replaying ``hello``)."""
        if not self._client.connected:
            self._client.connect()
        return self._client

    def _to_xpolicylab_observation(self, observation: Observation) -> dict[str, Any]:
        """Build an XPolicyLab Observation Data Format v1.0 dict."""
        out: dict[str, Any] = {"data_format_version": "v1.0"}
        instruction = observation.instruction or self._instruction
        if instruction is not None:
            out["instruction"] = instruction

        vision: dict[str, Any] = {}
        for slot, camera in self._cameras.items():
            image = observation.images.get(camera)
            if image is None:
                raise KeyError(
                    f"observation has no camera {camera!r} (mapped to XPolicyLab slot "
                    f"{slot!r}); available cameras: {sorted(observation.images)}. "
                    "Adjust the cameras= mapping, e.g. -P cameras=cam_head:base_rgb"
                )
            vision[slot] = {"color": image}
        if vision:
            out["vision"] = vision

        state: dict[str, Any] = {}
        for xpl_key, ir_key in self._state_map.items():
            value = observation.state.get(ir_key)
            if value is not None:
                state[xpl_key] = np.asarray(value)
        if state:
            out["state"] = state

        if self.info.control_hz is not None:
            out["additional_info"] = {"frequency": int(self.info.control_hz)}
        return out

    def _flatten_action(self, step: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Flatten one per-step action dict into a vector in the declared key order."""
        if not isinstance(step, Mapping):
            raise WsError(
                "infer_failed", f"expected an action dict per step, got {type(step).__name__}"
            )
        parts: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for key in self._action_keys:
            value = step.get(key)
            if value is None:
                raise WsError(
                    "infer_failed",
                    f"action dict is missing key {key!r}; got keys {sorted(step)}. "
                    "Adjust action_type/arms or pass action_keys=... explicitly",
                )
            parts.append(np.asarray(value, dtype=np.float64).ravel())
        vector = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        expected = self.info.action_space.shape[0]
        if vector.shape != (expected,):
            raise WsError(
                "infer_failed",
                f"flattened action has shape {vector.shape}, declared action space "
                f"expects ({expected},); check arm_dim/ee_dim/arms/action_type",
            )
        return vector
