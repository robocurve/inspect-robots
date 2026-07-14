"""Tests for the XPolicyLab policy adapter.

These run anywhere — no GPU, no XPolicyLab checkout, no external processes.
The stub server in ``conftest.py`` speaks the real wire protocol in-process.
"""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from _stub_server import StubPolicyServer
from inspect_robots_xpolicylab import XPolicyLabPolicy, xpolicylab_policy
from inspect_robots_xpolicylab._client import PolicyClient
from inspect_robots_xpolicylab._protocol import (
    Frame,
    MessageType,
    WsError,
    decode_frame,
    decode_wire,
    encode_frame,
)

from inspect_robots import (
    ActionSemantics,
    Box,
    EmbodimentInfo,
    ObservationSpace,
    Policy,
    Scene,
    StateField,
    StateSpec,
)
from inspect_robots.compat import check_compatibility
from inspect_robots.errors import CompatibilityError
from inspect_robots.registry import registered, resolve
from inspect_robots.types import Observation

_SCENE = Scene(id="s0", instruction="stack the bowls")


def _observation(instruction: str | None = None) -> Observation:
    return Observation(
        images={"base_rgb": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)},
        state={"joint_pos": np.arange(7, dtype=np.float64), "gripper": np.array([0.5])},
        instruction=instruction,
    )


def _policy(server: StubPolicyServer, **kwargs: Any) -> XPolicyLabPolicy:
    kwargs.setdefault("url", server.url)
    kwargs.setdefault("cameras", {"cam_head": "base_rgb"})
    kwargs.setdefault("connect_attempts", 1)
    kwargs.setdefault("connect_retry_delay_s", 0.01)
    kwargs.setdefault("request_timeout_s", 5.0)
    return XPolicyLabPolicy(**kwargs)


# --------------------------------------------------------------------------- #
# Registry, protocol conformance, and info — no server needed
# --------------------------------------------------------------------------- #


def test_registered_via_entry_point() -> None:
    assert "xpolicylab" in registered("policy")
    policy = resolve("policy", "xpolicylab", url="ws://localhost:1")
    assert isinstance(policy, XPolicyLabPolicy)
    policy.close()


def test_satisfies_policy_protocol() -> None:
    policy = xpolicylab_policy()
    assert isinstance(policy, Policy)
    policy.close()


def test_info_default_joint_profile() -> None:
    with XPolicyLabPolicy(control_hz=30.0) as policy:
        assert policy.info.name == "xpolicylab"
        # 7 arm joints + 1 gripper, matching the isaacsim default Franka profile.
        assert policy.info.action_space.dim == 8
        semantics = policy.info.action_space.semantics
        assert semantics is not None
        assert semantics.control_mode == "joint_pos"
        assert semantics.gripper == "continuous"
        assert policy.info.observation_space.state_keys == frozenset({"joint_pos", "gripper"})
        assert policy.info.control_hz == 30.0


def test_info_ee_profile() -> None:
    with XPolicyLabPolicy(action_type="ee") as policy:
        # 7-D pose [x, y, z, qw, qx, qy, qz] + 1 gripper.
        assert policy.info.action_space.dim == 8
        semantics = policy.info.action_space.semantics
        assert semantics is not None
        assert semantics.control_mode == "eef_abs_pose"
        assert semantics.rotation_repr == "quat_wxyz"
        # ee mode declares no required state by default.
        assert policy.info.observation_space.state_keys == frozenset()


def test_info_dual_arm_and_camera_specs() -> None:
    with XPolicyLabPolicy(
        arms=2,
        cameras="cam_head:base_rgb,cam_left_wrist:left_rgb",
        camera_height=224,
        camera_width=224,
    ) as policy:
        assert policy.info.action_space.dim == 16
        assert policy.info.observation_space.camera_names == frozenset({"base_rgb", "left_rgb"})


def test_construct_and_info_touch_no_network() -> None:
    # An unroutable URL: constructing and reading .info must not connect.
    with XPolicyLabPolicy(url="ws://192.0.2.1:19000") as policy:
        assert policy.info.action_space.dim == 8


def test_string_forms_parse() -> None:
    with XPolicyLabPolicy(
        cameras="cam_head:base_rgb",
        state_map="arm_joint_state:joint_pos",
        required_state_keys="joint_pos",
        action_keys="arm_joint_state,ee_joint_state",
        action_dim=8,
    ) as policy:
        assert policy.info.observation_space.state_keys == frozenset({"joint_pos"})
        assert policy.info.action_space.dim == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_type": "torque"},
        {"arms": 3},
        {"arm_dim": 0},
        {"ee_dim": -1},
        {"camera_height": 224},
        {"action_keys": ("arm_joint_state",)},  # action_dim missing
        {"cameras": "cam_head=base_rgb"},  # not key:value
        {"cameras": " , "},  # empty mapping
    ],
)
def test_invalid_constructor_args(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        XPolicyLabPolicy(**kwargs)


# --------------------------------------------------------------------------- #
# Rollout flow against the stub server
# --------------------------------------------------------------------------- #


def test_reset_act_returns_ordered_chunk(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server, control_hz=15.0) as policy:
        policy.reset(_SCENE)
        chunk = policy.act(_observation())
        assert len(chunk) == 2  # stub horizon
        # Step i: arm slots = i, ee slot = 100 + i — asserts arm-then-ee order.
        np.testing.assert_allclose(chunk.actions[0].data, [0.0] * 7 + [100.0])
        np.testing.assert_allclose(chunk.actions[1].data, [1.0] * 7 + [101.0])
        assert chunk.control_hz == 15.0
        assert chunk.inference_latency_s is not None and chunk.inference_latency_s > 0
        assert chunk.meta["server_latency_ms"] == 12.5


def test_dual_arm_flatten_order() -> None:
    server = StubPolicyServer(arms=2, horizon=1)
    try:
        with _policy(server, arms=2) as policy:
            policy.reset(_SCENE)
            chunk = policy.act(_observation())
            # left arm, left ee, right arm, right ee.
            np.testing.assert_allclose(
                chunk.actions[0].data, [0.0] * 7 + [100.0] + [0.0] * 7 + [100.0]
            )
    finally:
        server.stop()


def test_observation_translation(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server, control_hz=30.0) as policy:
        policy.reset(_SCENE)
        observation = _observation()
        policy.act(observation)
        infer = stub_server.frames_of(MessageType.INFER)[-1]
        sent = infer.payload["observation"]
        assert sent["data_format_version"] == "v1.0"
        assert sent["instruction"] == "stack the bowls"  # scene instruction fills in
        np.testing.assert_array_equal(
            sent["vision"]["cam_head"]["color"], observation.images["base_rgb"]
        )
        np.testing.assert_array_equal(
            sent["state"]["arm_joint_state"], observation.state["joint_pos"]
        )
        np.testing.assert_array_equal(sent["state"]["ee_joint_state"], [0.5])
        assert "ee_pose" not in sent["state"]  # unmapped-in-observation keys are skipped
        assert sent["additional_info"] == {"frequency": 30}


def test_observation_instruction_wins_and_no_frequency(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server) as policy:  # control_hz unset
        policy.reset(_SCENE)
        policy.act(_observation(instruction="pick up the fork"))
        sent = stub_server.frames_of(MessageType.INFER)[-1].payload["observation"]
        assert sent["instruction"] == "pick up the fork"
        assert "additional_info" not in sent


def test_trial_lifecycle_and_step_counter(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server) as policy:
        policy.reset(_SCENE)
        policy.act(_observation())
        policy.act(_observation())
        policy.reset(Scene(id="s1", instruction="next"))
        policy.act(_observation())

        resets = stub_server.frames_of(MessageType.RESET)
        assert [f.trial_id for f in resets] == ["s0-1", "s1-2"]
        # The second reset ends the first trial before starting the next one.
        trial_ends = stub_server.frames_of(MessageType.TRIAL_END)
        assert [f.trial_id for f in trial_ends] == ["s0-1"]
        # Envelope step: zeroed per trial, incremented per infer.
        assert [f.step for f in stub_server.frames_of(MessageType.INFER)] == [0, 1, 0]


def test_act_before_reset_raises(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server) as policy, pytest.raises(RuntimeError, match="reset"):
        policy.act(_observation())


def test_missing_camera_raises(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server, cameras={"cam_head": "head_rgb"}) as policy:
        policy.reset(_SCENE)
        with pytest.raises(KeyError, match="head_rgb"):
            policy.act(_observation())


def test_missing_action_key_raises(stub_server: StubPolicyServer) -> None:
    # Stub serves joint-mode keys; a policy expecting ee keys must fail clearly.
    with _policy(stub_server, action_type="ee") as policy:
        policy.reset(_SCENE)
        with pytest.raises(WsError, match="ee_pose"):
            policy.act(_observation())


def test_action_shape_mismatch_raises(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server, arm_dim=6) as policy:  # stub sends 7-D arms
        policy.reset(_SCENE)
        with pytest.raises(WsError, match="shape"):
            policy.act(_observation())


def test_empty_actions_raises() -> None:
    server = StubPolicyServer(horizon=0)
    try:
        with _policy(server) as policy:
            policy.reset(_SCENE)
            with pytest.raises(WsError, match="no actions"):
                policy.act(_observation())
    finally:
        server.stop()


def test_server_error_frame_surfaces(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server) as policy:
        policy.reset(_SCENE)
        with pytest.raises(WsError, match="boom") as excinfo:
            policy.act(_observation(instruction="error"))
        assert excinfo.value.code == "infer_failed"
        assert excinfo.value.details == {"who": "stub"}


def test_request_timeout(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server, request_timeout_s=0.3) as policy:
        policy.reset(_SCENE)
        with pytest.raises(WsError, match="timed out") as excinfo:
            policy.act(_observation(instruction="hang"))
        assert excinfo.value.code == "timeout"


def test_unreachable_server_actionable_error() -> None:
    with (
        XPolicyLabPolicy(
            url="ws://127.0.0.1:9", connect_attempts=2, connect_retry_delay_s=0.01
        ) as policy,
        pytest.raises(ConnectionError, match=r"setup_eval_policy_server\.sh"),
    ):
        policy.reset(_SCENE)


def test_reconnect_after_drop(stub_server: StubPolicyServer) -> None:
    with _policy(stub_server) as policy:
        policy.reset(_SCENE)
        with pytest.raises(ConnectionError):
            policy.act(_observation(instruction="drop"))
        # Next use reconnects (replaying hello) and the eval continues.
        policy.reset(Scene(id="s1", instruction="retry"))
        chunk = policy.act(_observation())
        assert len(chunk) == 2
    assert len(stub_server.frames_of(MessageType.HELLO)) == 2


def test_close_ends_trial_and_is_idempotent(stub_server: StubPolicyServer) -> None:
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    policy.close()
    policy.close()  # second close is a no-op
    assert [f.trial_id for f in stub_server.frames_of(MessageType.TRIAL_END)] == ["s0-1"]
    assert len(stub_server.frames_of(MessageType.CLOSE)) == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy.act(_observation())


def test_close_before_connect_is_safe() -> None:
    policy = XPolicyLabPolicy(url="ws://127.0.0.1:9")
    policy.close()
    policy.close()


def test_atexit_hook_registered_and_unregistered(monkeypatch: pytest.MonkeyPatch) -> None:
    import atexit

    registered_cbs: list[Any] = []
    unregistered_cbs: list[Any] = []
    monkeypatch.setattr(atexit, "register", lambda cb, *a, **k: registered_cbs.append(cb) or cb)
    monkeypatch.setattr(atexit, "unregister", lambda cb: unregistered_cbs.append(cb))
    policy = XPolicyLabPolicy()
    policy.close()
    assert registered_cbs == [policy._atexit_close]
    assert unregistered_cbs == [policy._atexit_close]


# --------------------------------------------------------------------------- #
# Compatibility with an isaacsim-like embodiment profile
# --------------------------------------------------------------------------- #


class _FrankaLikeEmbodiment:
    """Static info matching the isaacsim plugin's default Franka profile."""

    def __init__(self, action_dim: int = 8) -> None:
        state = StateSpec(
            fields=(
                StateField(key="joint_pos", shape=(7,), unit="rad"),
                StateField(key="joint_vel", shape=(7,), unit="rad/s"),
                StateField(key="eef_pos", shape=(3,), unit="m"),
                StateField(key="eef_quat", shape=(4,), unit="unit_quat"),
                StateField(key="gripper", shape=(1,), unit="normalized"),
            )
        )
        self.info = EmbodimentInfo(
            name="franka-stub",
            action_space=Box(
                shape=(action_dim,),
                semantics=ActionSemantics(control_mode="joint_pos", gripper="binary"),
            ),
            observation_space=ObservationSpace(state=state),
            is_simulated=True,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Any:  # pragma: no cover
        raise NotImplementedError

    def step(self, action: Any) -> Any:  # pragma: no cover - never called
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - never called
        pass


def test_compatible_with_franka_like_embodiment() -> None:
    with XPolicyLabPolicy() as policy:
        report = check_compatibility(policy, _FrankaLikeEmbodiment())
        assert report.ok, report.errors


def test_action_dim_mismatch_fails_fast() -> None:
    with XPolicyLabPolicy(arm_dim=6) as policy:
        report = check_compatibility(policy, _FrankaLikeEmbodiment())
        assert not report.ok
        assert "action_dim" in {issue.code for issue in report.errors}
        with pytest.raises(CompatibilityError):
            report.raise_for_errors()


# --------------------------------------------------------------------------- #
# Wire protocol: structural round-trips against upstream-shaped dicts
# --------------------------------------------------------------------------- #


def test_encoded_frame_matches_upstream_wire_shape() -> None:
    array = np.arange(3, dtype=np.float32)
    frame = Frame(
        message_type=MessageType.INFER,
        request_id="req-1",
        evaluation_id="eval-1",
        trial_id="trial-1",
        step=4,
        payload={"observation": {"state": {"arm_joint_state": array}}},
    )
    wire = msgpack.unpackb(encode_frame(frame), raw=False, object_hook=msgpack_numpy.decode)
    assert wire["message_type"] == "infer"
    assert wire["message_id"] == "req-1"  # request_id travels as message_id
    assert wire["evaluation_id"] == "eval-1"
    assert wire["trial_id"] == "trial-1"
    assert wire["step"] == 4
    decoded = wire["payload"]["observation"]["state"]["arm_joint_state"]
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, array)


def test_upstream_shaped_reply_decodes() -> None:
    reply = {
        "message_type": "infer_result",
        "message_id": "req-1",
        "evaluation_id": "eval-1",
        "action_case_id": None,
        "trial_id": "trial-1",
        "repeat_index": None,
        "step": 4,
        "sent_at": "2026-07-09T00:00:00+00:00",
        "payload": {
            "actions": [{"arm_joint_state": np.zeros(7, dtype=np.float32)}],
            "latency_ms": 3.2,
        },
    }
    packed = msgpack.packb(reply, default=msgpack_numpy.encode, use_bin_type=True)
    frame = decode_frame(packed)
    assert frame.message_type is MessageType.INFER_RESULT
    assert frame.request_id == "req-1"
    assert frame.payload["latency_ms"] == 3.2
    np.testing.assert_array_equal(frame.payload["actions"][0]["arm_joint_state"], np.zeros(7))


@pytest.mark.parametrize(
    "wire",
    [
        {"message_id": "r", "evaluation_id": "e"},  # missing message_type
        {"message_type": "nope", "message_id": "r", "evaluation_id": "e"},
        {"message_type": "hello", "evaluation_id": "e"},  # missing message_id
        {"message_type": "hello", "message_id": "r"},  # missing evaluation_id
        {"message_type": "hello", "message_id": "r", "evaluation_id": "e", "payload": [1]},
    ],
)
def test_invalid_envelopes_raise(wire: dict[str, Any]) -> None:
    with pytest.raises(WsError):
        Frame.from_wire(wire)


def test_decode_rejects_garbage_and_non_maps() -> None:
    with pytest.raises(WsError, match="decode failed"):
        decode_wire(b"\xc1not-msgpack")
    with pytest.raises(WsError, match="msgpack map"):
        decode_wire(msgpack.packb([1, 2, 3]))


def test_object_dtype_arrays_rejected() -> None:
    frame = Frame(
        message_type=MessageType.INFER,
        request_id="r",
        evaluation_id="e",
        payload={"observation": {"bad": np.array([object()])}},
    )
    with pytest.raises(WsError, match="object dtype"):
        encode_frame(frame)


def test_client_requires_connect_and_pairing() -> None:
    client = PolicyClient("ws://127.0.0.1:9", "eval-1")
    assert not client.connected
    with pytest.raises(ConnectionError, match="connect"):
        client.request(MessageType.RESET, {})
    client._ws = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="pairing"):
        client.request(MessageType.HELLO_ACK, {})
    client._ws = None
