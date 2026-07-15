"""Tests for the rosbridge embodiment adapter and its wire protocol."""

from __future__ import annotations

import base64
import math
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from io import BytesIO
from typing import Any, cast

import numpy as np
import pytest
from _stub_server import StubRosbridgeServer
from PIL import Image

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    Observation,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.registry import registered, resolve
from inspect_robots_ros import RosEmbodiment, ros_embodiment
from inspect_robots_ros._client import RosbridgeClient, TopicSample
from inspect_robots_ros._msgs import (
    build_float64_multi_array,
    build_gripper_command,
    build_joint_trajectory,
    message_type,
    parse_compressed_image,
    parse_joint_state,
    parse_pose_stamped,
)
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


def test_protocol_outbound_operations_match_rosbridge_v2_shapes() -> None:
    assert subscribe(
        "/joint_states",
        subscription_id="inspect-robots-joints",
        message_type="sensor_msgs/msg/JointState",
        throttle_rate=50,
    ) == {
        "op": "subscribe",
        "id": "inspect-robots-joints",
        "topic": "/joint_states",
        "type": "sensor_msgs/msg/JointState",
        "throttle_rate": 50,
        "queue_length": 1,
        "compression": "none",
    }
    assert unsubscribe("/joint_states", subscription_id="inspect-robots-joints") == {
        "op": "unsubscribe",
        "id": "inspect-robots-joints",
        "topic": "/joint_states",
    }
    assert advertise("/arm/command", message_type="trajectory_msgs/msg/JointTrajectory") == {
        "op": "advertise",
        "topic": "/arm/command",
        "type": "trajectory_msgs/msg/JointTrajectory",
    }
    assert unadvertise("/arm/command") == {
        "op": "unadvertise",
        "topic": "/arm/command",
    }
    assert publish("/arm/command", {"joint_names": ["j1"]}) == {
        "op": "publish",
        "topic": "/arm/command",
        "msg": {"joint_names": ["j1"]},
    }
    assert call_service("/home", request_id="service-1") == {
        "op": "call_service",
        "id": "service-1",
        "service": "/home",
        "args": {},
    }


def test_protocol_json_codec_is_compact_and_round_trips() -> None:
    operation = publish("/gripper/command", {"data": [0.25]})
    encoded = encode_message(operation)
    assert encoded == '{"op":"publish","topic":"/gripper/command","msg":{"data":[0.25]}}'
    assert decode_message(encoded) == operation
    assert decode_message(encoded.encode()) == operation


def test_protocol_parses_incoming_operations() -> None:
    assert parse_incoming(
        {"op": "publish", "topic": "/joint_states", "msg": {"position": [1.0]}}
    ) == PublishedMessage("/joint_states", {"position": [1.0]})
    assert parse_incoming(
        {
            "op": "service_response",
            "id": "service-1",
            "values": {"message": "homed"},
            "result": True,
        }
    ) == ServiceResponse("service-1", {"message": "homed"}, True)
    status = parse_incoming(
        {"op": "status", "id": "publish-1", "level": "error", "msg": "bad type"}
    )
    assert status == StatusMessage("error", "bad type", "publish-1")
    assert isinstance(status, StatusMessage)
    assert str(status.as_error()) == "status_error: bad type"
    assert parse_incoming({"op": "pong"}) is None


@pytest.mark.parametrize(
    "raw,match",
    [
        ("not json", "decode"),
        ("[]", "JSON object"),
        ('{"topic":"/joint_states"}', "missing string field 'op'"),
    ],
)
def test_protocol_rejects_invalid_json_frames(raw: str, match: str) -> None:
    with pytest.raises(RosbridgeError, match=match):
        decode_message(raw)


@pytest.mark.parametrize(
    "operation,match",
    [
        ({"op": "publish", "topic": 1, "msg": {}}, "topic"),
        ({"op": "publish", "topic": "/x", "msg": []}, "msg"),
        (
            {"op": "service_response", "id": "s", "values": {}, "result": "yes"},
            "result",
        ),
        ({"op": "status", "level": "error", "msg": "bad", "id": 1}, "id"),
    ],
)
def test_protocol_rejects_malformed_known_operations(
    operation: dict[str, object], match: str
) -> None:
    with pytest.raises(RosbridgeError, match=match):
        parse_incoming(operation)


def test_protocol_rejects_non_json_values_on_encode() -> None:
    with pytest.raises(RosbridgeError, match="encode"):
        encode_message({"op": "publish", "value": object()})


def test_non_error_status_does_not_create_an_error() -> None:
    assert StatusMessage("warning", "slow publisher").as_error() is None


def _client(server: StubRosbridgeServer, **kwargs: Any) -> RosbridgeClient:
    return RosbridgeClient(
        server.url,
        connect_timeout_s=2.0,
        request_timeout_s=1.0,
        **kwargs,
    )


def _wait_for_latched_error(client: RosbridgeClient) -> Exception:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        error = client.latched_error
        if error is not None:
            return error
        time.sleep(0.005)
    raise TimeoutError("client did not latch an error")


def _wait_for_topic_sequence(
    client: RosbridgeClient, topic: str, *, after_seq: int, timeout_s: float = 2.0
) -> TopicSample:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = client.latest(topic)
        if sample is not None and sample.seq > after_seq:
            return sample
        time.sleep(0.005)
    raise TimeoutError(f"client did not receive sequence newer than {after_seq} on {topic}")


def test_client_sends_while_receive_thread_updates_topic_cache(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    client.connect()
    client.connect()
    client.advertise("/temporary", message_type="std_msgs/msg/Float64MultiArray")
    client.unadvertise("/temporary")
    client.subscribe(
        "/joint_states",
        subscription_id="joints",
        message_type="sensor_msgs/msg/JointState",
        throttle_rate=50,
    )
    stub_server.wait_for(lambda ops: any(op.get("op") == "subscribe" for op in ops))

    def stream() -> None:
        for index in range(20):
            stub_server.publish("/joint_states", {"position": [float(index)]})
            time.sleep(0.002)

    publisher = threading.Thread(target=stream, daemon=True)
    publisher.start()
    for index in range(20):
        client.publish("/arm/command", {"data": [float(index)]})
    publisher.join(timeout=2)
    assert not publisher.is_alive()

    sample = client.wait_for_sample("/joint_states", timeout_s=1.0)
    stub_server.wait_for(lambda ops: len([op for op in ops if op.get("op") == "publish"]) == 20)
    assert sample.seq >= 1
    assert sample.msg["position"][0] >= 0.0
    assert len(stub_server.ops_of("publish")) == 20
    client.close()


def test_client_topic_sequence_increments_even_when_receive_stamps_tie(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server, clock=lambda: 7.0)
    client.connect()
    client.subscribe(
        "/joint_states",
        subscription_id="joints",
        message_type="sensor_msgs/msg/JointState",
        throttle_rate=0,
    )
    stub_server.wait_for(lambda ops: any(op.get("op") == "subscribe" for op in ops))
    stub_server.publish("/joint_states", {"position": [1.0]})
    first = _wait_for_topic_sequence(client, "/joint_states", after_seq=0)
    stub_server.publish("/joint_states", {"position": [2.0]})
    second = _wait_for_topic_sequence(client, "/joint_states", after_seq=first.seq)
    assert first.stamp == second.stamp == 7.0
    assert second.seq == first.seq + 1
    client.close()


def test_client_status_error_latches_and_surfaces_on_every_later_call(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    client.connect()
    stub_server.send_status("warning", "slow publisher")
    stub_server.send_status("error", "message type mismatch")
    error = _wait_for_latched_error(client)
    assert isinstance(error, RosbridgeError)
    assert "message type mismatch" in str(error)
    with pytest.raises(RosbridgeError, match="message type mismatch"):
        client.publish("/arm/command", {"data": []})
    with pytest.raises(RosbridgeError, match="message type mismatch"):
        client.latest("/joint_states")
    with pytest.raises(RosbridgeError, match="message type mismatch"):
        client.call_service("/home")
    client.close()


def test_client_socket_death_latches_connection_error(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    client.connect()
    stub_server.drop_connections()
    error = _wait_for_latched_error(client)
    assert isinstance(error, ConnectionError)
    assert stub_server.url in str(error)
    with pytest.raises(ConnectionError, match="lost"):
        client.sequence("/joint_states")
    client.close()


def test_client_service_response_is_correlated_by_id(stub_server: StubRosbridgeServer) -> None:
    stub_server.deferred_services.add("/home")
    client = _client(stub_server)
    client.connect()
    replies: list[ServiceResponse] = []
    failures: list[BaseException] = []

    def call() -> None:
        try:
            replies.append(client.call_service("/home"))
        except BaseException as exc:  # pragma: no cover - assertion reports unexpected failure
            failures.append(exc)

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    stub_server.wait_for(lambda ops: any(op.get("op") == "call_service" for op in ops))
    request_id = stub_server.ops_of("call_service")[0]["id"]
    stub_server.send_service_response("wrong-id", values={"wrong": True})
    time.sleep(0.02)
    assert caller.is_alive()
    stub_server.send_service_response(request_id, values={"homed": True})
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert not failures
    assert replies == [ServiceResponse(request_id, {"homed": True}, True)]
    client.close()


def test_client_failed_service_result_raises_protocol_error(
    stub_server: StubRosbridgeServer,
) -> None:
    stub_server.service_results["/home"] = (False, {"reason": "blocked"})
    client = _client(stub_server)
    client.connect()
    with pytest.raises(RosbridgeError, match=r"/home.*result=false.*blocked"):
        client.call_service("/home")
    client.close()


def test_client_service_timeout_names_service_and_url(stub_server: StubRosbridgeServer) -> None:
    stub_server.deferred_services.add("/home")
    client = RosbridgeClient(
        stub_server.url,
        connect_timeout_s=2.0,
        request_timeout_s=0.03,
    )
    client.connect()
    with pytest.raises(RosbridgeError, match=rf"/home.*{stub_server.url}"):
        client.call_service("/home")
    client.close()


def test_client_close_cleans_resources_joins_thread_and_is_idempotent(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    client.connect()
    client.advertise("/arm/command", message_type="std_msgs/msg/Float64MultiArray")
    client.subscribe(
        "/joint_states",
        subscription_id="joints",
        message_type="sensor_msgs/msg/JointState",
        throttle_rate=50,
    )
    stub_server.wait_for(lambda ops: len(ops) >= 2)
    client.close()
    client.close()
    stub_server.wait_for(
        lambda ops: (
            any(op.get("op") == "unsubscribe" for op in ops)
            and any(op.get("op") == "unadvertise" for op in ops)
        )
    )
    assert not client.connected
    assert not client.receiver_alive
    assert set(stub_server.subscriptions) == set()
    with pytest.raises(RuntimeError, match="closed"):
        client.publish("/arm/command", {"data": []})


def test_client_requires_connect_and_connect_failure_names_url(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    with pytest.raises(ConnectionError, match="call connect"):
        client.latest("/joint_states")
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.connect()

    unreachable = RosbridgeClient("ws://127.0.0.1:9", connect_timeout_s=0.05)
    with pytest.raises(ConnectionError, match=r"ws://127\.0\.0\.1:9"):
        unreachable.connect()
    unreachable.close()


def test_client_send_failure_latches_first_connection_error() -> None:
    class FailingSocket:
        def send(self, _message: str) -> None:
            raise OSError("broken pipe")

        def close(self) -> None:
            pass

    client = RosbridgeClient("ws://robot.example:9090")
    client._ws = cast(Any, FailingSocket())
    with pytest.raises(ConnectionError, match=r"robot\.example.*broken pipe") as exc_info:
        client.publish("/arm/command", {"data": []})
    assert client.latched_error is exc_info.value
    with pytest.raises(ConnectionError) as repeated:
        client.latest("/joint_states")
    assert repeated.value is exc_info.value
    client._latch(RuntimeError("later failure"))
    assert client.latched_error is exc_info.value
    client.close()


def test_client_rejects_unsupported_incoming_frame_type() -> None:
    class InvalidFrameSocket:
        def recv(self) -> object:
            return object()

        def close(self) -> None:
            pass

    client = RosbridgeClient("ws://robot.example:9090")
    client._ws = cast(Any, InvalidFrameSocket())
    client._receive_loop()
    error = client.latched_error
    assert isinstance(error, RosbridgeError)
    assert "unsupported frame type object" in str(error)
    client.close()


def test_client_service_wait_detects_concurrent_close() -> None:
    class NoopSocket:
        def send(self, _message: str) -> None:
            pass

        def close(self) -> None:
            pass

    client = RosbridgeClient("ws://robot.example:9090")

    def close_during_wait(_duration: float) -> None:
        client._closed = True

    client._sleep = close_during_wait
    client._ws = cast(Any, NoopSocket())
    with pytest.raises(ConnectionError, match=r"robot\.example.*closed"):
        client.call_service("/home")
    client._closed = False
    client.close()


def test_joint_state_reorders_shuffled_names() -> None:
    parsed = parse_joint_state(
        {
            "name": ["finger", "joint_2", "unused", "joint_1"],
            "position": [0.04, 2.0, 99.0, 1.0],
        },
        ("joint_1", "joint_2", "finger"),
    )
    assert parsed.dtype == np.float64
    np.testing.assert_array_equal(parsed, [1.0, 2.0, 0.04])


def test_joint_state_missing_joint_error_lists_available_names() -> None:
    with pytest.raises(ValueError, match=r"missing.*joint_2.*available.*joint_1.*spare"):
        parse_joint_state(
            {"name": ["spare", "joint_1"], "position": [9.0, 1.0]},
            ("joint_1", "joint_2"),
        )


@pytest.mark.parametrize(
    "msg,match",
    [
        ({"name": "joint_1", "position": [1.0]}, "name"),
        ({"name": ["joint_1"], "position": "1.0"}, "position"),
        ({"name": ["joint_1"], "position": [1.0, 2.0]}, "equal length"),
        ({"name": ["joint_1"], "position": [object()]}, "numeric"),
    ],
)
def test_joint_state_rejects_malformed_parallel_arrays(msg: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_joint_state(msg, ("joint_1",))


def _compressed_image_message(
    image: np.ndarray[Any, np.dtype[np.uint8]], image_format: str, ros_format: str
) -> dict[str, str]:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format=image_format)
    return {
        "format": ros_format,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


@pytest.mark.parametrize(
    "image_format,ros_format",
    [
        ("PNG", "png"),
        ("JPEG", "jpeg"),
        ("JPEG", "rgb8; jpeg compressed bgr8"),
    ],
)
def test_compressed_image_decodes_rgb_uint8(image_format: str, ros_format: str) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 240
    image[..., 1] = 10
    parsed = parse_compressed_image(_compressed_image_message(image, image_format, ros_format))
    assert parsed.shape == (2, 3, 3)
    assert parsed.dtype == np.uint8
    if image_format == "PNG":
        np.testing.assert_array_equal(parsed, image)
    else:
        np.testing.assert_allclose(parsed, image, atol=4)


@pytest.mark.parametrize(
    "msg",
    [
        {"format": "jpeg", "data": [1, 2, 3]},
        {"format": "jpeg", "data": "not base64!"},
        {"format": "jpeg", "data": base64.b64encode(b"not an image").decode()},
    ],
)
def test_compressed_image_rejects_bad_data(msg: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"CompressedImage|decode"):
        parse_compressed_image(msg)


def test_pose_stamped_reorders_xyzw_to_wxyz() -> None:
    parsed = parse_pose_stamped(
        {
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
            }
        }
    )
    np.testing.assert_array_equal(parsed, [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])


def test_pose_stamped_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="orientation"):
        parse_pose_stamped({"pose": {"position": {"x": 1.0}}})


@pytest.mark.parametrize(
    "ros_version,expected_type,expected_duration",
    [
        (
            1,
            "trajectory_msgs/JointTrajectory",
            {"secs": 0, "nsecs": 100_000_000},
        ),
        (
            2,
            "trajectory_msgs/msg/JointTrajectory",
            {"sec": 0, "nanosec": 100_000_000},
        ),
    ],
)
def test_joint_trajectory_golden_shapes_for_both_ros_versions(
    ros_version: int, expected_type: str, expected_duration: dict[str, int]
) -> None:
    assert message_type("joint_trajectory", ros_version) == expected_type
    assert build_joint_trajectory(
        ("joint_2", "joint_1"), [2.0, 1.0], period_s=0.1, ros_version=ros_version
    ) == {
        "joint_names": ["joint_2", "joint_1"],
        "points": [
            {
                "positions": [2.0, 1.0],
                "time_from_start": expected_duration,
            }
        ],
    }


def test_joint_trajectory_duration_rounding_carries_to_next_second() -> None:
    message = build_joint_trajectory(("joint_1",), [1.0], period_s=1.9999999996, ros_version=2)
    assert message["points"][0]["time_from_start"] == {"sec": 2, "nanosec": 0}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"joint_names": ("j1",), "positions": [1.0, 2.0], "period_s": 0.1},
            "names",
        ),
        (
            {"joint_names": ("j1",), "positions": [1.0], "period_s": -0.1},
            "period_s",
        ),
    ],
)
def test_joint_trajectory_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_joint_trajectory(**kwargs, ros_version=2)


def test_joint_trajectory_rejects_unknown_ros_version() -> None:
    with pytest.raises(ValueError, match="ros_version must be 1 or 2"):
        build_joint_trajectory(("joint_1",), [1.0], period_s=0.1, ros_version=3)


def test_float64_and_gripper_command_builders_preserve_raw_values() -> None:
    assert build_float64_multi_array(np.array([1.0, 2.0])) == {"data": [1.0, 2.0]}
    assert build_gripper_command(0.037, "float64") == {"data": 0.037}
    assert build_gripper_command(0.037, "float64_multi_array") == {"data": [0.037]}
    with pytest.raises(ValueError, match="command_type"):
        build_gripper_command(0.0, "action")


def test_supported_message_type_strings_are_versioned() -> None:
    assert message_type("joint_state", 1) == "sensor_msgs/JointState"
    assert message_type("joint_state", 2) == "sensor_msgs/msg/JointState"
    assert message_type("compressed_image", 2) == "sensor_msgs/msg/CompressedImage"
    assert message_type("pose_stamped", 2) == "geometry_msgs/msg/PoseStamped"
    assert message_type("float64_multi_array", 2) == "std_msgs/msg/Float64MultiArray"
    assert message_type("float64", 1) == "std_msgs/Float64"
    with pytest.raises(ValueError, match="ros_version"):
        message_type("joint_state", 3)


_SCENE = Scene(id="scene-1", instruction="place the red block in the tray")


def _embodiment(**kwargs: Any) -> RosEmbodiment:
    kwargs.setdefault("joints", ("joint_1", "joint_2"))
    kwargs.setdefault("command_topic", "/arm/command")
    kwargs.setdefault("action_low", (-2.0, -3.0))
    kwargs.setdefault("action_high", (2.0, 3.0))
    kwargs.setdefault("obs_timeout_s", 1.0)
    return RosEmbodiment(**kwargs)


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.sleep_calls: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleep_calls.append(duration)
        self.now += duration

    def advance(self, duration: float) -> None:
        self.now += duration


class _FakeClient:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.samples: dict[str, TopicSample] = {}
        self.operations: list[tuple[str, str, Any, float]] = []
        self.wait_timeouts: list[float] = []
        self.on_publish: Callable[[str, Mapping[str, Any]], None] | None = None
        self.on_wait: Callable[[str, int], None] | None = None
        self.service_response = ServiceResponse("fake-service", {}, True)
        self.closed = False

    def put(self, topic: str, msg: Mapping[str, Any], *, stamp: float | None = None) -> None:
        previous = self.samples.get(topic)
        self.samples[topic] = TopicSample(
            msg=dict(msg),
            stamp=self.clock() if stamp is None else stamp,
            seq=1 if previous is None else previous.seq + 1,
        )

    def connect(self) -> None:
        self.operations.append(("connect", "", None, self.clock()))

    def advertise(self, topic: str, *, message_type: str) -> None:
        self.operations.append(("advertise", topic, message_type, self.clock()))

    def subscribe(self, topic: str, **kwargs: Any) -> None:
        self.operations.append(("subscribe", topic, kwargs, self.clock()))

    def unsubscribe(self, topic: str, **kwargs: Any) -> None:
        self.operations.append(("unsubscribe", topic, kwargs, self.clock()))

    def publish(self, topic: str, msg: Mapping[str, Any]) -> None:
        self.operations.append(("publish", topic, dict(msg), self.clock()))
        if self.on_publish is not None:
            self.on_publish(topic, msg)

    def call_service(self, service: str) -> ServiceResponse:
        self.operations.append(("call_service", service, None, self.clock()))
        return self.service_response

    def latest(self, topic: str) -> TopicSample | None:
        return self.samples.get(topic)

    def sequence(self, topic: str) -> int:
        sample = self.samples.get(topic)
        return 0 if sample is None else sample.seq

    def wait_for_sample(self, topic: str, *, after_seq: int = 0, timeout_s: float) -> TopicSample:
        self.wait_timeouts.append(timeout_s)
        if self.on_wait is not None:
            self.on_wait(topic, after_seq)
        sample = self.samples.get(topic)
        if sample is not None and sample.seq > after_seq:
            return sample
        self.clock.sleep(timeout_s)
        raise TimeoutError(topic)

    def close(self) -> None:
        self.closed = True


def _joint_message(positions: list[float] | None = None) -> dict[str, Any]:
    return {
        "name": ["joint_2", "finger", "joint_1"],
        "position": positions or [0.2, 0.03, 0.1],
    }


def _pose_message() -> dict[str, Any]:
    return {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
        }
    }


def _ready_embodiment(
    *,
    clock: _FakeClock | None = None,
    joint_message: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[RosEmbodiment, _FakeClient, _FakeClock]:
    fake_clock = clock or _FakeClock()
    kwargs.setdefault("clock", fake_clock)
    kwargs.setdefault("sleep", fake_clock.sleep)
    embodiment = _embodiment(**kwargs)
    client = _FakeClient(fake_clock)
    client.put(embodiment.joint_states_topic, joint_message or _joint_message())
    if embodiment.eef_pose_topic is not None:
        client.put(embodiment.eef_pose_topic, _pose_message())
    for camera in embodiment.cameras.values():
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        client.put(camera.topic, _compressed_image_message(image, "PNG", "png"))

    def fresh_joint_state(topic: str, _msg: Mapping[str, Any]) -> None:
        if topic == embodiment.command_topic:
            current = client.samples[embodiment.joint_states_topic]
            client.put(embodiment.joint_states_topic, current.msg)

    client.on_publish = fresh_joint_state
    embodiment._client = cast(Any, client)
    embodiment._initialized = True
    embodiment._instruction = _SCENE.instruction
    return embodiment, client, fake_clock


class _TopicStreams:
    def __init__(
        self,
        server: StubRosbridgeServer,
        messages: Mapping[str, Mapping[str, Any] | Callable[[], Mapping[str, Any]]],
        *,
        interval_s: float,
    ) -> None:
        self.server = server
        self.messages = messages
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise TimeoutError("topic publisher thread did not stop")

    def _run(self) -> None:
        while not self._stop.is_set():
            for topic, source in self.messages.items():
                message = source() if callable(source) else source
                self.server.publish(topic, message)
            time.sleep(self.interval_s)


@contextmanager
def _topic_streams(
    server: StubRosbridgeServer,
    messages: Mapping[str, Mapping[str, Any] | Callable[[], Mapping[str, Any]]],
    *,
    interval_s: float = 0.005,
) -> Iterator[None]:
    streams = _TopicStreams(server, messages, interval_s=interval_s)
    streams.start()
    try:
        yield
    finally:
        streams.stop()


def test_registry_entry_point_and_cli_string_forms_include_non_square_camera() -> None:
    assert "ros" in registered("embodiment")
    embodiment = resolve(
        "embodiment",
        "ros",
        joints="joint_1,joint_2",
        command_topic="/arm/command",
        action_low="-2,-3",
        action_high="2,3",
        cameras="wrist:/camera/wrist/compressed:640x480",
    )
    assert isinstance(embodiment, RosEmbodiment)
    assert embodiment.joints == ("joint_1", "joint_2")
    camera = embodiment.info.observation_space.cameras[0]
    assert (camera.name, camera.height, camera.width) == ("wrist", 480, 640)
    embodiment.close()


def test_scalar_coerced_one_dof_bounds_are_accepted() -> None:
    embodiment = ros_embodiment(
        joints="joint_1",
        command_topic="/arm/command",
        action_low=-3.1,
        action_high=3.1,
    )
    assert embodiment.info.action_space.shape == (1,)
    np.testing.assert_array_equal(embodiment.info.action_space.low, [-3.1])
    embodiment.close()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"joints": None}, "joints is required"),
        ({"joints": ""}, "at least one joint name"),
        ({"command_topic": None}, "command_topic is required"),
        ({"action_low": None}, "action_low is required"),
        ({"action_high": None}, "action_high is required"),
        ({"action_low": "bad,1"}, "comma-separated list of numbers"),
        ({"action_low": ""}, "at least one numeric bound"),
        ({"action_low": (-1.0,)}, "1 bounds for 2 joints"),
        ({"action_high": (1.0,)}, "1 bounds for 2 joints"),
        ({"action_low": (-math.inf, -3.0)}, "bounds must all be finite"),
        ({"action_low": (3.0, -3.0)}, "elementwise <="),
        ({"joints": ("joint_1", "joint_1")}, "duplicate"),
        (
            {
                "joints": ("joint_1", "gripper"),
                "gripper_topic": "/gripper/command",
                "gripper_joint": "finger",
                "gripper_low": 0.0,
                "gripper_high": 1.0,
            },
            "named 'gripper'",
        ),
        (
            {
                "gripper_topic": "/gripper/command",
                "gripper_joint": "finger",
                "gripper_low": 1.0,
                "gripper_high": 1.0,
            },
            "less than",
        ),
        (
            {
                "gripper_topic": "/gripper/command",
                "gripper_joint": "finger",
                "gripper_low": 0.0,
                "gripper_high": math.inf,
            },
            "gripper_low and gripper_high must be finite",
        ),
        ({"gripper_topic": "/gripper/command"}, "required when gripper_topic"),
        ({"gripper_joint": "finger"}, "only when gripper_topic"),
        ({"control_hz": 0.0}, "positive and finite"),
        ({"control_hz": math.inf}, "positive and finite"),
        ({"ros_version": 3}, "ros_version"),
        ({"command_type": "effort"}, "command_type"),
        ({"gripper_command_type": "action"}, "gripper_command_type"),
        ({"gripper_closed_at": "left"}, "gripper_closed_at"),
        ({"fresh_obs_timeout_s": 0.0}, "fresh_obs_timeout_s must be positive and finite"),
        ({"camera_throttle_ms": -1}, "camera_throttle_ms must be >= 0"),
        ({"obs_timeout_s": 0.0}, "obs_timeout_s must be positive and finite"),
        ({"connect_timeout_s": math.inf}, "connect_timeout_s must be positive and finite"),
        ({"request_timeout_s": 0.0}, "request_timeout_s must be positive and finite"),
        ({"staleness_s": -1.0}, "staleness_s must be finite and >= 0"),
        ({"cameras": ","}, "parsed to no cameras"),
        ({"cameras": "wrist=/cam=640x480"}, "name:topic:WxH"),
        ({"cameras": "wrist:/cam:640by480"}, "must be WxH"),
        ({"cameras": "wrist:/cam:widex480"}, "integer width and height"),
        ({"cameras": {"wrist": ("/cam", 480)}}, "must map to"),
        ({"cameras": {"wrist-cam": ("/cam", 480, 640)}}, "valid identifier"),
        ({"cameras": {"wrist": ("", 480, 640)}}, "non-empty string"),
        ({"cameras": {"wrist": ("/cam", 0, 640)}}, "positive integers"),
        (
            {"cameras": "wrist:/a:640x480,wrist:/b:320x240"},
            "duplicate name",
        ),
    ],
)
def test_factory_validation_errors_are_actionable(kwargs: dict[str, Any], match: str) -> None:
    defaults: dict[str, Any] = {
        "joints": ("joint_1", "joint_2"),
        "command_topic": "/arm/command",
        "action_low": (-2.0, -3.0),
        "action_high": (2.0, 3.0),
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RosEmbodiment(**defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "joints": tuple(f"joint_{index}" for index in range(7)),
            "action_low": (-1.0,) * 7,
            "action_high": (1.0,) * 7,
            "eef_pose_topic": "/eef_pose",
        },
        {
            "joints": tuple(f"joint_{index}" for index in range(6)),
            "action_low": (-1.0,) * 6,
            "action_high": (1.0,) * 6,
            "gripper_topic": "/gripper/command",
            "gripper_joint": "finger",
            "gripper_low": 0.0,
            "gripper_high": 0.08,
            "eef_pose_topic": "/eef_pose",
        },
    ],
)
def test_seven_dimensional_action_with_eef_pose_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(
        ValueError,
        match="omit eef_pose_topic on a 7-dim action space until core supports "
        "key-priority reference matching",
    ):
        RosEmbodiment(command_topic="/arm/command", **kwargs)


def test_info_declares_spaces_semantics_capabilities_and_lazy_network() -> None:
    embodiment = _embodiment(
        url="ws://192.0.2.1:9090",
        gripper_topic="/gripper/command",
        gripper_joint="finger",
        gripper_low=0.0,
        gripper_high=0.08,
        eef_pose_topic="/eef_pose",
        cameras={"wrist": ("/camera/compressed", 480, 640)},
        reset_service="/home",
        simulated=True,
        control_hz=20.0,
        name="ros:test-arm",
    )
    info = embodiment.info
    assert info.name == "ros:test-arm"
    assert info.action_space.shape == (3,)
    np.testing.assert_array_equal(info.action_space.low, [-2.0, -3.0, 0.0])
    np.testing.assert_array_equal(info.action_space.high, [2.0, 3.0, 0.08])
    assert info.action_space.semantics == ActionSemantics(
        control_mode="joint_pos",
        rotation_repr="none",
        gripper="continuous",
        frame="base",
        dim_labels=("joint_1", "joint_2", "gripper"),
    )
    state_spec = info.observation_space.state
    assert state_spec is not None
    fields = {field.key: field for field in state_spec.fields}
    assert fields["joint_pos"].shape == (3,)
    assert fields["gripper"].shape == (1,)
    assert fields["eef_pose"].shape == (7,)
    assert info.capabilities == frozenset({"self_paced", "resettable"})
    assert info.control_hz == 20.0
    assert info.is_simulated
    assert info.supported_setups == info.supported_target_kinds == frozenset()
    assert not embodiment._client.connected
    embodiment.close()


@pytest.mark.parametrize("with_gripper", [False, True])
@pytest.mark.parametrize("with_eef_pose", [False, True])
def test_conformance_matrix_passes_away_from_seven_dimensions(
    with_gripper: bool, with_eef_pose: bool
) -> None:
    kwargs: dict[str, Any] = {}
    if with_gripper:
        kwargs.update(
            gripper_topic="/gripper/command",
            gripper_joint="finger",
            gripper_low=0.0,
            gripper_high=0.08,
        )
    if with_eef_pose:
        kwargs["eef_pose_topic"] = "/eef_pose"
    embodiment = _embodiment(**kwargs)
    assert_embodiment_conformant(embodiment.info)
    embodiment.close()


def test_gripperless_info_never_claims_real_hardware_oracle_capabilities() -> None:
    embodiment = _embodiment()
    assert embodiment.info.capabilities == frozenset({"self_paced"})
    semantics = embodiment.info.action_space.semantics
    assert semantics is not None
    assert semantics.dim_labels == ("joint_1", "joint_2")
    assert embodiment.info.observation_space.state_keys == frozenset({"joint_pos"})
    for capability in ("seedable", "auto_reset", "privileged_success"):
        assert capability not in embodiment.info.capabilities
    embodiment.close()


@pytest.mark.parametrize(
    "ros_version,command_type,expected",
    [
        (
            1,
            "joint_trajectory",
            {
                "joint_names": ["joint_1", "joint_2"],
                "points": [
                    {
                        "positions": [1.0, 2.0],
                        "time_from_start": {"secs": 0, "nsecs": 100_000_000},
                    }
                ],
            },
        ),
        (
            2,
            "joint_trajectory",
            {
                "joint_names": ["joint_1", "joint_2"],
                "points": [
                    {
                        "positions": [1.0, 2.0],
                        "time_from_start": {"sec": 0, "nanosec": 100_000_000},
                    }
                ],
            },
        ),
        (2, "float64_multi_array", {"data": [1.0, 2.0]}),
    ],
)
def test_step_publishes_versioned_arm_command_and_returns_nonterminal_result(
    ros_version: int, command_type: str, expected: dict[str, Any]
) -> None:
    embodiment, client, _clock = _ready_embodiment(
        ros_version=ros_version, command_type=command_type
    )
    result = embodiment.step(Action(data=np.asarray([1.0, 2.0])))
    arm_publish = next(op for op in client.operations if op[:2] == ("publish", "/arm/command"))
    assert arm_publish[2] == expected
    assert result.reward is None
    assert not result.terminated
    assert not result.truncated
    assert result.observation.instruction == _SCENE.instruction


@pytest.mark.parametrize(
    "ros_version,override,expected_message",
    [
        (1, None, {"data": 0.037}),
        (2, None, {"data": [0.037]}),
        (2, "float64", {"data": 0.037}),
        (1, "float64_multi_array", {"data": [0.037]}),
    ],
)
def test_step_splits_raw_gripper_command_with_version_default_and_override(
    ros_version: int, override: str | None, expected_message: dict[str, Any]
) -> None:
    embodiment, client, _clock = _ready_embodiment(
        ros_version=ros_version,
        gripper_topic="/gripper/command",
        gripper_command_type=override,
        gripper_joint="finger",
        gripper_low=0.0,
        gripper_high=0.08,
    )
    embodiment.step(Action(data=np.asarray([1.0, 2.0, 0.037])))
    publishes = [operation for operation in client.operations if operation[0] == "publish"]
    assert [operation[1] for operation in publishes] == [
        "/arm/command",
        "/gripper/command",
    ]
    assert publishes[1][2] == expected_message


def test_every_fast_back_to_back_arm_publish_interval_is_at_least_period() -> None:
    embodiment, client, _clock = _ready_embodiment(control_hz=10.0)
    for _ in range(5):
        embodiment.step(Action(data=np.asarray([0.1, 0.2])))
    publish_times = [
        operation[3]
        for operation in client.operations
        if operation[:2] == ("publish", "/arm/command")
    ]
    intervals = np.diff(publish_times)
    assert len(intervals) == 4
    assert bool(np.all(intervals >= 0.1 - 1e-12)), intervals


def test_pacing_adds_no_sleep_after_slow_inference_gap() -> None:
    embodiment, _client, clock = _ready_embodiment(control_hz=10.0)
    embodiment.step(Action(data=np.asarray([0.1, 0.2])))
    clock.advance(0.25)
    sleeps_before = len(clock.sleep_calls)
    embodiment.step(Action(data=np.asarray([0.2, 0.3])))
    assert len(clock.sleep_calls) == sleeps_before


def test_step_freshness_uses_sequence_when_receive_stamps_are_equal() -> None:
    clock = _FakeClock(now=5.0)
    embodiment, client, _clock = _ready_embodiment(clock=clock)
    before = client.samples[embodiment.joint_states_topic]
    result = embodiment.step(Action(data=np.asarray([0.1, 0.2])))
    after = client.samples[embodiment.joint_states_topic]
    assert before.stamp == after.stamp == 5.0
    assert after.seq == before.seq + 1
    assert result.observation.state_time == 5.0


def test_step_freshness_timeout_names_both_configuration_knobs() -> None:
    embodiment, client, _clock = _ready_embodiment(control_hz=20.0, fresh_obs_timeout_s=0.15)
    client.on_publish = None
    with pytest.raises(TimeoutError, match=r"fresh_obs_timeout_s=0.15s.*control_hz=20"):
        embodiment.step(Action(data=np.asarray([0.1, 0.2])))


def test_step_rejects_stale_cross_modal_sample() -> None:
    clock = _FakeClock(now=10.0)
    embodiment, client, _clock = _ready_embodiment(
        clock=clock,
        cameras={"wrist": ("/camera/compressed", 2, 3)},
        staleness_s=1.0,
    )
    camera_sample = client.samples["/camera/compressed"]
    client.samples["/camera/compressed"] = TopicSample(
        camera_sample.msg, stamp=8.0, seq=camera_sample.seq
    )
    with pytest.raises(TimeoutError, match=r"camera/compressed.*staleness_s=1"):
        embodiment.step(Action(data=np.asarray([0.1, 0.2])))


@pytest.mark.parametrize(
    "closed_at,raw,expected",
    [("low", 0.02, 0.25), ("high", 0.02, 0.75)],
)
def test_observation_gripper_normalization_and_polarity(
    closed_at: str, raw: float, expected: float
) -> None:
    embodiment, _client, _clock = _ready_embodiment(
        joint_message=_joint_message([0.2, raw, 0.1]),
        gripper_topic="/gripper/command",
        gripper_joint="finger",
        gripper_low=0.0,
        gripper_high=0.08,
        gripper_closed_at=closed_at,
    )
    observation = embodiment.step(Action(data=np.asarray([0.1, 0.2, raw]))).observation
    np.testing.assert_allclose(observation.state["joint_pos"], [0.1, 0.2, raw])
    np.testing.assert_allclose(observation.state["gripper"], [expected])


def test_observation_state_and_image_times_follow_oldest_state_convention() -> None:
    clock = _FakeClock(now=10.0)
    embodiment, client, _clock = _ready_embodiment(
        clock=clock,
        eef_pose_topic="/eef_pose",
        cameras={"wrist": ("/camera/compressed", 2, 3)},
        staleness_s=3.0,
    )
    eef = client.samples["/eef_pose"]
    camera = client.samples["/camera/compressed"]
    client.samples["/eef_pose"] = TopicSample(eef.msg, 8.5, eef.seq)
    client.samples["/camera/compressed"] = TopicSample(camera.msg, 9.0, camera.seq)
    observation = embodiment.step(Action(data=np.asarray([0.1, 0.2]))).observation
    assert observation.state_time == 8.5
    assert observation.image_times == {"wrist": 9.0}
    np.testing.assert_array_equal(
        observation.state["eef_pose"], [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]
    )


def test_step_rejects_wrong_action_shape() -> None:
    embodiment, _client, _clock = _ready_embodiment()
    with pytest.raises(ValueError, match=r"shape.*expected"):
        embodiment.step(Action(data=np.zeros((1, 2))))


def test_second_reset_without_physical_reset_path_warns_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    embodiment, client, _clock = _ready_embodiment()

    def fresh(topic: str, _after: int) -> None:
        client.put(topic, client.samples[topic].msg)

    client.on_wait = fresh
    embodiment.reset(_SCENE)
    embodiment.reset(_SCENE)
    embodiment.reset(_SCENE)
    stderr = capsys.readouterr().err
    assert stderr.count("no reset_service or operator_reset_confirm") == 1


def test_operator_confirm_post_wait_uses_obs_timeout_not_fresh_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    embodiment, client, _clock = _ready_embodiment(
        operator_reset_confirm=True,
        obs_timeout_s=3.0,
        fresh_obs_timeout_s=0.01,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    def fresh(topic: str, _after: int) -> None:
        client.put(topic, client.samples[topic].msg)

    client.on_wait = fresh
    observation = embodiment.reset(_SCENE)
    assert client.wait_timeouts[-1] == 3.0
    assert observation.instruction == _SCENE.instruction
    assert _SCENE.instruction in capsys.readouterr().out


def test_reset_accepts_and_ignores_seed_without_claiming_seedable() -> None:
    embodiment, client, _clock = _ready_embodiment()

    def fresh(topic: str, _after: int) -> None:
        client.put(topic, client.samples[topic].msg)

    client.on_wait = fresh
    observation = embodiment.reset(_SCENE, seed=123)
    assert observation.instruction == _SCENE.instruction
    assert "seedable" not in embodiment.info.capabilities


def test_operator_confirm_eof_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    embodiment, _client, _clock = _ready_embodiment(operator_reset_confirm=True)

    def eof(_prompt: str) -> str:
        raise EOFError("stdin is not interactive")

    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(EOFError, match="not interactive"):
        embodiment.reset(_SCENE)


def test_close_delegates_idempotently_to_client() -> None:
    embodiment, client, _clock = _ready_embodiment()
    embodiment.close()
    embodiment.close()
    assert client.closed


def test_context_manager_returns_embodiment_and_closes_client() -> None:
    embodiment, client, _clock = _ready_embodiment()
    with embodiment as entered:
        assert entered is embodiment
    assert client.closed


def test_initialization_advertises_configured_gripper() -> None:
    clock = _FakeClock()
    embodiment = _embodiment(
        clock=clock,
        sleep=clock.sleep,
        gripper_topic="/gripper/command",
        gripper_joint="finger",
        gripper_low=0.0,
        gripper_high=0.08,
    )
    client = _FakeClient(clock)

    def provide_joint_state(topic: str, _after: int) -> None:
        client.put(topic, _joint_message())

    client.on_wait = provide_joint_state
    embodiment._client = cast(Any, client)
    embodiment._ensure_initialized()
    advertisements = [operation for operation in client.operations if operation[0] == "advertise"]
    assert [(operation[1], operation[2]) for operation in advertisements] == [
        ("/arm/command", "trajectory_msgs/msg/JointTrajectory"),
        ("/gripper/command", "std_msgs/msg/Float64MultiArray"),
    ]


def test_preflight_stops_when_one_second_deadline_is_reached() -> None:
    clock = _FakeClock()
    embodiment = _embodiment(clock=clock, sleep=clock.sleep)
    client = _FakeClient(clock)

    def delayed_joint_state(topic: str, _after: int) -> None:
        clock.advance(1.0)
        client.put(topic, _joint_message())

    client.on_wait = delayed_joint_state
    embodiment._client = cast(Any, client)
    with pytest.warns(RuntimeWarning, match="exactly one message"):
        embodiment._joint_state_preflight()
    assert len(client.wait_timeouts) == 1


def test_shared_observation_deadline_names_later_missing_topic() -> None:
    clock = _FakeClock()
    embodiment = _embodiment(clock=clock, sleep=clock.sleep)
    client = _FakeClient(clock)

    def consume_deadline(topic: str, _after: int) -> None:
        clock.advance(1.0)
        client.put(topic, {})

    client.on_wait = consume_deadline
    embodiment._client = cast(Any, client)
    with pytest.raises(TimeoutError, match=r"missing topic '/camera'.*obs_timeout_s=1s"):
        embodiment._wait_for_sequences({"/joint_states": 0, "/camera": 0}, 1.0, "obs_timeout_s")


def test_camera_resolution_validation_is_only_applied_to_first_frame() -> None:
    embodiment, client, _clock = _ready_embodiment(cameras={"wrist": ("/camera/compressed", 2, 3)})
    embodiment._validate_camera_resolutions()
    client.put(
        "/camera/compressed",
        _compressed_image_message(np.zeros((4, 5, 3), dtype=np.uint8), "PNG", "png"),
    )
    embodiment._validate_camera_resolutions()


def test_observation_assembly_rejects_missing_cached_sample() -> None:
    embodiment, client, _clock = _ready_embodiment()
    client.samples.pop(embodiment.joint_states_topic)
    with pytest.raises(RuntimeError, match=r"no cached message.*joint_states"):
        embodiment._assemble_observation()


def test_connection_failure_names_url_and_both_rosbridge_launch_commands() -> None:
    embodiment = _embodiment(url="ws://127.0.0.1:9", connect_timeout_s=0.05)
    with pytest.raises(ConnectionError) as exc_info:
        embodiment.reset(_SCENE)
    message = str(exc_info.value)
    assert "ws://127.0.0.1:9" in message
    assert "roslaunch rosbridge_server rosbridge_websocket.launch" in message
    assert "ros2 launch rosbridge_server rosbridge_websocket_launch.xml" in message
    embodiment.close()


class _PolicyInfoStub:
    config = PolicyConfig()

    def __init__(self, info: PolicyInfo) -> None:
        self.info = info

    def reset(self, scene: Scene) -> None:
        del scene

    def act(self, observation: Observation) -> ActionChunk:
        del observation
        return ActionChunk(actions=(Action(data=np.zeros(self.info.action_space.shape)),))

    def close(self) -> None:
        pass


def test_compatibility_with_agent_and_xpolicylab_shaped_policy_info_stubs() -> None:
    embodiment = _embodiment(
        gripper_topic="/gripper/command",
        gripper_joint="finger",
        gripper_low=0.0,
        gripper_high=0.08,
        cameras={"wrist": ("/camera/compressed", 480, 640)},
    )
    agent = _PolicyInfoStub(
        PolicyInfo(
            name="agent-shaped",
            action_space=embodiment.info.action_space,
            observation_space=ObservationSpace(
                cameras=embodiment.info.observation_space.cameras,
                state_keys=frozenset({"joint_pos"}),
            ),
            control_hz=10.0,
        )
    )
    xpolicylab = _PolicyInfoStub(
        PolicyInfo(
            name="xpolicylab-shaped",
            action_space=Box(
                shape=(3,),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                ),
            ),
            observation_space=ObservationSpace(
                cameras=embodiment.info.observation_space.cameras,
                state_keys=frozenset({"joint_pos", "gripper"}),
            ),
            control_hz=10.0,
        )
    )
    assert check_compatibility(agent, embodiment).ok
    assert check_compatibility(xpolicylab, embodiment).ok

    camera_free = _embodiment(
        gripper_topic="/gripper/command",
        gripper_joint="finger",
        gripper_low=0.0,
        gripper_high=0.08,
    )
    report = check_compatibility(xpolicylab, camera_free)
    assert {issue.code for issue in report.errors} == {"missing_camera"}
    embodiment.close()
    camera_free.close()


def test_rosbridge_reset_subscriptions_preflight_handoff_and_fast_rate_no_warning(
    stub_server: StubRosbridgeServer,
) -> None:
    embodiment = _embodiment(url=stub_server.url, control_hz=10.0)
    with (
        _topic_streams(stub_server, {"/joint_states": _joint_message()}),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        observation = embodiment.reset(_SCENE)
    assert observation.instruction == _SCENE.instruction
    rate_warnings = [
        warning
        for warning in caught
        if "native rate" in str(warning.message) or "too few" in str(warning.message)
    ]
    assert not rate_warnings
    joint_ops = [
        operation
        for operation in stub_server.ops
        if operation.get("topic") == "/joint_states"
        and operation.get("op") in {"subscribe", "unsubscribe"}
    ]
    assert [(operation["op"], operation["id"]) for operation in joint_ops] == [
        ("subscribe", "inspect-robots-preflight-joint-states"),
        ("unsubscribe", "inspect-robots-preflight-joint-states"),
        ("subscribe", "inspect-robots-joint-states"),
    ]
    assert joint_ops[0]["throttle_rate"] == 0
    assert joint_ops[0]["queue_length"] == 1
    assert joint_ops[2]["throttle_rate"] == 50
    assert joint_ops[2]["queue_length"] == 1
    assert set(stub_server.subscriptions) == {"inspect-robots-joint-states"}
    embodiment.close()


def test_rosbridge_preflight_high_rate_publisher_above_poll_frequency_no_warning(
    stub_server: StubRosbridgeServer,
) -> None:
    """A 500 Hz publisher at control_hz=60 must not warn.

    The client poll loop coalesces messages faster than its 10 ms poll into the
    latest-value slot, so a sample-count rate estimate caps at ~100 Hz and
    falsely flags any rig whose 2x-control_hz threshold exceeds that cap; the
    sequence-delta estimate sees the coalesced messages.
    """
    embodiment = _embodiment(url=stub_server.url, control_hz=60.0)
    with (
        _topic_streams(stub_server, {"/joint_states": _joint_message()}, interval_s=0.002),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        observation = embodiment.reset(_SCENE)
    assert observation.instruction == _SCENE.instruction
    rate_warnings = [
        warning
        for warning in caught
        if "native rate" in str(warning.message) or "too few" in str(warning.message)
    ]
    assert not rate_warnings
    embodiment.close()


def test_rosbridge_preflight_warns_for_slow_native_joint_state_rate(
    stub_server: StubRosbridgeServer,
) -> None:
    embodiment = _embodiment(url=stub_server.url, control_hz=50.0)
    with (
        _topic_streams(stub_server, {"/joint_states": _joint_message()}, interval_s=0.03),
        pytest.warns(RuntimeWarning, match=r"native rate.*below 2x control_hz"),
    ):
        embodiment.reset(_SCENE)
    embodiment.close()


def test_rosbridge_preflight_single_message_warns_then_hands_off(
    stub_server: StubRosbridgeServer,
) -> None:
    stop = threading.Event()

    def publish_once_then_rollout() -> None:
        stub_server.wait_for(
            lambda ops: any(
                operation.get("op") == "subscribe"
                and operation.get("id") == "inspect-robots-preflight-joint-states"
                for operation in ops
            )
        )
        stub_server.publish("/joint_states", _joint_message())
        stub_server.wait_for(
            lambda ops: any(
                operation.get("op") == "subscribe"
                and operation.get("id") == "inspect-robots-joint-states"
                for operation in ops
            ),
            timeout_s=2.0,
        )
        while not stop.is_set():
            stub_server.publish("/joint_states", _joint_message())
            time.sleep(0.005)

    publisher = threading.Thread(target=publish_once_then_rollout, daemon=True)
    publisher.start()
    embodiment = _embodiment(url=stub_server.url, obs_timeout_s=1.0)
    try:
        with pytest.warns(RuntimeWarning, match="exactly one message"):
            embodiment.reset(_SCENE)
    finally:
        stop.set()
        publisher.join(timeout=2)
        assert not publisher.is_alive()
        embodiment.close()


def test_rosbridge_zero_joint_messages_fall_through_to_missing_topic_error(
    stub_server: StubRosbridgeServer,
) -> None:
    embodiment = _embodiment(url=stub_server.url, obs_timeout_s=0.03)
    with pytest.raises(TimeoutError, match=r"/joint_states.*obs_timeout_s=0.03"):
        embodiment.reset(_SCENE)
    embodiment.close()


def test_rosbridge_missing_optional_topic_names_that_topic(
    stub_server: StubRosbridgeServer,
) -> None:
    embodiment = _embodiment(
        url=stub_server.url,
        eef_pose_topic="/eef_pose",
        obs_timeout_s=0.05,
    )
    with (
        _topic_streams(stub_server, {"/joint_states": _joint_message()}),
        pytest.raises(TimeoutError, match=r"/eef_pose.*obs_timeout_s=0.05"),
    ):
        embodiment.reset(_SCENE)
    embodiment.close()


def test_rosbridge_camera_resolution_mismatch_names_declared_and_actual(
    stub_server: StubRosbridgeServer,
) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    embodiment = _embodiment(
        url=stub_server.url,
        cameras={"wrist": ("/camera/compressed", 3, 4)},
    )
    with (
        _topic_streams(
            stub_server,
            {
                "/joint_states": _joint_message(),
                "/camera/compressed": _compressed_image_message(image, "PNG", "png"),
            },
        ),
        pytest.raises(ValueError, match=r"declared resolution 4x3.*first frame is 3x2"),
    ):
        embodiment.reset(_SCENE)
    embodiment.close()


def test_rosbridge_subscribes_every_modality_with_declared_throttles_and_types(
    stub_server: StubRosbridgeServer,
) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    embodiment = _embodiment(
        url=stub_server.url,
        control_hz=20.0,
        eef_pose_topic="/eef_pose",
        cameras={"wrist": ("/camera/compressed", 2, 3)},
        reset_service="/home",
    )
    with _topic_streams(
        stub_server,
        {
            "/joint_states": _joint_message(),
            "/eef_pose": _pose_message(),
            "/camera/compressed": _compressed_image_message(image, "PNG", "png"),
        },
    ):
        observation = embodiment.reset(_SCENE)
    assert observation.images["wrist"].shape == (2, 3, 3)
    subscriptions = {operation["id"]: operation for operation in stub_server.ops_of("subscribe")}
    assert subscriptions["inspect-robots-preflight-joint-states"]["throttle_rate"] == 0
    assert subscriptions["inspect-robots-joint-states"]["throttle_rate"] == 25
    assert subscriptions["inspect-robots-eef-pose"]["throttle_rate"] == 25
    assert subscriptions["inspect-robots-camera-wrist"]["throttle_rate"] == 50
    assert subscriptions["inspect-robots-eef-pose"]["type"] == ("geometry_msgs/msg/PoseStamped")
    assert subscriptions["inspect-robots-camera-wrist"]["type"] == (
        "sensor_msgs/msg/CompressedImage"
    )
    assert all(operation["queue_length"] == 1 for operation in subscriptions.values())
    assert stub_server.ops_of("call_service")[0]["service"] == "/home"
    assert stub_server.ops_of("advertise")[0]["type"] == ("trajectory_msgs/msg/JointTrajectory")
    embodiment.close()


def test_rosbridge_reset_service_is_called_and_false_result_raises(
    stub_server: StubRosbridgeServer,
) -> None:
    stub_server.service_results["/home"] = (False, {"reason": "blocked"})
    embodiment = _embodiment(
        url=stub_server.url,
        reset_service="/home",
    )
    with (
        _topic_streams(stub_server, {"/joint_states": _joint_message()}),
        pytest.raises(RosbridgeError, match=r"/home.*result=false.*blocked"),
    ):
        embodiment.reset(_SCENE)
    assert stub_server.ops_of("call_service")[0]["args"] == {}
    embodiment.close()


def test_rosbridge_reset_returns_message_newer_than_initial_cache(
    stub_server: StubRosbridgeServer,
) -> None:
    counter = 0

    def changing_joint_state() -> Mapping[str, Any]:
        nonlocal counter
        counter += 1
        return _joint_message([float(counter), 0.03, float(counter) + 0.5])

    embodiment = _embodiment(url=stub_server.url)
    with _topic_streams(stub_server, {"/joint_states": changing_joint_state}):
        observation = embodiment.reset(_SCENE)
    assert counter > 5
    assert observation.state["joint_pos"][0] > 1.0
    embodiment.close()


def test_rosbridge_end_to_end_step_publishes_and_threads_instruction(
    stub_server: StubRosbridgeServer,
) -> None:
    embodiment = _embodiment(url=stub_server.url, fresh_obs_timeout_s=0.3)
    with _topic_streams(stub_server, {"/joint_states": _joint_message()}):
        reset_observation = embodiment.reset(_SCENE)
        step_result = embodiment.step(Action(data=np.asarray([0.4, 0.5])))
    assert (
        reset_observation.instruction == step_result.observation.instruction == _SCENE.instruction
    )
    command = [
        operation
        for operation in stub_server.ops_of("publish")
        if operation.get("topic") == "/arm/command"
    ][-1]
    assert command["msg"]["points"][0]["positions"] == [0.4, 0.5]
    embodiment.close()
