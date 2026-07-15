"""Tests for the rosbridge embodiment adapter and its wire protocol."""

from __future__ import annotations

import base64
import threading
import time
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from _stub_server import StubRosbridgeServer
from PIL import Image

from inspect_robots_ros._client import RosbridgeClient
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


def test_client_sends_while_receive_thread_updates_topic_cache(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    client.connect()
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

    publisher = threading.Thread(target=stream)
    publisher.start()
    for index in range(20):
        client.publish("/arm/command", {"data": [float(index)]})
    publisher.join(timeout=2)

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
        throttle_rate=50,
    )
    stub_server.wait_for(lambda ops: any(op.get("op") == "subscribe" for op in ops))
    stub_server.publish("/joint_states", {"position": [1.0]})
    first = client.wait_for_sample("/joint_states", timeout_s=1.0)
    stub_server.publish("/joint_states", {"position": [2.0]})
    second = client.wait_for_sample("/joint_states", after_seq=first.seq, timeout_s=1.0)
    assert first.stamp == second.stamp == 7.0
    assert second.seq == first.seq + 1
    client.close()


def test_client_status_error_latches_and_surfaces_on_every_later_call(
    stub_server: StubRosbridgeServer,
) -> None:
    client = _client(stub_server)
    client.connect()
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

    caller = threading.Thread(target=call)
    caller.start()
    stub_server.wait_for(lambda ops: any(op.get("op") == "call_service" for op in ops))
    request_id = stub_server.ops_of("call_service")[0]["id"]
    stub_server.send_service_response("wrong-id", values={"wrong": True})
    time.sleep(0.02)
    assert caller.is_alive()
    stub_server.send_service_response(request_id, values={"homed": True})
    caller.join(timeout=2)
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

    unreachable = RosbridgeClient("ws://127.0.0.1:9", connect_timeout_s=0.05)
    with pytest.raises(ConnectionError, match=r"ws://127\.0\.0\.1:9"):
        unreachable.connect()
    unreachable.close()


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
