"""Tests for the rosbridge embodiment adapter and its wire protocol."""

from __future__ import annotations

import pytest

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
