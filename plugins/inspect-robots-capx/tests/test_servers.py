"""CaP-X clients speak the recorded schemas with bounded retry behavior."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from conftest import CapxStub
from inspect_robots_capx._codec import npy_b64_decode
from inspect_robots_capx._servers import (
    CapxServerClients,
    GraspNetClient,
    PyrokiClient,
    Sam3Client,
)


class _FakeTime:
    def __init__(self) -> None:
        self.now = 10.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_clients_are_lazy_and_share_one_injected_http_pool(capx_stub: CapxStub) -> None:
    servers = CapxServerClients(
        sam3_url="http://sam.test",
        graspnet_url="http://grasp.test",
        pyroki_url="http://ik.test",
        transport=httpx.MockTransport(capx_stub.handler),
    )

    assert capx_stub.requests == []
    assert servers.sam3._http is servers.graspnet._http is servers.pyroki._http

    servers.close()
    assert servers._http.is_closed


def test_sam3_request_and_response_match_wire_schema(capx_stub: CapxStub) -> None:
    client = Sam3Client("http://sam.test", transport=httpx.MockTransport(capx_stub.handler))

    results = client.segment(np.zeros((2, 2, 3), dtype=np.uint8), "red cube")

    path, body = capx_stub.requests[0]
    assert path == "/segment"
    assert set(body) == {"image_base64", "text_prompt"}
    assert body["text_prompt"] == "red cube"
    assert results[0]["mask"].dtype == np.bool_
    assert np.array_equal(results[0]["mask"], np.eye(2, dtype=np.bool_))
    assert results[0]["box"] == [0, 0, 2, 2]
    client.close()


def test_graspnet_request_pins_defaults_and_npy_arrays(capx_stub: CapxStub) -> None:
    client = GraspNetClient("http://grasp.test", transport=httpx.MockTransport(capx_stub.handler))
    depth = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float64)
    mask = np.array([[True, False], [False, True]])

    grasps, scores = client.plan(depth, intrinsics, mask)

    path, body = capx_stub.requests[0]
    assert path == "/plan"
    assert set(body) == {
        "depth_base64",
        "cam_K_base64",
        "segmap_base64",
        "segmap_id",
        "local_regions",
        "filter_grasps",
        "skip_border_objects",
        "z_range",
        "forward_passes",
        "max_retries",
    }
    assert np.array_equal(npy_b64_decode(body["depth_base64"]), depth)
    assert np.array_equal(npy_b64_decode(body["cam_K_base64"]), intrinsics)
    assert np.array_equal(npy_b64_decode(body["segmap_base64"]), mask.astype(np.uint8))
    tuning = {
        key: value
        for key, value in body.items()
        if key not in {"depth_base64", "cam_K_base64", "segmap_base64"}
    }
    assert tuning == {
        "segmap_id": 1,
        "local_regions": True,
        "filter_grasps": True,
        "skip_border_objects": False,
        "z_range": [0.2, 2.0],
        "forward_passes": 2,
        "max_retries": 10,
    }
    assert grasps.shape == (1, 4, 4)
    assert np.array_equal(scores, np.array([0.8], dtype=np.float32))


def test_transient_503_retries_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="warming")
        return httpx.Response(200, json={"results": []})

    client = Sam3Client(
        "http://sam.test",
        transport=httpx.MockTransport(handler),
        backoff_s=0.0,
    )

    assert client.segment(np.zeros((1, 1, 3), dtype=np.uint8), "cube") == []
    assert calls == 2


def test_connection_error_names_url_and_launch_command() -> None:
    fake_time = _FakeTime()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = Sam3Client(
        "http://gpu-box:8114",
        request_timeout_s=0.25,
        transport=httpx.MockTransport(handler),
        clock=fake_time.monotonic,
        sleep=fake_time.sleep,
    )

    with pytest.raises(ConnectionError) as excinfo:
        client.segment(np.zeros((1, 1, 3), dtype=np.uint8), "cube")
    message = str(excinfo.value)
    assert "http://gpu-box:8114/segment" in message
    assert "launch_sam3_server.py --port 8114" in message


def test_total_timeout_budget_caps_each_attempt() -> None:
    fake_time = _FakeTime()
    attempt_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions["timeout"]
        attempt_timeouts.append(float(timeout["read"]))
        raise httpx.ReadTimeout("still loading", request=request)

    client = Sam3Client(
        "http://sam.test",
        request_timeout_s=1.5,
        transport=httpx.MockTransport(handler),
        clock=fake_time.monotonic,
        sleep=fake_time.sleep,
    )

    with pytest.raises(TimeoutError, match="still loading"):
        client.segment(np.zeros((1, 1, 3), dtype=np.uint8), "cube")
    assert attempt_timeouts == [1.5, 0.5]
    assert fake_time.now == pytest.approx(11.5)


def test_attempt_timeout_is_never_above_thirty_seconds() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"results": []})

    client = Sam3Client(
        "http://sam.test",
        request_timeout_s=120.0,
        transport=httpx.MockTransport(handler),
    )
    client.segment(np.zeros((1, 1, 3), dtype=np.uint8), "cube")

    assert seen[0]["read"] == 30.0


def test_pyroki_strips_arm_and_reuses_full_config_as_warm_start() -> None:
    bodies: list[dict[str, Any]] = []
    responses = [[0.1, -0.2, 0.7], [0.2, -0.1, 0.6]]

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"joint_positions": responses.pop(0)})

    client = PyrokiClient("http://ik.test", transport=httpx.MockTransport(handler))
    position = np.array([0.4, 0.1, 0.2])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    first = client.solve_ik(position, quaternion, arm_dof=2)
    second = client.solve_ik(position, quaternion, arm_dof=2)

    assert bodies[0] == {
        "target_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.2],
        "prev_cfg": None,
    }
    assert bodies[1]["prev_cfg"] == [0.1, -0.2, 0.7]
    assert np.array_equal(first, np.array([0.1, -0.2]))
    assert np.array_equal(second, np.array([0.2, -0.1]))

    client.reset()
    assert client._prev_cfg is None


def test_pyroki_short_response_is_actionable_and_not_warm_started() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"joint_positions": [0.1]})

    client = PyrokiClient("http://ik.test", transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError, match="URDF matching this embodiment"):
        client.solve_ik(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), arm_dof=2)
    assert client._prev_cfg is None
