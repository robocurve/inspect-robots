from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots_capx._codec import npy_b64_encode


@dataclass
class CapxStub:
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/segment":
            mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "mask_base64": base64.b64encode(mask.tobytes()).decode("ascii"),
                            "shape": [2, 2],
                            "box": [0, 0, 2, 2],
                            "score": 0.9,
                            "label": "cube",
                        }
                    ]
                },
            )
        if request.url.path == "/plan":
            return httpx.Response(
                200,
                json={
                    "grasps_base64": npy_b64_encode(np.eye(4, dtype=np.float32)[None]),
                    "scores_base64": npy_b64_encode(np.array([0.8], dtype=np.float32)),
                    "contact_pts_base64": npy_b64_encode(
                        np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
                    ),
                },
            )
        if request.url.path == "/ik":
            return httpx.Response(200, json={"joint_positions": [0.1, -0.2, 0.7]})
        raise AssertionError(f"unexpected request path {request.url.path}")


@pytest.fixture()
def capx_stub() -> CapxStub:
    return CapxStub()
