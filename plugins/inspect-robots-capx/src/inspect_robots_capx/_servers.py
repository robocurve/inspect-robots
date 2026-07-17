"""Lazy HTTP clients for the CaP-X perception and inverse-kinematics servers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import httpx
import numpy as np
import numpy.typing as npt

from inspect_robots_capx._codec import (
    grasp_arrays_decode,
    mask_decode,
    npy_b64_encode,
    png_b64_encode,
)

_ATTEMPT_TIMEOUT_S = 30.0

_SAM3_COMMAND = "uv run capx/serving/launch_sam3_server.py --port 8114"
_GRASPNET_COMMAND = "uv run capx/serving/launch_contact_graspnet_server.py --port 8115"
_PYROKI_COMMAND = (
    "uv run python -c 'from capx.serving.launch_pyroki_server import main; "
    'main(robot="panda_description", port=8116)\''
)


class _ServerClient:
    """Retrying JSON POST boundary shared by one logical CaP-X service."""

    def __init__(
        self,
        url: str,
        launch_command: str,
        *,
        request_timeout_s: float,
        http: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not np.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and > 0")
        if not np.isfinite(backoff_s) or backoff_s < 0:
            raise ValueError("backoff_s must be finite and >= 0")
        if http is not None and transport is not None:
            raise ValueError("pass either a shared http client or transport, not both")
        self._base_url = url.rstrip("/")
        self._launch_command = launch_command
        self._request_timeout_s = request_timeout_s
        self._backoff_s = backoff_s
        self._clock = clock
        self._sleep = sleep
        self._owns_http = http is None
        self._http = http or httpx.Client(transport=transport)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self._base_url + path
        deadline = self._clock() + self._request_timeout_s
        attempt = 0
        last_error = "server did not respond"
        timed_out = False
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                error_type = TimeoutError if timed_out else ConnectionError
                raise error_type(self._guidance(url, last_error))
            try:
                response = self._http.post(
                    url,
                    json=body,
                    timeout=min(remaining, _ATTEMPT_TIMEOUT_S),
                )
            except httpx.TimeoutException as exc:
                timed_out = True
                last_error = f"{type(exc).__name__}: {exc}"
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ConnectionError(
                            self._guidance(url, "server returned a non-object JSON payload")
                        )
                    return cast(dict[str, Any], payload)
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in (429,) and response.status_code < 500:
                    raise ConnectionError(self._guidance(url, last_error))

            remaining = deadline - self._clock()
            if remaining <= 0:
                continue
            delay = min(self._backoff_s * 2**attempt, remaining)
            self._sleep(delay)
            attempt += 1

    def _guidance(self, url: str, detail: str) -> str:
        return (
            f"CaP-X server request failed at {url}: {detail}.\n"
            f"launch it from a CaP-X checkout with: {self._launch_command}"
        )

    def close(self) -> None:
        """Release an independently owned HTTP pool, if this client created one."""
        if self._owns_http:
            self._http.close()


class Sam3Client(_ServerClient):
    """Text-prompt segmentation against CaP-X's SAM3 ``/segment`` endpoint."""

    def __init__(
        self,
        url: str,
        *,
        request_timeout_s: float = 120.0,
        http: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            url,
            _SAM3_COMMAND,
            request_timeout_s=request_timeout_s,
            http=http,
            transport=transport,
            backoff_s=backoff_s,
            clock=clock,
            sleep=sleep,
        )

    def segment(self, rgb: npt.NDArray[Any], text: str) -> list[dict[str, Any]]:
        """Return boolean masks and metadata for every SAM3 text match."""
        payload = self._post(
            "/segment",
            {"image_base64": png_b64_encode(rgb), "text_prompt": text},
        )
        results: list[dict[str, Any]] = []
        for raw in payload["results"]:
            shape = tuple(int(value) for value in raw["shape"])
            if len(shape) != 2:
                raise ValueError(f"SAM3 returned invalid mask shape {shape!r}; expected (H, W)")
            results.append(
                {
                    "mask": mask_decode(raw["mask_base64"], (shape[0], shape[1])),
                    "box": raw["box"],
                    "score": raw["score"],
                    "label": raw["label"],
                }
            )
        return results


class GraspNetClient(_ServerClient):
    """Grasp planning against CaP-X's Contact-GraspNet ``/plan`` endpoint."""

    def __init__(
        self,
        url: str,
        *,
        request_timeout_s: float = 120.0,
        http: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            url,
            _GRASPNET_COMMAND,
            request_timeout_s=request_timeout_s,
            http=http,
            transport=transport,
            backoff_s=backoff_s,
            clock=clock,
            sleep=sleep,
        )

    def plan(
        self,
        depth: npt.NDArray[Any],
        intrinsics: npt.NDArray[Any],
        mask: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Return camera-frame grasp poses and scores with stable empty shapes."""
        response = self._post(
            "/plan",
            {
                "depth_base64": npy_b64_encode(np.asarray(depth)),
                "cam_K_base64": npy_b64_encode(np.asarray(intrinsics)),
                "segmap_base64": npy_b64_encode(np.asarray(mask, dtype=np.uint8)),
                "segmap_id": 1,
                "local_regions": True,
                "filter_grasps": True,
                "skip_border_objects": False,
                "z_range": [0.2, 2.0],
                "forward_passes": 2,
                "max_retries": 10,
            },
        )
        return grasp_arrays_decode(response["grasps_base64"], response["scores_base64"])


class PyrokiClient(_ServerClient):
    """Stateful IK client retaining the server's full configuration as warm-start."""

    def __init__(
        self,
        url: str,
        *,
        request_timeout_s: float = 120.0,
        http: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            url,
            _PYROKI_COMMAND,
            request_timeout_s=request_timeout_s,
            http=http,
            transport=transport,
            backoff_s=backoff_s,
            clock=clock,
            sleep=sleep,
        )
        self._prev_cfg: list[float] | None = None

    def reset(self) -> None:
        """Clear the full-config IK warm-start at a trial boundary."""
        self._prev_cfg = None

    def solve_ik(
        self,
        position: npt.NDArray[Any],
        quaternion_wxyz: npt.NDArray[Any],
        arm_dof: int,
    ) -> npt.NDArray[np.float64]:
        """Solve one pose and return the leading embodiment arm joints only."""
        position_array = np.asarray(position, dtype=np.float64).reshape(-1)
        quaternion_array = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
        if position_array.shape != (3,) or quaternion_array.shape != (4,):
            raise ValueError("solve_ik expects position shape (3,) and quaternion_wxyz shape (4,)")
        response = self._post(
            "/ik",
            {
                "target_pose_wxyz_xyz": [
                    *quaternion_array.tolist(),
                    *position_array.tolist(),
                ],
                "prev_cfg": self._prev_cfg,
            },
        )
        full = np.asarray(response["joint_positions"], dtype=np.float64).reshape(-1)
        if len(full) < arm_dof:
            raise ValueError(
                "Pyroki returned "
                f"{len(full)} joints but the embodiment arm needs {arm_dof}; "
                "launch the Pyroki server with the URDF matching this embodiment"
            )
        self._prev_cfg = full.tolist()
        return full[:arm_dof].copy()


class CapxServerClients:
    """The three CaP-X service clients sharing one injectable HTTP connection pool."""

    def __init__(
        self,
        *,
        sam3_url: str,
        graspnet_url: str,
        pyroki_url: str,
        request_timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(transport=transport)
        common: dict[str, Any] = {
            "request_timeout_s": request_timeout_s,
            "http": self._http,
            "backoff_s": backoff_s,
            "clock": clock,
            "sleep": sleep,
        }
        self.sam3 = Sam3Client(sam3_url, **common)
        self.graspnet = GraspNetClient(graspnet_url, **common)
        self.pyroki = PyrokiClient(pyroki_url, **common)

    def close(self) -> None:
        """Release the single shared HTTP connection pool."""
        self._http.close()
