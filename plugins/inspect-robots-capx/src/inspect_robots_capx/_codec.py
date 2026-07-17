"""CaP-X HTTP wire codecs isolated for protocol-drift review.

The schemas mirror ``capx/serving`` in https://github.com/capgym/cap-x as
inspected on ``main`` in July 2026, when that repository contained seven
commits. The source plan did not preserve the upstream Git SHA, so it cannot be
truthfully recorded here from this network-isolated worktree. The small golden
tests below this module pin the exact bytes and shapes that were inspected.
"""

from __future__ import annotations

import base64
import io
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from inspect_robots_agent import encode_png


def png_b64_encode(rgb: npt.NDArray[Any]) -> str:
    """Encode one RGB observation as bare base64 PNG for SAM3 requests."""
    return base64.b64encode(encode_png(rgb)).decode("ascii")


def npy_b64_encode(array: npt.NDArray[Any]) -> str:
    """Encode an array as a base64 ``.npy`` payload without object pickles."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def npy_b64_decode(payload: str) -> npt.NDArray[Any]:
    """Decode a CaP-X base64 ``.npy`` payload while refusing object pickles."""
    raw = base64.b64decode(payload, validate=True)
    decoded = np.load(io.BytesIO(raw), allow_pickle=False)
    return cast(npt.NDArray[Any], decoded)


def mask_decode(payload: str, shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    """Decode a SAM3 raw-byte mask and normalize it to a boolean image."""
    raw = base64.b64decode(payload, validate=True)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.bool_)


def grasp_arrays_decode(
    grasps_payload: str,
    scores_payload: str,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Decode grasp poses/scores and normalize CaP-X's flat empty arrays.

    Contact-GraspNet serializes a no-grasp result as two ``shape == (0,)``
    arrays. Callers always receive the ordinary public shapes ``(K, 4, 4)``
    and ``(K,)``, including ``K == 0``.
    """
    grasps = npy_b64_decode(grasps_payload)
    scores = npy_b64_decode(scores_payload)
    if grasps.size == 0:
        grasps = grasps.reshape((0, 4, 4))
    if scores.size == 0:
        scores = scores.reshape((0,))
    return grasps, scores
