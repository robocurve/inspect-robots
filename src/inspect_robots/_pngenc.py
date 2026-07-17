"""Encode stored uint8 camera frames as PNGs without optional dependencies."""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import numpy as np
import numpy.typing as npt

_COLOR_TYPE_BY_CHANNELS = {1: 0, 3: 2, 4: 6}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def encode_png(image: npt.NDArray[Any]) -> bytes:
    """Encode a uint8 ``(H, W)`` or ``(H, W, {1,3,4})`` array as PNG bytes.

    Non-uint8 input raises ``TypeError`` rather than being silently coerced.
    Unsupported shapes raise ``ValueError``.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(f"PNG input must have dtype uint8, got {arr.dtype}")
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3 or arr.shape[2] not in _COLOR_TYPE_BY_CHANNELS:
        raise ValueError(f"unsupported PNG array shape {arr.shape}")
    arr = np.ascontiguousarray(arr)
    height, width, channels = arr.shape
    color_type = _COLOR_TYPE_BY_CHANNELS[channels]
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def png_data_url(image: npt.NDArray[Any]) -> str:
    """Return a PNG encoded as an inline base64 data URL."""
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")
