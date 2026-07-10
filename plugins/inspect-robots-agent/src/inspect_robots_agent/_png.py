"""Minimal stdlib PNG encoder for camera frames (no Pillow dependency).

LLM APIs take images as base64 data URLs; this encodes an ``(H, W, C)``
uint8/float array as an uncompressed-filter PNG using only zlib + struct.
"""

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
    """Encode an ``(H, W)`` or ``(H, W, {1,3,4})`` array as PNG bytes.

    Float arrays are assumed normalized to [0, 1] and scaled; everything else
    is cast to uint8.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
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
    """The ``data:image/png;base64,...`` form LLM APIs accept inline."""
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")
