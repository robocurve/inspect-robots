"""Tests for the strict dependency-free stored-frame PNG encoder."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from inspect_robots._pngenc import encode_png, png_data_url


@pytest.mark.parametrize("shape", [(3, 5), (3, 5, 1), (3, 5, 3), (3, 5, 4)])
def test_encode_png_signature_and_ihdr_dimensions(shape: tuple[int, ...]) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    encoded = encode_png(image)

    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    assert encoded[12:16] == b"IHDR"
    assert struct.unpack(">II", encoded[16:24]) == (5, 3)


def test_encode_png_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype uint8"):
        encode_png(np.zeros((2, 3, 3), dtype=np.float32))


@pytest.mark.parametrize("shape", [(4,), (2, 3, 2)])
def test_encode_png_rejects_unsupported_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="unsupported PNG array shape"):
        encode_png(np.zeros(shape, dtype=np.uint8))


def test_png_data_url_has_inline_png_prefix() -> None:
    assert png_data_url(np.zeros((1, 1), dtype=np.uint8)).startswith("data:image/png;base64,iVBOR")
