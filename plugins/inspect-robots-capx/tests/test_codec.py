"""Golden CaP-X wire payloads keep the dependency-free codecs honest."""

from __future__ import annotations

import base64

import numpy as np

from inspect_robots_capx._codec import (
    grasp_arrays_decode,
    mask_decode,
    npy_b64_decode,
    npy_b64_encode,
    png_b64_encode,
)

_DEPTH_WIRE = (
    "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzog"
    "KDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgIAoAAIA+AADAPwAAAEAAAAAA"
)
_EMPTY_WIRE = (
    "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzog"
    "KDAsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgIAo="
)
_ONE_GRASP_WIRE = (
    "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzog"
    "KDEsIDQsIDQpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgIAoAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAA"
    "AAAAAAAIA/"
)
_ONE_SCORE_WIRE = (
    "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzog"
    "KDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgIAoAAEA/"
)


def test_float32_depth_matches_recorded_npy_wire() -> None:
    depth = np.array([[0.25, 1.5], [2.0, 0.0]], dtype=np.float32)

    assert npy_b64_encode(depth) == _DEPTH_WIRE
    decoded = npy_b64_decode(_DEPTH_WIRE)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, depth)


def test_raw_mask_payload_decodes_to_bool_shape() -> None:
    expected = np.array([[True, False, True], [False, True, True]])
    payload = base64.b64encode(expected.astype(np.uint8).tobytes()).decode("ascii")

    decoded = mask_decode(payload, (2, 3))

    assert payload == "AQABAAEB"
    assert decoded.dtype == np.bool_
    assert np.array_equal(decoded, expected)


def test_recorded_grasp_payloads_keep_capx_shapes() -> None:
    grasps, scores = grasp_arrays_decode(_ONE_GRASP_WIRE, _ONE_SCORE_WIRE)

    assert grasps.shape == (1, 4, 4)
    assert scores.shape == (1,)
    assert np.array_equal(grasps[0], np.eye(4, dtype=np.float32))
    assert np.array_equal(scores, np.array([0.75], dtype=np.float32))


def test_flat_empty_grasp_payloads_normalize_to_public_shapes() -> None:
    raw = npy_b64_decode(_EMPTY_WIRE)
    assert raw.shape == (0,)

    grasps, scores = grasp_arrays_decode(_EMPTY_WIRE, _EMPTY_WIRE)

    assert grasps.shape == (0, 4, 4)
    assert scores.shape == (0,)


def test_png_encoding_is_bare_base64_not_a_data_url() -> None:
    encoded = png_b64_encode(np.zeros((1, 2, 3), dtype=np.uint8))

    assert not encoded.startswith("data:")
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
