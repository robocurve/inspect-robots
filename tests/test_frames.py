"""Stored frame identities must remain distinct across writes and readers."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import numpy as np
import pytest

from inspect_robots.frames import FrameStore, _safe


@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_frame_identities_preserve_pixels(tmp_path: Path, reverse: bool) -> None:
    identities = [("pick-e0", "top-e0_rgb", 11), ("pick-e0_top-e0", "rgb", 22)]
    if reverse:
        identities.reverse()
    store = FrameStore(str(tmp_path))
    refs = [
        (store.put(trial, 0, camera, np.full((2, 3, 3), value, dtype=np.uint8)), value)
        for trial, camera, value in identities
    ]
    assert len({ref.path for ref, _ in refs}) == 2
    assert len(list(tmp_path.glob("*.npy"))) == 2
    assert store.count == 2
    for ref, value in refs:
        np.testing.assert_array_equal(ref.load(), np.full((2, 3, 3), value, dtype=np.uint8))


def test_non_colliding_control(tmp_path: Path) -> None:
    store = FrameStore(str(tmp_path))
    refs = [
        store.put(trial, 0, "rgb", np.full((2, 3, 3), value, dtype=np.uint8))
        for trial, value in [("first-e0", 11), ("second-e0", 22)]
    ]
    assert len({ref.path for ref in refs}) == 2
    assert len(list(tmp_path.glob("*.npy"))) == 2
    for ref, value in zip(refs, [11, 22], strict=True):
        assert np.all(ref.load() == value)


def test_steps_rewrites_and_second_store(tmp_path: Path) -> None:
    store = FrameStore(str(tmp_path))
    refs = [
        store.put("pick-e0", step, "top-e0_rgb", np.array([step], dtype=np.uint8))
        for step in [0, 1]
    ]
    other = FrameStore(str(tmp_path))
    distinct = other.put("pick-e0_top-e0", 0, "rgb", np.array([22], dtype=np.uint8))
    rewrite = other.put("pick-e0", 0, "top-e0_rgb", np.array([33], dtype=np.uint8))
    assert rewrite.path == refs[0].path
    assert len(list(tmp_path.glob("*.npy"))) == 3
    assert store.count == other.count == 2
    assert refs[0].load().tolist() == [33]
    assert refs[1].load().tolist() == [1]
    assert distinct.load().tolist() == [22]


@pytest.mark.parametrize("step", [-1, 0, 999999, 1000000])
def test_versioned_names_round_trip_exact_identities(tmp_path: Path, step: int) -> None:
    from inspect_robots.frames import _parse_frame_filename

    names = ["a_b", "a%b", "a~b", "a/b", "a\\b", "a.b-0", "A", "a", "é", "é", "", ".."]
    store = FrameStore(str(tmp_path))
    paths = []
    for trial in names:
        for camera in names:
            ref = store.put(trial, step, camera, np.array([11], dtype=np.uint8))
            path = Path(ref.path)
            assert path.parent == tmp_path
            assert _parse_frame_filename(path.name) == (trial, camera, step)
            assert ref.camera == camera and ref.t == step
            paths.append(path.name.casefold())
    assert len(set(paths)) == len(names) ** 2


def test_versioned_filename_contract() -> None:
    from inspect_robots.frames import _frame_filename

    assert _frame_filename("pick-e0", "top-e0_rgb", 0) == "~f1~pick-e0~top-e0_rgb_000000.npy"
    assert _frame_filename("pick-e0_top-e0", "rgb", 0) == "~f1~pick-e0_top-e0~rgb_000000.npy"
    assert _frame_filename("A%~", "é/", 1000000) == "~f1~%41%25%7E~%C3%A9%2F_1000000.npy"


@pytest.mark.parametrize(
    "name",
    [
        "pick-e0_rgb_000000.npy",
        "~f2~a~b_000000.npy",
        "~f1~a~b~c_000000.npy",
        "~f1~a~b_0.npy",
        "~f1~a~b_0000000.npy",
        "~f1~a~b_-00000.npy",
        "~f1~A~b_000000.npy",
        "~f1~%61~b_000000.npy",
        "~f1~%7e~b_000000.npy",
        "~f1~%~b_000000.npy",
        "~f1~%GG~b_000000.npy",
        "~f1~%FF~b_000000.npy",
        "~f1~a/b~c_000000.npy",
        "~f1~é~b_000000.npy",
        "~f1~a~b_+00000.npy",
        "~f1~a~b_000000.npy\n",
        "~f1~a~b_000000.npz",
    ],
)
def test_parser_rejects_noncanonical_names(name: str) -> None:
    from inspect_robots.frames import _parse_frame_filename

    assert _parse_frame_filename(name) is None


@pytest.mark.parametrize("native", [False, True])
def test_write_length_error_preserves_existing_frames_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    store = FrameStore(str(tmp_path))
    frame = np.full((2, 2, 3), 11, dtype=np.uint8)
    ref = store.put("existing", 0, "rgb", frame)
    legacy = tmp_path / f"{'A' * 100}_rgb_000000.npy"
    np.save(legacy, frame)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    error = OSError(errno.EINVAL if native else errno.ENAMETOOLONG, "filename too long")
    if native:
        error.winerror = 206  # type: ignore[attr-defined]
    attempts: list[Path] = []

    def fail(path: Path, _image: object) -> None:
        attempts.append(path)
        raise error

    monkeypatch.setattr("inspect_robots.frames.np.save", fail)
    with pytest.raises(OSError) as caught:
        store.put("A" * 100, 0, "rgb", frame)
    assert caught.value is error
    assert store.count == 1
    assert len(attempts) == 1 and attempts[0].name.startswith("~f1~")
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
    np.testing.assert_array_equal(ref.load(), frame)


def test_actual_filesystem_length_error_has_no_legacy_retry(tmp_path: Path) -> None:
    if not hasattr(os, "pathconf"):
        pytest.skip("filesystem component limit query unavailable")
    limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    if limit <= 32 or limit > 4096:
        pytest.skip("filesystem has no practical testable component limit")
    trial = "A" * (limit // 2)
    frame = np.full((2, 2, 3), 11, dtype=np.uint8)
    legacy = tmp_path / f"{_safe(trial)}_rgb_000000.npy"
    np.save(legacy, frame)
    store = FrameStore(str(tmp_path))
    existing = store.put("existing", 0, "rgb", frame)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(OSError) as caught:
        store.put(trial, 0, "rgb", np.full_like(frame, 22))
    assert caught.value.errno == errno.ENAMETOOLONG
    assert store.count == 1
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
    np.testing.assert_array_equal(existing.load(), frame)
