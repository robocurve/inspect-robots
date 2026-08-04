import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

from inspect_robots._claims import claim_devices
from inspect_robots.conformance import DeviceSlot


def _slot(arg: str, kind: str = "can") -> DeviceSlot:
    return DeviceSlot(arg=arg, kind=kind, label=f"{arg} device")


def _lock_dir(runtime_dir: Path) -> Path:
    return runtime_dir / "inspect-robots" / "locks"


def _lock_path(runtime_dir: Path, value: str) -> Path:
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    return _lock_dir(runtime_dir) / f"{digest}.lock"


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
class TestClaimsPosix:
    def test_claim_then_conflict_reports_holder_pid(self, tmp_path: Path) -> None:
        slots = (_slot("left_channel"),)
        kvs = {"left_channel": "can0"}
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        claim = claim_devices(slots, kvs, env)

        try:
            with pytest.raises(SystemExit) as exc_info:
                claim_devices(slots, kvs, env)
            assert "can0" in str(exc_info.value)
            assert f"PID {os.getpid()}" in str(exc_info.value)
        finally:
            claim.release()

    def test_release_frees_the_device(self, tmp_path: Path) -> None:
        slots = (_slot("left_channel"),)
        kvs = {"left_channel": "can0"}
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        first = claim_devices(slots, kvs, env)

        first.release()
        first.release()
        second = claim_devices(slots, kvs, env)
        second.release()

    def test_conflict_releases_partial_acquisitions(self, tmp_path: Path) -> None:
        first_slot = _slot("left_channel")
        second_slot = _slot("right_channel")
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        existing = claim_devices((second_slot,), {"right_channel": "can1"}, env)

        try:
            with pytest.raises(SystemExit):
                claim_devices(
                    (first_slot, second_slot),
                    {"left_channel": "can0", "right_channel": "can1"},
                    env,
                )
            recovered = claim_devices((first_slot,), {"left_channel": "can0"}, env)
            recovered.release()
        finally:
            existing.release()

    def test_symlink_spellings_collide(self, tmp_path: Path) -> None:
        target = tmp_path / "video0"
        target.touch()
        link = tmp_path / "camera"
        link.symlink_to(target)
        slot = _slot("camera", "v4l2")
        env = {"XDG_RUNTIME_DIR": str(tmp_path / "runtime")}
        claim = claim_devices((slot,), {"camera": str(link)}, env)

        try:
            with pytest.raises(SystemExit):
                claim_devices((slot,), {"camera": str(target)}, env)
        finally:
            claim.release()

    def test_can_names_taken_verbatim(self, tmp_path: Path) -> None:
        slot = _slot("left_channel")
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        first = claim_devices((slot,), {"left_channel": "can0"}, env)
        second = claim_devices((slot,), {"left_channel": "can1"}, env)

        second.release()
        first.release()

    def test_none_and_missing_and_nonstring_values_skipped(self, tmp_path: Path) -> None:
        slots = (_slot("none"), _slot("missing"), _slot("number"))
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}

        claim = claim_devices(slots, {"none": None, "number": 3}, env)

        assert claim._fds == []
        assert not _lock_dir(tmp_path).exists()

    def test_duplicate_values_claimed_once(self, tmp_path: Path) -> None:
        slots = (_slot("left_channel"), _slot("right_channel"))
        kvs = {"left_channel": "can0", "right_channel": "can0"}
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        claim = claim_devices(slots, kvs, env)

        assert len(list(_lock_dir(tmp_path).iterdir())) == 1
        claim.release()
        replacement = claim_devices((_slot("left_channel"),), {"left_channel": "can0"}, env)
        replacement.release()

    def test_unusable_lock_dir_warns_and_noops(self, tmp_path: Path) -> None:
        file_parent = tmp_path / "not-a-directory"
        file_parent.write_text("blocked", encoding="utf-8")
        runtime_dir = file_parent / "runtime"

        with pytest.warns(RuntimeWarning, match=str(runtime_dir)):
            claim = claim_devices(
                (_slot("left_channel"),),
                {"left_channel": "can0"},
                {"XDG_RUNTIME_DIR": str(runtime_dir)},
            )

        assert claim._fds == []

    def test_symlink_lock_dir_warns_and_noops(self, tmp_path: Path) -> None:
        real_lock_dir = tmp_path / "real-locks"
        real_lock_dir.mkdir()
        lock_dir = _lock_dir(tmp_path)
        lock_dir.parent.mkdir()
        lock_dir.symlink_to(real_lock_dir, target_is_directory=True)

        with pytest.warns(RuntimeWarning, match=str(lock_dir)):
            claim = claim_devices(
                (_slot("left_channel"),),
                {"left_channel": "can0"},
                {"XDG_RUNTIME_DIR": str(tmp_path)},
            )

        assert claim._fds == []
        assert list(real_lock_dir.iterdir()) == []

    def test_foreign_owned_lock_dir_warns_and_noops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actual_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: actual_uid + 1)

        with pytest.warns(RuntimeWarning, match=str(_lock_dir(tmp_path))):
            claim = claim_devices(
                (_slot("left_channel"),),
                {"left_channel": "can0"},
                {"XDG_RUNTIME_DIR": str(tmp_path)},
            )

        assert claim._fds == []

    def test_gettempdir_fallback_used_without_runtime_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        claim = claim_devices((_slot("left_channel"),), {"left_channel": "can0"}, {})
        expected_runtime = tmp_path / f"inspect-robots-{os.getuid()}"

        try:
            assert len(list(_lock_dir(expected_runtime).iterdir())) == 1
        finally:
            claim.release()

    def test_unopenable_lock_file_warns_and_noops(self, tmp_path: Path) -> None:
        lock_path = _lock_path(tmp_path, "can0")
        lock_path.mkdir(parents=True)

        with pytest.warns(RuntimeWarning, match=str(lock_path)):
            claim = claim_devices(
                (_slot("left_channel"),),
                {"left_channel": "can0"},
                {"XDG_RUNTIME_DIR": str(tmp_path)},
            )

        assert claim._fds == []

    def test_symlink_lock_file_warns_and_noops(self, tmp_path: Path) -> None:
        target = tmp_path / "victim"
        target.write_text("keep me", encoding="utf-8")
        lock_path = _lock_path(tmp_path, "can0")
        lock_path.parent.mkdir(parents=True)
        lock_path.symlink_to(target)

        with pytest.warns(RuntimeWarning, match=str(lock_path)):
            claim = claim_devices(
                (_slot("left_channel"),),
                {"left_channel": "can0"},
                {"XDG_RUNTIME_DIR": str(tmp_path)},
            )

        assert claim._fds == []
        assert target.read_text(encoding="utf-8") == "keep me"

    def test_write_failure_warns_releases_and_noops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slots = (_slot("left_channel"),)
        kvs = {"left_channel": "can0"}
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}

        def fail_ftruncate(_fd: int, _length: int) -> None:
            raise OSError("truncate failed")

        with monkeypatch.context() as patch:
            patch.setattr(os, "ftruncate", fail_ftruncate)
            with pytest.warns(RuntimeWarning, match=str(_lock_path(tmp_path, "can0"))):
                claim = claim_devices(slots, kvs, env)

        assert claim._fds == []
        replacement = claim_devices(slots, kvs, env)
        replacement.release()

    def test_conflict_with_unparseable_holder_omits_pid(self, tmp_path: Path) -> None:
        slots = (_slot("left_channel"),)
        kvs = {"left_channel": "can0"}
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        claim = claim_devices(slots, kvs, env)
        _lock_path(tmp_path, "can0").write_text("not-a-pid holder\n", encoding="utf-8")

        try:
            with pytest.raises(SystemExit) as exc_info:
                claim_devices(slots, kvs, env)
            assert "can0" in str(exc_info.value)
            assert "PID" not in str(exc_info.value)
        finally:
            claim.release()

    def test_release_survives_externally_closed_fd(self, tmp_path: Path) -> None:
        slots = (_slot("left_channel"), _slot("right_channel"))
        kvs = {"left_channel": "can0", "right_channel": "can1"}
        env = {"XDG_RUNTIME_DIR": str(tmp_path)}
        claim = claim_devices(slots, kvs, env)

        os.close(claim._fds[0])
        claim.release()

        replacement = claim_devices((_slot("right_channel"),), {"right_channel": "can1"}, env)
        replacement.release()


def test_without_fcntl_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fcntl", None)
    slots = (_slot("left_channel"),)
    kvs = {"left_channel": "can0"}
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}

    first = claim_devices(slots, kvs, env)
    second = claim_devices(slots, kvs, env)

    assert first._fds == []
    assert second._fds == []
    assert not _lock_dir(tmp_path).exists()
