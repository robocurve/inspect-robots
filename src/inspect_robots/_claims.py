"""Advisory device claims held for the lifetime of open file descriptors.

Claims use POSIX ``flock`` locks. The kernel releases them when a process
dies, so stale locks cannot exist and lockfiles never need to be removed.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inspect_robots.conformance import DeviceSlot


class DeviceClaim:
    """Held advisory locks on a rig's devices; release() is idempotent."""

    def __init__(self, fds: list[int]) -> None:
        self._fds = fds

    def release(self) -> None:
        """Release every held device lock, tolerating already-closed handles."""
        if not self._fds:
            return

        import fcntl

        fds, self._fds = self._fds, []
        for fd in fds:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except OSError:
                # Unlock or close may race an external close; release is best-effort and idempotent.
                pass


def _lock_dir(env: Mapping[str, str]) -> Path | None:
    runtime_value = env.get("XDG_RUNTIME_DIR")
    if runtime_value:
        runtime_dir = Path(runtime_value)
    else:
        runtime_dir = Path(tempfile.gettempdir()) / f"inspect-robots-{os.getuid()}"
    lock_dir = runtime_dir / "inspect-robots" / "locks"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_dir_stat = os.lstat(lock_dir)
    except OSError as exc:
        warnings.warn(
            f"cannot create device claim lock directory {lock_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not stat.S_ISDIR(lock_dir_stat.st_mode) or lock_dir_stat.st_uid != os.getuid():
        warnings.warn(
            f"cannot use device claim lock directory {lock_dir}: not a directory we own",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return lock_dir


def _normalize(slot_kind: str, value: str) -> str:
    if slot_kind in ("v4l2", "serial"):
        return str(Path(value).resolve())
    return value


def claim_devices(
    slots: tuple[DeviceSlot, ...],
    kvs: Mapping[str, Any],
    env: Mapping[str, str],
) -> DeviceClaim:
    """Claim configured device-slot values without blocking on environment trouble.

    On platforms without ``fcntl`` this returns before filesystem setup. Thus
    the ``os.getuid`` fallback in lock-directory selection is reached only on
    POSIX, where ``fcntl`` is available.
    """
    try:
        import fcntl
    except ImportError:
        return DeviceClaim([])

    values: list[str] = []
    for slot in slots:
        value = kvs.get(slot.arg)
        if isinstance(value, str) and value:
            normalized = _normalize(slot.kind, value)
            if normalized not in values:
                values.append(normalized)

    if not values:
        return DeviceClaim([])

    lock_dir = _lock_dir(env)
    if lock_dir is None:
        return DeviceClaim([])

    claim = DeviceClaim([])
    for value in values:
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        lock_path = lock_dir / f"{digest}.lock"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            claim.release()
            warnings.warn(
                f"cannot open device claim lock file {lock_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return claim

        claim._fds.append(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                pid = int(os.read(fd, 4096).split()[0])
            except (OSError, ValueError, IndexError):
                holder_suffix = ""
            else:
                holder_suffix = f" (PID {pid})"
            claim.release()
            raise SystemExit(
                f"device {value!r} is already claimed by another inspect-robots"
                f" process{holder_suffix}: two evals must not drive one rig"
            ) from None

        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {value}\n".encode())
        except OSError as exc:
            claim.release()
            warnings.warn(
                f"cannot write device claim lock file {lock_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return claim

    return claim
