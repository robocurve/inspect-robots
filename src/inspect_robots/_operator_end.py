"""Process-wide ``SIGURG`` end-of-episode request that does not wait for the policy.

A keypress End is read between steps, so a trial that is blocked on a slow model
call first receives the reply and executes it. Remote supervisors (the rig
dashboard) can instead send ``SIGURG`` to the evaluator. While the rollout
thread is waiting on the policy, the handler raises ``OperatorEndRequested``
into that wait, so the reply is discarded unexecuted. At any other time the
signal only sets a flag that the rollout checks before the next policy call and
before every ``embodiment.step()``; it never interrupts motion already in flight.

``SIGURG`` is used because its default disposition is to ignore: if this
handler is missing or another component has replaced it, a misdirected End is
dropped instead of terminating the evaluator unparked. Python installs no
handler for it, and it is raised by the kernel only for socket out-of-band data
on a socket whose owner was set with ``F_SETOWN``.

The handler is installed once, on the main thread, and only over the default
disposition, so it never replaces another owner. Two files, both opt-in through
environment variables a supervisor sets when it launches the evaluator, form the
protocol:

- ``INSPECT_ROBOTS_END_SIGNAL_READY_FILE``: the PID is written here after
  installation, unless the run cannot honor End (parallel ``eval-set``). It is a
  hint for choosing signalled End over the keypress, not a guarantee.
- ``INSPECT_ROBOTS_END_SIGNAL_ACK_FILE``: the rollout writes the PID here when a
  main-thread trial has ended because of End, so no further action can start.
  The handler never writes it: a step can already be committed when the signal
  lands, and that step finishes before the ack. A supervisor removes the file,
  sends ``SIGURG``, and treats End as delivered only once the file names the PID.
"""

from __future__ import annotations

import os
import signal
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from types import FrameType

READY_FILE_ENV = "INSPECT_ROBOTS_END_SIGNAL_READY_FILE"
ACK_FILE_ENV = "INSPECT_ROBOTS_END_SIGNAL_ACK_FILE"


class OperatorEndRequested(BaseException):
    """Unwind a policy wait to end the trial; not an ``Exception``, so retries skip it."""


class _State:
    installed = False
    requested = False
    interruptible = False


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _write_pid(path: str) -> None:
    """Atomically replace ``path`` with this PID; raises ``OSError`` on failure."""
    partial = f"{path}.{os.getpid()}.tmp"
    with open(partial, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    os.replace(partial, path)


def _handle(signum: int, frame: FrameType | None) -> None:
    _State.requested = True
    if _State.interruptible:
        # Disarm before raising so a repeated signal cannot interrupt the unwind.
        _State.interruptible = False
        raise OperatorEndRequested()


def _refuse(reason: str, *, always: bool) -> bool:
    if always or os.environ.get(READY_FILE_ENV):
        warnings.warn(
            f"signalled End (SIGURG) unavailable: {reason}; End waits for the policy",
            RuntimeWarning,
            stacklevel=3,
        )
    return False


def _retract_ready() -> None:
    path = os.environ.get(READY_FILE_ENV)
    if not path:
        return
    # Delete only a file this process wrote, never one a later launch reused.
    owned = False
    with suppress(OSError), open(path, encoding="utf-8") as handle:
        owned = handle.read().strip() == str(os.getpid())
    if owned:
        with suppress(OSError):
            os.remove(path)


def install(*, advertise: bool = True) -> bool:
    """Install the handler once; return whether this process now honors the signal.

    Returns ``False`` without installing off POSIX, off the main thread, or when
    ``SIGURG`` already has a non-default handler (which also warns). On success
    with ``advertise``, writes the PID to the ready file when a supervisor set
    one. Callers whose rollouts run on worker threads pass ``advertise=False``.
    """
    urg = getattr(signal, "SIGURG", None)
    if urg is None:
        return _refuse("this platform has no SIGURG", always=False)
    if not _State.installed:
        if not _on_main_thread():
            return _refuse("not installed from the main thread", always=False)
        if signal.getsignal(urg) is not signal.SIG_DFL:
            return _refuse("another handler already owns SIGURG", always=True)
        signal.signal(urg, _handle)
        _State.installed = True
    path = os.environ.get(READY_FILE_ENV)
    if not path:
        return True
    if not advertise:
        warnings.warn(
            "signalled End (SIGURG) not advertised: this run executes trials on worker "
            "threads, which ignore it",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    try:
        _write_pid(path)
    except OSError as exc:
        warnings.warn(
            f"signalled End installed but its ready file {path!r} could not be written "
            f"({exc}); a supervisor will treat End as unsupported",
            RuntimeWarning,
            stacklevel=2,
        )
    return True


def active() -> bool:
    """True when this handler still owns ``SIGURG`` and the caller can be interrupted.

    Rollouts on worker threads (parallel simulation trials) ignore the signal
    entirely, so one trial cannot clear or consume another's request. If some
    other component replaced the handler after installation, End is withdrawn:
    the ready file is deleted and a warning is issued once. Until then, a
    replacement only means End goes unacknowledged.
    """
    if not _State.installed or not _on_main_thread():
        return False
    if signal.getsignal(signal.SIGURG) is not _handle:
        _State.installed = False
        _retract_ready()
        warnings.warn(
            "signalled End (SIGURG) withdrawn: another handler replaced it; "
            "End waits for the policy",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def begin_trial() -> None:
    """Discard a request that arrived before this trial started."""
    _State.requested = False


def acknowledge() -> None:
    """Record that a trial ended because of End; a write failure leaves no ack."""
    path = os.environ.get(ACK_FILE_ENV)
    if path:
        # A missing ack is the supervisor's cue that End did not land.
        with suppress(OSError):
            _write_pid(path)


def check() -> None:
    """Raise ``OperatorEndRequested`` if an end request arrived; no-op when inactive."""
    if active() and _State.requested:
        raise OperatorEndRequested()


@contextmanager
def interruptible() -> Iterator[None]:
    """Let an end request abort the enclosed wait, and reject its result afterwards."""
    if not active():
        yield
        return
    # Arm before checking: a signal landing between the two then raises from
    # the handler, so no request can slip into the wait as a flag only.
    _State.interruptible = True
    try:
        check()
        yield
    finally:
        _State.interruptible = False
    if _State.requested:
        # The wait finished (or a policy swallowed the interrupt) after the
        # request arrived: its result is late and must not be executed.
        raise OperatorEndRequested()
