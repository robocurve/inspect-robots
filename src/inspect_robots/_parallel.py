"""Bounded task dispatch shared by the Python and CLI evaluation entry points."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, suppress
from itertools import islice
from types import TracebackType
from typing import ParamSpec, TypeVar

from inspect_robots.errors import ConfigError, EmbodimentFault, SafetyAbort

_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")


class HaltPreservingExitStack(ExitStack):
    """Attempt every registered callback without replacing an active halt.

    Callback failures during SafetyAbort, EmbodimentFault or KeyboardInterrupt
    are secondary and reported as RuntimeWarnings. Otherwise callbacks retain
    ExitStack's ordinary exception behavior, including cleanup-originated halts.
    """

    def callback(
        self, callback: Callable[_P, _R], /, *args: _P.args, **kwds: _P.kwargs
    ) -> Callable[_P, _R]:
        """Register cleanup whose failure cannot mask an active stop signal."""

        def exit_callback(
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            try:
                callback(*args, **kwds)
            except BaseException as cleanup_error:
                if not isinstance(exc, (SafetyAbort, EmbodimentFault, KeyboardInterrupt)):
                    raise
                # Warning filters and custom reporters can also raise. Neither
                # may replace the halt or prevent the remaining cleanup.
                with suppress(BaseException):
                    warnings.warn(
                        f"cleanup failed while preserving {type(exc).__name__}: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return False

        self.push(exit_callback)
        return callback


# --- Argument validation helpers ---


def validate_max_workers(max_workers: int) -> None:
    """Reject invalid limits before constructing task resources."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ConfigError("max_workers must be a positive integer")


# --- End argument validation helpers ---


def run_parallel(tasks: Sequence[_T], run: Callable[[_T], _R], max_workers: int) -> list[_R]:
    """Keep at most max_workers tasks in flight and return results in input order.

    A raised exception stops admission. Active tasks finish and release their
    resources before the exception propagates, including on KeyboardInterrupt.
    Threads cannot forcibly interrupt an active backend call.
    """
    results: dict[int, _R] = {}
    remaining = iter(enumerate(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(run, task): index for index, task in islice(remaining, max_workers)
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Observe every completed exception before admitting another task.
            for future in done:
                results[pending.pop(future)] = future.result()
            for _ in done:
                item = next(remaining, None)
                if item is not None:
                    index, task = item
                    pending[executor.submit(run, task)] = index
    return [results[index] for index in range(len(tasks))]
