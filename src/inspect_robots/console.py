"""Threadless, non-blocking operator feedback and episode-end input.

The console owns no background reader. Callers poll it from the rollout loop,
which leaves stdin available for the ordinary post-trial verdict prompt.
"""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

USAGE = (
    "operator console: type a message + Enter to send it to the policy; Enter alone ends "
    "the episode; /y /n /p [note] ends it with a verdict"
)
USAGE_END_ONLY = (
    "operator console: Enter ends the episode; /y /n /p [note] records a verdict; "
    "typed notes are saved to the log"
)


@dataclass(frozen=True)
class EndRequest:
    """A request to stop the trial, optionally carrying its human verdict and note."""

    verdict: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class ConsolePoll:
    """All feedback and the first end request drained in one non-blocking poll.

    ``sources``, when non-empty, is parallel to ``messages``; an empty tuple means every
    message came from the console.
    """

    messages: tuple[str, ...] = ()
    end: EndRequest | None = None
    sources: tuple[str, ...] = ()


@runtime_checkable
class OperatorInput(Protocol):
    """Provide trial-scoped operator feedback without blocking the rollout loop."""

    def poll(self) -> ConsolePoll:
        """Return complete input accumulated since the previous poll without blocking."""
        ...

    def begin_trial(self) -> None:
        """Discard input that arrived outside the trial that is about to start."""
        ...


def _stdin_readable() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])  # pragma: no cover


def _stdin_read() -> str:
    return os.read(sys.stdin.fileno(), 65536).decode("utf-8", errors="replace")  # pragma: no cover


def _parse(line: str) -> tuple[str | None, EndRequest | None, bool]:
    text = line.rstrip("\n")
    stripped = text.strip()
    if not stripped:
        return None, EndRequest(), False

    token, separator, raw_note = stripped.partition(" ")
    verdict = {"/y": "y", "/n": "n", "/p": "partial"}.get(token.lower())
    if verdict is not None:
        note = raw_note.strip() if separator else ""
        return None, EndRequest(verdict=verdict, note=note or None), False
    if stripped.startswith("/"):
        return None, None, True
    return text, None, False


class OperatorConsole:
    """Poll stdin at fd level and preserve partial text until the operator presses Enter."""

    def __init__(
        self,
        readable: Callable[[], bool] | None = None,
        read: Callable[[], str] | None = None,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._readable = readable if readable is not None else _stdin_readable
        self._read = read if read is not None else _stdin_read
        self._output_fn = output_fn
        self._buffer = ""
        self._eof = False

    def poll(self) -> ConsolePoll:
        """Drain available bytes and return parsed complete lines, never unfinished input."""
        if self._eof:
            return ConsolePoll()

        messages: list[str] = []
        end: EndRequest | None = None
        while self._readable():
            chunk = self._read()
            if chunk == "":
                self._eof = True
                self._buffer = ""
                break
            self._buffer += chunk
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                message, parsed_end, show_usage = _parse(f"{line}\n")
                if message is not None:
                    messages.append(message)
                if parsed_end is not None and end is None:
                    end = parsed_end
                if show_usage:
                    self._output_fn(USAGE)
        return ConsolePoll(messages=tuple(messages), end=end)

    def begin_trial(self) -> None:
        """Drain pending fd input and clear partial text so trials cannot leak into each other."""
        if self._eof:
            return

        self._buffer = ""
        while self._readable():
            if self._read() == "":
                self._eof = True
                return
