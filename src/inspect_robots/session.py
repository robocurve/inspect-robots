"""Own attended-run operator input, prompts, and terminal rendering."""

from __future__ import annotations

import atexit
import os
import select
import shutil
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from inspect_robots.console import ConsolePoll, OperatorConsole, OperatorInput, _stdin_readable
from inspect_robots.errors import EmbodimentFault

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene


_PROMPT = "did the robot succeed? [y/n/partial/skip] (partial scores as failure) "
# "Enter for none", not "Enter to skip": `skip` is a literal verdict token one
# prompt earlier, and typing it here would record the word, not skip anything.
_NOTES_PROMPT = "grader notes (Enter for none): "
_PROMPT_ANSWERS = frozenset({"y", "yes", "n", "no", "partial", "skip"})
_DEFINITIVE_REASONS = frozenset({"success", "failure"})
_NO_TERMIOS_STATE = object()


def _flush_stdin_fd() -> None:
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()  # pragma: no cover
    while select.select([fd], [], [], 0)[0]:  # pragma: no cover
        if not os.read(fd, 65536):  # pragma: no cover
            break  # pragma: no cover


def _stdin_read_bytes() -> bytes:
    """Read up to 64KiB of raw stdin without decoding (the footer pump's default fd seam)."""
    return os.read(sys.stdin.fileno(), 65536)  # pragma: no cover


def _enter_cbreak() -> object:
    """Enter stdin cbreak mode without echo and return the exact attributes to restore."""
    import termios  # pragma: no cover

    fd = sys.stdin.fileno()  # pragma: no cover
    saved = termios.tcgetattr(fd)  # pragma: no cover
    current = saved.copy()  # pragma: no cover
    current[6] = saved[6].copy()  # pragma: no cover
    current[3] &= ~(termios.ECHO | termios.ICANON)  # pragma: no cover
    current[6][termios.VMIN] = 1  # pragma: no cover
    current[6][termios.VTIME] = 0  # pragma: no cover
    termios.tcsetattr(fd, termios.TCSANOW, current)  # pragma: no cover
    return (fd, saved)  # pragma: no cover


def _restore(state: object) -> None:
    """Restore one exact termios snapshot returned by ``_enter_cbreak``."""
    import termios  # pragma: no cover

    fd, attrs = cast(tuple[int, list[Any]], state)  # pragma: no cover
    termios.tcsetattr(fd, termios.TCSANOW, attrs)  # pragma: no cover


def _default_width() -> int:
    """Return the current terminal width in columns, re-read fresh on every call."""
    return shutil.get_terminal_size().columns


def _clip_tail(text: str, limit: int) -> str:
    """Return ``text`` unchanged if it fits ``limit`` characters, else just its trailing tail."""
    return text if len(text) <= limit else text[-limit:]


class _LineEditor:
    """Assemble raw stdin bytes into one editable line: append, backspace, Ctrl-U, Enter.

    Escape sequences (CSI ``ESC [ ... final``, SS3 ``ESC O final``, and a bare ``ESC``
    plus one byte) are discarded whole so arrow keys, Delete, and function keys never
    leak into the buffer or its echo (decision 4). The discard state persists across
    ``feed()`` calls so a sequence split across reads, or across polls, is still
    swallowed cleanly.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._escape_pending = False
        self._csi_discard = False

    def reset(self) -> None:
        """Clear the in-progress line and any mid-flight escape-discard state."""
        self._buffer.clear()
        self._escape_pending = False
        self._csi_discard = False

    def text(self) -> str:
        """Decode the unfinished line for display, replacing any not-yet-complete UTF-8 tail."""
        return self._buffer.decode("utf-8", errors="replace")

    def feed(self, chunk: bytes) -> list[str]:
        """Apply one raw read; return each line ``chunk`` completed, decoded once, in order."""
        completed: list[str] = []
        for b in chunk:
            if self._csi_discard:
                if 0x40 <= b <= 0x7E:
                    self._csi_discard = False
                continue
            if self._escape_pending:
                self._escape_pending = False
                if b in (0x5B, 0x4F):  # '[' (CSI) or 'O' (SS3)
                    self._csi_discard = True
                continue
            if b == 0x1B:  # ESC
                self._escape_pending = True
            elif b in (0x7F, 0x08):  # backspace / DEL
                self._delete_char()
            elif b == 0x15:  # Ctrl-U
                self._buffer.clear()
            elif b in (0x0A, 0x0D):  # \n or \r completes the line
                completed.append(self._buffer.decode("utf-8", errors="replace"))
                self._buffer.clear()
            elif b >= 0x20 and b != 0x7F:
                self._buffer.append(b)
            # else: other non-printable control bytes are ignored
        return completed

    def _delete_char(self) -> None:
        while self._buffer and 0x80 <= self._buffer[-1] <= 0xBF:
            self._buffer.pop()
        if self._buffer:
            self._buffer.pop()


class OperatorSession:
    """Coordinate all attended-run terminal input and output through injectable seams."""

    def __init__(
        self,
        *,
        console: OperatorConsole | None = None,
        input_fn: Callable[[str], str] | None = None,
        write: Callable[[str], None] | None = None,
        flush_fn: Callable[[], None] | None = None,
        fd_readable: Callable[[], bool] | None = None,
        fd_read: Callable[[], bytes] | None = None,
        width_fn: Callable[[], int] | None = None,
        isatty_fn: Callable[[], bool] | None = None,
        raw_mode_fn: Callable[[], object] | None = None,
        restore_fn: Callable[[object], None] | None = None,
        atexit_register: Callable[[Callable[[], None]], object] | None = None,
    ) -> None:
        self._input_fn = input_fn
        self._write_fn = write
        self._flush_fn = flush_fn if flush_fn is not None else _flush_stdin_fd
        self._status_open = False
        self._status_line: str | None = None
        self._owns_console = console is None
        self._footer_requested = False
        self._footer_active = False
        self._footer_label: str | None = None
        self._isatty_fn: Callable[[], bool] = (
            isatty_fn if isatty_fn is not None else sys.stdin.isatty
        )
        self._raw_mode_fn: Callable[[], object] = (
            raw_mode_fn if raw_mode_fn is not None else _enter_cbreak
        )
        self._restore_fn: Callable[[object], None] = (
            restore_fn if restore_fn is not None else _restore
        )
        self._atexit_register: Callable[[Callable[[], None]], object] = (
            atexit_register if atexit_register is not None else atexit.register
        )
        self._atexit_registered = False
        self._saved_termios_state: object = _NO_TERMIOS_STATE
        self._fd_readable: Callable[[], bool] = (
            fd_readable if fd_readable is not None else _stdin_readable
        )
        self._fd_read: Callable[[], bytes] = fd_read if fd_read is not None else _stdin_read_bytes
        self._width_fn: Callable[[], int] = width_fn if width_fn is not None else _default_width
        self._editor = _LineEditor()
        self._line_queue: list[str] = []
        self._pump_eof = False
        self._console = (
            console
            if console is not None
            else OperatorConsole(
                readable=self._dispatch_readable,
                read=self._dispatch_read,
                output_fn=self.write_line,
            )
        )
        self._attached_inputs: list[tuple[OperatorInput, str]] = []

    def _write(self, text: str) -> None:
        if self._write_fn is not None:
            self._write_fn(text)
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    def _input(self, prompt: str) -> str:
        if self._input_fn is not None:
            return self._input_fn(prompt)
        return input(prompt)

    def _dispatch_readable(self) -> bool:
        """Console-facing ``readable`` seam: footer queue/EOF state, else the raw fd seam."""
        if self._footer_active:
            return bool(self._line_queue) or self._pump_eof
        return self._fd_readable()

    def _dispatch_read(self) -> str:
        """Console-facing ``read`` seam: queued footer lines, or one decoded fd chunk."""
        if self._footer_active:
            if self._line_queue:
                chunk = "".join(f"{line}\n" for line in self._line_queue)
                self._line_queue = []
                return chunk
            return ""
        raw = self._fd_read()
        return "" if raw == b"" else raw.decode("utf-8", errors="replace")

    def _pump_input(self) -> None:
        """Drain readable stdin into the line editor, echoing the input row after each chunk."""
        while self._fd_readable():
            chunk = self._fd_read()
            if chunk == b"":
                self._pump_eof = True
                break
            self._line_queue.extend(self._editor.feed(chunk))
            self._draw_input_row()

    def _clipped_input_text(self) -> str:
        return _clip_tail(self._editor.text(), self._width_fn() - 4)

    def _clipped_status_text(self) -> str:
        assert self._status_line is not None
        return _clip_tail(self._status_line, self._width_fn() - 1)

    def _draw_input_row(self) -> None:
        self._write(f"\r\x1b[K> {self._clipped_input_text()}")

    def _enter_footer(self) -> None:
        """Enter the footer window for a new trial (decisions 5 and 7).

        Drains stale fd bytes with no echo, resets the editor and queue, closes a
        stale plain-open status line, then draws the (now empty) input row.
        """
        self._footer_active = True
        while self._fd_readable():
            if self._fd_read() == b"":
                self._pump_eof = True
                break
        self._editor.reset()
        self._line_queue = []
        prefix = ""
        if self._status_open:
            prefix = "\n"
            self._status_open = False
        self._write(f"{prefix}\r\x1b[K> {self._clipped_input_text()}")

    def _try_enter_footer(self) -> None:
        """Re-evaluate the per-trial gate ladder and silently retain plain mode on failure."""
        if not self._footer_requested or not self._owns_console:
            return
        try:
            if not self._isatty_fn():
                return
            saved_state = self._raw_mode_fn()
        except Exception:
            return

        self._saved_termios_state = saved_state
        try:
            self._enter_footer()
        except BaseException:
            self._reset_footer_state()
            self._restore_terminal()
            raise

    def _reset_footer_state(self) -> None:
        """Discard all renderer and editor state after a footer window closes or aborts."""
        self._footer_active = False
        self._status_open = False
        self._editor.reset()
        self._line_queue = []
        self._pump_eof = False

    def _restore_terminal(self) -> None:
        """Idempotently restore and forget the active termios snapshot."""
        if self._saved_termios_state is _NO_TERMIOS_STATE:
            return
        saved_state = self._saved_termios_state
        self._saved_termios_state = _NO_TERMIOS_STATE
        self._restore_fn(saved_state)

    def _footer_status(self, line: str | None) -> None:
        if line is None:
            if self._status_open:
                self._write(f"\r\x1b[K\x1b[A\r\x1b[K> {self._clipped_input_text()}")
                self._status_open = False
            return

        self._status_line = line
        if self._status_open:
            self._write(
                f"\x1b[A\r\x1b[K{self._clipped_status_text()}"
                f"\x1b[B\r\x1b[K> {self._clipped_input_text()}"
            )
        else:
            self._write(
                f"\r\x1b[K{self._clipped_status_text()}\n\r\x1b[K> {self._clipped_input_text()}"
            )
            self._status_open = True

    def _footer_write_line(self, text: str) -> None:
        if self._status_open:
            self._write(
                f"\r\x1b[K\x1b[A\r\x1b[K{text}\n"
                f"{self._clipped_status_text()}\n"
                f"\r\x1b[K> {self._clipped_input_text()}"
            )
        else:
            self._write(f"\r\x1b[K{text}\n\r\x1b[K> {self._clipped_input_text()}")

    def enable_footer(self, *, label: str) -> None:
        """Opt into the two-row footer renderer with its feedback confirmation label.

        A documented no-op when this session was built with a caller-injected
        ``console=...``: the footer requires the session-built console whose seams
        are the decision-7 dispatching closures wired in ``__init__``.
        """
        if self._owns_console:
            self._footer_requested = True
            self._footer_label = label
            if not self._atexit_registered:
                self._atexit_register(self._restore_terminal)
                self._atexit_registered = True

    def end_trial(self) -> None:
        """Idempotently clear footer rows and restore the trial's exact terminal attributes.

        Duck-typed hook (decision 1): ``rollout()`` calls this best-effort in the
        per-trial ``finally``. A session that did not enter footer mode writes no
        bytes and leaves plain-renderer state untouched.
        """
        if not self._footer_active:
            return
        try:
            if self._status_open:
                self._write("\r\x1b[K\x1b[A\r\x1b[K")
            else:
                self._write("\r\x1b[K")
        finally:
            self._reset_footer_state()
            self._restore_terminal()

    def poll(self) -> ConsolePoll:
        """Poll the console first, then merge every healthy attached input in order."""
        if self._footer_active:
            self._pump_input()
        console_poll = self._console.poll()
        if not self._attached_inputs:
            self._confirm_console_messages(console_poll)
            return console_poll

        messages = list(console_poll.messages)
        sources = ["console"] * len(messages)
        healthy_inputs: list[tuple[OperatorInput, str]] = []
        for source, label in self._attached_inputs:
            try:
                attached_poll = source.poll()
            except Exception as exc:
                self.write_line(f"{label} input disabled after {type(exc).__name__}: {exc}")
                continue
            healthy_inputs.append((source, label))
            for text in attached_poll.messages:
                self.write_line(f"{label}: {text}")
                messages.append(text)
                sources.append(label)
        self._attached_inputs = healthy_inputs
        poll = ConsolePoll(messages=tuple(messages), end=console_poll.end, sources=tuple(sources))
        self._confirm_console_messages(poll)
        return poll

    def _confirm_console_messages(self, poll: ConsolePoll) -> None:
        if not self._footer_active:
            return
        assert self._footer_label is not None
        for index, message in enumerate(poll.messages):
            if not poll.sources or poll.sources[index] == "console":
                self.write_line(f"[{self._footer_label}] {message}")

    def begin_trial(self) -> None:
        """Re-decide footer eligibility, then discard feedback predating the next trial."""
        self._try_enter_footer()
        self._console.begin_trial()
        healthy_inputs: list[tuple[OperatorInput, str]] = []
        for source, label in self._attached_inputs:
            try:
                source.begin_trial()
            except Exception as exc:
                self.write_line(f"{label} input disabled after {type(exc).__name__}: {exc}")
                continue
            healthy_inputs.append((source, label))
        self._attached_inputs = healthy_inputs

    def attach_input(self, source: OperatorInput, *, label: str) -> None:
        """Attach a feedback-only input and stamp its messages with ``label``.

        The session ignores the source's own provenance and any end request. A source that
        raises while polling or beginning a trial is permanently detached for this run.
        """
        self._attached_inputs.append((source, label))

    def status(self, line: str | None) -> None:
        """Render one in-place status line, or idempotently close an open line.

        In footer mode this drives the two-row/one-row renderer from decision 5
        instead of the plain single-line ticker.
        """
        if self._footer_active:
            self._footer_status(line)
            return
        if line is None:
            if self._status_open:
                self._write("\n")
                self._status_open = False
            return

        self._status_line = line
        self._write(f"\r  {line}   ")
        self._status_open = True

    def write_line(self, text: str) -> None:
        """Write scrollback text without losing an open status line (or the footer's input row)."""
        if self._footer_active:
            self._footer_write_line(text)
            return
        if self._status_open:
            self._write("\n")
        self._write(f"{text}\n")
        if self._status_open:
            assert self._status_line is not None
            self._write(f"\r  {self._status_line}   ")

    def gate(self, prompt: str, *, hint: str | None = None) -> None:
        """Block for readiness after flushing stale input, or fault when stdin is dead."""
        try:
            self._flush_fn()
            self._input(prompt)
        except (EOFError, OSError) as exc:
            message = (
                "Operator readiness gate could not read standard input. Run with a real TTY, "
                "provide injectable input, or use unattended mode."
            )
            if hint is not None:
                message = f"{message} {hint}"
            raise EmbodimentFault(message) from exc

    def prompt_verdict(self, record: TrialRecord, scene: Scene) -> None:
        """Capture or adopt the terminal operator's verdict on the trial record (R6).

        A verdict already captured by the console is announced and preserved.
        A terminated episode with a definitive embodiment verdict adopts and announces
        that verdict instead of asking the operator to confirm the same outcome a second
        time. Prompted verdicts are followed by one optional, stripped, case-preserved
        grader note.
        """
        from inspect_robots.transcript import operator_event

        del scene
        if record.operator_judgement is not None:
            self.write_line(f"operator verdict adopted from console: {record.operator_judgement}")
            return
        if record.terminated and record.termination_reason in _DEFINITIVE_REASONS:
            verdict = "y" if record.termination_reason == "success" else "n"
            record.operator_judgement = verdict
            record.events.append(
                operator_event(t=len(record.steps), verdict=verdict, source="embodiment")
            )
            self.write_line(
                f"operator verdict adopted from embodiment: {record.termination_reason}"
            )
            return
        if record.truncated and record.termination_reason == "max_steps":
            self.write_line("note: this trial hit the step limit before terminating")
        elif record.truncated and record.termination_reason is not None:
            # Neutral on purpose: the record cannot distinguish a
            # policy-requested stop from an embodiment truncation, so the
            # reason string carries the specifics.
            self.write_line(f"note: this trial ended early ({record.termination_reason!r})")
        while True:
            try:
                answer = self._input(_PROMPT).strip().lower()
            except EOFError:
                return
            if answer in _PROMPT_ANSWERS:
                break
            self.write_line(f"unrecognized answer {answer!r}; expected one of y/n/partial/skip")
        try:
            note = self._input(_NOTES_PROMPT).strip() or None
        except EOFError:
            note = None
        record.operator_note = note
        if answer != "skip":
            record.operator_judgement = answer
        # A skipped trial with a note still gets an event: the human said something
        # about this trial, and an event stream that recorded the verdict but not
        # the sentence typed in the same breath would be lying by omission. The
        # event then carries verdict="skip" with no judgement, which is why
        # operator_event documents "skip" as "no judgement" for its consumers.
        if answer != "skip" or note is not None:
            record.events.append(operator_event(t=len(record.steps), verdict=answer, note=note))
