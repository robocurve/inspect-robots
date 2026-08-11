"""Own attended-run operator input, prompts, terminal rendering, and the echo pump.

Footer windows may run a per-trial background echo pump thread (plan 0066) so typing
stays visible while the rollout thread is blocked in policy inference; the thread is
joined in ``end_trial()`` before any blocking stdin read can run.
"""

from __future__ import annotations

import atexit
import os
import select
import shutil
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from inspect_robots.console import (
    END_SENTINEL,
    USAGE,
    ConsolePoll,
    OperatorConsole,
    OperatorInput,
    _stdin_readable,
)
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
_END_HINT = "Esc ends the episode"
_END_HINT_PHRASE = "ends the episode"
# How long a trailing ESC must stay unanswered before a quiet pump reads it as a bare
# Esc keypress (vim-style escape timeout). With an echo pump running, the grace is
# tested at ECHO_INTERVAL_S cadence rather than only at quiet rollout polls, so it now
# applies mid-inference too; the accepted cost is unchanged from plan 0051: a tail
# delayed past this floor misreads as bare Esc and then types literally into the buffer.
_ESC_GRACE_S = 0.15
# Background echo pump cadence. Echo latency while the policy blocks in inference is
# bounded by this interval; the Esc grace settles within _ESC_GRACE_S + this interval.
ECHO_INTERVAL_S = 0.05


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

    Escape sequences (CSI ``ESC [ ... final``, SS3 ``ESC O final``, and ``ESC`` plus one
    other byte as an alt-combo) are discarded whole so arrow keys, Delete, and function
    keys never leak into the buffer or its echo (decision 4). The discard state persists
    across ``feed()`` calls so a sequence split across reads, or across polls, is still
    swallowed cleanly when its tail arrives.

    A pending ESC resolved by a second ESC or by Enter is a bare Esc keypress: ``feed()``
    emits the ``END_SENTINEL`` line (the end-of-episode request) and, for Enter, consumes
    the newline. The editor buffer is never touched by sentinel emission: a partial line
    typed before the Esc stays a partial and is abandoned when the trial ends. A trailing
    pending ESC with no resolving byte is settled by the session's pump grace
    (``_ESC_GRACE_S``). Accepted costs: Alt+Enter (``ESC \\r``) ends the episode, and a
    printable key pressed while an Esc is pending still resolves as an alt-combo,
    cancelling the intended end and eating the keystroke.
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
                if b == 0x1B:  # a second ESC: the pending one was a bare keypress
                    completed.append(END_SENTINEL)
                    continue
                self._escape_pending = False
                if b in (0x5B, 0x4F):  # '[' (CSI) or 'O' (SS3)
                    self._csi_discard = True
                elif b in (0x0A, 0x0D):  # Esc then Enter: bare keypress, newline consumed
                    completed.append(END_SENTINEL)
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

    @property
    def escape_pending(self) -> bool:
        """True while a lone ESC awaits its resolving byte (sequence, combo, or bare press)."""
        return self._escape_pending

    def take_bare_escape(self) -> bool:
        """Consume a pending ESC as a bare Esc keypress, reporting whether one was pending."""
        was_pending = self._escape_pending
        self._escape_pending = False
        return was_pending


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
        now_fn: Callable[[], float] | None = None,
        console_usage: str | None = None,
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
        self._now_fn: Callable[[], float] = now_fn if now_fn is not None else time.monotonic
        self._editor = _LineEditor()
        self._line_queue: list[str] = []
        self._pump_eof = False
        self._echo_interval_s: float | None = None
        self._lock = threading.RLock()
        self._pump_thread: threading.Thread | None = None
        self._pump_stop: threading.Event | None = None
        self._pump_error: Exception | None = None
        # Monotonic stamp of the byte-feeding pump that left an ESC pending; None means
        # no bare-Esc grace is armed.
        self._esc_stamp: float | None = None
        self._console = (
            console
            if console is not None
            else OperatorConsole(
                readable=self._dispatch_readable,
                read=self._dispatch_read,
                output_fn=self.write_line,
                usage=console_usage if console_usage is not None else USAGE,
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
        """Drain readable stdin into the line editor, echoing the input row after each chunk.

        Also settles a trailing pending ESC. A pump that fed bytes never runs the grace
        test (feed resolution always wins); if it ends with an ESC pending it arms the
        grace by stamping ``now_fn()``. Only a quiet pump (zero bytes) tests the grace,
        and it never re-stamps: once ``_ESC_GRACE_S`` has elapsed the pending ESC is
        consumed as a bare Esc keypress and the ``END_SENTINEL`` line is queued. EOF
        never fires the grace (the console latches EOF instead).
        """
        fed = False
        while self._fd_readable():
            chunk = self._fd_read()
            if chunk == b"":
                self._pump_eof = True
                break
            fed = True
            self._line_queue.extend(self._editor.feed(chunk))
            self._draw_input_row()
        if fed:
            self._esc_stamp = self._now_fn() if self._editor.escape_pending else None
            return
        if self._pump_eof or self._esc_stamp is None:
            return
        if self._now_fn() - self._esc_stamp < _ESC_GRACE_S:
            return
        self._esc_stamp = None
        self._editor.take_bare_escape()
        self._line_queue.append(END_SENTINEL)

    def _clipped_input_text(self) -> str:
        return _clip_tail(self._editor.text(), self._width_fn() - 4)

    def _status_texts(self) -> tuple[str, str]:
        """Return plugin text without a trailing gesture clause and with the owned hint.

        Accepted costs: a ``" | "`` segment containing the phrase without being
        gesture prose is dropped; a gesture mention mid-line, after a non-pipe
        separator, or as a separator-less whole-line status renders a duplicated or
        contradictory hint.
        """
        assert self._status_line is not None
        line = self._status_line
        head, sep, tail = line.rpartition(" | ")
        if sep and _END_HINT_PHRASE in tail:
            line = head
        if not line:
            return "", _END_HINT
        return line, f"{line} | {_END_HINT}"

    def _clipped_status_text(self) -> str:
        width = self._width_fn() - 1
        stripped, composed = self._status_texts()
        return composed if len(composed) <= width else _clip_tail(stripped, width)

    def _draw_input_row(self) -> None:
        self._write(f"\r\x1b[K> {self._clipped_input_text()}")

    def _enter_footer(self) -> None:
        """Enter the footer window for a new trial (decisions 5 and 7, plan 0066).

        Drains stale fd bytes with no echo, resets the editor and queue, closes a
        stale plain-open status line, draws the (now empty) input row, and finally
        starts the window's echo pump thread when an interval was configured.
        """
        self._footer_active = True
        while self._fd_readable():
            if self._fd_read() == b"":
                self._pump_eof = True
                break
        self._editor.reset()
        self._line_queue = []
        self._esc_stamp = None
        prefix = ""
        if self._status_open:
            prefix = "\n"
            self._status_open = False
        self._write(f"{prefix}\r\x1b[K> {self._clipped_input_text()}")
        if self._echo_interval_s is not None:
            stop = threading.Event()
            thread = threading.Thread(
                target=self._echo_pump_loop,
                args=(stop, self._echo_interval_s),
                name="operator-echo-pump",
                daemon=True,
            )
            # Assign only after a successful start: join() on a never-started thread
            # raises RuntimeError, and _try_enter_footer's abort path must leave no
            # handle a later end_trial's _stop_echo_pump would trip over.
            thread.start()
            self._pump_stop = stop
            self._pump_thread = thread

    def _echo_pump_loop(self, stop: threading.Event, interval: float) -> None:
        """Echo typed bytes between rollout polls while the policy blocks (plan 0066).

        Runs the same pump ``poll()`` runs, under the shared lock, every ``interval``
        seconds until ``stop`` is set. A raising pump is stored for the next ``poll()``
        to re-raise (rollout's existing degrade path) and ends this thread; the thread
        itself never prints or raises.
        """
        while not stop.wait(interval):
            with self._lock:
                try:
                    self._pump_input()
                except Exception as exc:
                    self._pump_error = exc
                    return

    def _stop_echo_pump(self) -> None:
        """Idempotently stop and join the echo pump thread for this footer window.

        The join is unbounded: a pump blocked writing to an XOFF-wedged terminal
        blocks it, exactly as the same write blocks today's synchronous pump on the
        rollout thread. Not a regression; accepted.
        """
        if self._pump_thread is None:
            return
        assert self._pump_stop is not None
        self._pump_stop.set()
        self._pump_thread.join()
        self._pump_thread = None
        self._pump_stop = None

    def _try_enter_footer(self) -> None:
        """Re-evaluate the per-trial gate ladder and silently retain plain mode on failure."""
        if self._footer_active:
            return
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
        self._esc_stamp = None
        # Cleared per closed window (decision 5, plan 0066): a stale error from trial N
        # must not kill trial N+1's plain-mode console when its footer entry fails the
        # gate ladder. Safe because poll() consumes the error before raising.
        self._pump_error = None

    def _restore_terminal(self) -> None:
        """Idempotently restore and forget the active termios snapshot.

        As the atexit crash net, also stop a still-live echo pump without taking the
        lock or joining: a pump blocked on a wedged stdout write must not deadlock
        interpreter shutdown, so one echo already in flight may land after the
        restore (accepted cost, plan 0066).
        """
        if self._saved_termios_state is _NO_TERMIOS_STATE:
            return
        stop = self._pump_stop
        if stop is not None:
            stop.set()
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

    def enable_footer(self, *, label: str, echo_interval_s: float | None = None) -> None:
        """Opt into the two-row footer renderer with its feedback confirmation label.

        ``echo_interval_s`` additionally opts each footer window into a background
        echo pump thread at that cadence, so typing echoes while the rollout thread
        is blocked in policy inference (plan 0066); ``None`` keeps the footer fully
        synchronous, which every embedded or injected-seam use wants, and the CLI
        passes ``ECHO_INTERVAL_S``. While a footer window is open (between
        ``begin_trial()`` and ``end_trial()``) the session owns stdin; blocking
        stdin reads are only legal outside it.

        A documented no-op when this session was built with a caller-injected
        ``console=...``: the footer requires the session-built console whose seams
        are the decision-7 dispatching closures wired in ``__init__``. The first
        enabling call also registers this session's idempotent terminal-restore
        with ``atexit`` as a crash net.
        """
        if self._owns_console:
            self._footer_requested = True
            self._footer_label = label
            self._echo_interval_s = echo_interval_s
            if not self._atexit_registered:
                self._atexit_register(self._restore_terminal)
                self._atexit_registered = True

    def end_trial(self) -> None:
        """Idempotently clear footer rows and restore the trial's exact terminal attributes.

        Duck-typed hook (decision 1): ``rollout()`` may call this best-effort
        mid-trial, including when ``begin_trial()`` itself raised, and again in the
        per-trial ``finally``. Begin-before-end pairing is not guaranteed, so this
        hook must remain idempotent. A session that did not enter footer mode writes
        no bytes and leaves plain-renderer state untouched. The echo pump is stopped
        and joined first — before the clearing writes and the termios restore — so a
        pump byte can never land after either (plan 0066).
        """
        try:
            self._stop_echo_pump()
        except BaseException:
            # A KeyboardInterrupt landing inside the join must not skip the terminal
            # restore that end_trial guaranteed before the pump existed. The stop
            # event is already set, so the pump dies within one interval; the rows
            # are abandoned unclear rather than risking a write race.
            self._reset_footer_state()
            self._restore_terminal()
            raise
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
        """Poll the console first, then merge every healthy attached input in order.

        A stored echo-pump-thread error is re-raised here first (plan 0066), reusing
        rollout's degrade-and-warn path for a raising ``poll()``. The lock is released
        before the attached-input merge: attached sources are rollout-thread state,
        and ``write_line`` re-acquires the lock itself.
        """
        with self._lock:
            if self._pump_error is not None:
                error = self._pump_error
                self._pump_error = None
                raise error
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
        # A message in a poll that also carries an end request is recorded to the log
        # but never delivered to the policy (rollout breaks before the next inference),
        # so "[sent]" would be a false delivery claim; confirm it as noted instead.
        label = "noted" if poll.end is not None else self._footer_label
        for index, message in enumerate(poll.messages):
            if not poll.sources or poll.sources[index] == "console":
                self.write_line(f"[{label}] {message}")

    def begin_trial(self) -> None:
        """Re-decide footer eligibility, then discard feedback predating the next trial."""
        self._try_enter_footer()
        with self._lock:
            # The window's echo pump may already be running; the console drain reads
            # the shared line queue through the dispatch seams.
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
            with self._lock:
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
            with self._lock:
                self._footer_write_line(text)
            return
        if self._status_open:
            self._write("\n")
        self._write(f"{text}\n")
        if self._status_open:
            assert self._status_line is not None
            self._write(f"\r  {self._status_line}   ")

    def gate(self, prompt: str, *, hint: str | None = None) -> None:
        """Close sticky status, flush stale input, and block for readiness.

        Must not be called while a footer window is open: the window owns stdin
        (and may be running its echo pump), so blocking reads are only legal
        outside ``begin_trial()``/``end_trial()`` (plan 0066).
        """
        self.status(None)
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

        Any sticky status is closed before output or prompting, and stale stdin is
        drained best-effort before a verdict answer is read.
        A verdict already captured by the console is announced and preserved.
        A terminated episode with a definitive embodiment verdict adopts and announces
        that verdict instead of asking the operator to confirm the same outcome a second
        time. Prompted verdicts are followed by one optional, stripped, case-preserved
        grader note.
        """
        self.status(None)
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
        with suppress(OSError):
            self._flush_fn()
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
