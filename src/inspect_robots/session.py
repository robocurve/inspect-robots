"""Own attended-run operator input, prompts, and terminal rendering."""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from inspect_robots.console import ConsolePoll, OperatorConsole
from inspect_robots.errors import EmbodimentFault
from inspect_robots.types import OPERATOR_END

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene


_PROMPT = "did the robot succeed? [y/n/partial/skip] (partial scores as failure) "
# "Enter for none", not "Enter to skip": `skip` is a literal verdict token one
# prompt earlier, and typing it here would record the word, not skip anything.
_NOTES_PROMPT = "grader notes (Enter for none): "
_PROMPT_ANSWERS = frozenset({"y", "yes", "n", "no", "partial", "skip"})
_DEFINITIVE_REASONS = frozenset({"success", "failure"})


def _flush_stdin_fd() -> None:
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()  # pragma: no cover
    while select.select([fd], [], [], 0)[0]:  # pragma: no cover
        if not os.read(fd, 65536):  # pragma: no cover
            break  # pragma: no cover


class OperatorSession:
    """Coordinate all attended-run terminal input and output through injectable seams."""

    def __init__(
        self,
        *,
        console: OperatorConsole | None = None,
        input_fn: Callable[[str], str] | None = None,
        write: Callable[[str], None] | None = None,
        flush_fn: Callable[[], None] | None = None,
    ) -> None:
        self._input_fn = input_fn
        self._write_fn = write
        self._flush_fn = flush_fn if flush_fn is not None else _flush_stdin_fd
        self._status_open = False
        self._status_line: str | None = None
        self._console = (
            console if console is not None else OperatorConsole(output_fn=self.write_line)
        )

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

    def poll(self) -> ConsolePoll:
        """Return the composed console's next non-blocking input batch."""
        return self._console.poll()

    def begin_trial(self) -> None:
        """Ask the composed console to discard input predating the next trial."""
        self._console.begin_trial()

    def status(self, line: str | None) -> None:
        """Render one in-place status line, or idempotently close an open line."""
        if line is None:
            if self._status_open:
                self._write("\n")
                self._status_open = False
            return

        self._status_line = line
        self._write(f"\r  {line}   ")
        self._status_open = True

    def write_line(self, text: str) -> None:
        """Write scrollback text without losing an open status line."""
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
        """Capture or adopt the terminal operator's verdict on the trial record (R6)."""
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

    def prompt_verdict_on_operator_end(self, record: TrialRecord, scene: Scene) -> None:
        """Prompt or announce only for a trial the operator demonstrably ended (R6-safe)."""
        if record.termination_reason == OPERATOR_END and record.operator_judgement is None:
            self.prompt_verdict(record, scene)
        elif record.termination_reason == OPERATOR_END:
            self.write_line(f"operator verdict adopted from console: {record.operator_judgement}")
