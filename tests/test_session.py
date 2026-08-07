"""Tests for the attended-run operator I/O owner."""

from __future__ import annotations

import builtins
import io
import os
import shutil
import sys
from collections.abc import Callable

import pytest

from inspect_robots.console import USAGE, ConsolePoll, EndRequest, OperatorConsole, OperatorInput
from inspect_robots.errors import EmbodimentFault
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import operator_scorer
from inspect_robots.session import _NOTES_PROMPT, _PROMPT, OperatorSession


class _RecordingConsole(OperatorConsole):
    def __init__(self) -> None:
        super().__init__(readable=lambda: False, read=lambda: "")
        self.poll_calls = 0
        self.begin_trial_calls = 0

    def poll(self) -> ConsolePoll:
        self.poll_calls += 1
        return ConsolePoll(messages=("recorded",))

    def begin_trial(self) -> None:
        self.begin_trial_calls += 1


class _RecordingLinesSession(OperatorSession):
    def __init__(self, input_fn: Callable[[str], str]) -> None:
        super().__init__(console=_RecordingConsole(), input_fn=input_fn)
        self.lines: list[str] = []

    def write_line(self, text: str) -> None:
        self.lines.append(text)


class _ScriptedAttachedInput:
    def __init__(
        self,
        polls: list[ConsolePoll | Exception],
        *,
        begin_error: Exception | None = None,
    ) -> None:
        self.polls = list(polls)
        self.begin_error = begin_error
        self.poll_calls = 0
        self.begin_calls = 0

    def poll(self) -> ConsolePoll:
        self.poll_calls += 1
        item = self.polls.pop(0) if self.polls else ConsolePoll()
        if isinstance(item, Exception):
            raise item
        return item

    def begin_trial(self) -> None:
        self.begin_calls += 1
        if self.begin_error is not None:
            raise self.begin_error


def _scripted_prompt_session(
    answers: list[str],
) -> tuple[OperatorSession, list[str], list[str]]:
    prompts: list[str] = []
    output: list[str] = []
    iterator = iter(answers)

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(iterator)

    return OperatorSession(input_fn=read, write=output.append), prompts, output


def test_console_protocol_methods_delegate_to_composed_console() -> None:
    console = _RecordingConsole()
    session = OperatorSession(console=console)

    poll = session.poll()
    assert poll == ConsolePoll(messages=("recorded",))
    assert poll.sources == ()
    session.begin_trial()

    assert console.poll_calls == 1
    assert console.begin_trial_calls == 1
    assert isinstance(session, OperatorInput)


def test_attached_inputs_merge_after_console_with_session_stamped_sources() -> None:
    output: list[str] = []
    session = OperatorSession(console=_RecordingConsole(), write=output.append)
    voice = _ScriptedAttachedInput(
        [
            ConsolePoll(
                messages=("first", "second"),
                end=EndRequest(verdict="y"),
                sources=("untrusted", "untrusted"),
            )
        ]
    )
    phone = _ScriptedAttachedInput([ConsolePoll(messages=("third",))])
    session.attach_input(voice, label="voice")
    session.attach_input(phone, label="phone")

    poll = session.poll()

    assert poll == ConsolePoll(
        messages=("recorded", "first", "second", "third"),
        end=None,
        sources=("console", "voice", "voice", "phone"),
    )
    assert output == ["voice: first\n", "voice: second\n", "phone: third\n"]


def test_poll_failure_permanently_detaches_only_that_source() -> None:
    output: list[str] = []
    session = OperatorSession(console=_RecordingConsole(), write=output.append)
    broken = _ScriptedAttachedInput(
        [RuntimeError("microphone failed"), ConsolePoll(messages=("must not return",))]
    )
    healthy = _ScriptedAttachedInput(
        [ConsolePoll(messages=("one",)), ConsolePoll(messages=("two",))]
    )
    session.attach_input(broken, label="voice")
    session.attach_input(healthy, label="phone")

    first = session.poll()
    second = session.poll()

    assert first.sources == ("console", "phone")
    assert second.sources == ("console", "phone")
    assert broken.poll_calls == 1
    assert healthy.poll_calls == 2
    assert output == [
        "voice input disabled after RuntimeError: microphone failed\n",
        "phone: one\n",
        "phone: two\n",
    ]


def test_begin_trial_failure_permanently_detaches_only_that_source() -> None:
    output: list[str] = []
    console = _RecordingConsole()
    session = OperatorSession(console=console, write=output.append)
    broken = _ScriptedAttachedInput(
        [ConsolePoll(messages=("must not return",))],
        begin_error=ValueError("reset failed"),
    )
    healthy = _ScriptedAttachedInput([ConsolePoll(messages=("fresh",))])
    session.attach_input(broken, label="voice")
    session.attach_input(healthy, label="phone")

    session.begin_trial()
    poll = session.poll()

    assert console.begin_trial_calls == 1
    assert broken.begin_calls == 1
    assert broken.poll_calls == 0
    assert healthy.begin_calls == 1
    assert poll.messages == ("recorded", "fresh")
    assert poll.sources == ("console", "phone")
    assert output == [
        "voice input disabled after ValueError: reset failed\n",
        "phone: fresh\n",
    ]


def test_default_console_routes_usage_through_write_seam() -> None:
    output: list[str] = []
    session = OperatorSession(write=output.append)
    chunks = ["/oops\n"]
    session._console._readable = lambda: bool(chunks)
    session._console._read = lambda: chunks.pop(0)

    assert session.poll() == ConsolePoll()
    assert output == [f"{USAGE}\n"]


def test_status_lifecycle_writes_exact_bytes_and_double_close_is_noop() -> None:
    output: list[str] = []
    session = OperatorSession(console=_RecordingConsole(), write=output.append)

    session.status("t = 3s")
    session.status("t = 4s")
    session.status(None)
    session.status(None)

    assert output == ["\r  t = 3s   ", "\r  t = 4s   ", "\n"]


def test_write_line_closes_and_repaints_open_status() -> None:
    output: list[str] = []
    session = OperatorSession(console=_RecordingConsole(), write=output.append)

    session.status("t = 3s")
    session.write_line("operator message")

    assert output == [
        "\r  t = 3s   ",
        "\n",
        "operator message\n",
        "\r  t = 3s   ",
    ]


def test_write_line_while_status_closed_writes_plain_line() -> None:
    output: list[str] = []
    session = OperatorSession(console=_RecordingConsole(), write=output.append)

    session.write_line("operator message")

    assert output == ["operator message\n"]


def test_default_write_resolves_stdout_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    session = OperatorSession(console=_RecordingConsole())
    first = io.StringIO()
    second = io.StringIO()

    monkeypatch.setattr(sys, "stdout", first)
    session.write_line("first")
    monkeypatch.setattr(sys, "stdout", second)
    session.write_line("second")

    assert first.getvalue() == "first\n"
    assert second.getvalue() == "second\n"


def test_gate_flushes_strictly_before_one_successful_input() -> None:
    order: list[str] = []

    def flush() -> None:
        order.append("flush")

    def read(prompt: str) -> str:
        order.append(f"input:{prompt}")
        return ""

    session = OperatorSession(input_fn=read, flush_fn=flush)
    gate: Callable[[str], object] = session.gate

    assert gate("Stand clear: ") is None
    assert order == ["flush", "input:Stand clear: "]


def test_gate_resolves_default_input_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    session = OperatorSession(flush_fn=lambda: None)

    def replacement(prompt: str) -> str:
        prompts.append(prompt)
        return "ready"

    monkeypatch.setattr(builtins, "input", replacement)

    session.gate("Ready? ")

    assert prompts == ["Ready? "]


def test_default_gate_flush_is_noop_when_stdin_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return "ready"

    session = OperatorSession(input_fn=read)

    session.gate("Ready? ")

    assert prompts == ["Ready? "]


@pytest.mark.parametrize("error_type", [EOFError, OSError])
@pytest.mark.parametrize("hint", [None, "Check the pendant cable."])
def test_gate_turns_dead_stdin_into_embodiment_fault(
    error_type: type[Exception], hint: str | None
) -> None:
    calls = {"flush": 0, "input": 0}

    def flush() -> None:
        calls["flush"] += 1

    def read(_prompt: str) -> str:
        calls["input"] += 1
        raise error_type("dead stdin")

    session = OperatorSession(input_fn=read, flush_fn=flush)

    with pytest.raises(EmbodimentFault) as exc_info:
        session.gate("Stand clear: ", hint=hint)

    message = str(exc_info.value)
    assert "standard input" in message
    assert "real TTY" in message
    assert "injectable input" in message
    assert "unattended mode" in message
    assert ("Check the pendant cable." in message) is (hint is not None)
    assert calls == {"flush": 1, "input": 1}


@pytest.mark.parametrize(
    ("termination_reason", "expected_verdict"),
    [("success", "y"), ("failure", "n")],
)
def test_prompt_operator_adopts_definitive_embodiment_verdict(
    termination_reason: str,
    expected_verdict: str,
) -> None:
    session = _RecordingLinesSession(lambda _prompt: pytest.fail("operator prompt must not fire"))
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=True,
        termination_reason=termination_reason,
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == expected_verdict
    assert record.operator_note is None
    assert session.lines == [f"operator verdict adopted from embodiment: {termination_reason}"]
    (event,) = record.events
    assert event.kind == "operator"
    assert event.t == 0
    assert event.data == {
        "verdict": expected_verdict,
        "source": "embodiment",
        "note": None,
    }


def test_prompt_operator_early_returns_for_pre_set_console_verdict_via_write_line() -> None:
    session = _RecordingLinesSession(lambda _prompt: pytest.fail("pre-set verdict must not prompt"))
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=True,
        termination_reason="success",
        operator_judgement="n",
        operator_note="console note",
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "n"
    assert record.operator_note == "console note"
    assert record.events == []
    assert session.lines == ["operator verdict adopted from console: n"]


@pytest.mark.parametrize(
    ("terminated", "termination_reason"),
    [
        (False, None),
        (False, "max_steps"),
        (False, "policy_stop"),
        (True, None),
    ],
)
def test_prompt_operator_still_prompts_without_definitive_verdict(
    terminated: bool,
    termination_reason: str | None,
) -> None:
    session, prompts, _output = _scripted_prompt_session(["y", ""])
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=terminated,
        termination_reason=termination_reason,
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert prompts == [_PROMPT, _NOTES_PROMPT]
    assert record.operator_judgement == "y"
    (event,) = record.events
    assert event.data == {"verdict": "y", "source": "prompt", "note": None}


def test_prompt_operator_prompts_for_truncated_success_reason() -> None:
    session, _prompts, _output = _scripted_prompt_session(["n", ""])
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=False,
        truncated=True,
        termination_reason="success",
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "n"
    (event,) = record.events
    assert event.data == {"verdict": "n", "source": "prompt", "note": None}


def test_prompt_operator_warns_before_judging_step_limited_trial() -> None:
    session, _prompts, output = _scripted_prompt_session(["n", ""])
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        truncated=True,
        termination_reason="max_steps",
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert output == ["note: this trial hit the step limit before terminating\n"]
    assert record.operator_judgement == "n"


@pytest.mark.parametrize(
    ("termination_reason", "expected_score"),
    [("success", True), ("failure", False)],
)
def test_operator_scorer_reads_adopted_embodiment_verdict(
    termination_reason: str,
    expected_score: bool,
) -> None:
    session = OperatorSession(
        input_fn=lambda _prompt: pytest.fail("operator prompt must not fire"),
        write=lambda _text: None,
    )
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=True,
        termination_reason=termination_reason,
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert operator_scorer()(record, None).value is expected_score


def test_prompt_operator_unit_semantics() -> None:
    scene = Scene(id="s0", instruction="reach")

    def record_with_steps() -> TrialRecord:
        record = TrialRecord(scene_id="s0", epoch=0, seed=0)
        record.steps = [None, None, None]  # type: ignore[list-item]
        return record

    record = record_with_steps()
    session, _prompts, _output = _scripted_prompt_session(["Partial", ""])
    session.prompt_verdict(record, scene)
    assert record.operator_judgement == "partial"
    (event,) = record.events
    assert event.kind == "operator"
    assert event.t == 3
    assert event.data["verdict"] == "partial"

    record = record_with_steps()
    session, _prompts, _output = _scripted_prompt_session(["skip", ""])
    session.prompt_verdict(record, scene)
    assert record.operator_judgement is None
    assert record.events == []

    prompts: list[str] = []

    def eof(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError

    record = record_with_steps()
    OperatorSession(input_fn=eof).prompt_verdict(record, scene)
    assert record.operator_judgement is None
    assert record.operator_note is None
    assert record.events == []
    assert prompts == [_PROMPT]


@pytest.mark.parametrize(
    ("verdict", "note_input", "expected_judgement", "expected_note", "expected_event"),
    [
        (
            "y",
            "gripper closed early",
            "y",
            "gripper closed early",
            {"verdict": "y", "source": "prompt", "note": "gripper closed early"},
        ),
        ("y", "", "y", None, {"verdict": "y", "source": "prompt", "note": None}),
        ("y", " \t ", "y", None, {"verdict": "y", "source": "prompt", "note": None}),
        (
            "y",
            "  Mixed CASE  ",
            "y",
            "Mixed CASE",
            {"verdict": "y", "source": "prompt", "note": "Mixed CASE"},
        ),
        (
            "skip",
            "  camera unplugged  ",
            None,
            "camera unplugged",
            {"verdict": "skip", "source": "prompt", "note": "camera unplugged"},
        ),
        ("skip", "", None, None, None),
    ],
)
def test_prompt_operator_records_optional_grader_notes(
    verdict: str,
    note_input: str,
    expected_judgement: str | None,
    expected_note: str | None,
    expected_event: dict[str, str | None] | None,
) -> None:
    session, _prompts, _output = _scripted_prompt_session([verdict, note_input])
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == expected_judgement
    assert record.operator_note == expected_note
    if expected_event is None:
        assert record.events == []
    else:
        (event,) = record.events
        assert event.data == expected_event


def test_prompt_operator_keeps_verdict_when_notes_prompt_reaches_eof() -> None:
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        if prompt == _PROMPT:
            return "n"
        raise EOFError

    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    OperatorSession(input_fn=answer).prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert prompts == [_PROMPT, _NOTES_PROMPT]
    assert record.operator_judgement == "n"
    assert record.operator_note is None
    (event,) = record.events
    assert event.data == {"verdict": "n", "source": "prompt", "note": None}


def test_prompt_verdict_prompts_for_operator_ended_trial() -> None:
    session, _prompts, _output = _scripted_prompt_session(["partial", "left gripper slipped"])
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=True,
        termination_reason="operator_end",
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "partial"
    assert record.operator_note == "left gripper slipped"


def test_prompt_verdict_notes_early_end_reason_before_asking() -> None:
    session, prompts, output = _scripted_prompt_session(["y", ""])
    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        truncated=True,
        termination_reason="give_up",
    )

    session.prompt_verdict(record, Scene(id="s0", instruction="reach"))

    assert "note: this trial ended early ('give_up')\n" in output
    assert prompts == [_PROMPT, _NOTES_PROMPT]
    assert record.operator_judgement == "y"


# --- Footer mode: line editor + pump + two-state renderer (plan 0051, task 1) -------------


class _ScriptedFd:
    """Feed pre-scripted raw byte chunks to the footer pump's fd seams.

    ``chunks`` is public and mutable so a test can queue more bytes between two
    ``poll()`` calls (exercising state that must persist across polls).
    """

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.chunks: list[bytes] = list(chunks) if chunks is not None else []

    def readable(self) -> bool:
        return bool(self.chunks)

    def read(self) -> bytes:
        return self.chunks.pop(0)


def _footer_session(
    output: list[str], fd: _ScriptedFd | None = None, width: int = 200
) -> tuple[OperatorSession, _ScriptedFd]:
    """Build a footer-enabled session, run ``begin_trial()``, then drop its input-row draw."""
    fd = fd if fd is not None else _ScriptedFd()
    session = OperatorSession(
        write=output.append,
        fd_readable=fd.readable,
        fd_read=fd.read,
        width_fn=lambda: width,
    )
    session.enable_footer()
    session.begin_trial()
    output.clear()
    return session, fd


def test_enable_footer_is_noop_on_caller_injected_console() -> None:
    output: list[str] = []
    session = OperatorSession(console=_RecordingConsole(), write=output.append)

    session.enable_footer()
    session.begin_trial()

    assert output == []


def test_default_console_dispatch_closures_delegate_to_fd_seams_when_footer_inactive() -> None:
    output: list[str] = []
    chunks = [b"/oops\n"]
    session = OperatorSession(
        write=output.append,
        fd_readable=lambda: bool(chunks),
        fd_read=lambda: chunks.pop(0),
    )

    assert session.poll() == ConsolePoll()
    assert output == [f"{USAGE}\n"]


def test_footer_begin_trial_draws_empty_input_row() -> None:
    output: list[str] = []
    fd = _ScriptedFd()
    session = OperatorSession(
        write=output.append, fd_readable=fd.readable, fd_read=fd.read, width_fn=lambda: 200
    )
    session.enable_footer()

    session.begin_trial()

    assert output == ["\r\x1b[K> "]


def test_footer_begin_trial_closes_plain_open_status_then_one_row_branch() -> None:
    output: list[str] = []
    fd = _ScriptedFd()
    session = OperatorSession(
        write=output.append, fd_readable=fd.readable, fd_read=fd.read, width_fn=lambda: 200
    )
    session.status("t = 3s")
    output.clear()
    session.enable_footer()

    session.begin_trial()

    assert output == ["\n\r\x1b[K> "]
    output.clear()

    session.status("t = 4s")

    assert output == ["\r\x1b[Kt = 4s\n\r\x1b[K> "]
    assert "\x1b[A" not in output[0]


def test_footer_begin_trial_drain_stops_on_real_eof() -> None:
    output: list[str] = []
    fd = _ScriptedFd([b""])
    session = OperatorSession(
        write=output.append, fd_readable=fd.readable, fd_read=fd.read, width_fn=lambda: 200
    )
    session.enable_footer()

    session.begin_trial()

    assert output == ["\r\x1b[K> "]
    poll = session.poll()
    assert poll == ConsolePoll()


def test_footer_status_none_on_already_closed_status_is_noop() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)

    session.status(None)

    assert output == []


def test_footer_default_width_fn_reads_terminal_size(monkeypatch: pytest.MonkeyPatch) -> None:
    output: list[str] = []
    fd = _ScriptedFd()
    monkeypatch.setattr(shutil, "get_terminal_size", lambda: os.terminal_size((10, 24)))
    session = OperatorSession(write=output.append, fd_readable=fd.readable, fd_read=fd.read)
    session.enable_footer()
    session.begin_trial()
    output.clear()
    fd.chunks = [b"abcdefgh"]

    session.poll()

    assert output == ["\r\x1b[K> cdefgh"]


def test_footer_begin_trial_discards_stale_bytes_with_no_echo_and_clears_state() -> None:
    output: list[str] = []
    fd = _ScriptedFd([b"stale", b"\n"])
    session = OperatorSession(
        write=output.append, fd_readable=fd.readable, fd_read=fd.read, width_fn=lambda: 200
    )
    session.enable_footer()

    session.begin_trial()

    assert output == ["\r\x1b[K> "]
    assert fd.chunks == []
    assert session.poll().messages == ()


def test_footer_one_row_write_line_has_no_cursor_up() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)

    session.write_line("hello")

    assert output == ["\r\x1b[Khello\n\r\x1b[K> "]
    assert "\x1b[A" not in output[0]


def test_footer_first_status_creates_status_row_from_one_row_state() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)

    session.status("t = 3s")

    assert output == ["\r\x1b[Kt = 3s\n\r\x1b[K> "]


def test_footer_two_row_status_update() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)
    session.status("t = 3s")
    output.clear()

    session.status("t = 4s")

    assert output == ["\x1b[A\r\x1b[Kt = 4s\x1b[B\r\x1b[K> "]


def test_footer_two_row_write_line_clears_input_row_first() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)
    session.status("t = 4s")
    output.clear()

    session.write_line("operator message")

    assert output == ["\r\x1b[K\x1b[A\r\x1b[Koperator message\nt = 4s\n\r\x1b[K> "]


def test_footer_status_none_collapses_two_row_to_one_row_retaining_input() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    session.status("t = 4s")
    output.clear()
    fd.chunks = [b"h", b"i"]
    session.poll()
    output.clear()

    session.status(None)

    assert output == ["\r\x1b[K\x1b[A\r\x1b[K> hi"]

    # The one-row branch is active again: no cursor-up in a follow-up status(line).
    output.clear()
    session.status("t = 5s")
    assert output == ["\r\x1b[Kt = 5s\n\r\x1b[K> hi"]


def test_footer_input_echo_one_row_state() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"h"]

    session.poll()

    assert output == ["\r\x1b[K> h"]


def test_footer_input_echo_two_row_state() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    session.status("t = 4s")
    output.clear()
    fd.chunks = [b"h"]

    session.poll()

    assert output == ["\r\x1b[K> h"]


def test_footer_end_trial_teardown_one_row_state() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)

    session.end_trial()

    assert output == ["\r\x1b[K"]


def test_footer_end_trial_teardown_two_row_state() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)
    session.status("t = 4s")
    output.clear()

    session.end_trial()

    assert output == ["\r\x1b[K\x1b[A\r\x1b[K"]


def test_footer_end_trial_teardown_is_idempotent() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output)

    session.end_trial()
    output.clear()
    session.end_trial()

    assert output == []


def test_footer_end_trial_noop_when_footer_never_entered_leaves_plain_status_untouched() -> None:
    output: list[str] = []
    session = OperatorSession(write=output.append)
    session.status("t = 3s")
    output.clear()

    session.end_trial()

    assert output == []
    # The plain-open status line survives: end_trial() must not touch it when footer
    # mode was never entered (the plain-mode byte-identity backstop).
    session.status(None)
    assert output == ["\n"]


def test_footer_input_row_tail_clips_at_width_minus_4() -> None:
    output: list[str] = []
    session, fd = _footer_session(output, width=10)
    fd.chunks = [b"abcdefgh"]

    session.poll()

    assert output == ["\r\x1b[K> cdefgh"]


def test_footer_status_row_clips_at_width_minus_1() -> None:
    output: list[str] = []
    session, _fd = _footer_session(output, width=6)

    session.status("0123456789")

    assert output == ["\r\x1b[K56789\n\r\x1b[K> "]


def test_footer_reclips_after_width_fn_change() -> None:
    output: list[str] = []
    width = {"cols": 200}
    fd = _ScriptedFd()
    session = OperatorSession(
        write=output.append,
        fd_readable=fd.readable,
        fd_read=fd.read,
        width_fn=lambda: width["cols"],
    )
    session.enable_footer()
    session.begin_trial()
    output.clear()
    fd.chunks = [b"abcdef"]
    session.poll()
    assert output == ["\r\x1b[K> abcdef"]
    output.clear()

    width["cols"] = 8
    fd.chunks = [b"g"]
    session.poll()

    assert output == ["\r\x1b[K> defg"]


def test_footer_pump_backspace_deletes_one_byte() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"a", b"b"]
    session.poll()
    output.clear()

    fd.chunks = [b"\x7f"]
    session.poll()

    assert output == ["\r\x1b[K> a"]


def test_footer_pump_backspace_on_empty_buffer_is_noop() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"\x7f"]

    session.poll()

    assert output == ["\r\x1b[K> "]


def test_footer_pump_backspace_after_multibyte_char_removes_whole_character() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"caf", "é".encode()]
    session.poll()
    output.clear()

    fd.chunks = [b"\x7f"]
    session.poll()

    assert output == ["\r\x1b[K> caf"]

    fd.chunks = [b"\n"]
    poll = session.poll()
    assert poll.messages == ("caf",)


def test_footer_pump_ctrl_u_kills_line() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"abc"]
    session.poll()
    output.clear()

    fd.chunks = [b"\x15"]
    session.poll()

    assert output == ["\r\x1b[K> "]


def test_footer_pump_ignores_other_control_bytes() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"a", b"\x01", b"\n"]

    poll = session.poll()

    assert poll.messages == ("a",)


@pytest.mark.parametrize("terminator", [b"\n", b"\r"])
def test_footer_pump_enter_completes_line_reaching_console_same_poll(terminator: bytes) -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"hello", terminator]

    poll = session.poll()

    assert poll.messages == ("hello",)
    assert output[-1] == "\r\x1b[K> "


def test_footer_pump_bytes_without_newline_echo_but_deliver_nothing() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"h", b"i"]

    poll = session.poll()

    assert poll == ConsolePoll()
    assert output == ["\r\x1b[K> h", "\r\x1b[K> hi"]


def test_footer_pump_split_utf8_across_chunks_decodes_cleanly_at_completion() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    lead, cont = "é".encode()[0:1], "é".encode()[1:2]
    fd.chunks = [lead, cont, b"\n"]

    poll = session.poll()

    assert poll.messages == ("é",)


def test_footer_pump_handles_64kib_paste_chunk() -> None:
    output: list[str] = []
    session, fd = _footer_session(output, width=50)
    pasted = b"a" * 65536
    fd.chunks = [pasted]

    session.poll()

    assert output == ["\r\x1b[K> " + "a" * 46]
    output.clear()

    fd.chunks = [b"\n"]
    poll = session.poll()

    assert poll.messages == (pasted.decode(),)


def test_footer_pump_swallows_arrow_key_csi_sequence() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"a", b"\x1b[A", b"b"]

    poll = session.poll()

    assert output == ["\r\x1b[K> a", "\r\x1b[K> a", "\r\x1b[K> ab"]
    fd.chunks = [b"\n"]
    poll = session.poll()
    assert poll.messages == ("ab",)


def test_footer_pump_swallows_csi_sequence_with_intermediate_byte() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"\x1b[3~"]

    session.poll()

    assert output == ["\r\x1b[K> "]


def test_footer_pump_swallows_ss3_sequence() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"\x1bOP"]

    session.poll()

    assert output == ["\r\x1b[K> "]


def test_footer_pump_swallows_bare_esc_plus_byte() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"\x1bq", b"z"]

    session.poll()

    assert output == ["\r\x1b[K> ", "\r\x1b[K> z"]


def test_footer_pump_escape_sequence_split_across_two_fd_read_chunks() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"\x1b", b"[", b"A", b"z"]

    session.poll()

    assert output == ["\r\x1b[K> ", "\r\x1b[K> ", "\r\x1b[K> ", "\r\x1b[K> z"]


def test_footer_pump_escape_sequence_split_across_two_poll_calls() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b"\x1b"]
    session.poll()
    output.clear()

    fd.chunks = [b"[A"]
    poll = session.poll()

    assert output == ["\r\x1b[K> "]
    assert poll == ConsolePoll()

    output.clear()
    fd.chunks = [b"z", b"\n"]
    poll = session.poll()
    assert poll.messages == ("z",)


def test_footer_dispatch_eof_contract_returns_empty_string_only_on_real_eof() -> None:
    output: list[str] = []
    session, fd = _footer_session(output)
    fd.chunks = [b""]

    poll = session.poll()

    assert poll == ConsolePoll()
