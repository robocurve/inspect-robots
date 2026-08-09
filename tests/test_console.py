"""Tests for the threadless operator console input channel."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from inspect_robots.console import (
    USAGE,
    USAGE_END_ONLY,
    ConsolePoll,
    EndRequest,
    OperatorConsole,
    OperatorInput,
)


def _scripted_source(
    chunks: list[str],
) -> tuple[Callable[[], bool], Callable[[], str]]:
    def readable() -> bool:
        return bool(chunks)

    def read() -> str:
        return chunks.pop(0)

    return readable, read


def _console(chunks: list[str], output: list[str] | None = None) -> OperatorConsole:
    readable, read = _scripted_source(chunks)
    return OperatorConsole(
        readable=readable,
        read=read,
        output_fn=(output if output is not None else []).append,
    )


def test_result_values_are_frozen_with_empty_defaults() -> None:
    end = EndRequest()
    poll = ConsolePoll()

    assert end == EndRequest(verdict=None, note=None)
    assert poll == ConsolePoll(messages=(), end=None, sources=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        end.verdict = "y"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        poll.end = end  # type: ignore[misc]


def test_feedback_preserves_spacing_apart_from_the_newline() -> None:
    console = _console(["  move farther left  \n"])

    assert console.poll() == ConsolePoll(messages=("  move farther left  ",))


def test_whitespace_only_line_prints_usage_reminder_without_ending() -> None:
    output: list[str] = []
    console = _console(["  \t  \n"], output)

    assert console.poll() == ConsolePoll()
    assert output == [USAGE]


def test_escape_sentinel_line_requests_an_unscored_end() -> None:
    # str.strip() removes the C0 controls \x1c-\x1f but not \x1b, so the sentinel
    # survives parsing even with surrounding whitespace.
    console = _console(["  \x1b  \n"])

    assert console.poll() == ConsolePoll(end=EndRequest())


def test_stop_command_requests_an_unscored_end() -> None:
    console = _console(["/stop\n"])

    assert console.poll() == ConsolePoll(end=EndRequest())


def test_stop_command_trailing_text_becomes_a_message_not_a_note() -> None:
    console = _console(["/STOP   wrap it up   \n"])

    assert console.poll() == ConsolePoll(messages=("wrap it up",), end=EndRequest())


def test_usage_reminder_prints_at_most_once_per_poll() -> None:
    output: list[str] = []
    console = _console(["\n\n/oops\n"], output)

    assert console.poll() == ConsolePoll()
    assert output == [USAGE]


def test_console_prints_its_configured_usage_string() -> None:
    output: list[str] = []
    readable, read = _scripted_source(["\n"])
    console = OperatorConsole(
        readable=readable, read=read, output_fn=output.append, usage=USAGE_END_ONLY
    )

    assert console.poll() == ConsolePoll()
    assert output == [USAGE_END_ONLY]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/y\n", EndRequest(verdict="y")),
        ("/N   \n", EndRequest(verdict="n")),
        ("/p note text here\n", EndRequest(verdict="partial", note="note text here")),
        (" /Y   spaced note   \n", EndRequest(verdict="y", note="spaced note")),
    ],
)
def test_verdict_commands_map_case_insensitively(line: str, expected: EndRequest) -> None:
    console = _console([line])

    assert console.poll() == ConsolePoll(end=expected)


def test_unknown_command_prints_usage_without_queuing_input() -> None:
    output: list[str] = []
    console = _console(["/oops please\n"], output)

    assert console.poll() == ConsolePoll()
    assert output == [USAGE]


def test_multiline_paste_in_one_chunk_is_fully_parsed() -> None:
    console = _console(["first\nsecond\nthird\n"])

    assert console.poll() == ConsolePoll(messages=("first", "second", "third"))


def test_partial_line_continues_in_a_later_poll() -> None:
    chunks = ["move a little"]
    console = _console(chunks)

    assert console.poll() == ConsolePoll()
    chunks.append(" farther\n")
    assert console.poll() == ConsolePoll(messages=("move a little farther",))


def test_multiple_chunks_are_drained_in_one_poll() -> None:
    console = _console(["first\nsec", "ond\nthird\n"])

    assert console.poll() == ConsolePoll(messages=("first", "second", "third"))


def test_end_then_feedback_returns_both_from_one_poll() -> None:
    console = _console(["/p needs another try\nkeep the wrist level\n"])

    assert console.poll() == ConsolePoll(
        messages=("keep the wrist level",),
        end=EndRequest(verdict="partial", note="needs another try"),
    )


def test_first_end_request_wins_while_later_lines_still_parse() -> None:
    output: list[str] = []
    console = _console(["/stop\n/n later verdict\n/oops\nafter end\n"], output)

    assert console.poll() == ConsolePoll(messages=("after end",), end=EndRequest())
    assert output == [USAGE]


def test_eof_discards_partial_input_and_latches_future_polls() -> None:
    chunks = ["unfinished"]
    read_calls = 0
    readable, scripted_read = _scripted_source(chunks)

    def read() -> str:
        nonlocal read_calls
        read_calls += 1
        return scripted_read()

    console = OperatorConsole(readable=readable, read=read)
    assert console.poll() == ConsolePoll()

    chunks.append("")
    assert console.poll() == ConsolePoll()
    chunks.append("must not be read\n")
    assert console.poll() == ConsolePoll()
    assert read_calls == 2


def test_eof_keeps_complete_lines_parsed_before_it() -> None:
    console = _console(["complete\npartial", ""])

    assert console.poll() == ConsolePoll(messages=("complete",))
    assert console.poll() == ConsolePoll()


def test_begin_trial_discards_pending_chunks_and_internal_partial() -> None:
    chunks = ["old partial"]
    console = _console(chunks)
    assert console.poll() == ConsolePoll()

    chunks.extend([" continuation\n", "another stale line\n"])
    console.begin_trial()
    chunks.append("fresh feedback\n")
    assert console.poll() == ConsolePoll(messages=("fresh feedback",))


def test_begin_trial_latches_always_readable_eof_without_spinning() -> None:
    readable_calls = 0
    read_calls = 0

    def readable() -> bool:
        nonlocal readable_calls
        readable_calls += 1
        return True

    def read() -> str:
        nonlocal read_calls
        read_calls += 1
        return ""

    console = OperatorConsole(readable=readable, read=read)
    console.begin_trial()
    console.begin_trial()
    assert console.poll() == ConsolePoll()
    assert readable_calls == 1
    assert read_calls == 1


def test_operator_console_satisfies_runtime_protocol() -> None:
    readable, read = _scripted_source([])
    console = OperatorConsole(readable=readable, read=read)

    assert isinstance(console, OperatorInput)


def test_tty_defaults_can_be_bound_without_reading_stdin() -> None:
    assert isinstance(OperatorConsole(), OperatorInput)
