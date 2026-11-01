"""Human-readable result rendering shared by ``run``, ``eval-set``, and ``inspect``:
outcome lines, step-limit notices, transcripts, and run summaries.."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from inspect_robots._html import _chat_content, _display_status, _is_chat_transcript

from ._shared import (
    _BOLD,
    _CYAN,
    _DIM,
    _GREEN,
    _OUTCOME_PHRASES,
    _RED,
    _YELLOW,
    _format_metric,
    _styled,
)

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog


def _step_limit_count(log: EvalLog) -> int:
    """Count recorded trials whose termination reason is the step horizon."""
    return sum(
        reason == "max_steps" for scene in log.samples for reason in scene.termination_reasons
    )


def _seconds_horizon(log: EvalLog) -> tuple[float, int, float | None] | None:
    """Return a validated declared/resolved seconds horizon from a saved log."""
    max_seconds = log.eval.max_seconds
    max_steps = log.eval.max_steps
    if not (
        isinstance(max_seconds, (int, float))
        and not isinstance(max_seconds, bool)
        and math.isfinite(max_seconds)
        and max_seconds > 0
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and max_steps > 0
    ):
        return None

    rate = log.eval.embodiment_info.get("control_hz")
    valid_rate = (
        float(rate)
        if isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(rate)
        and rate > 0
        else None
    )
    return float(max_seconds), max_steps, valid_rate


def _seconds_horizon_text(log: EvalLog) -> str | None:
    """Format a seconds-based horizon for compact CLI metadata surfaces."""
    horizon = _seconds_horizon(log)
    if horizon is None:
        return None
    max_seconds, max_steps, rate = horizon
    text = f"{max_seconds:g}s -> {max_steps} steps"
    return f"{text} at {rate:g} Hz" if rate is not None else text


def _outcome_line(log: EvalLog) -> tuple[str, bool] | None:
    """Return an outcome digest and unmapped flag, or ``None`` with no reasons."""
    reasons: list[object] = [
        reason for scene in log.samples for reason in scene.termination_reasons
    ]
    if not reasons:
        return None

    counts: dict[str, int] = {}
    has_unmapped = False
    for reason in reasons:
        text = "" if reason is None else str(reason)
        if not text:
            phrase = "no reason recorded"
        else:
            phrase = _OUTCOME_PHRASES.get(text, text)
            if text not in _OUTCOME_PHRASES:
                has_unmapped = True
        counts[phrase] = counts.get(phrase, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(reasons) == 1:
        return ordered[0][0], has_unmapped
    return ", ".join(f"{count} {phrase}" for phrase, count in ordered), has_unmapped


def _print_step_limit_notice(log: EvalLog, is_adhoc: bool) -> None:
    """Print the shared timeout note and the horizon-ownership hint when needed."""
    count = _step_limit_count(log)
    if count == 0:
        return

    note = f"note: {count}/{log.results.total_trials} trials hit the step limit before terminating"
    max_steps = log.eval.max_steps
    seconds_horizon = _seconds_horizon(log)
    if seconds_horizon is not None:
        max_seconds, resolved_steps, rate = seconds_horizon
        parenthetical = f"max_seconds={max_seconds:g}, resolved max_steps={resolved_steps}"
        if rate is not None:
            parenthetical += f" at {rate:g} Hz"
        note += f" ({parenthetical})"
    # Guards below reject bool/str values a hand-edited log can smuggle past
    # from_dict (bool is an int subclass, so isinstance alone lets True in).
    elif isinstance(max_steps, int) and not isinstance(max_steps, bool):
        parenthetical = f"max_steps={max_steps}"
        rate = log.eval.embodiment_info.get("control_hz")
        if (
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and math.isfinite(rate)
            and rate > 0
        ):
            parenthetical += f", ~{max_steps / rate:g}s at {rate:g} Hz"
        note += f" ({parenthetical})"
    print(_styled(note, _YELLOW))
    if is_adhoc:
        hint = "hint: raise it with --max-steps N or: inspect-robots config set max_steps N"
    elif seconds_horizon is not None:
        hint = f"hint: task {log.eval.task!r} defines its own max_seconds"
    else:
        hint = f"hint: task {log.eval.task!r} defines its own max_steps"
    print(_styled(hint, _DIM))


def _has_policy_transcripts(log: EvalLog) -> bool:
    """Whether any recorded trial carries a policy audit record."""
    return any(
        transcript is not None for scene in log.samples for transcript in scene.policy_transcripts
    )


def _print_degraded(line: str) -> None:
    """Print transcript-derived text, replacing unencodable code points.

    Transcripts are foreign data: a hostile or buggy model server can put lone
    UTF-16 surrogates in message content, and they survive the log's JSON
    round-trip but crash ``print`` on a strict-UTF-8 stdout. The forensic
    reader must degrade, never crash, on the episodes it exists to explain.
    """
    print(line.encode("utf-8", errors="replace").decode("utf-8"))


def _render_chat_transcript(transcript: list[object]) -> None:
    """Print roles, text, tool calls, and their indented results."""

    from inspect_robots import cli

    for raw_message in transcript:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message["role"])
        content = _chat_content(raw_message.get("content"))
        if role == "tool":
            suffix = "" if content is None else f" {content}"
            cli._print_degraded(f"        tool:{suffix}")
            continue
        suffix = "" if content is None else f" {content}"
        cli._print_degraded(f"    {role}:{suffix}")
        tool_calls = raw_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for raw_call in tool_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "unknown"))
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            cli._print_degraded(f"      -> {name}({arguments})")


def _print_policy_transcripts(log: EvalLog) -> None:
    """Append each available policy audit record in a tolerant human-readable form."""

    from inspect_robots import cli

    if not _has_policy_transcripts(log):
        print("no policy transcripts recorded")
        return
    print("policy transcripts:")
    for scene in log.samples:
        for trial, transcript in enumerate(scene.policy_transcripts):
            if transcript is None:
                continue
            cli._print_degraded(f"scene {scene.scene_id}, trial {trial}:")
            if _is_chat_transcript(transcript):
                _render_chat_transcript(transcript)
                continue
            for line in json.dumps(transcript, indent=2).splitlines():
                cli._print_degraded(f"    {line}")


def _print_survivor_warning(log: EvalLog) -> None:
    """Flag a successful run whose metrics no clean scene backs (issue #440)."""
    from inspect_robots.eval import _survivor_warning

    message = _survivor_warning(log)
    if message is not None:
        print(_styled(f"warning: {message}", _YELLOW))


def _print_run_summary(log: EvalLog, log_path: str, is_adhoc: bool) -> None:
    """Print the compact post-run summary and failure diagnostics."""

    from inspect_robots import cli

    failed = log.status != "success"
    errored_count = log.results.errored_trials
    status_color = _RED if failed else _GREEN
    print(f"{_styled('run status:', _CYAN)} {_styled(_display_status(log.status), status_color)}")
    outcome = _outcome_line(log)
    if outcome is not None:
        digest, has_unmapped = outcome
        line = f"{_styled('outcome:', _CYAN)} {digest}"
        if has_unmapped:
            cli._print_degraded(line)
        else:
            print(line)
    if failed and log.error is not None:
        print(f"{_styled('error:', _CYAN)} {_styled(log.error, _RED)}")
    if failed or errored_count:
        # Non-successful scenes are failure context. Errored scenes stay visible
        # even when the run succeeded (issue #73).
        for scene in log.samples:
            if scene.status != "success":
                detail = "" if scene.error in (None, log.error) else f": {scene.error}"
                print(f"  [{_styled(scene.status, _RED)}] {scene.scene_id}{detail}")
    _print_step_limit_notice(log, is_adhoc)
    _print_survivor_warning(log)
    trials = f"trials: {log.results.total_trials}"
    if errored_count:
        trials += f" ({errored_count} errored)"
    print(f"{_styled('scenes:', _CYAN)} {log.results.total_scenes}  {trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {_styled(_format_metric(value), _BOLD)}")
    print(f"{_styled('log:', _CYAN)} {_styled(log_path, _DIM)}")
    # Every run ends with the copy-pasteable read-back command (issue #90):
    # a bare path teaches a first-time user nothing about what to do next.
    print(_styled(f"hint: inspect it with: inspect-robots inspect {log_path}", _DIM))
    print(_styled(f"hint: HTML viewer: inspect-robots view {log_path}", _DIM))
    if _has_policy_transcripts(log):
        print(
            _styled(
                f"hint: agent conversation: inspect-robots inspect {log_path} --transcript",
                _DIM,
            )
        )
    if log.stats.frames_dir is not None:
        from inspect_robots._video import count_frames, resolve_frames_dir

        # Gate on frames actually existing: a camera-less --store-frames run
        # records a frames_dir but writes nothing, and the hint must not
        # point at a command that would exit "no frames found".
        root = resolve_frames_dir(log.stats.frames_dir, Path(log_path))
        if root is not None and count_frames(root):
            print(_styled(f"hint: render videos with: inspect-robots video {log_path}", _DIM))
    log_dir = Path(log_path).parent
    print(_styled(f"hint: browse all logs: inspect-robots view {log_dir}", _DIM))


def _print_eval_set_summary(success: bool, logs: Sequence[EvalLog], log_dir: str) -> None:
    """Print one overall status line, a compact row per task, then the shared log dir.

    Deliberately not N full ``_print_run_summary``s: a 10-task benchmark run
    should not scroll 10 screens of near-identical output. The status line and
    per-task labels reuse ``run``'s status vocabulary (issue #125) so the two
    commands read as one CLI.
    """
    status = "success" if success else "error"
    print(
        f"{_styled('run status:', _CYAN)} "
        f"{_styled(_display_status(status), _GREEN if success else _RED)}"
    )
    for log in logs:
        ok = log.status == "success"
        metrics = ", ".join(
            f"{name}={_format_metric(value)}" for name, value in sorted(log.results.metrics.items())
        )
        detail = metrics or (log.error or "")
        if metrics and not ok and log.error:
            # A run can fail yet keep metrics (e.g. ungraded trials abstain);
            # show why it failed next to them.
            detail = f"{metrics}  ({log.error})"
        row = f"  [{_styled(_display_status(log.status), _GREEN if ok else _RED)}] {log.eval.task}"
        horizon = _seconds_horizon_text(log)
        if horizon is not None:
            row += f" [{horizon}]"
        print(f"{row}  {detail}" if detail else row)
    print(f"{_styled('log dir:', _CYAN)} {_styled(log_dir, _DIM)}")
    if not success:
        print(
            _styled(
                f"hint: inspect a log with: inspect-robots inspect {log_dir}/<task>_<id>.json",
                _DIM,
            )
        )
    print(_styled(f"hint: browse all logs: inspect-robots view {log_dir}", _DIM))
