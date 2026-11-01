"""The ``inspect-robots eval-set`` command: several registered tasks against one
resolved policy/embodiment pair, serially or in parallel.."""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from inspect_robots.defaults import Defaults, load_defaults
from inspect_robots.session import OperatorSession

from ._shared import _DIM, _styled
from .components import (
    _announce_components,
    _apply_epochs_or_exit,
    _check_shared_run_conflicts,
    _match_tasks,
    _resolve_or_exit,
)
from .sessions import (
    _announce_live_view,
    _attended,
    _build_and_announce_guardrails,
    _build_grader,
    _build_operator_session,
    _build_voice_input,
    _close_voice_input,
    _connect_grader_session,
    _start_voice_input,
)
from .summary import _print_eval_set_summary

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog
    from inspect_robots.logging.sink import LogSink

_EVAL_SET_OUTPUT_LOCK = threading.Lock()


def _cmd_eval_set(args: argparse.Namespace) -> int:
    """Run matched tasks with one component lifecycle per serial set or parallel task."""

    from inspect_robots import cli
    from inspect_robots._parallel import run_parallel, validate_max_workers
    from inspect_robots.errors import ConfigError

    _check_shared_run_conflicts(args)
    try:
        validate_max_workers(args.max_workers)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_workers > 1 and (not args.no_prompt or args.voice):
        raise SystemExit("--max-workers > 1 requires --no-prompt and does not support --voice")
    task_names = _match_tasks(args.tasks)
    defaults = load_defaults(os.environ)
    try:
        if args.max_workers == 1:
            success, logs = _run_cli_eval_set(args, defaults, task_names)
        else:

            def run_task(name: str) -> tuple[bool, list[EvalLog]]:
                return _run_cli_eval_set(args, defaults, [name], parallel=True)

            results = run_parallel(task_names, run_task, args.max_workers)
            success = all(result[0] for result in results)
            logs = [log for _, task_logs in results for log in task_logs]
    except KeyboardInterrupt:
        cli._print_degraded(f"cancelled: partial logs are under {args.log_dir}")
        print(
            _styled(
                f"hint: inspect a log with: inspect-robots inspect {args.log_dir}/<task>_<id>.json",
                _DIM,
            )
        )
        print(_styled(f"hint: browse all logs: inspect-robots view {args.log_dir}", _DIM))
        return 130
    _print_eval_set_summary(success, logs, args.log_dir)
    return 0 if success else 1


def _run_cli_eval_set(
    args: argparse.Namespace,
    defaults: Defaults,
    task_names: Sequence[str],
    *,
    parallel: bool = False,
) -> tuple[bool, list[EvalLog]]:
    from inspect_robots import cli

    """Own components and sinks for one sequential set, or one parallel task."""
    from inspect_robots import eval_set
    from inspect_robots._parallel import HaltPreservingExitStack
    from inspect_robots.logging import JsonLogSink, LiveLogSink

    tasks = [_resolve_or_exit("task", name) for name in task_names]
    if args.epochs is not None:
        tasks = [_apply_epochs_or_exit(t, args.epochs, attribute_task=True) for t in tasks]

    # Attendedness picks the default grader, never gates grader wiring
    # (plan 0049). Built and preflighted before any component (plan 0085).
    grader = _build_grader(args, defaults)
    with HaltPreservingExitStack() as resources:
        resolved = (
            cli._resolve_components(args, defaults, resources=resources)
            if parallel
            else cli._resolve_components(args, defaults)
        )
        embodiment = resolved.embodiment
        # The CLI resolved the embodiment itself, so eval_set() does not own
        # its close. Every callback runs, and cleanup cannot replace a halt.
        if not parallel:
            resources.callback(resolved.claim.release)
        resources.callback(embodiment.close)
        if parallel and not embodiment.info.is_simulated:
            raise SystemExit("--max-workers > 1 requires a simulated embodiment")
        with _EVAL_SET_OUTPUT_LOCK:
            _announce_components(resolved)
            print(f"tasks: {', '.join(task_names)}")
            approver = _build_and_announce_guardrails(
                args, embodiment.info.action_space, resolved.embodiment
            )
            _announce_live_view(args, resolved)

        operator_input = None
        operator_session = None
        if _attended(args):
            operator_session, operator_input = _build_operator_session(resolved.policy, embodiment)
        voice_input = _build_voice_input(args, operator_input)
        resources.callback(_close_voice_input, voice_input)
        if voice_input is not None:
            _start_voice_input(
                voice_input,
                cast(OperatorSession, operator_input),
                resolved.policy,
            )
        _connect_grader_session(grader, operator_session)
        sink = JsonLogSink(args.log_dir)
        sinks: list[LogSink] = [sink]
        if not args.no_live_log:
            live_sink = LiveLogSink(args.log_dir)
            sinks.append(live_sink)

            def remove_live_snapshot() -> None:
                # A canonical-write failure makes the live snapshot the only
                # record of this run. Otherwise unlink it, including on halts.
                # An unlink failure must still allow the remaining closes.
                if live_sink.path is not None and not sink.write_failed:
                    with suppress(OSError):
                        live_sink.path.unlink(missing_ok=True)

            resources.callback(remove_live_snapshot)
        success, logs = eval_set(
            tasks,
            resolved.policy,
            embodiment,
            log_dir=args.log_dir,
            sinks=sinks,
            seed=args.seed,
            fail_on_error=args.fail_on_error if args.fail_on_error is not None else False,
            approver=approver,
            store_frames=(
                args.store_frames if args.store_frames is not None else defaults.store_frames
            ),
            retry_attempts=args.retry_attempts,
            operator_input=operator_input,
            grader=grader,
            environment_id=args.environment_id,
            environment_revision=args.environment_revision,
            policy_checkpoint=args.policy_checkpoint,
        )
        return success, logs
