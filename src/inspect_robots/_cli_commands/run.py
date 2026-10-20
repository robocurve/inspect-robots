"""The ``inspect-robots run`` command: one task or ad-hoc instruction on a resolved
policy/embodiment pair.."""

from __future__ import annotations

import argparse
import os
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from inspect_robots.defaults import _ADHOC_MAX_STEPS_FALLBACK, _ADHOC_SCORER_FALLBACK, load_defaults
from inspect_robots.session import OperatorSession

from ._shared import _CYAN, _DIM, _parse_kvs, _styled
from .components import (
    _announce_components,
    _apply_epochs_or_exit,
    _check_shared_run_conflicts,
    _resolve_or_exit,
)
from .sessions import (
    _announce_live_view,
    _attended,
    _build_and_announce_guardrails,
    _build_grader,
    _build_operator_session,
    _build_speaker_sink,
    _build_voice_input,
    _close_speaker_sink,
    _close_voice_input,
    _connect_grader_session,
    _start_speaker_sink,
    _start_voice_input,
)
from .summary import _print_run_summary

if TYPE_CHECKING:
    from inspect_robots.console import OperatorInput
    from inspect_robots.logging.sink import LogSink


def _cmd_run(args: argparse.Namespace) -> int:

    from inspect_robots import cli

    if args.rerun_port is not None and args.rerun_connect is not None:
        raise SystemExit(
            "--rerun-port spawns a local viewer and --rerun-connect streams "
            "to a remote one: pass only one"
        )
    if args.rerun_port is not None and args.rerun is False:
        raise SystemExit(
            "--no-rerun disables the live viewer and --rerun-port requests one: pass only one"
        )

    from inspect_robots import eval
    from inspect_robots.logging import JsonLogSink, LiveLogSink
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task

    is_auto = args.auto_task
    is_adhoc = args.instruction is not None or is_auto
    if args.auto_task_args and not is_auto:
        raise SystemExit("-A requires --auto-task")
    if is_auto and args.task_args:
        raise SystemExit("-T only applies to --task runs and cannot be used with --auto-task")
    selected_modes = sum((args.task is not None, args.instruction is not None, is_auto))
    if selected_modes > 1:
        raise SystemExit(
            "pass exactly one of --task, --instruction, or --auto-task, not both or all three"
        )
    if selected_modes == 0:
        raise SystemExit(
            "pass a registered --task name or an --instruction, or use --auto-task, to run"
        )
    if not is_adhoc:
        if args.max_steps is not None:
            raise SystemExit(
                "--max-steps only applies to --instruction or --auto-task runs; "
                "a registered task owns its horizon"
            )
        if args.scorer is not None:
            raise SystemExit(
                "--scorer only applies to --instruction or --auto-task runs; "
                "a registered task owns its scorers"
            )
    elif args.task_args:
        raise SystemExit(
            "-T only applies to --task runs; an ad-hoc instruction task takes no constructor args"
        )
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit(f"--max-steps must be >= 1, got {args.max_steps}")
    _check_shared_run_conflicts(args)

    defaults = load_defaults(os.environ)

    scorer_name: str | None = None
    max_steps: int | None = None
    if is_adhoc:
        scorer_name = args.scorer or defaults.scorer or _ADHOC_SCORER_FALLBACK
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else (defaults.max_steps or _ADHOC_MAX_STEPS_FALLBACK)
        )
        scorer = _resolve_or_exit("scorer", scorer_name)
        if is_auto:
            task = None
        else:
            task = Task(
                name="adhoc",
                scenes=[Scene(id="scene-0", instruction=args.instruction)],
                scorer=scorer,
                max_steps=max_steps,
                metadata={"instruction": args.instruction, "adhoc": True},
            )
    else:
        task = _resolve_or_exit("task", args.task, **_parse_kvs(args.task_args))

    # Attendedness picks the default grader, never gates grader wiring
    # (plan 0049). Built and preflighted before any component (plan 0085).
    grader = _build_grader(args, defaults)
    resolved = cli._resolve_components(args, defaults)
    embodiment = resolved.embodiment
    voice_input: OperatorInput | None = None
    speaker_sink: LogSink | None = None
    live_sink: LiveLogSink | None = None
    rerun_sink: LogSink | None = None
    try:
        if is_auto:
            import inspect_robots.taskgen
            from inspect_robots.errors import ConfigError

            # Config supplies generator defaults; explicit -A wins per key.
            # A persisted [taskgen.args] seed collides with the explicit
            # kwarg below and fails loudly: the seed knob is --seed.
            taskgen_kvs = {**defaults.taskgen_args, **_parse_kvs(args.auto_task_args)}
            try:
                scene = inspect_robots.taskgen.generate_scene(
                    resolved.embodiment,
                    seed=args.seed,
                    **taskgen_kvs,
                )
            except ConfigError as exc:
                raise SystemExit(str(exc)) from exc
            except TypeError as exc:
                raise SystemExit(
                    f"invalid arguments for automatic task generation: {exc}; "
                    "check [taskgen.args] and -A k=v"
                ) from exc
            task = Task(
                name="auto",
                scenes=[scene],
                scorer=scorer,
                max_steps=cast(int, max_steps),
                metadata={
                    "adhoc": True,
                    "auto_task": True,
                    "instruction": scene.instruction,
                },
            )
            print(f"auto task: {scene.instruction}")
            print("rubric:")
            rubric = cast(str, scene.metadata["rubric"])
            for line in rubric.splitlines():
                print(f"  {line}")
        task = cast(Task, task)
        if args.epochs is not None:
            task = _apply_epochs_or_exit(task, args.epochs)

        _announce_components(resolved)
        approver = _build_and_announce_guardrails(
            args, embodiment.info.action_space, resolved.embodiment
        )
        _announce_live_view(args, resolved)

        operator_input = None
        operator_session = None
        if _attended(args):
            operator_session, operator_input = _build_operator_session(resolved.policy, embodiment)
        voice_input = _build_voice_input(args, operator_input)
        if voice_input is not None:
            _start_voice_input(
                voice_input,
                cast(OperatorSession, operator_input),
                resolved.policy,
            )
        speaker_sink = _build_speaker_sink(args)
        if speaker_sink is not None:
            _start_speaker_sink(speaker_sink)
        _connect_grader_session(grader, operator_session)

        # Construct the sink explicitly so we can tell the user where the log went.
        sink = JsonLogSink(args.log_dir)
        sinks: list[LogSink] = [sink]
        if not args.no_live_log:
            live_sink = LiveLogSink(args.log_dir)
            sinks.append(live_sink)
        save_wanted = args.rerun_save if args.rerun_save is not None else defaults.rerun_save
        if args.rerun_connect is not None:
            from inspect_robots.logging.rerun_sink import RerunSink

            rerun_sink = RerunSink(
                connect_url=args.rerun_connect,
                recording_dir=args.log_dir if save_wanted else None,
            )
            sinks.append(rerun_sink)
            suffix = " (+ .rrd)" if save_wanted else ""
            print(f"{_styled('rerun:', _CYAN)} connect {args.rerun_connect}{suffix}")
        else:
            spawn_wanted = args.rerun_port is not None or (
                args.rerun if args.rerun is not None else defaults.rerun
            )
            if spawn_wanted:
                from inspect_robots.logging.rerun_sink import RerunSink

                # spawn=True opens the live viewer; the sink itself degrades to a
                # warn-once no-op when rerun-sdk is not installed.
                port = args.rerun_port if args.rerun_port is not None else defaults.rerun_port
                if port is None:
                    rerun_sink = RerunSink(
                        spawn=True,
                        recording_dir=args.log_dir if save_wanted else None,
                    )
                else:
                    rerun_sink = RerunSink(
                        spawn=True,
                        spawn_port=port,
                        recording_dir=args.log_dir if save_wanted else None,
                    )
                sinks.append(rerun_sink)
                suffix = " (+ .rrd)" if save_wanted else ""
                print(f"{_styled('rerun:', _CYAN)} live viewer{suffix}")
            elif args.rerun_save is True:
                from inspect_robots.logging.rerun_sink import RerunSink

                rerun_sink = RerunSink(recording_dir=args.log_dir)
                sinks.append(rerun_sink)
                print(f"{_styled('rerun:', _CYAN)} recording .rrd")
        # Speaker goes last so its bounded end-of-run drain never delays the
        # JSON log write or the Rerun flush in the on_eval_end fan-out.
        if speaker_sink is not None:
            sinks.append(speaker_sink)
        try:
            logs = eval(
                task,
                resolved.policy,
                embodiment,
                log_dir=args.log_dir,
                seed=args.seed,
                sinks=sinks,
                fail_on_error=args.fail_on_error if args.fail_on_error is not None else False,
                approver=approver,
                store_frames=(
                    args.store_frames if args.store_frames is not None else defaults.store_frames
                ),
                operator_input=operator_input,
                grader=grader,
                environment_id=args.environment_id,
                environment_revision=args.environment_revision,
                policy_checkpoint=args.policy_checkpoint,
            )
        except KeyboardInterrupt:
            if sink.path is not None and sink.path.exists():
                cli._print_degraded(f"cancelled: partial log written to {sink.path}")
                print(_styled(f"hint: inspect it with: inspect-robots inspect {sink.path}", _DIM))
            else:
                cli._print_degraded("cancelled: no log written")
            return 130
    finally:
        # The CLI resolved the embodiment itself, so eval() does not own it
        # ("close what we open"). Real-hardware embodiments release motor
        # torque in close(); skipping this leaves a robot energized. The span
        # starts right after resolution: --epochs/scorer validation between
        # here and eval() can raise, and that must not leak the embodiment.
        try:
            # A failing unlink (read-only remount, full disk) must not skip the
            # close chain below or replace the run's real exception.
            # Keep the snapshot only when a canonical write failed (including a
            # Ctrl-C during that write): then it is the only record of the run.
            # Otherwise (success, or Ctrl-C before the final write) remove it.
            # write_failed is sticky across eval_set tasks, so a later task's
            # half-written snapshot may also be kept; harmless, never lost.
            if live_sink is not None and live_sink.path is not None and not sink.write_failed:
                with suppress(OSError):
                    live_sink.path.unlink(missing_ok=True)
        finally:
            try:
                _close_speaker_sink(speaker_sink)
            finally:
                try:
                    _close_voice_input(voice_input)
                finally:
                    try:
                        embodiment.close()
                    finally:
                        resolved.claim.release()
    log = logs[0]
    _print_run_summary(log, str(sink.path), is_adhoc)
    resolved_recording_path = getattr(rerun_sink, "resolved_recording_path", None)
    if resolved_recording_path:
        print(f"{_styled('rrd:', _CYAN)} {_styled(str(resolved_recording_path), _DIM)}")
    return 0 if log.status == "success" else 1
