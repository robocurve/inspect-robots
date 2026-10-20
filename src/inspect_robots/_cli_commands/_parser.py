"""The ``inspect-robots`` argument parser: ``build_parser`` plus the flag-registration
helpers its subparsers share.."""

from __future__ import annotations

import argparse

from inspect_robots import __version__
from inspect_robots.defaults import _ADHOC_MAX_STEPS_FALLBACK, _ADHOC_SCORER_FALLBACK, _CONFIG_KEYS

from ._shared import _KIND_BY_PLURAL, _LIVE_FRAMES_BUDGET_MB, DEFAULT_RERUN_CONNECT_URL


def _add_shared_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags common to ``run`` and ``eval-set``.

    Component selection (``--policy``/``--embodiment``/``-P``/``-E``/``--sim``),
    guardrails, logging, prompting control, and epoch/error handling live here
    so a new shared flag lands in both commands at once instead of drifting
    between two copies.
    """
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="suppress the operator grader: never ask the terminal operator "
        "for a success verdict or grader notes",
    )
    parser.add_argument(
        "--grader",
        help="registered grader that captures the post-trial judgement, or "
        "'none' to disable grading (default: config, then operator when the "
        "run is attended)",
    )
    parser.add_argument("--policy", help="registered policy name (default: user config)")
    parser.add_argument("--embodiment", help="registered embodiment name (default: user config)")
    parser.add_argument("-P", dest="policy_args", action="append", metavar="k=v")
    parser.add_argument("-E", dest="embodiment_args", action="append", metavar="k=v")
    parser.add_argument("-G", dest="grader_args", action="append", metavar="k=v")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="enable attended-only microphone feedback (requires inspect-robots-voice)",
    )
    parser.add_argument(
        "-V",
        dest="voice_args",
        action="append",
        metavar="k=v",
        help="pass an argument to the inspect-robots-voice input (requires --voice)",
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--no-live-log",
        action="store_true",
        help="disable the transient JSON snapshots used by view --serve",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="override each task's epoch count")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="run on the configured sim_embodiment instead of the default "
        "(real-hardware) embodiment",
    )
    parser.add_argument(
        "--fail-on-error",
        type=float,
        default=None,
        metavar="X",
        help="halt on PolicyErrors: 1 = first error, 0<X<1 = proportion, X>1 = count",
    )
    parser.add_argument(
        "--store-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stream camera frames to a per-run directory under <log-dir>/frames "
        "instead of keeping them in memory (--no-store-frames overrides a "
        "store_frames config default)",
    )
    parser.add_argument(
        "--disable-guardrails",
        action="store_true",
        help="turn off the default safety approvers (bounds clamp + per-step "
        "delta limit); actions reach the embodiment unchecked",
    )
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=None,
        metavar="D",
        help="per-step change limit for the default guardrails, in the action "
        "space's native units (default: derived from the space's bounds)",
    )
    parser.add_argument(
        "--environment-id",
        default=None,
        help="environment identifier recorded in evaluation metadata",
    )
    parser.add_argument(
        "--environment-revision",
        default=None,
        help="environment revision or commit hash recorded in evaluation metadata",
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=None,
        help="policy model checkpoint path, hash, or revision recorded in evaluation metadata",
    )


def _port_number(text: str) -> int:
    """Parse a TCP port number for an argparse option."""
    port = int(text)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1-65535")
    return port


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the --config override to a subcommand that reads the config file."""
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="use this config file instead of "
        "<config-home>/inspect-robots/config.ini (sets $INSPECT_ROBOTS_CONFIG "
        "for the invocation; useful for hosts driving more than one rig)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots",
        description="Inspect Robots — the Inspect AI for robotics.",
    )
    parser.add_argument("--version", action="version", version=f"inspect-robots {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="list registered components")
    p_list.add_argument(
        "what",
        nargs="?",
        choices=sorted(_KIND_BY_PLURAL),
        help="component kind to list (default: all)",
    )

    p_run = sub.add_parser("run", help="run an evaluation")
    p_run.add_argument("--task", help="registered task name")
    p_run.add_argument(
        "--instruction",
        help="run a single ad-hoc scene with this language instruction "
        "(instead of a registered --task)",
    )
    p_run.add_argument(
        "--auto-task",
        action="store_true",
        help="generate one ad-hoc task and grading rubric from the initial camera frames",
    )
    p_run.add_argument("-T", dest="task_args", action="append", metavar="k=v")
    p_run.add_argument(
        "-A",
        dest="auto_task_args",
        action="append",
        metavar="k=v",
        help="pass an argument to automatic task generation (requires --auto-task)",
    )
    _add_shared_eval_args(p_run)
    _add_config_arg(p_run)
    p_run.add_argument(
        "--speak",
        action="store_true",
        help="speak live policy notes through the local audio output "
        "(requires inspect-robots-voice)",
    )
    p_run.add_argument(
        "-S",
        dest="speak_args",
        action="append",
        metavar="k=v",
        help="pass an argument to the inspect-robots-voice speaker, e.g. "
        "-S mode=blocking|interrupt|queue (requires --speak)",
    )
    p_run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="horizon of an --instruction or --auto-task run (default: config or "
        f"{_ADHOC_MAX_STEPS_FALLBACK}); invalid with --task",
    )
    p_run.add_argument(
        "--scorer",
        default=None,
        help="scorer for an --instruction or --auto-task run (default: config or "
        f"{_ADHOC_SCORER_FALLBACK!r}); invalid with --task",
    )
    p_run.add_argument(
        "--rerun",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stream the rollout (cameras, state, actions) to a live Rerun "
        "viewer window; needs rerun-sdk (--no-rerun overrides a rerun config "
        "default)",
    )
    p_run.add_argument(
        "--rerun-save",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also save the stream as a .rrd next to the eval log (default: on whenever "
        "the rerun viewer is active; without a viewer, --rerun-save records to a "
        ".rrd only; --no-rerun-save or a rerun_save config key overrides)",
    )
    p_run.add_argument(
        "--rerun-connect",
        nargs="?",
        const=DEFAULT_RERUN_CONNECT_URL,
        default=None,
        metavar="URL",
        help="stream the rollout to a Rerun viewer already running elsewhere "
        "(e.g. your laptop via an SSH reverse tunnel: ssh -R 9876:localhost:9876 ...); "
        f"URL defaults to {DEFAULT_RERUN_CONNECT_URL}",
    )
    p_run.add_argument(
        "--rerun-port",
        type=_port_number,
        default=None,
        metavar="PORT",
        help="spawn the live Rerun viewer on this port (implies --rerun; "
        "a per-rig rerun_port config key sets the default)",
    )

    p_eval_set = sub.add_parser("eval-set", help="run a set of registered tasks in one invocation")
    p_eval_set.add_argument(
        "tasks",
        nargs="+",
        metavar="TASK",
        help="registered task name(s); shell-quoted globs match by prefix, e.g. 'kitchenbench/*'",
    )
    _add_shared_eval_args(p_eval_set)
    _add_config_arg(p_eval_set)
    p_eval_set.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="maximum concurrent simulation tasks (requires --no-prompt)",
    )
    p_eval_set.add_argument(
        "--retry-attempts",
        type=int,
        default=0,
        help="passed through to eval_set(); resumption of a partial run is "
        "accepted but not yet honored",
    )

    p_inspect = sub.add_parser("inspect", help="print a saved eval log")
    p_inspect.add_argument("log", help="path to an EvalLog JSON file")
    p_inspect.add_argument(
        "--transcript",
        action="store_true",
        help="append recorded policy transcripts",
    )
    p_inspect.add_argument(
        "--wire",
        nargs="?",
        const=True,
        default=None,
        type=int,
        metavar="CALL",
        help="append the wire-call table, or dump every attempt for CALL",
    )
    p_inspect.add_argument(
        "--trial",
        default=None,
        metavar="SCENE-eEPOCH",
        help="select a captured trial for --wire CALL",
    )

    p_summarize = sub.add_parser(
        "summarize",
        help="distill a saved eval log into a markdown learnings file",
    )
    p_summarize.add_argument("log", help="path to an EvalLog JSON file")
    p_summarize.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model id (default: deterministic digest only)",
    )
    p_summarize.add_argument(
        "--base-url",
        default="https://api.anthropic.com/v1",
        help="OpenAI-compatible API base URL (default: https://api.anthropic.com/v1)",
    )
    p_summarize.add_argument(
        "--api-key-env",
        default="ANTHROPIC_API_KEY",
        metavar="VAR",
        help="environment variable holding the API key (default: ANTHROPIC_API_KEY)",
    )
    p_summarize.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="FILE",
        help="output markdown file (default: LOG_DIR/learnings/LOG_STEM.md; - writes stdout)",
    )

    p_view = sub.add_parser(
        "view",
        help="render saved eval logs as self-contained HTML reports",
        description=(
            "Render one EvalLog JSON file, or every top-level *.json in a logs "
            "directory with a browsable index. A directory log named index.json "
            "uses index_log.html (or the next collision-free suffix)."
        ),
    )
    p_view.add_argument(
        "log",
        help="path to an EvalLog JSON file, or a logs directory to render a browsable index",
    )
    p_view.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="PATH",
        help=(
            "output HTML file for one log (default: LOG.html; - writes to stdout), "
            "or output directory for a logs directory (default: LOG_DIR/html)"
        ),
    )
    p_view.add_argument(
        "--open",
        action="store_true",
        help="open the written report in the default web browser",
    )
    p_view.add_argument(
        "--no-frames",
        action="store_true",
        help="render placeholders instead of embedding stored camera frames",
    )
    p_view.add_argument(
        "--no-video",
        action="store_true",
        help="skip embedded MP4 encoding while keeping the frame flipbook",
    )
    p_view.add_argument(
        "--frames-budget",
        type=float,
        default=50,
        metavar="MB",
        help=(
            "maximum inline frame payload in decimal MB, applied to each rendered "
            "page (default: 50; 0 is unlimited)"
        ),
    )
    p_view.add_argument(
        "--live-frames-budget",
        type=float,
        default=_LIVE_FRAMES_BUDGET_MB,
        metavar="MB",
        help=(
            "maximum inline frame payload for each running page while serving "
            f"(default: {_LIVE_FRAMES_BUDGET_MB:g}; 0 is unlimited and is not "
            "recommended while serving)"
        ),
    )
    p_view.add_argument(
        "--force",
        action="store_true",
        help=(
            "re-render existing pages in directory mode; use after changing "
            "--no-frames, --no-video, --frames-budget, or --live-frames-budget "
            "(ignored for one file)"
        ),
    )
    p_view.add_argument(
        "--serve",
        action="store_true",
        help="serve a rendered logs directory over HTTP until stopped",
    )
    p_view.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="listen port for --serve (default: 8300)",
    )
    p_view.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="bind address for --serve (default: 127.0.0.1)",
    )

    p_video = sub.add_parser(
        "video",
        help="render a log's stored camera frames to one MP4 per camera stream",
    )
    p_video.add_argument("log", help="path to an EvalLog JSON file from a --store-frames run")
    p_video.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="output directory (default: the frames directory itself)",
    )
    p_video.add_argument(
        "--fps",
        type=float,
        default=None,
        help="playback rate (default: the log's control_hz, else 10)",
    )
    p_video.add_argument(
        "--ffmpeg",
        default=None,
        metavar="PATH",
        help="ffmpeg executable to use (default: found on PATH)",
    )

    p_doctor = sub.add_parser(
        "doctor",
        help="check an installed embodiment's declared spaces for adapter conformance",
    )
    p_doctor.add_argument("--embodiment", help="registered embodiment name (default: user config)")
    p_doctor.add_argument("-E", dest="embodiment_args", action="append", metavar="k=v")
    _add_config_arg(p_doctor)

    p_setup = sub.add_parser(
        "setup",
        help="interactive first-run wizard: pick defaults and discover camera devices, "
        "then write config.ini",
    )
    _add_config_arg(p_setup)

    p_config = sub.add_parser("config", help="view or set user defaults (config.ini)")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_set = config_sub.add_parser("set", help="persist a [defaults] key to the config file")
    p_set.add_argument("key", choices=_CONFIG_KEYS)
    p_set.add_argument("value")
    _add_config_arg(p_set)
    p_show = config_sub.add_parser("show", help="print resolved defaults and their sources")
    _add_config_arg(p_show)
    return parser
