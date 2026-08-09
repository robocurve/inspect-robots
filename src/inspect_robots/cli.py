"""The ``inspect_robots`` command-line interface.

Subcommands:

- ``inspect-robots list [tasks|policies|embodiments|scorers|sinks|operator_inputs]`` — show
  registered components (builtins + installed plugins).
- ``inspect-robots run --task T --policy P --embodiment E`` — run an eval, resolving
  components from the registry. Pass constructor args with ``-T/-P/-E k=v``;
  ``--epochs``, ``--fail-on-error``, and ``--store-frames`` tune the run. The
  written log's path is printed at the end.
- ``inspect-robots eval-set TASK [TASK ...] --policy P --embodiment E`` — run several
  registered tasks (exact names or ``fnmatch`` globs, e.g. ``'kitchenbench/*'``) against
  one resolved policy/embodiment pair via
  [`eval_set`][inspect_robots.eval.eval_set]. Prints one status line and a compact
  per-task row instead of a full summary per task.
- ``inspect-robots inspect LOG.json [--transcript] [--wire [CALL]]`` — print a
  saved eval log and optionally append policy conversations or captured wire calls.
- ``inspect-robots summarize LOG.json [--model M]`` — distill a saved eval log
  into a deterministic digest or model-written learnings file.
- ``inspect-robots view LOG.json|LOG_DIR [-o PATH] [--open] [--serve]`` — render
  one saved eval log or a browsable directory index as self-contained HTML,
  optionally serving a directory until stopped.
- ``inspect-robots video LOG.json`` — render a ``--store-frames`` run's stored
  camera frames to one MP4 per (trial, camera) stream via the ffmpeg binary.
- ``inspect-robots setup`` — interactively configure defaults and camera devices.

Zero-config form (plan 0005): ``inspect-robots "place the spoon on the plate"``
is sugar for ``run --instruction "..."`` — a single ad-hoc scene on the user's
default policy/embodiment (flags > ``INSPECT_ROBOTS_POLICY``/``_EMBODIMENT``
env vars > ``~/.config/inspect-robots/config.ini``). The sugar only fires for
a first argument with interior whitespace, so a mistyped subcommand
(``inspect-robots isnpect``) errors instead of starting a robot rollout;
single-word instructions use the explicit ``run --instruction`` form.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, cast

from inspect_robots import __version__
from inspect_robots._claims import DeviceClaim, claim_devices
from inspect_robots._dotenv import init_dotenv
from inspect_robots._html import (
    _chat_content,
    _display_status,
    _is_chat_transcript,
    _status_class,
    render_html,
)
from inspect_robots._html_index import IndexEntry, render_index
from inspect_robots._pointers import derive_blob_dir, read_jsonl_prefix, resolve_log_pointer
from inspect_robots.conformance import device_slots
from inspect_robots.console import USAGE, USAGE_END_ONLY
from inspect_robots.defaults import (
    _ADHOC_MAX_STEPS_FALLBACK,
    _ADHOC_SCORER_FALLBACK,
    _CONFIG_KEYS,
    _ENV_EMBODIMENT,
    _ENV_POLICY,
    _ENV_SIM_EMBODIMENT,
    Defaults,
    _parse_value,
    _set_default,
    load_defaults,
)
from inspect_robots.registry import registered
from inspect_robots.session import OperatorSession
from inspect_robots.types import OPERATOR_END

if TYPE_CHECKING:
    from inspect_robots.approver import Approver
    from inspect_robots.console import OperatorInput
    from inspect_robots.embodiment import Embodiment
    from inspect_robots.grader import Grader
    from inspect_robots.log import EvalLog
    from inspect_robots.logging.sink import LogSink
    from inspect_robots.spaces import Box


def _styled(text: str, code: str) -> str:
    """Wrap ``text`` in an ANSI style when stdout is an interactive terminal.

    Plain text is returned when piped/redirected or when ``NO_COLOR`` is set,
    so scripts and CI logs never see escape codes.
    """
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


_BOLD = "1"
_BOLD_BRIGHT_MAGENTA = "1;95"
_DIM = "2"
_CYAN = "36"
_GREEN = "32"
_RED = "31"
_YELLOW = "33"

_OUTCOME_PHRASES = {
    "success": "succeeded",
    "failure": "failed",
    "max_steps": "hit step limit",
    OPERATOR_END: "ended by operator",
    "give_up": "gave up",
    "done": "reported done",
    "policy_stop": "stopped by policy",
    "truncated": "truncated",
}


_KIND_BY_PLURAL = {
    "tasks": "task",
    "policies": "policy",
    "embodiments": "embodiment",
    "scorers": "scorer",
    "graders": "grader",
    "sinks": "sink",
    "operator_inputs": "operator_input",
}

_PLURAL_BY_KIND = {kind: plural for plural, kind in _KIND_BY_PLURAL.items()}

_SUBCOMMANDS = (
    "list",
    "run",
    "eval-set",
    "inspect",
    "summarize",
    "view",
    "video",
    "config",
    "setup",
    "doctor",
)

_ENV_BY_KIND = {"policy": _ENV_POLICY, "embodiment": _ENV_EMBODIMENT}

DEFAULT_RERUN_CONNECT_URL = "rerun+http://127.0.0.1:9876/proxy"
_SERVE_RERENDER_SECONDS = 60
_SERVE_LIVE_RERENDER_SECONDS = 2
# Two-level directory stamp (plan 0060): a suppressed-tier page (serve pass or
# --no-video) is stamped this far below the source log's mtime so a later
# full-tier pass can recognize and upgrade it. The delta spans many filesystem
# timestamp ticks (NTFS rounds to 100 ns) and the gate compares with slack so
# tick rounding can never turn a skip into a per-tick re-render.
_SUPPRESSED_STAMP_DELTA_NS = 2_000
_STAMP_TICK_SLACK_NS = 1_000
_LIVE_FRAMES_BUDGET_MB = 8.0
_serve_sleep = time.sleep


def _parse_kvs(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key] = _parse_value(value)
    return out


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
    p_run.add_argument("-T", dest="task_args", action="append", metavar="k=v")
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
        help="horizon of an --instruction run (default: config or "
        f"{_ADHOC_MAX_STEPS_FALLBACK}); invalid with --task",
    )
    p_run.add_argument(
        "--scorer",
        default=None,
        help="scorer for an --instruction run (default: config or "
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


def _cmd_list(what: str | None) -> int:
    from inspect_robots.registry import registered

    plurals = [what] if what else sorted(_KIND_BY_PLURAL)
    for plural in plurals:
        kind = _KIND_BY_PLURAL[plural]
        names = sorted(registered(kind))
        print(f"{plural}:")
        for name in names:
            print(f"  - {name}")
        if not names:
            print("  (none)")
    return 0


def _pick_component(
    kind: str, flag_value: str | None, default: str | None, source: str | None
) -> tuple[str, str]:
    """Resolve a component name via flag > defaults, or exit with guidance."""
    if flag_value:
        return flag_value, f"--{kind}"
    if default:
        return default, f"from {source}"
    from inspect_robots.registry import registered

    names = ", ".join(sorted(registered(kind))) or "(none)"
    raise SystemExit(
        f"no {kind} given and no default configured.\n"
        f"registered {_PLURAL_BY_KIND[kind]}: {names}\n"
        f"fix: pass --{kind} NAME, set ${_ENV_BY_KIND[kind]}, "
        "run 'inspect-robots setup', or "
        f"'inspect-robots config set {kind} NAME'"
    )


def _match_tasks(patterns: Sequence[str]) -> list[str]:
    """Resolve task-name patterns (exact names or ``fnmatch`` globs) against the registry.

    Preserves first-match order across patterns, deduplicated, so
    ``eval-set 'kb/*' 'kb/pour'`` does not run ``kb/pour`` twice. A pattern
    that matches nothing is an error naming every registered task, mirroring
    ``_pick_component``'s guidance-over-traceback style.
    """
    from inspect_robots.registry import registered

    names = sorted(registered("task"))
    matched: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        hits = [n for n in names if fnmatch.fnmatchcase(n, pattern)]
        if not hits:
            available = ", ".join(names) or "(none)"
            raise SystemExit(f"no task matches {pattern!r}.\nregistered tasks: {available}")
        for n in hits:
            if n not in seen:
                seen.add(n)
                matched.append(n)
    return matched


def _config_args(
    kind: str, name: str, owner: str | None, config_args: dict[str, Any]
) -> dict[str, Any]:
    """Config-file ``[<kind>.args]``, gated to the component they were written for.

    An args section is only valid for the ``[defaults]`` component it was
    configured alongside (its owner); handing it to a differently-selected
    component injects kwargs that constructor never asked for (issue #44).
    Dropping is loud on stderr: persisted rig calibration vanishing silently
    would be a worse failure than the crash this replaces.
    """
    if name == owner:
        return config_args
    if config_args:
        reason = f"they apply to {owner!r}" if owner else f"no default {kind} is configured"
        print(f"note: ignoring [{kind}.args] for {name!r}: {reason}", file=sys.stderr)
    return {}


def _pick_sim_embodiment(defaults: Defaults) -> tuple[str, str]:
    """The --sim chain: env var > config ``sim_embodiment``, or exit with guidance.

    Deliberately does NOT consult ``--embodiment``/``$INSPECT_ROBOTS_EMBODIMENT``
    (those pick the *real* default; an exported env var is a persistent
    preference, not per-invocation intent, so --sim simply ignores it).
    """
    if defaults.sim_embodiment:
        return defaults.sim_embodiment, f"--sim, from {defaults.sim_embodiment_source}"
    from inspect_robots.registry import registered

    names = ", ".join(sorted(registered("embodiment"))) or "(none)"
    raise SystemExit(
        "--sim given but no sim embodiment configured.\n"
        f"registered embodiments: {names}\n"
        f"fix: set ${_ENV_SIM_EMBODIMENT}, or run "
        "'inspect-robots config set sim_embodiment NAME'"
    )


def _resolve_or_exit(
    kind: str, name: str, args_section: str | None = None, /, **kwargs: Any
) -> Any:
    """Registry resolution with a clean error instead of a traceback.

    Unknown names raise ``KeyError``; a factory that cannot construct itself
    (e.g. the agent policy with no model/key configured) raises a guided
    ``ConfigError``. Invalid constructor arguments raise ``TypeError``. All
    three become user-facing messages rather than tracebacks. ``args_section``
    can identify a config section whose name differs from the registry kind.
    """
    from inspect_robots.errors import ConfigError
    from inspect_robots.registry import resolve

    try:
        return resolve(kind, name, **kwargs)
    except KeyError as exc:
        message = str(exc.args[0])
        # Only the registry's own unknown-name error earns the install hint; a
        # KeyError escaping a factory must not masquerade as a missing plugin.
        if (kind == "operator_input" and message.startswith("no operator_input named")) or (
            kind == "sink" and message.startswith("no sink named 'speaker'")
        ):
            message = f"{message}; fix: pip install inspect-robots-voice"
        raise SystemExit(message) from exc
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    except TypeError as exc:
        if args_section is None:
            args_section = f"{kind}.args"
        flag = {
            "task": "-T",
            "policy": "-P",
            "embodiment": "-E",
            "sink": "-S",
            "operator_input": "-V",
        }.get(kind, "the CLI args flag")
        raise SystemExit(
            f"invalid arguments for {kind} {name!r}: {exc}; check [{args_section}] and {flag} k=v"
        ) from exc


def _build_guardrails(
    space: Box,
    max_action_delta: float | None,
    embodiment: Embodiment | None = None,
) -> tuple[Approver, list[str], list[str]]:
    """The default CLI safety chain for an action space (plan 0008 §3e).

    Returns ``(approver, active, warnings)``. Degrades per component instead
    of blocking: a component that cannot apply to this space is skipped with
    a warning naming the actual refusal reason (the constructor's message),
    so the CLI is never *less* protective than running without guardrails —
    and never silently unprotected either.
    """
    from inspect_robots.approver import (
        AutoApprover,
        ChainApprover,
        ClampApprover,
        DeltaLimitApprover,
        GuardrailContribution,
    )

    parts: list[Approver] = []
    active: list[str] = []
    warnings: list[str] = []
    if space.low is None and space.high is None:
        warnings.append("bounds clamp skipped: the action space declares no low/high bounds")
    else:
        parts.append(ClampApprover(space))
        active.append("clamp")
    try:
        # Catch the constructor's refusal generically — whatever the §3a
        # reason — rather than pre-checking an enumerated list.
        parts.append(DeltaLimitApprover(space, max_delta=max_action_delta))
        active.append("delta-limit")
    except ValueError as exc:
        warnings.append(f"delta limit skipped: {exc}")

    missing = object()
    hook = getattr(embodiment, "contribute_guardrails", missing)
    if hook is not missing:
        if not callable(hook):
            assert embodiment is not None
            raise SystemExit(
                f"embodiment {embodiment.info.name!r} contribute_guardrails is "
                f"not callable (got {type(hook).__name__})"
            )
        contribution = hook(space)
        if not isinstance(contribution, GuardrailContribution):
            assert embodiment is not None
            raise SystemExit(
                f"embodiment {embodiment.info.name!r} contribute_guardrails returned "
                f"{type(contribution).__name__}, expected GuardrailContribution"
            )
        for name, approver in contribution.approvers:
            parts.append(approver)
            active.append(name)
        warnings.extend(contribution.warnings)

    if not parts:
        warnings.append(
            "no guardrails are active for this action space; declare bounds/semantics "
            "on the embodiment or pass --max-action-delta"
        )
        return AutoApprover(), active, warnings
    return ChainApprover(*parts), active, warnings


def _attended(args: argparse.Namespace) -> bool:
    """True when the operator can be prompted: a real TTY and no ``--no-prompt``.

    The single source of truth for the prompt gate shared by ``run`` and
    ``eval-set`` — the same anti-drift argument as ``_add_shared_eval_args``.
    """
    return not args.no_prompt and sys.stdin.isatty()


def _build_operator_session(
    policy: object, embodiment: Embodiment
) -> tuple[OperatorSession, OperatorSession | None]:
    """Build the prompt owner and enable its live input channel only when safe."""
    accepts_messages = bool(getattr(policy, "accepts_operator_messages", False))
    session = OperatorSession(console_usage=None if accepts_messages else USAGE_END_ONLY)
    connect_hook = getattr(embodiment, "connect_operator_session", None)
    if callable(connect_hook):
        if sys.platform == "win32":
            session.write_line(
                _styled(
                    "operator console unavailable: select cannot watch stdin on Windows; "
                    "feedback typing stays off",
                    _YELLOW,
                )
            )
            return session, None
        connect_hook(session)
        usage = USAGE if accepts_messages else USAGE_END_ONLY
        session.enable_footer(label="sent" if accepts_messages else "noted")
        label = "operator console:"
        session.write_line(f"{_styled(label, _CYAN)} {usage.removeprefix(label + ' ')}")
        return session, session

    if not accepts_messages:
        return session, None
    if sys.platform == "win32":
        session.write_line(
            _styled(
                "operator console unavailable: select cannot watch stdin on Windows; "
                "feedback typing stays off",
                _YELLOW,
            )
        )
        return session, None

    defer_hook = getattr(embodiment, "defer_operator_end", None)
    if callable(defer_hook):
        defer_hook()
    elif not embodiment.info.is_simulated:
        session.write_line(
            _styled(
                "operator console unavailable: this embodiment predates the operator console "
                "and still owns the end-of-episode keypress; feedback typing stays off",
                _YELLOW,
            )
        )
        return session, None

    label = "operator console:"
    session.enable_footer(label="sent")
    session.write_line(f"{_styled(label, _CYAN)} {USAGE.removeprefix(label + ' ')}")
    return session, session


def _build_voice_input(
    args: argparse.Namespace, operator_input: OperatorSession | None
) -> OperatorInput | None:
    """Validate voice flags and construct the registered input when requested."""
    if args.voice_args and not args.voice:
        raise SystemExit("-V requires --voice")
    if not args.voice:
        return None
    if not _attended(args):
        raise SystemExit("--voice requires attended mode (a TTY without --no-prompt)")
    if operator_input is None:
        raise SystemExit("--voice requires a live operator input channel")
    return cast(
        "OperatorInput",
        _resolve_or_exit("operator_input", "voice", **_parse_kvs(args.voice_args)),
    )


def _start_voice_input(
    voice_input: OperatorInput, session: OperatorSession, policy: object
) -> None:
    """Start and attach a voice input, honoring its optional startup hook.

    Announces the voice model before the startup hook runs: loading can download
    model weights on a first run, and the progress bars are otherwise unlabeled.
    """
    start_hook = getattr(voice_input, "start", None)
    if callable(start_hook):
        model = getattr(voice_input, "model", None)
        if isinstance(model, str):
            session.write_line(
                f"voice: loading speech-to-text model {model} (a first run downloads it)"
            )
        try:
            listening_line = start_hook()
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        if isinstance(listening_line, str):
            session.write_line(listening_line)
    session.attach_input(voice_input, label="voice")
    if not bool(getattr(policy, "accepts_operator_messages", False)):
        session.write_line("voice notes go to the log only; the policy does not receive them")


def _close_voice_input(voice_input: OperatorInput | None) -> None:
    """Close a constructed voice input through its optional shutdown hook."""
    if voice_input is None:
        return
    close_hook = getattr(voice_input, "close", None)
    if callable(close_hook):
        close_hook()


def _build_speaker_sink(args: argparse.Namespace) -> LogSink | None:
    """Validate speaker flags and construct the registered sink when requested."""
    if args.speak_args and not args.speak:
        raise SystemExit("-S requires --speak")
    if not args.speak:
        return None
    return cast(
        "LogSink",
        _resolve_or_exit("sink", "speaker", **_parse_kvs(args.speak_args)),
    )


def _start_speaker_sink(speaker_sink: LogSink) -> None:
    """Start a speaker sink through its optional startup hook."""
    start_hook = getattr(speaker_sink, "start", None)
    if callable(start_hook):
        try:
            start_hook()
        except Exception as exc:
            raise SystemExit(str(exc)) from exc


def _close_speaker_sink(speaker_sink: LogSink | None) -> None:
    """Close a constructed speaker sink through its optional shutdown hook."""
    if speaker_sink is None:
        return
    close_hook = getattr(speaker_sink, "close", None)
    if callable(close_hook):
        close_hook()


def _select_grader_name(args: argparse.Namespace, defaults: Defaults) -> str | None:
    """Resolve the grader name: flag > config > attended default (plan 0049).

    An explicit ``--grader`` is honored verbatim (``none`` disables grading;
    the contradictory pair with ``--no-prompt`` is rejected in
    ``_check_shared_run_conflicts``). A config-sourced ``operator`` needs a
    run that can actually be attended: under ``--no-prompt`` or without a TTY
    it downgrades to no grader with a stderr note, so a one-time config edit
    can never block a later cron/CI run at a prompt.
    """
    if args.grader is not None:
        return None if args.grader == "none" else args.grader
    name: str | None = defaults.grader
    if name is None:
        return "operator" if _attended(args) else None
    if name == "none":
        return None
    if name == "operator" and not _attended(args):
        print(
            "note: config grader 'operator' needs an attended terminal; "
            "grading is off for this run",
            file=sys.stderr,
        )
        return None
    return name


def _build_grader(
    args: argparse.Namespace, defaults: Defaults, session: OperatorSession | None
) -> Grader | None:
    """Construct the run's grader, sharing the operator session when one exists.

    Attendedness only picks the *default* name; an explicitly selected grader
    is built even without a session (its own fallback behavior then applies,
    e.g. the operator grader constructs a lazy session that degrades on dead
    stdin).
    """
    name = _select_grader_name(args, defaults)
    if name is None:
        return None
    grader = cast("Grader", _resolve_or_exit("grader", name))
    connect = getattr(grader, "connect_session", None)
    if session is not None and callable(connect):
        connect(session)
    return grader


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
    for raw_message in transcript:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message["role"])
        content = _chat_content(raw_message.get("content"))
        if role == "tool":
            suffix = "" if content is None else f" {content}"
            _print_degraded(f"        tool:{suffix}")
            continue
        suffix = "" if content is None else f" {content}"
        _print_degraded(f"    {role}:{suffix}")
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
            _print_degraded(f"      -> {name}({arguments})")


def _print_policy_transcripts(log: EvalLog) -> None:
    """Append each available policy audit record in a tolerant human-readable form."""
    if not _has_policy_transcripts(log):
        print("no policy transcripts recorded")
        return
    print("policy transcripts:")
    for scene in log.samples:
        for trial, transcript in enumerate(scene.policy_transcripts):
            if transcript is None:
                continue
            _print_degraded(f"scene {scene.scene_id}, trial {trial}:")
            if _is_chat_transcript(transcript):
                _render_chat_transcript(transcript)
                continue
            for line in json.dumps(transcript, indent=2).splitlines():
                _print_degraded(f"    {line}")


class _WireTrial(NamedTuple):
    """A readable wire sidecar and its trial/run filesystem context."""

    trial_id: str
    rows: list[dict[str, Any]]
    blob_dir: Path


_WIRE_BLOB_RE = re.compile(r"\$blob:([0-9a-f]{64})")


def _wire_blob_shas(value: object) -> list[str]:
    """Return every valid symbolic blob reference in a captured JSON value."""
    if isinstance(value, str):
        return [match.group(1) for match in _WIRE_BLOB_RE.finditer(value)]
    if isinstance(value, list):
        return [sha for item in value for sha in _wire_blob_shas(item)]
    if isinstance(value, dict):
        return [sha for item in value.values() for sha in _wire_blob_shas(item)]
    return []


def _load_wire_trials(log: EvalLog, log_path: Path) -> list[_WireTrial]:
    """Load every readable guarded wire sidecar referenced by an eval log."""
    trials: list[_WireTrial] = []
    for scene in log.samples:
        for epoch in range(len(scene.epochs)):
            if epoch >= len(scene.trial_metadata):
                continue
            target = resolve_log_pointer(log_path, scene.trial_metadata[epoch].get("wire_capture"))
            if target is None:
                continue
            loaded = read_jsonl_prefix(target)
            if loaded is None:
                continue
            blob_dir = derive_blob_dir(log_path, target)
            if blob_dir is None:
                continue
            trials.append(
                _WireTrial(
                    f"{scene.scene_id}-e{epoch}",
                    loaded,
                    blob_dir,
                )
            )
    return trials


def _wire_request(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row's request mapping, degrading malformed foreign data."""
    request = row.get("request")
    return cast(dict[str, Any], request) if isinstance(request, dict) else {}


def _print_wire_table(trials: list[_WireTrial]) -> None:
    """Print the compact all-trials wire attempt table."""
    if not trials:
        print("no wire capture recorded")
        return
    print("wire calls:")
    print("  trial  call  attempt  endpoint  status  duration  images  new blob bytes")
    for trial in trials:
        seen: set[str] = set()
        for row in trial.rows:
            shas = _wire_blob_shas(_wire_request(row))
            first_shas = set(shas).difference(seen)
            seen.update(shas)
            new_bytes = 0
            for sha in first_shas:
                with suppress(OSError):
                    new_bytes += (trial.blob_dir / f"{sha}.png").stat().st_size
            duration = row.get("duration_s")
            shown_duration = (
                f"{duration:.3f}s"
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else "-"
            )
            status = "-" if row.get("status") is None else str(row["status"])
            print(
                f"  {trial.trial_id}  {row.get('call', '-')}  "
                f"{row.get('attempt', '-')}  {row.get('endpoint', '-')}  "
                f"{status}  {shown_duration}  {len(shas)}  {new_bytes}"
            )


def _available_wire_trials(trials: list[_WireTrial]) -> str:
    """Format captured trial ids for a guided CLI error."""
    return ", ".join(trial.trial_id for trial in trials)


def _print_wire_call(trials: list[_WireTrial], call: int, selected_trial: str | None) -> None:
    """Pretty-print every captured attempt for one logical call index."""
    if not trials:
        raise SystemExit("no wire capture recorded; omit CALL to print an empty table")
    if selected_trial is None:
        if len(trials) > 1:
            raise SystemExit(
                "--trial is required for --wire CALL; available trials: "
                f"{_available_wire_trials(trials)}"
            )
        trial = trials[0]
    else:
        matches = [trial for trial in trials if trial.trial_id == selected_trial]
        if not matches:
            raise SystemExit(
                f"wire trial {selected_trial!r} not found; available trials: "
                f"{_available_wire_trials(trials)}"
            )
        trial = matches[0]

    rows = [row for row in trial.rows if row.get("call") == call]
    if not rows:
        raise SystemExit(f"wire call {call} not found in trial {trial.trial_id}")
    print(f"wire trial {trial.trial_id}, call {call}:")
    for row in rows:
        print(f"attempt {row.get('attempt', '-')}:")
        print(f"  endpoint: {row.get('endpoint', '-')}")
        print(f"  status: {row.get('status')}")
        print(f"  duration_s: {row.get('duration_s', '-')}")
        if "error" in row:
            _print_degraded(f"  error: {row['error']}")
        for label in ("request", "response"):
            _print_degraded(f"  {label}:")
            dumped = json.dumps(row.get(label), indent=2, sort_keys=True, ensure_ascii=False)
            for line in dumped.splitlines():
                _print_degraded(f"    {line}")


def _print_wire_capture(
    log: EvalLog,
    log_path: Path,
    wire: int | bool,
    selected_trial: str | None,
) -> None:
    """Print either the capture table or every attempt for one call."""
    trials = _load_wire_trials(log, log_path)
    if wire is True:
        if selected_trial is not None:
            raise SystemExit("--trial requires an integer --wire CALL")
        _print_wire_table(trials)
        return
    _print_wire_call(trials, wire, selected_trial)


def _print_run_summary(log: EvalLog, log_path: str, is_adhoc: bool) -> None:
    """Print the compact post-run summary and failure diagnostics."""
    failed = log.status != "success"
    errored_count = log.results.errored_trials
    status_color = _RED if failed else _GREEN
    print(f"{_styled('run status:', _CYAN)} {_styled(_display_status(log.status), status_color)}")
    outcome = _outcome_line(log)
    if outcome is not None:
        digest, has_unmapped = outcome
        line = f"{_styled('outcome:', _CYAN)} {digest}"
        if has_unmapped:
            _print_degraded(line)
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
    trials = f"trials: {log.results.total_trials}"
    if errored_count:
        trials += f" ({errored_count} errored)"
    print(f"{_styled('scenes:', _CYAN)} {log.results.total_scenes}  {trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {_styled(f'{value:.4g}', _BOLD)}")
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


class _ResolvedComponents(NamedTuple):
    """A resolved policy/embodiment pair plus the names/sources for the run header."""

    policy: Any
    policy_name: str
    policy_source: str
    embodiment: Any
    embodiment_name: str
    embodiment_source: str
    claim: DeviceClaim


def _check_shared_run_conflicts(args: argparse.Namespace) -> None:
    """Reject flag combinations and malformed values invalid for both ``run`` and ``eval-set``."""
    if args.sim and args.embodiment:
        raise SystemExit(
            "--sim selects your configured sim_embodiment; "
            "passing --embodiment already picks the embodiment — drop one"
        )
    if args.disable_guardrails and args.max_action_delta is not None:
        raise SystemExit(
            "--max-action-delta tunes the guardrails that --disable-guardrails turns off — drop one"
        )
    if args.no_prompt and args.grader == "operator":
        raise SystemExit(
            "--no-prompt suppresses the operator grader that --grader operator asks for — drop one"
        )
    if args.max_action_delta is not None and not (
        math.isfinite(args.max_action_delta) and args.max_action_delta > 0
    ):
        # A malformed *explicit* value is a CLI input error, not an embodiment
        # limitation: fail fast here rather than let _build_guardrails's
        # degrade-per-component path (meant for derived limits an embodiment's
        # space can't support) downgrade it to a warning and run with weaker
        # guardrails than the operator explicitly asked for.
        raise SystemExit(f"--max-action-delta must be finite and > 0, got {args.max_action_delta}")
    if args.fail_on_error is not None and not (
        math.isfinite(args.fail_on_error) and args.fail_on_error >= 0
    ):
        # Out-of-range values are not inert: a negative reaches
        # ``errors >= fail_on_error`` and silently halts on the first error like
        # ``1``, and a NaN fails every comparison and silently never halts.
        # 0 stays valid — it is the documented "never halt" value.
        raise SystemExit(f"--fail-on-error must be finite and >= 0, got {args.fail_on_error}")


def _resolve_components(args: argparse.Namespace, defaults: Defaults) -> _ResolvedComponents:
    """Pick and construct the policy/embodiment pair shared by ``run`` and ``eval-set``.

    The embodiment is constructed last, so callers can invoke this immediately
    before the ``try``/``finally`` that owns ``embodiment.close()`` and leave no
    window in which a resolved embodiment could leak past a later failure.
    """
    policy_name, policy_source = _pick_component(
        "policy", args.policy, defaults.policy, defaults.policy_source
    )
    if args.sim:
        embodiment_name, embodiment_source = _pick_sim_embodiment(defaults)
        embodiment_defaults = _config_args(
            "sim_embodiment",
            embodiment_name,
            defaults.sim_embodiment_args_owner,
            defaults.sim_embodiment_args,
        )
    else:
        embodiment_name, embodiment_source = _pick_component(
            "embodiment", args.embodiment, defaults.embodiment, defaults.embodiment_source
        )
        embodiment_defaults = _config_args(
            "embodiment", embodiment_name, defaults.embodiment_args_owner, defaults.embodiment_args
        )
    # Config-file args apply only to the component they were configured
    # alongside (issue #44); explicit -P/-E flags override same-named keys.
    policy_config_args = _config_args(
        "policy", policy_name, defaults.policy_args_owner, defaults.policy_args
    )
    policy_kvs = {**policy_config_args, **_parse_kvs(args.policy_args)}
    embodiment_kvs = {**embodiment_defaults, **_parse_kvs(args.embodiment_args)}

    policy = _resolve_or_exit("policy", policy_name, **policy_kvs)
    factories = registered("embodiment")
    slots = device_slots(factories[embodiment_name]) if embodiment_name in factories else ()
    claim = claim_devices(slots, embodiment_kvs, os.environ)
    try:
        if args.sim:
            embodiment = _resolve_or_exit(
                "embodiment", embodiment_name, "sim_embodiment.args", **embodiment_kvs
            )
        else:
            embodiment = _resolve_or_exit("embodiment", embodiment_name, **embodiment_kvs)
    except BaseException:
        claim.release()
        raise
    return _ResolvedComponents(
        policy, policy_name, policy_source, embodiment, embodiment_name, embodiment_source, claim
    )


def _announce_components(resolved: _ResolvedComponents) -> None:
    """Print the resolved policy/embodiment and where each came from.

    Defaults must never be silent: say what runs, and why, before it moves.
    """
    print(f"policy: {resolved.policy_name} ({resolved.policy_source})")
    print(f"embodiment: {resolved.embodiment_name} ({resolved.embodiment_source})")


def _build_and_announce_guardrails(
    args: argparse.Namespace, action_space: Box, embodiment: Embodiment
) -> Approver | None:
    """Build the default guardrail chain for a run, announcing what is active.

    Guardrails are on by default (plan 0008 §3e): the approver chain sits below
    the policy in rollout, so nothing the policy emits — a wild VLA action or a
    misbehaving LLM — reaches hardware unchecked. Returns ``None`` (the eval's
    own default) only when ``--disable-guardrails`` is given.
    """
    if args.disable_guardrails:
        print(
            "WARNING: guardrails disabled (--disable-guardrails); actions "
            "reach the embodiment unchecked.",
            file=sys.stderr,
        )
        print("guardrails: disabled (--disable-guardrails)")
        return None
    approver, active, guard_warnings = _build_guardrails(
        action_space, args.max_action_delta, embodiment
    )
    for warning in guard_warnings:
        print(f"guardrails warning: {warning}", file=sys.stderr)
    print(f"guardrails: {' + '.join(active) if active else 'none active'}")
    return approver


def _headless_session(env: Mapping[str, str], platform: str = sys.platform) -> bool:
    """Return whether the operator is expected to be remote from this process."""
    # _setup.py asks if a viewer window can open here; this asks if the human is elsewhere.
    if env.get("SSH_CONNECTION"):
        return True
    return platform.startswith("linux") and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def _announce_live_view(
    args: argparse.Namespace,
    resolved: _ResolvedComponents,
    env: Mapping[str, str] = os.environ,
) -> None:
    """Print the copyable browser-view command for agent runs with live logging."""
    if resolved.policy_name != "agent" or args.no_live_log:
        return
    log_dir = shlex.quote(args.log_dir)
    headless = _headless_session(env)
    command_tail = "--serve --host 0.0.0.0" if headless else "--serve --open"
    print(
        _styled(
            f"live: watch this run in your browser →  inspect-robots view {log_dir} {command_tail}",
            _BOLD_BRIGHT_MAGENTA,
        )
    )
    url = ""
    if headless:
        fields = env.get("SSH_CONNECTION", "").split()
        host = fields[2] if len(fields) == 4 else socket.gethostname()
        if ":" in host and not (host.startswith("[") and host.endswith("]")):
            host = f"[{host}]"
        url = f"; open http://{host}:8300/"
    print(
        _styled(
            "      each agent turn, notes, and operator/voice input, updating live" + url,
            _DIM,
        )
    )


def _cmd_run(args: argparse.Namespace) -> int:
    if args.rerun_port is not None and args.rerun_connect is not None:
        raise SystemExit(
            "--rerun-port spawns a local viewer and --rerun-connect streams "
            "to a remote one: pass only one"
        )
    if args.rerun_port is not None and args.rerun is False:
        raise SystemExit(
            "--no-rerun disables the live viewer and --rerun-port requests one: pass only one"
        )

    from dataclasses import replace

    from inspect_robots import eval
    from inspect_robots.logging import JsonLogSink, LiveLogSink
    from inspect_robots.scene import Scene
    from inspect_robots.task import Task

    is_adhoc = args.instruction is not None
    if is_adhoc and args.task:
        raise SystemExit("pass exactly one of --task or --instruction, not both")
    if not is_adhoc and not args.task:
        raise SystemExit("pass a registered --task name or an --instruction to run")
    if not is_adhoc:
        if args.max_steps is not None:
            raise SystemExit(
                "--max-steps only applies to --instruction runs; a registered task owns its horizon"
            )
        if args.scorer is not None:
            raise SystemExit(
                "--scorer only applies to --instruction runs; a registered task owns its scorers"
            )
    elif args.task_args:
        raise SystemExit(
            "-T only applies to --task runs; an ad-hoc instruction task takes no constructor args"
        )
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit(f"--max-steps must be >= 1, got {args.max_steps}")
    _check_shared_run_conflicts(args)

    defaults = load_defaults(os.environ)

    if is_adhoc:
        scorer_name = args.scorer or defaults.scorer or _ADHOC_SCORER_FALLBACK
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else (defaults.max_steps or _ADHOC_MAX_STEPS_FALLBACK)
        )
        task = Task(
            name="adhoc",
            scenes=[Scene(id="scene-0", instruction=args.instruction)],
            scorer=_resolve_or_exit("scorer", scorer_name),
            max_steps=max_steps,
            metadata={"instruction": args.instruction, "adhoc": True},
        )
    else:
        task = _resolve_or_exit("task", args.task, **_parse_kvs(args.task_args))

    resolved = _resolve_components(args, defaults)
    embodiment = resolved.embodiment
    voice_input: OperatorInput | None = None
    speaker_sink: LogSink | None = None
    live_sink: LiveLogSink | None = None
    rerun_sink: LogSink | None = None
    try:
        if args.epochs is not None:
            from inspect_robots.errors import ConfigError

            try:
                task = replace(task, epochs=args.epochs)
            except ConfigError as exc:
                raise SystemExit(f"--epochs: {exc}") from exc

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
        # Attendedness picks the default grader, never gates grader wiring
        # (plan 0049): an explicit --grader is built session-less and relies
        # on its own fallback.
        grader = _build_grader(args, defaults, operator_session)

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
            )
        except KeyboardInterrupt:
            if sink.path is not None and sink.path.exists():
                _print_degraded(f"cancelled: partial log written to {sink.path}")
                print(_styled(f"hint: inspect it with: inspect-robots inspect {sink.path}", _DIM))
            else:
                _print_degraded("cancelled: no log written")
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
            if live_sink is not None and live_sink.path is not None:
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
            f"{name}={value:.4g}" for name, value in sorted(log.results.metrics.items())
        )
        detail = metrics or (log.error or "")
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


def _cmd_eval_set(args: argparse.Namespace) -> int:
    """Resolve one policy/embodiment once, then drive every matched task through it.

    A thin wrapper over [`eval_set`][inspect_robots.eval.eval_set]: unlike
    calling ``eval_set()`` with string components (which resolves and closes
    the embodiment once per task), the CLI resolves the embodiment exactly
    once for the whole set, so a real robot is not reconnected between tasks.
    """
    from dataclasses import replace

    from inspect_robots import eval_set
    from inspect_robots.logging import JsonLogSink, LiveLogSink

    _check_shared_run_conflicts(args)
    task_names = _match_tasks(args.tasks)

    defaults = load_defaults(os.environ)
    tasks = [_resolve_or_exit("task", name) for name in task_names]
    if args.epochs is not None:
        from inspect_robots.errors import ConfigError
        from inspect_robots.task import Task

        patched: list[Task] = []
        for t in tasks:
            try:
                patched.append(replace(t, epochs=args.epochs))
            except ConfigError as exc:
                raise SystemExit(f"--epochs (task {t.name!r}): {exc}") from exc
        tasks = patched

    resolved = _resolve_components(args, defaults)
    embodiment = resolved.embodiment
    voice_input: OperatorInput | None = None
    live_sink: LiveLogSink | None = None
    try:
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
        if voice_input is not None:
            _start_voice_input(
                voice_input,
                cast(OperatorSession, operator_input),
                resolved.policy,
            )
        # Attendedness picks the default grader, never gates grader wiring
        # (plan 0049): an explicit --grader is built session-less and relies
        # on its own fallback.
        grader = _build_grader(args, defaults, operator_session)
        sink = JsonLogSink(args.log_dir)
        sinks: list[LogSink] = [sink]
        if not args.no_live_log:
            live_sink = LiveLogSink(args.log_dir)
            sinks.append(live_sink)
        try:
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
            )
        except KeyboardInterrupt:
            # eval_set writes one log per task; eval() persists a cancelled log
            # for the interrupted task before re-raising (#118). We don't hold
            # the per-task sink paths, so point at the shared dir. The finally
            # below still de-energizes the arm.
            _print_degraded(f"cancelled: partial logs are under {args.log_dir}")
            print(
                _styled(
                    f"hint: inspect a log with: inspect-robots inspect "
                    f"{args.log_dir}/<task>_<id>.json",
                    _DIM,
                )
            )
            print(
                _styled(
                    f"hint: browse all logs: inspect-robots view {args.log_dir}",
                    _DIM,
                )
            )
            return 130
    finally:
        # Same "close what we open" contract as _cmd_run: the CLI resolved the
        # embodiment itself, so it — not eval_set() — is responsible for
        # releasing it, exactly once, after every task has run.
        try:
            # A failing unlink must not skip the close chain (see _cmd_run).
            if live_sink is not None and live_sink.path is not None:
                with suppress(OSError):
                    live_sink.path.unlink(missing_ok=True)
        finally:
            try:
                _close_voice_input(voice_input)
            finally:
                try:
                    embodiment.close()
                finally:
                    resolved.claim.release()
    _print_eval_set_summary(success, logs, args.log_dir)
    return 0 if success else 1


def _cmd_inspect(
    path: str,
    *,
    transcript: bool = False,
    wire: int | bool | None = None,
    trial: str | None = None,
) -> int:
    from inspect_robots import read_eval_log

    log = read_eval_log(path)
    _print_step_limit_notice(log, log.eval.task == "adhoc")
    print(f"task:        {log.eval.task}")
    # One shared instruction (the adhoc case) reads as run-level identity;
    # differing instructions print per scene below instead. Instructions are
    # foreign text (dataset/operator-supplied), so printing degrades.
    instructions = {scene.instruction for scene in log.samples}
    shared = next(iter(instructions)) if len(instructions) == 1 else None
    if shared:
        _print_degraded(f"instruction: {shared}")
    print(f"policy:      {log.eval.policy}")
    print(f"embodiment:  {log.eval.embodiment}")
    print(f"run status:  {_display_status(log.status)}")
    outcome = _outcome_line(log)
    if outcome is not None:
        digest, has_unmapped = outcome
        line = f"outcome:     {digest}"
        if has_unmapped:
            _print_degraded(line)
        else:
            print(line)
    print(f"created:     {log.eval.created}")
    print(f"git:         {log.eval.git_commit}")
    horizon = _seconds_horizon_text(log)
    if horizon is not None:
        print(f"horizon:     {horizon}")
    trials = f"trials: {log.results.total_trials}"
    if log.results.errored_trials:
        trials += f" ({log.results.errored_trials} errored)"
    print(f"scenes:      {log.results.total_scenes}   {trials}")
    if log.stats.frames_dir is not None:
        from inspect_robots._video import count_frames, resolve_frames_dir

        root = resolve_frames_dir(log.stats.frames_dir, Path(path))
        if root is None:
            print(f"frames:      {log.stats.frames_dir} (not found from this directory)")
        else:
            # The resolved path, not the stored string: after a machine move
            # the stored string is exactly the path that does not work.
            n_frames = count_frames(root)
            plural = "frame" if n_frames == 1 else "frames"
            print(f"frames:      {root} ({n_frames} {plural})")
            if n_frames:
                print(_styled(f"hint: render videos with: inspect-robots video {path}", _DIM))
    print("metrics:")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {'n/a' if value is None else f'{value:.4g}'}")
    print("scenes:")
    for scene in log.samples:
        reduced = "  ".join(
            f"{k}={'n/a' if v is None else f'{v:.4g}'}" for k, v in sorted(scene.reduced.items())
        )
        step_limit_count = sum(reason == "max_steps" for reason in scene.termination_reasons)
        details = [reduced] if reduced else []
        if step_limit_count:
            details.append(f"({step_limit_count}/{len(scene.epochs)} trials hit max_steps)")
        print(f"  [{scene.status}] {scene.scene_id}: {'  '.join(details)}")
        if not shared and scene.instruction:
            _print_degraded(f"      instruction: {scene.instruction}")
    if log.error:
        print(f"error: {log.error}")
    if transcript:
        _print_policy_transcripts(log)
    elif _has_policy_transcripts(log):
        print("policy transcripts: recorded (--transcript to print)")
        print(_styled(f"hint: HTML viewer: inspect-robots view {path}", _DIM))
    if wire is not None:
        _print_wire_capture(log, Path(path), wire, trial)
    elif trial is not None:
        raise SystemExit("--trial requires --wire CALL")
    return 0 if log.status == "success" else 1


def _write_html(document: str, out_path: Path | None) -> int:
    """Write one document as UTF-8, returning the encoded byte count."""
    encoded = document.encode("utf-8", errors="replace")
    if out_path is None:
        sys.stdout.write(encoded.decode("utf-8"))
        return len(encoded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    return len(encoded)


def _write_html_atomic(document: str, out_path: Path) -> int:
    """Replace one UTF-8 document atomically and return its encoded byte count."""
    encoded = document.encode("utf-8", errors="replace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f"{out_path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    os.replace(tmp, out_path)
    return len(encoded)


def _render_log_page(
    log: EvalLog,
    log_path: Path,
    out_path: Path | None,
    *,
    no_frames: bool,
    frames_budget: float,
    live_frames_budget: float | None = None,
    refresh_seconds: int | None = None,
    atomic: bool = False,
    no_video: bool = False,
    serve_pass: bool = False,
) -> int:
    """Render one already-parsed log through the shared single-page pipeline."""
    frames_dir = None
    if not no_frames and log.stats.frames_dir is not None:
        from inspect_robots._video import resolve_frames_dir

        frames_dir = resolve_frames_dir(log.stats.frames_dir, log_path)
    document = render_html(
        log,
        title=f"{log.eval.task} - {log_path.name}",
        log_path=log_path,
        frames_dir=frames_dir,
        frames_budget_bytes=int(frames_budget * 1_000_000),
        live_frames_budget_bytes=(
            None if live_frames_budget is None else int(live_frames_budget * 1_000_000)
        ),
        wire_media_elided=log.status == "started",
        refresh_seconds=refresh_seconds,
        no_video=no_video,
        serve_pass=serve_pass,
    )
    if atomic and out_path is not None:
        return _write_html_atomic(document, out_path)
    return _write_html(document, out_path)


def _open_browser(uri: str, display: str | Path) -> None:
    """Open one rendered artifact, degrading browser failures to warnings."""
    import webbrowser

    try:
        opened = webbrowser.open(uri)
    except Exception as exc:
        print(f"warning: could not open browser for {display}: {exc}", file=sys.stderr)
    else:
        if not opened:
            print(f"warning: could not open browser for {display}", file=sys.stderr)


def _index_instruction(log: EvalLog) -> str:
    """Return the shared instruction or the multi-scene fallback."""
    instructions = {scene.instruction for scene in log.samples}
    shared = next(iter(instructions)) if len(instructions) == 1 else None
    if shared:
        return shared
    count = len(log.samples)
    return f"{count} scene" if count == 1 else f"{count} scenes"


def _index_termination(log: EvalLog) -> str:
    """Return the order-preserved union of human-facing termination reasons."""
    seen: set[str] = set()
    phrases: list[str] = []
    for scene in log.samples:
        for reason in scene.termination_reasons:
            if reason is None:
                continue
            text = str(reason)
            if text in seen:
                continue
            seen.add(text)
            phrases.append(_OUTCOME_PHRASES.get(text, text))
    return ", ".join(phrases)


def _index_entry(log: EvalLog, log_path: Path, page: str) -> IndexEntry:
    """Build one directory-index row from a successfully parsed log."""
    model = log.eval.policy_config.get("model")
    return IndexEntry(
        name=log_path.name,
        page=page,
        created=log.eval.created,
        instruction=_index_instruction(log),
        policy=log.eval.policy,
        model=None if model is None else str(model),
        status=_display_status(log.status),
        status_class=_status_class(log.status),
        metrics=log.results.metrics,
        errored_trials=log.results.errored_trials,
        termination=_index_termination(log),
        error=log.error,
    )


def _unreadable_index_entry(log_path: Path, error: Exception) -> IndexEntry:
    """Build an error row for a file that is not a readable EvalLog."""
    try:
        created = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        created = ""
    return IndexEntry(
        name=log_path.name,
        page=None,
        created=created,
        instruction="",
        policy="",
        model=None,
        status="error",
        status_class="status-error",
        metrics={},
        errored_trials=0,
        termination="",
        error=f"unreadable: {error}",
    )


def _directory_page_names(log_paths: Sequence[Path]) -> dict[Path, str]:
    """Assign collision-free report names without consulting existing files."""
    names = {path: f"{path.stem}.html" for path in log_paths if path.stem != "index"}
    reserved = set(names.values())
    for path in log_paths:
        if path.stem != "index":
            continue
        candidate = "index_log.html"
        suffix = 2
        while candidate in reserved:
            candidate = f"index_log_{suffix}.html"
            suffix += 1
        names[path] = candidate
        reserved.add(candidate)
    return names


class _DirectoryRenderResult(NamedTuple):
    """Paths produced by one logs-directory render pass."""

    out_dir: Path
    index_path: Path


_LIVE_REDIRECT_SENTINEL = "<!-- inspect-robots live redirect stub -->"


def _write_live_redirect_stub(path: Path) -> int:
    """Atomically replace an orphaned live page with an idempotent index redirect."""
    document = f"""{_LIVE_REDIRECT_SENTINEL}
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=index.html">
<title>Run completed</title></head>
<body><a href="index.html">Continue to the run index</a></body></html>
"""
    return _write_html_atomic(document, path)


def _render_view_directory(
    args: argparse.Namespace,
    log_dir: Path,
    *,
    force: bool,
    quiet: bool,
    refresh_seconds: int | None,
) -> _DirectoryRenderResult:
    """Render one incremental logs-directory pass and return its output paths."""
    from inspect_robots import read_eval_log

    if args.out == "-":
        raise SystemExit("-o - cannot be used with a logs directory; pass an output directory")

    log_paths = sorted(path for path in log_dir.glob("*.json") if path.is_file())
    if not log_paths and not args.serve:
        raise SystemExit(f"no top-level *.json logs found in {log_dir}")

    out_dir = log_dir / "html" if args.out is None else Path(args.out)
    if (out_dir.exists() or out_dir.is_symlink()) and not out_dir.is_dir():
        if args.out is not None:
            raise SystemExit(f"--out {out_dir} is not a directory; pass an output directory")
        raise SystemExit(
            f"default output path {out_dir} exists and is not a directory; move it or pass -o DIR"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    page_names = _directory_page_names(log_paths)
    # Derived from this pass's own glob, not the caller's earlier one: a live
    # log appearing between the two globs must not have its freshly rendered
    # page immediately stubbed as an orphan.
    backed_live_pages = {page_names[path] for path in log_paths if path.name.endswith(".live.json")}
    entries: list[IndexEntry] = []
    pages_written = 0
    bytes_written = 0
    total = len(log_paths)
    # One pass, one parse per log, nothing retained beyond its row: a log that
    # fails to read or render (e.g. replaced mid-run by a concurrent writer)
    # becomes an error row instead of sinking the index.
    for index, log_path in enumerate(log_paths, start=1):
        page_name = page_names[log_path]
        page_path = out_dir / page_name
        is_live_path = log_path.name.endswith(".live.json")
        try:
            source_stat = log_path.stat()
            log = read_eval_log(str(log_path))
            suppressed_tier = args.serve or args.no_video
            # The suppressed-tier stamp sits a full 2 microseconds below the
            # source mtime, and the gate allows 1 microsecond of slack:
            # filesystems round timestamps to their tick (100 ns on NTFS), so
            # a 1 ns delta would floor below its own target and re-render
            # suppressed pages on every serve tick on Windows.
            stamp_ns = max(
                0,
                source_stat.st_mtime_ns - (_SUPPRESSED_STAMP_DELTA_NS if suppressed_tier else 0),
            )
            render_page = (
                force
                or log.status == "started"
                or not page_path.exists()
                or page_path.stat().st_mtime_ns < stamp_ns - _STAMP_TICK_SLACK_NS
            )
            if render_page:
                if not quiet:
                    print(f"[{index}/{total}] rendering {log_path.name}", file=sys.stderr)
                bytes_written += _render_log_page(
                    log,
                    log_path,
                    page_path,
                    no_frames=args.no_frames,
                    frames_budget=args.frames_budget,
                    live_frames_budget=(
                        args.live_frames_budget if args.serve and log.status == "started" else None
                    ),
                    refresh_seconds=(
                        _SERVE_LIVE_RERENDER_SECONDS
                        if args.serve and log.status == "started"
                        else None
                    ),
                    atomic=log.status == "started",
                    no_video=args.no_video,
                    serve_pass=args.serve,
                )
                if log.status != "started":
                    # Started pages re-render every pass by design (plan 0058);
                    # the two-level stamp is a completed-page contract only.
                    os.utime(
                        page_path,
                        ns=(source_stat.st_atime_ns, stamp_ns),
                    )
                pages_written += 1
            entries.append(_index_entry(log, log_path, page_name))
        except FileNotFoundError:
            if is_live_path:
                continue
            exc = FileNotFoundError(log_path)
            print(f"warning: could not read or render {log_path.name}: {exc}", file=sys.stderr)
            entries.append(_unreadable_index_entry(log_path, exc))
        except Exception as exc:
            print(f"warning: could not read or render {log_path.name}: {exc}", file=sys.stderr)
            entries.append(_unreadable_index_entry(log_path, exc))

    for live_page in out_dir.glob("*.live.html"):
        if live_page.name in backed_live_pages:
            continue
        try:
            with live_page.open(encoding="utf-8", errors="replace") as handle:
                if handle.readline().rstrip("\r\n") == _LIVE_REDIRECT_SENTINEL:
                    continue
            bytes_written += _write_live_redirect_stub(live_page)
        except OSError as exc:
            print(f"warning: could not redirect {live_page.name}: {exc}", file=sys.stderr)

    index_path = out_dir / "index.html"
    bytes_written += _write_html_atomic(
        render_index(entries, refresh_seconds=refresh_seconds),
        index_path,
    )
    if not quiet:
        print(
            f"index: {index_path} "
            f"({total} logs, {pages_written} pages, {bytes_written / 1_000_000:.1f} MB)"
        )
    return _DirectoryRenderResult(out_dir=out_dir, index_path=index_path)


class _QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static reports without per-request stderr logging."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress the standard request log."""
        return None


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _serve_display_urls(bind_host: str, port: int) -> tuple[str, ...]:
    """Return the URLs shown for one bound address and socket-derived port."""
    display_host = socket.gethostname() if bind_host == "0.0.0.0" else bind_host
    primary = f"http://{display_host}:{port}/"
    if bind_host == "0.0.0.0":
        return primary, f"http://localhost:{port}/"
    return (primary,)


def _serve_open_url(bind_host: str, port: int) -> str:
    """Return the reachable URL that ``--open`` should target."""
    open_host = "127.0.0.1" if bind_host in _LOOPBACK_HOSTS | {"0.0.0.0"} else bind_host
    return f"http://{open_host}:{port}/"


def _raise_keyboard_interrupt(_signum: int, _frame: FrameType | None) -> NoReturn:
    """Map a termination signal onto the normal Ctrl-C shutdown path."""
    raise KeyboardInterrupt


def _serve_view_directory(
    args: argparse.Namespace,
    log_dir: Path,
    render_result: _DirectoryRenderResult,
) -> int:
    """Serve rendered logs and refresh the artifacts incrementally until stopped."""
    bind_host = cast(str, args.host)
    port = cast(int, args.port)
    handler = partial(_QuietHTTPRequestHandler, directory=str(render_result.out_dir))
    try:
        server = ThreadingHTTPServer((bind_host, port), handler)
    except OSError as exc:
        raise SystemExit(str(exc)) from exc

    actual_port = int(server.server_address[1])
    urls = _serve_display_urls(bind_host, actual_port)
    # ThreadingHTTPServer.__init__ has already bound and started listening, so
    # printing the URLs and opening one before serve_forever cannot race a request.
    print(f"serving logs at: {urls[0]}")
    for url in urls[1:]:
        print(f"                 {url}")
    if bind_host in _LOOPBACK_HOSTS:
        print(
            _styled(
                "serving to this machine only; pass --host 0.0.0.0 to serve to your network",
                _DIM,
            )
        )
    else:
        print(
            _styled(
                "serving to your network: anyone who can reach this machine can view these logs",
                _YELLOW,
            )
        )
    print("press Ctrl-C to stop")

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    sigterm_installed = False
    server_thread_started = False
    previous_live_set: set[str] = set()
    accumulated_seconds = 0
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        sigterm_installed = True
        server_thread.start()
        server_thread_started = True
        if args.open:
            open_url = _serve_open_url(bind_host, actual_port)
            _open_browser(open_url, open_url)
        while True:
            _serve_sleep(_SERVE_LIVE_RERENDER_SECONDS)
            accumulated_seconds += _SERVE_LIVE_RERENDER_SECONDS
            live_set = {path.name for path in log_dir.glob("*.live.json")}
            if not (
                live_set != previous_live_set
                or live_set
                or accumulated_seconds >= _SERVE_RERENDER_SECONDS
            ):
                continue
            try:
                _render_view_directory(
                    args,
                    log_dir,
                    force=False,
                    quiet=True,
                    refresh_seconds=(
                        _SERVE_LIVE_RERENDER_SECONDS if live_set else _SERVE_RERENDER_SECONDS
                    ),
                )
            except (Exception, SystemExit) as exc:
                print(f"warning: re-render failed: {exc}", file=sys.stderr)
                # Reset the accumulator on failure too: an idle directory with
                # a persistent render error must warn once per baseline period,
                # not once per 2s tick.
                accumulated_seconds = 0
            else:
                previous_live_set = live_set
                accumulated_seconds = 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            # shutdown() waits on an event only serve_forever sets; calling it
            # when the serve thread never started would hang forever.
            if server_thread_started:
                server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                if sigterm_installed:
                    # A non-Python-installed prior handler reads back as None,
                    # which signal.signal rejects; SIG_DFL is the sane restore.
                    restore = signal.SIG_DFL if previous_sigterm is None else previous_sigterm
                    signal.signal(signal.SIGTERM, restore)


def _cmd_view_directory(args: argparse.Namespace, log_dir: Path) -> int:
    """Render a logs directory and optionally serve it until stopped."""
    live_set = {path.name for path in log_dir.glob("*.live.json")}
    render_result = _render_view_directory(
        args,
        log_dir,
        force=args.force,
        quiet=False,
        refresh_seconds=(
            (_SERVE_LIVE_RERENDER_SECONDS if live_set else _SERVE_RERENDER_SECONDS)
            if args.serve
            else None
        ),
    )
    if args.serve:
        return _serve_view_directory(args, log_dir, render_result)
    if args.open:
        index_uri = render_result.index_path.resolve().as_uri()
        _open_browser(index_uri, render_result.index_path)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Render one saved log or a logs directory to UTF-8 HTML artifacts."""
    if args.port is not None and not args.serve:
        raise SystemExit("--port requires --serve")
    if args.host is not None and not args.serve:
        raise SystemExit("--host requires --serve")
    if args.port is not None and not (0 <= args.port <= 65535):
        raise SystemExit(f"--port must be between 0 and 65535, got {args.port}")
    args.port = 8300 if args.port is None else args.port
    args.host = "127.0.0.1" if args.host is None else args.host

    if not (math.isfinite(args.frames_budget) and args.frames_budget >= 0):
        raise SystemExit("--frames-budget must be a non-negative finite number")
    if not (math.isfinite(args.live_frames_budget) and args.live_frames_budget >= 0):
        raise SystemExit("--live-frames-budget must be a non-negative finite number")

    log_path = Path(args.log)
    if args.serve and not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    if log_path.is_dir():
        return _cmd_view_directory(args, log_path)
    if args.serve:
        raise SystemExit("--serve requires a logs directory")

    stdout_mode = args.out == "-"
    if stdout_mode and args.open:
        raise SystemExit("--open cannot be used with -o -: no file to open")
    out_path = (
        None
        if stdout_mode
        else (log_path.with_suffix(".html") if args.out is None else Path(args.out))
    )
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out {out_path} is a directory; pass an HTML file path")
    if out_path is not None and out_path.resolve() == log_path.resolve():
        # The one data-loss path in this command: rendering over the log it reads.
        raise SystemExit(f"--out {out_path} would overwrite the input log; pass a different path")

    from inspect_robots import read_eval_log

    document_size = _render_log_page(
        read_eval_log(str(log_path)),
        log_path,
        out_path,
        no_frames=args.no_frames,
        frames_budget=args.frames_budget,
        no_video=args.no_video,
    )
    if stdout_mode:
        return 0

    file_path = cast(Path, out_path)
    size_suffix = f" ({document_size / 1_000_000:.1f} MB)" if document_size > 1_000_000 else ""
    print(f"wrote {file_path}{size_suffix}")
    if args.open:
        _open_browser(file_path.resolve().as_uri(), file_path)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    """Distill a saved log and atomically write its markdown learnings artifact."""
    from inspect_robots._summarize import summarize
    from inspect_robots.errors import ConfigError

    stdout_mode = args.out == "-"
    log_path = Path(args.log)
    out_path = (
        None
        if stdout_mode
        else (
            log_path.parent / "learnings" / f"{log_path.stem}.md"
            if args.out is None
            else Path(args.out)
        )
    )
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out {out_path} is a directory; pass a Markdown file path")
    if out_path is not None and out_path.resolve() == log_path.resolve():
        # The one data-loss path in this command: rendering over the log it reads.
        raise SystemExit(f"--out {out_path} would overwrite the input log; pass a different path")

    try:
        document = summarize(
            log_path,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if stdout_mode:
        sys.stdout.write(document)
        return 0

    file_path = cast(Path, out_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(document)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, file_path)
    print(f"wrote {file_path}")
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    """Render a log's stored frames to one MP4 per (trial, camera) stream.

    Per-stream failures are isolated: each is reported on stderr, the
    remaining streams still encode, and the exit code is 1 if any failed.
    Results (``wrote ...`` lines, the fps note, the final summary) go to
    stdout; warnings and failure reports go to stderr.
    """
    from inspect_robots import read_eval_log
    from inspect_robots._video import (
        default_fps,
        discover_streams,
        encode_stream,
        frames_dir_candidates,
        resolve_frames_dir,
    )

    log = read_eval_log(args.log)
    frames_dir = log.stats.frames_dir
    if frames_dir is None:
        raise SystemExit("this log has no stored frames (re-run with --store-frames)")
    log_path = Path(args.log)
    root = resolve_frames_dir(frames_dir, log_path)
    if root is None:
        as_is, fallback = frames_dir_candidates(frames_dir, log_path)
        raise SystemExit(f"frames directory not found; tried {as_is} and {fallback}")

    streams, strays = discover_streams(root)
    for stray in strays:
        print(
            f"warning: skipping {stray.name}: does not match the frame filename pattern",
            file=sys.stderr,
        )
    if not streams:
        raise SystemExit(f"no frames found in {root}")

    if args.fps is not None:
        if not (math.isfinite(args.fps) and args.fps > 0):
            raise SystemExit("--fps must be a positive finite number")
        fps, fps_source = args.fps, "--fps"
    else:
        fps, fps_source = default_fps(log.eval.embodiment_info)

    if args.ffmpeg is not None:
        if not (os.path.isfile(args.ffmpeg) and os.access(args.ffmpeg, os.X_OK)):
            raise SystemExit(f"--ffmpeg {args.ffmpeg} is not an executable file")
        ffmpeg = args.ffmpeg
    else:
        which = shutil.which("ffmpeg")
        if which is None:
            raise SystemExit(
                "ffmpeg not found on PATH; install it (e.g. apt install ffmpeg) "
                "or pass --ffmpeg PATH"
            )
        ffmpeg = which

    out_dir = root if args.out is None else Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"--out {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    # All validation is done: only now may result-looking stdout appear.
    print(f"fps: {fps:g} ({fps_source})")
    failed = 0
    for prefix, frames in streams.items():
        out_path = out_dir / f"{prefix}.mp4"
        result = encode_stream(frames, out_path, fps, ffmpeg)
        if result.skipped_empty:
            plural = "frame" if result.skipped_empty == 1 else "frames"
            print(
                f"warning: {prefix}: skipped {result.skipped_empty} empty {plural}",
                file=sys.stderr,
            )
        if result.error is not None:
            failed += 1
            print(f"failed: {prefix}: {result.error}", file=sys.stderr)
        else:
            plural = "frame" if result.piped == 1 else "frames"
            print(f"wrote {out_path} ({result.piped} {plural})")
    total = len(streams)
    summary = f"wrote {total - failed}/{total} streams"
    if failed:
        summary += f", {failed} failed"
    print(summary)
    return 1 if failed else 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Preflight runtime requirements and conformance for an installed adapter.

    Purely declarative — the embodiment is constructed (adapters keep
    constructors hardware-free by convention) but never reset or stepped.
    """
    from inspect_robots.conformance import check_embodiment, missing_runtime_requirements
    from inspect_robots.registry import registered

    defaults = load_defaults(os.environ)
    name, source = _pick_component(
        "embodiment", args.embodiment, defaults.embodiment, defaults.embodiment_source
    )
    config_kvs = _config_args(
        "embodiment", name, defaults.embodiment_args_owner, defaults.embodiment_args
    )
    kvs = {**config_kvs, **_parse_kvs(args.embodiment_args)}
    print(f"embodiment: {name} ({source})")
    missing = missing_runtime_requirements(registered("embodiment").get(name))
    for module, remedy in missing.items():
        print(f"  [error] runtime-requirement: {module} missing → {remedy}")
    embodiment = _resolve_or_exit("embodiment", name, **kvs)
    try:
        report = check_embodiment(embodiment.info)
    finally:
        embodiment.close()
    print(report.summary())
    if not report.ok:
        print("see the adapter authoring guide: docs/guide/adapters.md")
    return 1 if not report.ok or missing else 0


def _cmd_setup() -> int:
    from inspect_robots._setup import run_setup

    return run_setup(
        os.environ,
        input_fn=input,
        out=sys.stdout,
        interactive=sys.stdin.isatty(),
    )


def _cmd_config(args: argparse.Namespace) -> int:
    if args.config_command == "set":
        path = _set_default(os.environ, args.key, args.value)
        print(f"wrote {args.key} = {args.value} to {path}")
        return 0
    defaults = load_defaults(os.environ)
    rows: list[tuple[str, object, str | None]] = [
        ("policy", defaults.policy, defaults.policy_source),
        ("embodiment", defaults.embodiment, defaults.embodiment_source),
        ("sim_embodiment", defaults.sim_embodiment, defaults.sim_embodiment_source),
        ("scorer", defaults.scorer, None),
        ("max_steps", defaults.max_steps, None),
        ("store_frames", defaults.store_frames, None),
        ("rerun", defaults.rerun, None),
        ("rerun_save", defaults.rerun_save, None),
        ("rerun_port", defaults.rerun_port, None),
    ]
    for key, value, source in rows:
        shown = "(unset)" if value is None else value
        suffix = f"  ({source})" if source else ""
        print(f"{key}: {shown}{suffix}")
    return 0


def _apply_instruction_sugar(argv: list[str]) -> list[str]:
    """``inspect-robots "wipe the table"`` → ``run --instruction "wipe the table"``.

    Fires only for a first argument that is not a subcommand or flag AND has
    interior whitespace after stripping: a mistyped subcommand
    (``inspect-robots isnpect``) or a whitespace-padded one
    (``inspect-robots " list "``) must never silently start a rollout.
    """
    if not argv:
        return argv
    tok = argv[0].strip()
    if tok in _SUBCOMMANDS or tok.startswith("-") or not any(ch.isspace() for ch in tok):
        return argv
    return ["run", "--instruction", argv[0], *argv[1:]]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch one subcommand, and return its process exit code."""
    init_dotenv(os.environ)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(_apply_instruction_sugar(argv_list))
    if config_override := getattr(args, "config", None):
        # os.environ, not a local mapping: external plugins (inspect-robots-yam's
        # default loader) re-read the config from os.environ while constructing
        # components, and subprocesses inherit it.
        override_path = Path(config_override).expanduser()
        if not override_path.is_absolute():
            override_path = Path.cwd() / override_path
        os.environ["INSPECT_ROBOTS_CONFIG"] = str(override_path)
    if args.command == "list":
        return _cmd_list(args.what)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "eval-set":
        return _cmd_eval_set(args)
    if args.command == "inspect":
        return _cmd_inspect(
            args.log,
            transcript=args.transcript,
            wire=args.wire,
            trial=args.trial,
        )
    if args.command == "summarize":
        return _cmd_summarize(args)
    if args.command == "view":
        return _cmd_view(args)
    if args.command == "video":
        return _cmd_video(args)
    if args.command == "config":
        return _cmd_config(args)
    if args.command == "setup":
        return _cmd_setup()
    if args.command == "doctor":
        return _cmd_doctor(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
