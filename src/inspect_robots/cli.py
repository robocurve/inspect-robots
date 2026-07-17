"""The ``inspect_robots`` command-line interface.

Subcommands:

- ``inspect-robots list [tasks|policies|embodiments|scorers|sinks]`` — show registered
  components (builtins + installed plugins).
- ``inspect-robots run --task T --policy P --embodiment E`` — run an eval, resolving
  components from the registry. Pass constructor args with ``-T/-P/-E k=v``;
  ``--epochs``, ``--fail-on-error``, and ``--store-frames`` tune the run. The
  written log's path is printed at the end.
- ``inspect-robots eval-set TASK [TASK ...] --policy P --embodiment E`` — run several
  registered tasks (exact names or ``fnmatch`` globs, e.g. ``'kitchenbench/*'``) against
  one resolved policy/embodiment pair via
  [`eval_set`][inspect_robots.eval.eval_set]. Prints one status line and a compact
  per-task row instead of a full summary per task.
- ``inspect-robots inspect LOG.json [--transcript]`` — print a saved eval log and
  optionally append recorded policy conversations.
- ``inspect-robots view LOG.json [-o OUT.html] [--open]`` — render a saved eval log
  as a self-contained HTML report.
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
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from inspect_robots import __version__
from inspect_robots._defaults import (
    ADHOC_MAX_STEPS_FALLBACK,
    ADHOC_SCORER_FALLBACK,
    CONFIG_KEYS,
    ENV_EMBODIMENT,
    ENV_POLICY,
    ENV_SIM_EMBODIMENT,
    Defaults,
    load_defaults,
    parse_value,
    set_default,
)
from inspect_robots._dotenv import init_dotenv
from inspect_robots._html import (
    _STATUS_DISPLAY as _STATUS_DISPLAY,
)
from inspect_robots._html import (
    _chat_content,
    _display_status,
    _is_chat_transcript,
    render_html,
)

if TYPE_CHECKING:
    from inspect_robots.approver import Approver
    from inspect_robots.log import EvalLog
    from inspect_robots.logging.sink import LogSink
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene
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
_DIM = "2"
_CYAN = "36"
_GREEN = "32"
_RED = "31"
_YELLOW = "33"

_OUTCOME_PHRASES = {
    "success": "succeeded",
    "failure": "failed",
    "max_steps": "hit step limit",
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
    "sinks": "sink",
}

_PLURAL_BY_KIND = {kind: plural for plural, kind in _KIND_BY_PLURAL.items()}

_SUBCOMMANDS = (
    "list",
    "run",
    "eval-set",
    "inspect",
    "view",
    "video",
    "config",
    "setup",
    "doctor",
)

_ENV_BY_KIND = {"policy": ENV_POLICY, "embodiment": ENV_EMBODIMENT}

DEFAULT_RERUN_CONNECT_URL = "rerun+http://127.0.0.1:9876/proxy"


def _parse_kvs(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key] = parse_value(value)
    return out


def _add_shared_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags common to ``run`` and ``eval-set``.

    Component selection (``--policy``/``--embodiment``/``-P``/``-E``/``--sim``),
    guardrails, logging, and epoch/error handling live here so a new shared flag
    lands in both commands at once instead of drifting between two copies.
    """
    parser.add_argument("--policy", help="registered policy name (default: user config)")
    parser.add_argument("--embodiment", help="registered embodiment name (default: user config)")
    parser.add_argument("-P", dest="policy_args", action="append", metavar="k=v")
    parser.add_argument("-E", dest="embodiment_args", action="append", metavar="k=v")
    parser.add_argument("--log-dir", default="logs")
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
    p_run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="horizon of an --instruction run (default: config or "
        f"{ADHOC_MAX_STEPS_FALLBACK}); invalid with --task",
    )
    p_run.add_argument(
        "--scorer",
        default=None,
        help="scorer for an --instruction run (default: config or "
        f"{ADHOC_SCORER_FALLBACK!r}); invalid with --task",
    )
    p_run.add_argument(
        "--no-prompt",
        action="store_true",
        help="never ask the terminal operator for a success verdict",
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
        "--rerun-connect",
        nargs="?",
        const=DEFAULT_RERUN_CONNECT_URL,
        default=None,
        metavar="URL",
        help="stream the rollout to a Rerun viewer already running elsewhere "
        "(e.g. your laptop via an SSH reverse tunnel: ssh -R 9876:localhost:9876 ...); "
        f"URL defaults to {DEFAULT_RERUN_CONNECT_URL}",
    )

    p_eval_set = sub.add_parser("eval-set", help="run a set of registered tasks in one invocation")
    p_eval_set.add_argument(
        "tasks",
        nargs="+",
        metavar="TASK",
        help="registered task name(s); shell-quoted globs match by prefix, e.g. 'kitchenbench/*'",
    )
    _add_shared_eval_args(p_eval_set)
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

    p_view = sub.add_parser("view", help="render a saved eval log as a self-contained HTML report")
    p_view.add_argument("log", help="path to an EvalLog JSON file")
    p_view.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="FILE",
        help="output HTML file (default: LOG with an .html suffix; - writes to stdout)",
    )
    p_view.add_argument(
        "--open",
        action="store_true",
        help="open the written report in the default web browser",
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

    sub.add_parser(
        "setup",
        help="interactive first-run wizard: pick defaults and discover camera devices, "
        "then write config.ini",
    )

    p_config = sub.add_parser("config", help="view or set user defaults (config.ini)")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_set = config_sub.add_parser("set", help="persist a [defaults] key to the config file")
    p_set.add_argument("key", choices=CONFIG_KEYS)
    p_set.add_argument("value")
    config_sub.add_parser("show", help="print resolved defaults and their sources")
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
        f"fix: set ${ENV_SIM_EMBODIMENT}, or run "
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
        raise SystemExit(str(exc.args[0])) from exc
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    except TypeError as exc:
        if args_section is None:
            args_section = f"{kind}.args"
        flag = {"task": "-T", "policy": "-P", "embodiment": "-E"}.get(kind, "the CLI args flag")
        raise SystemExit(
            f"invalid arguments for {kind} {name!r}: {exc}; check [{args_section}] and {flag} k=v"
        ) from exc


def _build_guardrails(
    space: Box, max_action_delta: float | None
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
    if not parts:
        warnings.append(
            "no guardrails are active for this action space; declare bounds/semantics "
            "on the embodiment or pass --max-action-delta"
        )
        return AutoApprover(), active, warnings
    return ChainApprover(*parts), active, warnings


_PROMPT = "did the robot succeed? [y/n/partial/skip] (partial scores as failure) "
_PROMPT_ANSWERS = frozenset({"y", "yes", "n", "no", "partial", "skip"})
_DEFINITIVE_REASONS = frozenset({"success", "failure"})


def _prompt_operator(record: TrialRecord, scene: Scene) -> None:
    """Capture or adopt the terminal operator's verdict on the record (R6).

    A terminated episode with a definitive embodiment verdict adopts and announces that
    verdict instead of asking the operator to confirm the same outcome a second time.
    """
    from inspect_robots.transcript import operator_event

    del scene
    if record.terminated and record.termination_reason in _DEFINITIVE_REASONS:
        verdict = "y" if record.termination_reason == "success" else "n"
        record.operator_judgement = verdict
        record.events.append(
            operator_event(t=len(record.steps), verdict=verdict, source="embodiment")
        )
        print(f"operator verdict adopted from embodiment: {record.termination_reason}")
        return
    if record.truncated and record.termination_reason == "max_steps":
        print("note: this trial hit the step limit before terminating")
    while True:
        try:
            answer = input(_PROMPT).strip().lower()
        except EOFError:
            answer = "skip"
        if answer in _PROMPT_ANSWERS:
            break
        print(f"unrecognized answer {answer!r}; expected one of y/n/partial/skip")
    if answer == "skip":
        return
    record.operator_judgement = answer
    record.events.append(operator_event(t=len(record.steps), verdict=answer))


def _step_limit_count(log: EvalLog) -> int:
    """Count recorded trials whose termination reason is the step horizon."""
    return sum(
        reason == "max_steps" for scene in log.samples for reason in scene.termination_reasons
    )


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
    # Guards below reject bool/str values a hand-edited log can smuggle past
    # from_dict (bool is an int subclass, so isinstance alone lets True in).
    if isinstance(max_steps, int) and not isinstance(max_steps, bool):
        parenthetical = f"max_steps={max_steps}"
        rate = log.eval.embodiment_info.get("control_hz")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
            parenthetical += f", ~{max_steps / rate:g}s at {rate:g} Hz"
        note += f" ({parenthetical})"
    print(_styled(note, _YELLOW))
    if is_adhoc:
        hint = "hint: raise it with --max-steps N or: inspect-robots config set max_steps N"
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


class _ResolvedComponents(NamedTuple):
    """A resolved policy/embodiment pair plus the names/sources for the run header."""

    policy: Any
    policy_name: str
    policy_source: str
    embodiment: Any
    embodiment_name: str
    embodiment_source: str


def _check_shared_run_conflicts(args: argparse.Namespace) -> None:
    """Reject flag combinations invalid for both ``run`` and ``eval-set``."""
    if args.sim and args.embodiment:
        raise SystemExit(
            "--sim selects your configured sim_embodiment; "
            "passing --embodiment already picks the embodiment — drop one"
        )
    if args.disable_guardrails and args.max_action_delta is not None:
        raise SystemExit(
            "--max-action-delta tunes the guardrails that --disable-guardrails turns off — drop one"
        )


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
    if args.sim:
        embodiment = _resolve_or_exit(
            "embodiment", embodiment_name, "sim_embodiment.args", **embodiment_kvs
        )
    else:
        embodiment = _resolve_or_exit("embodiment", embodiment_name, **embodiment_kvs)
    return _ResolvedComponents(
        policy, policy_name, policy_source, embodiment, embodiment_name, embodiment_source
    )


def _announce_components(resolved: _ResolvedComponents) -> None:
    """Print the resolved policy/embodiment and where each came from.

    Defaults must never be silent: say what runs, and why, before it moves.
    """
    print(f"policy: {resolved.policy_name} ({resolved.policy_source})")
    print(f"embodiment: {resolved.embodiment_name} ({resolved.embodiment_source})")


def _build_and_announce_guardrails(args: argparse.Namespace, action_space: Box) -> Approver | None:
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
    approver, active, guard_warnings = _build_guardrails(action_space, args.max_action_delta)
    for warning in guard_warnings:
        print(f"guardrails warning: {warning}", file=sys.stderr)
    print(f"guardrails: {' + '.join(active) if active else 'none active'}")
    return approver


def _cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from inspect_robots import eval
    from inspect_robots.logging import JsonLogSink
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
    _check_shared_run_conflicts(args)

    defaults = load_defaults(os.environ)

    if is_adhoc:
        scorer_name = args.scorer or defaults.scorer or ADHOC_SCORER_FALLBACK
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else (defaults.max_steps or ADHOC_MAX_STEPS_FALLBACK)
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
    try:
        if args.epochs is not None:
            task = replace(task, epochs=args.epochs)

        _announce_components(resolved)
        approver = _build_and_announce_guardrails(args, embodiment.info.action_space)

        before_scoring = None
        if (
            is_adhoc
            and not args.no_prompt
            and sys.stdin.isatty()
            and any(s.name == "operator" for s in task.scorers)
        ):
            # Ad-hoc runs only: a registered task with an operator scorer keeps
            # R6's non-blocking, unattended-safe behavior (judgement stays None).
            before_scoring = _prompt_operator

        # Construct the sink explicitly so we can tell the user where the log went.
        sink = JsonLogSink(args.log_dir)
        sinks: list[LogSink] = [sink]
        if args.rerun_connect is not None:
            from inspect_robots.logging.rerun_sink import RerunSink

            sinks.append(RerunSink(connect_url=args.rerun_connect))
            print(f"{_styled('rerun:', _CYAN)} connect {args.rerun_connect}")
        elif args.rerun if args.rerun is not None else defaults.rerun:
            from inspect_robots.logging.rerun_sink import RerunSink

            # spawn=True opens the live viewer; the sink itself degrades to a
            # warn-once no-op when rerun-sdk is not installed.
            sinks.append(RerunSink(spawn=True))
            print(f"{_styled('rerun:', _CYAN)} live viewer")
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
                before_scoring=before_scoring,
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
        embodiment.close()
    log = logs[0]
    _print_run_summary(log, str(sink.path), is_adhoc)
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
        print(f"{row}  {detail}" if detail else row)
    print(f"{_styled('log dir:', _CYAN)} {_styled(log_dir, _DIM)}")
    if not success:
        print(
            _styled(
                f"hint: inspect a log with: inspect-robots inspect {log_dir}/<task>_<id>.json",
                _DIM,
            )
        )
    print(
        _styled(
            f"hint: HTML viewer: inspect-robots view {log_dir}/<task>_<id>.json",
            _DIM,
        )
    )


def _cmd_eval_set(args: argparse.Namespace) -> int:
    """Resolve one policy/embodiment once, then drive every matched task through it.

    A thin wrapper over [`eval_set`][inspect_robots.eval.eval_set]: unlike
    calling ``eval_set()`` with string components (which resolves and closes
    the embodiment once per task), the CLI resolves the embodiment exactly
    once for the whole set, so a real robot is not reconnected between tasks.
    """
    from dataclasses import replace

    from inspect_robots import eval_set

    _check_shared_run_conflicts(args)
    task_names = _match_tasks(args.tasks)

    defaults = load_defaults(os.environ)
    tasks = [_resolve_or_exit("task", name) for name in task_names]
    if args.epochs is not None:
        tasks = [replace(t, epochs=args.epochs) for t in tasks]

    resolved = _resolve_components(args, defaults)
    embodiment = resolved.embodiment
    try:
        _announce_components(resolved)
        print(f"tasks: {', '.join(task_names)}")
        approver = _build_and_announce_guardrails(args, embodiment.info.action_space)
        try:
            success, logs = eval_set(
                tasks,
                resolved.policy,
                embodiment,
                log_dir=args.log_dir,
                seed=args.seed,
                fail_on_error=args.fail_on_error if args.fail_on_error is not None else False,
                approver=approver,
                store_frames=(
                    args.store_frames if args.store_frames is not None else defaults.store_frames
                ),
                retry_attempts=args.retry_attempts,
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
                    f"hint: HTML viewer: inspect-robots view {args.log_dir}/<task>_<id>.json",
                    _DIM,
                )
            )
            return 130
    finally:
        # Same "close what we open" contract as _cmd_run: the CLI resolved the
        # embodiment itself, so it — not eval_set() — is responsible for
        # releasing it, exactly once, after every task has run.
        embodiment.close()
    _print_eval_set_summary(success, logs, args.log_dir)
    return 0 if success else 1


def _cmd_inspect(path: str, *, transcript: bool = False) -> int:
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
        print(f"  {name}: {value:.4g}")
    print("scenes:")
    for scene in log.samples:
        reduced = "  ".join(f"{k}={v:.4g}" for k, v in sorted(scene.reduced.items()))
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
    return 0 if log.status == "success" else 1


def _cmd_view(args: argparse.Namespace) -> int:
    """Render a saved log to a UTF-8 HTML artifact and optionally open it."""
    import webbrowser

    from inspect_robots import read_eval_log

    stdout_mode = args.out == "-"
    if stdout_mode and args.open:
        raise SystemExit("--open cannot be used with -o -: no file to open")

    log_path = Path(args.log)
    out_path = (
        None
        if stdout_mode
        else (log_path.with_suffix(".html") if args.out is None else Path(args.out))
    )
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out {out_path} is a directory; pass an HTML file path")

    log = read_eval_log(args.log)
    document = render_html(log, title=f"{log.eval.task} - {log_path.name}")
    if stdout_mode:
        degraded = document.encode("utf-8", errors="replace").decode("utf-8")
        sys.stdout.write(degraded)
        return 0

    file_path = cast(Path, out_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    print(f"wrote {file_path}")

    if args.open:
        uri = file_path.resolve().as_uri()
        try:
            opened = webbrowser.open(uri)
        except Exception as exc:
            print(f"warning: could not open browser for {file_path}: {exc}", file=sys.stderr)
        else:
            if not opened:
                print(f"warning: could not open browser for {file_path}", file=sys.stderr)
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
        path = set_default(os.environ, args.key, args.value)
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
    if args.command == "list":
        return _cmd_list(args.what)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "eval-set":
        return _cmd_eval_set(args)
    if args.command == "inspect":
        return _cmd_inspect(args.log, transcript=args.transcript)
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
