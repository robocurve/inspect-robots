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
- ``inspect-robots inspect LOG.json`` — print a saved eval log.
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
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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


_KIND_BY_PLURAL = {
    "tasks": "task",
    "policies": "policy",
    "embodiments": "embodiment",
    "scorers": "scorer",
    "sinks": "sink",
}

_PLURAL_BY_KIND = {kind: plural for plural, kind in _KIND_BY_PLURAL.items()}

_SUBCOMMANDS = ("list", "run", "eval-set", "inspect", "config", "setup", "doctor")

_ENV_BY_KIND = {"policy": ENV_POLICY, "embodiment": ENV_EMBODIMENT}


def _parse_kvs(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key] = parse_value(value)
    return out


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
    p_run.add_argument("--policy", help="registered policy name (default: user config)")
    p_run.add_argument("--embodiment", help="registered embodiment name (default: user config)")
    p_run.add_argument("-T", dest="task_args", action="append", metavar="k=v")
    p_run.add_argument("-P", dest="policy_args", action="append", metavar="k=v")
    p_run.add_argument("-E", dest="embodiment_args", action="append", metavar="k=v")
    p_run.add_argument("--log-dir", default="logs")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--epochs", type=int, default=None, help="override the task's epoch count")
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
        "--sim",
        action="store_true",
        help="run on the configured sim_embodiment instead of the default "
        "(real-hardware) embodiment",
    )
    p_run.add_argument(
        "--no-prompt",
        action="store_true",
        help="never ask the terminal operator for a success verdict",
    )
    p_run.add_argument(
        "--fail-on-error",
        type=float,
        default=None,
        metavar="X",
        help="halt on PolicyErrors: 1 = first error, 0<X<1 = proportion, X>1 = count",
    )
    p_run.add_argument(
        "--store-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stream camera frames to a per-run directory under <log-dir>/frames "
        "instead of keeping them in memory (--no-store-frames overrides a "
        "store_frames config default)",
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
        "--disable-guardrails",
        action="store_true",
        help="turn off the default safety approvers (bounds clamp + per-step "
        "delta limit); actions reach the embodiment unchecked",
    )
    p_run.add_argument(
        "--max-action-delta",
        type=float,
        default=None,
        metavar="D",
        help="per-step change limit for the default guardrails, in the action "
        "space's native units (default: derived from the space's bounds)",
    )

    p_eval_set = sub.add_parser("eval-set", help="run a set of registered tasks in one invocation")
    p_eval_set.add_argument(
        "tasks",
        nargs="+",
        metavar="TASK",
        help="registered task name(s); shell-quoted globs match by prefix, e.g. 'kitchenbench/*'",
    )
    p_eval_set.add_argument("--policy", help="registered policy name (default: user config)")
    p_eval_set.add_argument(
        "--embodiment", help="registered embodiment name (default: user config)"
    )
    p_eval_set.add_argument("-P", dest="policy_args", action="append", metavar="k=v")
    p_eval_set.add_argument("-E", dest="embodiment_args", action="append", metavar="k=v")
    p_eval_set.add_argument("--log-dir", default="logs")
    p_eval_set.add_argument("--seed", type=int, default=0)
    p_eval_set.add_argument(
        "--epochs", type=int, default=None, help="override every matched task's epoch count"
    )
    p_eval_set.add_argument(
        "--sim",
        action="store_true",
        help="run on the configured sim_embodiment instead of the default "
        "(real-hardware) embodiment",
    )
    p_eval_set.add_argument(
        "--fail-on-error",
        type=float,
        default=None,
        metavar="X",
        help="halt on PolicyErrors: 1 = first error, 0<X<1 = proportion, X>1 = count",
    )
    p_eval_set.add_argument(
        "--retry-attempts",
        type=int,
        default=0,
        help="passed through to eval_set(); resumption of a partial run is "
        "accepted but not yet honored",
    )
    p_eval_set.add_argument(
        "--store-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stream camera frames to a per-run directory under <log-dir>/frames "
        "instead of keeping them in memory (--no-store-frames overrides a "
        "store_frames config default)",
    )
    p_eval_set.add_argument(
        "--disable-guardrails",
        action="store_true",
        help="turn off the default safety approvers (bounds clamp + per-step "
        "delta limit); actions reach the embodiment unchecked",
    )
    p_eval_set.add_argument(
        "--max-action-delta",
        type=float,
        default=None,
        metavar="D",
        help="per-step change limit for the default guardrails, in the action "
        "space's native units (default: derived from the space's bounds)",
    )

    p_inspect = sub.add_parser("inspect", help="print a saved eval log")
    p_inspect.add_argument("log", help="path to an EvalLog JSON file")

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


def _prompt_operator(record: TrialRecord, scene: Scene) -> None:
    """Capture the terminal operator's verdict on the record (R6)."""
    from inspect_robots.transcript import operator_event

    del scene
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


def _print_run_summary(log: EvalLog, log_path: str) -> None:
    """Print the compact post-run summary and failure diagnostics."""
    failed = log.status != "success"
    status_color = _RED if failed else _GREEN
    print(f"{_styled('status:', _CYAN)} {_styled(log.status, status_color)}")
    if failed and log.error is not None:
        print(f"{_styled('error:', _CYAN)} {_styled(log.error, _RED)}")
    if failed:
        for scene in log.samples:
            if scene.status == "error":
                detail = "" if scene.error in (None, log.error) else f": {scene.error}"
                print(f"  [{_styled('error', _RED)}] {scene.scene_id}{detail}")
    print(
        f"{_styled('scenes:', _CYAN)} {log.results.total_scenes}  "
        f"trials: {log.results.total_trials}"
    )
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {_styled(f'{value:.4g}', _BOLD)}")
    print(f"{_styled('log:', _CYAN)} {_styled(log_path, _DIM)}")
    if failed:
        print(_styled(f"hint: inspect-robots inspect {log_path}", _DIM))


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
    if args.sim and args.embodiment:
        raise SystemExit(
            "--sim selects your configured sim_embodiment; "
            "passing --embodiment already picks the embodiment — drop one"
        )
    if args.disable_guardrails and args.max_action_delta is not None:
        raise SystemExit(
            "--max-action-delta tunes the guardrails that --disable-guardrails turns off — drop one"
        )

    defaults = load_defaults(os.environ)
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

    policy = _resolve_or_exit("policy", policy_name, **policy_kvs)
    if args.sim:
        embodiment = _resolve_or_exit(
            "embodiment", embodiment_name, "sim_embodiment.args", **embodiment_kvs
        )
    else:
        embodiment = _resolve_or_exit("embodiment", embodiment_name, **embodiment_kvs)
    try:
        if args.epochs is not None:
            task = replace(task, epochs=args.epochs)

        # Defaults must never be silent: say what runs, and why, before it moves.
        print(f"policy: {policy_name} ({policy_source})")
        print(f"embodiment: {embodiment_name} ({embodiment_source})")

        # Guardrails are on by default (plan 0008 §3e): the approver chain
        # sits below the policy in rollout, so nothing the policy emits — a
        # wild VLA action or a misbehaving LLM — reaches hardware unchecked.
        approver: Approver | None = None
        if args.disable_guardrails:
            print(
                "WARNING: guardrails disabled (--disable-guardrails); actions "
                "reach the embodiment unchecked.",
                file=sys.stderr,
            )
            print("guardrails: disabled (--disable-guardrails)")
        else:
            approver, active, guard_warnings = _build_guardrails(
                embodiment.info.action_space, args.max_action_delta
            )
            for warning in guard_warnings:
                print(f"guardrails warning: {warning}", file=sys.stderr)
            print(f"guardrails: {' + '.join(active) if active else 'none active'}")

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
        if args.rerun if args.rerun is not None else defaults.rerun:
            from inspect_robots.logging.rerun_sink import RerunSink

            # spawn=True opens the live viewer; the sink itself degrades to a
            # warn-once no-op when rerun-sdk is not installed.
            sinks.append(RerunSink(spawn=True))
            print(f"{_styled('rerun:', _CYAN)} live viewer")
        logs = eval(
            task,
            policy,
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
    finally:
        # The CLI resolved the embodiment itself, so eval() does not own it
        # ("close what we open"). Real-hardware embodiments release motor
        # torque in close(); skipping this leaves a robot energized. The span
        # starts right after resolution: --epochs/scorer validation between
        # here and eval() can raise, and that must not leak the embodiment.
        embodiment.close()
    log = logs[0]
    _print_run_summary(log, str(sink.path))
    return 0 if log.status == "success" else 1


def _print_eval_set_summary(success: bool, logs: Sequence[EvalLog], log_dir: str) -> None:
    """Print one overall status line, a compact row per task, then the shared log dir.

    Deliberately not N full ``_print_run_summary``s: a 10-task benchmark run
    should not scroll 10 screens of near-identical output.
    """
    status = "success" if success else "error"
    print(f"{_styled('status:', _CYAN)} {_styled(status, _GREEN if success else _RED)}")
    for log in logs:
        failed = log.status != "success"
        metrics = ", ".join(
            f"{name}={value:.4g}" for name, value in sorted(log.results.metrics.items())
        )
        detail = metrics or (log.error or "")
        row = f"  [{_styled(log.status, _RED if failed else _GREEN)}] {log.eval.task}"
        print(f"{row}  {detail}" if detail else row)
    print(f"{_styled('log dir:', _CYAN)} {_styled(log_dir, _DIM)}")
    if not success:
        print(_styled(f"hint: inspect-robots inspect {log_dir}/<task>_<id>.json", _DIM))


def _cmd_eval_set(args: argparse.Namespace) -> int:
    """Resolve one policy/embodiment once, then drive every matched task through it.

    A thin wrapper over [`eval_set`][inspect_robots.eval.eval_set]: unlike
    calling ``eval_set()`` with string components (which resolves and closes
    the embodiment once per task), the CLI resolves the embodiment exactly
    once for the whole set, so a real robot is not reconnected between tasks.
    """
    from dataclasses import replace

    from inspect_robots import eval_set

    if args.sim and args.embodiment:
        raise SystemExit(
            "--sim selects your configured sim_embodiment; "
            "passing --embodiment already picks the embodiment — drop one"
        )
    if args.disable_guardrails and args.max_action_delta is not None:
        raise SystemExit(
            "--max-action-delta tunes the guardrails that --disable-guardrails turns off — drop one"
        )

    task_names = _match_tasks(args.tasks)

    defaults = load_defaults(os.environ)
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
    policy_config_args = _config_args(
        "policy", policy_name, defaults.policy_args_owner, defaults.policy_args
    )
    policy_kvs = {**policy_config_args, **_parse_kvs(args.policy_args)}
    embodiment_kvs = {**embodiment_defaults, **_parse_kvs(args.embodiment_args)}

    tasks = [_resolve_or_exit("task", name) for name in task_names]
    if args.epochs is not None:
        tasks = [replace(t, epochs=args.epochs) for t in tasks]

    policy = _resolve_or_exit("policy", policy_name, **policy_kvs)
    if args.sim:
        embodiment = _resolve_or_exit(
            "embodiment", embodiment_name, "sim_embodiment.args", **embodiment_kvs
        )
    else:
        embodiment = _resolve_or_exit("embodiment", embodiment_name, **embodiment_kvs)

    try:
        print(f"policy: {policy_name} ({policy_source})")
        print(f"embodiment: {embodiment_name} ({embodiment_source})")
        print(f"tasks: {', '.join(task_names)}")

        approver: Approver | None = None
        if args.disable_guardrails:
            print(
                "WARNING: guardrails disabled (--disable-guardrails); actions "
                "reach the embodiment unchecked.",
                file=sys.stderr,
            )
            print("guardrails: disabled (--disable-guardrails)")
        else:
            approver, active, guard_warnings = _build_guardrails(
                embodiment.info.action_space, args.max_action_delta
            )
            for warning in guard_warnings:
                print(f"guardrails warning: {warning}", file=sys.stderr)
            print(f"guardrails: {' + '.join(active) if active else 'none active'}")

        success, logs = eval_set(
            tasks,
            policy,
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
    finally:
        # Same "close what we open" contract as _cmd_run: the CLI resolved the
        # embodiment itself, so it — not eval_set() — is responsible for
        # releasing it, exactly once, after every task has run.
        embodiment.close()
    _print_eval_set_summary(success, logs, args.log_dir)
    return 0 if success else 1


def _cmd_inspect(path: str) -> int:
    from inspect_robots import read_eval_log

    log = read_eval_log(path)
    print(f"task:        {log.eval.task}")
    print(f"policy:      {log.eval.policy}")
    print(f"embodiment:  {log.eval.embodiment}")
    print(f"status:      {log.status}")
    print(f"created:     {log.eval.created}")
    print(f"git:         {log.eval.git_commit}")
    print(f"scenes:      {log.results.total_scenes}   trials: {log.results.total_trials}")
    print("metrics:")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}")
    print("scenes:")
    for scene in log.samples:
        reduced = "  ".join(f"{k}={v:.4g}" for k, v in sorted(scene.reduced.items()))
        print(f"  [{scene.status}] {scene.scene_id}: {reduced}")
    if log.error:
        print(f"error: {log.error}")
    return 0 if log.status == "success" else 1


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
        return _cmd_inspect(args.log)
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
