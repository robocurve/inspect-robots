"""Operator console, voice input, speaker sink, grader, and guardrail wiring shared
by ``run`` and ``eval-set``.."""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from inspect_robots.console import USAGE, USAGE_END_ONLY
from inspect_robots.defaults import Defaults
from inspect_robots.session import ECHO_INTERVAL_S, OperatorSession

from ._shared import _BOLD_BRIGHT_MAGENTA, _CYAN, _DIM, _YELLOW, _parse_kvs, _styled
from .components import _config_args, _resolve_or_exit, _ResolvedComponents

if TYPE_CHECKING:
    from inspect_robots.approver import Approver
    from inspect_robots.console import OperatorInput
    from inspect_robots.embodiment import Embodiment
    from inspect_robots.grader import Grader
    from inspect_robots.logging.sink import LogSink
    from inspect_robots.spaces import Box


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
        session.enable_footer(
            label="sent" if accepts_messages else "noted", echo_interval_s=ECHO_INTERVAL_S
        )
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
    session.enable_footer(label="sent", echo_interval_s=ECHO_INTERVAL_S)
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


def _build_grader(args: argparse.Namespace, defaults: Defaults) -> Grader | None:
    """Construct the run's grader and run its preflight before anything else.

    Called before components are resolved, so a grader whose preflight request
    is rejected stops the run before the robot is connected, the policy loads,
    or a task is generated (plan 0085). The operator session does not exist
    yet; ``_connect_grader_session`` attaches it once it does. Attendedness
    only picks the *default* name; an explicitly selected grader is built even
    without a session (its own fallback behavior then applies, e.g. the
    operator grader constructs a lazy session that degrades on dead stdin).
    """
    name = _select_grader_name(args, defaults)
    if name is None:
        if args.grader_args:
            raise SystemExit("-G requires a grader (--grader or a config default)")
        return None
    config_kvs = _config_args("grader", name, defaults.grader_args_owner, defaults.grader_args)
    grader_kvs = {**config_kvs, **_parse_kvs(args.grader_args)}
    grader = cast("Grader", _resolve_or_exit("grader", name, **grader_kvs))
    from inspect_robots.errors import ConfigError
    from inspect_robots.eval import _preflight_grader

    try:
        _preflight_grader(grader)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    return grader


def _connect_grader_session(grader: Grader | None, session: OperatorSession | None) -> None:
    """Share the run's operator session with a grader that wants one."""
    connect = getattr(grader, "connect_session", None)
    if session is not None and callable(connect):
        connect(session)


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
        host = fields[2] if len(fields) == 4 else ""
        # The suggested `--host 0.0.0.0` server listens on IPv4 only, so an
        # IPv6 address from SSH_CONNECTION would name a URL nothing serves.
        if not host or ":" in host:
            host = socket.gethostname()
        url = f"; open http://{host}:8300/"
    print(
        _styled(
            "      each agent turn, notes, and operator/voice input, updating live" + url,
            _DIM,
        )
    )
