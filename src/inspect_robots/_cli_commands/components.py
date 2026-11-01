"""Component resolution shared by ``run``, ``eval-set``, and ``doctor``: registry
lookups with guided errors, config-arg gating, and the device claim.."""

from __future__ import annotations

import argparse
import fnmatch
import math
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, NamedTuple

from inspect_robots._claims import DeviceClaim
from inspect_robots.conformance import device_slots
from inspect_robots.defaults import _ENV_SIM_EMBODIMENT, Defaults
from inspect_robots.registry import registered

from ._shared import _ENV_BY_KIND, _PLURAL_BY_KIND, _parse_kvs

if TYPE_CHECKING:
    from inspect_robots.task import Task


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
            "grader": "-G",
            "sink": "-S",
            "operator_input": "-V",
        }.get(kind, "the CLI args flag")
        raise SystemExit(
            f"invalid arguments for {kind} {name!r}: {exc}; check [{args_section}] and {flag} k=v"
        ) from exc


def _apply_epochs_or_exit(task: Task, epochs: int, *, attribute_task: bool = False) -> Task:
    """Apply ``--epochs`` with a guided error instead of a raw traceback.

    Only the count is overridden: the task's declared epoch reducer (e.g.
    ``Epochs(count=5, reducer="pass_at_2")``) is carried over, so the flag
    never silently swaps a benchmark's ``pass_at_k``/``max`` for ``mean``.
    ``Epochs.__post_init__`` rejects a count below 1 via ``ConfigError`` — the
    same validation-error class ``_resolve_or_exit`` already converts to
    ``SystemExit``.

    ``attribute_task`` names the offending task, which ``eval-set`` needs to
    say *which* of several tasks rejected the flag; ``run`` has only one.
    """
    from dataclasses import replace

    from inspect_robots.errors import ConfigError

    try:
        return replace(task, epochs=replace(task.epoch_spec, count=epochs))
    except ConfigError as exc:
        # `__post_init__` re-validates every field, but the task was already
        # valid and only `epochs` changed — so the epoch-count check is the
        # only rejection this call can newly trigger. That lets the message be
        # phrased in the flag's own terms instead of echoing Epochs' internal
        # wording, which read as "--epochs: Epochs count must be >= 1, got 0".
        where = f" (task {task.name!r})" if attribute_task else ""
        raise SystemExit(f"--epochs{where} must be >= 1, got {epochs}") from exc


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


def _resolve_components(
    args: argparse.Namespace,
    defaults: Defaults,
    *,
    resources: ExitStack | None = None,
) -> _ResolvedComponents:
    """Pick and construct the policy/embodiment pair shared by ``run`` and ``eval-set``.

    The embodiment is constructed last, so callers can invoke this immediately
    before the ``try``/``finally`` that owns ``embodiment.close()`` and leave no
    window in which a resolved embodiment could leak past a later failure.
    """

    from inspect_robots import cli

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
    if resources is not None:
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            resources.callback(close_policy)
    factories = registered("embodiment")
    slots = device_slots(factories[embodiment_name]) if embodiment_name in factories else ()
    claim = cli.claim_devices(slots, embodiment_kvs, os.environ)
    if resources is not None:
        resources.callback(claim.release)
    try:
        if args.sim:
            embodiment = _resolve_or_exit(
                "embodiment", embodiment_name, "sim_embodiment.args", **embodiment_kvs
            )
        else:
            embodiment = _resolve_or_exit("embodiment", embodiment_name, **embodiment_kvs)
    except BaseException:
        if resources is None:
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
