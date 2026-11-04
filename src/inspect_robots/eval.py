"""The ``eval()`` entry point — orchestrates scenes x epochs into an EvalLog.

Mirrors Inspect AI's ``eval()``: it runs a task's scenes (repeated over epochs),
scores each recorded trajectory, reduces epochs, aggregates metrics, and returns
a list of immutable [`EvalLog`][inspect_robots.log.EvalLog] (one per task). The tracer
slice accepts already-constructed objects; registry-string resolution
(``policy="openvla/7b"``) is layered on with the registry milestone.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from inspect_robots import __version__
from inspect_robots.action_manager import ActionManager, DefaultActionManager
from inspect_robots.compat import assert_compatible
from inspect_robots.embodiment import Embodiment, HealthReport, Readiness
from inspect_robots.errors import (
    ConfigError,
    EmbodimentFault,
    PolicyError,
    SafetyAbort,
    _CancelledTrial,
)
from inspect_robots.frames import FrameStore, _safe
from inspect_robots.grader import Grader
from inspect_robots.guard import Guard, PassThroughGuard, build_guard
from inspect_robots.log import (
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
    _json_safe_scene_metadata,
)
from inspect_robots.policy import Policy
from inspect_robots.rollout import Record, derive_seed, rollout
from inspect_robots.scene import Scene
from inspect_robots.scorer import (
    Score,
    get_reducer,
    reduce_scores,
    reducer_min_epochs,
    value_to_float,
)
from inspect_robots.task import Task
from inspect_robots.transcript import judgement_source

if TYPE_CHECKING:
    from inspect_robots.benchmark import Benchmark
    from inspect_robots.console import OperatorInput
    from inspect_robots.logging.sink import LogSink
    from inspect_robots.spaces import Box, ObservationSpace
    from inspect_robots.types import Action, Observation, StepResult


def _grading_hook(
    grader: Grader | str | None,
    before_scoring: Callable[[Record, Scene], None] | None,
) -> tuple[Callable[[Record, Scene], None] | None, Grader | None]:
    """Resolve ``grader``/``before_scoring`` into the pre-scoring hook and its grader.

    The two arguments write to the same seam, so passing both is always a
    caller bug and raises ``ConfigError``. A string resolves through the
    registry, and the result must satisfy the ``Grader`` protocol — a broken
    entry point fails here, at configuration time, not deep inside scoring.

    The resolved grader comes back alongside the hook so the caller can record
    what graded the run; a bare ``before_scoring`` callable has no identity to
    record and yields ``None``.
    """
    if grader is None:
        return before_scoring, None
    if before_scoring is not None:
        raise ConfigError("pass either grader or before_scoring, not both")
    if isinstance(grader, str):
        from inspect_robots.registry import resolve

        grader = cast(Grader, resolve("grader", grader))
    if not isinstance(grader, Grader):
        raise ConfigError(
            f"grader must implement the Grader protocol (a name and a "
            f"grade(record, scene) method); got {type(grader).__name__}"
        )
    return grader.grade, grader


def _preflight_grader(grader: Grader | None) -> None:
    """Run the grader's optional ``preflight()`` hook before any rollout.

    Duck-typed like ``config()`` so graders without it still satisfy the
    protocol. The hook raises ``ConfigError`` when every trial would be
    rejected; it is idempotent, so the CLI, ``eval_set`` and ``eval`` may all
    call it.
    """
    hook = getattr(grader, "preflight", None)
    if callable(hook):
        hook()


def _grader_identity(grader: Grader | None) -> tuple[str | None, dict[str, Any]]:
    """Return the grader's registry name and effective config for the eval spec.

    ``config()`` is an optional duck-typed hook, like the embodiment's
    ``bind_task``, deliberately not part of the ``Grader`` Protocol: adding it
    there would make every existing out-of-tree grader fail the protocol's
    ``isinstance`` check. A grader without one is still named in the log and
    simply records no configuration.
    """
    if grader is None:
        return None, {}
    hook = getattr(grader, "config", None)
    if not callable(hook):
        return grader.name, {}
    return grader.name, cast("dict[str, Any]", dict(hook()))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_action_log(
    record: Record,
    log_dir: str,
    run_stamp: str,
    action_space: Box,
) -> str | None:
    """Atomically persist one trial's executed actions, degrading on write failure."""
    trial_id = f"{_safe(record.scene_id)}-e{record.epoch}"
    relative_path = Path("actions") / run_stamp / f"{trial_id}.jsonl"
    path = Path(log_dir) / relative_path
    semantics = action_space.semantics
    labels = semantics.dim_labels if semantics is not None else None
    try:
        lines = [
            json.dumps(
                {
                    "kind": "header",
                    "run_id": run_stamp,
                    "scene_id": record.scene_id,
                    "epoch": record.epoch,
                    "action_dim": action_space.dim,
                    "labels": labels,
                },
                allow_nan=False,
            )
        ]
        lines.extend(
            json.dumps(
                {
                    "t": step.t,
                    "action": [float(value) for value in np.asarray(step.action.data).ravel()],
                },
                allow_nan=False,
            )
            for step in record.steps
        )
        payload = "\n".join(lines) + "\n"

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except (OSError, ValueError) as exc:
        warnings.warn(
            f"Action log disabled for this trial after {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return relative_path.as_posix()


# A key naming a credential, matched on its last underscore-separated part(s):
# "api_key", "hf_token" and "password" match; "max_tokens" and "api_key_env"
# (which names an environment variable, not its value) do not.
_SECRET_KEY = re.compile(
    r"(?i)(^|_)(api_?key|secret|token|password|passwd|credentials?|authorization"
    r"|access_?key|private_?key)$"
)
_REDACTED = "<redacted>"


def _describe(value: Any, depth: int = 0) -> Any:
    """A JSON-safe description of ``value`` for records, with credential-like keys redacted.

    Objects become ``{"type": "module.Class", ...public attributes}`` so an
    action manager or guard chain is recorded with its parameters. Arrays
    larger than 64 values are summarized by shape. Recursion stops at depth 4.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _describe(value.item(), depth)
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        return _describe(value.tolist(), depth)
    if depth >= 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED
            if _SECRET_KEY.search(str(key)) and item is not None
            else _describe(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_describe(item, depth + 1) for item in items]
    attributes = getattr(value, "__dict__", None)
    described: dict[str, Any] = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(attributes, dict):
        public = {key: item for key, item in attributes.items() if not key.startswith("_")}
        described.update(_describe(public, depth))
    # An optional config() reports settings held privately (a guard's resolved
    # limits), so records of runs with different settings stay distinguishable.
    config = getattr(value, "config", None)
    if callable(config) and not isinstance(value, type):
        try:
            reported = config()
        except Exception as exc:
            described["config_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(reported, Mapping):
                # A fresh depth budget: the hook's own (small) settings must
                # survive even when the object sits deep in a guard chain.
                described["config"] = _describe(reported)
    return described


def _write_atomic(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` through a fsynced temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _finite(value: Any) -> Any:
    """``value`` with non-finite floats as strings (``"nan"``, ``"inf"``), at any depth."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _record_json(value: Any, indent: int | None = None) -> str:
    """Strict JSON for record files; values json cannot encode are described, not raised.

    Non-finite floats anywhere (scene or rubric metadata, grader configuration)
    are written as strings, as non-finite action values are, so one NaN never
    costs the trial its record.
    """
    return json.dumps(_finite(value), indent=indent, allow_nan=False, default=_describe)


def _write_record(
    record: Record,
    records_dir: Path,
    log_dir: Path,
    events: tuple[dict[str, Any], ...],
    configuration: dict[str, Any],
    action_space: Box,
) -> str | None:
    """Persist one trial's record directory; return it relative to ``log_dir``, or None.

    ``record.json`` holds identity, configuration, termination and events;
    ``chunks.jsonl`` holds a header line, then one line per produced chunk.
    Non-finite action values are written as strings (``"nan"``, ``"inf"``) so
    a chunk that failed the trial is still recorded as the policy produced it.
    """
    directory = records_dir / f"{_safe(record.scene_id)}-e{record.epoch}"
    semantics = action_space.semantics
    document = {
        "record_id": record.record_id,
        "run_id": records_dir.name,
        "scene_id": record.scene_id,
        "epoch": record.epoch,
        "seed": record.seed,
        "status": record.status,
        "error": record.error,
        "termination": {
            "reason": record.termination_reason,
            "detail": record.termination_detail,
            "terminated": record.terminated,
            "truncated": record.truncated,
        },
        "steps": len(record.steps),
        "chunks": len(record.chunks),
        "actions": record.metadata.get("actions"),
        "configuration": configuration,
        "events": list(events),
    }
    header = {
        "kind": "header",
        "record_id": record.record_id,
        "action_dim": action_space.dim,
        "labels": semantics.dim_labels if semantics is not None else None,
    }
    try:
        # Values a policy, scorer or rubric supplies (numpy scalars, NaN latencies,
        # arbitrary metadata) go through _describe, so storage degrades with a
        # warning instead of escaping eval() and losing the run.
        lines = [_record_json(header)]
        lines.extend(
            _record_json(
                {
                    "t": chunk.t,
                    "actions": [_describe(action.tolist()) for action in chunk.actions],
                    "terminal": None
                    if chunk.terminal is None
                    else {"kind": chunk.terminal.kind, "reason": chunk.terminal.reason},
                    "latency_s": _describe(chunk.latency_s),
                    "wall_s": _describe(chunk.wall_s),
                }
            )
            for chunk in record.chunks
        )
        record_json = _record_json(document, indent=2) + "\n"
        # Both payloads are encoded before either file is written, so an
        # encoding failure never leaves a partial record directory behind.
        _write_atomic(directory / "chunks.jsonl", "\n".join(lines) + "\n")
        _write_atomic(directory / "record.json", record_json)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        warnings.warn(
            f"Record storage disabled for this trial after {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return directory.relative_to(log_dir).as_posix()


def _git_commit() -> str | None:
    """HEAD commit of the *current working directory's* repository, if any.

    This is deliberately the caller's repo (the code driving the eval), not
    Inspect Robots's own install. A ``-dirty`` suffix is appended when the working
    tree has uncommitted changes, so a log never silently claims a clean commit.
    """

    def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    head = _git("rev-parse", "HEAD")
    if head is None or head.returncode != 0 or not head.stdout.strip():
        return None
    commit = head.stdout.strip()
    tree = _git("status", "--porcelain")
    if tree is None or tree.returncode != 0:
        return None
    if tree.stdout.strip():
        commit += "-dirty"
    return commit


class _CriticalSinkError(Exception):
    """Carry a critical sink's failure past the remaining sinks' fan-out."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


class _Broadcast:
    """Fan a sink lifecycle out to several sinks, preserving hook order."""

    def __init__(self, sinks: list[LogSink]):
        self._sinks = sinks
        # Critical sinks (the canonical log) are served first, so a failed
        # final write is known before any sink discards its own record.
        self._ordered = sorted(sinks, key=lambda s: not getattr(s, "critical", False))
        policy_message_hooks: list[Callable[[int, Sequence[Any]], None]] = []
        for sink in sinks:
            hook = getattr(sink, "log_policy_messages", None)
            if callable(hook):
                policy_message_hooks.append(hook)
        self._policy_message_hooks = policy_message_hooks
        # Transcript hooks that raised this trial: skipped until the next
        # on_trial_start, preserving plan 0020's per-trial failure latch.
        self._failed_message_hooks: set[int] = set()
        if policy_message_hooks:
            self.log_policy_messages = self._fan_policy_messages

    @staticmethod
    def _safe_call(
        sink: Any,
        method_name: str,
        fn: Callable[..., Any],
        *args: Any,
        may_be_critical: bool = True,
    ) -> bool:
        """Call one sink hook, isolating its failure; return whether it succeeded.

        A failing sink warns instead of aborting the eval or starving the other
        sinks (#511), unless it declares ``critical = True`` (the canonical
        ``JsonLogSink``): then the error is re-raised once every sink has been
        offered the hook, so a lost eval log is never reported as success.
        """
        try:
            fn(*args)
        except (SafetyAbort, EmbodimentFault):
            raise
        except Exception as exc:
            if may_be_critical and getattr(sink, "critical", False):
                raise _CriticalSinkError(exc) from exc
            sink_name = type(sink).__name__ if sink is not None else "LogSink"
            warnings.warn(
                f"LogSink {sink_name}.{method_name}() failed with {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        return True

    def _fan_out(self, method_name: str, *args: Any, optional: bool = False) -> None:
        """Offer one hook to every sink, then surface a critical sink's failure.

        ``optional`` hooks are duck-typed: sinks without a callable attribute
        of that name are skipped. Critical sinks run first; if one fails,
        sinks marked ``discards_on_eval_end`` (the live snapshot, which deletes
        itself) are skipped so the only surviving record of the run is kept.
        """
        failure: Exception | None = None
        for s in self._ordered:
            hook: Any = getattr(s, method_name, None)
            if optional and not callable(hook):
                continue
            if (
                failure is not None
                and method_name == "on_eval_end"
                and getattr(s, "discards_on_eval_end", False)
            ):
                # Keep the record; tell the sink so a later run does not
                # delete it either (eval_set reuses sinks across tasks).
                retain = getattr(s, "retain_snapshot", None)
                if callable(retain):
                    retain()
                continue
            try:
                self._safe_call(s, method_name, hook, *args)
            except _CriticalSinkError as exc:
                if failure is None:
                    failure = exc.original
                else:
                    warnings.warn(
                        f"LogSink {type(s).__name__}.{method_name}() also failed with "
                        f"{type(exc.original).__name__}: {exc.original}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        if failure is not None:
            raise failure

    def _fan_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        for index, hook in enumerate(self._policy_message_hooks):
            if index in self._failed_message_hooks:
                continue
            sink = getattr(hook, "__self__", None)
            # Never critical: rollout downgrades transcript errors to "streaming
            # off" anyway, so a failing hook just warns and latches for the trial.
            ok = self._safe_call(
                sink, "log_policy_messages", hook, t, messages, may_be_critical=False
            )
            if not ok:
                self._failed_message_hooks.add(index)

    def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
        """Offer the resolved spaces to sinks that declare a bind_spaces hook.

        Duck-typed like ``log_policy_messages``: sinks without the attribute
        are unaffected, so the sink Protocol is unchanged.
        """
        self._fan_out("bind_spaces", action_space, observation_space, optional=True)

    def bind_frames_dir(self, frames_dir: str | None) -> None:
        """Offer the run's frame directory to sinks that declare the optional hook."""
        self._fan_out("bind_frames_dir", frames_dir, optional=True)

    def bind_scenes(self, scenes: Sequence[Scene]) -> None:
        """Offer the run's scenes to sinks that declare the optional hook."""
        self._fan_out("bind_scenes", scenes, optional=True)

    def on_eval_start(self, spec: EvalSpec) -> None:
        self._fan_out("on_eval_start", spec)

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        self._failed_message_hooks.clear()
        self._fan_out("on_trial_start", scene_id, epoch)

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        self._fan_out("log_step", t, observation, action, result)

    def on_trial_end(self, record: Record) -> None:
        self._fan_out("on_trial_end", record)

    def on_eval_end(self, log: EvalLog) -> None:
        self._fan_out("on_eval_end", log)


def _survivor_warning(log: EvalLog) -> str | None:
    """Describe a "success" whose metrics no clean scene backs, else ``None``.

    ``fail_on_error`` decides whether errored trials fail the run. When it
    tolerates them yet every scene errored, the metrics rest on whichever
    trials survived and can look entirely ordinary (issue #440), so callers
    and readers are warned instead of the status being overridden.
    """
    if log.status != "success" or not log.samples:
        return None
    if any(scene.status != "error" for scene in log.samples):
        return None
    errored, total = log.results.errored_trials, log.results.total_trials
    return (
        f"no scene completed cleanly ({errored} of {total} trial(s) errored); "
        "metrics may rest on a surviving minority of trials"
    )


def eval(
    task: Task | str,
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    log_dir: str = "logs",
    sinks: list[LogSink] | None = None,
    seed: int | None = 0,
    fail_on_error: bool | float = False,
    action_manager: ActionManager | None = None,
    guard: Guard | Sequence[str | Guard] | None = None,
    remap: dict[str, str] | None = None,
    store_frames: bool = False,
    store_actions: bool = True,
    store_records: bool = True,
    operator_input: OperatorInput | None = None,
    before_scoring: Callable[[Record, Scene], None] | None = None,
    grader: Grader | str | None = None,
    environment_id: str | None = None,
    environment_revision: str | None = None,
    policy_checkpoint: str | None = None,
    benchmark: Benchmark | None = None,
) -> list[EvalLog]:
    """Run ``task`` with ``policy`` on ``embodiment``; return ``[EvalLog]``.

    ``task``/``policy``/``embodiment`` may be objects or **registry names**
    (e.g. ``policy="scripted"``), resolved through the registry — the Inspect-style
    ergonomic that keeps logs and the CLI reproducible. An embodiment resolved
    from a registry name is owned by ``eval()`` and is closed when the run
    finishes (even on a halt); a caller-constructed embodiment object stays
    open — the caller owns its lifecycle.

    ``seed=None`` draws a fresh seed from the OS and records it in the log, so
    an "unseeded" run remains reproducible after the fact (and is distinct from
    ``seed=0``).

    ``fail_on_error`` follows Inspect semantics for ``PolicyError`` (``True`` =
    fail on first, ``False`` = never, ``0<x<1`` = proportion, ``x>1`` = count),
    checked after every trial. ``EmbodimentFault``/``SafetyAbort`` always halt
    regardless. Errored trials are recorded (with their partial trajectory
    delivered to sinks) but never scored, so a failed trial cannot masquerade
    as data in the metrics; it stays visible via ``SceneResult.status`` and an
    empty entry in ``SceneResult.epochs``.

    A run in which **every** trial errored (nothing was scored) always ends
    with ``status == "error"``, regardless of ``fail_on_error``. A run in which
    no scene completed cleanly keeps the status ``fail_on_error`` gives it, but
    emits a ``UserWarning`` (issue #440): its metrics may rest on a surviving
    minority of trials.

    Ctrl-C during a rollout records the partial trial and writes a log with
    ``status == "cancelled"``, then re-raises the interrupt (as a
    ``KeyboardInterrupt`` subclass chaining the original) after
    ``on_eval_end`` completes. An interrupt outside the rollout
    call (during scoring, reducers, or log assembly), or a second interrupt
    during the cancellation handlers, may still prevent the log from being
    written.

    When ``store_frames`` is set, camera frames are streamed to
    ``<log_dir>/frames`` as binary side-cars (R5) rather than kept in memory.

    ``store_actions`` defaults to ``True`` and writes each trial's complete
    executed action sequence to ``<log_dir>/actions`` as an atomic JSONL
    side-car. The ``actions`` trial-metadata key is framework-reserved. Set
    ``store_actions`` to ``False`` to disable these files.

    ``store_records`` defaults to ``True`` and writes one directory per trial
    under ``<log_dir>/records/<run>/``: ``record.json`` (the record id, the
    resolved per-trial configuration with credential-like values redacted, the
    termination, and every event) and ``chunks.jsonl`` (every action chunk as
    the policy produced it, with its inference latency). The log names each
    trial's directory in ``SceneResult.record_dirs``. A write failure degrades
    to a ``RuntimeWarning``.

    ``benchmark`` records the [`Benchmark`][inspect_robots.benchmark.Benchmark]
    this task runs in (its name and version) as ``EvalSpec.benchmark``;
    [`eval_benchmark`][inspect_robots.benchmark.eval_benchmark] sets it.

    ``guard`` is a guard object, or a list of registered guard specs
    (``"clamp"``, ``"delta-limit:max_delta=0.02"``) and guard objects applied
    in order; ``None`` passes every action through.

    ``operator_input`` supplies attended-console input; ``None`` disables the channel.

    Before a grader runs, an embodiment's optional duck-typed
    ``observe_parked()`` hook may move the robot to its parked/rest pose so the
    cameras see the scene unobstructed and return one fresh ``Observation``.
    Returning ``None`` declines. Other failures degrade with a
    ``RuntimeWarning`` and grading uses the last-step frames, except
    ``SafetyAbort`` and ``EmbodimentFault``, which halt the eval.

    ``before_scoring`` is called exactly once per trial that will be scored
    (never for errored or cancelled trials, which are recorded but not
    scored), after the rollout returns and before the scorers run. It may
    mutate the record — e.g. capture ``Record.operator_judgement`` (R6)
    so the ``operator`` scorer can read it, and ``Record.operator_note``
    alongside it, which is recorded but never scored. Exceptions it raises
    propagate to the caller. Note this fires on the *other* side of scoring
    from ``LogSink.on_trial_end``.

    ``grader`` is the component form of the same seam: a
    [`Grader`][inspect_robots.grader.Grader] object or registry name whose
    ``grade`` method becomes the pre-scoring hook. Pass either ``grader`` or
    ``before_scoring``, not both (``ConfigError``); disable grading with
    ``grader=None`` (the registry name ``"none"`` is CLI vocabulary, not an
    API value).

    Raises [`CompatibilityError`][inspect_robots.errors.CompatibilityError] (fail fast, before any
    rollout) if the policy and embodiment are incompatible, and
    [`ConfigError`][inspect_robots.errors.ConfigError] for an invalid epoch reducer
    (including ``pass_at_<k>`` with fewer than ``k`` planned epochs) or a
    grader whose ``preflight()`` request is rejected (checked before any
    string component is resolved). A trial the grader tried and failed to
    grade is scored as an abstention by the ``operator`` scorer, and the run
    then ends with ``status == "error"`` and an "N of M trial(s) ungraded"
    message.
    """
    if not isinstance(fail_on_error, bool) and not (
        isinstance(fail_on_error, (int, float))
        and math.isfinite(fail_on_error)
        and fail_on_error >= 0
    ):
        raise ConfigError(
            f"fail_on_error must be a boolean or finite float >= 0, got {fail_on_error!r}"
        )

    from inspect_robots.registry import resolve

    before_scoring, resolved_grader = _grading_hook(grader, before_scoring)
    # Before resolving string components: a grader that would reject every
    # trial must fail before any robot connection is opened (plan 0085).
    _preflight_grader(resolved_grader)
    owns_embodiment = isinstance(embodiment, str)
    task = cast(Task, resolve("task", task)) if isinstance(task, str) else task
    policy = cast(Policy, resolve("policy", policy)) if isinstance(policy, str) else policy
    embodiment = (
        cast(Embodiment, resolve("embodiment", embodiment))
        if isinstance(embodiment, str)
        else embodiment
    )
    try:
        return _run_eval(
            task,
            policy,
            embodiment,
            log_dir=log_dir,
            sinks=sinks,
            seed=seed,
            fail_on_error=fail_on_error,
            action_manager=action_manager,
            guard=guard,
            remap=remap,
            store_frames=store_frames,
            store_actions=store_actions,
            store_records=store_records,
            operator_input=operator_input,
            before_scoring=before_scoring,
            grader_identity=_grader_identity(resolved_grader),
            environment_id=environment_id,
            environment_revision=environment_revision,
            policy_checkpoint=policy_checkpoint,
            benchmark=benchmark,
        )
    finally:
        # Close what we opened: a registry-resolved embodiment is released even
        # when the run halts, so a real robot never leaks its connection.
        if owns_embodiment:
            embodiment.close()


def _run_eval(
    task: Task,
    policy: Policy,
    embodiment: Embodiment,
    *,
    log_dir: str,
    sinks: list[LogSink] | None,
    seed: int | None,
    fail_on_error: bool | float,
    action_manager: ActionManager | None,
    guard: Guard | Sequence[str | Guard] | None,
    remap: dict[str, str] | None,
    store_frames: bool,
    store_actions: bool,
    store_records: bool,
    operator_input: OperatorInput | None,
    before_scoring: Callable[[Record, Scene], None] | None,
    grader_identity: tuple[str | None, dict[str, Any]],
    environment_id: str | None = None,
    environment_revision: str | None = None,
    policy_checkpoint: str | None = None,
    benchmark: Benchmark | None = None,
) -> list[EvalLog]:
    """The body of [`eval`][inspect_robots.eval.eval], after resolution/ownership."""
    from inspect_robots.logging.json_log import JsonLogSink
    from inspect_robots.session import adopts_embodiment_verdict
    from inspect_robots.types import Observation

    # Embodiment-adaptive policies (plan 0008 §3c): an optional bind() hook
    # runs before the compatibility check so the policy can adopt the
    # embodiment's spaces. Duck-typed — bind is not part of the Policy
    # Protocol, so existing policies are untouched.
    bind = getattr(policy, "bind", None)
    if callable(bind):
        bind(embodiment.info)
    _bind_contributed_tools(policy, embodiment)

    # Fail fast on incompatible pairings before touching any hardware/sim.
    # This also validates the embodiment rate needed by a seconds-based task
    # before the resolved horizon is exposed to an adapter (plan 0026).
    report = assert_compatible(policy, embodiment, task, remap=remap)
    # Soft mismatches (e.g. action-frame differences) permit rollout with
    # reduced guarantees, but the caller must see them: surface every
    # warning issue as a Python UserWarning instead of silently dropping
    # the report (issue #593).
    for issue in report.warnings:
        warnings.warn(
            f"compatibility warning [{issue.code}]: {issue.message}",
            UserWarning,
            stacklevel=3,  # point at the caller's eval(...) call, like the survivor warning
        )
    task_envelope = task.resolve_envelope(embodiment.info.control_hz)

    # Horizon-aware embodiments (plan 0013): an optional bind_task() hook runs
    # after compatibility so the adapter receives the resolved rollout
    # envelope (e.g. for an operator countdown) before any hardware is
    # touched. Duck-typed — bind_task is not part of the Embodiment Protocol.
    bind_task = getattr(embodiment, "bind_task", None)
    if callable(bind_task):
        bind_task(task_envelope)

    # Horizon-aware policies (plan 0013 anticipated this): optional bind_task() hook
    policy_bind_task = getattr(policy, "bind_task", None)
    if callable(policy_bind_task):
        policy_bind_task(task_envelope)

    epoch_spec = task.epoch_spec
    # Resolve every scene's rubric scorers before any rollout, so a bad registry
    # name or a duplicate fails fast instead of after robot time is spent.
    scene_rubrics = {scene.id: task.rubrics_for(scene) for scene in task.scenes}
    scene_scorers = {
        scene_id: [(rubric.name, rubric.resolve_scorer()) for rubric in rubrics]
        for scene_id, rubrics in scene_rubrics.items()
    }
    rubric_names = list(
        dict.fromkeys(name for pairs in scene_scorers.values() for name, _ in pairs)
    )
    # Fail fast on an unknown/invalid epoch reducer, before any rollout runs.
    try:
        get_reducer(epoch_spec.reducer)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    # A reducer that needs more epochs than planned can never succeed; say so
    # before spending robot time. Abstentions are still checked at scoring.
    min_epochs = reducer_min_epochs(epoch_spec.reducer)
    if epoch_spec.count < min_epochs:
        raise ConfigError(
            f"epoch reducer {epoch_spec.reducer!r} needs at least {min_epochs} epochs, "
            f"but the task plans {epoch_spec.count}; raise Epochs.count or lower k"
        )

    if seed is None:
        # Draw and record a real seed so the run stays reproducible after the
        # fact; None must not silently alias seed=0 (see derive_seed).
        seed = int.from_bytes(os.urandom(4), "little")

    sink_list: list[LogSink] = sinks if sinks is not None else [JsonLogSink(log_dir)]
    bus = _Broadcast(sink_list)
    action_manager = action_manager or DefaultActionManager(policy.config.replan_interval)
    if isinstance(guard, (list, tuple)):
        guard = build_guard(guard, embodiment.info.action_space)
    guard = cast(Guard | None, guard) or PassThroughGuard()

    # One subdirectory per run: trial ids repeat across runs (scene-epoch),
    # so a shared directory would silently overwrite the previous run's
    # frames or transcripts.
    run_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"

    frame_store: FrameStore | None = None
    if store_frames:
        frame_store = FrameStore(str(Path(log_dir) / "frames" / run_stamp))

    grader_name, grader_config = grader_identity
    spec = EvalSpec(
        task=task.name,
        policy=policy.info.name,
        embodiment=embodiment.info.name,
        created=_now_iso(),
        inspect_robots_version=__version__,
        git_commit=_git_commit(),
        policy_config=_describe(asdict(policy.config)),
        embodiment_info={
            "control_hz": embodiment.info.control_hz,
            "is_simulated": embodiment.info.is_simulated,
            "capabilities": sorted(embodiment.info.capabilities),
            "morphology": getattr(embodiment.info, "morphology", None),
            "evaluation_interface": _evaluation_interface(embodiment),
        },
        grader=grader_name,
        grader_config=grader_config,
        seed=seed,
        max_steps=task_envelope.max_steps,
        max_seconds=task.max_seconds,
        rubrics=[rubric.describe() for rubric in task.task_rubrics],
        reducer=epoch_spec.reducer,
        benchmark=benchmark.describe() if benchmark is not None else None,
        environment_id=environment_id or getattr(embodiment.info, "environment_id", None),
        environment_revision=environment_revision
        or getattr(embodiment.info, "environment_revision", None),
        policy_checkpoint=policy_checkpoint or getattr(policy.info, "checkpoint", None),
    )
    records_dir: Path | None = Path(log_dir) / "records" / run_stamp if store_records else None
    run_configuration = {
        "task": task.name,
        "inspect_robots_version": spec.inspect_robots_version,
        "git_commit": spec.git_commit,
        "seed": seed,
        "max_steps": task_envelope.max_steps,
        "max_seconds": task.max_seconds,
        "policy": {
            "name": spec.policy,
            "checkpoint": spec.policy_checkpoint,
            "config": spec.policy_config,
        },
        "embodiment": {"name": spec.embodiment, **spec.embodiment_info},
        "action_manager": _describe(action_manager),
        "guard": _describe(guard),
        "grader": {"name": spec.grader, "config": spec.grader_config},
    }
    bus.bind_spaces(embodiment.info.action_space, embodiment.info.observation_space)
    bus.bind_frames_dir(str(frame_store.root) if frame_store is not None else None)
    bus.bind_scenes(task.scenes)
    bus.on_eval_start(spec)

    started = time.perf_counter()
    started_iso = _now_iso()

    scene_results: list[SceneResult] = []
    all_latencies: list[float] = []
    total_steps = 0
    total_trials = 0
    status = "success"
    error: str | None = None
    error_count = 0
    errored_trials = 0
    graded_attempts = 0
    ungraded_trials = 0
    first_grading_error: str | None = None
    abstentions: dict[str, int] = {}

    halted = False
    stopped = False
    cancelled_exc: _CancelledTrial | None = None
    # A SafetyAbort/EmbodimentFault raised by an evaluation-interface hook
    # (outside any trial) is re-raised once the log is written, so eval_set()
    # stops admitting tasks to a platform that just faulted.
    halt_exc: BaseException | None = None
    # A proportion threshold is a share of the whole eval, so the denominator is
    # every trial the run intends to attempt. Using the completed-so-far count
    # made the first error 1/1 = 100%, which trips any threshold below 1.
    planned_trials = len(task.scenes) * epoch_spec.count
    for scene in task.scenes:
        rubrics = scene_rubrics[scene.id]
        # Graders read the scene's effective rubrics, the task's merged with its own.
        graded_scene = replace(scene, rubrics=rubrics)
        per_scorer_scores: dict[str, list[Score]] = {
            name: [] for name, _ in scene_scorers[scene.id]
        }
        epoch_dicts: list[dict[str, float | None]] = []
        judgements: list[str | None] = []
        judgement_sources: list[str | None] = []
        notes: list[str | None] = []
        trial_metadatas: list[dict[str, Any]] = []
        termination_reasons: list[str | None] = []
        termination_details: list[str | None] = []
        operator_messages: list[tuple[dict[str, Any], ...]] = []
        policy_transcripts: list[Any] = []
        record_ids: list[str | None] = []
        record_dirs: list[str | None] = []
        trial_events: list[tuple[dict[str, Any], ...]] = []
        scene_metadata = _json_safe_scene_metadata(scene.metadata)
        scene_status = "success"
        scene_error: str | None = None

        for epoch in range(epoch_spec.count):
            trial_seed = derive_seed(seed, scene.init_seed, epoch)
            # The evaluation interface runs before the trial exists: a failed
            # check stops the run with no trial recorded (a failed home() is
            # parked first, since it may have moved the platform).
            not_ready, pre_trial_halt = _before_trial(embodiment, scene, epoch)
            if not_ready is not None:
                halt_exc = halt_exc or pre_trial_halt
                status = "error"
                error = not_ready
                scene_status = "error"
                scene_error = not_ready
                stopped = True
                break
            try:
                bus.on_trial_start(scene.id, epoch)
            except Exception:
                # A critical sink failing after home() moved the platform:
                # park it before the failure leaves eval().
                _park_before_escape(embodiment)
                raise
            record: Record | None = None
            policy_start_failed = False
            on_trial_start = getattr(policy, "on_trial_start", None)
            if callable(on_trial_start):
                try:
                    on_trial_start(scene.id, epoch, log_dir, run_stamp)
                except Exception as exc:
                    policy_start_failed = True
                    error_count += 1
                    scene_status = "error"
                    scene_error = f"policy.on_trial_start failed: {exc}"
                    record = Record(
                        scene_id=scene.id,
                        epoch=epoch,
                        seed=trial_seed,
                        status="error",
                        error=scene_error,
                    )
            if not policy_start_failed:
                try:
                    record = rollout(
                        policy,
                        embodiment,
                        scene,
                        max_steps=task_envelope.max_steps,
                        horizon_reason=(
                            "max_seconds" if task.max_seconds is not None else "max_steps"
                        ),
                        seed=trial_seed,
                        epoch=epoch,
                        action_manager=action_manager,
                        guard=guard,
                        sink=bus,
                        frame_store=frame_store,
                        operator_input=operator_input,
                    )
                except _CancelledTrial as exc:
                    status = "cancelled"
                    error = str(exc)
                    scene_status = "cancelled"
                    scene_error = error
                    halted = True
                    cancelled_exc = exc
                    record = exc.record
                except (EmbodimentFault, SafetyAbort) as exc:
                    # Hardware/safety failures always halt the whole eval; the
                    # partial trial record (if any) is preserved below.
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                    scene_status = "error"
                    scene_error = error
                    halted = True
                    record = exc.record
                except PolicyError as exc:
                    error_count += 1
                    scene_status = "error"
                    scene_error = f"{type(exc).__name__}: {exc}"
                    record = exc.record or Record(
                        scene_id=scene.id,
                        epoch=epoch,
                        seed=trial_seed,
                        status="error",
                        error=scene_error,
                    )
                except Exception:
                    # Anything else rollout() passes through (a frame store or
                    # critical sink failing mid-trial, a configuration error from
                    # the policy) still leaves a moved platform: park it first.
                    _park_before_escape(embodiment)
                    raise

            if record is not None:
                total_trials += 1
                total_steps += len(record.steps)
                all_latencies.extend(record.inference_latencies)
                if record.status != "success":
                    # Non-successful trials are not scored: a partial trial
                    # must not masquerade as data (e.g. an inf min-distance
                    # poisoning the metric mean). It stays visible via status.
                    epoch_dicts.append({})
                    if record.status == "error":
                        errored_trials += 1
                    judgements.append(None)
                    judgement_sources.append(None)
                    notes.append(None)
                else:
                    if before_scoring is not None:
                        # The only trials the hook sees are the ones scorers
                        # will read — an operator verdict on a crashed trial
                        # would be dead data (errored trials are never scored).
                        if record.operator_judgement is None and not adopts_embodiment_verdict(
                            record, graded_scene
                        ):
                            observe_parked = getattr(embodiment, "observe_parked", None)
                            if callable(observe_parked):
                                try:
                                    parked_observation = observe_parked()
                                except (SafetyAbort, EmbodimentFault):
                                    _park_before_escape(embodiment)
                                    raise
                                except Exception as exc:
                                    warnings.warn(
                                        "embodiment.observe_parked() failed with "
                                        f"{type(exc).__name__}: {exc}; grading from "
                                        "last-step frames",
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                                else:
                                    if isinstance(parked_observation, Observation):
                                        record.parked_observation = parked_observation
                                    elif parked_observation is not None:
                                        warnings.warn(
                                            "embodiment.observe_parked() returned "
                                            f"{type(parked_observation).__name__}; expected "
                                            "Observation or None; grading from last-step frames",
                                            RuntimeWarning,
                                            stacklevel=2,
                                        )
                        try:
                            before_scoring(record, graded_scene)
                        except Exception:
                            # Not BaseException: a Ctrl-C at the judgement prompt
                            # is a cancel, and a cancelled run is never parked.
                            _park_before_escape(embodiment)
                            raise
                        graded_attempts += 1
                        grading_error = record.metadata.get("grading_error")
                        if grading_error and record.operator_judgement is None:
                            ungraded_trials += 1
                            if first_grading_error is None:
                                first_grading_error = str(grading_error)
                    epoch_values: dict[str, float | None] = {}
                    judged = _judged_rubric(record, scene_scorers[scene.id])
                    for rubric_name, scorer in scene_scorers[scene.id]:
                        try:
                            if (
                                judged is not None
                                and rubric_name != judged
                                and _reads_judgement(scorer)
                            ):
                                # One verdict per trial: it counts for the rubric
                                # it was given for; the others abstain until judged.
                                score = Score(value=None, explanation="not judged in the run")
                            else:
                                score = scorer(record, scene.target)
                            value = value_to_float(score.value)
                        except (SafetyAbort, EmbodimentFault):
                            # Halt signals are not scoring errors: containing
                            # them here would let the next rollout start after
                            # an explicit safety abort or a hardware fault.
                            _park_before_escape(embodiment)
                            raise
                        except Exception as exc:
                            # A scorer failure degrades to an error log - it must
                            # never crash the eval and lose the trials that ran.
                            # Shorthand rubrics carry their scorer's name: say it once.
                            subject = (
                                f"scorer {scorer.name!r}"
                                if rubric_name == scorer.name
                                else f"rubric {rubric_name!r} (scorer {scorer.name!r})"
                            )
                            detail = f"{subject} failed: {exc}"
                            scene_status = "error"
                            scene_error = (
                                detail if scene_error is None else f"{scene_error}; {detail}"
                            )
                            if status == "success":
                                status = "error"
                                error = detail
                            continue
                        per_scorer_scores[rubric_name].append(score)
                        epoch_values[rubric_name] = value
                        if value is None:
                            abstentions[rubric_name] = abstentions.get(rubric_name, 0) + 1
                    epoch_dicts.append(epoch_values)
                    # Captured at the same instant as the judgement, on purpose:
                    # these fields are documented as strictly parallel, so a later
                    # mutation (e.g. from policy.on_trial_end) must not be able
                    # to reach one of them and miss the others.
                    judgements.append(record.operator_judgement)
                    judgement_sources.append(judgement_source(record))
                    notes.append(record.operator_note)

                # A never-reset trial must not persist the previous trial's
                # policy state under this trial's identity.
                if not policy_start_failed:
                    on_trial_end = getattr(policy, "on_trial_end", None)
                    if callable(on_trial_end):
                        try:
                            on_trial_end(record, log_dir, run_stamp)
                        except Exception as exc:
                            # Named `detail`, not `note`: a grader's note is a
                            # different thing entirely and is collected just above.
                            detail = f"policy.on_trial_end failed: {exc}"
                            scene_status = "error"
                            scene_error = (
                                detail if scene_error is None else f"{scene_error}; {detail}"
                            )
                            if status == "success":
                                status = "error"
                                error = detail

                if store_actions:
                    actions_path = _write_action_log(
                        record,
                        log_dir,
                        run_stamp,
                        embodiment.info.action_space,
                    )
                    if actions_path is not None:
                        record.metadata["actions"] = actions_path

                events = tuple(
                    {"kind": event.kind, "t": event.t, "data": _describe(event.data)}
                    for event in record.events
                )
                record_dir: str | None = None
                if records_dir is not None:
                    record_dir = _write_record(
                        record,
                        records_dir,
                        Path(log_dir),
                        events,
                        {
                            **run_configuration,
                            "trial_seed": trial_seed,
                            "scene": {
                                "id": scene.id,
                                "instruction": scene.instruction,
                                "init_seed": scene.init_seed,
                                "setup": scene.setup,
                                "metadata": scene_metadata,
                            },
                            "rubrics": [rubric.describe() for rubric in rubrics],
                        },
                        embodiment.info.action_space,
                    )
                record_ids.append(record.record_id)
                record_dirs.append(record_dir)
                trial_events.append(events)
                trial_metadatas.append(record.metadata)
                termination_reasons.append(record.termination_reason)
                termination_details.append(record.termination_detail)
                operator_messages.append(
                    tuple(
                        {
                            "t": event.t,
                            "text": event.data["text"],
                            "source": event.data.get("source", "console"),
                        }
                        for event in record.events
                        if event.kind == "operator_message"
                    )
                )
                policy_transcripts.append(record.policy_transcript)
                try:
                    bus.on_trial_end(record)
                except Exception:
                    if cancelled_exc is None:
                        _park_before_escape(embodiment)
                    raise

            # Park after every trial that ran, so the scene can be reset by hand,
            # and after a policy that failed to start: home() already moved the
            # platform. A cancelled run is left to close(): the operator asked
            # it to stop.
            if cancelled_exc is None:
                park_failure, park_halt = _park(embodiment)
                if park_failure is not None and halted:
                    warnings.warn(park_failure, RuntimeWarning, stacklevel=2)
                    # The trial's own halt ends only this task, but a faulted
                    # park means the platform itself must not run another one.
                    halt_exc = halt_exc or park_halt
                elif park_failure is not None:
                    status = "error"
                    error = park_failure
                    scene_status = "error"
                    scene_error = (
                        park_failure if scene_error is None else f"{scene_error}; {park_failure}"
                    )
                    halt_exc = halt_exc or park_halt
                    halted = True

            if halted:
                stopped = True
                break
            if _should_fail(fail_on_error, error_count, planned_trials):
                # Checked after every trial, so fail_on_error=True stops at the
                # first PolicyError instead of finishing the scene's epochs.
                status = "error"
                error = f"fail_on_error threshold exceeded ({error_count} errors)"
                stopped = True
                break

        reduced: dict[str, float | None] = {}
        for name, scene_scores in per_scorer_scores.items():
            if not scene_scores:
                continue
            try:
                reduced[name] = value_to_float(
                    reduce_scores(epoch_spec.reducer, scene_scores).value
                )
            except Exception as exc:
                # A reducer failure (e.g. pass_at_k over fewer epochs than k
                # after a halt, or mean over categorical scores) degrades to an
                # error log — it must never crash the eval and lose the log.
                detail = f"reducer {epoch_spec.reducer!r} failed for rubric {name!r}: {exc}"
                scene_status = "error"
                scene_error = detail if scene_error is None else f"{scene_error}; {detail}"
                if status == "success":
                    status = "error"
                    error = detail

        scene_results.append(
            SceneResult(
                scene_id=scene.id,
                status=scene_status,
                reduced=reduced,
                epochs=tuple(epoch_dicts),
                error=scene_error,
                instruction=scene.instruction,
                scene_metadata=scene_metadata,
                operator_judgements=tuple(judgements),
                judgement_sources=tuple(judgement_sources),
                operator_notes=tuple(notes),
                trial_metadata=tuple(trial_metadatas),
                termination_reasons=tuple(termination_reasons),
                termination_details=tuple(termination_details),
                rubrics=tuple(rubric.describe() for rubric in rubrics),
                operator_messages=tuple(operator_messages),
                policy_transcripts=tuple(policy_transcripts),
                record_ids=tuple(record_ids),
                record_dirs=tuple(record_dirs),
                events=tuple(trial_events),
            )
        )
        if stopped:
            break

    if status == "success" and total_trials > 0 and errored_trials == total_trials:
        # Every trial errored: there is no surviving data for fail_on_error's
        # flaky-trial tolerance to protect, and a "success" log would hide a
        # total failure (issue #73).
        status = "error"
        error = f"all {total_trials} trial(s) errored; nothing was scored"

    if ungraded_trials:
        # The grader tried and failed on these trials (plan 0085). They abstain
        # rather than score as failures, but a run with ungraded trials must
        # never read as a clean success. A more specific error keeps its
        # message and gains the count.
        count = f"{ungraded_trials} of {graded_attempts} trial(s) ungraded"
        if status == "success":
            status = "error"
            error = f"{count}: grader failed ({first_grading_error})"
        else:
            error = "; ".join(part for part in (error, count) if part)

    metrics: dict[str, float | None] = {}
    for name in rubric_names:
        vals = [sr.reduced[name] for sr in scene_results if name in sr.reduced]
        if vals:
            # Abstentions carry no verdict, so they are left out of the mean;
            # a rubric whose scorer abstained on every scene reports None, not 0.0.
            voted = [v for v in vals if v is not None]
            metrics[name] = mean(voted) if voted else None

    stats = EvalStats(
        started_at=started_iso,
        completed_at=_now_iso(),
        duration_s=time.perf_counter() - started,
        total_steps=total_steps,
        mean_inference_latency_s=(mean(all_latencies) if all_latencies else None),
        frames_dir=str(frame_store.root) if frame_store is not None else None,
        records_dir=str(records_dir) if records_dir is not None else None,
    )
    log = EvalLog(
        version=EvalLog.SCHEMA_VERSION,
        status=status,
        eval=spec,
        results=EvalResults(
            total_scenes=len(scene_results),
            total_trials=total_trials,
            metrics=metrics,
            errored_trials=errored_trials,
            abstentions=abstentions,
        ),
        stats=stats,
        samples=tuple(scene_results),
        error=error,
    )
    try:
        bus.on_eval_end(log)
    except Exception as exc:
        # A failed final write must not swallow the user's Ctrl-C or a hook
        # halt: eval_set would otherwise treat the OSError as a task error and
        # continue on a platform that asked to stop.
        if cancelled_exc is not None:
            raise cancelled_exc from exc
        if halt_exc is not None:
            raise halt_exc from exc
        raise
    survivor_warning = _survivor_warning(log)
    if survivor_warning is not None:
        warnings.warn(survivor_warning, UserWarning, stacklevel=3)
    if cancelled_exc is not None:
        raise cancelled_exc
    if halt_exc is not None:
        # The log is written; the caller can still reach it from the halt.
        with contextlib.suppress(AttributeError):
            halt_exc.log = log  # type: ignore[attr-defined]
        raise halt_exc
    return [log]


def _bind_contributed_tools(policy: Policy, embodiment: Embodiment) -> None:
    """Hand the embodiment's optional ``contribute_tools`` library to a policy that takes tools.

    Both hooks are duck-typed: an embodiment may offer
    ``contribute_tools(action_space) -> ToolLibrary`` and a tool-using policy
    ``bind_tools(library)``. Nothing happens unless both exist.
    """
    from inspect_robots.tools import ToolLibrary

    contribute = getattr(embodiment, "contribute_tools", None)
    bind_tools = getattr(policy, "bind_tools", None)
    if not callable(contribute) or not callable(bind_tools):
        return
    tools = contribute(embodiment.info.action_space)
    if not isinstance(tools, ToolLibrary):
        raise ConfigError(
            f"embodiment {embodiment.info.name!r}: contribute_tools returned "
            f"{type(tools).__name__}; expected ToolLibrary"
        )
    bind_tools(tools)


_EVALUATION_HOOKS = ("health", "prepare_trial", "home", "park")


def _evaluation_interface(embodiment: Embodiment) -> list[str]:
    """The evaluation-interface hooks this embodiment offers, in call order."""
    return [name for name in _EVALUATION_HOOKS if callable(getattr(embodiment, name, None))]


def _before_trial(
    embodiment: Embodiment, scene: Scene, epoch: int
) -> tuple[str | None, BaseException | None]:
    """Run ``health``, ``prepare_trial`` and ``home`` if offered; return why not to start.

    Any exception, a failed ``HealthReport`` or a not-ready ``Readiness`` stops
    the run: starting a trial on a platform that just said no is never safe.
    The second value is a raised ``SafetyAbort``/``EmbodimentFault``, which
    ``eval()`` re-raises after writing its log so ``eval_set()`` stops too.
    """
    failure = _before_trial_message(embodiment, scene, epoch)
    return (None, None) if failure is None else failure


def _halt_of(exc: BaseException) -> BaseException | None:
    """``exc`` when it is a halt signal (``SafetyAbort``/``EmbodimentFault``), else None."""
    return exc if isinstance(exc, (SafetyAbort, EmbodimentFault)) else None


def _before_trial_message(
    embodiment: Embodiment, scene: Scene, epoch: int
) -> tuple[str, BaseException | None] | None:
    trial = f"scene {scene.id!r} epoch {epoch}"
    health = getattr(embodiment, "health", None)
    if callable(health):
        try:
            report = health()
        except Exception as exc:
            return (
                f"embodiment.health() failed before {trial}: {type(exc).__name__}: {exc}",
                _halt_of(exc),
            )
        if not isinstance(report, HealthReport):
            return (
                f"embodiment.health() returned {type(report).__name__}; expected HealthReport",
                None,
            )
        if not report.ok:
            return f"embodiment health check failed before {trial}: {report.summary()}", None
    prepare_trial = getattr(embodiment, "prepare_trial", None)
    if callable(prepare_trial):
        try:
            readiness = prepare_trial(scene)
        except Exception as exc:
            return (
                f"embodiment.prepare_trial() failed before {trial}: {type(exc).__name__}: {exc}",
                _halt_of(exc),
            )
        if not isinstance(readiness, Readiness):
            return (
                f"embodiment.prepare_trial() returned {type(readiness).__name__}; "
                "expected Readiness",
                None,
            )
        if not readiness.ready:
            return (
                f"embodiment not ready for {trial}: {readiness.reason or 'no reason given'}",
                None,
            )
    home = getattr(embodiment, "home", None)
    if callable(home):
        try:
            home()
        except Exception as exc:
            message = f"embodiment.home() failed before {trial}: {type(exc).__name__}: {exc}"
            # home() may have moved the platform before failing, so it is parked
            # like any trial that ran; park is documented as safe after a fault.
            park_failure, park_halt = _park(embodiment)
            text = message if park_failure is None else f"{message}; {park_failure}"
            return text, _halt_of(exc) or park_halt
    return None


def _reads_judgement(scorer: object) -> bool:
    """Whether ``scorer`` scores from the trial's recorded judgement (the operator family)."""
    return callable(getattr(scorer, "score_annotations", None))


def _judged_rubric(record: Record, scorers: Sequence[tuple[str, object]]) -> str | None:
    """Which judgement rubric the trial's single verdict belongs to, when there are several.

    ``None`` with at most one judgement rubric or no verdict: the verdict then
    applies as before. Otherwise the grader's ``metadata["judged_rubric"]``
    when it names one of them, else the first; the choice is recorded there so
    re-scoring from annotations scopes the verdict the same way.
    """
    names = [name for name, scorer in scorers if _reads_judgement(scorer)]
    if len(names) < 2 or record.operator_judgement is None:
        # The verdict applies unscoped; drop any grader hint so annotations
        # lifted from the log scope it exactly as this eval scored it.
        record.metadata.pop("judged_rubric", None)
        return None
    chosen = record.metadata.get("judged_rubric")
    judged = chosen if chosen in names else names[0]
    record.metadata["judged_rubric"] = judged
    return str(judged)


def _park_before_escape(embodiment: Embodiment) -> None:
    """Park after a trial whose grading or scoring is about to raise out of ``eval()``.

    The robot moved during the trial, so it is parked as a completed trial
    would be; a park failure is warned about, never allowed to replace the
    exception that is escaping.
    """
    failure, halt = _park(embodiment)
    if halt is not None:
        # A faulted park outranks the escaping error: eval_set() must stop
        # before another task touches the platform.
        raise halt
    if failure is not None:
        warnings.warn(failure, RuntimeWarning, stacklevel=3)


def _park(embodiment: Embodiment) -> tuple[str | None, BaseException | None]:
    """Call ``park`` if offered; return the failure as a message and any halt it raised."""
    park = getattr(embodiment, "park", None)
    if not callable(park):
        return None, None
    try:
        park()
    except Exception as exc:
        return f"embodiment.park() failed: {type(exc).__name__}: {exc}", _halt_of(exc)
    return None, None


def _should_fail(fail_on_error: bool | float, errors: int, trials: int) -> bool:
    """Inspect-style ``fail_on_error`` evaluation for PolicyError-class failures.

    ``trials`` is the number of trials the eval plans to run, not the number
    completed so far: a proportion threshold is a share of the whole eval.
    """
    if not fail_on_error or errors == 0:  # covers False, 0, 0.0
        return False
    if fail_on_error is True:
        return True
    if 0 < fail_on_error < 1:
        return trials > 0 and (errors / trials) >= fail_on_error
    return errors >= fail_on_error


def _component_name(component: Policy | Embodiment) -> str:
    """Return a component's declared name without masking an earlier failure."""
    try:
        return component.info.name
    except Exception:
        return type(component).__name__


def _safe_info_attr(component: object, attr: str) -> str | None:
    try:
        info = getattr(component, "info", None)
        value = getattr(info, attr, None)
        return str(value) if value is not None else None
    except Exception:
        return None


def _error_log_for(
    task: Task | str,
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    seed: int | None,
    exc: Exception,
    environment_id: str | None = None,
    environment_revision: str | None = None,
    policy_checkpoint: str | None = None,
    benchmark: Benchmark | None = None,
) -> EvalLog:
    """Describe a task failure that occurred before or outside log production."""
    now = _now_iso()
    return EvalLog(
        version=EvalLog.SCHEMA_VERSION,
        status="error",
        eval=EvalSpec(
            task=task if isinstance(task, str) else task.name,
            policy=policy if isinstance(policy, str) else _component_name(policy),
            embodiment=(embodiment if isinstance(embodiment, str) else _component_name(embodiment)),
            created=now,
            inspect_robots_version=__version__,
            git_commit=_git_commit(),
            seed=seed,
            max_steps=None if isinstance(task, str) else task.max_steps,
            max_seconds=None if isinstance(task, str) else task.max_seconds,
            environment_id=environment_id
            or (
                _safe_info_attr(embodiment, "environment_id")
                if not isinstance(embodiment, str)
                else None
            ),
            environment_revision=environment_revision
            or (
                _safe_info_attr(embodiment, "environment_revision")
                if not isinstance(embodiment, str)
                else None
            ),
            policy_checkpoint=policy_checkpoint
            or (_safe_info_attr(policy, "checkpoint") if not isinstance(policy, str) else None),
            benchmark=benchmark.describe() if benchmark is not None else None,
        ),
        results=EvalResults(total_scenes=0, total_trials=0),
        stats=EvalStats(
            started_at=now,
            completed_at=now,
            duration_s=0.0,
            total_steps=0,
        ),
        samples=(),
        error=f"{type(exc).__name__}: {exc}",
    )


def eval_set(
    tasks: Task | str | Sequence[Task | str],
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    log_dir: str = "logs",
    sinks: list[LogSink] | None = None,
    seed: int | None = 0,
    fail_on_error: bool | float = False,
    action_manager: ActionManager | None = None,
    guard: Guard | Sequence[str | Guard] | None = None,
    remap: dict[str, str] | None = None,
    store_frames: bool = False,
    store_actions: bool = True,
    store_records: bool = True,
    operator_input: OperatorInput | None = None,
    before_scoring: Callable[[Record, Scene], None] | None = None,
    grader: Grader | str | None = None,
    retry_attempts: int = 0,
    max_workers: int = 1,
    environment_id: str | None = None,
    environment_revision: str | None = None,
    policy_checkpoint: str | None = None,
    benchmark: Benchmark | None = None,
) -> tuple[bool, list[EvalLog]]:
    """Run a set of tasks and return ``(success, logs)`` (mirrors Inspect AI).

    ``success`` is ``True`` iff every returned log has ``status == "success"``.
    A task that raises before or without producing a log contributes one
    ``status="error"`` log carrying the exception text, and the remaining
    tasks still run. A ``SafetyAbort`` or ``EmbodimentFault`` that escapes
    ``eval()`` (raised outside a trial, including from the embodiment's
    ``health``/``prepare_trial``/``home``/``park`` hooks) and
    ``KeyboardInterrupt`` still propagate and stop the set. A halt inside a
    trial ends that task with an error log and the set continues to the next
    task.
    ``CompatibilityError``, unknown policy or embodiment registry names, and
    task-factory ``ConfigError`` are therefore reported once per affected
    task. Only grading configuration errors raised before the task loop
    propagate.

    With a string embodiment, a non-safety exception from ``close()`` can
    produce an error row even when that task's completed JSON log is already
    on disk.

    ``store_actions`` and ``store_records`` follow ``eval()``'s default-on side-car contracts.

    ``grader``/``before_scoring`` follow ``eval()``'s contract (one pre-scoring
    hook, not both). Sequential runs resolve the grader once and share it;
    its optional ``preflight()`` runs before the first task and a rejection
    propagates instead of becoming a per-task error log. Parallel runs
    construct and preflight a grader per task before opening its components;
    grading configuration errors also propagate and stop task admission.

    Caller-supplied ``sinks`` are reused across the set's sequential runs. Each
    sink must reset its per-run state in ``on_eval_start`` and tolerate one
    complete lifecycle per task.

    ``max_workers > 1`` runs independent simulated tasks in threads. Tasks,
    policy, embodiment and optional grader must be registered names whose
    factories return fresh instances safe to run concurrently. Caller-owned
    sinks, action_managers, guards and hooks are rejected. Each task owns its
    resources and logs; returned logs retain input order. An escaping halt or
    interrupt stops admission and waits for active tasks to finish and close.
    Threads do not accelerate CPU-bound Python code or interrupt blocked calls.

    Resumption of a partially-completed run (skipping already-finished scenes via
    a stable run id) is reserved for a follow-up: ``retry_attempts`` is accepted
    now so callers don't get retrofitted, but is not yet honored.
    """
    from inspect_robots._parallel import (
        HaltPreservingExitStack,
        run_parallel,
        validate_max_workers,
    )

    validate_max_workers(max_workers)
    task_list = [tasks] if isinstance(tasks, Task | str) else list(tasks)
    if max_workers > 1:
        if not (
            all(isinstance(task, str) for task in task_list)
            and isinstance(policy, str)
            and isinstance(embodiment, str)
            and (grader is None or isinstance(grader, str))
        ):
            raise ConfigError(
                "parallel eval_set requires registered task, policy, embodiment "
                "and grader names; live objects cannot be shared"
            )
        # Guard specs are names, so each worker builds its own guards from them.
        guard_specs_only = isinstance(guard, (list, tuple)) and all(
            isinstance(spec, str) for spec in guard
        )
        shared_guard = None if guard_specs_only else guard
        if any(
            value is not None
            for value in (sinks, action_manager, shared_guard, operator_input, before_scoring)
        ):
            raise ConfigError(
                "parallel eval_set does not accept shared sinks, action_manager, "
                "guard objects, operator_input or before_scoring"
            )
        from inspect_robots.registry import registered, resolve

        registered("task")  # Finish plugin discovery before workers access the registry.

        def run_task(task: Task | str) -> list[EvalLog]:
            # Configuration rejection must escape the per-task error handler
            # and occur before opening components, as in sequential eval_set.
            _, task_grader = _grading_hook(grader, None)
            _preflight_grader(task_grader)
            try:
                with HaltPreservingExitStack() as resources:
                    task_policy = cast(Policy, resolve("policy", policy))
                    close_policy = getattr(task_policy, "close", None)
                    if callable(close_policy):
                        resources.callback(close_policy)
                    task_embodiment = cast(Embodiment, resolve("embodiment", embodiment))
                    resources.callback(task_embodiment.close)
                    if not task_embodiment.info.is_simulated:
                        raise ConfigError("parallel eval_set requires a simulated embodiment")
                    return eval(
                        task,
                        task_policy,
                        task_embodiment,
                        log_dir=log_dir,
                        seed=seed,
                        fail_on_error=fail_on_error,
                        remap=remap,
                        store_frames=store_frames,
                        store_actions=store_actions,
                        store_records=store_records,
                        guard=guard,
                        grader=task_grader,
                        environment_id=environment_id,
                        environment_revision=environment_revision,
                        policy_checkpoint=policy_checkpoint,
                        benchmark=benchmark,
                    )
            except (SafetyAbort, EmbodimentFault):
                raise
            except Exception as exc:
                return [
                    _error_log_for(
                        task,
                        policy,
                        embodiment,
                        seed=seed,
                        exc=exc,
                        environment_id=environment_id,
                        environment_revision=environment_revision,
                        policy_checkpoint=policy_checkpoint,
                        benchmark=benchmark,
                    )
                ]

        if not task_list:
            raise ConfigError("eval_set() requires at least one task; got an empty sequence")
        parallel_logs = [
            log for result in run_parallel(task_list, run_task, max_workers) for log in result
        ]
        return all(log.status == "success" for log in parallel_logs), parallel_logs

    before_scoring, resolved_grader = _grading_hook(grader, before_scoring)
    if not task_list:
        raise ConfigError("eval_set() requires at least one task; got an empty sequence")
    # Outside the per-task try below: a rejected preflight must stop the whole
    # set, not become the first task's error log while later tasks run.
    _preflight_grader(resolved_grader)
    logs: list[EvalLog] = []
    for task in task_list:
        try:
            logs.extend(
                eval(
                    task,
                    policy,
                    embodiment,
                    log_dir=log_dir,
                    sinks=sinks,
                    seed=seed,
                    fail_on_error=fail_on_error,
                    action_manager=action_manager,
                    guard=guard,
                    remap=remap,
                    store_frames=store_frames,
                    store_actions=store_actions,
                    store_records=store_records,
                    operator_input=operator_input,
                    # Hand the grader itself down, not just its bound method:
                    # each task builds its own EvalSpec and a bare callable
                    # would leave every eval_set log with no grader recorded.
                    grader=resolved_grader,
                    before_scoring=None if resolved_grader is not None else before_scoring,
                    environment_id=environment_id,
                    environment_revision=environment_revision,
                    policy_checkpoint=policy_checkpoint,
                    benchmark=benchmark,
                )
            )
        except (SafetyAbort, EmbodimentFault):
            raise
        except Exception as exc:
            logs.append(
                _error_log_for(
                    task,
                    policy,
                    embodiment,
                    seed=seed,
                    exc=exc,
                    environment_id=environment_id,
                    environment_revision=environment_revision,
                    policy_checkpoint=policy_checkpoint,
                    benchmark=benchmark,
                )
            )
    success = all(log.status == "success" for log in logs)
    return success, logs
