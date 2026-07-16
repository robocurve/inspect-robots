"""The ``eval()`` entry point — orchestrates scenes x epochs into an EvalLog.

Mirrors Inspect AI's ``eval()``: it runs a task's scenes (repeated over epochs),
scores each recorded trajectory, reduces epochs, aggregates metrics, and returns
a list of immutable [`EvalLog`][inspect_robots.log.EvalLog] (one per task). The tracer
slice accepts already-constructed objects; registry-string resolution
(``policy="openvla/7b"``) is layered on with the registry milestone.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, cast

from inspect_robots import __version__
from inspect_robots.approver import Approver, AutoApprover
from inspect_robots.compat import assert_compatible
from inspect_robots.controller import Controller, DefaultController
from inspect_robots.embodiment import Embodiment
from inspect_robots.errors import (
    ConfigError,
    EmbodimentFault,
    PolicyError,
    SafetyAbort,
    _CancelledTrial,
)
from inspect_robots.frames import FrameStore
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
from inspect_robots.policy import Policy
from inspect_robots.rollout import TrialRecord, derive_seed, rollout
from inspect_robots.scene import Scene
from inspect_robots.scorer import Score, get_reducer, reduce_scores, value_to_float
from inspect_robots.task import Task

if TYPE_CHECKING:
    from inspect_robots.logging.sink import LogSink
    from inspect_robots.types import Action, Observation, StepResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if tree is not None and tree.returncode == 0 and tree.stdout.strip():
        commit += "-dirty"
    return commit


class _Broadcast:
    """Fan a sink lifecycle out to several sinks, preserving hook order."""

    def __init__(self, sinks: list[LogSink]):
        self._sinks = sinks
        policy_message_hooks: list[Callable[[int, Sequence[Any]], None]] = []
        for sink in sinks:
            hook = getattr(sink, "log_policy_messages", None)
            if callable(hook):
                policy_message_hooks.append(hook)
        self._policy_message_hooks = policy_message_hooks
        if policy_message_hooks:
            self.log_policy_messages = self._fan_policy_messages

    def _fan_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        for hook in self._policy_message_hooks:
            hook(t, messages)

    def on_eval_start(self, spec: EvalSpec) -> None:
        for s in self._sinks:
            s.on_eval_start(spec)

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        for s in self._sinks:
            s.on_trial_start(scene_id, epoch)

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        for s in self._sinks:
            s.log_step(t, observation, action, result)

    def on_trial_end(self, record: TrialRecord) -> None:
        for s in self._sinks:
            s.on_trial_end(record)

    def on_eval_end(self, log: EvalLog) -> None:
        for s in self._sinks:
            s.on_eval_end(log)


def eval(
    task: Task | str,
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    log_dir: str = "logs",
    sinks: list[LogSink] | None = None,
    seed: int | None = 0,
    fail_on_error: bool | float = False,
    controller: Controller | None = None,
    approver: Approver | None = None,
    remap: dict[str, str] | None = None,
    store_frames: bool = False,
    before_scoring: Callable[[TrialRecord, Scene], None] | None = None,
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
    with ``status == "error"``, regardless of ``fail_on_error``.

    Ctrl-C during a rollout records the partial trial and writes a log with
    ``status == "cancelled"``, then re-raises the interrupt (as a
    ``KeyboardInterrupt`` subclass chaining the original) after
    ``on_eval_end`` completes. An interrupt outside the rollout
    call (during scoring, reducers, or log assembly), or a second interrupt
    during the cancellation handlers, may still prevent the log from being
    written.

    When ``store_frames`` is set, camera frames are streamed to
    ``<log_dir>/frames`` as binary side-cars (R5) rather than kept in memory.

    ``before_scoring`` is called exactly once per trial that will be scored
    (never for errored trials, which are recorded but not scored), after the
    rollout returns and before the scorers run. It may mutate the record —
    e.g. capture ``TrialRecord.operator_judgement`` (R6) so the ``operator``
    scorer can read it. Exceptions it raises propagate to the caller. Note
    this fires on the *other* side of scoring from ``LogSink.on_trial_end``.

    Raises [`CompatibilityError`][inspect_robots.errors.CompatibilityError] (fail fast, before any
    rollout) if the policy and embodiment are incompatible, and
    [`ConfigError`][inspect_robots.errors.ConfigError] for an invalid epoch reducer.
    """
    from inspect_robots.registry import resolve

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
            controller=controller,
            approver=approver,
            remap=remap,
            store_frames=store_frames,
            before_scoring=before_scoring,
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
    controller: Controller | None,
    approver: Approver | None,
    remap: dict[str, str] | None,
    store_frames: bool,
    before_scoring: Callable[[TrialRecord, Scene], None] | None,
) -> list[EvalLog]:
    """The body of [`eval`][inspect_robots.eval.eval], after resolution/ownership."""
    from inspect_robots.logging.json_log import JsonLogSink

    # Embodiment-adaptive policies (plan 0008 §3c): an optional bind() hook
    # runs before the compatibility check so the policy can adopt the
    # embodiment's spaces. Duck-typed — bind is not part of the Policy
    # Protocol, so existing policies are untouched.
    bind = getattr(policy, "bind", None)
    if callable(bind):
        bind(embodiment.info)

    # Horizon-aware embodiments (plan 0013): an optional bind_task() hook runs
    # here too, so the adapter can learn the rollout envelope (e.g. for an
    # operator countdown) before any hardware is touched. Duck-typed —
    # bind_task is not part of the Embodiment Protocol.
    bind_task = getattr(embodiment, "bind_task", None)
    if callable(bind_task):
        bind_task(task.envelope)

    # Fail fast on incompatible pairings before touching any hardware/sim.
    assert_compatible(policy, embodiment, task, remap=remap)

    epoch_spec = task.epoch_spec
    scorers = task.scorers
    # Fail fast on an unknown/invalid epoch reducer, before any rollout runs.
    try:
        get_reducer(epoch_spec.reducer)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if seed is None:
        # Draw and record a real seed so the run stays reproducible after the
        # fact; None must not silently alias seed=0 (see derive_seed).
        seed = int.from_bytes(os.urandom(4), "little")

    sink_list: list[LogSink] = sinks if sinks is not None else [JsonLogSink(log_dir)]
    bus = _Broadcast(sink_list)
    controller = controller or DefaultController(policy.config.replan_interval)
    approver = approver or AutoApprover()

    frame_store: FrameStore | None = None
    if store_frames:
        # One subdirectory per run: trial ids repeat across runs (scene-epoch),
        # so a shared directory would silently overwrite the previous run's
        # frames. The log's stats.frames_dir records the exact directory.
        run_stamp = (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        )
        frame_store = FrameStore(str(Path(log_dir) / "frames" / run_stamp))

    spec = EvalSpec(
        task=task.name,
        policy=policy.info.name,
        embodiment=embodiment.info.name,
        created=_now_iso(),
        inspect_robots_version=__version__,
        git_commit=_git_commit(),
        policy_config=asdict(policy.config),
        embodiment_info={
            "control_hz": embodiment.info.control_hz,
            "is_simulated": embodiment.info.is_simulated,
            "capabilities": sorted(embodiment.info.capabilities),
        },
        seed=seed,
        max_steps=task.max_steps,
    )
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

    halted = False
    stopped = False
    cancelled_exc: _CancelledTrial | None = None
    for scene in task.scenes:
        per_scorer_scores: dict[str, list[Score]] = {s.name: [] for s in scorers}
        epoch_dicts: list[dict[str, float]] = []
        judgements: list[str | None] = []
        termination_reasons: list[str | None] = []
        policy_transcripts: list[Any] = []
        scene_status = "success"
        scene_error: str | None = None

        for epoch in range(epoch_spec.count):
            trial_seed = derive_seed(seed, scene.init_seed, epoch)
            bus.on_trial_start(scene.id, epoch)
            record: TrialRecord | None
            try:
                record = rollout(
                    policy,
                    embodiment,
                    scene,
                    max_steps=task.max_steps,
                    seed=trial_seed,
                    epoch=epoch,
                    controller=controller,
                    approver=approver,
                    sink=bus,
                    frame_store=frame_store,
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
                record = exc.record or TrialRecord(
                    scene_id=scene.id,
                    epoch=epoch,
                    seed=trial_seed,
                    status="error",
                    error=scene_error,
                )

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
                    termination_reasons.append(record.termination_reason)
                    policy_transcripts.append(record.policy_transcript)
                else:
                    if before_scoring is not None:
                        # The only trials the hook sees are the ones scorers
                        # will read — an operator verdict on a crashed trial
                        # would be dead data (errored trials are never scored).
                        before_scoring(record, scene)
                    epoch_values: dict[str, float] = {}
                    for scorer in scorers:
                        score = scorer(record, scene.target)
                        per_scorer_scores[scorer.name].append(score)
                        epoch_values[scorer.name] = value_to_float(score.value)
                    epoch_dicts.append(epoch_values)
                    judgements.append(record.operator_judgement)
                    termination_reasons.append(record.termination_reason)
                    policy_transcripts.append(record.policy_transcript)
                bus.on_trial_end(record)

            if halted:
                stopped = True
                break
            if _should_fail(fail_on_error, error_count, total_trials):
                # Checked after every trial, so fail_on_error=True stops at the
                # first PolicyError instead of finishing the scene's epochs.
                status = "error"
                error = f"fail_on_error threshold exceeded ({error_count} errors)"
                stopped = True
                break

        reduced: dict[str, float] = {}
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
                note = f"reducer {epoch_spec.reducer!r} failed for scorer {name!r}: {exc}"
                scene_status = "error"
                scene_error = note if scene_error is None else f"{scene_error}; {note}"
                if status == "success":
                    status = "error"
                    error = note

        scene_results.append(
            SceneResult(
                scene_id=scene.id,
                status=scene_status,
                reduced=reduced,
                epochs=tuple(epoch_dicts),
                error=scene_error,
                instruction=scene.instruction,
                operator_judgements=tuple(judgements),
                termination_reasons=tuple(termination_reasons),
                policy_transcripts=tuple(policy_transcripts),
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

    metrics: dict[str, float] = {}
    for scorer in scorers:
        vals = [sr.reduced[scorer.name] for sr in scene_results if scorer.name in sr.reduced]
        if vals:
            metrics[scorer.name] = mean(vals)

    stats = EvalStats(
        started_at=started_iso,
        completed_at=_now_iso(),
        duration_s=time.perf_counter() - started,
        total_steps=total_steps,
        mean_inference_latency_s=(mean(all_latencies) if all_latencies else None),
        frames_dir=str(frame_store.root) if frame_store is not None else None,
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
        ),
        stats=stats,
        samples=tuple(scene_results),
        error=error,
    )
    bus.on_eval_end(log)
    if cancelled_exc is not None:
        raise cancelled_exc
    return [log]


def _should_fail(fail_on_error: bool | float, errors: int, trials: int) -> bool:
    """Inspect-style ``fail_on_error`` evaluation for PolicyError-class failures."""
    if not fail_on_error or errors == 0:  # covers False, 0, 0.0
        return False
    if fail_on_error is True:
        return True
    if 0 < fail_on_error < 1:
        return trials > 0 and (errors / trials) >= fail_on_error
    return errors >= fail_on_error


def eval_set(
    tasks: Task | str | Sequence[Task | str],
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    log_dir: str = "logs",
    seed: int | None = 0,
    fail_on_error: bool | float = False,
    controller: Controller | None = None,
    approver: Approver | None = None,
    remap: dict[str, str] | None = None,
    store_frames: bool = False,
    before_scoring: Callable[[TrialRecord, Scene], None] | None = None,
    retry_attempts: int = 0,
) -> tuple[bool, list[EvalLog]]:
    """Run a set of tasks and return ``(success, logs)`` (mirrors Inspect AI).

    ``success`` is ``True`` iff every task's log has ``status == "success"``.

    Resumption of a partially-completed run (skipping already-finished scenes via
    a stable run id) is reserved for a follow-up: ``retry_attempts`` is accepted
    now so callers don't get retrofitted, but is not yet honored.
    """
    task_list = [tasks] if isinstance(tasks, Task | str) else list(tasks)
    logs: list[EvalLog] = []
    for task in task_list:
        logs.extend(
            eval(
                task,
                policy,
                embodiment,
                log_dir=log_dir,
                seed=seed,
                fail_on_error=fail_on_error,
                controller=controller,
                approver=approver,
                remap=remap,
                store_frames=store_frames,
                before_scoring=before_scoring,
            )
        )
    success = all(log.status == "success" for log in logs)
    return success, logs
