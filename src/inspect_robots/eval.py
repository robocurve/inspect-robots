"""The ``eval()`` entry point — orchestrates scenes x epochs into an EvalLog.

Mirrors Inspect AI's ``eval()``: it runs a task's scenes (repeated over epochs),
scores each recorded trajectory, reduces epochs, aggregates metrics, and returns
a list of immutable [`EvalLog`][inspect_robots.log.EvalLog] (one per task). The tracer
slice accepts already-constructed objects; registry-string resolution
(``policy="openvla/7b"``) is layered on with the registry milestone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, cast

import numpy as np

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
from inspect_robots.frames import FrameStore, _safe
from inspect_robots.grader import Grader
from inspect_robots.log import (
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
    _json_safe_scene_metadata,
)
from inspect_robots.policy import Policy
from inspect_robots.rollout import TrialRecord, derive_seed, rollout
from inspect_robots.scene import Scene
from inspect_robots.scorer import Score, get_reducer, reduce_scores, value_to_float
from inspect_robots.task import Task
from inspect_robots.transcript import judgement_source

if TYPE_CHECKING:
    from inspect_robots.console import OperatorInput
    from inspect_robots.logging.json_log import JsonLogSink
    from inspect_robots.logging.sink import LogSink
    from inspect_robots.spaces import Box, ObservationSpace
    from inspect_robots.types import Action, Observation, StepResult


def _grading_hook(
    grader: Grader | str | None,
    before_scoring: Callable[[TrialRecord, Scene], None] | None,
) -> tuple[Callable[[TrialRecord, Scene], None] | None, Grader | None]:
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
    return datetime.now(timezone.utc).isoformat()


def _write_action_log(
    record: TrialRecord,
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

    def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
        """Offer the resolved spaces to sinks that declare a bind_spaces hook.

        Duck-typed like ``log_policy_messages``: sinks without the attribute
        are unaffected, so the sink Protocol is unchanged.
        """
        for sink in self._sinks:
            hook = getattr(sink, "bind_spaces", None)
            if callable(hook):
                hook(action_space, observation_space)

    def bind_frames_dir(self, frames_dir: str | None) -> None:
        """Offer the run's frame directory to sinks that declare the optional hook."""
        for sink in self._sinks:
            hook = getattr(sink, "bind_frames_dir", None)
            if callable(hook):
                hook(frames_dir)

    def bind_scenes(self, scenes: Sequence[Scene]) -> None:
        """Offer the run's scenes to sinks that declare the optional hook."""
        for sink in self._sinks:
            hook = getattr(sink, "bind_scenes", None)
            if callable(hook):
                hook(scenes)

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
    store_actions: bool = True,
    operator_input: OperatorInput | None = None,
    before_scoring: Callable[[TrialRecord, Scene], None] | None = None,
    grader: Grader | str | None = None,
    environment_id: str | None = None,
    environment_revision: str | None = None,
    policy_checkpoint: str | None = None,
    _policy_prebound: bool = False,
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

    ``store_actions`` defaults to ``True`` and writes each trial's complete
    executed action sequence to ``<log_dir>/actions`` as an atomic JSONL
    side-car. The ``actions`` trial-metadata key is framework-reserved. Set
    ``store_actions`` to ``False`` to disable these files.

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
    mutate the record — e.g. capture ``TrialRecord.operator_judgement`` (R6)
    so the ``operator`` scorer can read it, and ``TrialRecord.operator_note``
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
    [`ConfigError`][inspect_robots.errors.ConfigError] for an invalid epoch reducer.
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
            store_actions=store_actions,
            operator_input=operator_input,
            before_scoring=before_scoring,
            grader_identity=_grader_identity(resolved_grader),
            environment_id=environment_id,
            environment_revision=environment_revision,
            policy_checkpoint=policy_checkpoint,
            policy_prebound=_policy_prebound,
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
    store_actions: bool,
    operator_input: OperatorInput | None,
    before_scoring: Callable[[TrialRecord, Scene], None] | None,
    grader_identity: tuple[str | None, dict[str, Any]],
    environment_id: str | None = None,
    environment_revision: str | None = None,
    policy_checkpoint: str | None = None,
    policy_prebound: bool = False,
) -> list[EvalLog]:
    """The body of [`eval`][inspect_robots.eval.eval], after resolution/ownership."""
    from inspect_robots.logging.json_log import JsonLogSink
    from inspect_robots.session import _DEFINITIVE_REASONS
    from inspect_robots.types import Observation

    # Embodiment-adaptive policies (plan 0008 §3c): an optional bind() hook
    # runs before the compatibility check so the policy can adopt the
    # embodiment's spaces. Duck-typed — bind is not part of the Policy
    # Protocol, so existing policies are untouched.
    if not policy_prebound:
        bind = getattr(policy, "bind", None)
        if callable(bind):
            bind(embodiment.info)

    # Fail fast on incompatible pairings before touching any hardware/sim.
    # This also validates the embodiment rate needed by a seconds-based task
    # before the resolved horizon is exposed to an adapter (plan 0026).
    assert_compatible(policy, embodiment, task, remap=remap)
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

    # One subdirectory per run: trial ids repeat across runs (scene-epoch),
    # so a shared directory would silently overwrite the previous run's
    # frames or transcripts.
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"

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
        policy_config=asdict(policy.config),
        embodiment_info={
            "control_hz": embodiment.info.control_hz,
            "is_simulated": embodiment.info.is_simulated,
            "capabilities": sorted(embodiment.info.capabilities),
        },
        grader=grader_name,
        grader_config=grader_config,
        seed=seed,
        max_steps=task_envelope.max_steps,
        max_seconds=task.max_seconds,
        environment_id=environment_id or getattr(embodiment.info, "environment_id", None),
        environment_revision=environment_revision
        or getattr(embodiment.info, "environment_revision", None),
        policy_checkpoint=policy_checkpoint or getattr(policy.info, "checkpoint", None),
    )
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

    halted = False
    stopped = False
    cancelled_exc: _CancelledTrial | None = None
    # A proportion threshold is a share of the whole eval, so the denominator is
    # every trial the run intends to attempt. Using the completed-so-far count
    # made the first error 1/1 = 100%, which trips any threshold below 1.
    planned_trials = len(task.scenes) * epoch_spec.count
    for scene in task.scenes:
        per_scorer_scores: dict[str, list[Score]] = {s.name: [] for s in scorers}
        epoch_dicts: list[dict[str, float]] = []
        judgements: list[str | None] = []
        judgement_sources: list[str | None] = []
        notes: list[str | None] = []
        trial_metadatas: list[dict[str, Any]] = []
        termination_reasons: list[str | None] = []
        operator_messages: list[tuple[dict[str, Any], ...]] = []
        policy_transcripts: list[Any] = []
        scene_metadata = _json_safe_scene_metadata(scene.metadata)
        scene_status = "success"
        scene_error: str | None = None
        scene_errored_trials = 0
        scene_errors_retryable: list[bool] = []

        for epoch in range(epoch_spec.count):
            trial_seed = derive_seed(seed, scene.init_seed, epoch)
            bus.on_trial_start(scene.id, epoch)
            record: TrialRecord | None = None
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
                    scene_errors_retryable.append(False)
                    if isinstance(exc, (SafetyAbort, EmbodimentFault)):
                        halted = True
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                    record = TrialRecord(
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
                        seed=trial_seed,
                        epoch=epoch,
                        controller=controller,
                        approver=approver,
                        sink=bus,
                        frame_store=frame_store,
                        operator_input=operator_input,
                    )
                except _CancelledTrial as exc:
                    status = "cancelled"
                    error = str(exc)
                    scene_status = "cancelled"
                    scene_error = error
                    scene_errors_retryable.append(False)
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
                    scene_errors_retryable.append(False)
                    halted = True
                    record = exc.record
                except PolicyError as exc:
                    error_count += 1
                    scene_status = "error"
                    scene_error = f"{type(exc).__name__}: {exc}"
                    scene_errors_retryable.append(exc.retryable)
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
                        scene_errored_trials += 1
                    judgements.append(None)
                    judgement_sources.append(None)
                    notes.append(None)
                else:
                    if before_scoring is not None:
                        # The only trials the hook sees are the ones scorers
                        # will read — an operator verdict on a crashed trial
                        # would be dead data (errored trials are never scored).
                        if record.operator_judgement is None and not (
                            record.terminated and record.termination_reason in _DEFINITIVE_REASONS
                        ):
                            observe_parked = getattr(embodiment, "observe_parked", None)
                            if callable(observe_parked):
                                try:
                                    parked_observation = observe_parked()
                                except (SafetyAbort, EmbodimentFault):
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
                        before_scoring(record, scene)
                    epoch_values: dict[str, float] = {}
                    for scorer in scorers:
                        try:
                            score = scorer(record, scene.target)
                            value = value_to_float(score.value)
                        except (SafetyAbort, EmbodimentFault):
                            # Halt signals are not scoring errors: containing
                            # them here would let the next rollout start after
                            # an explicit safety abort or a hardware fault.
                            raise
                        except Exception as exc:
                            # A scorer failure degrades to an error log - it must
                            # never crash the eval and lose the trials that ran.
                            detail = f"scorer {scorer.name!r} failed: {exc}"
                            scene_status = "error"
                            scene_errors_retryable.append(False)
                            scene_error = (
                                detail if scene_error is None else f"{scene_error}; {detail}"
                            )
                            if status == "success":
                                status = "error"
                                error = detail
                            continue
                        per_scorer_scores[scorer.name].append(score)
                        epoch_values[scorer.name] = value
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
                            scene_errors_retryable.append(False)
                            scene_error = (
                                detail if scene_error is None else f"{scene_error}; {detail}"
                            )
                            if isinstance(exc, (SafetyAbort, EmbodimentFault)):
                                halted = True
                                status = "error"
                                error = f"{type(exc).__name__}: {exc}"
                            elif status == "success":
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

                trial_metadatas.append(record.metadata)
                termination_reasons.append(record.termination_reason)
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
                bus.on_trial_end(record)

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
                detail = f"reducer {epoch_spec.reducer!r} failed for scorer {name!r}: {exc}"
                scene_status = "error"
                scene_errors_retryable.append(False)
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
                operator_messages=tuple(operator_messages),
                policy_transcripts=tuple(policy_transcripts),
                frames_dir=str(frame_store.root) if frame_store is not None else None,
                errored_trials=scene_errored_trials,
                retryable_error=bool(scene_errors_retryable) and all(scene_errors_retryable),
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
        halted=halted,
    )
    bus.on_eval_end(log)
    if cancelled_exc is not None:
        raise cancelled_exc
    return [log]


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


def _error_log_for(
    task: Task | str,
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    seed: int | None,
    exc: Exception,
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
    controller: Controller | None = None,
    approver: Approver | None = None,
    remap: dict[str, str] | None = None,
    store_frames: bool = False,
    store_actions: bool = True,
    operator_input: OperatorInput | None = None,
    before_scoring: Callable[[TrialRecord, Scene], None] | None = None,
    grader: Grader | str | None = None,
    retry_attempts: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_inputs: Mapping[str, object] | None = None,
) -> tuple[bool, list[EvalLog]]:
    """Run a set of tasks and return ``(success, logs)`` (mirrors Inspect AI).

    ``success`` is ``True`` iff every returned log has ``status == "success"``.
    A task that raises before or without producing a log contributes one
    ``status="error"`` log carrying the exception text, and the remaining
    tasks still run. A ``SafetyAbort`` or ``EmbodimentFault`` that escapes
    ``eval()`` (raised outside a trial) and ``KeyboardInterrupt`` still
    propagate. A halt inside a trial ends that task with an error log and the
    set continues to the next task.
    ``CompatibilityError``, unknown policy or embodiment registry names, and
    task-factory ``ConfigError`` are therefore reported once per affected
    task. Only grading configuration errors raised before the task loop
    propagate.

    With a string embodiment, a non-safety exception from ``close()`` can
    produce an error row even when that task's completed JSON log is already
    on disk.

    ``store_actions`` follows ``eval()``'s default-on action side-car contract.

    ``grader``/``before_scoring`` follow ``eval()``'s contract (one pre-scoring
    hook, not both) and are resolved once here, so every task shares the same
    grader instance.

    Caller-supplied ``sinks`` are reused across the set's sequential runs. Each
    sink must reset its per-run state in ``on_eval_start`` and tolerate one
    complete lifecycle per task.

    ``checkpoint_path`` saves immutable attempt logs and a single-writer manifest.
    A later call with the same inputs reuses fully completed scenes. The optional
    ``retry_attempts`` budget permits only explicitly retryable policy failures
    to restart their scene during this invocation. Each retry begins at epoch zero.
    ``checkpoint_inputs`` adds caller-specific, JSON-serializable settings to the
    identity comparison. Its values are stored only as a SHA-256 digest.
    Built-in controllers and approvers contribute their effective settings,
    recursively for smoothing and approver chains. Custom implementations,
    including subclasses, must declare a JSON ``checkpoint_identity()`` hook.
    A custom ``before_scoring`` callback in checkpoint mode must declare a JSON
    ``checkpoint_identity()`` hook on the callable or its bound-method owner.
    Include its behavior revision and hidden grading settings in that identity.
    An attempt without a saved log leaves the checkpoint in flight; a later
    call fails before robot reset so the operator can reconcile the run.
    That invocation stops immediately and returns only the task logs collected
    so far, so a later task cannot clear the unresolved attempt's marker.
    """
    before_scoring, resolved_grader = _grading_hook(grader, before_scoring)
    task_list = [tasks] if isinstance(tasks, Task | str) else list(tasks)
    if not task_list:
        raise ConfigError("eval_set() requires at least one task; got an empty sequence")
    if (
        not isinstance(retry_attempts, int)
        or isinstance(retry_attempts, bool)
        or retry_attempts < 0
    ):
        raise ConfigError(f"retry_attempts must be an integer >= 0, got {retry_attempts!r}")
    if checkpoint_path is not None and not checkpoint_path:
        raise ConfigError("checkpoint_path must be a nonempty path")
    if retry_attempts or checkpoint_path is not None:
        return _resumable_eval_set(
            task_list,
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
            store_actions=store_actions,
            operator_input=operator_input,
            before_scoring=before_scoring,
            grader=resolved_grader,
            retry_attempts=retry_attempts,
            checkpoint_path=checkpoint_path,
            checkpoint_inputs=checkpoint_inputs,
        )
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
                    controller=controller,
                    approver=approver,
                    remap=remap,
                    store_frames=store_frames,
                    store_actions=store_actions,
                    operator_input=operator_input,
                    # Hand the grader itself down, not just its bound method:
                    # each task builds its own EvalSpec and a bare callable
                    # would leave every eval_set log with no grader recorded.
                    grader=resolved_grader,
                    before_scoring=None if resolved_grader is not None else before_scoring,
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
                )
            )
    success = all(log.status == "success" for log in logs)
    return success, logs


def _clear_json_sink_path(sink: JsonLogSink) -> None:
    """Forget a prior attempt path before a reused sink starts another eval."""
    sink.path = None


def _checkpoint_inputs_digest(
    inputs: Mapping[str, object] | None, *, name: str = "checkpoint_inputs"
) -> str | None:
    """Fingerprint external run settings without storing their raw values."""
    if inputs is None:
        return None
    try:
        encoded = json.dumps(dict(inputs), sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"{name} must be JSON serializable: {exc}") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _instance_identity(value: object) -> str | None:
    """Name an optional callback or component for checkpoint comparison."""
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _callback_identity(value: object) -> dict[str, object] | None:
    """Require custom grading callbacks to declare hidden settings for reuse."""
    if value is None:
        return None
    hook = getattr(value, "checkpoint_identity", None)
    if not callable(hook):
        owner = getattr(value, "__self__", None)
        hook = getattr(owner, "checkpoint_identity", None)
    if not callable(hook):
        raise ConfigError(
            "before_scoring needs a JSON checkpoint_identity() hook in checkpoint mode"
        )
    return {
        "type": _instance_identity(value),
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "config_sha256": _checkpoint_inputs_digest(
            {"identity": hook()}, name="before_scoring checkpoint_identity()"
        ),
    }


def _resumable_eval_set(
    task_list: list[Task | str],
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    log_dir: str,
    sinks: list[LogSink] | None,
    seed: int | None,
    fail_on_error: bool | float,
    controller: Controller | None,
    approver: Approver | None,
    remap: dict[str, str] | None,
    store_frames: bool,
    store_actions: bool,
    operator_input: OperatorInput | None,
    before_scoring: Callable[[TrialRecord, Scene], None] | None,
    grader: Grader | None,
    retry_attempts: int,
    checkpoint_path: str | None,
    checkpoint_inputs: Mapping[str, object] | None,
) -> tuple[bool, list[EvalLog]]:
    """Run unfinished scene subsets and publish aggregates from durable attempts."""
    from inspect_robots._eval_set_checkpoint import (
        _checkpoint_seed,
        _component_identity,
        _identity,
        _open_checkpoint,
    )
    from inspect_robots._eval_set_merge import _complete, _merge_task, _pending_scenes
    from inspect_robots.log import read_eval_log
    from inspect_robots.logging.json_log import JsonLogSink
    from inspect_robots.registry import resolve

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and (isinstance(policy, str) or isinstance(embodiment, str)):
        # Resolve registry components even when every scene is already complete.
        # Otherwise a changed factory could reuse scores without ever being opened.
        try:
            resolved_policy = (
                cast(Policy, resolve("policy", policy)) if isinstance(policy, str) else policy
            )
            owns_embodiment = isinstance(embodiment, str)
            resolved_embodiment = (
                cast(Embodiment, resolve("embodiment", embodiment))
                if isinstance(embodiment, str)
                else embodiment
            )
        except (SafetyAbort, EmbodimentFault):
            raise
        except Exception as exc:
            if checkpoint.exists():
                raise ConfigError(f"checkpoint component could not resolve: {exc}") from exc
            return False, [
                _error_log_for(task, policy, embodiment, seed=seed, exc=exc) for task in task_list
            ]
        try:
            return _resumable_eval_set(
                task_list,
                resolved_policy,
                resolved_embodiment,
                log_dir=log_dir,
                sinks=sinks,
                seed=seed,
                fail_on_error=fail_on_error,
                controller=controller,
                approver=approver,
                remap=remap,
                store_frames=store_frames,
                store_actions=store_actions,
                operator_input=operator_input,
                before_scoring=before_scoring,
                grader=grader,
                retry_attempts=retry_attempts,
                checkpoint_path=checkpoint_path,
                checkpoint_inputs=checkpoint_inputs,
            )
        finally:
            if owns_embodiment:
                resolved_embodiment.close()
    if checkpoint is not None:
        # Adaptive policies publish their actual spaces only after binding to
        # the robot. Each attempt skips rebinding this already identified policy.
        bind = getattr(policy, "bind", None)
        if callable(bind):
            bind(cast(Embodiment, embodiment).info)
    if seed is None:
        seed = _checkpoint_seed(checkpoint) if checkpoint is not None else None
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")

    resolved_tasks: list[Task | None] = []
    resolution_errors: dict[int, Exception] = {}
    for index, task in enumerate(task_list):
        if isinstance(task, str):
            try:
                resolved_tasks.append(cast(Task, resolve("task", task)))
            except Exception as exc:
                if checkpoint is not None:
                    raise ConfigError(f"checkpoint task {task!r} could not resolve: {exc}") from exc
                resolved_tasks.append(None)
                resolution_errors[index] = exc
        else:
            resolved_tasks.append(task)

    options: dict[str, object] = {
        "fail_on_error": fail_on_error,
        "remap": remap,
        "store_frames": store_frames,
        "store_actions": store_actions,
        "grader": _grader_identity(grader),
        "controller": (
            _component_identity(
                controller or DefaultController(cast(Policy, policy).config.replan_interval),
                name="controller",
            )
            if checkpoint is not None
            else None
        ),
        "approver": (
            _component_identity(approver or AutoApprover(), name="approver")
            if checkpoint is not None
            else None
        ),
        "operator_input": _instance_identity(operator_input),
        "before_scoring": (
            _callback_identity(before_scoring)
            if checkpoint is not None and grader is None
            else None
        ),
        "checkpoint_inputs_sha256": (
            _checkpoint_inputs_digest(checkpoint_inputs) if checkpoint is not None else None
        ),
    }
    identity = (
        _identity(
            [cast(Task, task) for task in resolved_tasks],
            policy,
            embodiment,
            seed=seed,
            log_dir=log_dir,
            options=options,
        )
        if checkpoint is not None
        else None
    )
    context = (
        _open_checkpoint(checkpoint, identity)
        if checkpoint is not None and identity is not None
        else nullcontext(None)
    )
    logs: list[EvalLog] = []
    with context as manifest:
        for index, raw_task in enumerate(task_list):
            resolved_task = resolved_tasks[index]
            if resolved_task is None:
                logs.append(
                    _error_log_for(
                        raw_task, policy, embodiment, seed=seed, exc=resolution_errors[index]
                    )
                )
                continue

            attempts: list[tuple[EvalLog, str]] = []
            if manifest is not None:
                for entry in manifest.attempts:
                    if entry["task_index"] == index:
                        path = manifest.attempt_log_path(entry)
                        attempts.append((read_eval_log(str(path)), str(path)))
            merged = _merge_task(resolved_task, attempts) if attempts else None
            to_run = (
                _pending_scenes(resolved_task, merged)
                if merged is not None
                else list(resolved_task.scenes)
            )
            first_run = not attempts
            attempt_counts: dict[str, int] = {scene.id: 0 for scene in resolved_task.scenes}
            while first_run or to_run:
                first_run = False
                scene_ids = [scene.id for scene in to_run]
                for scene_id in scene_ids:
                    attempt_counts[scene_id] += 1
                attempt_task = replace(resolved_task, scenes=tuple(to_run))
                if (
                    sinks
                    and isinstance(sinks[0], JsonLogSink)
                    and sinks[0].log_dir.resolve() == Path(log_dir).resolve()
                ):
                    json_sink = sinks[0]
                    attempt_sinks = sinks
                else:
                    json_sink = JsonLogSink(log_dir)
                    attempt_sinks = [json_sink, *(sinks or [])]
                _clear_json_sink_path(json_sink)
                if manifest is not None:
                    manifest.start_attempt()
                try:
                    (attempt_log,) = eval(
                        attempt_task,
                        policy,
                        embodiment,
                        log_dir=log_dir,
                        sinks=attempt_sinks,
                        seed=seed,
                        fail_on_error=fail_on_error,
                        controller=controller,
                        approver=approver,
                        remap=remap,
                        store_frames=store_frames,
                        store_actions=store_actions,
                        operator_input=operator_input,
                        grader=grader,
                        before_scoring=None if grader is not None else before_scoring,
                        _policy_prebound=manifest is not None,
                    )
                except KeyboardInterrupt:
                    if manifest is not None and json_sink.path is not None:
                        manifest.add_attempt(index, scene_ids, json_sink.path)
                    raise
                except (SafetyAbort, EmbodimentFault):
                    raise
                except Exception as exc:
                    if manifest is not None and json_sink.path is not None:
                        manifest.add_attempt(index, scene_ids, json_sink.path)
                    logs.append(
                        _error_log_for(resolved_task, policy, embodiment, seed=seed, exc=exc)
                    )
                    if manifest is not None and json_sink.path is None:
                        return False, logs
                    break
                if json_sink.path is None:
                    raise ConfigError("eval() returned without a durable attempt log")
                path = json_sink.path.resolve()
                if manifest is not None:
                    manifest.add_attempt(index, scene_ids, path)
                attempts.append((attempt_log, str(path)))
                merged = _merge_task(resolved_task, attempts)
                recorded = {sample.scene_id: sample for sample in attempt_log.samples}
                if attempt_log.halted:
                    to_run = []
                else:
                    to_run = [
                        scene
                        for scene in to_run
                        if scene.id in recorded
                        and recorded[scene.id].retryable_error
                        and not _complete(recorded[scene.id], resolved_task.epoch_spec.count)
                        and attempt_counts[scene.id] <= retry_attempts
                    ]
            else:
                aggregate = cast(EvalLog, merged)
                aggregate_sink = JsonLogSink(log_dir)
                aggregate_sink.on_eval_end(aggregate)
                if manifest is not None and aggregate_sink.path is not None:
                    manifest.set_aggregate(index, aggregate_sink.path)
                logs.append(aggregate)
    return all(log.status == "success" for log in logs), logs
