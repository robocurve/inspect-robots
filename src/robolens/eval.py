"""The ``eval()`` entry point — orchestrates scenes x epochs into an EvalLog.

Mirrors Inspect AI's ``eval()``: it runs a task's scenes (repeated over epochs),
scores each recorded trajectory, reduces epochs, aggregates metrics, and returns
a list of immutable :class:`~robolens.log.EvalLog` (one per task). The tracer
slice accepts already-constructed objects; registry-string resolution
(``policy="openvla/7b"``) is layered on with the registry milestone.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from statistics import mean
from typing import TYPE_CHECKING

from robolens import __version__
from robolens.approver import Approver, AutoApprover
from robolens.controller import Controller, DefaultController
from robolens.embodiment import Embodiment
from robolens.errors import EmbodimentFault, PolicyError, SafetyAbort
from robolens.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
from robolens.policy import Policy
from robolens.rollout import TrialRecord, derive_seed, rollout
from robolens.scorer import Score, reduce_scores, value_to_float
from robolens.task import Task

if TYPE_CHECKING:
    from robolens.logging.sink import LogSink
    from robolens.types import Action, Observation, StepResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


class _Broadcast:
    """Fan a sink lifecycle out to several sinks, preserving hook order."""

    def __init__(self, sinks: list[LogSink]):
        self._sinks = sinks

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
    task: Task,
    policy: Policy,
    embodiment: Embodiment,
    *,
    log_dir: str = "logs",
    sinks: list[LogSink] | None = None,
    seed: int | None = 0,
    fail_on_error: bool | float = False,
    controller: Controller | None = None,
    approver: Approver | None = None,
) -> list[EvalLog]:
    """Run ``task`` with ``policy`` on ``embodiment``; return ``[EvalLog]``.

    ``fail_on_error`` follows Inspect semantics for ``PolicyError`` (``True`` =
    fail on first, ``False`` = never, ``0<x<1`` = proportion, ``x>1`` = count).
    ``EmbodimentFault``/``SafetyAbort`` always halt regardless.
    """
    from robolens.logging.json_log import JsonLogSink

    sink_list: list[LogSink] = sinks if sinks is not None else [JsonLogSink(log_dir)]
    bus = _Broadcast(sink_list)
    controller = controller or DefaultController(policy.config.replan_interval)
    approver = approver or AutoApprover()

    spec = EvalSpec(
        task=task.name,
        policy=policy.info.name,
        embodiment=embodiment.info.name,
        created=_now_iso(),
        robolens_version=__version__,
        git_commit=_git_commit(),
        policy_config=asdict(policy.config),
        embodiment_info={
            "control_hz": embodiment.info.control_hz,
            "is_simulated": embodiment.info.is_simulated,
            "capabilities": sorted(embodiment.info.capabilities),
        },
        seed=seed,
    )
    bus.on_eval_start(spec)

    started = time.perf_counter()
    started_iso = _now_iso()
    epoch_spec = task.epoch_spec
    scorers = task.scorers

    scene_results: list[SceneResult] = []
    all_latencies: list[float] = []
    total_steps = 0
    total_trials = 0
    status = "success"
    error: str | None = None
    error_count = 0

    halted = False
    for scene in task.scenes:
        per_scorer_scores: dict[str, list[Score]] = {s.name: [] for s in scorers}
        epoch_dicts: list[dict[str, float]] = []
        scene_status = "success"
        scene_error: str | None = None

        for epoch in range(epoch_spec.count):
            trial_seed = derive_seed(seed, scene.init_seed, epoch)
            bus.on_trial_start(scene.id, epoch)
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
                    control_hz=task.control_hz,
                )
            except (EmbodimentFault, SafetyAbort) as exc:
                # Hardware/safety failures always halt the whole eval.
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                scene_status = "error"
                scene_error = error
                halted = True
                break
            except PolicyError as exc:
                error_count += 1
                scene_status = "error"
                scene_error = f"{type(exc).__name__}: {exc}"
                record = TrialRecord(
                    scene_id=scene.id,
                    epoch=epoch,
                    seed=trial_seed,
                    status="error",
                    error=scene_error,
                )

            total_trials += 1
            total_steps += len(record.steps)
            all_latencies.extend(record.inference_latencies)

            epoch_values: dict[str, float] = {}
            for scorer in scorers:
                score = scorer(record, scene.target)
                per_scorer_scores[scorer.name].append(score)
                epoch_values[scorer.name] = value_to_float(score.value)
            epoch_dicts.append(epoch_values)
            bus.on_trial_end(record)

        reduced = {
            name: value_to_float(reduce_scores(epoch_spec.reducer, scores).value)
            for name, scores in per_scorer_scores.items()
            if scores
        }
        scene_results.append(
            SceneResult(
                scene_id=scene.id,
                status=scene_status,
                reduced=reduced,
                epochs=epoch_dicts,
                error=scene_error,
            )
        )
        if halted or _should_fail(fail_on_error, error_count, total_trials):
            if not halted:
                status = "error"
                error = error or f"fail_on_error threshold exceeded ({error_count} errors)"
            break

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
    )
    log = EvalLog(
        version=EvalLog.SCHEMA_VERSION,
        status=status,
        eval=spec,
        results=EvalResults(
            total_scenes=len(scene_results),
            total_trials=total_trials,
            metrics=metrics,
        ),
        stats=stats,
        samples=scene_results,
        error=error,
    )
    bus.on_eval_end(log)
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
