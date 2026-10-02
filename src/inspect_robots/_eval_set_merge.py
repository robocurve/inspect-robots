"""Select scene results from immutable attempts and recompute task aggregates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite
from statistics import mean

from inspect_robots.errors import ConfigError
from inspect_robots.log import EvalLog, EvalResults, EvalStats, SceneResult
from inspect_robots.scene import Scene
from inspect_robots.task import Task


def _complete(sample: SceneResult, epochs: int) -> bool:
    """Treat a fully recorded successful scene as reusable regardless of score."""
    return sample.status == "success" and len(sample.epochs) == epochs


def _pending_scenes(task: Task, merged: EvalLog) -> list[Scene]:
    """Return the original task scenes lacking a full successful result."""
    selected = {sample.scene_id: sample for sample in merged.samples}
    return [
        scene
        for scene in task.scenes
        if scene.id not in selected or not _complete(selected[scene.id], task.epoch_spec.count)
    ]


def _merge_task(task: Task, attempts: Sequence[tuple[EvalLog, str]]) -> EvalLog:
    """Build one task log from attempts without rewriting their audit records."""
    if not attempts:
        raise ConfigError("cannot merge an evaluation task without attempt logs")
    selected: dict[str, SceneResult] = {}
    expected_ids = {scene.id for scene in task.scenes}
    epochs = task.epoch_spec.count
    for attempt, _ in attempts:
        for sample in attempt.samples:
            if sample.scene_id not in expected_ids:
                raise ConfigError(f"attempt contains unknown scene {sample.scene_id!r}")
            prior = selected.get(sample.scene_id)
            if prior is None or not _complete(prior, epochs):
                selected[sample.scene_id] = replace(
                    sample,
                    frames_dir=sample.frames_dir or attempt.stats.frames_dir,
                )
    samples = tuple(selected[scene.id] for scene in task.scenes if scene.id in selected)
    metrics: dict[str, float] = {}
    for scorer in task.scorers:
        values = [
            sample.reduced[scorer.name] for sample in samples if scorer.name in sample.reduced
        ]
        if values and all(isinstance(value, int | float) and isfinite(value) for value in values):
            metrics[scorer.name] = mean(values)
    incomplete = [
        scene
        for scene in task.scenes
        if scene.id not in selected or not _complete(selected[scene.id], epochs)
    ]
    status = "success" if not incomplete else "error"
    error = None
    if incomplete:
        error = f"{len(incomplete)} scene(s) incomplete"
        for scene in incomplete:
            candidate = selected.get(scene.id)
            if candidate is not None and candidate.error:
                error = candidate.error
                break
    first = attempts[0][0]
    last = attempts[-1][0]
    return EvalLog(
        version=first.version,
        status=status,
        eval=first.eval,
        results=EvalResults(
            total_scenes=len(samples),
            total_trials=sum(len(sample.epochs) for sample in samples),
            metrics=metrics,
            errored_trials=sum(sample.errored_trials for sample in samples),
        ),
        stats=EvalStats(
            started_at=first.stats.started_at,
            completed_at=last.stats.completed_at,
            duration_s=sum(log.stats.duration_s for log, _ in attempts),
            total_steps=sum(log.stats.total_steps for log, _ in attempts),
            mean_inference_latency_s=None,
            frames_dir=None,
        ),
        samples=samples,
        error=error,
        source_logs=tuple(path for _, path in attempts),
        halted=last.halted,
    )
