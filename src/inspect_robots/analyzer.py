"""Analyzers: read-only analysis of saved logs and their annotations.

The paper's analyzers have read-only access to records, as composable,
shareable modules. An [`Analyzer`][inspect_robots.analyzer.Analyzer] receives
[`AnnotatedLog`][inspect_robots.analyzer.AnnotatedLog] views, never writes, and
returns an [`Analysis`][inspect_robots.analyzer.Analysis]: text for a person,
JSON-safe ``data`` for a program, and optional ``files`` to save. Analyzers are
a registry kind (``@analyzer`` from ``inspect_robots.registry``, entry-point
group ``inspect_robots.analyzers``),
run by ``inspect-robots analyze NAME LOG...``.

[`rescore`][inspect_robots.analyzer.rescore] recomputes a log's results from
its annotations: scorers that offer ``score_annotations`` (the ``operator``
scorer does) score each trial from the judgement that currently counts, and
every other stored score is kept. Only values whose inputs changed are
recomputed, so a log with no edits re-scores to exactly its stored results.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, runtime_checkable

from inspect_robots.annotation import Annotation, lift_annotations, read_annotations, trial_key
from inspect_robots.log import EvalLog, SceneResult, read_eval_log
from inspect_robots.scorer import Score, reduce_scores, value_to_float


@dataclass(frozen=True)
class AnnotatedLog:
    """A saved log, where it came from, and every annotation that applies to it."""

    path: Path
    log: EvalLog
    annotations: tuple[Annotation, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> AnnotatedLog:
        """Read the log at ``path`` and its annotations (lifted and edited)."""
        log = read_eval_log(str(path))
        return cls(path=Path(path), log=log, annotations=read_annotations(path, log))


@dataclass(frozen=True)
class Analysis:
    """What an analyzer found: ``text`` to print, ``data`` to serialize, ``files`` to save.

    ``files`` maps a file name to its text content (for example an HTML page);
    the CLI writes them under ``--out``.
    """

    name: str
    text: str
    data: Mapping[str, Any] = field(default_factory=dict)
    files: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class Analyzer(Protocol):
    """Read-only analysis over one or more annotated logs."""

    name: str

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        """Analyze ``logs`` without changing them."""
        ...


# --------------------------------------------------------------------------- #
# Re-scoring
# --------------------------------------------------------------------------- #


def _scorer_for(rubric: Mapping[str, Any] | None, name: str) -> object | None:
    """The registered scorer behind a rubric, or ``None`` when it cannot be resolved."""
    from inspect_robots.registry import resolve

    rescored_by = rubric.get("rescored_by") if rubric is not None else None
    scorer_name = (
        rescored_by
        if isinstance(rescored_by, str)
        else (rubric.get("scorer") if rubric is not None else name)
    )
    if not isinstance(scorer_name, str):
        return None
    try:
        scorer: object = resolve("scorer", scorer_name)
    except Exception:
        return None
    return scorer


def _rescore_sample(
    sample: SceneResult, annotations: Sequence[Annotation], reducer: str
) -> tuple[SceneResult, set[str]]:
    """``sample`` re-scored from ``annotations``, with the rubrics whose trial scores changed."""
    rubrics = {str(r.get("name")): r for r in sample.rubrics if isinstance(r, Mapping)}
    names = list(dict.fromkeys(name for epoch in sample.epochs for name in epoch))
    scorers = {name: _scorer_for(rubrics.get(name), name) for name in names}
    epochs: list[dict[str, float | None]] = []
    changed_rubrics: set[str] = set()
    for index, stored in enumerate(sample.epochs):
        updated = dict(stored)
        for name in stored:
            score_annotations = getattr(scorers[name], "score_annotations", None)
            if not callable(score_annotations):
                continue
            score: Score = score_annotations(annotations, trial_key(sample, index), name)
            value = value_to_float(score.value)
            if value != stored[name]:
                updated[name] = value
                changed_rubrics.add(name)
        epochs.append(updated)
    if not changed_rubrics:
        return sample, changed_rubrics
    reduced = dict(sample.reduced)
    for name in changed_rubrics:
        scores = [Score(value=epoch[name]) for epoch in epochs if name in epoch]
        try:
            reduced[name] = value_to_float(reduce_scores(reducer, scores).value)
        except Exception as exc:
            # eval() records a reducer that cannot reduce (e.g. pass_at_k with
            # too few voting epochs) and still writes the log; re-scoring
            # degrades the same way, dropping the reduction and warning.
            reduced.pop(name, None)
            warnings.warn(
                f"rescore: reducer {reducer!r} failed for {name!r} in scene "
                f"{sample.scene_id!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return replace(sample, epochs=tuple(epochs), reduced=reduced), changed_rubrics


def rescore(log: EvalLog, annotations: Sequence[Annotation]) -> EvalLog:
    """``log`` with its results recomputed from ``annotations``; the raw data is untouched.

    Each trial is re-scored by rubric scorers that offer
    ``score_annotations(annotations, record_id, rubric)``; other scores stay
    as stored. Scene reductions use the log's recorded reducer (``"mean"`` for
    logs written before it was recorded), and a metric and its abstention
    count are recomputed only when one of its trial scores changed, even if
    the scene reduction did not. Errored trials stay unscored.
    """
    reducer = log.eval.reducer or "mean"
    samples: list[SceneResult] = []
    changed: set[str] = set()
    for sample in log.samples:
        rescored, changed_rubrics = _rescore_sample(sample, annotations, reducer)
        samples.append(rescored)
        changed.update(changed_rubrics)
    if not changed:
        return log
    metrics = dict(log.results.metrics)
    abstentions = dict(log.results.abstentions)
    for name in changed:
        values = [s.reduced[name] for s in samples if name in s.reduced]
        voted = [v for v in values if v is not None]
        metrics[name] = mean(voted) if voted else None
        abstaining = sum(1 for s in samples for e in s.epochs if name in e and e[name] is None)
        if abstaining:
            abstentions[name] = abstaining
        else:
            abstentions.pop(name, None)
    results = replace(log.results, metrics=metrics, abstentions=abstentions)
    return replace(log, results=results, samples=tuple(samples))


# --------------------------------------------------------------------------- #
# Builtin analyzers
# --------------------------------------------------------------------------- #


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


class _ScoresAnalyzer:
    """Each log's metrics as stored and as re-scored from its annotations."""

    name = "scores"

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        lines: list[str] = []
        data: dict[str, Any] = {}
        for item in logs:
            rescored = rescore(item.log, item.annotations)
            edits = len(item.annotations) - len(lift_annotations(item.log))
            stored = dict(item.log.results.metrics)
            now = dict(rescored.results.metrics)
            data[str(item.path)] = {
                "task": item.log.eval.task,
                "stored": stored,
                "rescored": now,
                "edits": edits,
            }
            lines.append(f"{item.path} ({item.log.eval.task}; {edits} edit(s) after the run)")
            for name in sorted(set(stored) | set(now)):
                before, after = stored.get(name), now.get(name)
                marker = "" if before == after else "  (changed)"
                lines.append(
                    f"  {name}: {_format_metric(before)} -> {_format_metric(after)}{marker}"
                )
        return Analysis(name=self.name, text="\n".join(lines) + "\n", data=data)


class _TerminationsAnalyzer:
    """How trials ended across logs: counts per reason, and the policy's stated reasons."""

    name = "terminations"

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        counts: Counter[str] = Counter()
        stated: list[dict[str, Any]] = []
        for item in logs:
            for sample in item.log.samples:
                for index, reason in enumerate(sample.termination_reasons):
                    counts[reason or "unknown"] += 1
                    detail = (
                        sample.termination_details[index]
                        if index < len(sample.termination_details)
                        else None
                    )
                    if detail:
                        stated.append(
                            {
                                "log": str(item.path),
                                "record_id": trial_key(sample, index),
                                "scene_id": sample.scene_id,
                                "reason": reason,
                                "detail": detail,
                            }
                        )
        total = sum(counts.values())
        lines = [f"{total} trial(s)"]
        lines += [f"  {reason}: {count}" for reason, count in counts.most_common()]
        if stated:
            lines.append("stated reasons:")
            lines += [f"  [{s['reason']}] {s['scene_id']}: {s['detail']}" for s in stated]
        return Analysis(
            name=self.name,
            text="\n".join(lines) + "\n",
            data={"counts": dict(counts), "stated": stated},
        )


class _BenchmarksAnalyzer:
    """Results grouped by benchmark and version: each task's status and metrics."""

    name = "benchmarks"

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        groups: dict[str, dict[str, Any]] = {}
        for item in logs:
            benchmark = item.log.eval.benchmark
            if not benchmark:
                continue
            key = f"{benchmark.get('name')} v{benchmark.get('version')}"
            rescored = rescore(item.log, item.annotations)
            groups.setdefault(key, {"benchmark": dict(benchmark), "tasks": []})["tasks"].append(
                {
                    "task": item.log.eval.task,
                    "status": item.log.status,
                    "metrics": dict(rescored.results.metrics),
                    "log": str(item.path),
                }
            )
        if not groups:
            return Analysis(name=self.name, text="no benchmark runs among these logs\n")
        lines: list[str] = []
        for key, group in groups.items():
            lines.append(f"{key}: {len(group['tasks'])} task(s)")
            for task in group["tasks"]:
                metrics = ", ".join(
                    f"{name} {_format_metric(value)}" for name, value in task["metrics"].items()
                )
                lines.append(f"  {task['task']} [{task['status']}] {metrics}".rstrip())
        return Analysis(name=self.name, text="\n".join(lines) + "\n", data=groups)


class _DigestAnalyzer:
    """The offline digest ``inspect-robots summarize`` writes without a model, per log."""

    name = "digest"

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        from inspect_robots._summarize import build_digest, load_transcripts

        parts = [build_digest(item.log, load_transcripts(item.log, item.path)) for item in logs]
        return Analysis(name=self.name, text="\n".join(parts))


class _HtmlAnalyzer:
    """The ``inspect-robots view`` page for each log, as files named after the logs."""

    name = "html"

    def analyze(self, logs: Sequence[AnnotatedLog]) -> Analysis:
        from inspect_robots._html import render_html
        from inspect_robots._video import resolve_frames_dir

        # Resolved from each log's own path, as `inspect-robots view` does, so the
        # page carries the same wire-capture and stored-frame evidence.
        files = {
            f"{item.path.stem}.html": render_html(
                item.log,
                title=item.path.name,
                log_path=Path(item.path),
                frames_dir=None
                if item.log.stats.frames_dir is None
                else resolve_frames_dir(item.log.stats.frames_dir, Path(item.path)),
            )
            for item in logs
        }
        return Analysis(
            name=self.name,
            text="".join(f"{name}\n" for name in files),
            files=files,
        )
