"""Maintain an atomic, schema-valid snapshot while an evaluation is running."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from inspect_robots.log import (
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
    _json_safe_scene_metadata,
)
from inspect_robots.logging.json_log import _sanitize, _slug
from inspect_robots.transcript import judgement_source

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene
    from inspect_robots.types import Action, Observation, StepResult


@dataclass
class _LiveScene:
    """Hold mutable parallel trial fields until the next immutable snapshot."""

    scene_id: str
    status: str = "success"
    error: str | None = None
    epochs: list[dict[str, float]] = field(default_factory=list)
    operator_judgements: list[str | None] = field(default_factory=list)
    judgement_sources: list[str | None] = field(default_factory=list)
    operator_notes: list[str | None] = field(default_factory=list)
    operator_messages: list[tuple[dict[str, Any], ...]] = field(default_factory=list)
    trial_metadata: list[dict[str, Any]] = field(default_factory=list)
    termination_reasons: list[str | None] = field(default_factory=list)
    policy_transcripts: list[Any] = field(default_factory=list)


class LiveLogSink:
    """Persist reusable in-progress snapshots without becoming the final-log sink.

    One instance may be reused across sequential runs: ``on_eval_start`` clears
    all prior state and chooses a new path. Pair this sink with
    [`JsonLogSink`][inspect_robots.logging.JsonLogSink] when a final log is
    required. A sinks list containing only ``LiveLogSink`` leaves no log after
    a successful run because ``on_eval_end`` removes the live snapshot.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        min_write_interval_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.log_dir = Path(log_dir)
        self.min_write_interval_s = min_write_interval_s
        self._clock = clock
        self.path: Path | None = None
        self._disabled = False
        self._warned = False
        self._spec: EvalSpec | None = None
        self._scenes: dict[str, _LiveScene] = {}
        self._current_scene: _LiveScene | None = None
        self._current_index: int | None = None
        self._current_transcript: list[Any] = []
        self._current_step = 0
        self._current_steps = 0
        self._completed_trials = 0
        self._completed_steps = 0
        self._errored_trials = 0
        self._latencies: list[float] = []
        self._started_clock = 0.0
        self._last_write_clock: float | None = None
        self._frames_dir: str | None = None
        self._bound_scenes: dict[str, tuple[str | None, dict[str, Any]]] = {}

    def bind_frames_dir(self, frames_dir: str | None) -> None:
        """Bind the frame directory that every snapshot for the next run records."""
        self._frames_dir = frames_dir

    def bind_scenes(self, scenes: Sequence[Scene]) -> None:
        """Freeze scene identity for live snapshots before the next run starts.

        Metadata is copied at bind time, so caller mutation during a run can
        transiently diverge from the final log that replaces the snapshot.
        """
        self._bound_scenes = {
            scene.id: (
                scene.instruction,
                _json_safe_scene_metadata(scene.metadata),
            )
            for scene in scenes
        }

    def _disable(self, exc: Exception) -> None:
        """Warn once and turn all later hooks into no-ops for this run."""
        if not self._warned:
            with suppress(Exception):
                print(
                    f"warning: live JSON logging disabled after {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            self._warned = True
        self._disabled = True

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Reset all run state and publish the first ``started`` snapshot."""
        self._disabled = False
        self._warned = False
        try:
            self._spec = spec
            self._scenes = {}
            self._current_scene = None
            self._current_index = None
            self._current_transcript = []
            self._current_step = 0
            self._current_steps = 0
            self._completed_trials = 0
            self._completed_steps = 0
            self._errored_trials = 0
            self._latencies = []
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.log_dir / f"{_slug(spec.task)}_{uuid.uuid4().hex[:8]}.live.json"
            now = self._clock()
            self._started_clock = now
            self._last_write_clock = None
            self._write(now, force=True)
        except Exception as exc:
            self._disable(exc)

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Append one strictly parallel in-progress slot and publish it immediately."""
        if self._disabled:
            return
        try:
            del epoch
            scene = self._scenes.setdefault(scene_id, _LiveScene(scene_id))
            scene.epochs.append({})
            scene.operator_judgements.append(None)
            scene.judgement_sources.append(None)
            scene.operator_notes.append(None)
            scene.operator_messages.append(())
            scene.trial_metadata.append({})
            scene.termination_reasons.append(None)
            self._current_transcript = []
            scene.policy_transcripts.append(self._current_transcript)
            self._current_scene = scene
            self._current_index = len(scene.epochs) - 1
            self._current_step = 0
            self._current_steps = 0
            self._write(self._clock(), force=True)
        except Exception as exc:
            self._disable(exc)

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Record the step index without serializing or touching the filesystem."""
        if self._disabled:
            return
        try:
            del observation, action, result
            self._current_step = int(t)
            self._current_steps = max(self._current_steps, self._current_step + 1)
        except Exception as exc:
            self._disable(exc)

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        """Accumulate a shallow delta snapshot and publish it at the throttled cadence."""
        if self._disabled:
            return
        try:
            self._current_step = int(t)
            self._current_transcript.extend(
                dict(message) if isinstance(message, dict) else message for message in messages
            )
            self._write(self._clock(), force=False)
        except Exception as exc:
            self._disable(exc)

    def on_trial_end(self, record: TrialRecord) -> None:
        """Replace the active slot with the completed record and force a snapshot."""
        if self._disabled:
            return
        try:
            if self._current_scene is None or self._current_index is None:
                raise RuntimeError("trial ended without a matching live trial start")
            scene = self._current_scene
            index = self._current_index
            scene.operator_judgements[index] = record.operator_judgement
            scene.judgement_sources[index] = judgement_source(record)
            scene.operator_notes[index] = record.operator_note
            scene.operator_messages[index] = tuple(
                {
                    "t": event.t,
                    "text": event.data["text"],
                    "source": event.data.get("source", "console"),
                }
                for event in record.events
                if event.kind == "operator_message"
            )
            scene.trial_metadata[index] = dict(record.metadata)
            scene.termination_reasons[index] = record.termination_reason
            scene.policy_transcripts[index] = (
                record.policy_transcript
                if record.policy_transcript is not None
                else self._current_transcript
            )
            if record.status == "error":
                scene.status = "error"
                scene.error = record.error
                self._errored_trials += 1
            elif record.status == "cancelled" and scene.status != "error":
                scene.status = "cancelled"
                scene.error = record.error
            self._completed_trials += 1
            self._completed_steps += len(record.steps)
            self._latencies.extend(record.inference_latencies)
            self._current_scene = None
            self._current_index = None
            self._current_transcript = []
            self._current_steps = 0
            self._write(self._clock(), force=True)
        except Exception as exc:
            self._disable(exc)

    def on_eval_end(self, log: EvalLog) -> None:
        """Remove the transient snapshot after the canonical sink writes the final log."""
        try:
            del log
            if self.path is not None:
                self.path.unlink(missing_ok=True)
        except Exception as exc:
            self._disable(exc)

    def _samples(self, updated_at: str) -> tuple[SceneResult, ...]:
        """Freeze the mutable scene state with an updated active-trial marker."""
        samples: list[SceneResult] = []
        for scene in self._scenes.values():
            bound_scene = self._bound_scenes.get(scene.scene_id)
            instruction = bound_scene[0] if bound_scene is not None else None
            scene_metadata = bound_scene[1] if bound_scene is not None else {}
            metadata = [dict(value) for value in scene.trial_metadata]
            if scene is self._current_scene and self._current_index is not None:
                metadata[self._current_index] = {
                    "live": {"step": self._current_step, "updated_at": updated_at}
                }
            samples.append(
                SceneResult(
                    scene_id=scene.scene_id,
                    status=scene.status,
                    reduced={},
                    epochs=tuple(scene.epochs),
                    error=scene.error,
                    instruction=instruction,
                    scene_metadata=dict(scene_metadata),
                    operator_judgements=tuple(scene.operator_judgements),
                    judgement_sources=tuple(scene.judgement_sources),
                    operator_notes=tuple(scene.operator_notes),
                    operator_messages=tuple(scene.operator_messages),
                    trial_metadata=tuple(metadata),
                    termination_reasons=tuple(scene.termination_reasons),
                    policy_transcripts=tuple(scene.policy_transcripts),
                )
            )
        return tuple(samples)

    def _write(self, now: float, *, force: bool) -> None:
        """Atomically write one snapshot when forced or outside the throttle window."""
        if (
            not force
            and self._last_write_clock is not None
            and now - self._last_write_clock < self.min_write_interval_s
        ):
            return
        if self._spec is None or self.path is None:
            raise RuntimeError("live sink has not started an evaluation")
        updated_at = datetime.now(timezone.utc).isoformat()
        duration = max(0.0, now - self._started_clock)
        live_log = EvalLog(
            version=EvalLog.SCHEMA_VERSION,
            status="started",
            eval=self._spec,
            results=EvalResults(
                total_scenes=len(self._scenes),
                total_trials=self._completed_trials,
                metrics={},
                errored_trials=self._errored_trials,
            ),
            stats=EvalStats(
                started_at=self._spec.created,
                completed_at=updated_at,
                duration_s=duration,
                total_steps=self._completed_steps + self._current_steps,
                mean_inference_latency_s=(mean(self._latencies) if self._latencies else None),
                frames_dir=self._frames_dir,
            ),
            samples=self._samples(updated_at),
        )
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                _sanitize(live_log.to_dict()),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        os.replace(tmp, self.path)
        self._last_write_clock = now
