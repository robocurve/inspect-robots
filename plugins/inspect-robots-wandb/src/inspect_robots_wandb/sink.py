"""Weights & Biases (W&B) logging sink implementation."""

from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog, EvalSpec


class WandbSink:
    """Stream evaluation runs, metrics, and trial summaries to Weights & Biases."""

    def __init__(
        self,
        project: str = "inspect-robots",
        name: str | None = None,
        group: str | None = None,
        mode: str | None = None,
    ) -> None:
        self.project = project
        self.name = name
        self.group = group
        self.mode = mode
        self._wandb: Any = None
        self._run: Any = None
        self._warned_missing = False

    def _ensure_wandb(self) -> Any:
        if self._wandb is not None:
            return self._wandb
        if importlib.util.find_spec("wandb") is None:
            if not self._warned_missing:
                warnings.warn(
                    "wandb is not installed; WandbSink is a no-op. "
                    "Install with: pip install 'inspect-robots-wandb[wandb]'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_missing = True
            return None
        import wandb

        self._wandb = wandb
        return wandb

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Initialize the W&B run at the start of evaluation."""
        wb = self._ensure_wandb()
        if wb is None:
            return
        run_name = self.name or f"{spec.task}_{spec.policy}"
        config = {
            "task": spec.task,
            "policy": spec.policy,
            "embodiment": spec.embodiment,
            "max_steps": spec.max_steps,
            "max_seconds": spec.max_seconds,
        }
        self._run = wb.init(
            project=self.project,
            name=run_name,
            group=self.group,
            mode=self.mode,
            config=config,
            reinit=True,
        )

    def on_eval_end(self, log: EvalLog) -> None:
        """Log final evaluation summary metrics and finish the W&B run."""
        wb = self._ensure_wandb()
        if wb is None or self._run is None:
            return
        metrics: dict[str, Any] = {
            "status": log.status,
            "total_scenes": log.results.total_scenes,
            "total_trials": log.results.total_trials,
            "errored_trials": log.results.errored_trials,
            "duration_s": log.stats.duration_s,
            "total_steps": log.stats.total_steps,
        }
        for k, v in log.results.metrics.items():
            metrics[f"metrics/{k}"] = v
        self._run.log(metrics)
        self._run.finish()
        self._run = None


def wandb_sink(
    project: str = "inspect-robots",
    name: str | None = None,
    group: str | None = None,
    mode: str | None = None,
) -> WandbSink:
    """Factory entry point for inspect_robots.sinks registry."""
    return WandbSink(project=project, name=name, group=group, mode=mode)
