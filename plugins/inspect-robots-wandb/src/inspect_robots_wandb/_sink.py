"""The W&B-backed implementation of the Inspect Robots sink protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from importlib import import_module
from typing import Any

from inspect_robots.log import EvalLog, EvalSpec
from inspect_robots.logging import NullSink


def _load_wandb() -> Any:
    """Import the optional SDK only when a W&B run is requested."""
    try:
        return import_module("wandb")
    except ImportError as exc:
        raise RuntimeError(
            "WandbSink requires the wandb package; install it with "
            "'pip install inspect-robots-wandb'"
        ) from exc


class WandbSink(NullSink):
    """Log Inspect Robots evaluation summaries to one W&B run per evaluation."""

    def __init__(
        self,
        project: str = "inspect-robots",
        *,
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: Sequence[str] | None = None,
        mode: str | None = None,
        dir: str | None = None,
    ) -> None:
        """Configure the W&B project and optional run metadata."""
        if not project.strip():
            raise ValueError("project must not be empty")
        self.project = project
        self.entity = entity
        self.name = name
        self.group = group
        self.tags = tuple(tags) if tags is not None else None
        self.mode = mode
        self.dir = dir
        self._run: Any = None

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Create a W&B run and record the immutable evaluation specification."""
        if self._run is not None:
            previous_run = self._run
            self._run = None
            previous_run.finish(exit_code=1)

        wandb = _load_wandb()
        kwargs: dict[str, Any] = {
            "project": self.project,
            "config": asdict(spec),
            "reinit": "create_new",
        }
        optional = {
            "entity": self.entity,
            "name": self.name,
            "group": self.group,
            "tags": list(self.tags) if self.tags is not None else None,
            "mode": self.mode,
            "dir": self.dir,
        }
        kwargs.update({key: value for key, value in optional.items() if value is not None})
        self._run = wandb.init(**kwargs)

    def on_eval_end(self, log: EvalLog) -> None:
        """Log aggregate evaluation values and close the W&B run."""
        run = self._run
        self._run = None
        if run is None:
            return

        payload: dict[str, Any] = {
            "eval/status": log.status,
            "eval/total_scenes": log.results.total_scenes,
            "eval/total_trials": log.results.total_trials,
            "eval/errored_trials": log.results.errored_trials,
            "eval/total_steps": log.stats.total_steps,
            "eval/duration_s": log.stats.duration_s,
        }
        payload.update({f"metric/{name}": value for name, value in log.results.metrics.items()})
        exit_code = 0 if log.status == "success" else 1
        try:
            run.log(payload)
        except Exception:
            exit_code = 1
            raise
        finally:
            run.finish(exit_code=exit_code)

    def on_eval_error(self, error: BaseException) -> None:
        """Record an escaped evaluation exception and close the active run."""
        run = self._run
        self._run = None
        if run is None:
            return

        try:
            run.log(
                {
                    "eval/status": "error",
                    "eval/error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            run.finish(exit_code=1)
