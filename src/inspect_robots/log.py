"""The immutable evaluation log — Inspect Robots's reproducible record of a run.

Mirrors Inspect AI's ``EvalLog``: ``version`` + ``status`` + ``eval`` spec +
``results`` + ``stats`` + per-scene ``samples`` + ``error``. Serialized to JSON
with a schema version so newer Inspect Robots always reads older logs (a read-back
guarantee enforced by golden tests in a later step).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

SCHEMA_VERSION = 1


@dataclass
class EvalSpec:
    """Top-level identity of an eval: what was run, with what, when."""

    task: str
    policy: str
    embodiment: str
    created: str
    inspect_robots_version: str
    git_commit: str | None = None
    policy_config: dict[str, Any] = field(default_factory=dict)
    embodiment_info: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


@dataclass
class EvalStats:
    """Timing and execution statistics for a run."""

    started_at: str
    completed_at: str
    duration_s: float
    total_steps: int
    mean_inference_latency_s: float | None = None
    # Directory of streamed camera frame side-cars, if frame logging was enabled.
    frames_dir: str | None = None


@dataclass
class SceneResult:
    """Per-scene result: the reduced score(s) plus the raw per-epoch scores."""

    scene_id: str
    status: str  # "success" | "error"
    reduced: dict[str, float] = field(default_factory=dict)
    epochs: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResults:
    """Aggregate results across all scenes."""

    total_scenes: int
    total_trials: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalLog:
    """The full record returned by [`eval`][inspect_robots.eval.eval] and persisted to disk."""

    version: int
    status: str  # "started" | "success" | "error"
    eval: EvalSpec
    results: EvalResults
    stats: EvalStats
    samples: list[SceneResult] = field(default_factory=list)
    error: str | None = None

    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalLog:
        if data.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported eval-log schema version {data.get('version')!r}; "
                f"this Inspect Robots reads version {SCHEMA_VERSION}"
            )
        return cls(
            version=data["version"],
            status=data["status"],
            eval=EvalSpec(**data["eval"]),
            results=EvalResults(**data["results"]),
            stats=EvalStats(**data["stats"]),
            samples=[SceneResult(**s) for s in data["samples"]],
            error=data.get("error"),
        )


def read_eval_log(path: str) -> EvalLog:
    """Read an [`EvalLog`][inspect_robots.log.EvalLog] back from a JSON file on disk."""
    with Path(path).open(encoding="utf-8") as fh:
        return EvalLog.from_dict(json.load(fh))
