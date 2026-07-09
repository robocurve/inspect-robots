"""The immutable evaluation log — Inspect Robots's reproducible record of a run.

Mirrors Inspect AI's ``EvalLog``: ``version`` + ``status`` + ``eval`` spec +
``results`` + ``stats`` + per-scene ``samples`` + ``error``. Serialized to JSON
with a schema version so newer Inspect Robots always reads older logs (a read-back
guarantee enforced by golden tests in a later step).

Immutability is *shallow*: the dataclasses are frozen and sequence fields are
tuples, so reassigning a field or mutating the sample list is impossible — but
dict-valued fields (``SceneResult.reduced``, the per-epoch score dicts,
``EvalResults.metrics``, ``EvalSpec.policy_config`` / ``embodiment_info``)
remain plain mutable dicts. Treat a log as read-only; nothing deep-freezes it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

SCHEMA_VERSION = 1


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class EvalStats:
    """Timing and execution statistics for a run."""

    started_at: str
    completed_at: str
    duration_s: float
    total_steps: int
    mean_inference_latency_s: float | None = None
    # Directory of streamed camera frame side-cars, if frame logging was enabled.
    frames_dir: str | None = None


@dataclass(frozen=True)
class SceneResult:
    """Per-scene result: the reduced score(s) plus the raw per-epoch scores."""

    scene_id: str
    status: str  # "success" | "error"
    reduced: dict[str, float] = field(default_factory=dict)
    epochs: tuple[dict[str, float], ...] = ()
    error: str | None = None
    # What the scene asked the policy to do — makes a log self-describing.
    instruction: str | None = None
    # Strictly parallel to ``epochs``: the operator's verdict per recorded
    # trial, ``None`` when the trial errored or no judgement was captured.
    # Defaults keep logs written before these fields existed readable.
    operator_judgements: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class EvalResults:
    """Aggregate results across all scenes."""

    total_scenes: int
    total_trials: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalLog:
    """The full record returned by [`eval`][inspect_robots.eval.eval] and persisted to disk."""

    version: int
    status: str  # "started" | "success" | "error"
    eval: EvalSpec
    results: EvalResults
    stats: EvalStats
    samples: tuple[SceneResult, ...] = ()
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
        samples = []
        for raw in data["samples"]:
            sample = dict(raw)
            # JSON has no tuple type: coerce the sequence fields it deserializes
            # as lists back into tuples so a read-back log is genuinely immutable
            # too, not just one freshly returned by eval(). ``.get`` covers a log
            # written before ``operator_judgements`` existed (newer reads older).
            sample["epochs"] = tuple(sample.get("epochs", ()))
            sample["operator_judgements"] = tuple(sample.get("operator_judgements", ()))
            samples.append(SceneResult(**sample))
        return cls(
            version=data["version"],
            status=data["status"],
            eval=EvalSpec(**data["eval"]),
            results=EvalResults(**data["results"]),
            stats=EvalStats(**data["stats"]),
            samples=tuple(samples),
            error=data.get("error"),
        )


def read_eval_log(path: str) -> EvalLog:
    """Read an [`EvalLog`][inspect_robots.log.EvalLog] back from a JSON file on disk."""
    with Path(path).open(encoding="utf-8") as fh:
        return EvalLog.from_dict(json.load(fh))
