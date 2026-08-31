"""The immutable evaluation log — Inspect Robots's reproducible record of a run.

Mirrors Inspect AI's ``EvalLog``: ``version`` + ``status`` + ``eval`` spec +
``results`` + ``stats`` + per-scene ``samples`` + ``error``. Serialized to JSON
with a schema version so newer Inspect Robots always reads older logs (a read-back
guarantee enforced by golden tests in a later step).

Immutability is *shallow*: the dataclasses are frozen and sequence fields are
tuples, so reassigning a field or mutating the sample list is impossible — but
dict-valued fields (``SceneResult.reduced``, the per-epoch score dicts,
``EvalResults.metrics``, ``EvalSpec.policy_config`` / ``embodiment_info``, and
``SceneResult.scene_metadata``)
remain plain mutable dicts, as do the dictionaries inside
``SceneResult.operator_messages``. ``SceneResult.policy_transcripts`` entries
are arbitrary mutable JSON values. Treat a log as read-only; nothing
deep-freezes it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar


def _json_safe_scene_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy each JSON-encodable value and omit values encoding rejects."""
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        try:
            safe[key] = json.loads(json.dumps(value))
        except (TypeError, ValueError, OverflowError):
            continue
    return safe


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalSpec:
    """Top-level identity and configured horizon of a reproducible eval.

    ``max_steps`` is always the resolved integer budget used by the rollout.
    ``max_seconds`` preserves a benchmark's declared physical-time budget when
    that integer was derived from the embodiment's control rate.

    The three optional provenance fields record what simulator build, asset
    revision, and policy checkpoint were used, so a published number can be
    re-derived months later even if the sim, scene assets, or policy have moved:

    - ``environment_id``: simulator or hardware-rig identifier (e.g. ``"isaacsim-2026.1.0"``).
    - ``environment_revision``: hash or tag of the scene/asset bundle (e.g. a git SHA).
    - ``policy_checkpoint``: hash, path, or Hugging Face revision of the model checkpoint.
    """

    task: str
    policy: str
    embodiment: str
    created: str
    inspect_robots_version: str
    git_commit: str | None = None
    policy_config: dict[str, Any] = field(default_factory=dict)
    embodiment_info: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    max_steps: int | None = None
    max_seconds: float | None = None
    # Environment and checkpoint provenance (plan 0053).
    environment_id: str | None = None
    environment_revision: str | None = None
    policy_checkpoint: str | None = None


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
    """Persist one scene's metadata, reduced scores, and raw per-epoch records."""

    scene_id: str
    status: str  # "success" | "error" | "cancelled"
    reduced: dict[str, float] = field(default_factory=dict)
    epochs: tuple[dict[str, float], ...] = ()
    error: str | None = None
    # What the scene asked the policy to do — makes a log self-describing.
    instruction: str | None = None
    # JSON-safe scene metadata, copied per key so one adapter-owned object does
    # not prevent the rest of the scene contract from being persisted.
    scene_metadata: dict[str, Any] = field(default_factory=dict)
    # Strictly parallel to ``epochs``: the operator's verdict per recorded
    # trial, ``None`` when the trial errored or no judgement was captured.
    # Defaults keep logs written before these fields existed readable.
    operator_judgements: tuple[str | None, ...] = ()
    # Strictly parallel to ``epochs``: which path produced each recorded
    # operator judgement ("console", "prompt", "embodiment", "vlm"), ``None``
    # when the trial has no judgement (errored trials, no grader, a skipped
    # prompt) or the grader recorded one without an operator event. The default
    # keeps older logs readable.
    judgement_sources: tuple[str | None, ...] = ()
    # Strictly parallel to ``epochs``: qualitative operator context per trial,
    # ``None`` when no note was captured. Read by nothing that scores.
    operator_notes: tuple[str | None, ...] = ()
    # Strictly parallel to ``epochs``: live feedback drained during each trial.
    # Both sequence layers are tuples to preserve the log's shallow immutability.
    operator_messages: tuple[tuple[dict[str, Any], ...], ...] = ()
    # Strictly parallel to ``epochs``: trial-specific metadata from the policy.
    trial_metadata: tuple[dict[str, Any], ...] = ()
    # Strictly parallel to ``epochs``: why each recorded trial ended, or
    # ``None`` for errored trials. The default keeps older schema-v1 logs readable.
    termination_reasons: tuple[str | None, ...] = ()
    # Strictly parallel to ``epochs``: the policy's audit record per trial,
    # ``None`` when unavailable. The default keeps older schema-v1 logs readable.
    policy_transcripts: tuple[Any, ...] = ()


@dataclass(frozen=True)
class EvalResults:
    """Aggregate results across all scenes."""

    total_scenes: int
    total_trials: int
    metrics: dict[str, float] = field(default_factory=dict)
    # Errored trials, which are recorded but never scored (visible per-scene
    # as empty entries in ``SceneResult.epochs``). The default keeps logs
    # written before this field existed readable.
    errored_trials: int = 0


@dataclass(frozen=True)
class EvalLog:
    """The full record returned by [`eval`][inspect_robots.eval.eval] and persisted to disk."""

    version: int
    status: str  # "started" | "success" | "error" | "cancelled"
    eval: EvalSpec
    results: EvalResults
    stats: EvalStats
    samples: tuple[SceneResult, ...] = ()
    error: str | None = None

    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete log to nested dictionaries and sequences."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalLog:
        """Reconstruct an immutable log, rejecting unsupported schema versions."""
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
            sample["judgement_sources"] = tuple(sample.get("judgement_sources", ()))
            sample["operator_notes"] = tuple(sample.get("operator_notes", ()))
            sample["operator_messages"] = tuple(
                tuple(messages) for messages in sample.get("operator_messages", ())
            )
            sample["trial_metadata"] = tuple(sample.get("trial_metadata", ()))
            sample["termination_reasons"] = tuple(sample.get("termination_reasons", ()))
            sample["policy_transcripts"] = tuple(sample.get("policy_transcripts", ()))
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
