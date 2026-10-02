"""Atomic, single-writer manifests for explicitly resumable evaluation sets."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from inspect_robots.embodiment import Embodiment
from inspect_robots.errors import ConfigError
from inspect_robots.log import read_eval_log
from inspect_robots.policy import Policy
from inspect_robots.scorer import Scorer
from inspect_robots.task import Task

_CHECKPOINT_VERSION = 1


def _identity_json_default(value: object) -> object:
    """Normalize the array bounds and string sets in component descriptions."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"unsupported checkpoint identity value: {type(value).__name__}")


def _scorer_identity(scorer: Scorer) -> dict[str, str]:
    """Fingerprint scorer settings without putting their raw values in the manifest."""
    scorer_object: object = scorer
    identity_hook = getattr(scorer, "checkpoint_identity", None)
    if callable(identity_hook):
        config = identity_hook()
    elif is_dataclass(scorer_object):
        config = asdict(cast(Any, scorer_object))
    else:
        raise ConfigError(f"scorer {scorer.name!r} needs a JSON checkpoint_identity() hook")
    encoded = json.dumps(config, sort_keys=True, allow_nan=False, default=_identity_json_default)
    cls = type(scorer)
    return {
        "name": scorer.name,
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "config_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _component_identity(value: object, *, name: str) -> dict[str, str]:
    """Capture effective action middleware settings and require opaque declarations."""
    from inspect_robots.approver import (
        AutoApprover,
        ChainApprover,
        ClampApprover,
        DeltaLimitApprover,
    )
    from inspect_robots.controller import (
        DefaultController,
        EnsemblingController,
        SmoothingController,
    )

    cls = type(value)
    hook = getattr(value, "checkpoint_identity", None)
    config: object
    if callable(hook):
        config = hook()
    elif cls is DefaultController:
        config = {"replan_interval": cast(DefaultController, value).replan_interval}
    elif cls is SmoothingController:
        smoothing = cast(SmoothingController, value)
        config = {
            "alpha": smoothing.alpha,
            "inner": _component_identity(smoothing.inner, name=f"{name}.inner"),
        }
    elif cls is EnsemblingController:
        ensembling = cast(EnsemblingController, value)
        config = {"m": ensembling.m, "action_space": asdict(ensembling.action_space)}
    elif cls is AutoApprover:
        config = {}
    elif cls is ClampApprover:
        config = {"action_space": asdict(cast(ClampApprover, value)._space)}
    elif cls is DeltaLimitApprover:
        limiter = cast(DeltaLimitApprover, value)
        config = (
            {"absolute": True, "delta": limiter._delta}
            if limiter._absolute
            else {"absolute": False, "low": limiter._low, "high": limiter._high}
        )
    elif cls is ChainApprover:
        config = {
            "approvers": [
                _component_identity(approver, name=f"{name}.approvers[{index}]")
                for index, approver in enumerate(cast(ChainApprover, value)._approvers)
            ]
        }
    else:
        raise ConfigError(f"{name} needs a JSON checkpoint_identity() hook in checkpoint mode")
    try:
        encoded = json.dumps(
            config, sort_keys=True, allow_nan=False, default=_identity_json_default
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"{name} checkpoint identity must be JSON serializable: {exc}") from exc
    return {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "config_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _identity(
    tasks: Sequence[Task],
    policy: Policy | str,
    embodiment: Embodiment | str,
    *,
    seed: int,
    log_dir: str,
    options: Mapping[str, object],
) -> dict[str, object]:
    """Describe run inputs whose change would make a saved scene unsafe to reuse."""
    try:
        task_specs = [
            {
                "name": task.name,
                "scenes": [asdict(scene) for scene in task.scenes],
                "epochs": asdict(task.epoch_spec),
                "scorers": [_scorer_identity(scorer) for scorer in task.scorers],
                "max_steps": task.max_steps,
                "max_seconds": task.max_seconds,
                "metadata": task.metadata,
            }
            for task in tasks
        ]
        policy_spec: object = (
            {
                "info": asdict(policy.info),
                "config": asdict(policy.config),
            }
            if not isinstance(policy, str)
            else {"name": policy}
        )
        embodiment_spec: object = (
            asdict(embodiment.info) if not isinstance(embodiment, str) else {"name": embodiment}
        )
        raw = {
            "tasks": task_specs,
            "policy": policy_spec,
            "embodiment": embodiment_spec,
            "seed": seed,
            "log_dir": str(Path(log_dir).resolve()),
            "options": dict(options),
        }
        # Normalize mappings/tuples to JSON values and reject objects or NaN
        # that cannot be compared reliably on a later invocation.
        return json.loads(  # type: ignore[no-any-return]
            json.dumps(raw, sort_keys=True, allow_nan=False, default=_identity_json_default)
        )
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ConfigError(f"checkpoint identity must be JSON serializable: {exc}") from exc


@dataclass
class _Manifest:
    """A validated checkpoint held under its sibling writer lock."""

    path: Path
    identity: dict[str, object]
    attempts: list[dict[str, object]] = field(default_factory=list)
    aggregates: dict[str, str] = field(default_factory=dict)
    in_flight: bool = False

    def attempt_log_path(self, entry: Mapping[str, object]) -> Path:
        """Resolve and validate a referenced immutable attempt log."""
        try:
            index = entry["task_index"]
            scene_ids = entry["scene_ids"]
            relative = entry["log"]
            tasks = self.identity["tasks"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(tasks, list)
                or index < 0
                or index >= len(tasks)
                or not isinstance(scene_ids, list)
                or not all(isinstance(value, str) for value in scene_ids)
                or not isinstance(relative, str)
                or Path(relative).is_absolute()
            ):
                raise ValueError("invalid attempt entry")
            path = (self.path.parent / relative).resolve()
            log_dir = Path(str(self.identity["log_dir"]))
            if not path.is_relative_to(log_dir) or not path.is_file():
                raise ValueError("attempt log is outside log_dir or missing")
            log = read_eval_log(str(path))
            task_spec = tasks[index]
            if not isinstance(task_spec, dict) or log.eval.task != task_spec["name"]:
                raise ValueError("attempt task does not match checkpoint")
            if log.eval.seed != self.identity["seed"]:
                raise ValueError("attempt seed does not match checkpoint")
            if any(sample.scene_id not in scene_ids for sample in log.samples):
                raise ValueError("attempt scenes do not match checkpoint")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ConfigError(f"invalid checkpoint attempt: {exc}") from exc
        return path

    def add_attempt(self, task_index: int, scene_ids: list[str], log_path: Path) -> None:
        """Publish an attempt only after its referenced log is readable."""
        entry: dict[str, object] = {
            "task_index": task_index,
            "scene_ids": scene_ids,
            "log": os.path.relpath(log_path, self.path.parent),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.attempt_log_path(entry)
        prior_in_flight = self.in_flight
        self.attempts.append(entry)
        self.in_flight = False
        try:
            self.write_atomic()
        except Exception:
            self.attempts.pop()
            self.in_flight = prior_in_flight
            raise

    def start_attempt(self) -> None:
        """Mark an attempt ambiguous until its durable log is in the manifest."""
        prior = self.in_flight
        self.in_flight = True
        try:
            self.write_atomic()
        except Exception:
            self.in_flight = prior
            raise

    def set_aggregate(self, task_index: int, log_path: Path) -> None:
        """Point to the latest aggregate without altering earlier attempt logs."""
        prior = self.aggregates.get(str(task_index))
        self.aggregates[str(task_index)] = os.path.relpath(log_path, self.path.parent)
        try:
            self.write_atomic()
        except Exception:
            if prior is None:
                del self.aggregates[str(task_index)]
            else:
                self.aggregates[str(task_index)] = prior
            raise

    def write_atomic(self) -> None:
        """Replace the manifest only after a synced temporary file is complete."""
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = {
            "version": _CHECKPOINT_VERSION,
            "identity": self.identity,
            "attempts": self.attempts,
            "aggregates": self.aggregates,
            "in_flight": self.in_flight,
        }
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)


def _checkpoint_seed(path: Path) -> int | None:
    """Read an existing seed before identity construction; open revalidates it."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = data["identity"]["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed is not an integer")
        return seed
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ConfigError(f"invalid checkpoint seed: {exc}") from exc


@contextmanager
def _open_checkpoint(path: Path, identity: dict[str, object]) -> Iterator[_Manifest]:
    """Open a matching checkpoint with exclusive ownership until exit."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ConfigError(f"checkpoint lock exists: {lock}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        if path.exists():
            try:
                data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
                if data["version"] != _CHECKPOINT_VERSION:
                    raise ValueError("unsupported checkpoint schema version")
                if data["identity"] != identity:
                    raise ConfigError("checkpoint identity differs from this evaluation set")
                attempts = data["attempts"]
                aggregates = data.get("aggregates", {})
                in_flight = data.get("in_flight", False)
                if (
                    not isinstance(attempts, list)
                    or not isinstance(aggregates, dict)
                    or not isinstance(in_flight, bool)
                ):
                    raise ValueError("invalid checkpoint entries")
                if in_flight:
                    raise ConfigError(
                        "checkpoint has an unfinished attempt without a saved log; "
                        "inspect the robot and attempt files before starting a new checkpoint"
                    )
                manifest = _Manifest(path, identity, attempts, aggregates, in_flight)
                for entry in manifest.attempts:
                    manifest.attempt_log_path(entry)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise ConfigError(f"invalid checkpoint: {exc}") from exc
        else:
            manifest = _Manifest(path, identity)
            manifest.write_atomic()
        yield manifest
    finally:
        lock.unlink()
