"""Runtime contract and trusted-file loader for setup-local task evaluators."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

import mujoco
import numpy as np
from inspect_robots.errors import ConfigError


@dataclass(frozen=True)
class Evaluation:
    """Task outcome after one physics control step."""

    success: bool
    distance: float
    info: dict[str, object]


@runtime_checkable
class TargetEvaluator(Protocol):
    """Setup-local state projection and success predicate."""

    def supports(self, model: mujoco.MjModel) -> bool: ...

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: object,
        control_hz: float,
        *,
        particles: np.ndarray | None = None,
    ) -> None: ...

    def evaluate(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        particles: np.ndarray | None = None,
    ) -> Evaluation: ...

    def state(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]: ...

    def marker(self) -> tuple[np.ndarray, float] | None: ...


@dataclass(frozen=True)
class SetupFiles:
    """The fixed files belonging to one trusted Newton setup directory."""

    directory: Path
    model: Path
    evaluator: Path


def resolve_setup(path: str | Path) -> SetupFiles:
    """Resolve a setup directory containing exactly the required runtime files."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigError(f"Setup directory not found: {directory}")
    model = directory / "model.xml"
    evaluator = directory / "evaluator.py"
    if not model.is_file():
        raise ConfigError(f"Setup is missing model.xml: {directory}")
    if not evaluator.is_file():
        raise ConfigError(f"Setup is missing evaluator.py: {directory}")
    return SetupFiles(directory, model, evaluator)


def load_evaluator(
    path: str | Path, target_kind: str, model: mujoco.MjModel
) -> tuple[TargetEvaluator, str]:
    """Execute a trusted setup evaluator and validate its small public contract."""
    evaluator_path = Path(path).resolve()
    try:
        source = evaluator_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Cannot read setup evaluator {evaluator_path}: {exc}") from exc
    digest = hashlib.sha256(source).hexdigest()
    module = _load_module(evaluator_path, digest)
    declared_kind = getattr(module, "TARGET_KIND", None)
    if declared_kind != target_kind:
        raise ConfigError(
            f"Setup evaluator supports target kind {declared_kind!r}, not {target_kind!r}"
        )
    factory = getattr(module, "create_evaluator", None)
    if not callable(factory):
        raise ConfigError("Setup evaluator must define create_evaluator()")
    try:
        evaluator = factory()
    except Exception as exc:
        raise ConfigError(f"Setup evaluator factory failed: {exc}") from exc
    if not isinstance(evaluator, TargetEvaluator):
        raise ConfigError("create_evaluator() returned an invalid evaluator")
    try:
        supported = evaluator.supports(model)
    except Exception as exc:
        raise ConfigError(f"Setup evaluator compatibility check failed: {exc}") from exc
    if supported is not True:
        raise ConfigError(
            f"MJCF does not provide the entities required by target kind {target_kind!r}"
        )
    return evaluator, digest


def _load_module(path: Path, digest: str) -> ModuleType:
    """Load by content identity so similarly named setup files cannot collide."""
    module_name = f"_inspect_robots_newton_setup_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Cannot load setup evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"Cannot execute setup evaluator {path}: {exc}") from exc
    return module
