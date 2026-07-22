"""Tests for seconds-based task horizons resolved against embodiment control rates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from inspect_robots import eval
from inspect_robots.compat import check_compatibility
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import CompatibilityError, ConfigError
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task


def _make_scene() -> Scene:
    return Scene(id="s0", instruction="reach target", init_seed=0)


def test_task_validation_requires_exactly_one_horizon() -> None:
    # Neither specified
    with pytest.raises(ConfigError, match="specify exactly one of max_steps or max_seconds"):
        Task(name="t", scenes=[_make_scene()], scorer=success_at_end())

    # Both specified
    with pytest.raises(ConfigError, match="specify exactly one of max_steps or max_seconds"):
        Task(
            name="t",
            scenes=[_make_scene()],
            scorer=success_at_end(),
            max_steps=10,
            max_seconds=5.0,
        )


def test_task_validation_invalid_max_steps() -> None:
    with pytest.raises(ConfigError, match="max_steps must be an integer >= 1"):
        Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_steps=0)

    with pytest.raises(ConfigError, match="max_steps must be an integer >= 1"):
        Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_steps=cast(Any, True))


def test_task_validation_invalid_max_seconds() -> None:
    with pytest.raises(ConfigError, match="max_seconds must be a number > 0"):
        Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=0.0)

    with pytest.raises(ConfigError, match="max_seconds must be a number > 0"):
        Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=-1.5)

    with pytest.raises(ConfigError, match="max_seconds must be a number > 0"):
        Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=cast(Any, True))


def test_task_envelope_with_max_seconds_raises() -> None:
    task = Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=5.0)
    with pytest.raises(ConfigError, match="step envelope must be resolved at eval"):
        _ = task.envelope


def test_compat_error_when_control_hz_missing() -> None:
    task = Task(name="t", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=5.0)
    emb = CubePickEmbodiment()

    # Modify embodiment to have no control_hz
    emb.info = EmbodimentInfo(
        name="no_hz_emb",
        action_space=emb.info.action_space,
        observation_space=emb.info.observation_space,
        control_hz=None,
    )

    report = check_compatibility(ScriptedPolicy(), emb, task)
    assert any(issue.code == "control_hz_missing" for issue in report.errors)

    with pytest.raises(CompatibilityError, match="control_hz_missing"):
        eval(task, ScriptedPolicy(), emb)


def test_eval_resolves_max_seconds_with_ceil(tmp_path: Path) -> None:
    # 5.5 seconds at 10.0 Hz control_hz => ceil(55.0) = 55 steps
    task = Task(name="t_seconds", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=5.5)
    emb = CubePickEmbodiment()  # 10 Hz

    (log,) = eval(task, ScriptedPolicy(), emb, log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.eval.max_seconds == 5.5
    assert log.eval.max_steps == 55


def test_eval_resolves_max_seconds_fractional_ceil(tmp_path: Path) -> None:
    # 0.12 seconds at 10.0 Hz => ceil(1.2) = 2 steps
    task = Task(name="t_short", scenes=[_make_scene()], scorer=success_at_end(), max_seconds=0.12)
    emb = CubePickEmbodiment()  # 10 Hz

    (log,) = eval(task, ScriptedPolicy(), emb, log_dir=str(tmp_path))
    assert log.eval.max_seconds == 0.12
    assert log.eval.max_steps == 2
