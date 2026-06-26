"""EvalLog schema: round-trip, golden read-back guarantee, atomicity; eval_set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robolens import eval_set, read_eval_log
from robolens.log import (
    SCHEMA_VERSION,
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
)
from robolens.mock import CubePickEmbodiment, ScriptedPolicy
from robolens.scene import Scene
from robolens.scorer import success_at_end
from robolens.task import Task


def _golden_log() -> EvalLog:
    return EvalLog(
        version=SCHEMA_VERSION,
        status="success",
        eval=EvalSpec(
            task="demo",
            policy="scripted",
            embodiment="cubepick",
            created="2026-06-26T00:00:00+00:00",
            robolens_version="0.0.0",
            git_commit="deadbeef",
            seed=0,
        ),
        results=EvalResults(total_scenes=1, total_trials=1, metrics={"success_at_end": 1.0}),
        stats=EvalStats(
            started_at="2026-06-26T00:00:00+00:00",
            completed_at="2026-06-26T00:00:01+00:00",
            duration_s=1.0,
            total_steps=12,
        ),
        samples=[SceneResult(scene_id="s0", status="success", reduced={"success_at_end": 1.0})],
    )


def test_eval_log_round_trips_through_dict() -> None:
    log = _golden_log()
    restored = EvalLog.from_dict(log.to_dict())
    assert restored.to_dict() == log.to_dict()
    assert restored.results.metrics["success_at_end"] == 1.0


def test_golden_log_reads_back(tmp_path: Path) -> None:
    # A log written today must remain readable: persist, then read.
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(_golden_log().to_dict()), encoding="utf-8")
    restored = read_eval_log(str(path))
    assert restored.version == SCHEMA_VERSION
    assert restored.eval.git_commit == "deadbeef"
    assert restored.samples[0].scene_id == "s0"


def test_unsupported_schema_version_rejected() -> None:
    data = _golden_log().to_dict()
    data["version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        EvalLog.from_dict(data)


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    from robolens import eval

    task = Task(
        name="demo",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=60,
    )
    eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.tmp"))  # atomic temp+rename left nothing behind


def test_store_frames_writes_side_cars(tmp_path: Path) -> None:
    from robolens import eval

    task = Task(
        name="demo",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=60,
    )
    logs = eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), store_frames=True
    )
    assert logs[0].stats.frames_dir is not None
    assert list((tmp_path / "frames").glob("*.npy"))


def test_eval_set_runs_multiple_tasks(tmp_path: Path) -> None:
    def task(name: str) -> Task:
        return Task(
            name=name,
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            scorer=success_at_end(),
            max_steps=60,
        )

    success, logs = eval_set(
        [task("a"), task("b")],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    assert success is True
    assert len(logs) == 2
    assert {log.eval.task for log in logs} == {"a", "b"}
