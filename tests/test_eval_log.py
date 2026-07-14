"""EvalLog schema: round-trip, golden read-back guarantee, atomicity; eval_set."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from inspect_robots import eval_set, read_eval_log
from inspect_robots.log import (
    SCHEMA_VERSION,
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
)
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task


def _golden_log() -> EvalLog:
    return EvalLog(
        version=SCHEMA_VERSION,
        status="success",
        eval=EvalSpec(
            task="demo",
            policy="scripted",
            embodiment="cubepick",
            created="2026-06-26T00:00:00+00:00",
            inspect_robots_version="0.0.0",
            git_commit="deadbeef",
            seed=0,
            max_steps=1200,
        ),
        results=EvalResults(total_scenes=1, total_trials=1, metrics={"success_at_end": 1.0}),
        stats=EvalStats(
            started_at="2026-06-26T00:00:00+00:00",
            completed_at="2026-06-26T00:00:01+00:00",
            duration_s=1.0,
            total_steps=12,
        ),
        samples=(
            SceneResult(
                scene_id="s0",
                status="success",
                reduced={"success_at_end": 1.0},
                epochs=({"success_at_end": 1.0},),
                instruction="reach the cube",
                operator_judgements=("yes",),
                termination_reasons=("success",),
            ),
        ),
    )


def test_eval_log_round_trips_through_dict() -> None:
    log = _golden_log()
    restored = EvalLog.from_dict(log.to_dict())
    assert restored.to_dict() == log.to_dict()
    assert restored.results.metrics["success_at_end"] == 1.0


def test_results_without_errored_trials_reads_with_default() -> None:
    """Logs written before EvalResults.errored_trials existed must still read."""
    data = _golden_log().to_dict()
    data["results"].pop("errored_trials", None)
    log = EvalLog.from_dict(data)
    assert log.results.errored_trials == 0


def test_golden_log_reads_back(tmp_path: Path) -> None:
    # A log written today must remain readable: persist, then read.
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(_golden_log().to_dict()), encoding="utf-8")
    restored = read_eval_log(str(path))
    assert restored.version == SCHEMA_VERSION
    assert restored.eval.git_commit == "deadbeef"
    assert restored.samples[0].scene_id == "s0"
    assert restored.samples[0].instruction == "reach the cube"
    assert restored.samples[0].operator_judgements == ("yes",)
    assert restored.samples[0].termination_reasons == ("success",)
    assert restored.eval.max_steps == 1200


def test_v1_log_without_additive_fields_reads_back(tmp_path: Path) -> None:
    # Older schema-v1 logs missing additive fields must remain readable.
    data = _golden_log().to_dict()
    del data["eval"]["max_steps"]
    for sample in data["samples"]:
        del sample["instruction"]
        del sample["operator_judgements"]
        del sample["termination_reasons"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    restored = read_eval_log(str(path))
    assert restored.samples[0].reduced == {"success_at_end": 1.0}
    assert restored.samples[0].instruction is None
    assert restored.samples[0].operator_judgements == ()
    assert restored.samples[0].termination_reasons == ()
    assert restored.eval.max_steps is None


def test_eval_log_and_friends_are_frozen() -> None:
    # An EvalLog is documented as immutable; each dataclass in it must actually
    # refuse attribute reassignment, and the sequence fields (previously plain
    # lists) must refuse in-place mutation too.
    log = _golden_log()
    with pytest.raises(FrozenInstanceError):
        log.status = "error"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        log.eval.seed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        log.results.total_trials = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        log.stats.total_steps = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        log.samples[0].status = "error"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        log.samples.clear()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        log.samples[0].operator_judgements.append("no")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        log.samples[0].termination_reasons.append("failure")  # type: ignore[attr-defined]


def test_unsupported_schema_version_rejected() -> None:
    data = _golden_log().to_dict()
    data["version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        EvalLog.from_dict(data)


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    from inspect_robots import eval

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
    from inspect_robots import eval

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
    assert list((tmp_path / "frames").rglob("*.npy"))


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


def test_store_frames_runs_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """Each eval gets its own frames subdir; a second run must not clobber the first."""
    from inspect_robots import eval

    task = Task(
        name="demo",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=5,
    )
    dirs = set()
    for _ in range(2):
        logs = eval(
            task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), store_frames=True
        )
        assert logs[0].stats.frames_dir is not None
        dirs.add(logs[0].stats.frames_dir)
    assert len(dirs) == 2  # distinct per-run directories
    for d in dirs:
        assert list(Path(d).glob("*.npy"))  # both runs' frames still on disk
