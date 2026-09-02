"""EvalLog schema: round-trip, golden read-back guarantee, atomicity; eval_set."""

from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from inspect_robots import eval_set, read_eval_log
from inspect_robots._html import render_html
from inspect_robots._summarize import TrialTranscript, build_digest
from inspect_robots.errors import EmbodimentFault, SafetyAbort
from inspect_robots.log import (
    SCHEMA_VERSION,
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
    _json_safe_scene_metadata,
)
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Epochs, Task


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
            grader="vlm",
            grader_config={
                "model": "judge-model",
                "base_url": "https://example.invalid/v1",
                "rubric": "Grade success if the cube moved.",
                "max_cameras": 4,
                "effort": None,
            },
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
                scene_metadata={
                    "rubric": "touch the cube",
                    "taskgen": {"model": "vision"},
                },
                operator_judgements=("yes",),
                judgement_sources=("prompt",),
                operator_notes=("gripper closed early",),
                operator_messages=(({"t": 3, "text": "keep left <now>"},),),
                trial_metadata=({"foo": "bar"},),
                termination_reasons=("success",),
                policy_transcripts=(
                    [
                        {"role": "user", "content": "reach the cube"},
                        {"role": "assistant", "content": "moving"},
                    ],
                ),
            ),
        ),
    )


def test_json_safe_scene_metadata_filters_and_deep_copies() -> None:
    """Retain JSON-safe values while detaching nested data from its source."""
    nested = {"thresholds": [1, 2]}
    metadata = {
        "rubric": "touch the cube",
        "nested": nested,
        "adapter_object": object(),
    }

    safe = _json_safe_scene_metadata(metadata)

    assert safe == {
        "rubric": "touch the cube",
        "nested": {"thresholds": [1, 2]},
    }
    nested["thresholds"].append(3)
    assert safe["nested"] == {"thresholds": [1, 2]}


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
    assert restored.samples[0].scene_metadata == {
        "rubric": "touch the cube",
        "taskgen": {"model": "vision"},
    }
    assert restored.samples[0].operator_judgements == ("yes",)
    assert restored.samples[0].judgement_sources == ("prompt",)
    assert restored.eval.grader == "vlm"
    assert restored.eval.grader_config["model"] == "judge-model"
    assert restored.eval.grader_config["rubric"] == "Grade success if the cube moved."
    assert restored.eval.grader_config["max_cameras"] == 4
    # ``None`` (no reasoning_effort sent) must survive as null, not vanish.
    assert restored.eval.grader_config["effort"] is None
    assert "effort" in restored.eval.grader_config
    assert restored.samples[0].operator_notes == ("gripper closed early",)
    assert restored.samples[0].operator_messages == (({"t": 3, "text": "keep left <now>"},),)
    assert isinstance(restored.samples[0].operator_messages, tuple)
    assert isinstance(restored.samples[0].operator_messages[0], tuple)
    assert restored.samples[0].trial_metadata == ({"foo": "bar"},)
    assert restored.samples[0].termination_reasons == ("success",)
    assert restored.samples[0].policy_transcripts == (
        [
            {"role": "user", "content": "reach the cube"},
            {"role": "assistant", "content": "moving"},
        ],
    )
    assert restored.eval.max_steps == 1200


def test_operator_messages_without_source_remain_readable(tmp_path: Path) -> None:
    path = tmp_path / "old-operator-message.json"
    path.write_text(json.dumps(_golden_log().to_dict()), encoding="utf-8")

    restored = read_eval_log(str(path))

    assert restored.samples[0].operator_messages == (({"t": 3, "text": "keep left <now>"},),)


def test_v1_log_without_additive_fields_reads_back(tmp_path: Path) -> None:
    # Older schema-v1 logs missing additive fields must remain readable.
    data = _golden_log().to_dict()
    del data["eval"]["max_steps"]
    del data["eval"]["max_seconds"]
    del data["eval"]["grader"]
    del data["eval"]["grader_config"]
    for sample in data["samples"]:
        del sample["instruction"]
        del sample["scene_metadata"]
        del sample["operator_judgements"]
        del sample["judgement_sources"]
        del sample["operator_notes"]
        del sample["operator_messages"]
        del sample["trial_metadata"]
        del sample["termination_reasons"]
        del sample["policy_transcripts"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    restored = read_eval_log(str(path))
    assert restored.samples[0].reduced == {"success_at_end": 1.0}
    assert restored.samples[0].instruction is None
    assert restored.samples[0].scene_metadata == {}
    assert restored.samples[0].operator_judgements == ()
    assert restored.samples[0].judgement_sources == ()
    assert restored.samples[0].operator_notes == ()
    assert restored.samples[0].operator_messages == ()
    assert restored.samples[0].trial_metadata == ()
    assert restored.samples[0].termination_reasons == ()
    assert restored.samples[0].policy_transcripts == ()
    assert restored.eval.max_steps is None
    assert restored.eval.max_seconds is None
    assert restored.eval.grader is None
    assert restored.eval.grader_config == {}


def test_seconds_horizon_round_trips_declared_and_resolved_values() -> None:
    spec = EvalSpec(
        task="timed",
        policy="scripted",
        embodiment="cubepick",
        created="2026-07-21T00:00:00+00:00",
        inspect_robots_version="0.0.0",
        embodiment_info={"control_hz": 15.0},
        max_steps=1800,
        max_seconds=120.0,
    )
    restored = EvalLog.from_dict(
        EvalLog(
            version=SCHEMA_VERSION,
            status="success",
            eval=spec,
            results=EvalResults(total_scenes=0, total_trials=0),
            stats=EvalStats(started_at="a", completed_at="b", duration_s=0.0, total_steps=0),
        ).to_dict()
    )
    assert restored.eval.max_seconds == 120.0
    assert restored.eval.max_steps == 1800


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
        log.samples[0].operator_messages[0].append({})  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        log.samples[0].trial_metadata.append({})  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        log.samples[0].termination_reasons.append("failure")  # type: ignore[attr-defined]


def test_unsupported_schema_version_rejected() -> None:
    data = _golden_log().to_dict()
    data["version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        EvalLog.from_dict(data)


def test_operator_feedback_appears_in_digest_and_escaped_html() -> None:
    log = _golden_log()
    transcripts = [TrialTranscript("s0", 0, None, "none")]

    digest = build_digest(log, transcripts)
    document = render_html(log, title="operator feedback")

    assert "operator feedback: keep left <now>" in digest
    assert document.index("Operator feedback") > document.index("Trial 0 transcript")
    assert 'class="feedback-chip"' not in document
    assert "keep left &lt;now&gt;" in document
    assert "keep left <now>" not in document


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


def test_eval_persists_only_json_safe_scene_metadata(tmp_path: Path) -> None:
    from inspect_robots import eval

    circular: list[object] = []
    circular.append(circular)
    task = Task(
        name="metadata",
        scenes=[
            Scene(
                id="s0",
                instruction="reach",
                init_seed=0,
                metadata={
                    "rubric": "touch the cube",
                    "nested": {"thresholds": [1, 2]},
                    "adapter_object": object(),
                    "circular": circular,
                },
            )
        ],
        scorer=success_at_end(),
        max_steps=60,
    )

    log = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))[0]

    assert log.status == "success"
    assert log.samples[0].scene_metadata == {
        "rubric": "touch the cube",
        "nested": {"thresholds": [1, 2]},
    }
    # The persisted copy is deep: mutating the live scene metadata after the
    # fact (as an adapter might mid-run) must not reach into the log.
    cast(dict[str, Any], task.scenes[0].metadata["nested"])["thresholds"].append(object())
    assert log.samples[0].scene_metadata["nested"] == {"thresholds": [1, 2]}
    written = read_eval_log(str(next(tmp_path.glob("*.json"))))
    assert written.samples[0].scene_metadata == log.samples[0].scene_metadata


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


def test_eval_set_reports_failed_task_and_keeps_completed_logs(tmp_path: Path) -> None:
    """Return earlier logs plus an in-memory error row when a later task fails."""

    def task(name: str, *, reducer: str = "mean") -> Task:
        return Task(
            name=name,
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            scorer=success_at_end(),
            max_steps=60,
            epochs=Epochs(count=1, reducer=reducer),
        )

    good = task("good")
    bad = task("bad", reducer="bogus")
    success, logs = eval_set(
        [good, bad],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    assert success is False
    assert len(logs) == 2
    assert logs[0].status == "success"
    assert logs[0].eval.task == good.name
    assert logs[1].status == "error"
    assert logs[1].error is not None and "bogus" in logs[1].error
    assert logs[1].eval.task == bad.name
    assert logs[1].eval.policy == "scripted"
    assert logs[1].eval.embodiment == "cubepick"
    assert logs[1].eval.max_steps == bad.max_steps
    assert logs[1].samples == ()
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_eval_set_error_log_preserves_string_component_names(tmp_path: Path) -> None:
    """Keep registry strings in a synthetic spec when task resolution fails."""
    success, logs = eval_set(
        "missing-task",
        "scripted",
        "cubepick",
        log_dir=str(tmp_path),
        seed=17,
    )

    assert success is False
    assert len(logs) == 1
    assert logs[0].status == "error"
    assert logs[0].eval.task == "missing-task"
    assert logs[0].eval.policy == "scripted"
    assert logs[0].eval.embodiment == "cubepick"
    assert logs[0].eval.seed == 17
    assert logs[0].eval.max_steps is None
    assert logs[0].eval.max_seconds is None
    assert not list(tmp_path.glob("*.json"))


def test_eval_set_error_log_falls_back_when_embodiment_info_raises(
    tmp_path: Path,
) -> None:
    """A broken adapter descriptor must not mask the task's error log."""

    class _RaisingInfoEmbodiment:
        @property
        def info(self) -> Any:
            raise RuntimeError("broken embodiment info")

    task = Task(
        name="broken-info",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=60,
    )
    embodiment = _RaisingInfoEmbodiment()

    success, logs = eval_set(
        task,
        ScriptedPolicy(),
        cast(Any, embodiment),
        log_dir=str(tmp_path),
    )

    assert success is False
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "error"
    assert log.eval.embodiment == type(embodiment).__name__
    assert log.error is not None and "RuntimeError" in log.error


@pytest.mark.parametrize(
    "error",
    [SafetyAbort("unsafe"), EmbodimentFault("faulted"), KeyboardInterrupt("stopped")],
)
def test_eval_set_propagates_halts_and_interrupts(
    error: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never contain safety halts, embodiment faults, or keyboard interrupts."""

    def raising_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args, kwargs
        raise error

    monkeypatch.setattr(sys.modules["inspect_robots.eval"], "eval", raising_eval)

    with pytest.raises(type(error), match=str(error)):
        eval_set(
            "cubepick-reach",
            "scripted",
            "cubepick",
            log_dir=str(tmp_path),
        )


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
