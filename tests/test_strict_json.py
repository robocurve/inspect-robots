"""Strict RFC 8259 JSON logs and rollout non-finite gates, end to end.

The written eval log must be parseable by any conforming JSON parser: no
``Infinity``/``NaN`` literals (non-finite floats become ``null``). A non-finite
action introduced by an approver halts the eval as ``SafetyAbort``, and the log
still reaches disk.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval, read_eval_log
from inspect_robots.logging.json_log import _sanitize
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import min_distance_to_goal, success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Action, StepResult


def _task(scorer: object = None) -> Task:
    return Task(
        name="strict-json",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=scorer or success_at_end(),  # type: ignore[arg-type]
        max_steps=40,
    )


def _forbid_constants(name: str) -> float:
    raise AssertionError(f"non-RFC-8259 constant in log: {name}")


def _read_strict(path: Path) -> dict[str, object]:
    """Parse with a ``parse_constant`` that rejects Infinity/NaN literals."""
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_forbid_constants)
    assert isinstance(data, dict)
    return data


class _NoDistanceEmbodiment(CubePickEmbodiment):
    """Reports no distance signal, so min_distance_to_goal scores inf."""

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        return replace(result, info={"success": result.info.get("success", False)})


class _NaNApprover:
    """Replace a finite policy action with a NaN action."""

    def review(self, action: Action, store: dict[str, object]) -> Action:
        del store
        return replace(action, data=np.full(2, np.nan))


def test_sanitize_maps_non_finite_floats_to_none() -> None:
    dirty = {
        "inf": float("inf"),
        "ninf": float("-inf"),
        "nan": float("nan"),
        "fine": 1.5,
        "int": 3,
        "flag": True,
        "nested": [float("inf"), {"d": float("nan")}, (2.0, float("-inf"))],
    }
    clean = _sanitize(dirty)
    assert clean == {
        "inf": None,
        "ninf": None,
        "nan": None,
        "fine": 1.5,
        "int": 3,
        "flag": True,
        "nested": [None, {"d": None}, [2.0, None]],
    }


def test_inf_metric_written_as_null(tmp_path: Path) -> None:
    task = _task(scorer=min_distance_to_goal())
    (log,) = eval(task, ScriptedPolicy(), _NoDistanceEmbodiment(), log_dir=str(tmp_path))
    assert log.results.metrics["min_distance_to_goal"] == float("inf")  # in-memory sentinel

    (path,) = tmp_path.glob("*.json")
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    data = _read_strict(path)  # a strict parser accepts the whole file
    results = data["results"]
    assert isinstance(results, dict)
    metrics = results["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["min_distance_to_goal"] is None  # inf → null at the JSON boundary


def test_nan_action_halts_as_safety_abort_and_log_reaches_disk(tmp_path: Path) -> None:
    embodiment = CubePickEmbodiment()
    approver = _NaNApprover()
    (log,) = eval(_task(), ScriptedPolicy(), embodiment, approver=approver, log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.error is not None and "non-finite" in log.error
    assert "_NaNApprover" in log.error

    (path,) = tmp_path.glob("*.json")
    restored = read_eval_log(str(path))
    assert restored.status == "error"
    _read_strict(path)  # strict parseable even for a halted run


def test_long_task_name_still_writes_its_log(tmp_path: Path) -> None:
    # The filename is derived from the task name, so an unbounded name pushed the
    # path past the 255-byte limit and on_eval_end raised OSError *after* every
    # trial had run, leaving log_dir empty and the run unrecoverable (#292).
    name = "a" * 300
    task = Task(
        name=name,
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=3,
    )
    (log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))

    (path,) = tmp_path.glob("*.json")
    assert len(path.name.encode()) <= 255
    # The full name is preserved in the log body; only the filename is capped.
    assert log.eval.task == name
    assert read_eval_log(str(path)).eval.task == name


def test_json_dump_backstop_rejects_unsanitized_non_finite(tmp_path: Path) -> None:
    # The allow_nan=False regression backstop: if a non-finite value ever
    # slipped past _sanitize, the write would fail loudly.
    with (
        pytest.raises(ValueError),
        (tmp_path / "x.json").open("w", encoding="utf-8") as fh,
    ):
        json.dump({"bad": float("inf")}, fh, allow_nan=False)


def test_scene_instruction_and_judgements_serialize_strict(tmp_path: Path) -> None:
    # The new SceneResult fields reach disk as strict JSON: the instruction
    # verbatim, and one judgement slot per epoch (None when nobody judged).
    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "success"
    (path,) = tmp_path.glob("*.json")
    data = _read_strict(path)
    samples = data["samples"]
    assert isinstance(samples, list)
    sample = samples[0]
    assert isinstance(sample, dict)
    assert sample["instruction"] == "reach"
    assert sample["operator_judgements"] == [None]
    assert sample["operator_notes"] == [None]


def test_populated_operator_judgement_and_note_serialize_strict(tmp_path: Path) -> None:
    # A captured judgement and grader note both survive the strict JSON sink.
    def judge(record: TrialRecord, scene: Scene) -> None:
        record.operator_judgement = "y"
        record.operator_note = "Gripper Closed Early"

    (log,) = eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        before_scoring=judge,
    )
    assert log.status == "success"
    (path,) = tmp_path.glob("*.json")
    data = _read_strict(path)
    samples = data["samples"]
    assert isinstance(samples, list)
    sample = samples[0]
    assert isinstance(sample, dict)
    assert sample["instruction"] == "reach"
    assert sample["operator_judgements"] == ["y"]
    assert sample["operator_notes"] == ["Gripper Closed Early"]


def test_policy_transcript_non_finite_floats_write_as_null(tmp_path: Path) -> None:
    class _NonFiniteTranscriptPolicy(ScriptedPolicy):
        def transcript(self) -> object:
            return {"inf": float("inf"), "nan": float("nan")}

    eval(_task(), _NonFiniteTranscriptPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    (path,) = tmp_path.glob("*.json")
    data = _read_strict(path)
    samples = data["samples"]
    assert isinstance(samples, list)
    transcript = samples[0]["policy_transcripts"][0]
    assert transcript == {"inf": None, "nan": None}
