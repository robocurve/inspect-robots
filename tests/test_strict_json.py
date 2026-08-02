"""Strict RFC 8259 JSON logs and the ClampApprover NaN gate, end to end.

The written eval log must be parseable by any conforming JSON parser: no
``Infinity``/``NaN`` literals (non-finite floats become ``null``). A NaN action
caught by the ClampApprover halts the eval as ``SafetyAbort`` — and the log
still reaches disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots import eval, read_eval_log
from inspect_robots.approver import ClampApprover
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats
from inspect_robots.logging.json_log import JsonLogSink, _sanitize
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import min_distance_to_goal, success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation, StepResult


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


def _sink_log() -> EvalLog:
    """Build the smallest immutable log needed for direct sink tests."""
    return EvalLog(
        version=1,
        status="success",
        eval=EvalSpec(
            task="strict json",
            policy="p",
            embodiment="e",
            created="now",
            inspect_robots_version="0",
        ),
        results=EvalResults(total_scenes=0, total_trials=0, metrics={}),
        stats=EvalStats(
            started_at="start",
            completed_at="end",
            duration_s=0,
            total_steps=0,
        ),
        samples=(),
    )


class _NoDistanceEmbodiment(CubePickEmbodiment):
    """Reports no distance signal, so min_distance_to_goal scores inf."""

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        return replace(result, info={"success": result.info.get("success", False)})


class _NaNPolicy(ScriptedPolicy):
    """Emits a NaN action on the first inference."""

    def act(self, observation: Observation) -> ActionChunk:
        chunk = super().act(observation)
        return ActionChunk(actions=[Action(data=np.full(2, np.nan)), *chunk.actions])


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
    approver = ClampApprover(embodiment.info.action_space)
    (log,) = eval(_task(), _NaNPolicy(), embodiment, approver=approver, log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.error is not None and "NaN" in log.error

    (path,) = tmp_path.glob("*.json")
    restored = read_eval_log(str(path))
    assert restored.status == "error"
    _read_strict(path)  # strict parseable even for a halted run


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


def test_json_sink_redraws_filename_and_tmp_name_after_link_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed final name triggers a fresh stem and per-attempt temp UUID."""
    values = iter(("a" * 64, "1" * 64, "b" * 64, "2" * 64))
    monkeypatch.setattr(
        "inspect_robots.logging.json_log.uuid.uuid4",
        lambda: SimpleNamespace(hex=next(values)),
    )
    actual_link = os.link
    links: list[tuple[Path, Path]] = []

    def collide_once(source: Path, target: Path) -> None:
        links.append((source, target))
        if len(links) == 1:
            raise FileExistsError
        actual_link(source, target)

    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", collide_once)
    sink = JsonLogSink(str(tmp_path))

    sink.on_eval_end(_sink_log())

    assert sink.path == tmp_path / "strict-json_bbbbbbbb.json"
    assert sink.path.is_file()
    assert links[0][0].name == f".strict-json_aaaaaaaa.{'1' * 64}.tmp"
    assert links[1][0].name == f".strict-json_bbbbbbbb.{'2' * 64}.tmp"
    assert not list(tmp_path.glob(".*.tmp"))


def test_json_sink_uses_full_uuid_after_sixteen_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sixteen exhausted short claims fall back to one full UUID stem."""
    counter = 0

    def next_uuid() -> SimpleNamespace:
        nonlocal counter
        counter += 1
        return SimpleNamespace(hex=f"{counter:064x}")

    actual_link = os.link
    calls = 0

    def collide_sixteen(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls <= 16:
            raise FileExistsError
        actual_link(source, target)

    monkeypatch.setattr("inspect_robots.logging.json_log.uuid.uuid4", next_uuid)
    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", collide_sixteen)
    sink = JsonLogSink(str(tmp_path))

    sink.on_eval_end(_sink_log())

    assert sink.path is not None
    stem_uuid = sink.path.stem.removeprefix("strict-json_")
    assert len(stem_uuid) == 64
    assert calls == 17


def test_json_sink_propagates_the_impossible_full_uuid_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a full-UUID collision never overwrites an already claimed log."""

    def always_collide(_source: Path, _target: Path) -> None:
        raise FileExistsError

    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", always_collide)
    sink = JsonLogSink(str(tmp_path))

    with pytest.raises(FileExistsError):
        sink.on_eval_end(_sink_log())

    assert sink.path is None
    assert not list(tmp_path.glob(".*.tmp"))


def test_json_sink_falls_back_to_replace_when_hard_links_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-collision hard-link failure retains the atomic replace behavior."""
    replaced: list[tuple[Path, Path]] = []
    actual_replace = os.replace

    def no_links(_source: Path, _target: Path) -> None:
        raise OSError("hard links unavailable")

    def record_replace(source: Path, target: Path) -> None:
        replaced.append((source, target))
        actual_replace(source, target)

    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", no_links)
    monkeypatch.setattr("inspect_robots.logging.json_log.os.replace", record_replace)
    sink = JsonLogSink(str(tmp_path))

    sink.on_eval_end(_sink_log())

    assert sink.path is not None and sink.path.is_file()
    assert replaced == [(replaced[0][0], sink.path)]
    assert not replaced[0][0].exists()
