"""Strict RFC 8259 JSON logs and rollout non-finite gates, end to end.

The written eval log must be parseable by any conforming JSON parser: no
``Infinity``/``NaN`` literals (non-finite floats become ``null``). A non-finite
action introduced by an approver halts the eval as ``SafetyAbort``, and the log
still reaches disk.
"""

from __future__ import annotations

import errno
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
        "np_nan32": np.float32(np.nan),
        "np_inf32": np.float32(np.inf),
        "np_ninf32": np.float32(-np.inf),
        "np_nan16": np.float16(np.nan),
        "np_fine32": np.float32(2.5),
        "fine": 1.5,
        "int": 3,
        "flag": True,
        "nested": [float("inf"), {"d": float("nan")}, (2.0, float("-inf"), np.float32(np.nan))],
    }
    clean = _sanitize(dirty)
    assert clean == {
        "inf": None,
        "ninf": None,
        "nan": None,
        "np_nan32": None,
        "np_inf32": None,
        "np_ninf32": None,
        "np_nan16": None,
        "np_fine32": 2.5,
        "fine": 1.5,
        "int": 3,
        "flag": True,
        "nested": [None, {"d": None}, [2.0, None, None]],
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


@pytest.mark.parametrize("unsupported_links", [False, True])
def test_json_sink_never_overwrites_a_colliding_log_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported_links: bool
) -> None:
    """A reused random suffix cannot replace an earlier immutable attempt log."""
    from types import SimpleNamespace

    from inspect_robots.logging.json_log import JsonLogSink

    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    monkeypatch.setattr(
        "inspect_robots.logging.json_log.uuid.uuid4",
        lambda: SimpleNamespace(hex="a" * 32),
    )
    sink = JsonLogSink(str(tmp_path))
    sink.on_eval_end(log)
    assert sink.path is not None
    saved_path = sink.path
    saved_path.write_text("previous immutable log")

    if unsupported_links:

        def no_links(source: object, destination: object) -> None:
            del source, destination
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

        monkeypatch.setattr("inspect_robots.logging.json_log.os.link", no_links)
    with pytest.raises(FileExistsError):
        sink.on_eval_end(log)
    assert saved_path.read_text() == "previous immutable log"
    assert vars(sink)["path"] is None
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("error_code", [errno.EOPNOTSUPP, errno.ENOSYS, errno.EPERM, errno.EINVAL])
def test_eval_publishes_json_when_hard_links_are_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_code: int
) -> None:
    """A completed evaluation still publishes a readable log on portable filesystems."""

    def no_links(source: object, destination: object) -> None:
        del source, destination
        raise OSError(error_code, "hard links unsupported")

    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", no_links)
    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    (path,) = tmp_path.glob("*.json")
    assert read_eval_log(str(path)).to_dict() == log.to_dict()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lock"))


def test_portable_json_publication_excludes_a_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers selecting one name cannot replace each other's completed log."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    from inspect_robots.logging.json_log import JsonLogSink

    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    monkeypatch.setattr(
        "inspect_robots.logging.json_log.uuid.uuid4",
        lambda: SimpleNamespace(hex="c" * 32),
    )
    entered = Event()
    release = Event()

    def no_links(source: object, destination: object) -> None:
        del source, destination
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", no_links)
    first = JsonLogSink(str(tmp_path))
    second = JsonLogSink(str(tmp_path))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(first.on_eval_end, log)
        try:
            assert entered.wait(5)
            with pytest.raises(FileExistsError):
                second.on_eval_end(replace(log, status="error", error="second writer"))
        finally:
            release.set()
        pending.result()

    assert first.path is not None
    assert read_eval_log(str(first.path)).status == "success"
    assert second.path is None
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lock"))


@pytest.mark.parametrize("failure_stage", ["link", "replace"])
def test_json_publication_disk_error_cleans_owned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Real publication failures propagate without exposing partial JSON or stale claims."""
    from inspect_robots.logging.json_log import JsonLogSink

    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path / "seed"))

    def failed_link(source: object, destination: object) -> None:
        del source, destination
        code = errno.EIO if failure_stage == "link" else errno.EOPNOTSUPP
        raise OSError(code, "link failure")

    def failed_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError(errno.EIO, "replacement failure")

    monkeypatch.setattr("inspect_robots.logging.json_log.os.link", failed_link)
    monkeypatch.setattr("inspect_robots.logging.json_log.os.replace", failed_replace)
    log_dir = tmp_path / "failure"
    sink = JsonLogSink(str(log_dir))
    with pytest.raises(OSError) as error:
        sink.on_eval_end(log)
    assert error.value.errno == errno.EIO
    assert sink.path is None
    assert list(log_dir.iterdir()) == []


def test_json_sink_preserves_another_writers_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A temp-name collision cannot unlink a different writer's pending log."""
    from types import SimpleNamespace

    from inspect_robots.logging.json_log import JsonLogSink

    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    monkeypatch.setattr(
        "inspect_robots.logging.json_log.uuid.uuid4",
        lambda: SimpleNamespace(hex="b" * 32),
    )
    pending = tmp_path / f"strict-json_{'b' * 32}.json.tmp"
    pending.write_text("another writer's pending log")
    sink = JsonLogSink(str(tmp_path))
    sink.on_eval_end(log)
    assert pending.read_text() == "another writer's pending log"
    assert sink.path is not None and sink.path.exists()


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


def test_numpy_float_non_finite_writes_as_null(tmp_path: Path) -> None:
    def hook(record: TrialRecord, scene: Scene) -> None:
        record.metadata["np_nan32"] = np.float32(np.nan)
        record.metadata["np_inf32"] = np.float32(np.inf)
        record.metadata["np_fine32"] = np.float32(3.14)

    eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        before_scoring=hook,
    )
    (path,) = tmp_path.glob("*.json")
    data = _read_strict(path)
    samples = data["samples"]
    assert isinstance(samples, list)
    meta = samples[0]["trial_metadata"][0]
    assert meta["np_nan32"] is None
    assert meta["np_inf32"] is None
    assert meta["np_fine32"] == pytest.approx(3.14, rel=1e-3)
