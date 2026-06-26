"""RerunSink: graceful no-op when rerun-sdk is absent; real logging when present."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import robolens.logging.rerun_sink as rerun_mod
from robolens import eval
from robolens.logging import RerunSink
from robolens.mock import CubePickEmbodiment, ScriptedPolicy
from robolens.registry import registered
from robolens.scene import Scene
from robolens.scorer import success_at_end
from robolens.task import Task

_RERUN_INSTALLED = importlib.util.find_spec("rerun") is not None


def _task() -> Task:
    return Task(
        name="demo",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=60,
    )


def test_rerun_sink_registered() -> None:
    assert "rerun" in registered("sink")


@pytest.mark.skipif(_RERUN_INSTALLED, reason="rerun installed; testing the absent path")
def test_noop_and_warns_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerun_mod, "_WARNED", False)
    sink = RerunSink()
    with pytest.warns(RuntimeWarning, match="rerun-sdk is not installed"):
        assert sink.available is False


@pytest.mark.skipif(_RERUN_INSTALLED, reason="rerun installed; testing the absent path")
def test_eval_runs_with_absent_rerun_sink(tmp_path: Path) -> None:
    # A full eval with only the (unavailable) RerunSink must still complete.
    logs = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), sinks=[RerunSink()])
    assert logs[0].status == "success"


@pytest.mark.skipif(not _RERUN_INSTALLED, reason="requires rerun-sdk")
def test_rerun_sink_writes_recording(tmp_path: Path) -> None:
    rrd = tmp_path / "run.rrd"
    sink = RerunSink(str(rrd))
    assert sink.available is True
    eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), sinks=[sink])
    assert rrd.exists()
