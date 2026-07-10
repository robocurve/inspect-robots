"""RerunSink: graceful no-op when rerun-sdk is absent; real logging when present."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from inspect_robots import eval
from inspect_robots.logging import RerunSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.registry import registered
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

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
def test_noop_and_warns_when_absent() -> None:
    sink = RerunSink()
    with pytest.warns(RuntimeWarning, match="rerun-sdk is not installed"):
        assert sink.available is False
    # Warned once per instance; a second check stays quiet.
    assert sink.available is False
    # ...but a fresh instance warns again (no hidden module-global state).
    with pytest.warns(RuntimeWarning, match="rerun-sdk is not installed"):
        assert RerunSink().available is False


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


def test_viewer_failure_disables_sink_instead_of_crashing() -> None:
    """A missing viewer binary (rr.init raising) must not kill the eval."""
    from inspect_robots.log import EvalSpec

    class _FakeRR:
        def init(self, *a: object, **k: object) -> None:
            raise RuntimeError("Failed to find Rerun Viewer executable in PATH.")

    sink = RerunSink(spawn=True)
    sink._rr = _FakeRR()
    with pytest.warns(RuntimeWarning, match="RerunSink disabled"):
        sink.on_eval_start(
            EvalSpec(
                task="t", policy="p", embodiment="e", created="now", inspect_robots_version="0"
            )
        )
    assert sink.available is False  # dormant from here on
    sink.log_step(0, None, None, None)  # type: ignore[arg-type]  # must not raise
