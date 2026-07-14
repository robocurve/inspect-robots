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


def test_spawn_and_connect_are_mutually_exclusive() -> None:
    """A sink cannot spawn locally and connect to a remote viewer."""
    with pytest.raises(ValueError, match="spawn and connect_url are mutually exclusive"):
        RerunSink(spawn=True, connect_url="rerun+http://127.0.0.1:9876/proxy")


def test_connect_grpc_is_called_only_when_configured() -> None:
    """Startup connects to the configured URL and skips gRPC when it is unset."""

    class _FakeRR:
        def __init__(self) -> None:
            self.init_count = 0
            self.connect_urls: list[str] = []

        def init(self, *args: object, **kwargs: object) -> None:
            self.init_count += 1

        def connect_grpc(self, url: str) -> None:
            self.connect_urls.append(url)

    url = "rerun+http://127.0.0.1:9876/proxy"
    fake = _FakeRR()
    connected = RerunSink(connect_url=url)
    connected._rr = fake
    connected.on_eval_start(None)  # type: ignore[arg-type]
    unconnected = RerunSink()
    unconnected._rr = fake
    unconnected.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.init_count == 2
    assert fake.connect_urls == [url]


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


def test_connection_failure_disables_sink_instead_of_crashing() -> None:
    """An unreachable gRPC viewer must warn and leave the sink dormant."""

    class _FakeRR:
        def init(self, *args: object, **kwargs: object) -> None:
            return None

        def connect_grpc(self, url: str) -> None:
            raise RuntimeError(f"could not connect to {url}")

    sink = RerunSink(connect_url="rerun+http://127.0.0.1:9876/proxy")
    sink._rr = _FakeRR()
    with pytest.warns(RuntimeWarning, match="RerunSink disabled"):
        sink.on_eval_start(None)  # type: ignore[arg-type]
    assert sink.available is False
