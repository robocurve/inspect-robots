"""RerunSink: graceful no-op when rerun-sdk is absent; real logging when present."""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.logging import RerunSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.registry import registered
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Action, Observation, StepResult

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


class _RawImage:
    """Fake rr.Image archetype without a compress method (old SDK surface)."""

    def __init__(self, img: object) -> None:
        self.img = img


class _CompressibleImage(_RawImage):
    """Fake rr.Image archetype whose compress returns a marker value."""

    def compress(self, *, jpeg_quality: int) -> tuple[str, int]:
        """Return a marker so tests can assert compression was applied."""
        return ("Compressed", jpeg_quality)


class _ExplodingImage(_RawImage):
    """Fake rr.Image archetype whose compress always fails."""

    def compress(self, *, jpeg_quality: int) -> tuple[str, int]:
        """Raise to exercise the raw-image fallback."""
        raise ValueError("cannot encode")


def _install_fake_rerun(
    monkeypatch: pytest.MonkeyPatch,
    *,
    image_cls: type[_RawImage] = _CompressibleImage,
    gate: threading.Event | None = None,
    log_error: Exception | None = None,
) -> list[tuple[str, object]]:
    """Install a fake ``rerun`` module (new-API surface); return the (path, value) log."""
    logged: list[tuple[str, object]] = []
    fake = types.ModuleType("rerun")

    def _log(path: str, value: object = None, **_kwargs: object) -> None:
        if gate is not None:
            gate.wait(timeout=30.0)
        if log_error is not None:
            raise log_error
        logged.append((path, value))

    fake.init = lambda *a, **k: None  # type: ignore[attr-defined]
    fake.save = lambda p: None  # type: ignore[attr-defined]
    fake.set_time = lambda *a, **k: None  # type: ignore[attr-defined]
    fake.log = _log  # type: ignore[attr-defined]
    fake.Image = image_cls  # type: ignore[attr-defined]
    fake.Scalars = lambda v: ("Scalars", v)  # type: ignore[attr-defined]
    fake.TextLog = lambda t: ("TextLog", t)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rerun", fake)
    return logged


def _obs(*, with_image: bool = True) -> Observation:
    images = {"cam": np.zeros((4, 4, 3), dtype=np.uint8)} if with_image else {}
    return Observation(images=images, state={"q": np.array([1.0])})


def _step_result() -> StepResult:
    return StepResult(observation=Observation(), reward=1.0)


def _log_one(sink: RerunSink, t: int = 0, *, with_image: bool = True) -> None:
    sink.log_step(t, _obs(with_image=with_image), Action(data=np.array([0.5])), _step_result())


def test_images_jpeg_compressed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_CompressibleImage)
    sink = RerunSink()
    _log_one(sink)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert camera == [("Compressed", 75)]


def test_jpeg_quality_none_logs_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_CompressibleImage)
    sink = RerunSink(jpeg_quality=None)
    _log_one(sink)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert len(camera) == 1 and isinstance(camera[0], _CompressibleImage)


def test_old_sdk_without_compress_logs_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_RawImage)
    sink = RerunSink()
    _log_one(sink)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert len(camera) == 1 and isinstance(camera[0], _RawImage)


def test_compress_failure_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_ExplodingImage)
    sink = RerunSink()
    _log_one(sink)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert len(camera) == 1 and isinstance(camera[0], _ExplodingImage)
