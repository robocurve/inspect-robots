"""RerunSink: graceful no-op when rerun-sdk is absent; real logging when present."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
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
    assert sink.flush(timeout=5.0)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert camera == [("Compressed", 75)]
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_jpeg_quality_none_logs_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_CompressibleImage)
    sink = RerunSink(jpeg_quality=None)
    _log_one(sink)
    assert sink.flush(timeout=5.0)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert len(camera) == 1 and isinstance(camera[0], _CompressibleImage)
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_old_sdk_without_compress_warns_once_and_logs_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_RawImage)
    sink = RerunSink()
    with pytest.warns(RuntimeWarning, match="could not JPEG-compress") as record:
        _log_one(sink, 0)
        _log_one(sink, 1)
        assert sink.flush(timeout=5.0)
    # Warned once for the whole sink, not once per frame.
    assert len([w for w in record if "JPEG-compress" in str(w.message)]) == 1
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert len(camera) == 2 and all(isinstance(c, _RawImage) for c in camera)
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_compress_failure_warns_and_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch, image_cls=_ExplodingImage)
    sink = RerunSink()
    with pytest.warns(RuntimeWarning, match="could not JPEG-compress"):
        _log_one(sink)
        assert sink.flush(timeout=5.0)
    camera = [v for p, v in logged if p == "trial/camera/cam"]
    assert len(camera) == 1 and isinstance(camera[0], _ExplodingImage)
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_log_step_never_blocks_when_viewer_stalls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The producer side is bounded: overflow is dropped, never waited on."""
    gate = threading.Event()
    _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink(queue_size=4)
    try:
        _log_one(sink, 0)
        # Pin payload 0 in-flight so payload 1 enqueues below the image
        # watermark and keeps its images; a later eviction then
        # deterministically hits an image-bearing payload.
        _wait_for_inflight(sink)
        for t in range(1, 20):
            _log_one(sink, t)
        with sink._cond:
            assert len(sink._queue) <= 4
        assert sink._dropped_steps > 0
        assert sink._dropped_frames > 0
    finally:
        gate.set()
    assert sink.flush(timeout=5.0)


def test_scalars_survive_frame_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under pressure images are stripped but every step's scalars still arrive."""
    gate = threading.Event()
    logged = _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink(queue_size=8)  # image watermark = 2
    try:
        for t in range(6):
            _log_one(sink, t)
    finally:
        gate.set()
    assert sink.flush(timeout=5.0)
    state_paths = [p for p, _ in logged if p == "trial/state/q/0"]
    camera_paths = [p for p, _ in logged if p == "trial/camera/cam"]
    assert len(state_paths) == 6  # no whole-step drops at queue_size=8
    assert len(camera_paths) == 6 - sink._dropped_frames
    # Worker pop timing makes the exact count race between 3 and 4.
    assert 3 <= sink._dropped_frames <= 4


def test_flush_times_out_while_stalled_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink()
    try:
        _log_one(sink)
        assert sink.flush(timeout=0.05) is False
    finally:
        gate.set()
    assert sink.flush(timeout=5.0)
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_eval_end_shuts_down_worker_and_log_step_restarts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    sink = RerunSink()
    _log_one(sink, 0)
    sink.on_eval_end(None)  # type: ignore[arg-type]
    assert sink._worker is None
    _log_one(sink, 1)  # restarts the worker
    assert sink.flush(timeout=5.0)
    assert len([p for p, _ in logged if p == "trial/camera/cam"]) == 2
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_emit_failure_warns_once_and_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rerun(monkeypatch, log_error=ValueError("boom"))
    sink = RerunSink()
    with pytest.warns(RuntimeWarning, match="failed to emit") as record:
        for t in range(3):
            _log_one(sink, t)
        assert sink.flush(timeout=5.0)
    assert len([w for w in record if "failed to emit" in str(w.message)]) == 1
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_queue_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="queue_size"):
        RerunSink(queue_size=0)


def test_trial_end_flushes_queued_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trial boundaries drain the queue so an eval abort loses at most one trial's tail."""
    logged = _install_fake_rerun(monkeypatch)
    sink = RerunSink()
    _log_one(sink, 0)
    sink.on_trial_end(None)  # type: ignore[arg-type]
    assert [p for p, _ in logged if p == "trial/camera/cam"] == ["trial/camera/cam"]
    sink.on_eval_end(None)  # type: ignore[arg-type]


def _wait_for_inflight(sink: RerunSink) -> None:
    """Spin until the worker has popped a payload and is inside the (gated) SDK call."""
    state = sink._state
    assert state is not None
    for _ in range(500):
        with sink._cond:
            if state.inflight:
                return
        time.sleep(0.01)
    pytest.fail("worker never picked up the payload")


def test_wedged_worker_is_disowned_and_backlog_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker stuck in the SDK is abandoned; a restarted worker owns the queue alone."""
    gate = threading.Event()
    logged = _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink(flush_timeout=0.05)
    try:
        _log_one(sink, 0)
        _wait_for_inflight(sink)  # worker A is now wedged inside rr.log on payload 0
        _log_one(sink, 1)  # payload 1 queued behind the wedge
        worker_a = sink._worker
        assert worker_a is not None
        with pytest.warns(RuntimeWarning) as record:
            sink.on_eval_end(None)  # type: ignore[arg-type]  # flush+join time out; A disowned
        messages = [str(w.message) for w in record]
        assert any("stalled" in m for m in messages)
        assert any("dropped 1 camera frame(s) and 1 full step(s)" in m for m in messages)
        assert sink._worker is None and sink._state is None
        assert sink._dropped_steps == 0  # reported and reset
        _log_one(sink, 2)  # starts worker B
    finally:
        gate.set()
    assert sink.flush(timeout=5.0)
    worker_a.join(timeout=5.0)
    assert not worker_a.is_alive()  # A exited; it never became a second consumer
    camera = [p for p, _ in logged if p == "trial/camera/cam"]
    assert len(camera) == 2  # A's in-flight payload 0, B's payload 2; payload 1 dropped
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_eval_end_reports_dropped_data(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink(queue_size=2)
    try:
        for t in range(6):
            _log_one(sink, t)
    finally:
        gate.set()
    with pytest.warns(RuntimeWarning, match="dropped"):
        sink.on_eval_end(None)  # type: ignore[arg-type]
    # Counters reset: a quiet follow-up eval must not re-report old drops.
    assert sink._dropped_frames == 0 and sink._dropped_steps == 0


def test_trial_end_skips_flush_once_stalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """After one timed-out flush, later trial boundaries stop re-paying the timeout."""
    gate = threading.Event()
    _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink(flush_timeout=0.05)
    try:
        _log_one(sink, 0)
        _wait_for_inflight(sink)  # worker wedged: the flush below must time out
        flush_timeouts: list[float | None] = []
        original_flush = sink.flush

        def _counting_flush(timeout: float | None = None) -> bool:
            flush_timeouts.append(timeout)
            return original_flush(timeout)

        monkeypatch.setattr(sink, "flush", _counting_flush)
        sink.on_trial_end(None)  # type: ignore[arg-type]  # times out, marks stalled
        state = sink._state
        assert state is not None and state.stalled
        sink.on_trial_end(None)  # type: ignore[arg-type]  # skipped: no second flush
        assert len(flush_timeouts) == 1
    finally:
        gate.set()
    assert sink.flush(timeout=5.0)
    sink.on_eval_end(None)  # type: ignore[arg-type]
