"""RerunSink: graceful no-op when rerun-sdk is absent; real logging when present."""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.logging import RerunSink
from inspect_robots.logging.rerun_sink import (
    _render_message,
    _StepPayload,
    _TranscriptPayload,
    _WorkerState,
)
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


def test_spawn_and_connect_are_mutually_exclusive() -> None:
    """A sink cannot spawn locally and connect to a remote viewer."""
    with pytest.raises(ValueError, match="spawn and connect_url are mutually exclusive"):
        RerunSink(spawn=True, connect_url="rerun+http://127.0.0.1:9876/proxy")


def test_spawn_and_recording_are_mutually_exclusive() -> None:
    """A sink cannot spawn a local viewer and record to .rrd at once."""
    with pytest.raises(ValueError, match="spawn and recording_path are mutually exclusive"):
        RerunSink("run.rrd", spawn=True)


def test_recording_and_connect_are_mutually_exclusive() -> None:
    """A sink cannot record to .rrd and stream to a remote viewer at once."""
    with pytest.raises(ValueError, match="recording_path and connect_url are mutually exclusive"):
        RerunSink("run.rrd", connect_url="rerun+http://127.0.0.1:9876/proxy")


class _StartupRR:
    """Capture startup calls made through the fake Rerun SDK surface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def init(self, application_id: str, **kwargs: object) -> None:
        """Capture initialization arguments."""
        self.calls.append(("init", (application_id, kwargs)))

    def spawn(self, **kwargs: object) -> None:
        """Capture viewer-spawn arguments."""
        self.calls.append(("spawn", kwargs))

    def connect_grpc(self, url: str) -> None:
        """Capture the remote viewer URL."""
        self.calls.append(("connect_grpc", url))


def test_spawn_uses_bounded_memory_limit_after_plain_init() -> None:
    """Local startup initializes without spawn= and applies the 2 GiB viewer cap."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True)
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("spawn", {"memory_limit": "2GiB"}),
    ]


def test_custom_spawn_memory_limit_is_forwarded_verbatim() -> None:
    """A caller-provided viewer memory limit reaches rr.spawn unchanged."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True, spawn_memory_limit="4GiB")
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls[-1] == ("spawn", {"memory_limit": "4GiB"})


def test_default_startup_never_spawns_a_viewer() -> None:
    """The default non-spawn mode only initializes the recording."""
    fake = _StartupRR()
    sink = RerunSink()
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls == [("init", ("inspect_robots", {}))]


def test_connect_grpc_follows_init_without_spawning() -> None:
    """Remote startup initializes, connects to the URL, and never spawns locally."""

    url = "rerun+http://127.0.0.1:9876/proxy"
    fake = _StartupRR()
    sink = RerunSink(connect_url=url)
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("connect_grpc", url),
    ]


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


def test_init_failure_disables_sink_instead_of_crashing() -> None:
    """A recording initialization failure must warn instead of killing the eval."""
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


def test_spawn_failure_disables_sink_instead_of_crashing() -> None:
    """A missing viewer binary reported by rr.spawn must leave the sink dormant."""

    class _FakeRR:
        def init(self, *args: object, **kwargs: object) -> None:
            return None

        def spawn(self, **kwargs: object) -> None:
            raise RuntimeError("Failed to find Rerun Viewer executable in PATH.")

    sink = RerunSink(spawn=True)
    sink._rr = _FakeRR()
    with pytest.warns(RuntimeWarning, match="RerunSink disabled"):
        sink.on_eval_start(None)  # type: ignore[arg-type]
    assert sink.available is False


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

    def _set_time(timeline: str, **kwargs: object) -> None:
        logged.append(("set_time", (timeline, kwargs)))

    fake.set_time = _set_time  # type: ignore[attr-defined]
    fake.log = _log  # type: ignore[attr-defined]
    fake.Image = image_cls  # type: ignore[attr-defined]
    fake.Scalars = lambda v: ("Scalars", v)  # type: ignore[attr-defined]
    fake.TextLog = lambda t, *, level=None: ("TextLog", t, level)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rerun", fake)
    return logged


class _BlockingRecording:
    """Fake recording whose flush remains blocked until the test releases it."""

    def __init__(self, gate: threading.Event, finished: threading.Event) -> None:
        self._gate = gate
        self._finished = finished

    def flush(self) -> None:
        """Wait long enough to exceed the sink's bounded probe timeout."""
        self._gate.wait(timeout=30.0)
        self._finished.set()


class _HealthyRecording:
    """Fake recording with an immediately completed flush."""

    def flush(self) -> None:
        """Return immediately to represent a healthy SDK connection."""


class _ExplodingRecording:
    """Fake recording whose available flush surface raises an exception."""

    def flush(self) -> None:
        """Raise so the probe can treat the completed call as healthy."""
        raise RuntimeError("flush failed")


class _RecordingWithoutFlush:
    """Represent the rerun-sdk 0.20/0.21 recording surface."""


def _probe_fake(
    recording: object | None,
    *,
    unregister_calls: list[None] | None = None,
    unregister_error: Exception | None = None,
) -> types.ModuleType:
    """Build a fake SDK exposing the recording probe and optional atexit shim."""
    fake = types.ModuleType("rerun")
    fake.init = lambda *a, **k: None  # type: ignore[attr-defined]
    fake.get_global_data_recording = lambda: recording  # type: ignore[attr-defined]
    if unregister_calls is not None:

        def _unregister_shutdown() -> None:
            unregister_calls.append(None)
            if unregister_error is not None:
                raise unregister_error

        fake.unregister_shutdown = _unregister_shutdown  # type: ignore[attr-defined]
    return fake


def _obs(*, with_image: bool = True) -> Observation:
    images = {"cam": np.zeros((4, 4, 3), dtype=np.uint8)} if with_image else {}
    return Observation(images=images, state={"q": np.array([1.0])})


def _step_result() -> StepResult:
    return StepResult(observation=Observation(), reward=1.0)


def _log_one(sink: RerunSink, t: int = 0, *, with_image: bool = True) -> None:
    sink.log_step(t, _obs(with_image=with_image), Action(data=np.array([0.5])), _step_result())


def _step_payload(t: int, *, with_image: bool = True) -> _StepPayload:
    """Build a queue payload without starting a worker."""
    images = {"cam": np.zeros((1, 1, 3), dtype=np.uint8)} if with_image else {}
    return _StepPayload(
        prefix="trial/scene/e0",
        t=t,
        images=images,
        state={},
        action=np.array([0.0]),
        reward=None,
        terminated=False,
        termination_reason=None,
    )


def _arm_completed_worker(sink: RerunSink) -> None:
    """Give shutdown a completed generation so it reports existing counters."""
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    sink._worker = worker
    sink._state = _WorkerState(stop=threading.Event())


def test_policy_messages_emit_ordered_levels_on_the_step_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    sink = RerunSink()
    sink.on_trial_start("scene", 2)

    sink.log_policy_messages(
        7,
        [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
            {"role": "tool", "content": "result"},
            {"role": "system", "content": "prompt"},
            {"role": "critic", "content": "other"},
        ],
    )

    assert sink.flush(timeout=5.0)
    assert logged == [
        ("set_time", ("step", {"sequence": 7})),
        ("trial/scene/e2/llm", ("TextLog", "assistant: answer", "INFO")),
        ("trial/scene/e2/llm", ("TextLog", "user: question", "INFO")),
        ("trial/scene/e2/llm", ("TextLog", "tool: result", "DEBUG")),
        ("trial/scene/e2/llm", ("TextLog", "system: prompt", "TRACE")),
        ("trial/scene/e2/llm", ("TextLog", "critic: other", "TRACE")),
    ]
    sink.on_eval_end(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"role": "assistant", "content": "hello"}, ("INFO", "assistant: hello")),
        (
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "camera 'top':"},
                    {"type": "image_url", "image_url": {"url": "elided"}},
                    {"type": "text", "text": "after"},
                    7,
                ],
            },
            ("INFO", "user: camera 'top':\n[image_url part]\nafter\n[unknown part]"),
        ),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_move",
                        "type": "function",
                        "function": {"name": "move_by", "arguments": '{"dx": 0.1}'},
                    }
                ],
            },
            ("INFO", 'assistant: tool_call move_by({"dx": 0.1})'),
        ),
        ("plain row", ("INFO", "plain row")),
        ({"content": "orphan"}, ("TRACE", "unknown: orphan")),
        ({"role": "tool", "content": "result"}, ("DEBUG", "tool: result")),
        ({"role": "system", "content": "prompt"}, ("TRACE", "system: prompt")),
        ({"role": "assistant", "content": ""}, ("INFO", "assistant: ")),
        (
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [{"function": "malformed"}, "malformed"],
            },
            ("INFO", "assistant: tool_call ()\ntool_call ()"),
        ),
    ],
)
def test_policy_message_rendering_table(message: Any, expected: tuple[str, str]) -> None:
    assert _render_message(message) == expected


def test_mixed_queue_eviction_counts_transcripts_steps_and_images() -> None:
    sink = RerunSink(queue_size=2)
    sink._image_watermark = 99
    sink._enqueue(_TranscriptPayload("trial/scene/e0", 0, (("INFO", "first"),)))
    sink._enqueue(_step_payload(1))

    sink._enqueue(_TranscriptPayload("trial/scene/e0", 2, (("INFO", "second"),)))
    assert sink._dropped_transcripts == 1
    assert sink._dropped_steps == 0
    assert sink._dropped_frames == 0

    sink._enqueue(_step_payload(3, with_image=False))
    assert sink._dropped_transcripts == 1
    assert sink._dropped_steps == 1
    assert sink._dropped_frames == 1

    with sink._cond:
        sink._queue.clear()
    _arm_completed_worker(sink)
    with pytest.warns(RuntimeWarning, match=r"1 transcript update\(s\)"):
        sink.on_eval_end(None)  # type: ignore[arg-type]


def test_transcript_only_drops_warn_and_reset_all_counters() -> None:
    sink = RerunSink(queue_size=1)
    sink._enqueue(_TranscriptPayload("trial/scene/e0", 0, (("INFO", "first"),)))
    sink._enqueue(_TranscriptPayload("trial/scene/e0", 1, (("INFO", "second"),)))
    with sink._cond:
        sink._queue.clear()
    _arm_completed_worker(sink)

    with pytest.warns(RuntimeWarning, match=r"1 transcript update\(s\)"):
        sink.on_eval_end(None)  # type: ignore[arg-type]

    assert sink._dropped_frames == 0
    assert sink._dropped_steps == 0
    assert sink._dropped_transcripts == 0


def test_policy_messages_as_first_trial_call_spawn_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    sink = RerunSink()

    sink.log_policy_messages(0, [{"role": "assistant", "content": "first"}])

    assert sink._worker is not None
    assert sink.flush(timeout=5.0)
    assert ("trial/llm", ("TextLog", "assistant: first", "INFO")) in logged
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_wedged_shutdown_accounts_for_mixed_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    _install_fake_rerun(monkeypatch, gate=gate)
    sink = RerunSink(flush_timeout=0.05)
    worker: threading.Thread | None = None
    try:
        _log_one(sink, 0)
        _wait_for_inflight(sink)
        worker = sink._worker
        assert worker is not None
        with sink._cond:
            sink._queue.append(_TranscriptPayload("trial/scene/e0", 1, (("INFO", "queued"),)))
            sink._queue.append(_step_payload(2))
        with pytest.warns(RuntimeWarning) as caught:
            sink.on_eval_end(None)  # type: ignore[arg-type]
        messages = [str(item.message) for item in caught]
        assert any("1 transcript update(s)" in message for message in messages)
        assert any("1 camera frame(s) and 1 full step(s)" in message for message in messages)
    finally:
        gate.set()
    assert worker is not None
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_disabled_sink_silently_ignores_policy_messages() -> None:
    sink = RerunSink()
    sink._disabled = True

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sink.log_policy_messages(0, [{"role": "assistant", "content": "unused"}])

    assert sink._worker is None
    assert not sink._queue


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
    assert not hasattr(sys.modules["rerun"], "get_global_data_recording")
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


def test_wedged_recording_flush_unregisters_atexit_and_disables_sink() -> None:
    """A timed-out SDK flush abandons queued data and prevents unsafe re-init."""
    gate = threading.Event()
    finished = threading.Event()
    unregister_calls: list[None] = []
    fake = _probe_fake(_BlockingRecording(gate, finished), unregister_calls=unregister_calls)
    init_calls: list[None] = []

    def _init(*args: object, **kwargs: object) -> None:
        init_calls.append(None)

    fake.init = _init  # type: ignore[attr-defined]
    sink = RerunSink(flush_timeout=0.05)
    sink._rr = fake
    try:
        with pytest.warns(RuntimeWarning, match="viewer connection is stalled"):
            sink.on_eval_end(None)  # type: ignore[arg-type]
        assert unregister_calls == [None]
        assert sink._disabled
        sink.on_eval_start(None)  # type: ignore[arg-type]
        assert sink._rr is fake
        assert init_calls == []
    finally:
        gate.set()
    assert finished.wait(timeout=5.0)


def test_disabled_sink_skips_probe_on_later_shutdowns() -> None:
    """A wedge-disabled sink stays dormant: no repeated stall, warning, or unregister."""
    gate = threading.Event()
    finished = threading.Event()
    unregister_calls: list[None] = []
    sink = RerunSink(flush_timeout=0.05)
    sink._rr = _probe_fake(_BlockingRecording(gate, finished), unregister_calls=unregister_calls)
    try:
        with pytest.warns(RuntimeWarning, match="viewer connection is stalled"):
            sink.on_eval_end(None)  # type: ignore[arg-type]
        assert unregister_calls == [None]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sink.on_eval_end(None)  # type: ignore[arg-type]
        assert unregister_calls == [None]
    finally:
        gate.set()
    assert finished.wait(timeout=5.0)


def test_healthy_recording_flush_keeps_atexit_and_sink_enabled() -> None:
    """A completed SDK flush leaves the normal Rerun atexit cleanup registered."""
    unregister_calls: list[None] = []
    sink = RerunSink()
    sink._rr = _probe_fake(_HealthyRecording(), unregister_calls=unregister_calls)

    sink.on_eval_end(None)  # type: ignore[arg-type]

    assert unregister_calls == []
    assert not sink._disabled


def test_shutdown_probe_shims_leave_inconclusive_sinks_enabled() -> None:
    """Missing recording APIs keep the older SDK's existing atexit posture unchanged."""
    uninitialized = RerunSink()
    uninitialized.on_eval_end(None)  # type: ignore[arg-type]
    assert not uninitialized._disabled

    no_get_recording = types.ModuleType("rerun")
    no_recording = _probe_fake(None)
    unregister_calls: list[None] = []
    no_flush = _probe_fake(_RecordingWithoutFlush(), unregister_calls=unregister_calls)

    for fake in (no_get_recording, no_recording, no_flush):
        sink = RerunSink()
        sink._rr = fake
        sink.on_eval_end(None)  # type: ignore[arg-type]
        assert not sink._disabled

    assert unregister_calls == []


def test_wedged_recording_without_unregister_shutdown_still_disables() -> None:
    """A missing unregister shim cannot prevent the wedged sink from going dormant."""
    gate = threading.Event()
    finished = threading.Event()
    sink = RerunSink(flush_timeout=0.05)
    sink._rr = _probe_fake(_BlockingRecording(gate, finished))
    try:
        with pytest.warns(RuntimeWarning, match="viewer connection is stalled"):
            sink.on_eval_end(None)  # type: ignore[arg-type]
        assert sink._disabled
    finally:
        gate.set()
    assert finished.wait(timeout=5.0)


def test_unregister_shutdown_failure_is_swallowed() -> None:
    """A failing atexit-unregister shim does not escape the shutdown path."""
    gate = threading.Event()
    finished = threading.Event()
    unregister_calls: list[None] = []
    sink = RerunSink(flush_timeout=0.05)
    sink._rr = _probe_fake(
        _BlockingRecording(gate, finished),
        unregister_calls=unregister_calls,
        unregister_error=RuntimeError("cannot unregister"),
    )
    try:
        with pytest.warns(RuntimeWarning, match="viewer connection is stalled"):
            sink.on_eval_end(None)  # type: ignore[arg-type]
        assert unregister_calls == [None]
        assert sink._disabled
    finally:
        gate.set()
    assert finished.wait(timeout=5.0)


def test_recording_flush_exception_counts_as_completed_probe() -> None:
    """An exception from an available flush is swallowed and treated as non-wedged."""
    unregister_calls: list[None] = []
    sink = RerunSink()
    sink._rr = _probe_fake(_ExplodingRecording(), unregister_calls=unregister_calls)

    sink.on_eval_end(None)  # type: ignore[arg-type]

    assert unregister_calls == []
    assert not sink._disabled


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


def test_real_rerun_process_exits_when_tcp_peer_never_reads() -> None:
    """The real SDK atexit path is bounded after a connected peer stops reading."""
    rr = pytest.importorskip("rerun")
    if not hasattr(rr, "connect_grpc"):
        pytest.skip("pre-gRPC rerun-sdk cannot run the connect-mode wedge scenario")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    accepted = threading.Event()
    release_listener = threading.Event()

    def _accept_without_reading() -> None:
        try:
            connection, _ = server.accept()
        except OSError:
            return
        with connection:
            accepted.set()
            release_listener.wait(timeout=45.0)

    listener = threading.Thread(target=_accept_without_reading, daemon=True)
    listener.start()
    script = textwrap.dedent(
        """
        import sys

        import numpy as np

        from inspect_robots.logging import RerunSink
        from inspect_robots.types import Action, Observation, StepResult

        sink = RerunSink(
            connect_url=sys.argv[1],
            jpeg_quality=None,
            flush_timeout=0.2,
        )
        sink.on_eval_start(None)
        image = np.random.default_rng(0).integers(
            0, 256, size=(4096, 4096, 3), dtype=np.uint8
        )
        observation = Observation(images={"camera": image})
        result = StepResult(observation=Observation(), reward=1.0)
        sink.log_step(0, observation, Action(data=np.array([0.5])), result)
        assert sink.flush(timeout=10.0)
        sink.on_eval_end(None)
        """
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                f"rerun+http://127.0.0.1:{port}/proxy",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    finally:
        release_listener.set()
        server.close()
        listener.join(timeout=5.0)

    assert accepted.is_set(), completed.stderr
    assert completed.returncode == 0, completed.stderr
    # Guard against the trivial pass: the wedge branch must actually fire.
    assert "viewer connection is stalled" in completed.stderr, completed.stderr
