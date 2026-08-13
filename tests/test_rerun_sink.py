"""RerunSink: graceful no-op when rerun-sdk is absent; real logging when present."""

from __future__ import annotations

import importlib.util
import inspect
import re
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.log import EvalSpec
from inspect_robots.logging import RerunSink
from inspect_robots.logging import rerun_sink as rerun_sink_module
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
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
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


def test_recording_path_and_dir_are_mutually_exclusive() -> None:
    """A fixed recording file and per-eval recording directory cannot both be selected."""
    with pytest.raises(ValueError, match="recording_path and recording_dir are mutually exclusive"):
        RerunSink("run.rrd", recording_dir="recordings")


class _GrpcSinkTarget(NamedTuple):
    """Captured gRPC sink constructor arguments."""

    url: str


class _FileSinkTarget(NamedTuple):
    """Captured file sink constructor arguments."""

    path: Path


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

    def save(self, path: Path) -> None:
        """Capture the viewer-less recording path."""
        self.calls.append(("save", path))

    def GrpcSink(self, *, url: str) -> _GrpcSinkTarget:
        """Return a marker carrying the explicit live-view URL."""
        return _GrpcSinkTarget(url)

    def FileSink(self, path: Path) -> _FileSinkTarget:
        """Return a marker carrying the recording path."""
        return _FileSinkTarget(path)

    def set_sinks(self, *sinks: object) -> None:
        """Capture the ordered tee targets."""
        self.calls.append(("set_sinks", sinks))


class _OldStartupRR:
    """Capture the pre-0.24 startup surface without tee attributes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def init(self, application_id: str, **kwargs: object) -> None:
        """Capture initialization arguments."""
        self.calls.append(("init", (application_id, kwargs)))

    def spawn(self, **kwargs: object) -> None:
        """Capture the legacy local-view startup."""
        self.calls.append(("spawn", kwargs))

    def connect_grpc(self, url: str) -> None:
        """Capture the legacy remote-view startup."""
        self.calls.append(("connect_grpc", url))


def _eval_spec(task: str = "demo") -> EvalSpec:
    """Build the minimal eval identity needed by startup path derivation."""
    return EvalSpec(
        task=task,
        policy="p",
        embodiment="e",
        created="now",
        inspect_robots_version="0",
    )


def test_spawn_uses_bounded_memory_limit_after_plain_init() -> None:
    """Local startup initializes without spawn= and applies the 2 GiB viewer cap."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True)
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("spawn", {"memory_limit": "2GiB", "port": 9876}),
    ]


def test_custom_spawn_memory_limit_is_forwarded_verbatim() -> None:
    """A caller-provided viewer memory limit reaches rr.spawn unchanged."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True, spawn_memory_limit="4GiB")
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls[-1] == ("spawn", {"memory_limit": "4GiB", "port": 9876})


def test_custom_spawn_port_is_forwarded_verbatim() -> None:
    """A caller-provided viewer port reaches rr.spawn unchanged."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True, spawn_port=9877)
    sink._rr = fake

    sink.on_eval_start(None)  # type: ignore[arg-type]

    assert fake.calls[-1] == ("spawn", {"memory_limit": "2GiB", "port": 9877})


@pytest.mark.parametrize("spawn_port", [0, 65536])
def test_spawn_port_out_of_range_raises(spawn_port: int) -> None:
    """Viewer spawn ports outside the valid TCP range are rejected."""
    with pytest.raises(ValueError, match="spawn_port"):
        RerunSink(spawn=True, spawn_port=spawn_port)


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


def test_spawn_tee_uses_explicit_default_url_and_order(tmp_path: Path) -> None:
    """A local tee starts a disconnected viewer before attaching its two sinks."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True, recording_dir=str(tmp_path))
    sink._rr = fake

    sink.on_eval_start(_eval_spec("Demo Task"))

    path = sink.resolved_recording_path
    assert path is not None
    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("spawn", {"memory_limit": "2GiB", "port": 9876, "connect": False}),
        (
            "set_sinks",
            (
                _GrpcSinkTarget("rerun+http://127.0.0.1:9876/proxy"),
                _FileSinkTarget(path),
            ),
        ),
    ]


def test_spawn_tee_uses_custom_port_in_explicit_url(tmp_path: Path) -> None:
    """The tee URL follows a custom spawn port instead of an SDK default."""
    fake = _StartupRR()
    sink = RerunSink(spawn=True, spawn_port=9988, recording_dir=str(tmp_path))
    sink._rr = fake

    sink.on_eval_start(_eval_spec())

    assert fake.calls[1] == (
        "spawn",
        {"memory_limit": "2GiB", "port": 9988, "connect": False},
    )
    set_sinks = fake.calls[2]
    assert set_sinks[0] == "set_sinks"
    targets = set_sinks[1]
    assert isinstance(targets, tuple)
    assert targets[0] == _GrpcSinkTarget("rerun+http://127.0.0.1:9988/proxy")


def test_connect_tee_sets_sinks_without_legacy_connect(tmp_path: Path) -> None:
    """A remote tee attaches the URL and file together without connect_grpc."""
    url = "rerun+http://viewer.example:9988/proxy"
    fake = _StartupRR()
    sink = RerunSink(connect_url=url, recording_dir=str(tmp_path))
    sink._rr = fake

    sink.on_eval_start(_eval_spec())

    path = sink.resolved_recording_path
    assert path is not None
    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("set_sinks", (_GrpcSinkTarget(url), _FileSinkTarget(path))),
    ]


def test_fixed_path_tee_attaches_the_configured_file(tmp_path: Path) -> None:
    """A fixed recording path combines with a local viewer through the same tee."""
    fake = _StartupRR()
    target = tmp_path / "run.rrd"
    sink = RerunSink(str(target), spawn=True)
    sink._rr = fake

    sink.on_eval_start(_eval_spec())

    assert sink.resolved_recording_path == target
    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("spawn", {"memory_limit": "2GiB", "port": 9876, "connect": False}),
        (
            "set_sinks",
            (
                _GrpcSinkTarget("rerun+http://127.0.0.1:9876/proxy"),
                _FileSinkTarget(target),
            ),
        ),
    ]


def test_recording_dir_saves_derived_fresh_path_per_eval(tmp_path: Path) -> None:
    """Directory mode creates its parent and draws a task-slugged path for each eval."""
    recording_dir = tmp_path / "nested" / "recordings"
    fake = _StartupRR()
    sink = RerunSink(recording_dir=str(recording_dir))
    sink._rr = fake

    sink.on_eval_start(_eval_spec("Demo Task"))
    first = sink.resolved_recording_path
    sink.on_eval_start(_eval_spec("Demo Task"))
    second = sink.resolved_recording_path

    assert recording_dir.is_dir()
    assert first is not None and second is not None and first != second
    assert first.parent == recording_dir
    assert second.parent == recording_dir
    assert re.fullmatch(r"demo-task_[0-9a-f]{8}\.rrd", first.name)
    assert re.fullmatch(r"demo-task_[0-9a-f]{8}\.rrd", second.name)
    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("save", first),
        ("init", ("inspect_robots", {})),
        ("save", second),
    ]


def test_fixed_recording_path_is_exposed_as_path(tmp_path: Path) -> None:
    """Fixed mode normalizes the configured string into the public Path contract."""
    fake = _StartupRR()
    target = tmp_path / "fixed.rrd"
    sink = RerunSink(str(target))
    sink._rr = fake

    sink.on_eval_start(_eval_spec())

    assert sink.resolved_recording_path == target
    assert fake.calls[-1] == ("save", target)


def test_no_recording_target_leaves_resolved_path_none() -> None:
    """Bare initialization never advertises a recording file."""
    fake = _StartupRR()
    sink = RerunSink()
    sink._rr = fake

    sink.on_eval_start(_eval_spec())

    assert sink.resolved_recording_path is None


def test_old_sdk_spawn_fallback_warns_once_and_clears_path(tmp_path: Path) -> None:
    """A pre-0.24 SDK keeps its local viewer but never advertises the skipped file."""
    fake = _OldStartupRR()
    sink = RerunSink(spawn=True, recording_dir=str(tmp_path))
    sink._rr = fake

    with pytest.warns(RuntimeWarning, match="predates set_sinks") as caught:
        sink.on_eval_start(_eval_spec())
        sink.on_eval_start(_eval_spec())

    assert len(caught) == 1
    assert sink.resolved_recording_path is None
    assert fake.calls == [
        ("init", ("inspect_robots", {})),
        ("spawn", {"memory_limit": "2GiB", "port": 9876}),
        ("init", ("inspect_robots", {})),
        ("spawn", {"memory_limit": "2GiB", "port": 9876}),
    ]


def test_old_sdk_connect_fallback_keeps_remote_view(tmp_path: Path) -> None:
    """A pre-0.24 SDK preserves the legacy remote connection while skipping the file."""
    url = "rerun+http://viewer.example:9876/proxy"
    fake = _OldStartupRR()
    sink = RerunSink(connect_url=url, recording_dir=str(tmp_path))
    sink._rr = fake

    with pytest.warns(RuntimeWarning, match="predates set_sinks"):
        sink.on_eval_start(_eval_spec())

    assert sink.resolved_recording_path is None
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
def test_recording_dir_stays_unresolved_when_sdk_is_absent(tmp_path: Path) -> None:
    """The SDK-less early return does not create or advertise a phantom file."""
    recording_dir = tmp_path / "recordings"
    sink = RerunSink(recording_dir=str(recording_dir))

    with pytest.warns(RuntimeWarning, match="rerun-sdk is not installed"):
        sink.on_eval_start(_eval_spec())

    assert sink.resolved_recording_path is None
    assert not recording_dir.exists()


@pytest.mark.skipif(_RERUN_INSTALLED, reason="rerun installed; testing the absent path")
def test_eval_runs_with_absent_rerun_sink(tmp_path: Path) -> None:
    # A full eval with only the (unavailable) RerunSink must still complete.
    logs = eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[RerunSink()],
    )
    assert logs[0].status == "success"


@pytest.mark.skipif(not _RERUN_INSTALLED, reason="requires rerun-sdk")
def test_rerun_sink_writes_recording(tmp_path: Path) -> None:
    rrd = tmp_path / "run.rrd"
    sink = RerunSink(str(rrd))
    assert sink.available is True
    eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        sinks=[sink],
        store_actions=False,
    )
    assert rrd.exists()


def test_init_failure_disables_sink_instead_of_crashing(tmp_path: Path) -> None:
    """A recording initialization failure must warn instead of killing the eval."""

    class _FakeRR:
        def init(self, *a: object, **k: object) -> None:
            raise RuntimeError("Failed to find Rerun Viewer executable in PATH.")

    sink = RerunSink(spawn=True, recording_dir=str(tmp_path))
    sink._rr = _FakeRR()
    with pytest.warns(RuntimeWarning, match="RerunSink disabled"):
        sink.on_eval_start(
            EvalSpec(
                task="t", policy="p", embodiment="e", created="now", inspect_robots_version="0"
            )
        )
    assert sink.available is False  # dormant from here on
    assert sink.resolved_recording_path is None
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


class _TextDocument(NamedTuple):
    text: str
    media_type: str


class _BlueprintNode(NamedTuple):
    kind: str
    args: tuple[object, ...]
    kwargs: dict[str, object]


class _BlueprintFactory:
    def __init__(self, recorder: _BlueprintRecorder, kind: str) -> None:
        self._recorder = recorder
        self._kind = kind

    def __call__(self, *args: object, **kwargs: object) -> _BlueprintNode:
        if self._recorder.fail_on == self._kind:
            raise ValueError(f"cannot build {self._kind}")
        node = _BlueprintNode(self._kind, args, kwargs)
        self._recorder.calls.append(node)
        return node


class _BlueprintRecorder:
    def __init__(self, *, send: bool = True, fail_on: str | None = None) -> None:
        self.send = send
        self.fail_on = fail_on
        self.calls: list[_BlueprintNode] = []
        self.sent: list[object] = []
        self.module = types.ModuleType("rerun.blueprint")
        for kind in (
            "TimeSeriesView",
            "Spatial2DView",
            "TextDocumentView",
            "TextLogView",
            "Vertical",
            "Tabs",
            "Horizontal",
            "Blueprint",
        ):
            setattr(self.module, kind, _BlueprintFactory(self, kind))

    def install(self, fake: types.ModuleType) -> None:
        fake.blueprint = self.module  # type: ignore[attr-defined]
        if self.send:
            fake.send_blueprint = self.sent.append  # type: ignore[attr-defined]


def _install_fake_rerun(
    monkeypatch: pytest.MonkeyPatch,
    *,
    image_cls: type[_RawImage] = _CompressibleImage,
    gate: threading.Event | None = None,
    log_error: Exception | None = None,
    text_document: bool = True,
    blueprint: _BlueprintRecorder | None = None,
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
    if text_document:
        fake.TextDocument = _TextDocument  # type: ignore[attr-defined]
    if blueprint is not None:
        blueprint.install(fake)
        monkeypatch.setitem(sys.modules, "rerun.blueprint", blueprint.module)
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


def _labeled_action_space() -> Box:
    return Box(
        shape=(4,),
        semantics=ActionSemantics(
            "joint_pos",
            dim_labels=("left_j0", "left_gripper", "right_j0", "right_gripper"),
        ),
    )


def _blueprint_step(sink: RerunSink, t: int = 0) -> None:
    sink.log_step(
        t,
        Observation(
            images={"top": np.zeros((4, 4, 3), dtype=np.uint8)},
            state={"joint_pos": np.arange(4, dtype=np.float64)},
        ),
        Action(data=np.arange(4, dtype=np.float64)),
        StepResult(observation=Observation(), reward=1.0),
    )


def _views(recorder: _BlueprintRecorder, kind: str) -> list[_BlueprintNode]:
    return [call for call in recorder.calls if call.kind == kind]


def _sent_tree(root: object) -> list[_BlueprintNode]:
    """Flatten every node reachable from one sent Blueprint, depth-first."""
    if not isinstance(root, _BlueprintNode):
        return []
    nodes = [root]
    for arg in root.args:
        nodes.extend(_sent_tree(arg))
    return nodes


def _node(value: object) -> _BlueprintNode:
    """Narrow a recorded child to a node so structure asserts can nest."""
    assert isinstance(value, _BlueprintNode)
    return value


def test_joint_groups_split_on_label_prefix() -> None:
    labels = tuple(
        f"{side}_{part}" for side in ("left", "right") for part in ("j0", "j1", "gripper")
    )
    assert rerun_sink_module._joint_groups(labels, 6) == [
        ("left", [0, 1, 2]),
        ("right", [3, 4, 5]),
    ]
    assert rerun_sink_module._joint_groups(None, 3) == [("joints", [0, 1, 2])]
    assert rerun_sink_module._joint_groups(("a_x", "a_y"), 2) == [("joints", [0, 1])]
    assert rerun_sink_module._joint_groups(("ax", "by"), 2) == [("joints", [0, 1])]
    assert rerun_sink_module._joint_groups(("l_a", "up", "r_b"), 3) == [("joints", [0, 1, 2])]
    assert rerun_sink_module._joint_groups(("l_a", "r_b"), 3) == [("joints", [0, 1, 2])]


def test_camera_order_ranks_left_top_right() -> None:
    """Ranked prefixes lead in spatial order; unranked names trail alphabetically."""
    assert rerun_sink_module._camera_order(("top_cam", "right_cam", "left_cam")) == [
        "left_cam",
        "top_cam",
        "right_cam",
    ]
    assert rerun_sink_module._camera_order(("top", "right", "left")) == ["left", "top", "right"]
    assert rerun_sink_module._camera_order(("zed", "wrist_cam", "top_cam")) == [
        "top_cam",
        "wrist_cam",
        "zed",
    ]
    assert rerun_sink_module._camera_order(("Top_cam", "left_cam")) == ["left_cam", "Top_cam"]
    assert rerun_sink_module._camera_order(()) == []


def test_blueprint_sent_per_trial_prefix_with_per_group_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(
        _labeled_action_space(),
        ObservationSpace(
            cameras=(CameraSpec(name="top", height=4, width=4),),
            state=StateSpec(fields=(StateField(key="joint_pos", shape=(4,)),)),
        ),
    )
    sink.on_eval_start(None)  # type: ignore[arg-type]
    sink.on_trial_start("s0", 0)

    _blueprint_step(sink)
    assert sink.flush(timeout=5.0)

    assert len(recorder.sent) == 1
    # The sent Blueprint must actually CONTAIN the views (a constructed view
    # that never gets wired into a column would pass constructor-level
    # assertions while the viewer shows nothing).
    reachable = _sent_tree(recorder.sent[0])
    reachable_names = {
        view.kwargs.get("name") for view in reachable if view.kind == "TimeSeriesView"
    }
    assert {"left", "right", "reward"} == reachable_names
    assert any(view.kind == "Spatial2DView" for view in reachable)
    assert any(view.kind == "TextLogView" for view in reachable)
    assert any(view.kind == "TextDocumentView" for view in reachable)
    # Two rows: cameras across the top, then the text tabs beside the plots.
    # The tab order matters: the latest-message document is the default tab.
    vertical = _node(_node(recorder.sent[0]).args[0])
    assert vertical.kind == "Vertical"
    camera_row, plot_row = (_node(child) for child in vertical.args)
    assert camera_row.kind == "Horizontal"
    assert [_node(child).kind for child in camera_row.args] == ["Spatial2DView"]
    assert plot_row.kind == "Horizontal"
    assert [_node(child).kind for child in plot_row.args] == [
        "Tabs",
        "TimeSeriesView",
        "TimeSeriesView",
    ]
    tabs = _node(plot_row.args[0])
    assert [_node(child).kind for child in tabs.args] == [
        "TextDocumentView",
        "TextLogView",
        "TimeSeriesView",
    ]
    assert _node(tabs.args[2]).kwargs.get("contents") == ["+ trial/s0/e0/reward"]
    time_series = _views(recorder, "TimeSeriesView")
    left = next(view for view in time_series if view.kwargs.get("name") == "left")
    right = next(view for view in time_series if view.kwargs.get("name") == "right")
    assert left.kwargs["contents"] == [
        "+ trial/s0/e0/action/0",
        "+ trial/s0/e0/action/1",
        "+ trial/s0/e0/state/joint_pos/0",
        "+ trial/s0/e0/state/joint_pos/1",
    ]
    assert right.kwargs["contents"] == [
        "+ trial/s0/e0/action/2",
        "+ trial/s0/e0/action/3",
        "+ trial/s0/e0/state/joint_pos/2",
        "+ trial/s0/e0/state/joint_pos/3",
    ]
    assert any(
        view.kwargs
        == {
            "name": "top",
            "origin": "/",
            "contents": ["+ trial/s0/e0/camera/top"],
        }
        for view in _views(recorder, "Spatial2DView")
    )
    assert any(
        view.kwargs.get("contents") == ["+ trial/s0/e0/llm", "+ trial/s0/e0/event/**"]
        for view in _views(recorder, "TextLogView")
    )
    assert any(
        view.kwargs.get("contents") == ["+ trial/s0/e0/llm/latest"]
        for view in _views(recorder, "TextDocumentView")
    )
    # The reward series lives behind the text tabs, not in a plot slot:
    # agent-policy runs never log rewards, so a permanently visible reward
    # panel would sit empty.
    assert [_node(child).kwargs.get("name") for child in plot_row.args[1:]] == ["left", "right"]

    sink.on_trial_start("s1", 0)
    _blueprint_step(sink, 1)
    assert sink.flush(timeout=5.0)
    assert len(recorder.sent) == 2
    assert any(
        view.kwargs.get("contents") == ["+ trial/s1/e0/llm/latest"]
        for view in _views(recorder, "TextDocumentView")
    )

    _blueprint_step(sink, 2)
    assert sink.flush(timeout=5.0)
    assert len(recorder.sent) == 2
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_blueprint_camera_row_is_ordered_left_top_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared camera order (YAM lists top first) does not drive the row."""
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(
        _labeled_action_space(),
        ObservationSpace(
            cameras=tuple(
                CameraSpec(name=name, height=4, width=4)
                for name in ("top_cam", "wrist_cam", "right_cam", "left_cam")
            )
        ),
    )
    sink.on_eval_start(None)  # type: ignore[arg-type]
    sink.on_trial_start("s0", 0)
    _blueprint_step(sink)
    assert sink.flush(timeout=5.0)

    camera_row = _node(_node(_node(recorder.sent[0]).args[0]).args[0])
    assert camera_row.kind == "Horizontal"
    assert [_node(view).kwargs.get("name") for view in camera_row.args] == [
        "left_cam",
        "top_cam",
        "right_cam",
        "wrist_cam",
    ]
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_transcript_first_payload_sends_blueprint(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(_labeled_action_space(), ObservationSpace())
    sink.on_eval_start(None)  # type: ignore[arg-type]
    sink.on_trial_start("s0", 0)

    sink.log_policy_messages(0, [{"role": "assistant", "content": "first"}])
    assert sink.flush(timeout=5.0)

    assert len(recorder.sent) == 1
    # A camera-less run sends the plot row as the root, not a Vertical with
    # an empty top row.
    row = _node(_node(recorder.sent[0]).args[0])
    assert row.kind == "Horizontal"
    assert _node(row.args[0]).kind == "Tabs"
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_blueprint_state_keys_without_statespec_get_own_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(
        Box(shape=(4,)),
        ObservationSpace(state_keys=frozenset({"eef_pos", "cube_pos"})),
    )
    sink.on_eval_start(None)  # type: ignore[arg-type]
    sink.on_trial_start("s0", 0)
    _blueprint_step(sink)
    assert sink.flush(timeout=5.0)

    views = _views(recorder, "TimeSeriesView")
    assert any(view.kwargs.get("contents") == ["+ trial/s0/e0/state/eef_pos/**"] for view in views)
    assert any(view.kwargs.get("contents") == ["+ trial/s0/e0/state/cube_pos/**"] for view in views)
    assert _views(recorder, "Spatial2DView") == []
    sink.on_eval_end(None)  # type: ignore[arg-type]

    rich_sink = RerunSink()
    rich_sink.bind_spaces(
        _labeled_action_space(),
        ObservationSpace(
            state=StateSpec(
                fields=(
                    StateField(key="joint_pos", shape=(4,)),
                    StateField(key="eef_pose", shape=(7,)),
                    StateField(key="matrix", shape=(2, 2)),
                )
            )
        ),
    )
    rich_sink.on_eval_start(None)  # type: ignore[arg-type]
    rich_sink.on_trial_start("s1", 0)
    _blueprint_step(rich_sink)
    assert rich_sink.flush(timeout=5.0)

    rich_views = _views(recorder, "TimeSeriesView")
    assert any(
        view.kwargs.get("contents") == ["+ trial/s1/e0/state/eef_pose/**"] for view in rich_views
    )
    assert any(
        view.kwargs.get("contents") == ["+ trial/s1/e0/state/matrix/**"] for view in rich_views
    )
    assert not any(
        view.kwargs.get("contents") == ["+ trial/s1/e0/state/joint_pos/**"] for view in rich_views
    )
    rich_sink.on_eval_end(None)  # type: ignore[arg-type]


def test_blueprint_skipped_without_bind_spaces_or_blueprint_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    unbound = RerunSink()
    unbound.on_eval_start(None)  # type: ignore[arg-type]
    _log_one(unbound)
    assert unbound.flush(timeout=5.0)
    assert recorder.sent == []
    unbound.on_eval_end(None)  # type: ignore[arg-type]

    _install_fake_rerun(monkeypatch)
    no_blueprint = RerunSink()
    no_blueprint.bind_spaces(_labeled_action_space(), ObservationSpace())
    no_blueprint.on_eval_start(None)  # type: ignore[arg-type]
    _log_one(no_blueprint)
    assert no_blueprint.flush(timeout=5.0)
    no_blueprint.on_eval_end(None)  # type: ignore[arg-type]

    no_send_recorder = _BlueprintRecorder(send=False)
    _install_fake_rerun(monkeypatch, blueprint=no_send_recorder)
    no_send = RerunSink()
    no_send.bind_spaces(_labeled_action_space(), ObservationSpace())
    no_send.on_eval_start(None)  # type: ignore[arg-type]
    _log_one(no_send)
    assert no_send.flush(timeout=5.0)
    assert no_send_recorder.sent == []
    no_send.on_eval_end(None)  # type: ignore[arg-type]


def test_blueprint_build_failure_warns_once_and_run_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _BlueprintRecorder(fail_on="TimeSeriesView")
    logged = _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(_labeled_action_space(), ObservationSpace())
    sink.on_eval_start(None)  # type: ignore[arg-type]

    with pytest.warns(RuntimeWarning, match="automatic layout") as caught:
        sink.on_trial_start("s0", 0)
        _blueprint_step(sink)
        assert sink.flush(timeout=5.0)
        sink.on_trial_start("s1", 0)
        _blueprint_step(sink, 1)
        assert sink.flush(timeout=5.0)

    assert len(caught) == 1
    assert len([path for path, _ in logged if path.endswith("/action/0")]) == 2
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_reused_sink_resends_blueprint_for_same_trial_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(_labeled_action_space(), ObservationSpace())

    for _ in range(2):
        sink.on_eval_start(None)  # type: ignore[arg-type]
        sink.on_trial_start("s0", 0)
        _blueprint_step(sink)
        assert sink.flush(timeout=5.0)
        sink.on_eval_end(None)  # type: ignore[arg-type]

    assert len(recorder.sent) == 2


def test_eval_drives_the_blueprint_through_bind_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: eval() wires bind_spaces into RerunSink and a layout is sent."""
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    eval(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        sinks=[RerunSink()],
        store_actions=False,
    )
    assert len(recorder.sent) >= 1
    reachable = _sent_tree(recorder.sent[0])
    # CubePick's dx/dy labels collapse to the single combined joints view.
    assert any(
        view.kind == "TimeSeriesView" and view.kwargs.get("name") == "joints" for view in reachable
    )


def test_blueprint_skipped_for_zero_dim_action_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0-dim action space sends no layout instead of empty-content views."""
    recorder = _BlueprintRecorder()
    _install_fake_rerun(monkeypatch, blueprint=recorder)
    sink = RerunSink()
    sink.bind_spaces(Box(shape=(0,)), ObservationSpace())
    sink.on_trial_start("s0", 0)
    _blueprint_step(sink)
    assert sink.flush(timeout=5.0)
    assert recorder.sent == []
    sink.on_eval_end(None)  # type: ignore[arg-type]


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
        (
            "trial/scene/e2/llm/latest",
            _TextDocument(
                "**[INFO]** assistant: answer",
                media_type="text/markdown",
            ),
        ),
    ]
    sink.on_eval_end(None)  # type: ignore[arg-type]


def test_transcript_emission_logs_rows_and_markdown_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    rr = sys.modules["rerun"]
    payload = _TranscriptPayload(
        "trial/scene/e0",
        3,
        (
            ("assistant", "INFO", "assistant: answer"),
            ("tool", "DEBUG", "tool: result"),
        ),
    )

    RerunSink()._emit_transcript(rr, payload)

    assert logged == [
        ("set_time", ("step", {"sequence": 3})),
        ("trial/scene/e0/llm", ("TextLog", "assistant: answer", "INFO")),
        ("trial/scene/e0/llm", ("TextLog", "tool: result", "DEBUG")),
        (
            "trial/scene/e0/llm/latest",
            _TextDocument(
                "**[INFO]** assistant: answer",
                media_type="text/markdown",
            ),
        ),
    ]


def test_transcript_document_composes_multiline_entries_with_hard_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    rr = sys.modules["rerun"]
    message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "camera 'top':"},
            {"type": "image_url", "image_url": {"url": "elided"}},
            {"type": "text", "text": "after"},
        ],
    }
    payload = _TranscriptPayload(
        "trial/scene/e0",
        4,
        (_render_message(message), ("tool", "DEBUG", "tool: result")),
    )

    RerunSink()._emit_transcript(rr, payload)

    documents = [value for path, value in logged if path == "trial/scene/e0/llm/latest"]
    assert documents == [
        _TextDocument(
            "**[INFO]** assistant: camera 'top':  \n[image_url part]  \nafter",
            media_type="text/markdown",
        )
    ]


def test_transcript_payload_emits_exactly_one_document_for_multiple_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    rr = sys.modules["rerun"]
    payload = _TranscriptPayload(
        "trial/scene/e0",
        5,
        (
            ("assistant", "INFO", "assistant: first"),
            ("tool", "DEBUG", "tool: second"),
            ("assistant", "INFO", "assistant: third"),
        ),
    )

    RerunSink()._emit_transcript(rr, payload)

    documents = [value for path, value in logged if path == "trial/scene/e0/llm/latest"]
    assert documents == [
        _TextDocument(
            "**[INFO]** assistant: first\n\n---\n\n**[INFO]** assistant: third",
            media_type="text/markdown",
        )
    ]


def test_tool_only_transcript_delta_emits_row_without_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-only delta preserves its row without replacing the reading pane."""
    logged = _install_fake_rerun(monkeypatch)
    rr = sys.modules["rerun"]
    payload = _TranscriptPayload(
        "trial/scene/e0",
        6,
        (("tool", "DEBUG", "tool: result"),),
    )

    RerunSink()._emit_transcript(rr, payload)

    assert ("trial/scene/e0/llm", ("TextLog", "tool: result", "DEBUG")) in logged
    assert all(path != "trial/scene/e0/llm/latest" for path, _ in logged)


def test_mixed_transcript_delta_document_contains_only_assistant_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed delta limits the reading pane to assistant content."""
    logged = _install_fake_rerun(monkeypatch)
    rr = sys.modules["rerun"]
    payload = _TranscriptPayload(
        "trial/scene/e0",
        6,
        (
            ("user", "INFO", "user: question"),
            ("assistant", "INFO", "assistant: answer"),
            ("tool", "DEBUG", "tool: result"),
        ),
    )

    RerunSink()._emit_transcript(rr, payload)

    documents = [value for path, value in logged if path == "trial/scene/e0/llm/latest"]
    assert documents == [
        _TextDocument(
            "**[INFO]** assistant: answer",
            media_type="text/markdown",
        )
    ]


def test_empty_transcript_payload_emits_no_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch)
    rr = sys.modules["rerun"]

    RerunSink()._emit_transcript(rr, _TranscriptPayload("trial/scene/e0", 6, ()))

    assert logged == [("set_time", ("step", {"sequence": 6}))]


def test_sdk_without_text_document_keeps_transcript_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged = _install_fake_rerun(monkeypatch, text_document=False)
    rr = sys.modules["rerun"]
    payload = _TranscriptPayload(
        "trial/scene/e0",
        7,
        (
            ("assistant", "INFO", "assistant: answer"),
            ("tool", "DEBUG", "tool: result"),
        ),
    )

    RerunSink()._emit_transcript(rr, payload)

    assert logged == [
        ("set_time", ("step", {"sequence": 7})),
        ("trial/scene/e0/llm", ("TextLog", "assistant: answer", "INFO")),
        ("trial/scene/e0/llm", ("TextLog", "tool: result", "DEBUG")),
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            {"role": "assistant", "content": "hello"},
            ("assistant", "INFO", "assistant: hello"),
        ),
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
            (
                "user",
                "INFO",
                "user: camera 'top':\n[image_url part]\nafter\n[unknown part]",
            ),
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
            ("assistant", "INFO", 'assistant: tool_call move_by({"dx": 0.1})'),
        ),
        ("plain row", ("", "INFO", "plain row")),
        ({"content": "orphan"}, ("unknown", "TRACE", "unknown: orphan")),
        ({"role": "tool", "content": "result"}, ("tool", "DEBUG", "tool: result")),
        (
            {"role": "system", "content": "prompt"},
            ("system", "TRACE", "system: prompt"),
        ),
        (
            {"role": "assistant", "content": ""},
            ("assistant", "INFO", "assistant: "),
        ),
        (
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [{"function": "malformed"}, "malformed"],
            },
            ("assistant", "INFO", "assistant: tool_call ()\ntool_call ()"),
        ),
    ],
)
def test_policy_message_rendering_table(message: Any, expected: tuple[str, str, str]) -> None:
    assert _render_message(message) == expected


def test_mixed_queue_eviction_counts_transcripts_steps_and_images() -> None:
    sink = RerunSink(queue_size=2)
    sink._image_watermark = 99
    sink._enqueue(_TranscriptPayload("trial/scene/e0", 0, (("", "INFO", "first"),)))
    sink._enqueue(_step_payload(1))

    sink._enqueue(_TranscriptPayload("trial/scene/e0", 2, (("", "INFO", "second"),)))
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
    sink._enqueue(_TranscriptPayload("trial/scene/e0", 0, (("", "INFO", "first"),)))
    sink._enqueue(_TranscriptPayload("trial/scene/e0", 1, (("", "INFO", "second"),)))
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
            sink._queue.append(_TranscriptPayload("trial/scene/e0", 1, (("", "INFO", "queued"),)))
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


def test_real_rerun_accepts_the_transcript_document_call() -> None:
    """The real SDK accepts the exact TextDocument construction the sink emits."""
    rr = pytest.importorskip("rerun")
    if not hasattr(rr, "TextDocument"):
        pytest.skip("pre-TextDocument rerun-sdk lacks the archetype")
    rr.TextDocument("**[INFO]** assistant: hi  \nthere", media_type="text/markdown")


@pytest.mark.skipif(not _RERUN_INSTALLED, reason="requires rerun-sdk")
def test_real_rerun_tee_surface_accepts_forwarded_arguments(tmp_path: Path) -> None:
    """The locked SDK exposes every tee constructor and keyword the sink forwards."""
    rr = pytest.importorskip("rerun")
    assert all(hasattr(rr, name) for name in ("set_sinks", "GrpcSink", "FileSink"))
    assert {"port", "memory_limit", "connect"} <= set(inspect.signature(rr.spawn).parameters)
    grpc_sink = rr.GrpcSink(url="rerun+http://127.0.0.1:9876/proxy")
    file_sink = rr.FileSink(tmp_path / "signature.rrd")
    assert grpc_sink is not None
    assert file_sink is not None


@pytest.mark.skipif(not _RERUN_INSTALLED, reason="requires rerun-sdk")
def test_real_rerun_file_sink_round_trip(tmp_path: Path) -> None:
    """A scalar flushed through the real FileSink produces a nonempty recording."""
    rr = pytest.importorskip("rerun")
    target = tmp_path / "t.rrd"
    rr.init("inspect_robots_test_file_sink")
    rr.set_sinks(rr.FileSink(target))
    rr.log("scalar", rr.Scalars(1.0))
    recording = rr.get_global_data_recording()
    assert recording is not None
    recording.flush()
    assert target.exists()
    assert target.stat().st_size > 0


@pytest.mark.skipif(not _RERUN_INSTALLED, reason="rerun-sdk not installed")
def test_real_rerun_accepts_the_blueprint(tmp_path: Path) -> None:
    """The real SDK accepts every constructor and send call in the trial blueprint."""
    rr = pytest.importorskip("rerun")
    sink = RerunSink(str(tmp_path / "blueprint.rrd"))
    sink._rr = rr
    sink.bind_spaces(
        _labeled_action_space(),
        ObservationSpace(
            cameras=(CameraSpec(name="top", height=4, width=4),),
            state=StateSpec(fields=(StateField(key="joint_pos", shape=(4,)),)),
        ),
    )
    sink.on_eval_start(None)  # type: ignore[arg-type]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sink._send_blueprint(rr, "trial/s0/e0")

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
