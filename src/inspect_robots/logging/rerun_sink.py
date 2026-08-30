"""Optional Rerun visualization sink.

Logs camera images, proprioception, action vectors, and success markers to
`Rerun <https://github.com/rerun-io/rerun>`_. The sink can write a ``.rrd``
recording, spawn a local viewer, connect over gRPC to a remote viewer, or tee
the same stream to a file and live viewer with rerun-sdk 0.24 or newer. A viewer
spawned by the sink has a 2 GiB memory limit by default, which makes
the viewer purge its oldest events instead of accumulating an unbounded
session history.
``rerun-sdk`` is imported lazily *inside* methods so the core package never
depends on it; if it is not installed, the sink warns once and becomes a no-op
(so unattended runs and the core-only import gate are unaffected).

Emission happens on a daemon worker thread: ``log_step`` snapshots the
transition and enqueues it, so a slow or stalled viewer connection can never
block the control-rate rollout loop. Under backpressure the sink degrades
visualization instead of delaying control: camera frames are dropped first
(scalar plots stay complete), then whole steps or transcript rows, and the drop
counts are reported as a ``RuntimeWarning`` when the eval ends. The queue is drained at
every trial boundary (bounded by ``flush_timeout``), so an eval that aborts
mid-run loses at most the current trial's queued tail. Camera frames are
JPEG-compressed by default (``jpeg_quality=75``); pass ``jpeg_quality=None``
for lossless raw frames. If compression is unavailable (an SDK without
``Image.compress``, or pillow missing), the sink warns once and logs raw
frames. Caller-path startup performs ``init``, ``spawn``, ``connect_grpc``,
``save``, ``set_sinks``, ``GrpcSink``/``FileSink`` construction, and the
``recording_dir`` mkdir. All later Rerun SDK calls happen on the worker because
the SDK's timeline state is thread-local, except for the shutdown-path flush
probe and ``unregister_shutdown`` on the caller path. The
probe invokes ``RecordingStream.flush`` on a bounded daemon thread; flush is
internally synchronized, and ``unregister_shutdown`` only manipulates an
``atexit`` hook, so neither depends on timeline state. Worker state is
generation-scoped so a worker wedged in a blocked SDK call is disowned at
shutdown and can never double-consume after a restart. If the SDK flush probe
also wedges, the sink unregisters the SDK's unbounded ``atexit`` flush,
disables itself, and abandons queued SDK-side data.

The viewer limit applies only to viewers this package spawns; a viewer already
running on the same port keeps the limit it started with. The bounded exit
probe requires rerun-sdk 0.22 or newer because older recording streams expose
no ``flush`` method. A new sink in the same process can still hang in
``rr.init`` after a connection wedges, and paths such as Ctrl-C that skip
``on_eval_end`` retain the SDK's unbounded ``atexit`` hook.

Each trial's entities are namespaced under ``trial/<scene_id>/e<epoch>`` so
successive trials never overwrite one another on the shared step timeline.
The worker sends an explicit per-trial blueprint from ``_emit`` for the live namespace.
Transcripts are emitted as ``TextLog`` rows at ``{prefix}/llm`` paired with a
markdown ``TextDocument`` at ``{prefix}/llm/latest`` holding the step's
assistant message(s) for a wrapped, timeline-synced reading pane.

Install with ``pip install "inspect-robots[rerun]"``.
"""

from __future__ import annotations

import dataclasses
import threading
import warnings
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from inspect_robots.logging.json_log import _slug
from inspect_robots.types import ImageArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from inspect_robots.log import EvalLog, EvalSpec
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.spaces import Box, ObservationSpace
    from inspect_robots.types import Action, Observation, StepResult


def _render_message(message: Any) -> tuple[str, str, str]:
    if not isinstance(message, dict):
        return "", "INFO", str(message)

    role = str(message.get("role", "unknown"))
    level = {
        "assistant": "INFO",
        "user": "INFO",
        "tool": "DEBUG",
    }.get(role, "TRACE")
    lines: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        lines.append(content)
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                part_type = part.get("type", "unknown") if isinstance(part, dict) else "unknown"
                parts.append(f"[{part_type} part]")
        if parts:
            lines.append("\n".join(parts))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if not isinstance(function, dict):
                function = {}
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            lines.append(f"tool_call {name}({arguments})")
    return role, level, f"{role}: " + "\n".join(lines)


def _joint_groups(labels: tuple[str, ...] | None, dim: int) -> list[tuple[str, list[int]]]:
    """Group action dims by their label's first-underscore prefix.

    YAM-style labels (``left_j0`` .. ``right_gripper``) yield one group per
    side, in first-appearance order. Missing labels, a label count that does
    not match the dim count, any label without an underscore (CubePick's
    ``dx``/``dy``, eef-style ``x``/``y``/``grip``), or fewer than two distinct
    prefixes all collapse to a single "joints" group: one plot beats one
    clutter-view per dim.
    """
    if labels is None or len(labels) != dim or not all("_" in label for label in labels):
        return [("joints", list(range(dim)))]
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label.split("_", 1)[0], []).append(index)
    if len(groups) < 2:
        return [("joints", list(range(dim)))]
    return list(groups.items())


_CAMERA_RANK = {"left": 0, "top": 1, "right": 2}


def _camera_order(names: tuple[str, ...]) -> list[str]:
    """Order camera views left, top, right by name prefix.

    The rank key is the name's first-underscore prefix, lowercased, so
    ``left_cam``, ``Left_cam``, and a bare ``left`` all rank as ``left``.
    Operators read the workspace spatially, which rarely matches the
    declared camera order (YAM declares top first). Names without a ranked
    prefix sort after the ranked ones; ties break in codepoint order.
    """
    return sorted(
        names,
        key=lambda name: (
            _CAMERA_RANK.get(name.split("_", 1)[0].lower(), len(_CAMERA_RANK)),
            name,
        ),
    )


@dataclasses.dataclass(frozen=True)
class _StepPayload:
    """One transition, snapshotted so no live buffers are shared across threads."""

    prefix: str
    t: int
    images: dict[str, ImageArray]
    state: dict[str, npt.NDArray[np.float64]]
    action: npt.NDArray[np.float64]
    reward: float | None
    terminated: bool
    termination_reason: str | None


@dataclasses.dataclass(frozen=True)
class _TranscriptPayload:
    """Rendered (role, level, text) rows snapshotted for one control step."""

    prefix: str
    t: int
    entries: tuple[tuple[str, str, str], ...]


@dataclasses.dataclass
class _WorkerState:
    """Per-worker-generation state, so an abandoned worker cannot corrupt its successor."""

    stop: threading.Event
    inflight: int = 0
    stalled: bool = False


class RerunSink:
    """Record and visualize one Rerun stream without delaying the control loop.

    A recording target can be combined with either live mode when rerun-sdk
    0.24 or newer provides ``set_sinks``. ``recording_dir`` creates one file per
    eval named ``{task_slug}_{eight_hex_digits}.rrd``; ``recording_path`` reuses
    one fixed file. ``resolved_recording_path`` exposes the file attached for
    the current eval, or ``None`` when no file was attached. For a reused sink,
    the next eval's ``rr.init`` releases the previous ``FileSink`` after
    ``on_eval_end`` has probed its flush. A fixed path is truncated and rewritten
    on each eval, matching ``rr.save`` semantics.

    ``spawn_memory_limit`` is passed verbatim to Rerun only when ``spawn=True``.
    """

    def __init__(
        self,
        recording_path: str | None = None,
        *,
        recording_dir: str | None = None,
        application_id: str = "inspect_robots",
        spawn: bool = False,
        spawn_memory_limit: str = "2GiB",
        spawn_port: int = 9876,
        connect_url: str | None = None,
        jpeg_quality: int | None = 75,
        queue_size: int = 64,
        flush_timeout: float = 10.0,
    ):
        """Configure file and live targets, buffering, and the viewer memory ceiling.

        ``recording_path`` and ``recording_dir`` are mutually exclusive. Either
        may be combined with ``spawn`` or ``connect_url`` for a tee on
        rerun-sdk 0.24 or newer. The resolved target remains ``None`` until
        ``on_eval_start`` successfully attaches the file sink.
        ``spawn_memory_limit`` is consulted only when ``spawn`` is true.
        ``spawn_port`` is consulted only when ``spawn`` is true.
        """
        # A local and remote viewer still conflict. File recording combines
        # with either viewer through set_sinks on rerun-sdk 0.24 or newer.
        if spawn and connect_url is not None:
            raise ValueError("spawn and connect_url are mutually exclusive")
        if recording_path is not None and recording_dir is not None:
            raise ValueError("recording_path and recording_dir are mutually exclusive")
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        if not 1 <= spawn_port <= 65535:
            raise ValueError(f"spawn_port must be in 1-65535, got {spawn_port}")
        self.recording_path = recording_path
        self.recording_dir = recording_dir
        self.resolved_recording_path: Path | None = None
        self.application_id = application_id
        self.spawn = spawn
        self.spawn_memory_limit = spawn_memory_limit
        self.spawn_port = spawn_port
        self.connect_url = connect_url
        self.jpeg_quality = jpeg_quality
        self.queue_size = queue_size
        self.flush_timeout = flush_timeout
        self._rr: Any | None = None
        self._warned = False
        self._set_sinks_warned = False
        self._disabled = False
        self._prefix = "trial"
        self._queue: deque[_StepPayload | _TranscriptPayload] = deque()
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None
        self._state: _WorkerState | None = None
        self._dropped_frames = 0
        self._dropped_steps = 0
        self._dropped_transcripts = 0
        self._image_watermark = max(1, queue_size // 4)
        self._emit_warned = False
        self._compress_warned = False
        self._dim_labels: tuple[str, ...] | None = None
        self._action_dim: int | None = None
        self._camera_names: tuple[str, ...] = ()
        self._state_keys: tuple[str, ...] = ()
        self._state_lengths: dict[str, int] = {}
        self._blueprint_prefix: str | None = None
        self._blueprint_warned = False

    def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
        """Distill the resolved spaces into the fields the blueprint needs.

        Called by ``eval()`` before ``on_eval_start`` (and therefore before
        the worker thread exists); storing plain tuples rather than the space
        objects keeps the worker free of shared mutable state.
        """
        self._action_dim = action_space.dim
        semantics = action_space.semantics
        self._dim_labels = None if semantics is None else semantics.dim_labels
        self._camera_names = tuple(camera.name for camera in observation_space.cameras)
        self._state_keys = tuple(sorted(observation_space.state_keys))
        state = observation_space.state
        self._state_lengths = (
            {field.key: field.shape[0] for field in state.fields if len(field.shape) == 1}
            if state is not None
            else {}
        )

    def _ensure_rerun(self) -> Any | None:
        if self._disabled:
            return None
        if self._rr is not None:
            return self._rr
        try:
            import rerun as rr
        except ImportError:
            if not self._warned:
                warnings.warn(
                    "rerun-sdk is not installed; RerunSink is a no-op. "
                    'Install with: pip install "inspect-robots[rerun]"',
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return None
        self._rr = rr
        return rr

    @property
    def available(self) -> bool:
        """Whether the optional SDK can currently accept visualization events."""
        return self._ensure_rerun() is not None

    @staticmethod
    def _set_step(rr: Any, t: int) -> None:
        if hasattr(rr, "set_time"):  # rerun-sdk >= 0.23
            rr.set_time("step", sequence=t)
        else:  # older SDKs
            rr.set_time_sequence("step", t)

    @staticmethod
    def _scalar(rr: Any, value: float) -> Any:
        scalars = getattr(rr, "Scalars", None)  # rerun-sdk >= 0.23
        if scalars is not None:
            return scalars(value)
        return rr.Scalar(value)  # older SDKs

    def _image(self, rr: Any, image: ImageArray) -> Any:
        img = rr.Image(image)
        if self.jpeg_quality is None:
            return img
        compress = getattr(img, "compress", None)
        if compress is None:  # pre-compress SDK surface: log raw
            self._warn_compress_fallback("this rerun-sdk has no Image.compress")
            return img
        try:
            return compress(jpeg_quality=self.jpeg_quality)
        except Exception as exc:  # encode failure (e.g. missing pillow): log raw
            self._warn_compress_fallback(str(exc))
            return img

    def _warn_compress_fallback(self, reason: str) -> None:
        if self._compress_warned:
            return
        self._compress_warned = True
        warnings.warn(
            f"RerunSink could not JPEG-compress camera frames ({reason}); "
            "logging raw frames instead. Install pillow for compression.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _send_blueprint(self, rr: Any, prefix: str) -> None:
        """Send an explicit layout for one trial namespace, if the SDK can.

        The layout is two rows (a run with no cameras sends only the second
        row): the camera views in left/top/right order across the top, then
        a tabbed text panel beside the per-group joint plots. The latest
        LLM message is the default tab; the full transcript log and the
        reward series sit behind tabs, so reward-logging runs keep it one
        click away while agent-policy runs, which never log rewards, do not
        spend a permanently empty plot slot on it.

        Rerun's entity queries support ``/**`` only as a suffix, so the views
        name each trial's entities with concrete paths and the layout is
        re-sent when a new trial namespace begins (the viewer follows the
        live trial; a single-trial ``run`` sends exactly once). Skipped when
        spaces were never bound or the SDK predates blueprints; any build or
        send failure degrades to the automatic layout with a single warning.
        """
        # Falsy covers both "never bound" (None) and a 0-dim action space,
        # which would otherwise build empty-content views.
        if not self._action_dim:
            return
        rrb = getattr(rr, "blueprint", None)
        send = getattr(rr, "send_blueprint", None)
        if rrb is None or send is None:
            return
        try:
            plots = []
            for name, indices in _joint_groups(self._dim_labels, self._action_dim):
                contents = [f"+ {prefix}/action/{index}" for index in indices]
                for key, length in self._state_lengths.items():
                    if length == self._action_dim:
                        contents += [f"+ {prefix}/state/{key}/{index}" for index in indices]
                plots.append(rrb.TimeSeriesView(name=name, origin="/", contents=contents))
            aligned = {
                key for key, length in self._state_lengths.items() if length == self._action_dim
            }
            for key in self._state_keys:
                if key not in aligned:
                    plots.append(
                        rrb.TimeSeriesView(
                            name=key,
                            origin="/",
                            contents=[f"+ {prefix}/state/{key}/**"],
                        )
                    )
            cameras = [
                rrb.Spatial2DView(
                    name=camera,
                    origin="/",
                    contents=[f"+ {prefix}/camera/{camera}"],
                )
                for camera in _camera_order(self._camera_names)
            ]
            text = rrb.Tabs(
                rrb.TextDocumentView(name="latest", contents=[f"+ {prefix}/llm/latest"]),
                rrb.TextLogView(
                    name="llm",
                    contents=[f"+ {prefix}/llm", f"+ {prefix}/event/**"],
                ),
                rrb.TimeSeriesView(
                    name="reward",
                    origin="/",
                    contents=[f"+ {prefix}/reward"],
                ),
            )
            row = rrb.Horizontal(text, *plots)
            if cameras:
                send(rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*cameras), row)))
            else:
                send(rrb.Blueprint(row))
        except Exception as exc:
            if not self._blueprint_warned:
                self._blueprint_warned = True
                warnings.warn(
                    f"RerunSink could not send the blueprint layout ({exc}); "
                    "the viewer will use its automatic layout",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _emit(self, rr: Any, payload: _StepPayload | _TranscriptPayload) -> None:
        if payload.prefix != self._blueprint_prefix:
            self._blueprint_prefix = payload.prefix
            self._send_blueprint(rr, payload.prefix)
        if isinstance(payload, _TranscriptPayload):
            self._emit_transcript(rr, payload)
            return
        self._set_step(rr, payload.t)
        pre = payload.prefix
        for cam, image in payload.images.items():
            rr.log(f"{pre}/camera/{cam}", self._image(rr, image))
        for key, vec in payload.state.items():
            for i, scalar in enumerate(vec):
                rr.log(f"{pre}/state/{key}/{i}", self._scalar(rr, float(scalar)))
        for i, scalar in enumerate(payload.action):
            rr.log(f"{pre}/action/{i}", self._scalar(rr, float(scalar)))
        if payload.reward is not None:
            rr.log(f"{pre}/reward", self._scalar(rr, payload.reward))
        if payload.terminated:
            rr.log(
                f"{pre}/event/terminated",
                rr.TextLog(payload.termination_reason or "terminated"),
            )

    def _emit_transcript(self, rr: Any, payload: _TranscriptPayload) -> None:
        self._set_step(rr, payload.t)
        for _role, level, text in payload.entries:
            rr.log(f"{payload.prefix}/llm", rr.TextLog(text, level=level))
        self._emit_transcript_document(rr, payload)

    def _emit_transcript_document(self, rr: Any, payload: _TranscriptPayload) -> None:
        text_document = getattr(rr, "TextDocument", None)
        if text_document is None or not payload.entries:
            return
        assistant_entries = [
            (level, text) for role, level, text in payload.entries if role == "assistant"
        ]
        if not assistant_entries:
            return
        body = "\n\n---\n\n".join(
            f"**[{level}]** " + text.replace("\n", "  \n") for level, text in assistant_entries
        )
        rr.log(
            f"{payload.prefix}/llm/latest",
            text_document(body, media_type="text/markdown"),
        )

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        state = _WorkerState(stop=threading.Event())
        self._state = state
        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(state,),
            name="inspect-robots-rerun-sink",
            daemon=True,
        )
        self._worker.start()

    def _enqueue(self, payload: _StepPayload | _TranscriptPayload) -> None:
        with self._cond:
            if (
                isinstance(payload, _StepPayload)
                and payload.images
                and len(self._queue) >= self._image_watermark
            ):
                self._dropped_frames += len(payload.images)
                payload = dataclasses.replace(payload, images={})
            if len(self._queue) >= self.queue_size:
                evicted = self._queue.popleft()
                if isinstance(evicted, _StepPayload):
                    self._dropped_steps += 1
                    self._dropped_frames += len(evicted.images)
                else:
                    self._dropped_transcripts += 1
            self._queue.append(payload)
            self._cond.notify_all()

    def _worker_loop(self, state: _WorkerState) -> None:
        while True:
            with self._cond:
                while self._state is state and not self._queue and not state.stop.is_set():
                    self._cond.wait()
                if self._state is not state or not self._queue:
                    # Disowned after a wedged shutdown, or stopped and drained:
                    # either way this generation is done and must not consume.
                    return
                payload = self._queue.popleft()
                state.inflight = 1
            try:
                self._emit(self._rr, payload)
            except Exception as exc:
                self._warn_emit_failure(exc)
            finally:
                with self._cond:
                    state.inflight = 0
                    self._cond.notify_all()

    def _warn_emit_failure(self, exc: Exception) -> None:
        if self._emit_warned:
            return
        self._emit_warned = True
        warnings.warn(
            f"RerunSink failed to emit a step ({exc}); further emit failures are silent",
            RuntimeWarning,
            stacklevel=2,
        )

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every queued event was handed to the SDK; False on timeout."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._queue and (self._state is None or self._state.inflight == 0),
                timeout,
            )

    def _shutdown(self) -> None:
        worker, state = self._worker, self._state
        wedged = False
        if worker is not None and state is not None:
            self.flush(timeout=self.flush_timeout)
            state.stop.set()
            with self._cond:
                self._cond.notify_all()
            worker.join(timeout=self.flush_timeout)
            self._worker = None
            wedged = worker.is_alive()
            with self._cond:
                self._state = None
                if wedged:
                    # Disown the wedged worker: clear its backlog into the drop
                    # counters so a restarted worker never double-consumes it.
                    for payload in self._queue:
                        if isinstance(payload, _StepPayload):
                            self._dropped_steps += 1
                            self._dropped_frames += len(payload.images)
                        else:
                            self._dropped_transcripts += 1
                    self._queue.clear()
            self._emit_warned = False

        self._probe_recording_flush()
        if worker is None or state is None:
            return
        if wedged:
            warnings.warn(
                "RerunSink shutdown timed out with visualization data still "
                "queued; the viewer connection appears stalled",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._dropped_frames or self._dropped_steps or self._dropped_transcripts:
            transcript_fragment = (
                f" and {self._dropped_transcripts} transcript update(s)"
                if self._dropped_transcripts
                else ""
            )
            warnings.warn(
                f"RerunSink dropped {self._dropped_frames} camera frame(s) and "
                f"{self._dropped_steps} full step(s){transcript_fragment} "
                "to keep the control loop "
                "unblocked; record to a .rrd file only (no live viewer) or reduce camera bandwidth "
                "to avoid drops",
                RuntimeWarning,
                stacklevel=2,
            )
            self._dropped_frames = 0
            self._dropped_steps = 0
            self._dropped_transcripts = 0

    def _probe_recording_flush(self) -> None:
        rr = self._rr
        if rr is None or self._disabled:
            return
        get_rec = getattr(rr, "get_global_data_recording", None)
        rec = get_rec() if get_rec is not None else None
        if rec is None:
            return
        flush = getattr(rec, "flush", None)
        if flush is None:
            return

        def _flush() -> None:
            try:  # noqa: SIM105 - the probe contract explicitly swallows exceptions
                flush()
            except Exception:
                pass

        probe = threading.Thread(
            target=_flush,
            name="inspect-robots-rerun-flush-probe",
            daemon=True,
        )
        probe.start()
        probe.join(timeout=self.flush_timeout)
        if not probe.is_alive():
            return

        unregister = getattr(rr, "unregister_shutdown", None)
        if unregister is not None:
            try:  # noqa: SIM105 - keep the compatibility shim as a guarded call
                unregister()
            except Exception:
                pass
        self._disabled = True
        warnings.warn(
            "RerunSink viewer connection is stalled; visualization is disabled "
            "for this sink and queued SDK-side data was abandoned",
            RuntimeWarning,
            stacklevel=3,
        )

    def on_eval_start(self, spec: EvalSpec) -> None:
        """Initialize recording, disabling this noncritical sink after startup failure."""
        # A crash path can skip on_eval_end; stop the previous eval's worker before
        # resetting blueprint state so its writes can't land after the reset.
        if self._worker is not None and self._worker.is_alive():
            self._shutdown()
        self.resolved_recording_path = None
        self._blueprint_prefix = None
        self._blueprint_warned = False
        rr = self._ensure_rerun()
        if rr is None:
            return
        try:
            if self.recording_dir is not None:
                recording_dir = Path(self.recording_dir)
                recording_dir.mkdir(parents=True, exist_ok=True)
                self.resolved_recording_path = (
                    recording_dir / f"{_slug(spec.task)}_{uuid4().hex[:8]}.rrd"
                )
            elif self.recording_path is not None:
                self.resolved_recording_path = Path(self.recording_path)

            rr.init(self.application_id)
            live = self.spawn or self.connect_url is not None
            if live and self.resolved_recording_path is not None:
                if all(hasattr(rr, name) for name in ("set_sinks", "GrpcSink", "FileSink")):
                    if self.spawn:
                        rr.spawn(
                            memory_limit=self.spawn_memory_limit,
                            port=self.spawn_port,
                            connect=False,
                        )
                    url = (
                        self.connect_url
                        if self.connect_url is not None
                        else f"rerun+http://127.0.0.1:{self.spawn_port}/proxy"
                    )
                    rr.set_sinks(
                        rr.GrpcSink(url=url),
                        rr.FileSink(self.resolved_recording_path),
                    )
                else:
                    if not self._set_sinks_warned:
                        warnings.warn(
                            "this rerun-sdk predates set_sinks; the live view continues but "
                            "the .rrd was skipped; upgrade to rerun-sdk>=0.24",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._set_sinks_warned = True
                    self.resolved_recording_path = None
                    if self.spawn:
                        rr.spawn(memory_limit=self.spawn_memory_limit, port=self.spawn_port)
                    if self.connect_url is not None:
                        rr.connect_grpc(self.connect_url)
            elif live:
                if self.spawn:
                    rr.spawn(memory_limit=self.spawn_memory_limit, port=self.spawn_port)
                if self.connect_url is not None:
                    rr.connect_grpc(self.connect_url)
            elif self.resolved_recording_path is not None:
                rr.save(self.resolved_recording_path)
        except Exception as exc:
            # A visualization sink must never take the eval down with it — a
            # missing viewer binary (spawn), unreachable viewer (connect), or
            # unwritable recording path degrades to a warned no-op, exactly
            # like a missing rerun-sdk.
            warnings.warn(
                f"RerunSink disabled: could not start the Rerun recording/viewer ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            self.resolved_recording_path = None
            self._rr = None
            self._disabled = True

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        """Select the entity namespace for the incoming scene and epoch."""
        # Namespace this trial's entities so trials never overwrite each other.
        self._prefix = f"trial/{scene_id}/e{epoch}"

    def log_step(
        self, t: int, observation: Observation, action: Action, result: StepResult
    ) -> None:
        """Snapshot one transition's observations, action, reward, and termination marker."""
        rr = self._ensure_rerun()
        if rr is None:
            return
        payload = _StepPayload(
            prefix=self._prefix,
            t=t,
            images={cam: np.array(img) for cam, img in observation.images.items()},
            state={
                key: np.atleast_1d(np.array(value, dtype=np.float64))
                for key, value in observation.state.items()
            },
            action=np.atleast_1d(np.array(action.data, dtype=np.float64)),
            reward=None if result.reward is None else float(result.reward),
            terminated=result.terminated,
            termination_reason=result.termination_reason,
        )
        self._ensure_worker()
        self._enqueue(payload)

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        """Queue best-effort transcript rows without mutating policy messages."""
        rr = self._ensure_rerun()
        if rr is None:
            return
        payload = _TranscriptPayload(
            prefix=self._prefix,
            t=t,
            entries=tuple(_render_message(message) for message in messages),
        )
        self._ensure_worker()
        self._enqueue(payload)

    def on_trial_end(self, record: TrialRecord) -> None:
        """Drain queued events between trials, bounding loss if the eval aborts mid-run.

        ``eval()`` does not guarantee ``on_eval_end`` on every failure path
        (scorer/hook exceptions, or Ctrl-C outside the rollout window), so
        trial boundaries are the flush points that cap tail loss at one trial.
        Blocking here is bounded by ``flush_timeout`` and happens between trials,
        never inside the control-rate loop. Once a flush times out, the connection
        is treated as stalled for the rest of this worker generation and later
        trial boundaries return immediately instead of re-paying the timeout;
        the eval-end drop report still accounts for whatever never drained.
        """
        state = self._state
        if state is not None and state.stalled:
            return
        if not self.flush(timeout=self.flush_timeout) and state is not None:
            state.stalled = True

    def on_eval_end(self, log: EvalLog) -> None:
        """Stop the worker and probe the SDK flush; waits at most ~3x the timeout."""
        self._shutdown()
