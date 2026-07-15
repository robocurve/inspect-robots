"""Optional Rerun visualization sink.

Logs camera images, proprioception, action vectors, and success markers to
`Rerun <https://github.com/rerun-io/rerun>`_. The sink can write a ``.rrd``
recording, spawn a local viewer, or connect over gRPC to a remote viewer. A
viewer spawned by the sink has a 2 GiB memory limit by default, which makes
the viewer purge its oldest events instead of accumulating an unbounded
session history.
``rerun-sdk`` is imported lazily *inside* methods so the core package never
depends on it; if it is not installed, the sink warns once and becomes a no-op
(so unattended runs and the core-only import gate are unaffected).

Emission happens on a daemon worker thread: ``log_step`` snapshots the
transition and enqueues it, so a slow or stalled viewer connection can never
block the control-rate rollout loop. Under backpressure the sink degrades
visualization instead of delaying control: camera frames are dropped first
(scalar plots stay complete), then whole steps, and the drop counts are
reported as a ``RuntimeWarning`` when the eval ends. The queue is drained at
every trial boundary (bounded by ``flush_timeout``), so an eval that aborts
mid-run loses at most the current trial's queued tail. Camera frames are
JPEG-compressed by default (``jpeg_quality=75``); pass ``jpeg_quality=None``
for lossless raw frames. If compression is unavailable (an SDK without
``Image.compress``, or pillow missing), the sink warns once and logs raw
frames. All Rerun SDK calls after ``init``/``spawn``/``connect_grpc``/``save``
happen on the worker because the SDK's timeline state is thread-local, except
for the shutdown-path flush probe and ``unregister_shutdown`` on the caller path. The
probe invokes ``RecordingStream.flush`` on a bounded daemon thread; flush is
internally synchronized, and ``unregister_shutdown`` only manipulates an
``atexit`` hook, so neither depends on timeline state. Worker state is
generation-scoped so a worker wedged in a blocked SDK call is disowned at
shutdown and can never double-consume after a restart. If the SDK flush probe
also wedges, the sink unregisters the SDK's unbounded ``atexit`` flush,
disables itself, and abandons queued SDK-side data.

The viewer limit applies only to viewers this package spawns; a viewer already
running on the default port keeps the limit it started with. The bounded exit
probe requires rerun-sdk 0.22 or newer because older recording streams expose
no ``flush`` method. A new sink in the same process can still hang in
``rr.init`` after a connection wedges, and paths such as Ctrl-C that skip
``on_eval_end`` retain the SDK's unbounded ``atexit`` hook.

Each trial's entities are namespaced under ``trial/<scene_id>/e<epoch>`` so
successive trials never overwrite one another on the shared step timeline.

Install with ``pip install "inspect-robots[rerun]"``.
"""

from __future__ import annotations

import dataclasses
import threading
import warnings
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from inspect_robots.types import ImageArray

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog, EvalSpec
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.types import Action, Observation, StepResult


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


@dataclasses.dataclass
class _WorkerState:
    """Per-worker-generation state, so an abandoned worker cannot corrupt its successor."""

    stop: threading.Event
    inflight: int = 0
    stalled: bool = False


class RerunSink:
    """Write a ``.rrd``, spawn a bounded local viewer, or connect to a remote one.

    ``spawn_memory_limit`` is passed verbatim to Rerun only when ``spawn=True``.
    """

    def __init__(
        self,
        recording_path: str | None = None,
        *,
        application_id: str = "inspect_robots",
        spawn: bool = False,
        spawn_memory_limit: str = "2GiB",
        connect_url: str | None = None,
        jpeg_quality: int | None = 75,
        queue_size: int = 64,
        flush_timeout: float = 10.0,
    ):
        """Configure output mode, buffering, and the local viewer memory ceiling.

        ``spawn_memory_limit`` is consulted only when ``spawn`` is true.
        """
        # rerun's save/connect/spawn calls each *replace* the global sink, so
        # combining any two modes would silently drop one of the streams.
        if spawn and connect_url is not None:
            raise ValueError("spawn and connect_url are mutually exclusive")
        if spawn and recording_path is not None:
            raise ValueError("spawn and recording_path are mutually exclusive")
        if recording_path is not None and connect_url is not None:
            raise ValueError("recording_path and connect_url are mutually exclusive")
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        self.recording_path = recording_path
        self.application_id = application_id
        self.spawn = spawn
        self.spawn_memory_limit = spawn_memory_limit
        self.connect_url = connect_url
        self.jpeg_quality = jpeg_quality
        self.queue_size = queue_size
        self.flush_timeout = flush_timeout
        self._rr: Any | None = None
        self._warned = False
        self._disabled = False
        self._prefix = "trial"
        self._queue: deque[_StepPayload] = deque()
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None
        self._state: _WorkerState | None = None
        self._dropped_frames = 0
        self._dropped_steps = 0
        self._image_watermark = max(1, queue_size // 4)
        self._emit_warned = False
        self._compress_warned = False

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

    def _emit(self, rr: Any, payload: _StepPayload) -> None:
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

    def _enqueue(self, payload: _StepPayload) -> None:
        with self._cond:
            if payload.images and len(self._queue) >= self._image_watermark:
                self._dropped_frames += len(payload.images)
                payload = dataclasses.replace(payload, images={})
            if len(self._queue) >= self.queue_size:
                evicted = self._queue.popleft()
                self._dropped_steps += 1
                self._dropped_frames += len(evicted.images)
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
                    self._dropped_steps += len(self._queue)
                    self._dropped_frames += sum(len(p.images) for p in self._queue)
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
        if self._dropped_frames or self._dropped_steps:
            warnings.warn(
                f"RerunSink dropped {self._dropped_frames} camera frame(s) and "
                f"{self._dropped_steps} full step(s) to keep the control loop "
                "unblocked; record to a .rrd file or reduce camera bandwidth "
                "to avoid drops",
                RuntimeWarning,
                stacklevel=2,
            )
            self._dropped_frames = 0
            self._dropped_steps = 0

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
        rr = self._ensure_rerun()
        if rr is None:
            return
        try:
            rr.init(self.application_id)
            if self.spawn:
                rr.spawn(memory_limit=self.spawn_memory_limit)
            if self.connect_url is not None:
                rr.connect_grpc(self.connect_url)
            if self.recording_path is not None:
                rr.save(self.recording_path)
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
