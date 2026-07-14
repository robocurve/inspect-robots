# Non-Blocking RerunSink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `RerunSink` must never add latency to the rollout control loop, no matter how slow the live viewer connection is; under backpressure it drops camera frames (then whole steps) instead of blocking, and it JPEG-compresses frames to cut bandwidth ~10-20x.

**Architecture:** `log_step` becomes a cheap snapshot-and-enqueue: it copies the step data into an immutable `_StepPayload` and appends it to a bounded deque. A single daemon worker thread owns every `rerun` SDK call (`set_time`, JPEG encode, `log`) because the SDK's timeline state is thread-local. Under pressure the enqueue path first strips images from new payloads (scalar plots stay gap-free), then evicts the oldest whole payloads; drops are counted and reported once at eval end. Worker state is generation-scoped (`_WorkerState`, one per started worker) so a worker wedged inside a blocked SDK call can be disowned at shutdown and can never double-consume or corrupt `flush()` accounting after a restart. `on_trial_end` drains the queue (bounded) so an eval that aborts mid-run loses at most the current trial's tail; `on_eval_end` flushes with a timeout and joins the worker; a wedged SDK gets abandoned (daemon thread), never waited on forever.

**Tech Stack:** stdlib only (`threading`, `collections.deque`, `dataclasses`) — the core stays NumPy-only. `rerun-sdk` remains a lazily imported optional extra; JPEG encoding uses `rr.Image(...).compress(jpeg_quality=...)` with graceful fallback to raw images on older SDKs or encode failure.

## Motivation (observed failure)

Live-viewer runs on real hardware (yam arms, 2026-07-14) emitted
`WARN re_quota_channel::sync: batcher_output: Sender has been blocked for over 5 seconds waiting for space in channel`
and the episode timer visibly stalled. Raw camera frames exceed what the SDK's
bounded channels + gRPC link to the viewer can drain; the SDK *blocks* the
calling thread rather than dropping data, and `log_step` is called inline from
the single control-rate rollout loop (a core invariant, plan 0001), so the
robot itself hiccups. Opening the viewer shows buffered video fast-forwarding
to catch up. Visualization must degrade (dropped frames), control must not.

## Global Constraints

- Core stays NumPy-only: no new runtime deps; `threading`/`collections`/`dataclasses` are stdlib. The `core-only-import` CI job must stay green.
- `pytest --cov` at 100% coverage (`--cov-fail-under=100`); every new branch needs a test, including fallback and timeout branches.
- `mypy --strict` clean over `src` and `tests`.
- `ruff check .` and `ruff format --check .` clean; D1 rules require docstrings on every public module/class/function (underscore-prefixed helpers are exempt). Docstrings state the contract, not the symbol name.
- Python 3.10-3.13 compatibility (no 3.11+-only syntax).
- `rerun`/`torch` must never be imported at module top in core or `mock/` (lazy import inside methods only).
- Public API surface: no new `inspect_robots.__all__` entries (only `RerunSink` keyword args change), so `tests/test_api_snapshot.py` stays untouched.
- Dependency change (pillow added to the `rerun`/`viz` extras): run `uv lock` and commit `uv.lock` in the same commit, or CI fails with "the lockfile needs to be updated".
- Run the coverage gate in a rerun-free dev env (`uv sync --extra dev`): the rerun-absent tests are `skipif`-gated and are required for 100% coverage, so an env with `rerun-sdk` installed (this rig may have one for hardware runs) fails the gate for pre-existing reasons, not because of this change.
- Public-facing text (README) follows the repo writing-style rules: no em dashes in prose, no mid-sentence bold, no decorative emoji.
- Commit after each task (small focused commits, author `Jay Chooi` per repo config).

## File Structure

```
plans/0012-nonblocking-rerun-sink.md        (this plan)
pyproject.toml                              (modify: pillow in rerun/viz extras)
uv.lock                                     (regenerate)
src/inspect_robots/logging/rerun_sink.py    (modify: payload, worker, drop policy)
tests/test_rerun_sink.py                    (modify: new fake-backend + threading tests)
tests/test_coverage_completion.py           (modify: flush after direct log_step calls)
README.md                                   (modify: one sentence on non-blocking viz)
src/inspect_robots/CLAUDE.md                (modify: logging/ row mentions non-blocking sink)
```

All sink changes stay inside `rerun_sink.py`; no other module's behavior changes. `cli.py:553`'s `RerunSink(spawn=True)` picks up the new defaults with no CLI change.

---

### Task 1: Snapshot payload + single emit path + JPEG compression

Refactor `log_step` so all SDK calls go through one `_emit(rr, payload)` method operating on an immutable snapshot, and add JPEG compression with fallback. Behavior stays synchronous in this task; the worker thread arrives in Task 2. This split keeps each diff reviewable and keeps the suite green at every commit.

**Files:**
- Modify: `src/inspect_robots/logging/rerun_sink.py`
- Modify: `pyproject.toml:38-39` (extras) + regenerate `uv.lock`
- Test: `tests/test_rerun_sink.py`

**Interfaces:**
- Consumes: existing `RerunSink._ensure_rerun`, `_set_step`, `_scalar` (unchanged).
- Produces (Task 2 and 3 rely on these exact names):
  - `_StepPayload` frozen dataclass with fields `prefix: str`, `t: int`, `images: dict[str, ImageArray]`, `state: dict[str, npt.NDArray[np.float64]]`, `action: npt.NDArray[np.float64]`, `reward: float | None`, `terminated: bool`, `termination_reason: str | None`.
  - `RerunSink._emit(self, rr: Any, payload: _StepPayload) -> None`.
  - `RerunSink._image(rr: Any, image: ImageArray, jpeg_quality: int | None) -> Any` (staticmethod).
  - `RerunSink.__init__` gains keyword-only `jpeg_quality: int | None = 75`.

- [ ] **Step 1: Add pillow to the rerun extras and relock**

`rr.Image(...).compress()` needs PIL at runtime; without it every frame silently falls back to raw and the whole point is lost on real rigs. In `pyproject.toml` change:

```toml
rerun = ["rerun-sdk>=0.20", "pillow>=10"]
viz = ["rerun-sdk>=0.20", "pillow>=10"]
```

Run: `uv lock`
Expected: `uv.lock` updated without errors. (Core deps unchanged; pillow is extra-only, so the `core-only-import` job is unaffected.)

- [ ] **Step 2: Write the failing tests for `_image` compression behavior**

Append to `tests/test_rerun_sink.py` (new imports at top of file: `import sys`, `import threading`, `import time`, `import types`, `import numpy as np`, and `from inspect_robots.types import Action, Observation, StepResult`; keep the existing imports — `time` is used from Task 2 on):

```python
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
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_rerun_sink.py -v -k "jpeg or compress or old_sdk"`
Expected: 4 FAILs (`RerunSink.__init__` has no `jpeg_quality` parameter / camera value is the bare `_CompressibleImage`, not the compressed marker).

- [ ] **Step 4: Implement `_StepPayload`, `_image`, `_emit`, and the sync rewire**

In `src/inspect_robots/logging/rerun_sink.py`:

Ensure the imports include the following — `warnings`, `TYPE_CHECKING`, `Any`, and `numpy as np` are already imported, so only add what's missing; keep `from __future__ import annotations`:

```python
import dataclasses

import numpy.typing as npt

from inspect_robots.types import ImageArray
```

(`inspect_robots.types` is NumPy-only, so a runtime import keeps the core-only gate green; there is no import cycle because `types.py` imports nothing from `logging/`.)

Add the payload dataclass above the class:

```python
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
```

Extend `__init__` (new parameter shown in context; `queue_size`/`flush_timeout` arrive in Task 2):

```python
    def __init__(
        self,
        recording_path: str | None = None,
        *,
        application_id: str = "inspect_robots",
        spawn: bool = False,
        jpeg_quality: int | None = 75,
    ):
        self.recording_path = recording_path
        self.application_id = application_id
        self.spawn = spawn
        self.jpeg_quality = jpeg_quality
        self._rr: Any | None = None
        self._warned = False
        self._disabled = False
        self._prefix = "trial"
```

Add the compression helper and the emit path:

```python
    @staticmethod
    def _image(rr: Any, image: ImageArray, jpeg_quality: int | None) -> Any:
        img = rr.Image(image)
        if jpeg_quality is None:
            return img
        compress = getattr(img, "compress", None)
        if compress is None:  # pre-compress SDK surface: log raw
            return img
        try:
            return compress(jpeg_quality=jpeg_quality)
        except Exception:  # encode failure (e.g. missing PIL): log raw
            return img

    def _emit(self, rr: Any, payload: _StepPayload) -> None:
        self._set_step(rr, payload.t)
        pre = payload.prefix
        for cam, image in payload.images.items():
            rr.log(f"{pre}/camera/{cam}", self._image(rr, image, self.jpeg_quality))
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
```

Rewrite `log_step` to snapshot and (for now, synchronously) emit. The copies are load-bearing: embodiments may reuse camera buffers, and from Task 2 on the payload crosses a thread boundary.

```python
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
        self._emit(rr, payload)
```

Delete nothing else; `_set_step`/`_scalar` stay as they are.

- [ ] **Step 5: Run the full suite and gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all pass, coverage 100%. The pre-existing fake in `tests/test_coverage_completion.py` uses `fake.Image = lambda img: ("Image",)`, whose tuple has no `compress` attribute, so the `compress is None` branch is covered there; the four new tests cover the remaining three branches of `_image`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/inspect_robots/logging/rerun_sink.py tests/test_rerun_sink.py
git commit -m "feat(rerun): snapshot payloads and JPEG-compress camera frames"
```

---

### Task 2: Worker thread, bounded queue, drop policy, flush, shutdown

Move all emission onto a daemon worker thread. `log_step` becomes enqueue-only and can never block on the SDK. This is the atomic switch to async, so it also updates the two existing fake-backend tests that would otherwise race.

**Files:**
- Modify: `src/inspect_robots/logging/rerun_sink.py`
- Modify: `tests/test_coverage_completion.py:481-509` (flush after direct `log_step` calls)
- Test: `tests/test_rerun_sink.py`

**Interfaces:**
- Consumes: `_StepPayload`, `RerunSink._emit` from Task 1 (exact signatures above).
- Produces (Task 3 relies on these exact names):
  - `_WorkerState` module-private dataclass: `stop: threading.Event`, `inflight: int = 0` — one instance per worker generation.
  - `RerunSink.__init__` gains keyword-only `queue_size: int = 64` (validated `>= 1`, else `ValueError`) and `flush_timeout: float = 10.0` (stored as `self.queue_size`, `self.flush_timeout`).
  - `RerunSink.flush(self, timeout: float | None = None) -> bool` (public, docstringed).
  - `RerunSink.on_trial_end` now drains the queue, bounded by `flush_timeout`.
  - `RerunSink._shutdown(self) -> None` (called from `on_eval_end`; Task 3 adds warnings inside it).
  - Instance state: `self._queue: deque[_StepPayload]`, `self._cond: threading.Condition`, `self._worker: threading.Thread | None`, `self._state: _WorkerState | None`, `self._dropped_frames: int`, `self._dropped_steps: int`, `self._image_watermark: int`, `self._emit_warned: bool`.

Threading model (document this in the module docstring in Task 4):

- Single producer (the rollout loop calls `log_step` sequentially; verified: `rollout.py` calls sinks from the one control loop via the sequential `_Broadcast` in `eval.py`), single consumer (the current worker). All queue state is guarded by one `Condition`.
- Rerun's timeline (`set_time`) is thread-local, so every SDK call after `init`/`save` must happen on the worker; emitting from two threads would stamp wrong step indices.
- Drop policy on enqueue: if the queue already holds `_image_watermark` (= `max(1, queue_size // 4)`) items, strip the new payload's images (`_dropped_frames += 1`); if the queue is at `queue_size`, evict the oldest whole payload (`_dropped_steps += 1`, plus `_dropped_frames += 1` if it still carried images, so the frame count in the eval-end report does not undercount). The goal of the two tiers is scalar continuity: state/action/reward plots stay gap-free as long as whole steps survive, while video degrades first. Images are stripped from the incoming payload because that is the cheapest point to intervene (no queue re-scan); this favors already-queued frames, which is acceptable since the win is scalar completeness, not frame recency.
- Worker generations: each started worker gets its own `_WorkerState` (its stop event and in-flight flag) and exits as soon as it observes `self._state is not state` — so a worker abandoned mid-`_emit` by a timed-out shutdown finishes its one in-flight payload and then dies instead of rejoining the queue as a rogue second consumer, and its `inflight` flag can never corrupt a successor's `flush()` accounting. A shared `self._stop`/`self._busy` would break exactly this way after a wedge-then-restart (stale event replaced, bool clobbered across generations); do not "simplify" back to that.
- The worker is started lazily from `log_step` and restarts after `_shutdown`, so direct `log_step` calls in tests (and any sink reuse across evals) keep working.
- Shutdown on a wedged worker: after `flush` times out and `join` times out, the worker is disowned (`self._state = None`), its backlog is cleared into the drop counters (a restarted worker must never double-consume it), and the daemon thread is left to die when it unwedges. Worst-case `on_eval_end` wait is 2 x `flush_timeout` (flush + join); document that on `on_eval_end`.
- Eval-abort data loss: `bus.on_eval_end` in `eval.py` is not wrapped in `try/finally` — an unguarded scorer raise (`eval.py:338`), a `before_scoring` raise (documented to propagate), a sibling sink raising, or Ctrl-C on the rig all skip it. The synchronous sink had already delivered everything on those paths; an async queue would silently lose its tail. Mitigation: `on_trial_end` flushes (bounded by `flush_timeout`), so at most the current trial's queue is at risk, and blocking *between* trials does not violate the control-loop invariant. The residual mid-trial window is accepted and documented (see non-goals); changing `eval.py` is out of scope for this plan.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rerun_sink.py`:

```python
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
        sink.on_eval_end(None)  # type: ignore[arg-type]  # flush+join time out; A disowned
        assert sink._worker is None and sink._state is None
        assert sink._dropped_steps == 1  # payload 1 went down with the backlog
        _log_one(sink, 2)  # starts worker B
    finally:
        gate.set()
    assert sink.flush(timeout=5.0)
    worker_a.join(timeout=5.0)
    assert not worker_a.is_alive()  # A exited; it never became a second consumer
    camera = [p for p, _ in logged if p == "trial/camera/cam"]
    assert len(camera) == 2  # A's in-flight payload 0, B's payload 2; payload 1 dropped
    sink.on_eval_end(None)  # type: ignore[arg-type]
```

Notes for the implementer:
- `gate.set()` sits in `finally` blocks so a failing assertion can't leave the daemon worker wedged for the 30 s gate timeout while later tests run.
- The assertions are deliberately range-based where worker pop timing races (`3 <= _dropped_frames <= 4`): the worker may or may not have popped payload 0 before payload 2 is enqueued. Do not tighten them. Where an exact count or a specific branch is required (`_dropped_steps == 1` in the wedge test; the evicted-with-images branch in the never-blocks test, which the 100% gate needs), `_wait_for_inflight` pins the interleaving first — do not remove those calls.
- `_wait_for_inflight` is defined later in the file than its first caller; that is fine (resolution happens at call time), keep the append order as written.
- In the wedge test, asserting `len(camera) == 2` only after `worker_a.join(...)` is load-bearing: joining A guarantees its in-flight payload 0 was emitted before the count is read.
- `pytest.warns` captures warnings raised on the worker thread because the `warnings` module's state is process-global; the `flush` inside the block guarantees the emit attempts happened before the block closes. (py3.14's context-local warnings will change this — if the weekly canary starts failing here, that's why.)
- `test_log_step_never_blocks_when_viewer_stalls` and `test_scalars_survive_frame_drops` intentionally end without `on_eval_end`: from Task 3 on, shutting down a sink with nonzero drop counters warns, and wrapping cleanup in `pytest.warns` before Task 3 exists would fail. A parked daemon worker with no producers is harmless.
- Task 3 rewrites the middle of the wedge test (the disowning `on_eval_end` starts warning and resetting counters there); write it bare here exactly as shown — Task 3 Step 3 has the replacement.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_rerun_sink.py -v -k "stall or survive or flush or restarts or emit_failure or queue_size or trial_end or wedged"`
Expected: FAILs/ERRORs (`RerunSink.__init__` has no `queue_size`; `sink.flush`/`sink._queue`/`sink._cond`/`sink._state` do not exist).

- [ ] **Step 3: Implement the worker**

In `src/inspect_robots/logging/rerun_sink.py` add imports:

```python
import threading
from collections import deque
```

Extend `__init__` (final signature; replaces Task 1's):

```python
    def __init__(
        self,
        recording_path: str | None = None,
        *,
        application_id: str = "inspect_robots",
        spawn: bool = False,
        jpeg_quality: int | None = 75,
        queue_size: int = 64,
        flush_timeout: float = 10.0,
    ):
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        self.recording_path = recording_path
        self.application_id = application_id
        self.spawn = spawn
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
```

Add the per-generation worker state next to `_StepPayload` (module level):

```python
@dataclasses.dataclass
class _WorkerState:
    """Per-worker-generation state, so an abandoned worker cannot corrupt its successor."""

    stop: threading.Event
    inflight: int = 0
```

Add the queue/worker machinery:

```python
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
                payload = dataclasses.replace(payload, images={})
                self._dropped_frames += 1
            if len(self._queue) >= self.queue_size:
                evicted = self._queue.popleft()
                self._dropped_steps += 1
                if evicted.images:
                    self._dropped_frames += 1
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
                lambda: not self._queue
                and (self._state is None or self._state.inflight == 0),
                timeout,
            )

    def _shutdown(self) -> None:
        worker, state = self._worker, self._state
        if worker is None or state is None:
            return
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
                self._dropped_frames += sum(1 for p in self._queue if p.images)
                self._queue.clear()
        self._emit_warned = False
```

Note the exit condition in `_worker_loop`: a *current* worker with its stop set drains the queue before exiting (`_shutdown` flushed first, so this is normally instant), while a *disowned* worker (`self._state is not state`) exits immediately even with items queued — the backlog belongs to its successor (or was cleared). `flush()` reads only the current generation's `inflight`, so a zombie finishing its last emit cannot flip the result.

Change the tail of `log_step` from direct emit to enqueue (only the last line changes versus Task 1):

```python
        self._ensure_worker()
        self._enqueue(payload)
```

Change `on_trial_end` and `on_eval_end` to drain/stop the worker:

```python
    def on_trial_end(self, record: TrialRecord) -> None:
        """Drain queued events between trials, bounding loss if the eval aborts mid-run.

        ``eval()`` does not guarantee ``on_eval_end`` on every failure path
        (scorer/hook exceptions, Ctrl-C), so trial boundaries are the flush
        points that cap tail loss at one trial. Blocking here is bounded by
        ``flush_timeout`` and happens between trials, never inside the
        control-rate loop.
        """
        self.flush(timeout=self.flush_timeout)

    def on_eval_end(self, log: EvalLog) -> None:
        """Flush queued data and stop the worker; waits at most ~2x ``flush_timeout``."""
        self._shutdown()
```

- [ ] **Step 4: Fix the two pre-existing tests that now race**

In `tests/test_coverage_completion.py`, the direct `log_step` call after the eval (around line 501) restarts the worker; its branches are only deterministically covered after a flush. Immediately after that `sink.log_step(...)` call block, add:

```python
    assert sink.flush(timeout=5.0)
    sink.on_eval_end(None)  # type: ignore[arg-type]
```

(The eval-driven assertions earlier in that test are already safe: `eval()` calls `on_eval_end`, which flushes before returning.)

In `tests/test_rerun_sink.py`, the four Task 1 tests call `log_step` and assert immediately; insert `assert sink.flush(timeout=5.0)` between the `_log_one(sink)` call and the assertions in each of them.

- [ ] **Step 5: Run the full suite and gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all pass, coverage 100%. If a coverage report shows worker-loop lines as uncovered, check that assertions flush before the test returns; coverage.py traces spawned threads by default.

- [ ] **Step 6: Commit**

```bash
git add src/inspect_robots/logging/rerun_sink.py tests/test_rerun_sink.py tests/test_coverage_completion.py
git commit -m "feat(rerun): emit on a worker thread; never block the control loop"
```

---

### Task 3: Backpressure observability (drop report + stalled-shutdown warning)

Silent drops would read as "covered everything" when the viz is actually sparse; report them once per eval.

**Files:**
- Modify: `src/inspect_robots/logging/rerun_sink.py` (only `_shutdown`)
- Test: `tests/test_rerun_sink.py`

**Interfaces:**
- Consumes: `_shutdown`, `flush`, drop counters from Task 2.
- Produces: warning behavior only; no new names.

- [ ] **Step 1: Write the failing test and update the wedge test**

Append to `tests/test_rerun_sink.py`:

```python
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
```

In `test_wedged_worker_is_disowned_and_backlog_dropped`, replace the section from the disowning `on_eval_end` through `_log_one(sink, 2)` with (the disowning shutdown now emits both warnings and resets the counters):

```python
        with pytest.warns(RuntimeWarning) as record:
            sink.on_eval_end(None)  # type: ignore[arg-type]  # flush+join time out; A disowned
        messages = [str(w.message) for w in record]
        assert any("stalled" in m for m in messages)
        assert any("dropped 1 camera frame(s) and 1 full step(s)" in m for m in messages)
        assert sink._worker is None and sink._state is None
        assert sink._dropped_steps == 0  # reported and reset
        _log_one(sink, 2)  # starts worker B
```

The test's final `sink.on_eval_end(None)` stays bare: worker B shuts down cleanly and the counters were already reported and reset, so it must not warn (that silence is itself part of the contract under test).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rerun_sink.py -v -k "reports_dropped or wedged"`
Expected: `test_eval_end_reports_dropped_data` FAILS with `DID NOT WARN`; the wedge test FAILS on the `pytest.warns` block (no warning emitted yet).

- [ ] **Step 3: Implement the warnings in `_shutdown`**

Append to the end of `_shutdown` (after the existing `self._emit_warned = False` line):

```python
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
```

(`wedged` already exists in `_shutdown` from Task 2. A worker that was slow but finished during `join` is not `wedged` and lost nothing — the drain-then-exit loop emptied the queue — so no stalled warning fires for it.)

- [ ] **Step 4: Run the full suite and gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all pass, 100% coverage. Both warning branches are exercised by the two new tests; the no-warning path is exercised by every clean-shutdown test from Task 2.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/logging/rerun_sink.py tests/test_rerun_sink.py
git commit -m "feat(rerun): report dropped frames and stalled viewer at eval end"
```

---

### Task 4: Documentation

**Files:**
- Modify: `src/inspect_robots/logging/rerun_sink.py` (module docstring only)
- Modify: `README.md` (the "Rerun visualization" feature bullet, around line 174)
- Modify: `src/inspect_robots/CLAUDE.md` (the `logging/` row in the module map)

- [ ] **Step 1: Update the module docstring**

Insert after the first paragraph of the `rerun_sink.py` module docstring:

```
Emission happens on a daemon worker thread: ``log_step`` snapshots the
transition and enqueues it, so a slow or stalled viewer connection can never
block the control-rate rollout loop. Under backpressure the sink degrades
visualization instead of delaying control: camera frames are dropped first
(scalar plots stay complete), then whole steps, and the drop counts are
reported as a ``RuntimeWarning`` when the eval ends. The queue is drained at
every trial boundary (bounded by ``flush_timeout``), so an eval that aborts
mid-run loses at most the current trial's queued tail. Camera frames are
JPEG-compressed by default (``jpeg_quality=75``); pass ``jpeg_quality=None``
for lossless raw frames. All Rerun SDK calls after ``init``/``save`` happen on
the worker because the SDK's timeline state is thread-local; worker state is
generation-scoped so a worker wedged in a blocked SDK call is disowned at
shutdown and can never double-consume after a restart.
```

- [ ] **Step 2: Update README and the package CLAUDE.md**

README feature bullet (keep the repo writing-style rules: no em dashes, no mid-sentence bold): extend the existing "Rerun visualization." bullet with one sentence:

```
Logging is non-blocking: a slow viewer connection drops frames instead of stalling the robot control loop, and camera streams are JPEG-compressed by default.
```

`src/inspect_robots/CLAUDE.md` module map, `logging/` row: change to

```
| `logging/` | `LogSink` protocol, `JsonLogSink` (atomic), optional `RerunSink` (non-blocking worker thread; drops frames, never delays control) |
```

- [ ] **Step 3: Run all gates one final time**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: all pass, 100% coverage.

- [ ] **Step 4: Commit**

```bash
git add src/inspect_robots/logging/rerun_sink.py README.md src/inspect_robots/CLAUDE.md
git commit -m "docs(rerun): document non-blocking sink and drop policy"
```

---

## Explicit non-goals (YAGNI)

- No CLI flags for `jpeg_quality`/`queue_size`; the defaults are chosen to be correct for the live-viewer path and callers constructing `RerunSink` directly can pass kwargs.
- No `rr.send_columns` scalar batching, no video-codec streams, no multi-producer support (the rollout loop is the single producer by design).
- No change to `on_eval_start` failure handling, entity naming, or the `_set_step`/`_scalar` SDK-version shims.
- No change to `eval.py` (wrapping `bus.on_eval_end` in `try/finally`) and no `atexit`/`weakref.finalize` hook. The accepted residual window: if the process dies mid-trial (scorer/hook exception, Ctrl-C), up to the current trial's queued viz events are lost. The `on_trial_end` flush bounds it to one trial, and the `EvalLog` itself is unaffected (the JSON sink is synchronous). Revisit only if partial-trial viz proves valuable in practice.
