"""Rollout hardening: error taxonomy, transcript events, approver, FrameStore."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.approver import Approver, AutoApprover, ClampApprover
from inspect_robots.controller import DefaultController
from inspect_robots.errors import EmbodimentFault, PolicyError, SafetyAbort, _CancelledTrial
from inspect_robots.frames import FrameStore
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.rollout import TrialRecord, derive_seed, rollout
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation

_SCENE = Scene(id="s", instruction="reach", init_seed=0)
_BOX = Box(shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"))


def _run(
    policy: object,
    embodiment: object,
    *,
    approver: Approver | None = None,
    frame_store: FrameStore | None = None,
) -> TrialRecord:
    return rollout(
        policy,  # type: ignore[arg-type]
        embodiment,  # type: ignore[arg-type]
        _SCENE,
        max_steps=40,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=approver or AutoApprover(),
        sink=NullSink(),
        frame_store=frame_store,
    )


class _BoomPolicy:
    def __init__(self) -> None:
        self.info = PolicyInfo(name="boom", action_space=_BOX)
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        raise RuntimeError("inference exploded")


class _FaultyEmbodiment(CubePickEmbodiment):
    def step(self, action: Action):  # type: ignore[no-untyped-def]
        raise RuntimeError("motor stalled")


class _VetoApprover:
    def review(self, action: Action, store: dict[str, object]) -> Action:
        raise SafetyAbort("operator pressed e-stop")


def test_policy_exception_wrapped_as_policy_error() -> None:
    with pytest.raises(PolicyError, match="inference exploded"):
        _run(_BoomPolicy(), CubePickEmbodiment())


def test_embodiment_exception_wrapped_as_fault() -> None:
    with pytest.raises(EmbodimentFault, match="motor stalled"):
        _run(ScriptedPolicy(), _FaultyEmbodiment())


def test_keyboard_interrupt_from_step_carries_partial_record_and_transcript() -> None:
    original = KeyboardInterrupt("stop now")

    class _InterruptingEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def step(self, action: Action):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 2:
                raise original
            return super().step(action)

    class _AuditedPolicy(ScriptedPolicy):
        def transcript(self) -> object:
            return {"inferences": self.num_inferences}

    with pytest.raises(_CancelledTrial) as excinfo:
        _run(_AuditedPolicy(chunk_size=1), _InterruptingEmbodiment())

    record = excinfo.value.record
    assert record.status == "cancelled"
    assert len(record.steps) == 1
    assert record.policy_transcript == {"inferences": 2}
    assert record.events[-1].kind == "error"
    assert record.events[-1].t == 1
    assert excinfo.value.__cause__ is original


def test_safety_abort_propagates() -> None:
    with pytest.raises(SafetyAbort, match="e-stop"):
        _run(ScriptedPolicy(), CubePickEmbodiment(), approver=_VetoApprover())


def test_transcript_records_events() -> None:
    record = _run(ScriptedPolicy(), CubePickEmbodiment())
    kinds = [e.kind for e in record.events]
    assert kinds[0] == "reset"
    assert "inference" in kinds
    assert "step" in kinds
    # The final step event carries the termination reason.
    last_step = [e for e in record.events if e.kind == "step"][-1]
    assert last_step.data["terminated"] is True
    assert last_step.data["reason"] == "success"


def test_clamp_approver_bounds_action() -> None:
    space = Box(shape=(2,), low=np.array([-0.05, -0.05]), high=np.array([0.05, 0.05]))
    approver = ClampApprover(space)
    out = approver.review(Action(data=np.array([0.5, -0.5])), {})
    assert np.allclose(out.data, [0.05, -0.05])
    assert out.meta.get("clamped") is True


def test_clamp_approver_nan_raises_safety_abort() -> None:
    # A NaN has no meaningful clamp; it must never reach hardware.
    space = Box(shape=(2,), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
    approver = ClampApprover(space)
    with pytest.raises(SafetyAbort, match="NaN"):
        approver.review(Action(data=np.array([0.0, float("nan")])), {})


def test_clamp_approver_nan_aborts_even_without_bounds() -> None:
    approver = ClampApprover(Box(shape=(2,)))  # no low/high
    with pytest.raises(SafetyAbort, match="NaN"):
        approver.review(Action(data=np.array([float("nan"), 0.0])), {})


def test_clamp_approver_inf_clamps_without_abort() -> None:
    # ±inf is out-of-range, not poisonous: it clamps to the finite bound.
    space = Box(shape=(2,), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
    approver = ClampApprover(space)
    out = approver.review(Action(data=np.array([float("inf"), float("-inf")])), {})
    assert np.allclose(out.data, [1.0, -1.0])
    assert out.meta.get("clamped") is True


def test_clamp_approver_low_only_bound() -> None:
    approver = ClampApprover(Box(shape=(2,), low=np.array([0.0, 0.0])))
    out = approver.review(Action(data=np.array([-0.5, 0.5])), {})
    assert np.allclose(out.data, [0.0, 0.5])
    assert out.meta.get("clamped") is True
    in_bounds = Action(data=np.array([0.5, 0.5]))
    assert approver.review(in_bounds, {}) is in_bounds  # identity pass-through


def test_clamp_approver_high_only_bound() -> None:
    approver = ClampApprover(Box(shape=(2,), high=np.array([1.0, 1.0])))
    out = approver.review(Action(data=np.array([2.0, 0.5])), {})
    assert np.allclose(out.data, [1.0, 0.5])
    assert out.meta.get("clamped") is True
    # inf on the unbounded side has nothing to clamp against: pass-through.
    unbounded_side = Action(data=np.array([float("-inf"), 0.0]))
    assert approver.review(unbounded_side, {}) is unbounded_side


class _StopPolicy:
    """Emits one hold-still action carrying the given meta every inference."""

    def __init__(self, meta: dict[str, object]):
        self.info = PolicyInfo(name="stopper", action_space=_BOX)
        self.config = PolicyConfig()
        self._meta = meta

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(2), meta=dict(self._meta))])


def test_policy_request_stop_truncates_with_reason() -> None:
    policy = _StopPolicy({"request_stop": True, "stop_reason": "done"})
    record = _run(policy, CubePickEmbodiment())
    assert record.truncated is True
    assert record.terminated is False
    assert record.termination_reason == "done"
    assert len(record.steps) == 1  # the flagged action still executed, once


def test_policy_request_stop_default_reason() -> None:
    record = _run(_StopPolicy({"request_stop": True}), CubePickEmbodiment())
    assert record.termination_reason == "policy_stop"


def test_request_stop_survives_approver_rewrite() -> None:
    class _MetaStrippingApprover:
        def review(self, action: Action, store: dict[str, object]) -> Action:
            return replace(action, data=np.asarray(action.data) * 0.5, meta={})

    policy = _StopPolicy({"request_stop": True, "stop_reason": "done"})
    record = _run(policy, CubePickEmbodiment(), approver=_MetaStrippingApprover())
    assert record.truncated is True
    assert record.termination_reason == "done"


def test_embodiment_termination_wins_over_request_stop() -> None:
    class _InstantSuccessEmbodiment(CubePickEmbodiment):
        def step(self, action: Action):  # type: ignore[no-untyped-def]
            result = super().step(action)
            return replace(result, terminated=True, termination_reason="success")

    policy = _StopPolicy({"request_stop": True, "stop_reason": "done"})
    record = _run(policy, _InstantSuccessEmbodiment())
    assert record.terminated is True
    assert record.truncated is False
    assert record.termination_reason == "success"


def test_rollout_records_delta_clamped_approval_detail() -> None:
    from inspect_robots.approver import DeltaLimitApprover

    emb = CubePickEmbodiment()
    # CubePick's scripted steps are 0.1-magnitude deltas; a tighter explicit
    # limit forces a clamp so the approval event carries the new detail.
    approver = DeltaLimitApprover(emb.info.action_space, max_delta=0.05)
    record = _run(ScriptedPolicy(), emb, approver=approver)
    approvals = [e for e in record.events if e.kind == "approval"]
    assert approvals
    assert approvals[0].data["detail"] == "delta_clamped"


def test_frame_store_sanitizes_without_collisions(tmp_path: Path) -> None:
    store = FrameStore(str(tmp_path / "frames"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    a = store.put("a/b", 0, "cam", img)
    b = store.put("a-b", 0, "cam", img)
    assert a.path != b.path  # sanitization must not introduce collisions
    assert Path(a.path).exists() and Path(b.path).exists()


def test_frame_store_streams_to_disk(tmp_path: Path) -> None:
    store = FrameStore(str(tmp_path / "frames"))
    record = _run(ScriptedPolicy(), CubePickEmbodiment(), frame_store=store)
    assert store.count > 0
    first = record.steps[0]
    assert not first.observation.images  # images stripped from the record
    assert first.image_refs is not None and "top" in first.image_refs
    loaded = first.image_refs["top"].load()
    assert loaded.shape == (32, 32, 3)


def test_per_trial_seed_varies_by_epoch() -> None:
    s0 = derive_seed(7, 3, 0)
    s1 = derive_seed(7, 3, 1)
    assert s0 != s1
    # deterministic
    assert derive_seed(7, 3, 0) == s0


def test_derive_seed_distinguishes_none_from_zero() -> None:
    # "unseeded" must not silently alias seed=0.
    assert derive_seed(None, 3, 0) != derive_seed(0, 3, 0)
    assert derive_seed(7, None, 0) != derive_seed(7, 0, 0)


# --------------------------------------------------------------------------- #
# Partial-record preservation: every in-trial error carries the forensic record.
# --------------------------------------------------------------------------- #
class _BoomLaterPolicy:
    """Delivers one good chunk, then explodes on the second inference."""

    def __init__(self) -> None:
        self.info = PolicyInfo(name="boom-later", action_space=_BOX)
        self.config = PolicyConfig()
        self._calls = 0

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("inference exploded later")
        return ActionChunk(actions=[Action(data=np.zeros(2)) for _ in range(4)])


def test_policy_error_carries_partial_record() -> None:
    with pytest.raises(PolicyError, match="exploded later") as excinfo:
        _run(_BoomLaterPolicy(), CubePickEmbodiment())
    rec = excinfo.value.record
    assert rec is not None
    assert rec.status == "error"
    assert rec.error is not None and "exploded later" in rec.error
    assert len(rec.steps) == 4  # the first chunk executed before the failure
    assert rec.events[-1].kind == "error"


def test_embodiment_fault_carries_partial_record() -> None:
    with pytest.raises(EmbodimentFault, match="motor stalled") as excinfo:
        _run(ScriptedPolicy(), _FaultyEmbodiment())
    rec = excinfo.value.record
    assert rec is not None and rec.status == "error"
    kinds = [e.kind for e in rec.events]
    assert kinds[0] == "reset" and kinds[-1] == "error"


def test_safety_abort_carries_partial_record() -> None:
    with pytest.raises(SafetyAbort, match="e-stop") as excinfo:
        _run(ScriptedPolicy(), CubePickEmbodiment(), approver=_VetoApprover())
    rec = excinfo.value.record
    assert rec is not None and rec.status == "error"


class _CrashingApprover:
    def review(self, action: Action, store: dict[str, object]) -> Action:
        raise ZeroDivisionError("approver crashed")


def test_approver_crash_wrapped_as_safety_abort() -> None:
    # An approver that crashed cannot vouch for safety: treat it as an abort.
    with pytest.raises(SafetyAbort, match="approver crashed") as excinfo:
        _run(ScriptedPolicy(), CubePickEmbodiment(), approver=_CrashingApprover())
    assert excinfo.value.record is not None


# --------------------------------------------------------------------------- #
# Reset failures are wrapped/attributed like in-loop failures.
# --------------------------------------------------------------------------- #
class _ResetBoomPolicy(_BoomLaterPolicy):
    def reset(self, scene: Scene) -> None:
        raise RuntimeError("policy reset failed")


class _TypedResetPolicy(_BoomLaterPolicy):
    def reset(self, scene: Scene) -> None:
        raise PolicyError("typed reset failure")


class _ResetFaultEmbodiment(CubePickEmbodiment):
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        raise RuntimeError("homing failed")


class _TypedResetEmbodiment(CubePickEmbodiment):
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        raise EmbodimentFault("e-stop during reset")


def test_policy_reset_exception_wrapped() -> None:
    with pytest.raises(PolicyError, match="policy reset failed") as excinfo:
        _run(_ResetBoomPolicy(), CubePickEmbodiment())
    assert excinfo.value.record is not None


def test_policy_reset_typed_error_propagates() -> None:
    with pytest.raises(PolicyError, match="typed reset failure") as excinfo:
        _run(_TypedResetPolicy(), CubePickEmbodiment())
    assert excinfo.value.record is not None


def test_embodiment_reset_exception_wrapped() -> None:
    with pytest.raises(EmbodimentFault, match="homing failed") as excinfo:
        _run(ScriptedPolicy(), _ResetFaultEmbodiment())
    assert excinfo.value.record is not None


def test_embodiment_reset_typed_error_propagates() -> None:
    with pytest.raises(EmbodimentFault, match="e-stop during reset") as excinfo:
        _run(ScriptedPolicy(), _TypedResetEmbodiment())
    assert excinfo.value.record is not None


# --------------------------------------------------------------------------- #
# Runtime action-shape validation: a malformed action is a PolicyError, not a
# halting EmbodimentFault misattributed to the robot.
# --------------------------------------------------------------------------- #
class _WrongDimPolicy:
    def __init__(self) -> None:
        self.info = PolicyInfo(name="wrong-dim", action_space=_BOX)  # declares 2-D
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(3))])  # emits 3-D


def test_wrong_dim_action_attributed_to_policy() -> None:
    with pytest.raises(PolicyError, match="expects 2-D") as excinfo:
        _run(_WrongDimPolicy(), CubePickEmbodiment())
    rec = excinfo.value.record
    assert rec is not None and rec.status == "error"


# --------------------------------------------------------------------------- #
# Approval events: a modified action is recorded in the transcript.
# --------------------------------------------------------------------------- #
def test_clamp_approver_records_approval_event() -> None:
    space = Box(shape=(2,), low=np.array([-0.05, -0.05]), high=np.array([0.05, 0.05]))
    record = _run(ScriptedPolicy(), CubePickEmbodiment(), approver=ClampApprover(space))
    approvals = [e for e in record.events if e.kind == "approval"]
    assert approvals
    assert approvals[0].data["modified"] is True
    assert approvals[0].data["detail"] == "clamped"


class _HalvingApprover:
    """Modifies the action without setting the ``clamped`` meta flag."""

    def review(self, action: Action, store: dict[str, object]) -> Action:
        return replace(action, data=np.asarray(action.data, dtype=np.float64) * 0.5)


def test_modified_action_records_approval_event_without_detail() -> None:
    record = _run(ScriptedPolicy(), CubePickEmbodiment(), approver=_HalvingApprover())
    approvals = [e for e in record.events if e.kind == "approval"]
    assert approvals
    assert approvals[0].data["detail"] is None


def test_fail_on_error_proportion_halts(tmp_path: Path) -> None:
    task = Task(
        name="t",
        scenes=[Scene(id=f"s{i}", instruction="x") for i in range(4)],
        scorer=success_at_end(),
        max_steps=20,
    )
    # Every trial raises -> proportion 1.0 >= 0.5 threshold -> eval status error.
    logs = eval(task, _BoomPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), fail_on_error=0.5)
    assert logs[0].status == "error"


# --------------------------------------------------------------------------- #
# Policy transcripts: best-effort capture on every post-reset exit path.
# --------------------------------------------------------------------------- #
class _TranscriptPolicy(_StopPolicy):
    """Stops after one action and exposes a configurable audit payload."""

    def __init__(self, transcript: object) -> None:
        super().__init__({"request_stop": True})
        self._transcript = transcript

    def transcript(self) -> object:
        return self._transcript


def test_policy_transcript_captured_on_success() -> None:
    raw = [{"role": "assistant", "content": "done"}]
    record = _run(_TranscriptPolicy(raw), CubePickEmbodiment())
    assert record.policy_transcript == raw
    assert record.policy_transcript is not raw


def test_policy_transcript_captured_on_partial_policy_error() -> None:
    class _TranscriptBoomLater(_BoomLaterPolicy):
        def transcript(self) -> object:
            return {"calls": self._calls}

    with pytest.raises(PolicyError) as excinfo:
        _run(_TranscriptBoomLater(), CubePickEmbodiment())
    assert excinfo.value.record is not None
    assert excinfo.value.record.policy_transcript == {"calls": 2}


def test_policy_transcript_not_collected_when_policy_reset_fails() -> None:
    class _StaleTranscriptPolicy(_ResetBoomPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.transcript_calls = 0

        def transcript(self) -> object:
            self.transcript_calls += 1
            return {"stale": True}

    policy = _StaleTranscriptPolicy()
    with pytest.raises(PolicyError) as excinfo:
        _run(policy, CubePickEmbodiment())
    assert policy.transcript_calls == 0
    assert excinfo.value.record is not None
    assert excinfo.value.record.policy_transcript is None


def test_raising_policy_transcript_becomes_error_marker() -> None:
    class _RaisingTranscriptPolicy(_StopPolicy):
        def transcript(self) -> object:
            raise RuntimeError("audit exploded")

    record = _run(_RaisingTranscriptPolicy({"request_stop": True}), CubePickEmbodiment())
    assert record.policy_transcript == {"transcript_error": "RuntimeError: audit exploded"}


def test_unprintable_transcript_exception_uses_type_only_marker() -> None:
    class _UnprintableError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("cannot format")

    class _RaisingTranscriptPolicy(_StopPolicy):
        def transcript(self) -> object:
            raise _UnprintableError

    record = _run(_RaisingTranscriptPolicy({"request_stop": True}), CubePickEmbodiment())
    assert record.policy_transcript == {"transcript_error": "_UnprintableError"}


def test_hookless_policy_records_none_transcript() -> None:
    record = _run(_StopPolicy({"request_stop": True}), CubePickEmbodiment())
    assert record.policy_transcript is None


def test_policy_base_default_transcript_is_none() -> None:
    class _MinimalPolicy(PolicyBase):
        info = PolicyInfo(name="minimal", action_space=_BOX)

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    policy = _MinimalPolicy()
    assert policy.transcript() is None
    assert _run(policy, CubePickEmbodiment()).policy_transcript is None


def test_policy_transcript_normalizes_numpy_leaves_to_strings() -> None:
    array = np.arange(10_000)
    raw = {"scalar": np.float32(1.25), "array": array}
    record = _run(_TranscriptPolicy(raw), CubePickEmbodiment())
    assert record.policy_transcript == {"scalar": "1.25", "array": str(array)}
    assert "..." in record.policy_transcript["array"]


def test_circular_policy_transcript_becomes_error_marker() -> None:
    circular: list[object] = []
    circular.append(circular)
    record = _run(_TranscriptPolicy(circular), CubePickEmbodiment())
    assert record.policy_transcript == {
        "transcript_error": "ValueError: Circular reference detected"
    }


def test_oversized_policy_transcript_becomes_dropped_marker() -> None:
    payload = "x" * (2 * 1024 * 1024)
    record = _run(_TranscriptPolicy(payload), CubePickEmbodiment())
    assert record.policy_transcript == {
        "transcript_dropped": True,
        "bytes": 2 * 1024 * 1024 + 2,
        "note": "exceeds inline limit; policies must not embed binary data",
    }
