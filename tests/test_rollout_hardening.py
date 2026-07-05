"""Rollout hardening: error taxonomy, transcript events, approver, FrameStore."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import eval
from inspect_robots.approver import Approver, AutoApprover, ClampApprover
from inspect_robots.controller import DefaultController
from inspect_robots.errors import EmbodimentFault, PolicyError, SafetyAbort
from inspect_robots.frames import FrameStore
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
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
