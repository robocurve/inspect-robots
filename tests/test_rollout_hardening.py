"""Rollout hardening: error taxonomy, transcript events, approver, FrameStore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roboinspect import eval
from roboinspect.approver import AutoApprover, ClampApprover
from roboinspect.controller import DefaultController
from roboinspect.errors import EmbodimentFault, PolicyError, SafetyAbort
from roboinspect.frames import FrameStore
from roboinspect.logging.sink import NullSink
from roboinspect.mock import CubePickEmbodiment, ScriptedPolicy
from roboinspect.policy import PolicyConfig, PolicyInfo
from roboinspect.rollout import derive_seed, rollout
from roboinspect.scene import Scene
from roboinspect.scorer import success_at_end
from roboinspect.spaces import ActionSemantics, Box
from roboinspect.task import Task
from roboinspect.types import Action, ActionChunk, Observation

_SCENE = Scene(id="s", instruction="reach", init_seed=0)
_BOX = Box(shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"))


def _run(policy: object, embodiment: object, *, approver: object = None, frame_store=None):  # type: ignore[no-untyped-def]
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
