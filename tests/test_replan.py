"""Replanning on observation: triggers that cut a chunk short so the policy can look again."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.action_manager import (
    _APPROVALS_KEY,
    _CHUNK_START_KEY,
    _LAST_ACTION_KEY,
    DefaultActionManager,
    GripperSlip,
    ReplanTrigger,
    UnreachableTarget,
)
from inspect_robots.guard import ClampGuard
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation


class _Chunker:
    """Returns five-action chunks; remembers the observations it was called with."""

    def __init__(self, value: float = 0.0) -> None:
        self.info = PolicyInfo(name="chunker", action_space=CubePickEmbodiment().info.action_space)
        self.config = PolicyConfig()
        self.value = value
        self.seen: list[Observation] = []

    def reset(self, scene: Scene) -> None:
        self.seen.clear()

    def act(self, observation: Observation) -> ActionChunk:
        self.seen.append(observation)
        return ActionChunk(actions=[Action(data=np.full(2, self.value)) for _ in range(5)])


class _FireAt:
    """Fires once, at one step."""

    name = "test_trigger"

    def __init__(self, step: int) -> None:
        self.step = step

    def should_replan(self, observation: Observation, t: int, store: dict[str, Any]) -> str | None:
        return "something changed" if t == self.step else None


def _task(max_steps: int = 10) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


def test_a_firing_trigger_drops_the_chunk_and_records_why(tmp_path: Path) -> None:
    policy = _Chunker()
    trigger = _FireAt(2)
    assert isinstance(trigger, ReplanTrigger)
    (log,) = ir_eval(
        _task(),
        policy,
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        action_manager=DefaultActionManager(triggers=[trigger]),
    )
    (events,) = log.samples[0].events
    inference_steps = [e["t"] for e in events if e["kind"] == "inference"]
    assert inference_steps == [0, 2, 7]  # replanned at 2 instead of 5
    (replan,) = [e for e in events if e["kind"] == "replan"]
    assert replan == {
        "kind": "replan",
        "t": 2,
        "data": {"trigger": "test_trigger", "reason": "something changed"},
    }
    assert policy.seen[1].extra["replan"] == {
        "trigger": "test_trigger",
        "reason": "something changed",
    }
    assert "replan" not in policy.seen[0].extra


def test_triggers_are_not_asked_at_a_chunk_boundary() -> None:
    asked: list[int] = []

    class _Recording:
        name = "recording"

        def should_replan(self, observation: Observation, t: int, store: Any) -> str | None:
            asked.append(t)
            return None

    manager = DefaultActionManager(replan_interval=2, triggers=[_Recording()])
    store: dict[str, Any] = {}
    for t in range(4):
        manager.next_action(_Chunker(), Observation(), t, store)
    assert asked == [1, 3]


def test_triggers_must_offer_should_replan() -> None:
    with pytest.raises(ValueError, match="has no should_replan method"):
        DefaultActionManager(triggers=[object()])  # type: ignore[list-item]


# -- unreachable target ------------------------------------------------------


def test_unreachable_target_fires_on_a_bounds_clamp_in_the_current_chunk() -> None:
    trigger = UnreachableTarget()
    store: dict[str, Any] = {_CHUNK_START_KEY: 5, _APPROVALS_KEY: [{"t": 3, "detail": "clamped"}]}
    assert trigger.should_replan(Observation(), 6, store) is None  # before this chunk
    store[_APPROVALS_KEY].append({"t": 6, "detail": "delta_clamped"})
    assert trigger.should_replan(Observation(), 7, store) is None  # a speed limit only
    store[_APPROVALS_KEY].append({"t": 7, "detail": None})
    assert trigger.should_replan(Observation(), 8, store) is None
    store[_APPROVALS_KEY].append({"t": 8, "detail": "clamped, delta_clamped"})
    reason = trigger.should_replan(Observation(), 9, store)
    assert reason is not None and "step 8" in reason and "out of reach" in reason


def test_unreachable_target_replans_a_policy_aiming_outside_the_bounds(tmp_path: Path) -> None:
    embodiment = CubePickEmbodiment()
    policy = _Chunker(value=5.0)  # far outside CubePick's action bounds
    (log,) = ir_eval(
        _task(max_steps=6),
        policy,
        embodiment,
        log_dir=str(tmp_path),
        action_manager=DefaultActionManager(triggers=[UnreachableTarget()]),
        guard=ClampGuard(embodiment.info.action_space),
    )
    (events,) = log.samples[0].events
    replans = [e for e in events if e["kind"] == "replan"]
    assert [e["t"] for e in replans] == [1, 2, 3, 4, 5]
    assert all(e["data"]["trigger"] == "unreachable_target" for e in replans)


# -- gripper slip ------------------------------------------------------------


def _slip_store(command: float) -> dict[str, Any]:
    return {_LAST_ACTION_KEY: Action(data=np.array([0.0, command]))}


def _gripper(value: float) -> Observation:
    return Observation(state={"joint_pos": np.array([0.0, value])})


def test_gripper_slip_fires_when_a_held_gripper_closes_on_nothing() -> None:
    trigger = GripperSlip("joint_pos", 1, threshold=0.05, settle_steps=2)
    store = _slip_store(command=0.0)
    readings = [0.5, 0.45, 0.30, 0.30, 0.29, 0.20]
    results = [trigger.should_replan(_gripper(r), t, store) for t, r in enumerate(readings)]
    # Steps 0-2: a new command, then settling; step 2 captures the 0.30 baseline.
    assert results[:5] == [None, None, None, None, None]
    assert results[5] is not None and "gripper moved 0.100 toward its command of 0" in results[5]
    # The baseline moves with the slip, so it reports once.
    assert trigger.should_replan(_gripper(0.20), 6, store) is None


def test_a_new_command_restarts_the_settling() -> None:
    trigger = GripperSlip("joint_pos", 1, settle_steps=0)
    store = _slip_store(command=0.0)
    assert trigger.should_replan(_gripper(0.5), 0, store) is None
    assert trigger.should_replan(_gripper(0.5), 1, store) is None  # baseline
    store[_LAST_ACTION_KEY] = Action(data=np.array([0.0, 1.0]))
    assert trigger.should_replan(_gripper(0.2), 2, store) is None  # opening: a new command
    assert trigger.should_replan(_gripper(0.9), 3, store) is None


def test_gripper_slip_waits_for_an_action_and_a_reading() -> None:
    trigger = GripperSlip("joint_pos", 1)
    assert trigger.should_replan(_gripper(0.5), 0, {}) is None
    assert trigger.should_replan(Observation(), 0, _slip_store(0.0)) is None


def test_gripper_slip_triggers_are_built_from_gripper_labels() -> None:
    space = Box(
        shape=(3,),
        semantics=ActionSemantics("joint_pos", dim_labels=("j0", "left_gripper", "Gripper_R")),
    )
    triggers = GripperSlip.for_embodiment(space, "joint_pos", threshold=0.1)
    assert [(t.index, t.label, t.threshold) for t in triggers] == [
        (1, "left_gripper", 0.1),
        (2, "Gripper_R", 0.1),
    ]
    assert GripperSlip.for_embodiment(Box(shape=(2,)), "joint_pos") == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"settle_steps": -1}, "settle_steps"),
        ({"settle_steps": True}, "settle_steps"),
    ],
)
def test_gripper_slip_validates_its_settings(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GripperSlip("joint_pos", 1, **kwargs)


def test_gripper_slip_follows_the_executed_command_not_the_requested_one() -> None:
    from inspect_robots.guard import DeltaLimitGuard
    from inspect_robots.logging import NullSink
    from inspect_robots.rollout import rollout

    space = Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=ActionSemantics("joint_pos", dim_labels=("joint", "gripper")),
    )

    class _Tracker(CubePickEmbodiment):
        """Follows every executed command exactly: a gripper that never slips."""

        def __init__(self) -> None:
            super().__init__()
            self.info = self.info.__class__(
                **{**self.info.__dict__, "action_space": space, "name": "tracker"}
            )
            self.q = np.array([0.0, 1.0])

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self.q = np.array([0.0, 1.0])
            return Observation(state={"joint_pos": self.q.copy()})

        def step(self, action: Action) -> Any:
            from inspect_robots.types import StepResult

            self.q = np.asarray(action.data, dtype=np.float64).copy()
            return StepResult(observation=Observation(state={"joint_pos": self.q.copy()}))

    class _Closer:
        """Holds the gripper open once (the limiter's reference), then commands it shut."""

        info = PolicyInfo(name="closer", action_space=space)
        config = PolicyConfig()

        def __init__(self) -> None:
            self.calls = 0

        def reset(self, scene: Scene) -> None:
            self.calls = 0

        def act(self, observation: Observation) -> ActionChunk:
            self.calls += 1
            if self.calls == 1:
                return ActionChunk(actions=[Action(data=np.array([0.0, 1.0]))])
            return ActionChunk(actions=[Action(data=np.array([0.0, 0.0]))] * 8)

    trigger = GripperSlip("joint_pos", 1)
    record = rollout(
        _Closer(),
        _Tracker(),
        Scene(id="s", instruction="close"),
        max_steps=30,
        seed=0,
        epoch=0,
        action_manager=DefaultActionManager(triggers=[trigger]),
        guard=DeltaLimitGuard(space, max_delta=0.04),
        sink=NullSink(),
    )
    # The guard walks the gripper shut 0.04 per step; tracking that is not a slip.
    assert [event for event in record.events if event.kind == "replan"] == []
    assert record.steps[-1].action.data[1] == pytest.approx(0.0)


def test_gripper_slip_still_fires_behind_a_smoothing_manager() -> None:
    from inspect_robots.action_manager import SmoothingActionManager
    from inspect_robots.guard import PassThroughGuard
    from inspect_robots.logging import NullSink
    from inspect_robots.rollout import rollout

    space = Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=ActionSemantics("joint_pos", dim_labels=("joint", "gripper")),
    )

    class _HoldsAnObject(CubePickEmbodiment):
        """The gripper stops at 0.4 on an object until step 60, when the object slips out."""

        def __init__(self) -> None:
            super().__init__()
            self.info = self.info.__class__(**{**self.info.__dict__, "action_space": space})
            self.t = 0

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self.t = 0
            return Observation(state={"joint_pos": np.array([0.0, 1.0])})

        def step(self, action: Action) -> Any:
            from inspect_robots.types import StepResult

            self.t += 1
            commanded = float(np.asarray(action.data)[1])
            gripper = commanded if self.t >= 60 else max(commanded, 0.4)
            return StepResult(
                observation=Observation(state={"joint_pos": np.array([0.0, gripper])})
            )

    class _Squeeze:
        info = PolicyInfo(name="squeeze", action_space=space)
        config = PolicyConfig()

        def reset(self, scene: Scene) -> None:
            pass

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.array([0.0, 0.0]))] * 200)

    record = rollout(
        _Squeeze(),
        _HoldsAnObject(),
        Scene(id="s", instruction="hold"),
        max_steps=120,
        seed=0,
        epoch=0,
        action_manager=SmoothingActionManager(
            DefaultActionManager(triggers=[GripperSlip("joint_pos", 1)]), alpha=0.5
        ),
        guard=PassThroughGuard(),
        sink=NullSink(),
    )
    replans = [event for event in record.events if event.kind == "replan"]
    assert replans and 60 <= replans[0].t <= 62
    assert "the grasp may have slipped" in replans[0].data["reason"]


@pytest.mark.parametrize("requested", [0.0, -0.2])
def test_gripper_slip_fires_when_a_clamp_holds_an_out_of_bounds_squeeze(requested: float) -> None:
    from inspect_robots.guard import ClampGuard
    from inspect_robots.logging import NullSink
    from inspect_robots.rollout import rollout

    space = Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=ActionSemantics("joint_pos", dim_labels=("joint", "gripper")),
    )

    class _HoldsAnObject(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = self.info.__class__(**{**self.info.__dict__, "action_space": space})
            self.t = 0

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self.t = 0
            return Observation(state={"joint_pos": np.array([0.0, 1.0])})

        def step(self, action: Action) -> Any:
            from inspect_robots.types import StepResult

            self.t += 1
            commanded = float(np.asarray(action.data)[1])
            gripper = commanded if self.t >= 60 else max(commanded, 0.4)
            return StepResult(
                observation=Observation(state={"joint_pos": np.array([0.0, gripper])})
            )

    class _Squeeze:
        info = PolicyInfo(name="squeeze", action_space=space)
        config = PolicyConfig()

        def reset(self, scene: Scene) -> None:
            pass

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.array([0.0, requested]))] * 200)

    record = rollout(
        _Squeeze(),
        _HoldsAnObject(),
        Scene(id="s", instruction="hold"),
        max_steps=100,
        seed=0,
        epoch=0,
        action_manager=DefaultActionManager(triggers=[GripperSlip("joint_pos", 1)]),
        guard=ClampGuard(space),
        sink=NullSink(),
    )
    replans = [event for event in record.events if event.kind == "replan"]
    assert replans and 60 <= replans[0].t <= 62


def test_gripper_slip_ignores_a_limiter_creeping_below_the_settle_tolerance() -> None:
    from inspect_robots.action_manager import _EXECUTED_ACTION_KEY

    trigger = GripperSlip("joint_pos", 1, threshold=0.05, settle_steps=1)
    store = _slip_store(command=0.0)
    executed = 1.0
    for t in range(60):
        executed = max(0.0, executed - 0.004)  # under 0.1 * threshold per step
        store[_EXECUTED_ACTION_KEY] = Action(data=np.array([0.0, executed]))
        # The gripper tracks the creeping command exactly: nothing slipped.
        assert trigger.should_replan(_gripper(executed), t, store) is None
