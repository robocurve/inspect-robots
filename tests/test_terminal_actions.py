"""Terminal actions: a policy ends a trial through ``ActionChunk.terminal``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inspect_robots import DONE, GIVE_UP, Termination
from inspect_robots import eval as ir_eval
from inspect_robots.action_manager import (
    DefaultActionManager,
    EnsemblingActionManager,
    SmoothingActionManager,
)
from inspect_robots.errors import PolicyError
from inspect_robots.guard import PassThroughGuard
from inspect_robots.log import read_eval_log
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.rollout import Record, rollout
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

_SCENE = Scene(id="s", instruction="reach", init_seed=0)
_BOX = Box(shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"))
_DECLARED = frozenset({DONE, GIVE_UP})


class _TerminalPolicy:
    """Plays ``moves`` zero actions per chunk, then ends with ``terminal``."""

    def __init__(
        self,
        terminal: Termination,
        *,
        moves: int = 2,
        chunks_before: int = 0,
        declared: frozenset[str] = _DECLARED,
    ) -> None:
        self.info = PolicyInfo(name="terminal", action_space=_BOX, terminal_actions=declared)
        self.config = PolicyConfig()
        self._terminal = terminal
        self._moves = moves
        self._chunks_before = chunks_before
        self.calls = 0

    def reset(self, scene: Scene) -> None:
        self.calls = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.calls += 1
        actions = [Action(data=np.zeros(2)) for _ in range(self._moves)]
        if self.calls <= self._chunks_before:
            return ActionChunk(actions=actions or [Action(data=np.zeros(2))])
        return ActionChunk(actions=actions, terminal=self._terminal)


def _run(policy: object, embodiment: object | None = None, *, max_steps: int = 40) -> Record:
    return rollout(
        policy,  # type: ignore[arg-type]
        embodiment or CubePickEmbodiment(),  # type: ignore[arg-type]
        _SCENE,
        max_steps=max_steps,
        seed=0,
        epoch=0,
        action_manager=DefaultActionManager(),
        guard=PassThroughGuard(),
        sink=NullSink(),
    )


# -- the types ---------------------------------------------------------------


@pytest.mark.parametrize("kind", ["", "   ", 3])
def test_termination_needs_a_kind(kind: object) -> None:
    with pytest.raises(ValueError, match="kind must be a non-empty string"):
        Termination(kind=kind)  # type: ignore[arg-type]


def test_termination_reason_must_be_text() -> None:
    with pytest.raises(ValueError, match="reason must be a string"):
        Termination(kind=DONE, reason=None)  # type: ignore[arg-type]


def test_only_a_terminal_chunk_may_be_empty() -> None:
    with pytest.raises(ValueError, match="unless it is terminal"):
        ActionChunk(actions=[])
    chunk = ActionChunk(actions=[], terminal=Termination(GIVE_UP, "cannot reach"))
    assert len(chunk) == 0


# -- action managers ---------------------------------------------------------


def test_default_manager_marks_the_last_action_it_plays() -> None:
    policy = _TerminalPolicy(Termination(DONE, "stacked"), moves=3)
    store: dict[str, object] = {}
    manager = DefaultActionManager()
    played = [manager.next_action(policy, Observation(), t, store) for t in range(3)]
    assert [getattr(a, "terminal", None) for a in played] == [
        None,
        None,
        Termination(DONE, "stacked"),
    ]


def test_short_replan_interval_keeps_the_terminal() -> None:
    policy = _TerminalPolicy(Termination(DONE), moves=5)
    store: dict[str, object] = {}
    manager = DefaultActionManager(replan_interval=2)
    first = manager.next_action(policy, Observation(), 0, store)
    second = manager.next_action(policy, Observation(), 1, store)
    assert isinstance(first, Action) and first.terminal is None
    assert isinstance(second, Action) and second.terminal == Termination(DONE)


@pytest.mark.parametrize(
    "make",
    [
        DefaultActionManager,
        lambda: SmoothingActionManager(DefaultActionManager()),
        lambda: EnsemblingActionManager(_BOX),
    ],
    ids=["default", "smoothing", "ensembling"],
)
def test_terminal_only_chunk_returns_the_termination(make: object) -> None:
    policy = _TerminalPolicy(Termination(GIVE_UP, "blocked"), moves=0)
    store: dict[str, object] = {}
    result = make().next_action(policy, Observation(), 0, store)  # type: ignore[operator]
    assert result == Termination(GIVE_UP, "blocked")
    assert store["_controller_inferences"] == [(None, 0)]


def test_ensembling_marks_the_blended_action() -> None:
    policy = _TerminalPolicy(Termination(DONE), moves=2)
    result = EnsemblingActionManager(_BOX).next_action(policy, Observation(), 0, {})
    assert isinstance(result, Action)
    assert result.terminal == Termination(DONE)


def test_smoothing_keeps_the_terminal_on_the_smoothed_action() -> None:
    policy = _TerminalPolicy(Termination(DONE), moves=1)
    result = SmoothingActionManager(DefaultActionManager()).next_action(
        policy, Observation(), 0, {}
    )
    assert isinstance(result, Action)
    assert result.terminal == Termination(DONE)


# -- the rollout -------------------------------------------------------------


def test_terminal_chunk_ends_the_trial_after_its_actions() -> None:
    record = _run(_TerminalPolicy(Termination(DONE, "stacked all three"), moves=3))
    assert len(record.steps) == 3
    assert (record.truncated, record.terminated) == (True, False)
    assert record.termination_reason == DONE
    assert record.termination_detail == "stacked all three"
    terminal = [e for e in record.events if e.kind == "policy_terminal"]
    assert [(e.t, e.data) for e in terminal] == [(2, {"kind": DONE, "reason": "stacked all three"})]


def test_terminal_only_chunk_ends_without_stepping() -> None:
    policy = _TerminalPolicy(Termination(GIVE_UP, "out of reach"), moves=0, chunks_before=2)
    record = _run(policy)
    assert len(record.steps) == 2  # the two earlier one-action chunks, then nothing
    assert record.termination_reason == GIVE_UP
    assert record.termination_detail == "out of reach"
    assert record.events[-1].kind == "policy_terminal"


def test_empty_reason_is_recorded_as_no_detail() -> None:
    record = _run(_TerminalPolicy(Termination(DONE), moves=1))
    assert record.termination_detail is None


@pytest.mark.parametrize("moves", [0, 1], ids=["terminal-only", "with-actions"])
def test_undeclared_terminal_kind_is_a_policy_error(moves: int) -> None:
    policy = _TerminalPolicy(Termination("refuse", "unsafe"), moves=moves)
    expected = r"(?s)terminal action 'refuse'.*declared: \['done', 'give_up'\].*forward the inner"
    with pytest.raises(PolicyError, match=expected):
        _run(policy)


def test_terminal_on_the_last_allowed_step_is_not_reported_as_horizon() -> None:
    record = _run(_TerminalPolicy(Termination(DONE), moves=2), max_steps=2)
    assert record.termination_reason == DONE


def test_embodiment_termination_wins_over_a_terminal_action() -> None:
    class _InstantSuccess(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:
            result = super().step(action)
            return replace(result, terminated=True, termination_reason="success")

    record = _run(_TerminalPolicy(Termination(GIVE_UP), moves=1), _InstantSuccess())
    assert record.termination_reason == "success"
    assert record.termination_detail is None


def test_terminal_survives_a_guard_that_rebuilds_the_action() -> None:
    class _Rebuild:
        def review(self, action: Action, store: dict[str, object]) -> Action:
            return Action(data=np.asarray(action.data) * 0.5)

    record = rollout(
        _TerminalPolicy(Termination(DONE), moves=2),
        CubePickEmbodiment(),
        _SCENE,
        max_steps=40,
        seed=0,
        epoch=0,
        action_manager=DefaultActionManager(),
        guard=_Rebuild(),
        sink=NullSink(),
    )
    assert record.termination_reason == DONE


class _LegacyStopPolicy:
    def __init__(self, meta: dict[str, object]) -> None:
        self.info = PolicyInfo(name="legacy", action_space=_BOX)
        self.config = PolicyConfig()
        self._meta = meta

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(2), meta=dict(self._meta))])


@pytest.mark.parametrize(
    ("meta", "reason", "detail"),
    [
        (
            {"request_stop": True, "stop_reason": "done", "stop_detail": "jar open"},
            "done",
            "jar open",
        ),
        ({"request_stop": True, "stop_reason": "give_up", "stop_detail": 7}, "give_up", None),
        ({"request_stop": True, "stop_reason": ""}, "policy_stop", None),
    ],
    ids=["with-detail", "non-text-detail", "blank-reason"],
)
def test_legacy_stop_meta_still_ends_the_trial(
    meta: dict[str, object], reason: str, detail: str | None
) -> None:
    record = _run(_LegacyStopPolicy(meta))
    assert record.termination_reason == reason
    assert record.termination_detail == detail


def test_termination_details_reach_the_log(tmp_path: Path) -> None:
    task = Task(name="t", scenes=[_SCENE], scorer=success_at_end(), max_steps=40, epochs=2)
    logs = ir_eval(
        task,
        _TerminalPolicy(Termination(GIVE_UP, "the jar is glued shut"), moves=1),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    (sample,) = logs[0].samples
    assert sample.termination_reasons == (GIVE_UP, GIVE_UP)
    assert sample.termination_details == ("the jar is glued shut", "the jar is glued shut")
    (log_file,) = tmp_path.glob("*.json")
    assert read_eval_log(str(log_file)).samples[0].termination_details == sample.termination_details
