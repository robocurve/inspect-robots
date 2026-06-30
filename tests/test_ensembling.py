"""Temporal-ensembling controller: blend math, cadence, eviction, semantics."""

from __future__ import annotations

import numpy as np
import pytest

from roboinspect import eval
from roboinspect.controller import EnsemblingController
from roboinspect.mock import CubePickEmbodiment, ScriptedPolicy
from roboinspect.scene import Scene
from roboinspect.scorer import success_at_end
from roboinspect.spaces import ActionSemantics, Box
from roboinspect.task import Task
from roboinspect.types import Action, ActionChunk, Observation

_DELTA_SPACE = Box(shape=(2,), semantics=ActionSemantics(control_mode="eef_delta_pos"))


class _FixedChunkPolicy:
    """Returns the same two-action chunk every inference; counts inferences."""

    def __init__(self, a0: list[float], a1: list[float], chunk_len: int = 2):
        self._actions = [np.array(a0, dtype=np.float64), np.array(a1, dtype=np.float64)]
        self.chunk_len = chunk_len
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        acts = [Action(data=self._actions[i % 2]) for i in range(self.chunk_len)]
        return ActionChunk(actions=acts)


def test_blend_uniform_is_mean_at_overlap() -> None:
    policy = _FixedChunkPolicy([1.0, 0.0], [0.0, 1.0])
    ctrl = EnsemblingController(_DELTA_SPACE, m=0.0)
    store: dict[str, object] = {}
    obs = Observation()
    a0 = ctrl.next_action(policy, obs, 0, store)
    assert np.allclose(a0.data, [1.0, 0.0])  # only the fresh chunk's actions[0]
    a1 = ctrl.next_action(policy, obs, 1, store)
    # overlap: chunk@0 -> actions[1]=[0,1]; chunk@1 -> actions[0]=[1,0]; mean
    assert np.allclose(a1.data, [0.5, 0.5])


def test_large_m_favors_oldest() -> None:
    policy = _FixedChunkPolicy([1.0, 0.0], [0.0, 1.0])
    ctrl = EnsemblingController(_DELTA_SPACE, m=50.0)
    store: dict[str, object] = {}
    obs = Observation()
    ctrl.next_action(policy, obs, 0, store)
    a1 = ctrl.next_action(policy, obs, 1, store)
    # oldest contribution at t=1 is chunk@0 -> actions[1] = [0,1]
    assert np.allclose(a1.data, [0.0, 1.0], atol=1e-6)


def test_buffer_eviction_bounded_by_chunk_len() -> None:
    policy = _FixedChunkPolicy([0.1, 0.0], [0.0, 0.1], chunk_len=3)
    ctrl = EnsemblingController(_DELTA_SPACE, m=0.1)
    store: dict[str, object] = {}
    obs = Observation()
    for t in range(10):
        ctrl.next_action(policy, obs, t, store)
    assert len(store["_ensemble_chunks"]) <= 3  # type: ignore[arg-type]


def test_inference_every_step() -> None:
    policy = _FixedChunkPolicy([0.1, 0.0], [0.0, 0.1])
    ctrl = EnsemblingController(_DELTA_SPACE, m=0.1)
    store: dict[str, object] = {}
    obs = Observation()
    for t in range(6):
        ctrl.next_action(policy, obs, t, store)
    assert policy.num_inferences == 6  # re-infers every step (vs DefaultController)


def test_rejects_unaverageable_semantics() -> None:
    with pytest.raises(ValueError, match="rotation_repr"):
        EnsemblingController(
            Box(shape=(7,), semantics=ActionSemantics("eef_abs_pose", rotation_repr="quat_wxyz"))
        )
    with pytest.raises(ValueError, match="binary gripper"):
        EnsemblingController(
            Box(shape=(3,), semantics=ActionSemantics("eef_delta_pos", gripper="binary"))
        )


def test_warns_when_semantics_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import roboinspect.controller as ctrl_mod

    monkeypatch.setattr(ctrl_mod, "_ENSEMBLE_WARNED", False)
    with pytest.warns(RuntimeWarning, match="no semantics"):
        EnsemblingController(Box(shape=(2,)))


def test_eval_with_ensembling_succeeds(tmp_path: object) -> None:
    task = Task(
        name="demo",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=120,
    )
    ctrl = EnsemblingController(CubePickEmbodiment().info.action_space, m=0.01)
    logs = eval(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        controller=ctrl,
        log_dir=str(tmp_path),
    )
    assert logs[0].status == "success"
    assert logs[0].results.metrics["success_at_end"] == 1.0
