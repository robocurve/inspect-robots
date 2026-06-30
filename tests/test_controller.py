"""Controller middleware: open-loop replanning and composition."""

from __future__ import annotations

import numpy as np

from roboinspect.controller import DefaultController, SmoothingController
from roboinspect.mock import CubePickEmbodiment, ScriptedPolicy
from roboinspect.types import Observation


def _obs(embodiment: CubePickEmbodiment) -> Observation:
    from roboinspect.scene import Scene

    return embodiment.reset(Scene(id="s", instruction="x"), seed=0)


def test_default_controller_plays_whole_chunk_then_replans() -> None:
    policy = ScriptedPolicy(chunk_size=5)
    embodiment = CubePickEmbodiment()
    obs = _obs(embodiment)
    store: dict[str, object] = {}
    ctrl = DefaultController()  # replan_interval None -> whole chunk
    for _ in range(5):
        ctrl.next_action(policy, obs, 0, store)
    assert policy.num_inferences == 1  # one inference covered all 5 actions


def test_replan_interval_reinfers_periodically() -> None:
    policy = ScriptedPolicy(chunk_size=8)
    embodiment = CubePickEmbodiment()
    obs = _obs(embodiment)
    store: dict[str, object] = {}
    ctrl = DefaultController(replan_interval=2)
    for _ in range(6):
        ctrl.next_action(policy, obs, 0, store)
    assert policy.num_inferences == 3  # 6 actions / 2 per inference


def test_smoothing_controller_composes() -> None:
    policy = ScriptedPolicy(chunk_size=4)
    embodiment = CubePickEmbodiment()
    obs = _obs(embodiment)
    store: dict[str, object] = {}
    inner = DefaultController()
    smooth = SmoothingController(inner, alpha=0.5)
    a0 = smooth.next_action(policy, obs, 0, store)
    a1 = smooth.next_action(policy, obs, 1, store)
    # First action passes through; the second is the EMA of raw and previous.
    assert isinstance(a0.data, np.ndarray)
    assert a1.data.shape == (2,)
    # Inference bookkeeping still flows from the wrapped controller.
    assert policy.num_inferences == 1
