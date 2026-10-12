"""Controller middleware: open-loop replanning and composition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots.controller import DefaultController, SmoothingController
from inspect_robots.errors import PolicyError
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Action, ActionChunk, Observation


def _obs(embodiment: CubePickEmbodiment) -> Observation:
    from inspect_robots.scene import Scene

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


@pytest.mark.parametrize("bad_interval", [0, -1, 2.5, True, False, "5"])
def test_default_controller_rejects_invalid_replan_interval(bad_interval: Any) -> None:
    with pytest.raises(ValueError, match="replan_interval must be an integer >= 1 or None"):
        DefaultController(replan_interval=bad_interval)


def test_default_controller_raises_policy_error_on_empty_chunk() -> None:
    class _EmptyChunkPolicy:
        info = PolicyInfo(
            name="empty-policy",
            action_space=Box(shape=(1,), semantics=ActionSemantics("joint_delta")),
        )
        config = PolicyConfig()

        def act(self, obs: Observation) -> ActionChunk:
            # Bypass ActionChunk.__post_init__ to simulate an empty sequence returned by an adapter
            chunk = object.__new__(ActionChunk)
            object.__setattr__(chunk, "actions", ())
            object.__setattr__(chunk, "inference_latency_s", None)
            return chunk

    ctrl = DefaultController()
    store: dict[str, object] = {}
    with pytest.raises(PolicyError, match="returned an empty ActionChunk"):
        ctrl.next_action(_EmptyChunkPolicy(), Observation(), 0, store)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_alpha", [0.0, -0.5, 1.5, float("nan")])
def test_smoothing_controller_rejects_invalid_alpha(bad_alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\]"):
        SmoothingController(DefaultController(), alpha=bad_alpha)


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


class _VaryingPolicy:
    """Emits a scripted, non-constant scalar per inference."""

    def __init__(self, values: list[float]):
        self._values = values
        self._index = 0
        self.info = PolicyInfo(
            name="varying",
            action_space=Box(shape=(1,), semantics=ActionSemantics(control_mode="joint_delta")),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Scene) -> None:
        self._index = 0

    def act(self, observation: Observation) -> ActionChunk:
        value = self._values[self._index]
        self._index += 1
        return ActionChunk(actions=[Action(data=np.array([value]))])


def test_stacked_smoothing_controllers_keep_separate_state() -> None:
    # Two smoothing layers in one chain shared a single module-level store key,
    # so each smoothed against the other's last write rather than its own,
    # silently producing neither layer's intended output (#296). A varying
    # stream is required: with a constant one the corruption is invisible.
    values = [1.0, 5.0, 2.0, 8.0, 3.0]
    obs = Observation(instruction="x")
    outer = SmoothingController(SmoothingController(DefaultController(), alpha=0.5), alpha=0.9)

    store: dict[str, object] = {}
    policy = _VaryingPolicy(values)
    actual = [float(outer.next_action(policy, obs, t, store).data[0]) for t in range(len(values))]

    inner_prev: float | None = None
    outer_prev: float | None = None
    expected: list[float] = []
    for value in values:
        inner_prev = value if inner_prev is None else 0.5 * value + 0.5 * inner_prev
        outer_prev = inner_prev if outer_prev is None else 0.9 * inner_prev + 0.1 * outer_prev
        expected.append(outer_prev)

    assert np.allclose(actual, expected)
