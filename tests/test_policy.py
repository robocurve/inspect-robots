"""Unit tests for Policy, PolicyConfig, PolicyInfo, and PolicyBase."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation


def test_policy_config_defaults_and_custom() -> None:
    cfg = PolicyConfig()
    assert cfg.action_horizon == 1
    assert cfg.replan_interval is None
    assert cfg.temperature is None

    cfg2 = PolicyConfig(action_horizon=8, replan_interval=4, temperature=0.5)
    assert cfg2.action_horizon == 8
    assert cfg2.replan_interval == 4
    assert cfg2.temperature == 0.5


@pytest.mark.parametrize("bad_horizon", [0, -1, True, False, 1.5, "4"])
def test_policy_config_rejects_invalid_action_horizon(bad_horizon: Any) -> None:
    with pytest.raises(ValueError, match="action_horizon must be an integer >= 1"):
        PolicyConfig(action_horizon=bad_horizon)


@pytest.mark.parametrize("bad_interval", [0, -1, True, False, 2.5, "2"])
def test_policy_config_rejects_invalid_replan_interval(bad_interval: Any) -> None:
    with pytest.raises(ValueError, match="replan_interval must be an integer >= 1 or None"):
        PolicyConfig(replan_interval=bad_interval)


@pytest.mark.parametrize(
    "bad_temp",
    [-0.1, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "0.5"],
)
def test_policy_config_rejects_invalid_temperature(bad_temp: Any) -> None:
    with pytest.raises(ValueError, match=r"temperature must be a finite number >= 0\.0 or None"):
        PolicyConfig(temperature=bad_temp)


def test_policy_info_valid() -> None:
    box = Box(shape=(3,))
    obs_space = ObservationSpace()
    info = PolicyInfo(
        name="my-policy",
        action_space=box,
        observation_space=obs_space,
        control_hz=20.0,
        checkpoint="sha256:abc",
    )
    assert info.name == "my-policy"
    assert info.action_space is box
    assert info.observation_space is obs_space
    assert info.control_hz == 20.0
    assert info.checkpoint == "sha256:abc"


@pytest.mark.parametrize("bad_name", ["", "   ", "\t\n", None, 123])
def test_policy_info_rejects_invalid_name(bad_name: Any) -> None:
    box = Box(shape=(3,))
    with pytest.raises(ValueError, match="PolicyInfo name must be a non-empty string"):
        PolicyInfo(name=bad_name, action_space=box)


def test_policy_info_rejects_invalid_action_space() -> None:
    with pytest.raises(TypeError, match="action_space must be a Box"):
        PolicyInfo(name="pol", action_space="not-a-box")  # type: ignore[arg-type]


def test_policy_info_rejects_invalid_observation_space() -> None:
    box = Box(shape=(3,))
    with pytest.raises(TypeError, match="observation_space must be an ObservationSpace"):
        PolicyInfo(name="pol", action_space=box, observation_space="not-an-obs-space")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_hz",
    [0, -5.0, float("nan"), float("inf"), float("-inf"), True, False, "10"],
)
def test_policy_info_rejects_invalid_control_hz(bad_hz: Any) -> None:
    box = Box(shape=(3,))
    with pytest.raises(ValueError, match="control_hz must be a positive finite number or None"):
        PolicyInfo(name="pol", action_space=box, control_hz=bad_hz)


@pytest.mark.parametrize("bad_ckpt", [123, True, ["hash"]])
def test_policy_info_rejects_invalid_checkpoint(bad_ckpt: Any) -> None:
    box = Box(shape=(3,))
    with pytest.raises(TypeError, match="checkpoint must be a string or None"):
        PolicyInfo(name="pol", action_space=box, checkpoint=bad_ckpt)


def test_policy_base_defaults() -> None:
    class MinimalPolicy(PolicyBase):
        def __init__(self) -> None:
            self.info = PolicyInfo(name="minimal", action_space=Box(shape=(2,)))

        def act(self, observation: Observation) -> ActionChunk:
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    policy = MinimalPolicy()
    assert policy.config.action_horizon == 1
    # Check default no-op lifecycle hooks don't raise
    policy.bind(None)  # type: ignore[arg-type]
    policy.bind_task(None)  # type: ignore[arg-type]
    policy.on_trial_start("s0", 0, "logs", "run-1")
    policy.on_trial_end(None, "logs", "run-1")  # type: ignore[arg-type]
    policy.reset(Scene(id="s0", instruction="test", init_seed=0))
    assert policy.transcript() is None
    chunk = policy.act(Observation())
    assert len(chunk.actions) == 1
