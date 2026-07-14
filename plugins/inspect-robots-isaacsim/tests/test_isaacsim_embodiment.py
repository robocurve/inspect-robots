"""Tests for the Isaac Lab embodiment adapter.

These run on any machine — Isaac Sim is NOT required. They verify the parts that
matter before a simulator ever boots: the declared spaces/semantics, protocol
conformance, registry resolution, Inspect Robots compatibility checking, and that
calling reset() without Isaac fails with a clear, actionable error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots_isaacsim import IsaacSimEmbodiment, isaacsim_embodiment

from inspect_robots import (
    ActionSemantics,
    Box,
    Embodiment,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.compat import check_compatibility
from inspect_robots.errors import CompatibilityError
from inspect_robots.types import ActionChunk, Observation

_FRANKA_SEM = ActionSemantics(
    control_mode="joint_pos", rotation_repr="none", gripper="binary", frame="base"
)


class _FrankaPolicy:
    """A minimal policy whose spaces match the default Franka profile (dim 8)."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="franka-stub",
            action_space=Box(shape=(8,), semantics=_FRANKA_SEM),
            observation_space=ObservationSpace(state_keys=frozenset({"joint_pos"})),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Scene) -> None: ...

    def act(self, observation: Observation) -> ActionChunk:  # pragma: no cover - unused
        from inspect_robots import Action

        return ActionChunk(actions=[Action(data=np.zeros(8))])


def test_info_describes_franka_joint_pos() -> None:
    emb = IsaacSimEmbodiment()
    assert emb.info.name == "isaacsim"
    assert emb.info.is_simulated is True
    # 7 arm joints + 1 binary gripper command.
    assert emb.info.action_space.dim == 8
    sem = emb.info.action_space.semantics
    assert sem is not None
    assert sem.control_mode == "joint_pos"
    assert sem.gripper == "binary"
    assert "privileged_success" in emb.info.capabilities
    assert "self_paced" not in emb.info.capabilities  # sim is framework-paced


def test_action_dim_tracks_arm_joints() -> None:
    assert IsaacSimEmbodiment(num_arm_joints=6).info.action_space.dim == 7


def test_rejects_bad_joint_count() -> None:
    with pytest.raises(ValueError):
        IsaacSimEmbodiment(num_arm_joints=0)


def test_observation_space_has_state_and_cameras() -> None:
    emb = IsaacSimEmbodiment()
    obs_space = emb.info.observation_space
    assert "base_rgb" in obs_space.camera_names
    assert {"joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper"} <= obs_space.state_keys


def test_satisfies_embodiment_protocol() -> None:
    assert isinstance(IsaacSimEmbodiment(), Embodiment)


def test_factory_and_kwargs() -> None:
    emb = isaacsim_embodiment(task_id="Isaac-Open-Drawer-Franka-v0", control_hz=20.0)
    assert isinstance(emb, IsaacSimEmbodiment)
    assert emb.task_id == "Isaac-Open-Drawer-Franka-v0"
    assert emb.info.control_hz == 20.0


def test_compatible_with_matching_franka_policy() -> None:
    report = check_compatibility(_FrankaPolicy(), IsaacSimEmbodiment())
    assert report.ok, report.errors


def test_incompatible_with_2d_policy_fails_fast() -> None:
    class _2DPolicy(_FrankaPolicy):
        def __init__(self) -> None:
            self.info = PolicyInfo(
                name="cube2d",
                action_space=Box(
                    shape=(2,),
                    semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
                ),
            )
            self.config = PolicyConfig()

    report = check_compatibility(_2DPolicy(), IsaacSimEmbodiment())
    assert not report.ok
    codes = {i.code for i in report.errors}
    assert "action_dim" in codes
    with pytest.raises(CompatibilityError):
        report.raise_for_errors()


def test_reset_without_isaac_raises_clear_error() -> None:
    emb = IsaacSimEmbodiment()
    with pytest.raises(RuntimeError, match="Isaac Sim / Isaac Lab is not importable"):
        emb.reset(Scene(id="s0", instruction="lift the cube"))


def test_close_is_safe_before_launch_and_idempotent() -> None:
    emb = IsaacSimEmbodiment()
    emb.close()  # no env/app yet -> no-op, must not raise
    emb.close()  # second call must also be safe


def test_context_manager_calls_close() -> None:
    closed = {"n": 0}

    class _Emb(IsaacSimEmbodiment):
        def close(self) -> None:
            closed["n"] += 1

    with _Emb() as emb:
        assert isinstance(emb, IsaacSimEmbodiment)
    assert closed["n"] == 1


# --------------------------------------------------------------------------- #
# Memory-safety: drive reset()/step() many times against a fake Isaac env and
# assert the adapter accumulates no state and leaks no RAM. Runs without Isaac.
# --------------------------------------------------------------------------- #
class _FakeIsaacEnv:
    """A stand-in for the Isaac Lab gym env: dict obs batched over num_envs=1."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.reset_calls = 0
        self.step_calls = 0

    def _obs(self) -> dict[str, dict[str, np.ndarray]]:
        # Fresh arrays each call, exactly like Isaac handing back buffer reads.
        return {
            "policy": {
                "base_rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
                "joint_pos": np.zeros((1, 7), dtype=np.float32),
                "joint_vel": np.zeros((1, 7), dtype=np.float32),
                "eef_pos": np.zeros((1, 3), dtype=np.float32),
                "eef_quat": np.zeros((1, 4), dtype=np.float32),
                "gripper": np.zeros((1, 1), dtype=np.float32),
            }
        }

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.reset_calls += 1
        return self._obs(), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self.step_calls += 1
        return (
            self._obs(),
            np.array([0.0]),
            np.array([False]),
            np.array([False]),
            {"success": False},
        )

    def close(self) -> None: ...


class _FakeTorch:
    @staticmethod
    def as_tensor(value: Any, device: str | None = None) -> np.ndarray:
        return np.asarray(value)


def _inject_fake(emb: IsaacSimEmbodiment) -> _FakeIsaacEnv:
    fake = _FakeIsaacEnv(emb.info.action_space.dim)
    emb._env = fake
    emb._torch = _FakeTorch()
    return fake


def test_step_translation_against_fake_env() -> None:
    from inspect_robots import Action

    emb = IsaacSimEmbodiment()
    _inject_fake(emb)
    emb.reset(Scene(id="s", instruction="lift"))
    result = emb.step(Action(data=np.zeros(8)))
    assert set(result.observation.images) == {"base_rgb"}
    assert result.observation.images["base_rgb"].dtype == np.uint8
    assert {"joint_pos", "eef_pos", "gripper"} <= set(result.observation.state)
    assert result.terminated is False
    assert result.info["success"] is False


def test_no_ram_leak_over_many_steps() -> None:
    """RAM must stay flat over thousands of steps (caller drops each result)."""
    import gc
    import tracemalloc

    from inspect_robots import Action

    emb = IsaacSimEmbodiment()
    fake = _inject_fake(emb)
    act = Action(data=np.zeros(8, dtype=np.float32))

    # Warm up so one-time allocations (interned strings, caches) settle.
    emb.reset(Scene(id="s", instruction="lift"))
    for _ in range(200):
        emb.step(act)

    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(3000):
        emb.step(act)  # result intentionally dropped each iteration
    gc.collect()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grew = sum(s.size_diff for s in after.compare_to(before, "filename") if s.size_diff > 0)
    # 3000 steps each allocating a 150KB image would be ~450MB if retained; a
    # leak-free loop holds nothing, so allow only a small slack for allocator noise.
    assert grew < 5_000_000, f"RAM grew {grew} bytes over 3000 steps; suspect a leak"
    # The adapter itself must hold no per-step accumulation.
    assert fake.step_calls == 3200
    for value in vars(emb).values():
        assert not isinstance(value, (list, dict)) or len(value) <= 1


def test_disable_debug_vis_walks_nested_configs() -> None:
    from inspect_robots_isaacsim.embodiment import _disable_debug_vis

    class _Term:
        def __init__(self, debug_vis: bool) -> None:
            self.debug_vis = debug_vis

    class _Commands:
        def __init__(self) -> None:
            self.object_pose = _Term(True)
            self.extra = [_Term(True), {"nested": _Term(True)}]

    class _EnvCfg:
        def __init__(self) -> None:
            self.commands = _Commands()
            self.scene = _Term(False)  # already off: stays off
            self.not_a_flag = _Term(True)
            self.debug_vis = "keep"  # non-bool sentinel: must not be touched

    cfg = _EnvCfg()
    cfg.commands.object_pose.debug_vis = True
    # A reference cycle must not hang the walk.
    cfg.commands.parent = cfg  # type: ignore[attr-defined]

    _disable_debug_vis(cfg)

    assert cfg.commands.object_pose.debug_vis is False
    assert cfg.commands.extra[0].debug_vis is False
    assert cfg.commands.extra[1]["nested"].debug_vis is False
    assert cfg.not_a_flag.debug_vis is False
    assert cfg.scene.debug_vis is False
    assert cfg.debug_vis == "keep"


def test_request_named_obs_terms_flips_concatenate_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from inspect_robots_isaacsim.embodiment import _request_named_obs_terms

    class _Group:
        concatenate_terms = True

    class _Obs:
        policy = _Group()

    class _Cfg:
        observations = _Obs()

    cfg = _Cfg()
    with caplog.at_level(logging.WARNING, logger="inspect_robots_isaacsim.embodiment"):
        _request_named_obs_terms(cfg, "policy")
        assert cfg.observations.policy.concatenate_terms is False
        assert not caplog.records  # success path stays quiet

        # Unknown group / missing attrs: a warned no-op, never raises — a
        # silently-flat group would mean an empty Observation.state.
        _request_named_obs_terms(cfg, "nonexistent")
        _request_named_obs_terms(object(), "policy")
    assert len(caplog.records) == 2
    assert all("named observation terms" in r.message for r in caplog.records)
