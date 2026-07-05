"""An ``Embodiment`` adapter that wraps an Isaac Lab (Isaac Sim) environment.

This is the "body + world" half of a Inspect Robots eval, backed by a real physics
simulation. It conforms to :class:`inspect_robots.Embodiment` (the runtime-checkable
protocol), so once installed it can be paired with any compatible
:class:`inspect_robots.Policy` and run through ``inspect_robots.eval``.

Design mirrors Inspect Robots's own ``RerunSink``: the heavy, GPU-bound, non-PyPI
dependencies (``isaacsim`` / ``isaaclab`` / ``torch``) are imported **lazily,
inside methods**. Constructing the adapter and reading ``.info`` therefore work
on any machine (so ``inspect-robots list embodiments`` and the fail-fast compatibility
check run without Isaac); only :meth:`reset` / :meth:`step` actually launch the
simulator.

Default profile: a 7-DoF **Franka Panda** under **joint-position** control with a
binary gripper (action dim ``= num_arm_joints + 1``). Everything is configurable
so any Isaac Lab manipulation task can be wrapped.

Version note: Isaac Lab's gym wrapper returns a dict observation grouped by
``"policy"`` and batched over ``num_envs``. The obs/action *group keys* and the
``success`` signal differ across tasks; the constructor exposes ``obs_group``,
``image_keys``, ``state_keys`` and ``success_info_key`` so you can map your task
without editing this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from inspect_robots import (
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentInfo,
    ObservationSpace,
    Observation,
    Scene,
    StateField,
    StateSpec,
    StepResult,
)
from inspect_robots.spaces import CANONICAL_STATE_UNITS

if TYPE_CHECKING:
    from inspect_robots import Action

# Isaac Lab simulators expose a privileged success oracle, are resettable and
# seedable, and can render — but they are NOT self-paced (they step as fast as the
# GPU allows; Inspect Robots owns wall-clock pacing).
_DEFAULT_CAPABILITIES = frozenset({"seedable", "resettable", "privileged_success", "renderable"})

# Isaac Sim allows exactly ONE SimulationApp per process. Track it module-side so a
# second embodiment instance reuses the live app instead of launching a duplicate
# (which crashes) or leaking one. Cleared by close().
_ACTIVE_APP: Any | None = None


def _missing_isaac(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "Isaac Sim / Isaac Lab is not importable in this environment "
        f"({exc}). The inspect-robots-isaacsim adapter needs a working Isaac Lab "
        "install (NVIDIA Omniverse + GPU). Constructing the embodiment and "
        "reading .info work without it, but reset()/step() require it. "
        "See: https://isaac-sim.github.io/IsaacLab/"
    )


def _default_state_fields(num_arm_joints: int) -> tuple[StateField, ...]:
    """Canonical proprioception for a Franka-like arm (keys/units from Inspect Robots)."""

    def unit(key: str) -> str:
        return CANONICAL_STATE_UNITS.get(key, "")

    return (
        StateField("joint_pos", (num_arm_joints,), unit("joint_pos")),
        StateField("joint_vel", (num_arm_joints,), unit("joint_vel")),
        StateField("eef_pos", (3,), unit("eef_pos")),
        StateField("eef_quat", (4,), unit("eef_quat")),
        StateField("gripper", (1,), unit("gripper")),
    )


class IsaacSimEmbodiment:
    """Wrap an Isaac Lab gym task as a Inspect Robots :class:`~inspect_robots.Embodiment`.

    Parameters
    ----------
    task_id:
        The Isaac Lab / gymnasium task id, e.g. ``"Isaac-Lift-Cube-Franka-v0"``.
    num_arm_joints:
        Number of actuated arm joints (Franka Panda = 7). The action is
        ``num_arm_joints`` joint targets plus one binary gripper command, so the
        action dimension is ``num_arm_joints + 1``.
    cameras:
        Camera streams the task renders, as ``(name, height, width[, channels])``
        tuples. Match the names your policy requires (Inspect Robots remaps if needed).
    headless / device:
        Forwarded to Isaac's ``AppLauncher`` / env. ``headless=True`` is required
        on machines without a display (the usual eval box).
    obs_group / image_keys / state_keys / success_info_key:
        How to read the task's observation dict and success flag (see module docs).
    name / supported_setups / supported_target_kinds:
        Surfaced on ``EmbodimentInfo`` for logging and R7 scene-realizability
        checks. Empty ``supported_*`` means "unconstrained".
    """

    def __init__(
        self,
        task_id: str = "Isaac-Lift-Cube-Franka-v0",
        *,
        num_arm_joints: int = 7,
        cameras: Sequence[tuple[str, int, int] | tuple[str, int, int, int]] = (
            ("base_rgb", 224, 224, 3),
        ),
        control_hz: float = 30.0,
        headless: bool = True,
        device: str = "cuda:0",
        obs_group: str = "policy",
        image_keys: Mapping[str, str] | None = None,
        state_keys: Mapping[str, str] | None = None,
        success_info_key: str = "success",
        name: str = "isaacsim",
        supported_setups: Sequence[str] = (),
        supported_target_kinds: Sequence[str] = (),
    ) -> None:
        if num_arm_joints < 1:
            raise ValueError("num_arm_joints must be >= 1")

        self.task_id = task_id
        self.num_arm_joints = num_arm_joints
        self.headless = headless
        self.device = device
        self.obs_group = obs_group
        # Map Inspect Robots obs keys -> the task's raw dict keys. Identity by default.
        self.image_keys = dict(image_keys or {})
        self.state_keys = dict(state_keys or {})
        self.success_info_key = success_info_key

        camera_specs = tuple(CameraSpec(c[0], c[1], c[2], *(c[3:] or (3,))) for c in cameras)
        action_dim = num_arm_joints + 1  # arm joint targets + 1 binary gripper

        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(
                shape=(action_dim,),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="binary",
                    frame="base",
                ),
            ),
            observation_space=ObservationSpace(
                cameras=camera_specs,
                state=StateSpec(fields=_default_state_fields(num_arm_joints)),
            ),
            control_hz=control_hz,
            is_simulated=True,
            capabilities=_DEFAULT_CAPABILITIES,
            supported_setups=frozenset(supported_setups),
            supported_target_kinds=frozenset(supported_target_kinds),
        )

        # Lazily-initialised Isaac handles (None until first reset()).
        self._app: Any | None = None
        self._env: Any | None = None
        self._torch: Any | None = None

    # ------------------------------------------------------------------ #
    # Lazy Isaac bring-up (app launch and env creation are separable so the
    # simulator boot can be validated independently of any task registry)
    # ------------------------------------------------------------------ #
    def _ensure_app(self) -> Any:
        """Launch the Isaac Sim ``SimulationApp`` on first use and return it.

        If another instance already launched the app this process, reuse it —
        Isaac Sim is a hard process singleton, so launching twice would crash.
        """
        global _ACTIVE_APP
        if self._app is not None:
            return self._app
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without Isaac
            raise _missing_isaac(exc) from exc
        self._torch = torch

        if _ACTIVE_APP is not None:
            self._app = _ACTIVE_APP
            return self._app

        try:
            from isaaclab.app import AppLauncher
        except ImportError as exc:  # pragma: no cover - exercised only without Isaac
            raise _missing_isaac(exc) from exc
        self._app = AppLauncher(headless=self.headless, device=self.device).app
        _ACTIVE_APP = self._app
        return self._app

    def _ensure_env(self) -> Any:
        """Build the gym env on first use (boots the app if needed)."""
        if self._env is not None:
            return self._env
        self._ensure_app()
        # Importing the tasks registers the gym ids; must happen AFTER the app boots.
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  (registers Isaac-* gym ids)

        self._env = gym.make(self.task_id, num_envs=1, render_mode="rgb_array")
        return self._env

    # ------------------------------------------------------------------ #
    # Embodiment protocol
    # ------------------------------------------------------------------ #
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        env = self._ensure_env()
        obs, _info = env.reset(seed=seed)
        return self._to_observation(obs, scene.instruction)

    def step(self, action: Action) -> StepResult:
        env = self._ensure_env()
        torch = self._torch
        assert torch is not None  # set alongside _env
        tensor = torch.as_tensor(
            np.asarray(action.data, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        obs, reward, terminated, truncated, info = env.step(tensor)

        success = self._read_success(info, terminated)
        term = bool(_scalar(terminated))
        return StepResult(
            observation=self._to_observation(obs, None),
            reward=float(_scalar(reward)),
            terminated=term,
            termination_reason="success" if (term and success) else None,
            truncated=bool(_scalar(truncated)),
            info={"success": success},
        )

    def close(self) -> None:
        """Release the gym env and Isaac Sim app (and the GPU memory they hold).

        Idempotent: safe to call before launch or twice. ``eval()`` closes what
        it resolves — an embodiment looked up by registry name is closed when
        the run finishes, but an embodiment object you construct is yours to
        close. Use it as a context manager (or call ``close()`` in a
        ``finally``) to guarantee the simulator is torn down and GPU memory is
        freed when a run ends.
        """
        global _ACTIVE_APP
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._app is not None:
            self._app.close()
            if _ACTIVE_APP is self._app:
                _ACTIVE_APP = None
            self._app = None
        self._torch = None

    def __enter__(self) -> IsaacSimEmbodiment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Translation: Isaac Lab dict/tensors <-> Inspect Robots types
    # ------------------------------------------------------------------ #
    def _to_observation(self, raw: Any, instruction: str | None) -> Observation:
        group = raw[self.obs_group] if isinstance(raw, Mapping) and self.obs_group in raw else raw
        images: dict[str, np.ndarray] = {}
        state: dict[str, np.ndarray] = {}

        for cam in self.info.observation_space.cameras:
            key = self.image_keys.get(cam.name, cam.name)
            if isinstance(group, Mapping) and key in group:
                images[cam.name] = _to_image(group[key])

        if self.info.observation_space.state is not None:
            for field in self.info.observation_space.state.fields:
                key = self.state_keys.get(field.key, field.key)
                if isinstance(group, Mapping) and key in group:
                    state[field.key] = _to_float_array(group[key])

        return Observation(images=images, state=state, instruction=instruction)

    def _read_success(self, info: Any, terminated: Any) -> bool:
        if isinstance(info, Mapping) and self.success_info_key in info:
            return bool(_scalar(info[self.success_info_key]))
        # Fall back to "terminated implies success" when the task exposes no oracle.
        return bool(_scalar(terminated))


# --------------------------------------------------------------------------- #
# Tensor/array helpers (kept torch-free at import time; operate duck-typed)
# --------------------------------------------------------------------------- #
def _np(value: Any) -> np.ndarray:
    """Best-effort conversion of a (possibly GPU torch) value to a NumPy array."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    arr = _np(value).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _to_float_array(value: Any) -> np.ndarray:
    arr = _np(value).astype(np.float64)
    # Drop the leading num_envs axis (we run num_envs=1).
    return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr


def _to_image(value: Any) -> np.ndarray:
    arr = _np(value)
    if arr.ndim >= 1 and arr.shape[0] == 1:  # drop num_envs axis
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255)
    return arr.astype(np.uint8)
