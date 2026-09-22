"""``JevPolicy``: Jev steers the YAM arms one menu pick at a time.

Per step: read gripper poses from ``observation.state["eef_state"]``, update
the tag-based world model, let the phase machine decide the stage, write the
curated state and menu, ask Jev one Choice question, and map the pick onto a
bounded Cartesian ``ActionChunk``. Every action carries ``meta["jev"]`` with
the phase, the pick, and the world so scorers can read it from the recorded
steps (scoring runs before ``on_trial_end``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_jev._decisions import DecisionsClient
from inspect_robots_jev.calibration import Calibration
from inspect_robots_jev.menu import HOLD, Move, build_menu, parse_option
from inspect_robots_jev.motion import MotionMapper
from inspect_robots_jev.perceiver import Perceiver, TagLayout
from inspect_robots_jev.phase import Phase, PhaseMachine, TaskConfig
from inspect_robots_jev.serializer import build_state, instructions_for
from inspect_robots_jev.world import ARMS, AXES, Arm, GripperView, Vec3, WorldState

#: Phases during which the cube is in the gripper and its tags are expected to be hidden.
_HELD_PHASES = frozenset({"lift", "carry", "lower"})
#: Free-space steering phases: the only ones that stall when the target tag goes stale.
_STALE_GATED = frozenset({"approach", "carry"})
#: Commanded-vs-measured divergence on the active arm's x/y/z beyond which the next
#: chunk's position interpolation restarts from the measured pose (gripper and
#: orientation keep their commanded values).
_RESYNC_M = 0.02


@dataclass(frozen=True)
class JevPolicyConfig:
    """Everything ``-P k=v`` can set; file paths are resolved lazily on first ``act``."""

    api: str = "openrouter"
    model: str | None = None
    steps_cm: tuple[float, ...] = (0.5, 2.0, 5.0)
    camera: str = "top_cam"
    tags: str = "tags.json"
    calibration: str = "calibration.json"
    history: int = 5
    stale_after: int = 10
    task: TaskConfig = field(default_factory=TaskConfig)


class JevPolicy(PolicyBase):
    """Text-only decision model as a policy over AprilTag world state."""

    def __init__(
        self,
        config: JevPolicyConfig | None = None,
        *,
        client: DecisionsClient | None = None,
        perceiver: Perceiver | None = None,
    ) -> None:
        self.settings = config or JevPolicyConfig()
        self._client = client
        self._perceiver = perceiver
        self._mapper: MotionMapper | None = None
        self._bounds: tuple[Vec3, Vec3] | None = None
        self._machine: PhaseMachine | None = None
        self._history: list[str] = []
        self._transcript: list[dict[str, Any]] = []
        self._held: tuple[Arm, str] | None = None
        self._decisions = 0
        self._stalls = 0
        self._last_world: WorldState | None = None
        self._last_target: np.ndarray | None = None
        self._last_move: Move | None = None
        self.info = PolicyInfo(name="jev", action_space=Box(shape=(14,)))

    # -- lifecycle -----------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment's Cartesian box; fail fast if it is not label-addressable."""
        mapper = MotionMapper(embodiment_info.action_space, control_hz=embodiment_info.control_hz)
        lows = [mapper.bounds_xyz(arm)[0] for arm in ARMS]
        highs = [mapper.bounds_xyz(arm)[1] for arm in ARMS]
        low = tuple(max(lo[i] for lo in lows) for i in range(3))
        high = tuple(min(hi[i] for hi in highs) for i in range(3))
        self._bounds = ((low[0], low[1], low[2]), (high[0], high[1], high[2]))
        self._mapper = mapper
        self.info = PolicyInfo(
            name="jev",
            action_space=embodiment_info.action_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        """Forget the previous trial: world memory, phase, history, transcript, counters."""
        self._machine = None
        self._history = []
        self._transcript = []
        self._held = None
        self._decisions = 0
        self._stalls = 0
        self._last_world = None
        self._last_target = None
        self._last_move = None
        if self._perceiver is not None:
            self._perceiver.reset()

    def transcript(self) -> list[dict[str, Any]]:
        """Per-step record of what Jev saw and answered (JSON-serialisable)."""
        return list(self._transcript)

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Write a summary into the trial's metadata for the log and report."""
        record.metadata["jev"] = {
            "final_phase": self._machine.phase.name if self._machine is not None else None,
            "final_world": self._world_dict(self._last_world),
            "decisions": self._decisions,
            "stalls": self._stalls,
        }

    # -- the step ------------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        """One perceive → phase → ask → move cycle."""
        mapper = self._require_mapper()
        cfg = self.settings
        eef = np.asarray(observation.state["eef_state"], dtype=np.float64)
        grippers: dict[Arm, GripperView] = {
            arm: GripperView(mapper.gripper_position(eef, arm), mapper.gripper_opening(eef, arm))
            for arm in ARMS
        }
        # The world clock counts policy decisions (one per act), not control
        # ticks: one pick plays 1-20 ticks, so `env_step` would make every
        # stale budget 5-20x too short. `env_step` is kept for the transcript.
        step = self._decisions + self._stalls
        tick = int(observation.extra.get("env_step", -1))
        held_for_world = self._held
        world = self._perceiver_instance().update(
            observation, step, grippers=grippers, held=held_for_world
        )
        self._last_world = world
        # Interpolate from the last commanded pose, not the measured one, so IK
        # tracking lag never shows up as a spurious delta clamp on the first tick
        # and a hold repeats the command instead of walking the arm back.
        start = self._last_target if self._last_target is not None else eef
        if self._last_target is not None and self._machine is not None:
            arm = self._machine.phase.arm
            xyz = [mapper.index(f"{arm}_{axis}") for axis in AXES]
            if bool(np.any(np.abs(self._last_target[xyz] - eef[xyz]) > _RESYNC_M)):
                # The arm could not follow (contact, IK hold): restart the position
                # interpolation from where the arm really is. Gripper and
                # orientation keep their commanded values: a jaw that stopped on
                # the cube at 0.3 must stay commanded closed, or the grasp relaxes.
                start = self._last_target.copy()
                start[xyz] = eef[xyz]
        if self._machine is None:
            if cfg.task.cube not in world.objects or cfg.task.bowl not in world.objects:
                return self._stall(
                    mapper, start, world, None, "cube or bowl not yet seen", None, tick
                )
            self._machine = PhaseMachine(cfg.task, bounds=self._bounds)
            phase = self._machine.reset(world)
        else:
            last = self._last_move
            descended_m = (
                -last.delta_m if last is not None and last.axis == "z" and last.delta_m < 0 else 0.0
            )
            phase = self._machine.advance(world, descended_m=descended_m)
        self._held = (phase.arm, cfg.task.cube) if phase.name in _HELD_PHASES else None
        if phase.name == "done":
            return self._finish(mapper, start, world, phase, held_for_world, tick)
        # Only the free-space steering phases wait for a fresh sighting. In
        # descend/lower the target is under the gripper and its tag is expected
        # to be hidden; the goal is the remembered pose the state text already
        # tells Jev to trust. Held phases pin the cube to the gripper.
        if phase.name in _STALE_GATED and (
            phase.target not in world.objects
            or world.steps_since_seen(phase.target) > cfg.stale_after
        ):
            return self._stall(
                mapper, start, world, phase, f"{phase.target} tag not seen", held_for_world, tick
            )
        state = build_state(
            world,
            phase,
            self._history,
            history_len=cfg.history,
            open_threshold=cfg.task.open_threshold,
            closed_threshold=cfg.task.closed_on_object,
        )
        menu = build_menu(phase.arm, phase.menu_kind, phase.target, cfg.steps_cm)
        instructions = instructions_for(phase)
        answer = self._client_instance().choose(
            state=state, instructions=instructions, criteria=menu
        )
        move = parse_option(answer.choice)
        if move.is_hold:
            move = Move(phase.arm, None, 0.0, None)
        chunk = mapper.chunk(move, start)
        self._last_target = np.asarray(chunk.actions[-1].data, dtype=np.float64)
        self._last_move = move
        self._decisions += 1
        self._history.append(menu[answer.choice].split(";")[0])
        summary = self._summary(
            phase, world, choice=answer.choice, confidence=answer.confidence, held=held_for_world
        )
        self._transcript.append(
            {
                "step": step,
                "tick": tick,
                "phase": phase.name,
                "arm": phase.arm,
                "state": state,
                "instructions": instructions,
                "menu": menu,
                "answer": {
                    "choice": answer.choice,
                    "probabilities": dict(answer.probabilities),
                    "confidence": answer.confidence,
                    "model": answer.model,
                    "usage": dict(answer.usage),
                },
                "actions": len(chunk.actions),
            }
        )
        return self._with_meta(chunk, {"jev": summary})

    # -- helpers -------------------------------------------------------------

    def _require_mapper(self) -> MotionMapper:
        if self._mapper is None:
            raise RuntimeError("JevPolicy.bind(embodiment_info) must run before act()")
        return self._mapper

    def _client_instance(self) -> DecisionsClient:
        if self._client is None:
            self._client = DecisionsClient(api=self.settings.api, model=self.settings.model)
        return self._client

    def _perceiver_instance(self) -> Perceiver:
        if self._perceiver is None:
            self._perceiver = Perceiver(
                layout=TagLayout.load(Path(self.settings.tags)),
                camera=self.settings.camera,
                calibration=Calibration.load(Path(self.settings.calibration)),
            )
        return self._perceiver

    @staticmethod
    def _world_dict(world: WorldState | None) -> dict[str, dict[str, Any]]:
        """Per object: position per arm frame plus how many steps since its tag was seen."""
        if world is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for name, view in world.objects.items():
            entry: dict[str, Any] = {
                arm: [float(v) for v in pos] for arm, pos in view.in_frame.items()
            }
            entry["seen_ago"] = world.steps_since_seen(name)
            entry["from_gripper"] = view.from_gripper
            out[name] = entry
        return out

    def _summary(
        self,
        phase: Phase | None,
        world: WorldState,
        *,
        choice: str,
        confidence: float | None,
        held: tuple[Arm, str] | None,
    ) -> dict[str, Any]:
        # `held` is the value the world was BUILT with, not the one chosen for
        # the next step, so the scorer reads a flag that matches `world`.
        return {
            "phase": phase.name if phase is not None else None,
            "arm": phase.arm if phase is not None else None,
            "choice": choice,
            "confidence": confidence,
            "held": held is not None,
            "world": self._world_dict(world),
        }

    @staticmethod
    def _with_meta(chunk: ActionChunk, meta: Mapping[str, Any]) -> ActionChunk:
        actions = [replace(a, meta={**a.meta, **meta}) for a in chunk.actions]
        return ActionChunk(
            actions=actions, control_hz=chunk.control_hz, inference_latency_s=None, meta=chunk.meta
        )

    def _hold(self, mapper: MotionMapper, start: Any, meta: Mapping[str, Any]) -> ActionChunk:
        chunk = mapper.chunk(Move("left", None, 0.0, None), start)
        return self._with_meta(chunk, meta)

    def _stall(
        self,
        mapper: MotionMapper,
        start: Any,
        world: WorldState,
        phase: Phase | None,
        why: str,
        held: tuple[Arm, str] | None = None,
        tick: int = -1,
    ) -> ActionChunk:
        self._stalls += 1
        self._transcript.append(
            {
                "step": world.step,
                "tick": tick,
                "phase": phase.name if phase is not None else None,
                "stall": True,
                "reason": why,
            }
        )
        summary = self._summary(phase, world, choice=HOLD, confidence=None, held=held)
        meta = {"jev": {**summary, "stall": why}}
        if phase is not None:
            # The likeliest reason the tag is unseen is the gripper hovering over
            # it, so holding would never let it reappear: rise instead. Bounded by
            # the action box like every other move.
            rise = Move(phase.arm, "z", self.settings.task.hover_m, None)
            chunk = mapper.chunk(rise, start)
            self._last_target = np.asarray(chunk.actions[-1].data, dtype=np.float64)
            self._last_move = rise
            return self._with_meta(chunk, meta)
        return self._hold(mapper, start, meta)

    def _finish(
        self,
        mapper: MotionMapper,
        eef: Any,
        world: WorldState,
        phase: Phase,
        held: tuple[Arm, str] | None,
        tick: int,
    ) -> ActionChunk:
        self._transcript.append(
            {"step": world.step, "tick": tick, "phase": "done", "request_stop": True}
        )
        summary = self._summary(phase, world, choice=HOLD, confidence=None, held=held)
        return self._hold(
            mapper, eef, {"request_stop": True, "stop_reason": "task_done", "jev": summary}
        )


def _floats(value: Any) -> tuple[float, ...]:
    parts = value.split(",") if isinstance(value, str) else list(value)
    steps = tuple(float(part) for part in parts if str(part).strip())
    if not steps or any(step <= 0 for step in steps):
        raise ValueError(f"steps_cm must be positive numbers, got {value!r}")
    return steps


def jev_policy(**kwargs: Any) -> JevPolicy:
    """Registry entry for ``--policy jev``; ``-P`` values arrive as strings."""
    task = TaskConfig(
        cube=str(kwargs.pop("cube", TaskConfig.cube)), bowl=str(kwargs.pop("bowl", TaskConfig.bowl))
    )
    if "steps_cm" in kwargs:
        kwargs["steps_cm"] = _floats(kwargs["steps_cm"])
    for key in ("history", "stale_after"):
        if key in kwargs:
            kwargs[key] = int(kwargs[key])
    allowed = {f.name for f in JevPolicyConfig.__dataclass_fields__.values()} - {"task"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise TypeError(f"unknown jev policy args: {sorted(unknown)}")
    config = JevPolicyConfig(task=task, **kwargs)
    return JevPolicy(config)
