"""The rollout engine — the closed control loop at the heart of RoboLens.

One :func:`rollout` runs a single trial (one scene, one epoch): it drives the
policy↔embodiment loop through the :class:`~robolens.controller.Controller`
(open-loop chunk execution) and the :class:`~robolens.approver.Approver` safety
gate, logging each step to the sinks, and returns an immutable
:class:`TrialRecord` that scorers consume.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robolens.approver import Approver
from robolens.controller import Controller
from robolens.embodiment import SELF_PACED, Embodiment
from robolens.policy import Policy
from robolens.scene import Scene
from robolens.types import Action, Observation, StepResult

if TYPE_CHECKING:
    from robolens.logging.sink import LogSink


def derive_seed(eval_seed: int | None, scene_seed: int | None, epoch: int) -> int:
    """Deterministically combine eval/scene seeds and the epoch index (R2).

    Distinct epochs of the same scene get distinct seeds so repeats actually vary
    for stochastic policies, while a fixed ``(eval_seed, scene_seed, epoch)``
    reproduces bitwise.
    """
    payload = f"{eval_seed or 0}:{scene_seed or 0}:{epoch}".encode()
    return zlib.crc32(payload) & 0xFFFFFFFF


@dataclass(frozen=True, eq=False)
class StepRecord:
    """One step of a recorded trajectory."""

    t: int
    observation: Observation
    action: Action
    result: StepResult


@dataclass
class TrialRecord:
    """The full record of one trial — the unit scorers consume."""

    scene_id: str
    epoch: int
    seed: int | None
    steps: list[StepRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    status: str = "success"  # "success" (ran to completion) | "error"
    error: str | None = None
    inference_latencies: list[float] = field(default_factory=list)


def _effective_control_hz(
    chunk_hz: float | None, task_hz: float | None, embodiment_hz: float | None
) -> float | None:
    """First non-None of chunk → task → embodiment rate (R1)."""
    for hz in (chunk_hz, task_hz, embodiment_hz):
        if hz is not None:
            return hz
    return None


def rollout(
    policy: Policy,
    embodiment: Embodiment,
    scene: Scene,
    *,
    max_steps: int,
    seed: int | None,
    epoch: int,
    controller: Controller,
    approver: Approver,
    sink: LogSink,
    control_hz: float | None = None,
) -> TrialRecord:
    """Run a single trial and return its record."""
    record = TrialRecord(scene_id=scene.id, epoch=epoch, seed=seed)
    store: dict[str, Any] = {}

    policy.reset(scene)
    obs = embodiment.reset(scene, seed=seed)

    t = 0
    while t < max_steps:
        action = controller.next_action(policy, obs, t, store)
        action = approver.review(action, store)
        result: StepResult = embodiment.step(action)
        sink.log_step(t, obs, action, result)
        record.steps.append(StepRecord(t=t, observation=obs, action=action, result=result))
        t += 1

        if result.terminated:
            record.terminated = True
            record.termination_reason = result.termination_reason
            break
        if result.truncated:
            record.truncated = True
            record.termination_reason = result.termination_reason or "truncated"
            break
        obs = result.observation
    else:
        record.truncated = True
        record.termination_reason = "max_steps"

    latencies = store.get("_controller_inference_latencies", [])
    record.inference_latencies = list(latencies)
    # ``control_hz`` / SELF_PACED are wired here; real-time pacing (sleep) is added
    # in rollout hardening so the test suite stays fast.
    _ = _effective_control_hz(None, control_hz, embodiment.info.control_hz)
    _ = SELF_PACED
    return record
