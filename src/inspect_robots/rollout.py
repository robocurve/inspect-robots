"""The rollout engine — the closed control loop at the heart of Inspect Robots.

One [`rollout`][inspect_robots.rollout.rollout] runs a single trial (one scene, one epoch): it
drives the policy↔embodiment loop through the [`Controller`][inspect_robots.controller.Controller]
(open-loop chunk execution) and the [`Approver`][inspect_robots.approver.Approver] safety
gate, logging each step to the sinks, and returns an immutable
[`TrialRecord`][inspect_robots.rollout.TrialRecord] that scorers consume.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from inspect_robots.approver import Approver
from inspect_robots.controller import Controller
from inspect_robots.embodiment import SELF_PACED, Embodiment
from inspect_robots.errors import EmbodimentFault, InspectRobotsError, PolicyError
from inspect_robots.frames import FrameRef, FrameStore
from inspect_robots.policy import Policy
from inspect_robots.scene import Scene
from inspect_robots.transcript import (
    Event,
    error_event,
    inference_event,
    reset_event,
    step_event,
)
from inspect_robots.types import Action, Observation, StepResult

if TYPE_CHECKING:
    from inspect_robots.logging.sink import LogSink


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
    """One step of a recorded trajectory.

    When a [`FrameStore`][inspect_robots.frames.FrameStore] is used, ``observation`` has its
    images stripped and ``image_refs`` holds on-disk handles instead (R5).
    """

    t: int
    observation: Observation
    action: Action
    result: StepResult
    image_refs: Mapping[str, FrameRef] | None = None


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
    # Human operator's success verdict, captured once during rollout (R6). Read
    # by OperatorScorer; remains None for unattended/CI runs.
    operator_judgement: str | None = None
    # Typed transcript of what happened during the trial.
    events: list[Event] = field(default_factory=list)


def _effective_control_hz(
    chunk_hz: float | None, task_hz: float | None, embodiment_hz: float | None
) -> float | None:
    """First non-None of chunk → task → embodiment rate (R1)."""
    for hz in (chunk_hz, task_hz, embodiment_hz):
        if hz is not None:
            return hz
    return None


def _store_frames(
    frame_store: FrameStore | None, trial_id: str, t: int, obs: Observation
) -> tuple[Observation, Mapping[str, FrameRef] | None]:
    """If a frame store is configured, stream images to disk and strip them."""
    if frame_store is None or not obs.images:
        return obs, None
    refs = {cam: frame_store.put(trial_id, t, cam, image) for cam, image in obs.images.items()}
    return replace(obs, images={}), refs


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
    frame_store: FrameStore | None = None,
) -> TrialRecord:
    """Run a single trial and return its record.

    Generic exceptions raised by the policy are wrapped as
    [`PolicyError`][inspect_robots.errors.PolicyError]; by the embodiment as
    [`EmbodimentFault`][inspect_robots.errors.EmbodimentFault]. Already-typed Inspect Robots errors
    (incl. [`SafetyAbort`][inspect_robots.errors.SafetyAbort]) propagate unchanged, so the
    eval orchestrator can apply the correct continue-vs-halt policy.
    """
    trial_id = f"{scene.id}-e{epoch}"
    record = TrialRecord(scene_id=scene.id, epoch=epoch, seed=seed)
    record.events.append(reset_event(seed))
    store: dict[str, Any] = {}

    policy.reset(scene)
    obs = embodiment.reset(scene, seed=seed)

    t = 0
    while t < max_steps:
        prev_inferences = len(store.get("_controller_inferences", []))
        try:
            action = controller.next_action(policy, obs, t, store)
        except InspectRobotsError:
            raise
        except Exception as exc:
            record.events.append(error_event(t, "PolicyError", str(exc)))
            raise PolicyError(str(exc)) from exc

        inferences = store.get("_controller_inferences", [])
        if len(inferences) > prev_inferences:
            latency, chunk_len = inferences[-1]
            record.events.append(inference_event(t, latency, chunk_len))

        action = approver.review(action, store)  # may raise SafetyAbort

        try:
            result: StepResult = embodiment.step(action)
        except InspectRobotsError:
            raise
        except Exception as exc:
            record.events.append(error_event(t, "EmbodimentFault", str(exc)))
            raise EmbodimentFault(str(exc)) from exc

        sink.log_step(t, obs, action, result)
        obs_rec, refs = _store_frames(frame_store, trial_id, t, obs)
        record.steps.append(
            StepRecord(t=t, observation=obs_rec, action=action, result=result, image_refs=refs)
        )
        record.events.append(
            step_event(t, result.terminated, result.truncated, result.termination_reason)
        )
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

    record.inference_latencies = [
        lat for lat, _ in store.get("_controller_inferences", []) if lat is not None
    ]
    # ``control_hz`` / SELF_PACED are wired here; real-time pacing (sleep) is added
    # with a real-robot adapter so the test suite stays fast.
    _ = _effective_control_hz(None, control_hz, embodiment.info.control_hz)
    _ = SELF_PACED
    return record
