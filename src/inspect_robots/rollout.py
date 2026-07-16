"""The rollout engine — the closed control loop at the heart of Inspect Robots.

One [`rollout`][inspect_robots.rollout.rollout] runs a single trial (one scene, one epoch): it
drives the policy↔embodiment loop through the [`Controller`][inspect_robots.controller.Controller]
(open-loop chunk execution) and the [`Approver`][inspect_robots.approver.Approver] safety
gate, logging each step to the sinks, and returns an immutable
[`TrialRecord`][inspect_robots.rollout.TrialRecord] that scorers consume.
"""

from __future__ import annotations

import json
import warnings
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from inspect_robots.approver import Approver
from inspect_robots.controller import _INFER_KEY, Controller
from inspect_robots.embodiment import Embodiment
from inspect_robots.errors import (
    EmbodimentFault,
    InspectRobotsError,
    PolicyError,
    SafetyAbort,
    _CancelledTrial,
)
from inspect_robots.frames import FrameRef, FrameStore
from inspect_robots.policy import Policy
from inspect_robots.scene import Scene
from inspect_robots.transcript import (
    Event,
    approval_event,
    error_event,
    inference_event,
    reset_event,
    step_event,
)
from inspect_robots.types import Action, Observation, StepResult

if TYPE_CHECKING:
    from inspect_robots.logging.sink import LogSink

_TRANSCRIPT_BYTE_LIMIT = 2 * 1024 * 1024


def derive_seed(eval_seed: int | None, scene_seed: int | None, epoch: int) -> int:
    """Deterministically combine eval/scene seeds and the epoch index (R2).

    Distinct epochs of the same scene get distinct seeds so repeats actually vary
    for stochastic policies, while a fixed ``(eval_seed, scene_seed, epoch)``
    reproduces bitwise. ``None`` and ``0`` hash differently, so an unseeded
    input does not silently alias ``seed=0``.
    """
    payload = f"{eval_seed}:{scene_seed}:{epoch}".encode()
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
    status: str = "success"  # "success" (ran to completion) | "error" | "cancelled"
    error: str | None = None
    inference_latencies: list[float] = field(default_factory=list)
    # Human operator's success verdict, captured once during rollout (R6). Read
    # by OperatorScorer; remains None for unattended/CI runs.
    operator_judgement: str | None = None
    # Typed transcript of what happened during the trial.
    events: list[Event] = field(default_factory=list)
    # The policy's optional per-trial audit record (e.g. an LLM conversation),
    # collected via the duck-typed transcript() hook and normalized to plain
    # JSON types by _collect_transcript. None when the policy has no hook.
    policy_transcript: Any = None


def _collect_transcript(policy: object) -> Any:
    """Normalize a policy's optional audit hook without affecting trial outcome."""
    try:
        transcript = getattr(policy, "transcript", None)
        if not callable(transcript):
            return None
        raw = transcript()
        if raw is None:
            return None
        dumped = json.dumps(raw, default=str)
        normalized = json.loads(dumped)
        size = len(dumped.encode())
        if size > _TRANSCRIPT_BYTE_LIMIT:
            return {
                "transcript_dropped": True,
                "bytes": size,
                "note": "exceeds inline limit; policies must not embed binary data",
            }
        return normalized
    except Exception as exc:
        try:
            detail = f"{type(exc).__name__}: {exc}"
        except Exception:
            detail = type(exc).__name__
        return {"transcript_error": detail}


def _effective_control_hz(
    chunk_hz: float | None, task_hz: float | None, embodiment_hz: float | None
) -> float | None:
    """First non-None of chunk → task → embodiment rate (R1).

    Real-time pacing (sleeping the control loop to this rate, honoring the
    ``SELF_PACED`` capability) is wired up together with the first real-robot
    adapter; until then the test suite stays fast and this helper is unused by
    the loop below.
    """
    for hz in (chunk_hz, task_hz, embodiment_hz):
        if hz is not None:
            return hz
    return None


def _record_failure(record: TrialRecord, exc: InspectRobotsError, t: int) -> InspectRobotsError:
    """Mark ``record`` failed and attach it to ``exc`` (see ``InspectRobotsError.record``).

    The partial record — steps walked and transcript events up to the failure —
    is forensic data the orchestrator preserves in the eval log.
    """
    message = str(exc)
    record.events.append(error_event(t, type(exc).__name__, message))
    record.status = "error"
    record.error = f"{type(exc).__name__}: {message}"
    exc.record = record
    return exc


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
    [`EmbodimentFault`][inspect_robots.errors.EmbodimentFault]; by the approver as
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort] (an approver that crashed cannot
    vouch for safety). Already-typed Inspect Robots errors (incl.
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort]) propagate unchanged, so the
    eval orchestrator can apply the correct continue-vs-halt policy. Every error
    raised from inside the trial carries the partial ``TrialRecord`` on
    ``exc.record`` for the orchestrator to preserve.

    ``control_hz`` is accepted for R1's rate-precedence chain; real-time pacing
    lands with the first real-robot adapter (see ``_effective_control_hz``).
    """
    trial_id = f"{scene.id}-e{epoch}"
    record = TrialRecord(scene_id=scene.id, epoch=epoch, seed=seed)
    record.events.append(reset_event(seed))
    store: dict[str, Any] = {}
    expected_dim = embodiment.info.action_space.dim
    policy_reset_ok = False
    delta_hook: Any = getattr(policy, "transcript_delta", None)
    messages_hook: Any = getattr(sink, "log_policy_messages", None)
    stream_ok = callable(delta_hook) and callable(messages_hook)

    try:
        t = -1
        try:
            policy.reset(scene)
            policy_reset_ok = True
        except InspectRobotsError as exc:
            _record_failure(record, exc, -1)
            raise
        except Exception as exc:
            raise _record_failure(record, PolicyError(str(exc)), -1) from exc
        try:
            obs = embodiment.reset(scene, seed=seed)
        except InspectRobotsError as exc:
            _record_failure(record, exc, -1)
            raise
        except Exception as exc:
            raise _record_failure(record, EmbodimentFault(str(exc)), -1) from exc

        t = 0
        while t < max_steps:
            prev_inferences = len(store.get(_INFER_KEY, []))
            try:
                action = controller.next_action(
                    policy, replace(obs, extra={**obs.extra, "env_step": t}), t, store
                )
            except InspectRobotsError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, PolicyError(str(exc)), t) from exc

            inferences = store.get(_INFER_KEY, [])
            if len(inferences) > prev_inferences:
                latency, chunk_len = inferences[-1]
                record.events.append(inference_event(t, latency, chunk_len))
                if stream_ok:
                    try:
                        delta = delta_hook()
                        entries = list(delta) if delta is not None else []
                        if entries:
                            messages_hook(t, entries)
                    except Exception as exc:
                        stream_ok = False
                        warnings.warn(
                            "Live policy transcript streaming disabled after "
                            f"{type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            # A malformed action is the policy's fault; catching it here keeps it
            # from surfacing inside the approver/embodiment as a halting fault.
            emitted_dim = int(np.asarray(action.data).size)
            if emitted_dim != expected_dim:
                raise _record_failure(
                    record,
                    PolicyError(
                        f"policy emitted a {emitted_dim}-D action but embodiment "
                        f"{embodiment.info.name!r} expects {expected_dim}-D"
                    ),
                    t,
                )

            # Policy-requested stop (plan 0008 §3d), captured from the
            # PRE-review action so an approver rewrite cannot erase the
            # intent. Note: EnsemblingController rebuilds actions with chunk
            # meta, so this channel works under Default/SmoothingController
            # (which preserve per-action meta), not under ensembling.
            requested_stop = bool(action.meta.get("request_stop"))
            stop_reason = str(action.meta.get("stop_reason", "policy_stop"))

            try:
                reviewed = approver.review(action, store)  # may raise SafetyAbort
            except InspectRobotsError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, SafetyAbort(str(exc)), t) from exc
            if reviewed is not action:
                flags = [k for k in ("clamped", "delta_clamped") if reviewed.meta.get(k)]
                detail = ", ".join(flags) or None
                record.events.append(approval_event(t, modified=True, detail=detail))
            action = reviewed

            try:
                result: StepResult = embodiment.step(action)
            except InspectRobotsError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, EmbodimentFault(str(exc)), t) from exc

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
            if requested_stop:
                # Embodiment-reported termination above wins (ground truth);
                # otherwise the policy's stop ends the trial as a truncation —
                # scoring stays the scorer's job, done() is not success.
                record.truncated = True
                record.termination_reason = stop_reason
                break
            obs = result.observation
        else:
            record.truncated = True
            record.termination_reason = "max_steps"
    except KeyboardInterrupt as exc:
        record.status = "cancelled"
        record.error = "cancelled by user (KeyboardInterrupt)"
        record.events.append(error_event(t, "KeyboardInterrupt", "cancelled by user"))
        raise _CancelledTrial(record.error, record) from exc
    finally:
        # Preserve measured latencies even when the trial ends in an error.
        record.inference_latencies = [
            lat for lat, _ in store.get(_INFER_KEY, []) if lat is not None
        ]
        if policy_reset_ok:  # pragma: no branch - false only while an exception unwinds
            record.policy_transcript = _collect_transcript(policy)
    return record
