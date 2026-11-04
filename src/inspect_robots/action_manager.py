"""Action managers: the rollout middleware layer (Inspect's ``@solver`` analog).

An [`ActionManager`][inspect_robots.action_manager.ActionManager] owns the
per-control-step decision of which action to send to the embodiment. It decides
*when* to call ``policy.act()`` (a slow VLA inference returning an
[`ActionChunk`][inspect_robots.types.ActionChunk]), buffers the returned chunk,
and pops the next action each step. This single-method, stateful shape (R3) is
what lets advanced action managers, such as a temporal-ensembling manager that
re-infers every step and blends overlapping predictions, compose without forking
the rollout loop.

A terminal chunk (``chunk.terminal`` set) ends the trial: the manager marks the
last action it executes from that chunk with ``Action.terminal``, or, for a
chunk with no actions, returns the [`Termination`][inspect_robots.types.Termination]
itself so the rollout ends the trial without stepping the embodiment.

``DefaultActionManager`` plays the first ``replan_interval`` actions of each chunk,
then re-infers (``replan_interval=None`` ⇒ play the whole chunk before replanning).
A terminal chunk ends the trial after the actions it plays, so a short
``replan_interval`` never drops the policy's decision to stop.

The paper's action manager also decides, each time an action completes, whether
to rerun the policy on the latest observation. ``DefaultActionManager`` asks its
[`ReplanTrigger`][inspect_robots.action_manager.ReplanTrigger]s before playing
the next buffered action; when one fires, the rest of the chunk is dropped, the
policy is called on the current observation (which carries the trigger's
``extra["replan"]``), and the rollout records a ``replan`` event with the reason.
A pending stop in the dropped actions is dropped with them on purpose: the
policy chose to stop for a world the trigger says has changed, so it decides
again from the new observation.
Built in: [`UnreachableTarget`][inspect_robots.action_manager.UnreachableTarget]
and [`GripperSlip`][inspect_robots.action_manager.GripperSlip]. Reaching the end
of a chunk always replans and needs no trigger.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from itertools import count
from typing import Any, Protocol, runtime_checkable

import numpy as np

from inspect_robots.policy import Policy
from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk, Observation, Termination

_BUFFER_KEY = "_controller_action_buffer"
# Each entry is (inference_latency_s | None, chunk_len): one per policy.act() call.
_INFER_KEY = "_controller_inferences"
# Guard modifications, {"t", "detail"}, appended by the rollout.
_APPROVALS_KEY = "_rollout_approvals"
# Replans a trigger caused, {"t", "trigger", "reason"}; the rollout records each as an event.
_REPLAN_KEY = "_controller_replans"
# The action most recently returned for execution, and the step its chunk began.
_LAST_ACTION_KEY = "_controller_last_action"
# The action the rollout actually sent to the embodiment, after smoothing and
# guard review; set by the rollout just before each step.
_EXECUTED_ACTION_KEY = "_rollout_executed_action"
# How little (as a share of GripperSlip.threshold) the executed gripper command
# may move between steps for the grasp to count as held.
_EXECUTED_TOLERANCE = 0.1
_CHUNK_START_KEY = "_controller_chunk_start"


@runtime_checkable
class ActionManager(Protocol):
    """Decides the next action to execute, calling the policy as needed."""

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action | Termination:
        """Choose step ``t``'s action, using ``store`` for trial-local action manager state.

        Returns a [`Termination`][inspect_robots.types.Termination] instead of an
        action when the policy ends the trial without another action to execute.
        """
        ...


def _empty_chunk(policy: Policy) -> Exception:
    from inspect_robots.errors import PolicyError

    return PolicyError(f"policy {policy.info.name!r} returned an empty ActionChunk")


def _terminal_actions(chunk: ActionChunk, actions: list[Action]) -> list[Action]:
    """Mark the last action this manager will execute from a terminal chunk."""
    if chunk.terminal is not None and actions:
        actions[-1] = replace(actions[-1], terminal=chunk.terminal)
    return actions


@runtime_checkable
class ReplanTrigger(Protocol):
    """Decides, before the next buffered action plays, that the policy must look again."""

    name: str

    def should_replan(self, observation: Observation, t: int, store: dict[str, Any]) -> str | None:
        """Return why the policy should rerun on ``observation`` now, or ``None`` to continue.

        ``store`` is the trial's action-manager state; a trigger keeps its own
        state there under a key unique to it.
        """
        ...


class UnreachableTarget:
    """Replan when the guard clamped an action of the current chunk to the action bounds.

    A clamped action means the policy asked for a target outside what the
    embodiment allows, so the rest of the chunk is heading somewhere it cannot
    go. Speed-limit (delta) clamps alone do not count.
    """

    name = "unreachable_target"

    def should_replan(self, observation: Observation, t: int, store: dict[str, Any]) -> str | None:
        """Fire on the first bounds clamp since the current chunk began."""
        start = store.get(_CHUNK_START_KEY, 0)
        for approval in store.get(_APPROVALS_KEY, []):
            detail = approval.get("detail") or ""
            if approval.get("t", -1) >= start and "clamped" in detail.split(", "):
                return (
                    f"the guard clamped the action at step {approval['t']} to the action "
                    "bounds; the target is out of reach"
                )
        return None


class GripperSlip:
    """Replan when a gripper moves toward its command although no new command was sent.

    Holding an object, a gripper rests short of its closing command. If, after
    ``settle_steps`` steps of an unchanged command, the measured value moves
    more than ``threshold`` toward the command, the object has likely slipped
    out. ``state_key`` and ``index`` locate the gripper in the observation
    state and in the action vector, which must be absolute targets.
    """

    def __init__(
        self,
        state_key: str,
        index: int,
        *,
        label: str = "gripper",
        threshold: float = 0.05,
        settle_steps: int = 5,
    ) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and > 0")
        if isinstance(settle_steps, bool) or not isinstance(settle_steps, int) or settle_steps < 0:
            raise ValueError("settle_steps must be an integer >= 0")
        self.name = "gripper_slip"
        self.state_key = state_key
        self.index = index
        self.label = label
        self.threshold = threshold
        self.settle_steps = settle_steps
        self._key = f"_gripper_slip:{state_key}:{index}"

    @classmethod
    def for_embodiment(
        cls, action_space: Box, state_key: str, *, threshold: float = 0.05, settle_steps: int = 5
    ) -> list[GripperSlip]:
        """One trigger per action dimension whose label contains ``"gripper"``."""
        labels = action_space.semantics.dim_labels if action_space.semantics else None
        return [
            cls(state_key, index, label=label, threshold=threshold, settle_steps=settle_steps)
            for index, label in enumerate(labels or ())
            if "gripper" in label.lower()
        ]

    def should_replan(self, observation: Observation, t: int, store: dict[str, Any]) -> str | None:
        """Compare the gripper with its reading once the unchanged command has settled.

        The command is the policy's requested gripper target. Settling counts
        only while the command the embodiment actually executed (after smoothing
        and any guard) has stopped moving, so a delta limiter or a smoother still
        walking the gripper is never read as a slip, while a smoothed command
        that only approaches its target, or a clamp that holds an out-of-bounds
        request at the bound, still lets the check run.
        """
        last = store.get(_LAST_ACTION_KEY)
        state = observation.state.get(self.state_key)
        if last is None or state is None:
            return None
        command = float(np.asarray(last.data, dtype=np.float64).ravel()[self.index])
        executed_action = store.get(_EXECUTED_ACTION_KEY, last)
        executed = float(np.asarray(executed_action.data, dtype=np.float64).ravel()[self.index])
        measured = float(np.asarray(state, dtype=np.float64).ravel()[self.index])
        tracked = store.get(self._key)
        if tracked is None or tracked["command"] != command:
            store[self._key] = {
                "command": command,
                "executed": executed,
                "steady": 0,
                "baseline": None,
            }
            return None
        moved = abs(executed - tracked["executed"])
        tracked["executed"] = executed
        if moved > self.threshold * _EXECUTED_TOLERANCE:
            # The executed command is still moving: not yet a held grasp.
            tracked["steady"] = 0
            tracked["baseline"] = None
            return None
        tracked["steady"] += 1
        if tracked["steady"] < self.settle_steps:
            return None
        if tracked["baseline"] is None:
            tracked["baseline"] = measured
            tracked["baseline_executed"] = executed
            return None
        # Movement the executed command itself made toward the target (a slow
        # limiter creeping below the settle tolerance) is commanded, not slip.
        commanded = abs(tracked["baseline_executed"] - command) - abs(executed - command)
        closer = abs(tracked["baseline"] - command) - abs(measured - command) - commanded
        if closer < 0:
            # The gripper lagged the command (it closed on something): measure
            # later movement from here, so a stalled stretch never cancels a slip.
            tracked["baseline"] = measured
            tracked["baseline_executed"] = executed
            return None
        if closer > self.threshold:
            tracked["baseline"] = measured
            tracked["baseline_executed"] = executed
            return (
                f"{self.label} moved {closer:.3f} toward its command of {command:g} with no "
                "new command; the grasp may have slipped"
            )
        return None


class DefaultActionManager:
    """Open-loop chunk execution with periodic replanning and optional replan triggers."""

    def __init__(
        self,
        replan_interval: int | None = None,
        triggers: Sequence[ReplanTrigger] = (),
    ):
        if replan_interval is not None and (
            not isinstance(replan_interval, int)
            or isinstance(replan_interval, bool)
            or replan_interval < 1
        ):
            raise ValueError("replan_interval must be an integer >= 1 or None")
        for trigger in triggers:
            if not callable(getattr(trigger, "should_replan", None)):
                raise ValueError(f"replan trigger {trigger!r} has no should_replan method")
        self.replan_interval = replan_interval
        self.triggers = tuple(triggers)

    def _triggered(
        self, observation: Observation, t: int, store: dict[str, Any]
    ) -> dict[str, Any] | None:
        for trigger in self.triggers:
            reason = trigger.should_replan(observation, t, store)
            if reason:
                replan = {"t": t, "trigger": trigger.name, "reason": reason}
                store.setdefault(_REPLAN_KEY, []).append(replan)
                return replan
        return None

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action | Termination:
        """Play buffered actions until the chunk or interval ends, or a trigger fires."""
        buffer: deque[Action] = store.setdefault(_BUFFER_KEY, deque())
        if buffer and self.triggers:
            replan = self._triggered(observation, t, store)
            if replan is not None:
                buffer.clear()
                observation = replace(
                    observation,
                    extra={
                        **observation.extra,
                        "replan": {"trigger": replan["trigger"], "reason": replan["reason"]},
                    },
                )
        if not buffer:
            store[_CHUNK_START_KEY] = t
            chunk = policy.act(observation)
            if not chunk.actions:
                if chunk.terminal is None:
                    raise _empty_chunk(policy)
                store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, 0))
                return chunk.terminal
            take = self.replan_interval or len(chunk)
            taken = _terminal_actions(chunk, list(chunk.actions)[:take])
            buffer.extend(taken)
            # Record the number of actions actually buffered (the chunk may be
            # shorter than replan_interval), so the transcript never overstates.
            store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, len(taken)))
        action = buffer.popleft()
        store[_LAST_ACTION_KEY] = action
        return action


_SMOOTH_KEY = "_smoothing_prev_action"
# Distinguishes stacked instances' keys; process-local and never persisted.
_smooth_serial = count()


class SmoothingActionManager:
    """Wrap another action manager and exponentially smooth its action stream.

    Demonstrates the middleware composition the single-method interface enables:
    the wrapped action manager owns inference/replanning while this layer applies an
    exponential moving average (``alpha`` toward the new action) on top. Only
    valid for additive/continuous action spaces (the caller's responsibility).
    """

    def __init__(self, inner: ActionManager, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.inner = inner
        self.alpha = alpha
        # Per instance, not per module: this is the one action manager state written
        # destructively rather than appended to, so two smoothing layers in the
        # same chain would otherwise overwrite each other's previous action.
        self._store_key = f"{_SMOOTH_KEY}:{next(_smooth_serial)}"

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action | Termination:
        """Apply an exponential moving average to the wrapped action manager's output."""
        action = self.inner.next_action(policy, observation, t, store)
        if isinstance(action, Termination):
            return action
        raw = np.asarray(action.data, dtype=np.float64)
        prev = store.get(self._store_key)
        smoothed = raw if prev is None else self.alpha * raw + (1 - self.alpha) * prev
        store[self._store_key] = smoothed
        return replace(action, data=smoothed)


_ENSEMBLE_KEY = "_ensemble_chunks"
# Control modes whose actions live in a vector space and may be linearly averaged.
_AVERAGEABLE_MODES = frozenset(
    {"joint_pos", "joint_vel", "joint_delta", "eef_delta_pos", "eef_abs_pose", "eef_delta_pose"}
)
# Rotation representations that survive linear averaging. "none" has no rotation;
# "rot6d" is averaged un-normalized here on the assumption the consumer applies
# Gram-Schmidt on decode (true for the standard rot6d->matrix path).
_AVERAGEABLE_ROT = frozenset({"none", "rot6d"})


class EnsemblingActionManager:
    """ACT/ALOHA-style temporal ensembling over overlapping action chunks.

    Queries the policy every control step and blends, for the current step, the
    predictions of all still-relevant recent chunks. A chunk queried at global
    step ``q`` predicts step ``t`` via its action at index ``t - q`` (valid while
    ``0 <= t - q < len(chunk)``). Predictions are weighted ``exp(-m * i)`` with
    ``i = 0`` for the **oldest** contributing chunk (ALOHA's convention: older
    predictions dominate, which smooths motion); larger ``m`` favors the oldest.

    Only valid for additive action representations: the constructor refuses
    rotation reps and binary grippers that cannot be linearly averaged (R8).
    """

    def __init__(self, action_space: Box, m: float = 0.1):
        if isinstance(m, bool) or not math.isfinite(m) or m < 0:
            raise ValueError("m must be a finite number >= 0")
        self.action_space = action_space
        self.m = m
        sem = action_space.semantics
        if sem is None:
            warnings.warn(
                "EnsemblingActionManager: action space has no semantics; cannot "
                "verify that actions are safe to average.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if sem.control_mode not in _AVERAGEABLE_MODES:  # pragma: no cover
                # Defensive: every valid ControlMode literal is currently averageable.
                raise ValueError(
                    f"EnsemblingActionManager cannot average control_mode {sem.control_mode!r}"
                )
            if sem.rotation_repr not in _AVERAGEABLE_ROT:
                raise ValueError(
                    f"EnsemblingActionManager cannot linearly average rotation_repr "
                    f"{sem.rotation_repr!r}; only {sorted(_AVERAGEABLE_ROT)} are safe"
                )
            if sem.gripper == "binary":
                raise ValueError(
                    "EnsemblingActionManager cannot average a binary gripper; threshold "
                    "it downstream or use a continuous gripper"
                )

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action | Termination:
        """Query once for step ``t`` and blend every retained prediction for that step.

        A terminal chunk ends the trial after this step's blended action, or at
        once when it has no actions.
        """
        chunk = policy.act(observation)
        if not chunk.actions:
            if chunk.terminal is None:
                raise _empty_chunk(policy)
            store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, 0))
            return chunk.terminal
        store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, len(chunk)))

        buffer: list[tuple[int, list[Any], dict[str, Any]]] = store.setdefault(_ENSEMBLE_KEY, [])
        buffer.append(
            (
                t,
                [np.asarray(a.data, dtype=np.float64) for a in chunk.actions],
                dict(chunk.meta),
            )
        )
        # Keep only chunks that predict the current step; evict the stale ones.
        buffer[:] = [(q, acts, meta) for (q, acts, meta) in buffer if 0 <= t - q < len(acts)]
        # Oldest first (ascending query time) so weight index 0 is the oldest.
        buffer.sort(key=lambda e: e[0])

        predictions = [acts[t - q] for (q, acts, _meta) in buffer]
        weights = np.exp(-self.m * np.arange(len(predictions)))
        weights = weights / weights.sum()
        blended = np.average(np.stack(predictions), axis=0, weights=weights)
        newest_meta = buffer[-1][2]  # largest query time = newest chunk
        return Action(data=blended, meta=newest_meta, terminal=chunk.terminal)
