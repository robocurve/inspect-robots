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
"""

from __future__ import annotations

import math
import warnings
from collections import deque
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


class DefaultActionManager:
    """Open-loop chunk execution with periodic replanning."""

    def __init__(self, replan_interval: int | None = None):
        if replan_interval is not None and (
            not isinstance(replan_interval, int)
            or isinstance(replan_interval, bool)
            or replan_interval < 1
        ):
            raise ValueError("replan_interval must be an integer >= 1 or None")
        self.replan_interval = replan_interval

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action | Termination:
        """Reuse buffered chunk actions until the configured replanning boundary."""
        buffer: deque[Action] = store.setdefault(_BUFFER_KEY, deque())
        if not buffer:
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
        return buffer.popleft()


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
