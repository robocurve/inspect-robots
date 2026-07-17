"""The Approver — a safety gate between policy output and the embodiment.

Every action passes through ``Approver.review`` before ``embodiment.step``. This
is the robotics analog of Inspect AI's ``ApprovalPolicy`` and is more
safety-critical: an approver may pass, clamp, or veto an action (a veto raises
[`SafetyAbort`][inspect_robots.errors.SafetyAbort]). In the tracer slice the default approver
passes everything through; clamping/operator approval land in rollout hardening.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspect_robots.errors import SafetyAbort
from inspect_robots.spaces import Box
from inspect_robots.types import Action


@runtime_checkable
class Approver(Protocol):
    """Reviews an action before it reaches the embodiment.

    May return the action unchanged, return a modified (e.g. clamped) action, or
    raise [`SafetyAbort`][inspect_robots.errors.SafetyAbort] to halt the eval.
    """

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Return the action to execute, or raise ``SafetyAbort`` to halt the evaluation."""
        ...


class AutoApprover:
    """Approve every action unchanged (the permissive default)."""

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Pass the action through without modifying its identity."""
        return action


class ClampApprover:
    """Clamp actions to a box's ``low``/``high`` bounds before they reach hardware.

    One-sided boxes are honored: a ``low``-only box clamps from below, a
    ``high``-only box from above. A modified action is flagged via
    ``action.meta["clamped"]`` so the rollout can record an approval event;
    when nothing clamps, the *same* action object is returned (the rollout
    detects modification by identity).

    Non-finite values are the safety cases: ``NaN`` anywhere in the action
    raises [`SafetyAbort`][inspect_robots.errors.SafetyAbort] — a NaN is a
    poisonous value with no meaningful clamp, and it must never reach hardware.
    ``±inf`` is *not* an abort: it clamps to the finite bound on that side like
    any other out-of-range value (and passes through if that side is
    unbounded).
    """

    def __init__(self, action_space: Box):
        self._space = action_space

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Reject NaNs and clamp bounded dimensions, preserving identity when unchanged."""
        data = np.asarray(action.data, dtype=np.float64)
        if bool(np.isnan(data).any()):
            raise SafetyAbort("ClampApprover: action contains NaN; refusing to pass it on")
        low, high = self._space.low, self._space.high
        if low is None and high is None:
            return action
        clamped = np.clip(data, low, high)
        if np.array_equal(clamped, data):
            return action
        return replace(action, data=clamped, meta={**dict(action.meta), "clamped": True})


# Control modes whose actions are absolute targets (delta-limited against the
# last approved action) vs displacements/rates (the action *is* the per-step
# change, limited directly). Together these cover every ControlMode literal.
_ABSOLUTE_MODES = frozenset({"joint_pos", "eef_abs_pose"})
_POSE_MODES = frozenset({"eef_abs_pose", "eef_delta_pose"})
# Absolute pose modes only: clamping an absolute euler/quat orientation per
# dimension has wraparound and axis-coupling problems, so those reps are
# refused. Displacement pose modes (eef_delta_pose) carry small rotation
# *deltas*, which clamp per dimension like any other bounded displacement.
_ABSOLUTE_POSE_MODES = _ABSOLUTE_MODES & _POSE_MODES
# Rotation reps that survive independent per-dimension clamping of an absolute
# orientation (same set the EnsemblingController accepts for per-dimension
# averaging).
_LIMITABLE_ROT = frozenset({"none", "rot6d"})
_LAST_APPROVED_KEY = "delta_limit:last"


class DeltaLimitApprover:
    """The "no wild swings" gate — semantics-aware per-step change limiting.

    Absolute-target modes (``joint_pos``, ``eef_abs_pose``) clamp each
    dimension to at most ``max_delta`` away from the **last approved action**
    of the trial; the first action passes through un-delta-limited (there is
    no trustworthy reference yet — bounds clamping still applies upstream).
    The derived default is 5% of ``high - low`` per step.

    Displacement/rate modes (``eef_delta_pos``, ``eef_delta_pose``,
    ``joint_delta``, ``joint_vel``) *are* the per-step change, so each
    dimension clamps to the intersection of the box and
    ``[-max_delta, +max_delta]``. The derived default is the box alone —
    core cannot assume displacement bounds are per-step-sized, so without an
    explicit ``max_delta`` the limiter adds nothing beyond ``ClampApprover``.

    Construction never guesses: missing semantics, a needed missing/non-finite
    bound without an explicit ``max_delta``, or an **absolute** pose mode whose
    rotation representation cannot be clamped per-dimension all raise
    ``ValueError`` (a displacement pose mode's rotation deltas clamp fine).
    ``NaN`` anywhere in a reviewed action raises
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort]. A modified action is
    flagged ``meta["delta_clamped"]``; an unmodified one is returned as the
    same object (rollout detects modification by identity). The reference
    lives in the rollout ``store`` (fresh per trial) under a namespaced key.
    """

    def __init__(self, action_space: Box, max_delta: float | Any | None = None):
        sem = action_space.semantics
        if sem is None:
            raise ValueError(
                "DeltaLimitApprover: action space declares no semantics; the "
                "limiter cannot tell absolute targets from displacements"
            )
        if sem.control_mode in _ABSOLUTE_POSE_MODES and sem.rotation_repr not in _LIMITABLE_ROT:
            raise ValueError(
                f"DeltaLimitApprover: cannot clamp absolute rotation_repr "
                f"{sem.rotation_repr!r} per dimension; only {sorted(_LIMITABLE_ROT)} "
                f"are safe (displacement pose modes carry rotation deltas and are fine)"
            )
        self._absolute = sem.control_mode in _ABSOLUTE_MODES
        dim = action_space.dim
        low, high = action_space.low, action_space.high

        explicit = _validate_max_delta(max_delta, dim) if max_delta is not None else None
        if self._absolute:
            if explicit is not None:
                self._delta = explicit
            else:
                if low is None or high is None or not bool(np.all(np.isfinite(high - low))):
                    raise ValueError(
                        "DeltaLimitApprover: deriving a default needs finite low/high "
                        "bounds; pass max_delta explicitly"
                    )
                self._delta = 0.05 * (high - low)
        else:
            if explicit is None and (low is None or high is None):
                raise ValueError(
                    "DeltaLimitApprover: a displacement-mode space without both "
                    "bounds needs an explicit max_delta"
                )
            self._low = low if explicit is None else _intersect(low, -explicit, np.maximum)
            self._high = high if explicit is None else _intersect(high, explicit, np.minimum)

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Limit per-step change, retaining absolute-mode history in trial state."""
        data = np.asarray(action.data, dtype=np.float64)
        if bool(np.isnan(data).any()):
            raise SafetyAbort("DeltaLimitApprover: action contains NaN; refusing to pass it on")
        if self._absolute:
            reference = store.get(_LAST_APPROVED_KEY)
            if reference is None:
                approved = data
            else:
                ref = np.asarray(reference, dtype=np.float64)
                approved = np.clip(data, ref - self._delta, ref + self._delta)
            store[_LAST_APPROVED_KEY] = approved
        else:
            approved = np.clip(data, self._low, self._high)
        if np.array_equal(approved, data):
            return action
        return replace(action, data=approved, meta={**dict(action.meta), "delta_clamped": True})


def _validate_max_delta(max_delta: float | Any, dim: int) -> npt.NDArray[np.float64]:
    try:
        arr = np.broadcast_to(np.asarray(max_delta, dtype=np.float64), (dim,))
    except ValueError as exc:
        raise ValueError(
            f"DeltaLimitApprover: max_delta does not broadcast to {dim} dimensions"
        ) from exc
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr <= 0)):
        raise ValueError("DeltaLimitApprover: max_delta must be finite and > 0")
    return arr


def _intersect(
    bound: npt.NDArray[np.floating[Any]] | None,
    limit: npt.NDArray[np.float64],
    tighter: Any,
) -> npt.NDArray[np.float64]:
    """Combine an optional box side with the symmetric limit, keeping the tighter."""
    result: npt.NDArray[np.float64] = limit if bound is None else tighter(bound, limit)
    return result


class ChainApprover:
    """Run approvers in sequence, feeding each the previous one's result.

    Gives "guardrails" one composable name, e.g.
    ``ChainApprover(ClampApprover(space), DeltaLimitApprover(space))``.
    Identity is preserved end to end: if no approver modifies the action, the
    original object comes back, so the rollout's modification check still works.
    """

    def __init__(self, *approvers: Approver):
        self._approvers = approvers

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Apply each configured gate in order to the preceding gate's output."""
        for approver in self._approvers:
            action = approver.review(action, store)
        return action
