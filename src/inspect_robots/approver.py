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

from inspect_robots.errors import SafetyAbort
from inspect_robots.spaces import Box
from inspect_robots.types import Action


@runtime_checkable
class Approver(Protocol):
    """Reviews an action before it reaches the embodiment.

    May return the action unchanged, return a modified (e.g. clamped) action, or
    raise [`SafetyAbort`][inspect_robots.errors.SafetyAbort] to halt the eval.
    """

    def review(self, action: Action, store: dict[str, Any]) -> Action: ...


class AutoApprover:
    """Approve every action unchanged (the permissive default)."""

    def review(self, action: Action, store: dict[str, Any]) -> Action:
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
