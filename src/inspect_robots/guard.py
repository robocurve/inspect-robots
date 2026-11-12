"""The Guard — a safety gate between policy output and the embodiment.

Every action passes through ``Guard.review`` before ``embodiment.step``. This
is the robotics analog of Inspect AI's ``ApprovalPolicy`` and is more
safety-critical: a guard may pass, clamp, or veto an action (a veto raises
[`SafetyAbort`][inspect_robots.errors.SafetyAbort]).

Guards are also configuration. ``guard`` is a registry kind whose factories
take the action space plus keyword parameters (``clamp`` and ``delta-limit``
are built in), and a guard spec ``NAME`` or ``NAME:key=value,key=value``
names one. [`build_guard`][inspect_robots.guard.build_guard] turns specs into a
[`GuardChain`][inspect_robots.guard.GuardChain], which records which guard
changed an action and the action before and after, so the record shows every
intervention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspect_robots.errors import ConfigError, SafetyAbort
from inspect_robots.spaces import ABSOLUTE_CONTROL_MODES, Box, RotationRepr
from inspect_robots.types import Action


@runtime_checkable
class Guard(Protocol):
    """Reviews an action before it reaches the embodiment.

    May return the action unchanged, return a modified (e.g. clamped) action, or
    raise [`SafetyAbort`][inspect_robots.errors.SafetyAbort] to halt the eval.
    """

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Return the action to execute, or raise ``SafetyAbort`` to halt the evaluation."""
        ...


@dataclass(frozen=True)
class GuardContribution:
    """Guards an embodiment adds to the CLI's default guardrail chain.

    ``guards`` pairs a short display name (shown in the ``guardrails:``
    banner) with each guard to append. ``warnings`` names contributions
    the embodiment declined to make or made in a degraded state, and why, so
    neither condition is invisible. Warnings may accompany active guards.

    A contributed guard must preserve the incoming ``Action`` object's
    identity when approving it unmodified. If it substitutes another target,
    such as holding an earlier pose, it must call
    ``DeltaLimitGuard.rewind_reference`` with that executed pose so later
    deltas are measured from reality. It vetoes by raising ``SafetyAbort``;
    other exceptions are bugs and propagate. The rollout enforces finiteness
    before review and again on the returned action, so contributed guards
    need not duplicate that validation.
    """

    guards: tuple[tuple[str, Guard], ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, guard in self.guards:
            if not name or "\n" in name or "\r" in name:
                raise ValueError("guardrail display names must be non-empty and newline-free")
            if not callable(getattr(guard, "review", None)):
                raise ValueError(f"guardrail guard {name!r} must have a callable review")


class PassThroughGuard:
    """Approve every action unchanged (the permissive default)."""

    name = "pass_through"

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Pass the action through without modifying its identity."""
        return action


class ClampGuard:
    """Clamp actions to a box's ``low``/``high`` bounds before they reach hardware.

    Registered as the ``clamp`` guard.

    One-sided boxes are honored: a ``low``-only box clamps from below, a
    ``high``-only box from above. A modified action is flagged via
    ``action.meta["clamped"]`` so the rollout can record an approval event;
    when nothing clamps, the *same* action object is returned (the rollout
    detects modification by identity).

    Non-finite values are the safety cases for direct ``review()`` callers,
    since rollout rejects non-finite policy actions before review: ``NaN``
    anywhere in the action raises
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort]. A NaN is a poisonous
    value with no meaningful clamp, and it must never reach hardware. ``±inf``
    is *not* an abort: it clamps to the finite bound on that side like any other
    out-of-range value (and passes through if that side is unbounded).
    """

    name = "clamp"

    def __init__(self, action_space: Box):
        self._space = action_space

    def config(self) -> dict[str, Any]:
        """The bounds this guard clamps to, as recorded in a trial's record."""
        low, high = self._space.low, self._space.high
        return {
            "low": None if low is None else low.tolist(),
            "high": None if high is None else high.tolist(),
        }

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Reject NaNs and clamp bounded dimensions, preserving identity when unchanged."""
        data = np.asarray(action.data, dtype=np.float64)
        if bool(np.isnan(data).any()):
            raise SafetyAbort("ClampGuard: action contains NaN; refusing to pass it on")
        low, high = self._space.low, self._space.high
        if low is None and high is None:
            return action
        clamped = np.clip(data, low, high)
        if np.array_equal(clamped, data):
            return action
        return replace(action, data=clamped, meta={**dict(action.meta), "clamped": True})


_POSE_MODES = frozenset({"eef_abs_pose", "eef_delta_pose"})
# Absolute pose modes only: clamping an absolute euler/quat orientation per
# dimension has wraparound and axis-coupling problems, so those reps are
# refused. Displacement pose modes (eef_delta_pose) carry small rotation
# *deltas*, which mostly clamp per dimension like any other bounded
# displacement — except reps whose no-op is not the origin (see
# _DISPLACEMENT_POSE_UNSAFE_ROT below).
_ABSOLUTE_POSE_MODES = ABSOLUTE_CONTROL_MODES & _POSE_MODES
_DISPLACEMENT_POSE_MODES = _POSE_MODES - _ABSOLUTE_POSE_MODES
# Rotation reps that survive independent per-dimension clamping of an absolute
# orientation (same set the EnsemblingActionManager accepts for per-dimension
# averaging).
_LIMITABLE_ROT = frozenset({"none", "rot6d"})
# Displacement pose modes carry rotation *deltas*, whose no-op is not the zero
# vector for these reps (a quaternion identity is (1, 0, 0, 0); a rot6d
# identity is (1, 0, 0, 0, 1, 0)). Clamping such a delta per dimension toward
# a symmetric ±max_delta box drags it away from identity, and downstream
# re-normalization (Gram-Schmidt orthonormalization for rot6d) can then
# amplify the rotation instead of limiting it — the same failure class as
# clamping an absolute quaternion. euler_xyz/axis_angle deltas have no such
# problem (their no-op is the zero vector) and clamp fine. Note rot6d is
# limitable for *absolute* orientations (_LIMITABLE_ROT above) but not for
# displacement deltas — the two sets are deliberately not related.
_DISPLACEMENT_POSE_UNSAFE_ROT = frozenset({"quat_wxyz", "quat_xyzw", "rot6d"})
# Derived from RotationRepr itself (not hand-listed) so a rep added there
# defaults to the safe side of this message without anyone remembering to
# update it — the refusal set above is still the one source of truth for
# what actually gets rejected.
_DISPLACEMENT_SAFE_ROT = frozenset(get_args(RotationRepr)) - _DISPLACEMENT_POSE_UNSAFE_ROT
_LAST_APPROVED_KEY = "delta_limit:last"


class DeltaLimitGuard:
    """The "no wild swings" gate — semantics-aware per-step change limiting.

    Absolute-target modes (``joint_pos``, ``eef_abs_pose``) clamp each
    dimension to at most ``max_delta`` away from the **last approved action**
    of the trial; the first action passes through un-delta-limited (there is
    no trustworthy reference yet — bounds clamping still applies upstream).
    The derived default is a declared ``ActionSemantics.max_step`` where
    present, otherwise 5% of ``high - low`` per step.

    Displacement/rate modes (``eef_delta_pos``, ``eef_delta_pose``,
    ``joint_delta``, ``joint_vel``) *are* the per-step change, so each
    dimension clamps to the intersection of the box and
    ``[-max_delta, +max_delta]``. The derived default is the box alone —
    core cannot assume displacement bounds are per-step-sized, so without an
    explicit ``max_delta`` the limiter adds nothing beyond ``ClampGuard``.

    Construction never guesses: missing semantics, a needed missing/non-finite
    bound without an explicit ``max_delta``, an absolute pose mode whose
    rotation representation cannot be clamped per-dimension, or a displacement
    pose mode carrying a quaternion or rot6d delta (whose identity is not the
    zero vector, so per-dimension clamping distorts it) all raise ``ValueError``.
    Non-finite values (``NaN`` or ``±inf``) anywhere in a reviewed action raise
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort]. Unlike ``ClampGuard``
    (which clamps ``±inf`` to finite box bounds), a delta limiter cannot clamp
    ``±inf`` into anything meaningful because it has no trustworthy finite bound
    on the first absolute step. A modified action is
    flagged ``meta["delta_clamped"]``; an unmodified one is returned as the
    same object (rollout detects modification by identity). The reference
    lives in the rollout ``store`` (fresh per trial) under a namespaced key.
    """

    name = "delta-limit"

    def __init__(self, action_space: Box, max_delta: float | Any | None = None):
        sem = action_space.semantics
        if sem is None:
            raise ValueError(
                "DeltaLimitGuard: action space declares no semantics; the "
                "limiter cannot tell absolute targets from displacements"
            )
        if sem.control_mode in _ABSOLUTE_POSE_MODES and sem.rotation_repr not in _LIMITABLE_ROT:
            raise ValueError(
                f"DeltaLimitGuard: cannot clamp absolute rotation_repr "
                f"{sem.rotation_repr!r} per dimension; only {sorted(_LIMITABLE_ROT)} "
                f"are safe"
            )
        if (
            sem.control_mode in _DISPLACEMENT_POSE_MODES
            and sem.rotation_repr in _DISPLACEMENT_POSE_UNSAFE_ROT
        ):
            raise ValueError(
                f"DeltaLimitGuard: cannot clamp displacement rotation_repr "
                f"{sem.rotation_repr!r} per dimension; its identity is not at the "
                f"origin, so per-dimension clamping distorts it instead of "
                f"limiting it; only {sorted(_DISPLACEMENT_SAFE_ROT)} are safe here"
            )
        self._absolute = sem.control_mode in ABSOLUTE_CONTROL_MODES
        dim = action_space.dim
        low, high = action_space.low, action_space.high

        explicit = (
            _validate_max_delta(max_delta, action_space.shape, dim)
            if max_delta is not None
            else None
        )
        if self._absolute:
            if explicit is not None:
                self._delta = explicit
            else:
                declared = sem.max_step or (None,) * dim
                undeclared = np.fromiter(
                    (entry is None for entry in declared), dtype=np.bool_, count=dim
                )
                if bool(np.any(undeclared)) and (low is None or high is None):
                    raise ValueError(
                        "DeltaLimitGuard: deriving a default needs finite low/high "
                        "bounds; pass max_delta explicitly"
                    )
                self._delta = np.empty(dim, dtype=np.float64)
                if bool(np.any(undeclared)):
                    assert low is not None and high is not None
                    low_undeclared = low.reshape(-1)[undeclared]
                    high_undeclared = high.reshape(-1)[undeclared]
                    with np.errstate(over="ignore", invalid="ignore"):
                        native_default = 0.05 * (high_undeclared - low_undeclared)
                    if not bool(
                        np.all(np.isfinite(low_undeclared))
                        and np.all(np.isfinite(high_undeclared))
                        and np.all(np.isfinite(native_default))
                    ):
                        raise ValueError(
                            "DeltaLimitGuard: deriving a default needs finite low/high "
                            "bounds; pass max_delta explicitly"
                        )
                    self._delta[undeclared] = native_default
                for index, entry in enumerate(declared):
                    if entry is not None:
                        self._delta[index] = entry
                # review() clips against the action's own shape; a flat delta
                # would fail to broadcast for multi-dimensional boxes.
                self._delta = self._delta.reshape(action_space.shape)
        else:
            if explicit is None and (low is None or high is None):
                raise ValueError(
                    "DeltaLimitGuard: a displacement-mode space without both "
                    "bounds needs an explicit max_delta"
                )
            self._low = low if explicit is None else _intersect(low, -explicit, np.maximum)
            self._high = high if explicit is None else _intersect(high, explicit, np.minimum)

    def config(self) -> dict[str, Any]:
        """The resolved limits, as recorded in a trial's record.

        Absolute modes report the per-dimension ``max_delta`` around the last
        approved action; displacement modes report the clamp box after
        intersecting the space with ``max_delta``.
        """
        if self._absolute:
            return {"mode": "absolute", "max_delta": self._delta.tolist()}
        return {
            "mode": "displacement",
            "low": None if self._low is None else np.asarray(self._low).tolist(),
            "high": None if self._high is None else np.asarray(self._high).tolist(),
        }

    @staticmethod
    def rewind_reference(store: dict[str, Any], pose: npt.NDArray[np.float64]) -> None:
        """Reset an existing limiter reference to a copy of an executed pose.

        A downstream guard that substitutes a target, such as holding an
        earlier safe pose, must use this before the next action is reviewed so
        the limiter measures subsequent deltas from the pose that actually ran.
        If the limiter has not established a reference, this is a no-op.
        """
        if not bool(np.all(np.isfinite(pose))):
            raise ValueError(
                "DeltaLimitGuard: rewind_reference received a non-finite pose; "
                "refusing to corrupt reference state"
            )
        if _LAST_APPROVED_KEY in store:
            store[_LAST_APPROVED_KEY] = pose.copy()

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Limit per-step change, retaining absolute-mode history in trial state."""
        data = np.asarray(action.data, dtype=np.float64)
        if not bool(np.all(np.isfinite(data))):
            raise SafetyAbort(
                "DeltaLimitGuard: action contains a non-finite value (NaN or ±inf); "
                "refusing to pass it on"
            )
        if self._absolute:
            reference = store.get(_LAST_APPROVED_KEY)
            if reference is None:
                approved = data
            else:
                ref = np.asarray(reference, dtype=np.float64)
                approved = np.clip(data, ref - self._delta, ref + self._delta)
            # A copy, as in rewind_reference: the first approved action is the
            # caller's own array, and a policy that reuses that buffer for its
            # next action would otherwise move the reference along with it.
            store[_LAST_APPROVED_KEY] = approved.copy()
        else:
            approved = np.clip(data, self._low, self._high)
        if np.array_equal(approved, data):
            return action
        return replace(action, data=approved, meta={**dict(action.meta), "delta_clamped": True})


def _validate_max_delta(
    max_delta: float | Any, shape: tuple[int, ...], dim: int
) -> npt.NDArray[np.float64]:
    raw = np.asarray(max_delta, dtype=np.float64)
    try:
        arr = np.broadcast_to(raw, shape)
    except ValueError:
        try:
            arr = np.broadcast_to(raw, (dim,)).reshape(shape)
        except ValueError as exc:
            raise ValueError(
                f"DeltaLimitGuard: max_delta does not broadcast to {shape} (or {dim} dimensions)"
            ) from exc
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr <= 0)):
        raise ValueError("DeltaLimitGuard: max_delta must be finite and > 0")
    # Shaped like the box for the same reason the derived default is: review()
    # clips against the action's own shape, and the displacement branch
    # intersects with bounds of that shape, so a flat delta fails to broadcast
    # for multi-dimensional boxes.
    return arr.reshape(shape)


def _intersect(
    bound: npt.NDArray[np.floating[Any]] | None,
    limit: npt.NDArray[np.float64],
    tighter: Any,
) -> npt.NDArray[np.float64]:
    """Combine an optional box side with the symmetric limit, keeping the tighter."""
    result: npt.NDArray[np.float64] = limit if bound is None else tighter(bound, limit)
    return result


# Where a GuardChain lists the guards that changed the action under review, as
# {"guard", "before", "after"}; the rollout moves them into the approval event.
_GUARD_TRACE_KEY = "_guard_trace"


def guard_name(guard: object) -> str:
    """A guard's ``name`` attribute, or its class name when it has none."""
    name = getattr(guard, "name", None)
    return name if isinstance(name, str) and name else type(guard).__name__


def _values(action: Action) -> list[float]:
    return [float(value) for value in np.asarray(action.data, dtype=np.float64).ravel()]


class GuardChain:
    """Run guards in sequence, feeding each the previous one's result.

    Gives "guardrails" one composable name, e.g.
    ``GuardChain(ClampGuard(space), DeltaLimitGuard(space))``.
    Identity is preserved end to end: if no guard modifies the action, the
    original object comes back, so the rollout's modification check still works.
    ``names`` labels the guards in the record (default: each guard's
    [`guard_name`][inspect_robots.guard.guard_name]). Each guard that changes the
    action is noted with the action before and after it.
    """

    name = "chain"

    def __init__(self, *guards: Guard, names: Sequence[str] | None = None):
        for guard in guards:
            if not callable(getattr(guard, "review", None)):
                raise ValueError(f"GuardChain element must have a callable review, got {guard!r}")
        if names is not None and len(names) != len(guards):
            raise ValueError(f"GuardChain got {len(names)} names for {len(guards)} guards")
        self.guards = guards
        self.names = tuple(names) if names is not None else tuple(guard_name(g) for g in guards)

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Apply each configured gate in order to the preceding gate's output.

        Every member that limits absolute deltas measures against the same
        reference: the action the previous chain review approved. Without this,
        an earlier limiter would advance the shared reference to its own output
        before a later, tighter limiter read it, so the tighter limit could never
        bind. After the chain, the reference is the chain's final output.
        """
        trace: list[dict[str, Any]] = store.setdefault(_GUARD_TRACE_KEY, [])
        had_reference = _LAST_APPROVED_KEY in store
        reference = store.get(_LAST_APPROVED_KEY)
        limited = had_reference
        for name, guard in zip(self.names, self.guards, strict=True):
            if had_reference:
                store[_LAST_APPROVED_KEY] = reference
            else:
                store.pop(_LAST_APPROVED_KEY, None)
            reviewed = guard.review(action, store)
            limited = limited or _LAST_APPROVED_KEY in store
            if reviewed is not action:
                trace.append({"guard": name, "before": _values(action), "after": _values(reviewed)})
            action = reviewed
        if limited:
            # The chain's output is what executes, including a substituted hold
            # target, so it is the next review's reference.
            store[_LAST_APPROVED_KEY] = np.asarray(action.data, dtype=np.float64).copy()
        return action


def parse_guard_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Split ``NAME`` or ``NAME:key=value,key=value`` into a name and parsed parameters.

    Values are parsed like CLI ``-P k=v`` values (booleans, numbers, ``none``,
    quoted strings).
    """
    from inspect_robots.defaults import _parse_value

    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name:
        raise ConfigError(f"guard spec {spec!r} has no name.\nfix: write NAME or NAME:k=v,k=v")
    params: dict[str, Any] = {}
    for item in filter(None, (part.strip() for part in rest.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(
                f"guard spec {spec!r}: {item!r} is not key=value.\nfix: write NAME:k=v,k=v"
            )
        params[key.strip()] = _parse_value(value.strip())
    return name, params


def build_guard(specs: Sequence[str | Guard], action_space: Box) -> Guard:
    """One guard from specs or guard objects, applied in order.

    A string is a registered guard spec (``NAME`` or ``NAME:k=v,...``); its
    factory is called with ``action_space`` and the parameters. An unknown
    name or a factory that rejects its parameters raises ``ConfigError``.
    """
    from inspect_robots.registry import registered, resolve

    guards: list[Guard] = []
    names: list[str] = []
    for spec in specs:
        if not isinstance(spec, str):
            guards.append(spec)
            names.append(guard_name(spec))
            continue
        name, params = parse_guard_spec(spec)
        if name not in registered("guard"):
            known = ", ".join(sorted(registered("guard"))) or "(none)"
            raise ConfigError(f"unknown guard {name!r}; registered guards: {known}")
        try:
            guard = resolve("guard", name, action_space=action_space, **params)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"guard {spec!r}: {exc}") from exc
        if not callable(getattr(guard, "review", None)):
            raise ConfigError(f"guard {name!r} factory returned {guard!r}, which has no review")
        guards.append(guard)
        names.append(name)
    return GuardChain(*guards, names=names)
