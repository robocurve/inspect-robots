"""DeltaLimitApprover / ChainApprover tests (plan 0008 §3a).

The delta limiter is semantics-aware: absolute-target modes clamp around the
last approved action; displacement/rate modes clamp the action magnitude
itself. Construction never guesses — missing semantics, missing needed
bounds, or unaverageable rotation reps refuse loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.errors import SafetyAbort
from inspect_robots.spaces import ActionSemantics, Box, RotationRepr
from inspect_robots.types import Action


def _abs_space() -> Box:
    return Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=ActionSemantics("joint_pos"),
    )


def _delta_space() -> Box:
    # Asymmetric second dim ([0, 1]) mirrors a normalized-gripper delta dim.
    return Box(
        shape=(2,),
        low=np.array([-0.1, 0.0]),
        high=np.array([0.1, 1.0]),
        semantics=ActionSemantics("joint_delta"),
    )


# --- construction refusals: never guess -------------------------------------


def test_refuses_missing_semantics() -> None:
    with pytest.raises(ValueError, match="semantics"):
        DeltaLimitApprover(Box(shape=(2,), low=np.zeros(2), high=np.ones(2)))


def test_refuses_absolute_pose_mode_with_unaverageable_rotation() -> None:
    # Absolute euler/quat orientations have wraparound + axis coupling, so
    # per-dim clamping is unsound and refused.
    space = Box(
        shape=(7,),
        low=np.full(7, -1.0),
        high=np.full(7, 1.0),
        semantics=ActionSemantics("eef_abs_pose", rotation_repr="quat_wxyz"),
    )
    with pytest.raises(ValueError, match="rotation_repr"):
        DeltaLimitApprover(space)


@pytest.mark.parametrize("rotation_repr", ["euler_xyz", "quat_wxyz", "axis_angle"])
def test_delta_pose_rotation_deltas_clamp_per_dim(rotation_repr: RotationRepr) -> None:
    # eef_delta_pose carries small rotation *deltas*, not absolute orientations,
    # so any rotation_repr clamps per dimension like a bounded displacement (#143).
    # BridgeData V2 shape: 3 xyz + 3 euler deltas + 1 gripper.
    space = Box(
        shape=(7,),
        low=np.full(7, -0.1),
        high=np.full(7, 0.1),
        semantics=ActionSemantics("eef_delta_pose", rotation_repr=rotation_repr),
    )
    approver = DeltaLimitApprover(space, max_delta=0.05)
    # A large rotation-delta dim (dpitch, index 4) clamps to the ±max_delta box.
    out = approver.review(Action(data=np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])), {})
    assert np.isclose(out.data[4], 0.05)
    assert out.meta.get("delta_clamped") is True


def test_refuses_derived_default_without_bounds() -> None:
    for mode in ("joint_pos", "joint_delta"):
        unbounded = Box(shape=(2,), semantics=ActionSemantics(mode))
        with pytest.raises(ValueError, match="bound"):
            DeltaLimitApprover(unbounded)
        # An explicit max_delta removes the bounds requirement.
        DeltaLimitApprover(unbounded, max_delta=0.5)


def test_refuses_nonpositive_or_nonfinite_max_delta() -> None:
    with pytest.raises(ValueError, match="max_delta"):
        DeltaLimitApprover(_abs_space(), max_delta=0.0)
    with pytest.raises(ValueError, match="max_delta"):
        DeltaLimitApprover(_abs_space(), max_delta=float("inf"))
    with pytest.raises(ValueError, match="max_delta"):
        DeltaLimitApprover(_abs_space(), max_delta=np.array([0.1, -0.1]))
    with pytest.raises(ValueError, match="max_delta"):
        DeltaLimitApprover(_abs_space(), max_delta=np.array([0.1, 0.1, 0.1]))


# --- absolute-target modes ---------------------------------------------------


def test_absolute_first_action_passes_then_limits() -> None:
    approver = DeltaLimitApprover(_abs_space(), max_delta=0.1)
    store: dict[str, object] = {}
    first = Action(data=np.array([0.9, 0.9]))
    assert approver.review(first, store) is first  # no reference yet
    # Second action is limited to ±0.1 around the last approved one.
    out = approver.review(Action(data=np.array([0.0, 1.0])), store)
    assert np.allclose(out.data, [0.8, 1.0])
    assert out.meta.get("delta_clamped") is True


def test_absolute_identity_within_limit_and_reference_advances() -> None:
    approver = DeltaLimitApprover(_abs_space(), max_delta=0.5)
    store: dict[str, object] = {}
    approver.review(Action(data=np.array([0.0, 0.0])), store)
    small = Action(data=np.array([0.3, 0.4]))
    assert approver.review(small, store) is small  # identity when nothing clamps
    # Reference advanced to [0.3, 0.4]: a jump back to 0.9 clamps to 0.8.
    out = approver.review(Action(data=np.array([0.9, 0.4])), store)
    assert np.allclose(out.data, [0.8, 0.4])


def test_absolute_reference_is_the_approved_action_not_the_request() -> None:
    approver = DeltaLimitApprover(_abs_space(), max_delta=0.1)
    store: dict[str, object] = {}
    approver.review(Action(data=np.array([0.0, 0.0])), store)
    approver.review(Action(data=np.array([1.0, 0.0])), store)  # approved as 0.1
    out = approver.review(Action(data=np.array([1.0, 0.0])), store)
    assert np.allclose(out.data, [0.2, 0.0])  # walks, one clamped step at a time


def test_absolute_derived_default_is_five_percent_of_range() -> None:
    approver = DeltaLimitApprover(_abs_space())  # ranges: 2.0 and 1.0
    store: dict[str, object] = {}
    approver.review(Action(data=np.array([0.0, 0.5])), store)
    out = approver.review(Action(data=np.array([1.0, 0.0])), store)
    assert np.allclose(out.data, [0.1, 0.45])  # 0.05 * (high - low) per dim


def test_stores_are_isolated_per_trial() -> None:
    approver = DeltaLimitApprover(_abs_space(), max_delta=0.1)
    trial_a: dict[str, object] = {}
    approver.review(Action(data=np.array([0.9, 0.9])), trial_a)
    # A fresh store (new trial) has no reference: first action passes again.
    trial_b: dict[str, object] = {}
    big = Action(data=np.array([-0.9, 0.1]))
    assert approver.review(big, trial_b) is big


# --- displacement / rate modes ----------------------------------------------


def test_displacement_explicit_limit_intersects_box() -> None:
    approver = DeltaLimitApprover(_delta_space(), max_delta=0.05)
    out = approver.review(Action(data=np.array([0.08, -0.5])), {})
    # dim0: min(high=0.1, +0.05) → 0.05; dim1: max(low=0.0, -0.05) → 0.0.
    assert np.allclose(out.data, [0.05, 0.0])
    assert out.meta.get("delta_clamped") is True


def test_displacement_derived_default_is_box_alone() -> None:
    approver = DeltaLimitApprover(_delta_space())
    inside = Action(data=np.array([0.1, 1.0]))  # at the box edge: untouched
    assert approver.review(inside, {}) is inside
    out = approver.review(Action(data=np.array([0.2, -0.2])), {})
    assert np.allclose(out.data, [0.1, 0.0])


def test_joint_vel_is_rate_limited_like_displacement() -> None:
    space = Box(
        shape=(1,),
        low=np.array([-2.0]),
        high=np.array([2.0]),
        semantics=ActionSemantics("joint_vel"),
    )
    approver = DeltaLimitApprover(space, max_delta=1.0)
    out = approver.review(Action(data=np.array([1.5])), {})
    assert np.allclose(out.data, [1.0])


# --- shared contracts ---------------------------------------------------------


def test_nan_raises_safety_abort_in_both_branches() -> None:
    for space in (_abs_space(), _delta_space()):
        approver = DeltaLimitApprover(space, max_delta=0.1)
        with pytest.raises(SafetyAbort, match="NaN"):
            approver.review(Action(data=np.array([float("nan"), 0.0])), {})


def test_chain_runs_approvers_in_order() -> None:
    space = _delta_space()
    chain = ChainApprover(ClampApprover(space), DeltaLimitApprover(space, max_delta=0.05))
    out = chain.review(Action(data=np.array([5.0, 5.0])), {})
    # Clamp to box ([0.1, 1.0]) first, then delta-limit to 0.05.
    assert np.allclose(out.data, [0.05, 0.05])
    inside = Action(data=np.array([0.01, 0.01]))
    assert chain.review(inside, {}) is inside  # identity survives the chain
