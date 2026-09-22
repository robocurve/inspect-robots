import pytest

from inspect_robots_jev.phase import PhaseMachine, TaskConfig
from inspect_robots_jev.world import GripperView, ObjectView, Vec3, WorldState

CUBE: Vec3 = (0.30, 0.10, 0.013)
BOWL: Vec3 = (0.35, -0.10, 0.02)


def world(
    left: Vec3 = (0.20, 0.0, 0.20),
    opening: float = 1.0,
    cube: Vec3 = CUBE,
    step: int = 0,
    right: Vec3 = (0.2, 0.5, 0.2),
) -> WorldState:
    return WorldState(
        step=step,
        objects={
            "cube": ObjectView(
                "cube", {"left": cube, "right": (cube[0], cube[1] + 0.5, cube[2])}, step
            ),
            "bowl": ObjectView(
                "bowl", {"left": BOWL, "right": (BOWL[0], BOWL[1] + 0.5, BOWL[2])}, step
            ),
        },
        grippers={"left": GripperView(left, opening), "right": GripperView(right, 1.0)},
    )


def test_phase_before_reset_raises() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        _ = PhaseMachine(TaskConfig()).phase


def test_reset_picks_closer_arm_and_hover_goal() -> None:
    m = PhaseMachine(TaskConfig())
    p = m.reset(world())
    assert (p.name, p.arm, p.target, p.menu_kind) == ("approach", "left", "cube", "xyz")
    assert p.goal == pytest.approx((0.30, 0.10, 0.063))
    assert p.positional


def test_reset_picks_arm_by_base_distance_not_gripper() -> None:
    # The right gripper is parked on top of the cube, but the cube is far from
    # the right arm's base (y = 0.60 in its frame) and near the left base.
    m = PhaseMachine(TaskConfig())
    p = m.reset(world(left=(0.0, -0.4, 0.4), right=(0.30, 0.60, 0.013)))
    assert p.arm == "left"


def test_reset_can_pick_right_arm() -> None:
    m = PhaseMachine(TaskConfig())
    w = WorldState(
        step=0,
        objects={
            "cube": ObjectView(
                "cube", {"left": (0.30, 0.60, 0.013), "right": (0.30, 0.10, 0.013)}, 0
            ),
            "bowl": ObjectView("bowl", {"left": BOWL, "right": BOWL}, 0),
        },
        grippers={
            "left": GripperView((0.2, 0.0, 0.2), 1.0),
            "right": GripperView((0.2, 0.0, 0.2), 1.0),
        },
    )
    p = m.reset(w)
    assert p.arm == "right"
    assert p.goal == pytest.approx((0.30, 0.10, 0.063))


def test_goals_are_clamped_into_bounds_so_floor_does_not_deadlock() -> None:
    bounds = ((0.15, -0.25, 0.03), (0.48, 0.25, 0.40))
    m = PhaseMachine(TaskConfig(), bounds=bounds)
    m.reset(world())
    m.advance(world(left=(0.30, 0.10, 0.063)))
    p = m.phase
    assert p.name == "descend" and p.goal == pytest.approx((0.30, 0.10, 0.03))
    # gripper stops at the floor; the transition must still fire
    assert m.advance(world(left=(0.30, 0.10, 0.03))).name == "grasp"


def test_full_happy_path() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    assert m.advance(world(left=(0.30, 0.10, 0.063))).name == "descend"
    p = m.advance(world(left=(0.30, 0.10, 0.013)))
    assert (p.name, p.menu_kind, p.goal) == ("grasp", "grip_close", None)
    assert not p.positional
    p = m.advance(world(left=(0.30, 0.10, 0.013), opening=0.3))
    assert (p.name, p.menu_kind) == ("lift", "z")
    assert p.goal == pytest.approx((0.30, 0.10, 0.093))
    p = m.advance(world(left=(0.30, 0.10, 0.093), opening=0.3, cube=(0.30, 0.10, 0.093)))
    assert (p.name, p.target, p.menu_kind) == ("carry", "bowl", "xyz")
    assert p.goal == pytest.approx((0.35, -0.10, 0.0827))
    p = m.advance(world(left=(0.35, -0.10, 0.093), opening=0.3, cube=(0.35, -0.10, 0.093)))
    assert p.name == "lower" and p.goal == pytest.approx((0.35, -0.10, -0.0073))
    p = m.advance(world(left=(0.35, -0.10, -0.0073), opening=0.3, cube=(0.35, -0.10, -0.0073)))
    assert p.name == "release" and p.menu_kind == "grip_open"
    p = m.advance(world(left=(0.35, -0.10, -0.0073), opening=0.9))
    assert p.name == "retreat" and p.goal == pytest.approx((0.35, -0.10, 0.10))
    p = m.advance(world(left=(0.35, -0.10, 0.10), opening=0.9))
    assert (p.name, p.menu_kind, p.target) == ("verify", "z", "cube")
    assert p.goal == pytest.approx((0.35, -0.10, 0.15))
    # the cube is seen again this step (not inferred from the gripper) -> done
    p = m.advance(world(left=(0.35, -0.10, 0.15), opening=0.9, cube=(0.35, -0.10, 0.013), step=1))
    assert (p.name, p.menu_kind, p.goal) == ("done", "done", None)
    assert m.phase.name == "done"
    assert m.advance(world()).name == "done"


def test_grasp_on_air_reopens_then_returns_to_approach() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    m.advance(world(left=(0.30, 0.10, 0.063)))
    m.advance(world(left=(0.30, 0.10, 0.013)))
    assert m.phase.name == "grasp"
    p = m.advance(world(left=(0.30, 0.10, 0.013), opening=0.01))
    assert (p.name, p.menu_kind, p.target, p.goal) == ("reopen", "grip_open", "cube", None)
    # still closed: stay in reopen, never bounce back to grasp
    assert m.advance(world(left=(0.30, 0.10, 0.013), opening=0.01)).name == "reopen"
    assert m.advance(world(left=(0.30, 0.10, 0.013), opening=0.9)).name == "approach"


def test_grasp_waits_while_still_open() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    m.advance(world(left=(0.30, 0.10, 0.063)))
    m.advance(world(left=(0.30, 0.10, 0.013)))
    assert m.advance(world(left=(0.30, 0.10, 0.013), opening=0.9)).name == "grasp"


def test_no_transition_when_far() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    assert m.advance(world(left=(0.10, 0.10, 0.30))).name == "approach"
    assert m.advance(world(left=(0.30, 0.10, 0.30))).name == "approach"


def test_lift_goal_without_grasp_point_falls_back_to_cube() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    m._name = "lift"
    p = m.advance(world(left=(0.30, 0.10, 0.013), opening=0.3))
    assert p.goal == pytest.approx((0.30, 0.10, 0.093))


def test_verify_waits_for_a_real_sighting_then_gives_up() -> None:
    m = PhaseMachine(TaskConfig(verify_decisions=2))
    m.reset(world())
    m._name = "verify"
    # cube view still pinned to the gripper: not a sighting
    pinned = WorldState(
        step=5,
        objects={
            "cube": ObjectView("cube", {"left": CUBE, "right": CUBE}, 5, from_gripper=True),
            "bowl": ObjectView("bowl", {"left": BOWL, "right": BOWL}, 5),
        },
        grippers={
            "left": GripperView((0.35, -0.10, 0.15), 1.0),
            "right": GripperView((0.2, 0.5, 0.2), 1.0),
        },
    )
    assert m.advance(pinned).name == "verify"
    assert m.advance(pinned).name == "verify"
    assert m.advance(pinned).name == "done"  # budget spent: end anyway, scorer will fail it


def test_verify_ends_on_stale_but_unpinned_view_only_when_fresh() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    m._name = "verify"
    stale = world(left=(0.35, -0.10, 0.15), step=9, cube=CUBE)
    stale_view = WorldState(
        step=9,
        objects={
            "cube": ObjectView("cube", {"left": CUBE, "right": CUBE}, 4),
            "bowl": stale.objects["bowl"],
        },
        grippers=stale.grippers,
    )
    assert m.advance(stale_view).name == "verify"
    assert m.advance(world(left=(0.35, -0.10, 0.15), step=10)).name == "done"


def test_contact_rule_ends_descend_and_lower_when_height_stops_dropping() -> None:
    cfg = TaskConfig(contact_decisions=3)
    m = PhaseMachine(cfg)
    m.reset(world())
    m._name = "descend"
    # fingers rest on the table at z=0.05, well above the cube-centre goal (0.013)
    for _ in range(3):
        assert m.advance(world(left=(0.30, 0.10, 0.05))).name == "descend"
    assert m.advance(world(left=(0.30, 0.10, 0.05))).name == "grasp"
    m._name = "lower"
    m._grasp_point = (0.30, 0.10, 0.05)
    for z in (0.10, 0.08, 0.06):  # still descending 2 cm per decision: no contact yet
        assert m.advance(world(left=(0.35, -0.10, z), opening=0.3)).name == "lower"
    # the window must be flat for contact_decisions=3 consecutive decisions
    for _ in range(2):
        assert m.advance(world(left=(0.35, -0.10, 0.06), opening=0.3)).name == "lower"
    assert m.advance(world(left=(0.35, -0.10, 0.06), opening=0.3)).name == "release"


def test_contact_rule_needs_real_commanded_descent() -> None:
    # four 0.5 cm DOWN picks that the arm under-travels to 0.3 cm each: span 0.9 cm
    # (<= z_tol) but the commanded descent (1.5 cm) exceeds z_tol -> contact fires;
    # with 0.2 cm picks (0.6 cm commanded over the window) it must not.
    m = PhaseMachine(TaskConfig(contact_decisions=3))
    m.reset(world())
    m._name = "descend"
    for z in (0.100, 0.097, 0.094, 0.091):
        p = m.advance(world(left=(0.30, 0.10, z)), descended_m=0.002)
    assert p.name == "descend"
    m._z_trace = []
    for z in (0.100, 0.097, 0.094):
        assert m.advance(world(left=(0.30, 0.10, z)), descended_m=0.005).name == "descend"
    assert m.advance(world(left=(0.30, 0.10, 0.091)), descended_m=0.005).name == "grasp"


def test_z_trace_resets_outside_descending_phases() -> None:
    m = PhaseMachine(TaskConfig())
    m.reset(world())
    for _ in range(5):
        m.advance(world(left=(0.10, 0.10, 0.30)))
    assert m._z_trace == []


def test_contact_rule_ignores_rising_and_uncommanded_decisions() -> None:
    m = PhaseMachine(TaskConfig(contact_decisions=3))
    m.reset(world())
    m._name = "descend"
    # rising for four decisions is not contact
    for z in (0.10, 0.11, 0.12, 0.13):
        assert m.advance(world(left=(0.30, 0.10, z))).name == "descend"
    m._z_trace = []
    # flat height but no DOWN was commanded (holds): not contact either
    for _ in range(5):
        assert m.advance(world(left=(0.30, 0.10, 0.08)), descended_m=0.0).name == "descend"
    assert m._z_trace == []
