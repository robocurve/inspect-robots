import pytest

from inspect_robots_jev.menu import HOLD, MenuKind, Move, build_menu, parse_option


def test_xyz_menu_shape_and_hints() -> None:
    menu = build_menu("left", "xyz", "red cube", [0.5, 2, 5])
    assert len(menu) == 19 and HOLD in menu
    assert menu["left_x_plus_2cm"] == (
        "move the left gripper FORWARD by 2 cm; "
        "use when the red cube is 2 to 5 cm away in that direction"
    )
    assert menu["left_z_minus_0.5cm"].startswith(
        "move the left gripper DOWN by 0.5 cm; use when the red cube is less than 2 cm away"
    )
    assert menu["left_y_minus_5cm"].endswith(
        "use only when the red cube is more than 5 cm away in that direction"
    )
    assert menu[HOLD].startswith("do not move this turn")


def test_single_step_hint() -> None:
    assert build_menu("left", "xyz", "c", [2])["left_x_plus_2cm"].endswith("not yet reached")


def test_z_menu() -> None:
    menu = build_menu("right", "z", "bowl", [0.5, 2])
    assert set(menu) == {
        "right_z_plus_0.5cm",
        "right_z_minus_0.5cm",
        "right_z_plus_2cm",
        "right_z_minus_2cm",
        HOLD,
    }


def test_gripper_menus() -> None:
    assert set(build_menu("left", "grip_close", "red cube", [1])) == {"left_close", HOLD}
    assert set(build_menu("left", "grip_open", "bowl", [1])) == {"left_open", HOLD}
    assert set(build_menu("left", "done", "bowl", [1])) == {HOLD}


@pytest.mark.parametrize(
    ("option", "move"),
    [
        ("left_x_plus_2cm", Move("left", "x", 0.02, None)),
        ("right_z_minus_0.5cm", Move("right", "z", -0.005, None)),
        ("left_close", Move("left", None, 0.0, 0.0)),
        ("right_open", Move("right", None, 0.0, 1.0)),
    ],
)
def test_parse_option(option: str, move: Move) -> None:
    parsed = parse_option(option)
    assert (parsed.arm, parsed.axis, parsed.gripper) == (move.arm, move.axis, move.gripper)
    assert parsed.delta_m == pytest.approx(move.delta_m)
    assert not parsed.is_hold


def test_every_menu_option_parses() -> None:
    kinds: tuple[MenuKind, ...] = ("xyz", "z", "grip_close", "grip_open", "done")
    for kind in kinds:
        for option in build_menu("right", kind, "t", [0.5, 2, 5]):
            parse_option(option)


@pytest.mark.parametrize("bad", ["left_sideways_3cm", "left_x_plus_abccm", "up", "left_x_plus_3"])
def test_parse_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="unrecognised"):
        parse_option(bad)


def test_parse_hold() -> None:
    assert parse_option(HOLD).is_hold
