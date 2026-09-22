import pytest

from inspect_robots_jev.world import (
    DIRECTION,
    GripperView,
    ObjectView,
    WorldState,
    describe_offset,
    distance,
    round_half_cm,
)


@pytest.mark.parametrize(
    ("m", "expected"),
    [(0.0349, 0.035), (0.0374, 0.035), (0.0376, 0.04), (-0.0126, -0.015), (0.002, 0.0)],
)
def test_round_half_cm(m: float, expected: float) -> None:
    assert round_half_cm(m) == pytest.approx(expected)


def test_describe_offset_words() -> None:
    words = describe_offset(target=(0.135, 0.20, 0.06), origin=(0.10, 0.20, 0.10))
    assert words == ["3.5 cm FORWARD", "aligned on y", "4 cm BELOW"]


def test_describe_offset_other_signs() -> None:
    words = describe_offset(target=(0.08, 0.15, 0.20), origin=(0.10, 0.20, 0.10))
    assert words == ["2 cm BACK", "5 cm RIGHT", "10 cm ABOVE"]


def test_direction_table_is_complete() -> None:
    assert set(DIRECTION.values()) == {"FORWARD", "BACK", "LEFT", "RIGHT", "UP", "DOWN"}


def test_world_offset_and_seen() -> None:
    world = WorldState(
        step=7,
        objects={
            "cube": ObjectView(
                "cube", {"left": (0.3, 0.0, 0.02), "right": (0.3, 0.4, 0.02)}, last_seen_step=4
            )
        },
        grippers={
            "left": GripperView((0.2, 0.0, 0.1), 1.0),
            "right": GripperView((0.2, 0.0, 0.1), 1.0),
        },
    )
    assert world.offset("left", "cube") == pytest.approx((0.1, 0.0, -0.08))
    assert world.steps_since_seen("cube") == 3
    assert distance((0.0, 0.0, 0.0), (0.0, 0.03, 0.04)) == pytest.approx(0.05)
