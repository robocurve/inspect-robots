import json

from inspect_robots_jev.phase import Phase
from inspect_robots_jev.serializer import build_state, gripper_word, instructions_for
from inspect_robots_jev.world import GripperView, ObjectView, WorldState


def world(step: int = 3, cube_seen: int = 3, opening: float = 1.0) -> WorldState:
    return WorldState(
        step=step,
        objects={
            "cube": ObjectView(
                "cube", {"left": (0.30, 0.10, 0.013), "right": (0.3, 0.6, 0.013)}, cube_seen
            ),
            "bowl": ObjectView(
                "bowl", {"left": (0.35, -0.10, 0.02), "right": (0.35, 0.4, 0.02)}, step
            ),
        },
        grippers={
            "left": GripperView((0.265, 0.10, 0.103), opening),
            "right": GripperView((0.2, 0.5, 0.2), 1.0),
        },
    )


APPROACH = Phase("approach", "left", "cube", "xyz", (0.30, 0.10, 0.063))


def test_approach_state_exact() -> None:
    state = build_state(world(), APPROACH, ["move the left gripper FORWARD by 5 cm"])
    assert state == {
        "task": (
            "Move the left gripper to the target point 5 cm above the cube. "
            "Move toward that point, never away."
        ),
        "target_point_relative_to_left_gripper": ["3.5 cm FORWARD", "aligned on y", "4 cm BELOW"],
        "straight_line_distance_cm": 5.3,
        "left_gripper": "open",
        "other_objects_on_table_ignore_them": ["bowl"],
        "recent_moves_oldest_first": ["move the left gripper FORWARD by 5 cm"],
    }
    json.dumps(state)


def test_note_when_target_unseen() -> None:
    state = build_state(world(step=5, cube_seen=2), APPROACH, [])
    assert state["note"] == "the cube was last seen 3 decisions ago; assume it has not moved"


def test_gripper_phase_has_no_offset() -> None:
    grasp = Phase("grasp", "left", "cube", "grip_close", None)
    state = build_state(world(opening=0.5), grasp, [])
    assert "target_point_relative_to_left_gripper" not in state
    assert "straight_line_distance_cm" not in state
    assert state["task"] == "Close the left gripper on the cube."
    assert state["left_gripper"] == "partly closed"


def test_history_truncates_oldest_first() -> None:
    state = build_state(world(), APPROACH, [f"m{i}" for i in range(8)], history_len=5)
    assert state["recent_moves_oldest_first"] == ["m3", "m4", "m5", "m6", "m7"]
    assert build_state(world(), APPROACH, ["a"], history_len=0)["recent_moves_oldest_first"] == []


def test_every_phase_has_task_text() -> None:
    for name, kind in (
        ("descend", "z"),
        ("lift", "z"),
        ("carry", "xyz"),
        ("lower", "z"),
        ("retreat", "z"),
    ):
        phase = Phase(name, "right", "bowl", kind, (0.3, 0.4, 0.1))  # type: ignore[arg-type]
        assert "right gripper" in build_state(world(), phase, [])["task"]
    for name, kind in (("release", "grip_open"), ("reopen", "grip_open"), ("done", "done")):
        phase = Phase(name, "right", "bowl", kind, None)  # type: ignore[arg-type]
        assert build_state(world(), phase, [])["task"]


def test_missing_target_object_is_tolerated() -> None:
    phase = Phase("approach", "left", "ghost", "xyz", (0.3, 0.1, 0.06))
    state = build_state(world(), phase, [])
    assert "note" not in state
    assert state["other_objects_on_table_ignore_them"] == ["bowl", "cube"]


def test_instructions() -> None:
    assert instructions_for(APPROACH).startswith("Which move takes the left gripper toward")
    assert (
        instructions_for(Phase("grasp", "right", "cube", "grip_close", None))
        == "Should the right gripper act now?"
    )
    assert instructions_for(Phase("done", "left", "bowl", "done", None)) == "Pick hold."


def test_gripper_word() -> None:
    words = [gripper_word(o) for o in (1.0, 0.8, 0.5, 0.35, 0.0)]
    assert words == ["open", "open", "partly closed", "closed", "closed"]
    assert gripper_word(0.5, open_threshold=0.4) == "open"
    assert gripper_word(0.5, closed_threshold=0.6) == "closed"


def test_build_state_threads_thresholds() -> None:
    state = build_state(world(opening=0.5), APPROACH, [], open_threshold=0.4)
    assert state["left_gripper"] == "open"
