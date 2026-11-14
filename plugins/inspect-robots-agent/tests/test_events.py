from __future__ import annotations

import numpy as np
import pytest

from inspect_robots.types import Observation
from inspect_robots_agent._events import ObservationEvents, routine_camera_observation


def _detector() -> ObservationEvents:
    return ObservationEvents(("eef_state", ("left_z", "left_gripper", "right_gripper")))


def _observation(left: float = 1.0, right: float = 1.0, z: float = 0.10) -> Observation:
    return Observation(state={"eef_state": np.array([z, left, right])})


def test_watchdog_counts_observations_and_resets_with_a_new_detector() -> None:
    events = _detector()
    assert [events.observe(_observation(z=i / 20)) for i in range(9)] == [
        "initial",
        None,
        None,
        None,
        "watchdog",
        None,
        None,
        None,
        "watchdog",
    ]
    assert _detector().observe(_observation()) == "initial"


def test_detect_command_then_readback_without_retriggering_on_held_target() -> None:
    events = _detector()
    assert events.observe(_observation()) == "initial"
    # A command is actionable even if the gripper has not moved yet.
    assert (
        events.observe(_observation(), previous_target=np.array([0.10, 0.0, 1.0]))
        == "gripper_command"
    )
    assert (
        events.observe(_observation(left=0.4), previous_target=np.array([0.08, 0.0, 1.0]))
        == "gripper_motion"
    )
    # The object can keep the readback far from its held command on later turns.
    assert (
        events.observe(_observation(left=0.41), previous_target=np.array([0.07, 0.0, 1.0])) is None
    )


@pytest.mark.parametrize(
    "unknown",
    [Observation(), _observation(left=float("nan")), _observation(left=45.0)],
)
def test_missing_state_preserves_fidelity_until_readback_is_available_again(
    unknown: Observation,
) -> None:
    events = _detector()
    assert events.observe(_observation()) == "initial"
    assert events.observe(unknown) == "missing_gripper_state"
    assert events.observe(_observation()) == "missing_gripper_state"
    assert events.observe(_observation()) is None


def test_recovery_overrides_routine_even_without_gripper_movement() -> None:
    events = _detector()
    assert events.observe(_observation()) == "initial"
    assert events.observe(_observation(), recovery=True) == "recovery"
    assert (
        events.observe(
            Observation(state=_observation().state, extra={"replan": {"reason": "gripper slip"}})
        )
        == "recovery"
    )


def test_thumbnail_averages_pixels_and_keeps_original_frames() -> None:
    stripes = np.tile(np.array([0, 200], dtype=np.uint8), 112)
    wrist = np.broadcast_to(stripes[np.newaxis, :, np.newaxis], (224, 224, 3)).copy()
    wide = np.full((160, 320, 3), 50, dtype=np.uint8)
    overhead = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = Observation(images={"left_cam": wrist, "right_cam": wide, "top_cam": overhead})
    routine = routine_camera_observation(observation)
    assert routine.images["left_cam"].shape == (112, 112, 3)
    assert routine.images["right_cam"].shape == (56, 112, 3)
    assert np.all(routine.images["left_cam"] == 100)
    assert np.all(routine.images["right_cam"] == 50)
    assert routine.images["top_cam"] is overhead
    assert observation.images["left_cam"] is wrist
    assert observation.images["right_cam"] is wide
    assert wrist[0, :2, 0].tolist() == [0, 200]


def test_thumbnail_keeps_small_wrist_and_unrecognized_view() -> None:
    thumbnail = np.zeros((40, 50, 3), dtype=np.uint8)
    unknown = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = Observation(images={"left_cam": thumbnail, "wrist": unknown})
    routine = routine_camera_observation(observation)
    assert routine is observation
    assert routine.images["left_cam"] is thumbnail
    assert routine.images["wrist"] is unknown
