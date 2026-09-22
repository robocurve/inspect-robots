from typing import Any

import numpy as np

from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_jev.scorer import cube_in_bowl


def record(meta: Any) -> TrialRecord:
    obs = Observation()
    step = StepRecord(
        t=0,
        observation=obs,
        action=Action(data=np.zeros(1), meta=meta),
        result=StepResult(observation=obs),
    )
    return TrialRecord(scene_id="s", epoch=0, seed=None, steps=[step])


def world(
    cube: tuple[float, float, float],
    bowl: tuple[float, float, float],
    *,
    seen_ago: int = 0,
    held: bool = False,
) -> dict[str, Any]:
    return {
        "jev": {
            "held": held,
            "world": {
                "cube": {
                    "left": list(cube),
                    "right": [cube[0], cube[1] + 0.5, cube[2]],
                    "seen_ago": seen_ago,
                },
                "bowl": {
                    "left": list(bowl),
                    "right": [bowl[0], bowl[1] + 0.5, bowl[2]],
                    "seen_ago": 0,
                },
            },
        }
    }


def test_success() -> None:
    score = cube_in_bowl()(record(world((0.35, -0.09, 0.03), (0.35, -0.10, 0.02))), None)
    assert score.value is True and score.metadata["arm"] == "left"
    assert score.explanation == "horizontal 1.0 cm, +1.0 cm above bowl tag"


def test_horizontal_miss_and_too_high() -> None:
    assert (
        cube_in_bowl()(record(world((0.40, -0.10, 0.03), (0.35, -0.10, 0.02))), None).value is False
    )
    assert (
        cube_in_bowl()(record(world((0.35, -0.10, 0.08), (0.35, -0.10, 0.02))), None).value is False
    )


def test_not_reseen_or_still_held_fails() -> None:
    s = cube_in_bowl()(record(world((0.35, -0.10, 0.03), (0.35, -0.10, 0.02), seen_ago=5)), None)
    assert s.value is False and "not observed after release" in str(s.explanation)
    s = cube_in_bowl()(record(world((0.35, -0.10, 0.03), (0.35, -0.10, 0.02), held=True)), None)
    assert s.value is False and "still held" in str(s.explanation)


def test_missing_data() -> None:
    scorer = cube_in_bowl()
    assert (
        scorer(TrialRecord(scene_id="s", epoch=0, seed=None), None).explanation
        == "no jev world state recorded"
    )
    assert scorer(record({}), None).explanation == "no jev world state recorded"
    s = scorer(record({"jev": {"world": {"cube": {"left": [0, 0, 0]}}}}), None)
    assert s.value is False and "never observed" in str(s.explanation)


def test_right_frame_fallback_and_no_shared_frame() -> None:
    meta = {
        "jev": {
            "held": False,
            "world": {
                "cube": {"right": [0.35, 0.4, 0.03], "seen_ago": 0},
                "bowl": {"right": [0.35, 0.4, 0.02]},
            },
        }
    }
    s = cube_in_bowl()(record(meta), None)
    assert s.value is True and s.metadata["arm"] == "right"
    meta = {
        "jev": {
            "held": False,
            "world": {"cube": {"left": [0, 0, 0], "seen_ago": 0}, "bowl": {"right": [0, 0, 0]}},
        }
    }
    assert "share no arm frame" in str(cube_in_bowl()(record(meta), None).explanation)


def test_custom_names_and_name() -> None:
    scorer = cube_in_bowl(cube="red cube", bowl="dish", radius_m=0.1)
    assert scorer.name == "jev_cube_in_bowl"
    meta = {
        "jev": {
            "held": False,
            "world": {
                "red cube": {"left": [0.0, 0.05, 0.0], "seen_ago": 0},
                "dish": {"left": [0, 0, 0]},
            },
        }
    }
    assert scorer(record(meta), None).value is True


def test_pose_taken_from_gripper_is_not_credited() -> None:
    meta = world((0.35, -0.10, 0.03), (0.35, -0.10, 0.02))
    meta["jev"]["world"]["cube"]["from_gripper"] = True
    s = cube_in_bowl()(record(meta), None)
    assert s.value is False and "still held" in str(s.explanation)
