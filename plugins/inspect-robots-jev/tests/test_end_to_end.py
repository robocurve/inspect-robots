"""Full ``eval()`` on the fake rig with a greedy scripted Jev."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots_jev._decisions import ChoiceAnswer
from inspect_robots_jev.calibration import Calibration, pose_to_matrix
from inspect_robots_jev.perceiver import Perceiver, TagLayout
from inspect_robots_jev.policy import JevPolicy, JevPolicyConfig
from inspect_robots_jev.scorer import cube_in_bowl
from tests._fake_rig import BOX, RIGHT_OFFSET, FakeRig

FIXTURE = Path(__file__).parent / "fixtures" / "tags_cube_bowl.json"
_PHRASE = re.compile(r"([\d.]+) cm (FORWARD|BACK|LEFT|RIGHT|ABOVE|BELOW)")
_AXIS = {
    "FORWARD": ("x", "plus"),
    "BACK": ("x", "minus"),
    "LEFT": ("y", "plus"),
    "RIGHT": ("y", "minus"),
    "ABOVE": ("z", "plus"),
    "BELOW": ("z", "minus"),
}


class GreedyJev:
    """Read the directional words in the state; take the largest legal step on the largest gap."""

    def __init__(self) -> None:
        self.calls = 0

    def choose(
        self,
        *,
        state: dict[str, Any],
        instructions: str,
        criteria: dict[str, str],
        question_id: str = "move",
    ) -> ChoiceAnswer:
        self.calls += 1
        choice = self._pick(state, criteria)
        return ChoiceAnswer(
            choice=choice,
            probabilities={choice: 1.0},
            confidence=1.0,
            model="greedy",
            usage={},
            raw={},
        )

    @staticmethod
    def _pick(state: dict[str, Any], criteria: dict[str, str]) -> str:
        grip_options = [k for k in criteria if k.endswith(("_close", "_open"))]
        if grip_options:
            return grip_options[0]
        key = next((k for k in state if k.startswith("target_point_relative_to_")), None)
        if key is None:
            return "hold"
        arm = key[len("target_point_relative_to_") : -len("_gripper")]
        gaps = []
        for phrase in state[key]:
            m = _PHRASE.match(phrase)
            if m:
                gaps.append((float(m.group(1)), *_AXIS[m.group(2)]))
        if not gaps:
            return "hold"
        gap, axis, word = max(gaps)
        steps = sorted(
            {
                float(k.split("_")[3][:-2])
                for k in criteria
                if k.count("_") == 3 and k.split("_")[1] == axis
            }
        )
        fitting = [s for s in steps if s <= gap + 1e-9] or steps[:1]
        return f"{arm}_{axis}_{word}_{fitting[-1]:g}cm"


@pytest.mark.parametrize(
    ("occlude", "floor_z"),
    [(False, None), (True, None), (True, 0.02), (False, 0.03)],
    ids=["clear", "occluding-camera", "occluding+shallow-bowl", "shallow-bowl"],
)
def test_cube_into_bowl_end_to_end(tmp_path: Path, occlude: bool, floor_z: float | None) -> None:
    rig = FakeRig(occlude=occlude, floor_z=floor_z)
    cal = Calibration(
        camera_to_arm={"left": np.eye(4), "right": pose_to_matrix(np.eye(3), RIGHT_OFFSET)}
    )
    perceiver = Perceiver(
        layout=TagLayout.load(FIXTURE), camera="top_cam", calibration=cal, detector=rig.detector
    )
    jev = GreedyJev()
    policy = JevPolicy(JevPolicyConfig(steps_cm=(0.5, 2.0, 5.0)), client=jev, perceiver=perceiver)  # type: ignore[arg-type]
    task = Task(
        name="jev-cube-in-bowl",
        scenes=[Scene(id="s0", instruction="put the cube in the bowl")],
        scorer=[cube_in_bowl()],
        max_steps=600,
    )
    logs = ir_eval(task, policy, rig, log_dir=str(tmp_path))
    log = logs[0]
    assert log.status == "success", log.error
    sample = log.samples[0]
    assert sample.status == "success"
    assert sample.termination_reasons == ("task_done",)
    assert sample.epochs[0]["jev_cube_in_bowl"] == 1.0
    assert sample.trial_metadata[0]["jev"]["final_phase"] == "done"
    assert sample.trial_metadata[0]["jev"]["decisions"] > 10
    transcript = sample.policy_transcripts[0]
    assert isinstance(transcript, list) and transcript[-1]["phase"] == "done"
    assert rig.holding is False
    assert np.hypot(*(rig.cube[:2] - rig.bowl[:2])) <= 0.04
    # the trial must not have ended while the cube was still hidden under the gripper
    final_cube = sample.trial_metadata[0]["jev"]["final_world"]["cube"]
    assert final_cube["from_gripper"] is False and final_cube["seen_ago"] == 0
    low, high = np.asarray(BOX.low), np.asarray(BOX.high)
    for cmd in rig.commanded:
        assert np.all(cmd >= low - 1e-12) and np.all(cmd <= high + 1e-12)
    assert jev.calls > 10
    assert log.results.total_trials == 1
