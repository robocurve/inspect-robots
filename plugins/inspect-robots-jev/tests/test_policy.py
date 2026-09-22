import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Observation, StepResult
from inspect_robots_jev._decisions import ChoiceAnswer
from inspect_robots_jev.calibration import Calibration
from inspect_robots_jev.perceiver import TagLayout
from inspect_robots_jev.phase import TaskConfig
from inspect_robots_jev.policy import JevPolicy, JevPolicyConfig, jev_policy
from inspect_robots_jev.scorer import cube_in_bowl
from inspect_robots_jev.world import Arm, GripperView, ObjectView, Vec3, WorldState

LABELS = tuple(
    f"{arm}_{p}"
    for arm in ("left", "right")
    for p in ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
)
ARM_LOW = (0.15, -0.25, 0.03, -math.pi, 0.0, 0.0, 0.0)
ARM_HIGH = (0.48, 0.25, 0.40, math.pi, 0.0, 0.0, 1.0)
BOX = Box(
    low=np.array(ARM_LOW * 2),
    high=np.array(ARM_HIGH * 2),
    shape=(14,),
    semantics=ActionSemantics(
        control_mode="eef_abs_pose", gripper="continuous", frame="base", dim_labels=LABELS
    ),
)
INFO = EmbodimentInfo(
    name="fake", action_space=BOX, observation_space=ObservationSpace(), control_hz=10.0
)
CUBE: Vec3 = (0.30, 0.10, 0.04)
BOWL: Vec3 = (0.35, -0.10, 0.05)


def eef(left: Vec3 = (0.20, 0.0, 0.20), opening: float = 1.0) -> np.ndarray:
    s = np.zeros(14)
    s[0:3] = left
    s[6] = opening
    s[7:10] = (0.20, 0.0, 0.20)
    s[13] = 1.0
    return s


def obs(left: Vec3 = (0.20, 0.0, 0.20), opening: float = 1.0, step: int = 0) -> Observation:
    return Observation(images={}, state={"eef_state": eef(left, opening)}, extra={"env_step": step})


class FakePerceiver:
    """Scripted world: objects fixed; optionally drop the cube from view."""

    def __init__(self, *, objects: dict[str, Vec3] | None = None) -> None:
        self.objects = objects if objects is not None else {"cube": CUBE, "bowl": BOWL}
        self.seen_step: dict[str, int] = {}
        self.visible = True
        self.resets = 0
        self.held_seen: list[tuple[Arm, str] | None] = []

    def reset(self) -> None:
        self.resets += 1
        self.seen_step = {}

    def update(
        self,
        observation: Observation,
        step: int,
        *,
        grippers: dict[Arm, GripperView],
        held: Any = None,
    ) -> WorldState:
        self.held_seen.append(held)
        views: dict[str, ObjectView] = {}
        for name, pos in self.objects.items():
            if self.visible or name != "cube":
                self.seen_step[name] = step
            if name in self.seen_step:
                pinned = held is not None and held[1] == name
                views[name] = ObjectView(
                    name,
                    {"left": pos, "right": (pos[0], pos[1] + 0.5, pos[2])},
                    self.seen_step[name],
                    from_gripper=pinned,
                )
        return WorldState(step=step, objects=views, grippers=grippers)


class FakeClient:
    def __init__(self, choices: list[str]) -> None:
        self.choices = choices
        self.calls: list[dict[str, Any]] = []

    def choose(
        self, *, state: Any, instructions: str, criteria: dict[str, str], question_id: str = "move"
    ) -> ChoiceAnswer:
        self.calls.append({"state": state, "instructions": instructions, "criteria": criteria})
        choice = self.choices.pop(0)
        return ChoiceAnswer(
            choice=choice,
            probabilities={choice: 1.0},
            confidence=0.9,
            model="m",
            usage={"input_tokens": 1},
            raw={},
        )


def make(
    choices: list[str], perceiver: FakePerceiver | None = None
) -> tuple[JevPolicy, FakeClient, FakePerceiver]:
    client = FakeClient(choices)
    perc = perceiver or FakePerceiver()
    policy = JevPolicy(JevPolicyConfig(), client=client, perceiver=perc)  # type: ignore[arg-type]
    policy.bind(INFO)
    policy.reset(Scene(id="s", instruction="put the cube in the bowl"))
    return policy, client, perc


def test_act_before_bind_raises() -> None:
    with pytest.raises(RuntimeError, match="bind"):
        JevPolicy(JevPolicyConfig(), client=FakeClient([]), perceiver=FakePerceiver()).act(obs())  # type: ignore[arg-type]


def test_bind_adopts_action_space_and_intersects_bounds() -> None:
    policy, _, _ = make([])
    assert policy.info.action_space is BOX and policy.info.control_hz == 10.0
    assert policy._bounds == ((0.15, -0.25, 0.03), (0.48, 0.25, 0.40))


def test_first_act_resets_machine_and_asks_with_xyz_menu() -> None:
    policy, client, perc = make(["left_x_plus_5cm"])
    chunk = policy.act(obs())
    call = client.calls[0]
    assert set(call["criteria"]) >= {"left_x_plus_5cm", "hold"} and len(call["criteria"]) == 19
    assert call["state"]["task"].startswith(
        "Move the left gripper to the target point 5 cm above the cube"
    )
    assert call["state"]["target_point_relative_to_left_gripper"] == [
        "10 cm FORWARD",
        "10 cm LEFT",
        "11 cm BELOW",
    ]
    assert chunk.actions[-1].data[0] == pytest.approx(0.25)
    meta = chunk.actions[-1].meta["jev"]
    assert (
        meta["phase"] == "approach"
        and meta["choice"] == "left_x_plus_5cm"
        and meta["confidence"] == 0.9
    )
    assert meta["world"]["cube"]["left"] == list(CUBE) and meta["world"]["cube"]["seen_ago"] == 0
    assert meta["held"] is False
    assert "request_stop" not in chunk.actions[-1].meta
    assert perc.resets == 1 and perc.held_seen == [None]
    assert policy.transcript()[0]["answer"]["choice"] == "left_x_plus_5cm"
    json.dumps(policy.transcript())


def test_history_accumulates_descriptions() -> None:
    policy, client, _ = make(["left_x_plus_5cm", "left_y_plus_2cm"])
    policy.act(obs())
    policy.act(obs(step=1))
    assert client.calls[1]["state"]["recent_moves_oldest_first"] == [
        "move the left gripper FORWARD by 5 cm"
    ]


def test_hold_pick_yields_current_pose_and_counts_decision() -> None:
    policy, _, _ = make(["hold"])
    chunk = policy.act(obs())
    assert len(chunk.actions) == 1
    np.testing.assert_allclose(chunk.actions[0].data, eef())
    assert policy.transcript()[0]["answer"]["choice"] == "hold"


def test_chunks_start_from_last_commanded_target_not_measured_pose() -> None:
    policy, _, _ = make(["left_x_plus_5cm", "left_y_plus_2cm", "hold"])
    first = policy.act(obs())
    assert first.actions[-1].data[0] == pytest.approx(0.25)
    # the arm lags a little (1 cm): the next chunk continues from the commanded 0.25
    second = policy.act(obs(left=(0.24, 0.0, 0.20), step=1))
    assert second.actions[0].data[0] == pytest.approx(0.25)
    assert second.actions[-1].data[1] == pytest.approx(0.02)
    third = policy.act(obs(left=(0.245, 0.01, 0.20), step=2))
    np.testing.assert_allclose(third.actions[0].data, second.actions[-1].data)


def test_large_divergence_resyncs_to_the_measured_pose() -> None:
    policy, _, _ = make(["left_x_plus_5cm", "left_y_plus_2cm"])
    policy.act(obs())  # commands x 0.20 -> 0.25
    # the arm is blocked at 0.21: 4 cm behind the command, beyond the 2 cm resync band
    second = policy.act(obs(left=(0.21, 0.0, 0.20), step=1))
    assert second.actions[0].data[0] == pytest.approx(0.21)
    assert second.actions[-1].data[1] == pytest.approx(0.02)


def test_stall_when_cube_never_seen() -> None:
    perc = FakePerceiver(objects={"bowl": BOWL})
    policy, client, _ = make([], perc)
    chunk = policy.act(obs())
    assert client.calls == [] and len(chunk.actions) == 1
    assert chunk.actions[0].meta["jev"]["stall"] == "cube or bowl not yet seen"
    assert policy.transcript()[0]["stall"] is True


def test_stall_when_target_stale() -> None:
    perc = FakePerceiver()
    policy, client, _ = make(["left_x_plus_0.5cm"] * 11, perc)
    policy.act(obs())
    perc.visible = False
    for step in range(1, 12):
        # env_step counts control ticks and jumps by many per decision; the
        # stale budget must count decisions, so these jumps must not matter
        chunk = policy.act(obs(step=step * 7))
    # decisions 1..10 are within stale_after=10 and still ask Jev; decision 11 stalls
    assert len(client.calls) == 11
    assert policy.transcript()[1]["tick"] == 7
    assert chunk.actions[0].meta["jev"]["stall"] == "cube tag not seen"
    assert policy.transcript()[-1]["reason"] == "cube tag not seen"


def test_done_requests_stop_and_gripper_phase_menu() -> None:
    perc = FakePerceiver()
    policy, client, _ = make(
        ["left_z_minus_2cm", "left_close", "left_open", "left_z_plus_5cm"], perc
    )
    policy._machine = None
    # drive to grasp: gripper already at the cube centre after descend transitions
    policy.act(obs(left=(0.30, 0.10, 0.09)))  # reset -> approach; near hover -> descend on next
    assert policy._machine is not None
    policy._machine._name = "grasp"  # jump ahead: grasp phase, menu is close/hold
    policy.act(obs(left=(0.30, 0.10, 0.04), step=1))
    assert set(client.calls[-1]["criteria"]) == {"left_close", "hold"}
    assert client.calls[-1]["instructions"] == "Should the left gripper act now?"
    policy._machine._name = "release"
    policy.act(obs(left=(0.35, -0.10, 0.05), opening=0.3, step=2))
    policy._machine._name = "verify"
    # the cube is visible again (FakePerceiver marks it seen every step) -> done
    chunk = policy.act(obs(left=(0.35, -0.10, 0.20), opening=1.0, step=3))
    assert chunk.actions[0].meta["request_stop"] is True
    assert chunk.actions[0].meta["stop_reason"] == "task_done"
    assert chunk.actions[0].meta["jev"]["phase"] == "done"
    assert policy.transcript()[-1]["request_stop"] is True


def test_held_flag_follows_phases() -> None:
    perc = FakePerceiver()
    policy, _client, _ = make(["left_z_plus_5cm", "left_z_plus_2cm", "left_z_plus_2cm"], perc)
    policy.act(obs())
    assert policy._machine is not None
    policy._machine._name = "lift"
    policy.act(obs(left=(0.30, 0.10, 0.04), opening=0.3, step=1))
    assert policy._held == ("left", "cube")
    policy.act(obs(left=(0.30, 0.10, 0.20), opening=0.3, step=2))
    assert perc.held_seen[-1] == ("left", "cube")


def test_on_trial_end_writes_metadata_and_reset_clears() -> None:
    policy, _, _ = make(["left_x_plus_5cm"])
    policy.act(obs())
    record = TrialRecord(scene_id="s", epoch=0, seed=None)
    policy.on_trial_end(record, "logs", "run")
    assert (
        record.metadata["jev"]["decisions"] == 1
        and record.metadata["jev"]["final_phase"] == "approach"
    )
    assert record.metadata["jev"]["final_world"]["bowl"]["left"] == list(BOWL)
    policy.reset(Scene(id="s2", instruction="again"))
    assert policy.transcript() == [] and policy._machine is None
    empty = TrialRecord(scene_id="s2", epoch=0, seed=None)
    policy.on_trial_end(empty, "logs", "run")
    assert empty.metadata["jev"] == {
        "final_phase": None,
        "final_world": {},
        "decisions": 0,
        "stalls": 0,
    }


def test_lazy_client_and_perceiver_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from inspect_robots_jev import policy as policy_mod

    built: list[str] = []

    class StubClient:
        def __init__(self, **kw: Any) -> None:
            built.append(f"client:{kw['api']}:{kw['model']}")

    class StubPerceiver:
        def __init__(self, **kw: Any) -> None:
            built.append(f"perceiver:{kw['camera']}")

        def reset(self) -> None:
            pass

    monkeypatch.setattr(policy_mod, "DecisionsClient", StubClient)
    monkeypatch.setattr(policy_mod, "Perceiver", StubPerceiver)
    monkeypatch.setattr(TagLayout, "load", classmethod(lambda cls, p: "layout"))
    monkeypatch.setattr(Calibration, "load", classmethod(lambda cls, p: "cal"))
    policy = JevPolicy(JevPolicyConfig(api="typesafe", model="jev-latest", camera="cam"))
    policy._client_instance()
    policy._perceiver_instance()
    policy.reset(Scene(id="s", instruction="x"))
    assert built == ["client:typesafe:jev-latest", "perceiver:cam"]


def test_jev_policy_registry_entry_parses_strings() -> None:
    policy = jev_policy(
        steps_cm="0.5,2", history="3", stale_after="4", cube="red cube", api="typesafe"
    )
    assert policy.settings.steps_cm == (0.5, 2.0)
    assert (
        policy.settings.history == 3
        and policy.settings.stale_after == 4
        and policy.settings.api == "typesafe"
    )
    assert policy.settings.task == TaskConfig(cube="red cube")
    assert jev_policy(steps_cm=[1, 2]).settings.steps_cm == (1.0, 2.0)
    with pytest.raises(TypeError, match="unknown jev policy args"):
        jev_policy(bogus=1)


def test_reset_without_perceiver_is_safe() -> None:
    policy = JevPolicy()
    policy.reset(Scene(id="s", instruction="x"))
    assert policy.transcript() == [] and policy.settings == JevPolicyConfig()


def test_descend_does_not_stall_when_cube_tag_is_hidden() -> None:
    perc = FakePerceiver()
    policy, client, _ = make(["left_z_minus_2cm"] * 3, perc)
    policy.act(obs())
    assert policy._machine is not None
    policy._machine._name = "descend"
    perc.visible = False
    for step in range(1, 3):
        policy.act(obs(left=(0.30, 0.10, 0.08), step=step))
    # both descend acts asked Jev although the cube was unseen: the goal is the remembered pose
    assert len(client.calls) == 3
    assert not any(e.get("stall") for e in policy.transcript())


def test_release_with_hold_is_not_scored_as_success() -> None:
    perc = FakePerceiver()
    policy, _, _ = make(["left_z_plus_2cm", "left_z_minus_0.5cm", "hold"], perc)
    policy.act(obs())
    assert policy._machine is not None
    policy._machine._name = "lower"
    policy.act(obs(left=(0.35, -0.10, 0.09), opening=0.3, step=1))  # held during lower
    assert policy._held == ("left", "cube")
    policy._machine._name = "release"
    chunk = policy.act(obs(left=(0.35, -0.10, 0.03), opening=0.3, step=2))  # Jev says hold
    meta = chunk.actions[-1].meta["jev"]
    # the world for this step was built with the cube pinned to the gripper
    assert meta["held"] is True and meta["world"]["cube"]["from_gripper"] is True
    record = TrialRecord(scene_id="s", epoch=0, seed=None)
    record.steps.append(
        StepRecord(
            t=0, observation=obs(), action=chunk.actions[-1], result=StepResult(observation=obs())
        )
    )
    assert cube_in_bowl()(record, None).value is False


def test_steps_cm_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        jev_policy(steps_cm="0,2")
    with pytest.raises(ValueError, match="positive"):
        jev_policy(steps_cm="")


def test_resync_keeps_the_gripper_commanded_closed() -> None:
    perc = FakePerceiver()
    policy, _, _ = make(["left_z_minus_2cm", "left_close", "left_z_plus_2cm"], perc)
    policy.act(obs())
    assert policy._machine is not None
    policy._machine._name = "grasp"
    closing = policy.act(obs(left=(0.20, 0.0, 0.18), step=1))
    assert closing.actions[-1].data[6] == 0.0
    # the jaws stopped on the cube at 0.3 (5 cm off in x too: position resyncs)
    policy._machine._name = "lift"
    lift = policy.act(obs(left=(0.25, 0.0, 0.18), opening=0.3, step=2))
    assert lift.actions[0].data[0] == pytest.approx(0.25)  # position restarted from measured
    assert all(a.data[6] == 0.0 for a in lift.actions)  # grip stays commanded closed


def test_stall_in_approach_rises() -> None:
    perc = FakePerceiver(objects={"bowl": BOWL})
    policy, _, _ = make([], perc)
    perc.objects["cube"] = CUBE
    perc.seen_step["cube"] = -20  # remembered long ago
    perc.visible = False
    policy.act(obs())  # reset happens (cube in memory) then the target is stale
    chunk = policy.act(obs(step=1))
    assert chunk.actions[-1].meta["jev"]["stall"] == "cube tag not seen"
    assert chunk.actions[-1].data[2] == pytest.approx(0.20 + 0.05)
