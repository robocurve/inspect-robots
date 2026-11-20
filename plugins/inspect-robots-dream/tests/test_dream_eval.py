"""The policy through `eval()` against a fake YAM and a fake Dream client: no Dream client, Pillow
or network needed."""

from __future__ import annotations

from pathlib import Path

import pytest
from _dream_fakes import HOME, STATE, Clock, DreamRefusal, FakeClient, FakeYam, encode

from inspect_robots import GIVE_UP, Scene, Task, eval, resolve
from inspect_robots_dream import DreamPolicy
from inspect_robots_dream.session import DreamSession


def task(steps: int) -> Task:
    scene = Scene(id="cube", instruction="put the red cube on the plate")
    return Task(name="dream-e2e", scenes=[scene], scorer="episode_length", max_steps=steps)


def session(client: FakeClient, clock: Clock) -> DreamSession:
    return DreamSession(lambda: client, encode=encode, sleep=clock.sleep, clock=clock)


def test_dream_drives_a_yam_through_eval(tmp_path: Path, clock: Clock) -> None:
    client = FakeClient()
    yam = FakeYam()
    with DreamPolicy(session=session(client, clock)) as policy:
        [log] = eval(task(82), policy, yam, log_dir=str(tmp_path))

    assert log.status == "success"
    assert [target.tolist() for target in yam.targets[:32]] == [HOME.tolist()] * 32
    assert all(target.tolist() == STATE.tolist() for target in yam.targets[32:])
    ticks = client.args_of("set_observation")
    assert [timestamp for *_, timestamp in ticks] == pytest.approx(
        [step / 20 for step in range(82)]
    )
    assert set(ticks[0][0].values()) == {"frame-90"}
    assert log.samples[0].termination_reasons == ("max_steps",)
    transcript = log.samples[0].policy_transcripts[0]
    assert isinstance(transcript, dict)
    assert [request["step"] for request in transcript["requests"]] == [32, 80]
    assert transcript["tracking_error_rad"] == {"mean": 0.0, "max": 0.0}


def test_a_refused_instruction_is_logged_as_give_up_not_as_an_error(
    tmp_path: Path, clock: Clock
) -> None:
    reason = "Pulling the tablecloth risks pulling breakable plates onto the floor."
    client = FakeClient(errors=[DreamRefusal(f"unsafe instruction, refused: {reason}")])
    yam = FakeYam()
    with DreamPolicy(session=session(client, clock)) as policy:
        [log] = eval(task(82), policy, yam, log_dir=str(tmp_path))

    assert log.status == "success" and log.results is not None
    [sample] = log.samples
    assert sample.error is None
    assert sample.termination_reasons == (GIVE_UP,)
    assert sample.termination_details == (reason,)
    assert [target.tolist() for target in yam.targets] == [HOME.tolist()] * 32
    assert client.names().count("get_action") == 1
    transcript = sample.policy_transcripts[0]
    assert isinstance(transcript, dict) and transcript["refusal"] == reason


def test_the_entry_point_builds_the_policy() -> None:
    policy = resolve("policy", "dream", rows_per_request=12, dry_run=True)
    assert isinstance(policy, DreamPolicy)
    assert (policy.rows_per_request, policy.dry_run) == (12, True)
