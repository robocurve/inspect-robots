from __future__ import annotations

from pathlib import Path

import pytest
from _dream_fakes import HOME, STATE, Clock, FakeClient, FakeYam, serving

from inspect_robots import Scene, Task, eval, resolve
from inspect_robots_dream import DreamPolicy
from inspect_robots_dream.bridge import BridgeServer
from inspect_robots_dream.session import DreamSession


def task(steps: int) -> Task:
    scene = Scene(id="cube", instruction="put the red cube on the plate")
    return Task(name="dream-e2e", scenes=[scene], scorer="episode_length", max_steps=steps)


def test_dream_drives_a_yam_through_eval(tmp_path: Path, clock: Clock) -> None:
    client = FakeClient()
    yam = FakeYam()
    session = DreamSession(lambda: client, sleep=clock.sleep, clock=clock)
    with DreamPolicy(session=session) as policy:
        [log] = eval(task(82), policy, yam, log_dir=str(tmp_path))

    assert log.status == "success"
    assert [target.tolist() for target in yam.targets[:32]] == [HOME.tolist()] * 32
    assert all(target.tolist() == STATE.tolist() for target in yam.targets[32:])
    timestamps = [timestamp for *_, timestamp in client.args_of("set_observation")]
    assert timestamps == pytest.approx([step / 20 for step in range(82)])
    transcript = log.samples[0].policy_transcripts[0]
    assert isinstance(transcript, dict)
    assert [request["step"] for request in transcript["requests"]] == [32, 80]
    assert transcript["tracking_error_rad"] == {"mean": 0.0, "max": 0.0}


def test_dream_drives_a_yam_through_the_bridge(tmp_path: Path, clock: Clock) -> None:
    client = FakeClient()
    session = DreamSession(lambda: client, sleep=clock.sleep, clock=clock)
    with (
        serving(BridgeServer(session, "127.0.0.1", 0)) as url,
        DreamPolicy(server_url=url) as policy,
    ):
        [log] = eval(task(40), policy, FakeYam(), log_dir=str(tmp_path))

    assert log.status == "success"
    assert client.names().count("set_observation") == 40
    assert client.names().count("get_action") == 1
    transcript = log.samples[0].policy_transcripts[0]
    assert isinstance(transcript, dict) and transcript["server_url"] == url


def test_the_entry_point_builds_the_policy() -> None:
    policy = resolve("policy", "dream", rows_per_request=12, dry_run=True)
    assert isinstance(policy, DreamPolicy)
    assert (policy.rows_per_request, policy.dry_run) == (12, True)
