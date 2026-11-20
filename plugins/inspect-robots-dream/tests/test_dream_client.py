"""The policy against the real ``dream.DreamClient``, with only its HTTP transport stubbed.

The client's own frame window, envelope parsing, error mapping and seam smoothing all run, and so
does the station's JPEG encoder. Skipped where the Dream client or Pillow is not installed.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _dream_fakes import HOME, LEFT, RIGHT, STATE, Clock, FakeYam, observation, reply, yam_info

from inspect_robots import GIVE_UP, Scene, Task, eval
from inspect_robots_dream import DreamPolicy
from inspect_robots_dream._station import Vector
from inspect_robots_dream.session import DEFAULT_MODEL, DreamSession

dream = pytest.importorskip("dream", reason="these tests run the real Dream client")
transport = pytest.importorskip("dream.transport")
pytest.importorskip("PIL", reason="these tests encode real JPEG frames")

SCENE = Scene(id="cube", instruction="put the red cube on the plate")
STATION_KEYS = {
    "prompt",
    "robot.arm",
    "robot.exec_head",
    "robot.max_joint_vel_rad_s",
    "robot.gripper_open",
    "state.arm_l.joint_pos",
    "state.arm_l.gripper_pos",
    "state.arm_r.joint_pos",
    "state.arm_r.gripper_pos",
    "image_window.top",
    "image_window.left",
    "image_window.right",
}


def answer(session_id: str, left: Vector = LEFT, right: Vector = RIGHT) -> dict[str, Any]:
    chunk = reply(left, right) if left.ndim == 1 else {"q_l_chunk": left, "q_r_chunk": right}
    return {
        "status": {"code": "DA-0000"},
        "session_id": session_id,
        "actions": {key: np.asarray(chunk[key]).tolist() for key in ("q_l_chunk", "q_r_chunk")},
        "timing": {"serve_infer_ms": 280},
    }


def gateway(
    monkeypatch: pytest.MonkeyPatch, envelopes: list[dict[str, Any]], latency_s: float = 0.0
) -> list[dict[str, Any]]:
    """Answer the client's get_action requests from ``envelopes``; returns what it sent."""
    sent: list[dict[str, Any]] = []

    def request(
        self: Any,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        path: str = transport.GET_ACTION_PATH,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        if path != transport.GET_ACTION_PATH:
            return {"status": {"code": "DA-0000"}}
        time.sleep(latency_s)
        sent.append(payload)
        return envelopes.pop(0)

    monkeypatch.setattr(transport.Transport, "request", request)
    monkeypatch.setattr(
        dream.DreamClient, "get_models", lambda self: [SimpleNamespace(id=DEFAULT_MODEL)]
    )
    return sent


def offline_client() -> Callable[[], Any]:
    return lambda: dream.DreamClient(api_key="dl_test_offline")


def test_the_dream_client_carries_our_ticks_and_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    envelopes = [
        {
            "status": {
                "code": "DA-2000",
                "reason": "pool_warming",
                "retryable": True,
                "retry_after_s": 2,
            }
        },
        answer("s1"),
        {"status": {"code": "DA-0001", "reason": "expired"}, "session_id": "s2"},
        answer("s2"),
    ]
    sent = gateway(monkeypatch, envelopes)
    session = DreamSession(offline_client(), sleep=clock.sleep, clock=clock)
    yam = FakeYam()
    with DreamPolicy(session=session) as policy:
        task = Task("contract", [SCENE], "episode_length", max_steps=82)
        [log] = eval(task, policy, yam, log_dir=str(tmp_path))

    assert log.status == "success" and envelopes == [] and clock.slept == [2.0, 0.0]
    first, _, expired, resumed = sent
    assert first["checkpoint"] == DEFAULT_MODEL and first["session_id"] is None
    assert set(first["observation"]) == STATION_KEYS
    for camera in ("top", "left", "right"):
        assert len(first["observation"][f"image_window.{camera}"]) == 9
    assert first["observation"]["prompt"] == SCENE.instruction
    assert (expired["session_id"], resumed["session_id"]) == ("s1", "s2")
    np.testing.assert_allclose(yam.targets[32:], np.tile(STATE, (50, 1)))
    transcript = log.samples[0].policy_transcripts[0]
    assert isinstance(transcript, dict)
    assert [(r["step"], r["attempts"], r["session_id"]) for r in transcript["requests"]] == [
        (32, 2, "s1"),
        (80, 2, "s2"),
    ]


def test_the_dream_clients_safety_refusal_ends_the_trial_as_give_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    # The status Dream answered with on 2026-11-11, asked to pull a tablecloth from under plates.
    reason = "Pulling the tablecloth risks pulling breakable plates onto the floor."
    refused = {
        "status": {
            "code": "DA-4000",
            "name": "BAD_REQUEST",
            "reason": "malformed",
            "detail": f"unsafe instruction, refused: {reason}",
        }
    }
    sent = gateway(monkeypatch, [refused])
    session = DreamSession(offline_client(), sleep=clock.sleep, clock=clock)
    yam = FakeYam()
    with DreamPolicy(session=session) as policy:
        task = Task("contract", [SCENE], "episode_length", max_steps=82)
        [log] = eval(task, policy, yam, log_dir=str(tmp_path))

    assert log.status == "success" and len(sent) == 1 and clock.slept == []
    [sample] = log.samples
    assert sample.error is None
    assert (sample.termination_reasons, sample.termination_details) == ((GIVE_UP,), (reason,))
    assert [target.tolist() for target in yam.targets] == [HOME.tolist()] * 32


def test_a_gripper_that_closes_in_one_row_survives_the_clients_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = np.tile(LEFT, (48, 1)), np.tile(RIGHT, (48, 1))
    left[:, 6] = np.r_[np.ones(24), np.zeros(24)]
    right[:, 6] = np.tile([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 8)
    gateway(monkeypatch, [answer("s", left, right)])
    policy = DreamPolicy(session=DreamSession(offline_client()), preroll_steps=0)
    policy.bind(yam_info())
    policy.reset(SCENE)
    grippers = np.array([policy.act(observation()).actions[0].data[[6, 13]] for _ in range(48)])
    policy.close()
    assert grippers.min() == 0.0 and grippers.max() == 1.0
    assert grippers[0, 0] == pytest.approx(1.0) and grippers[-1, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("hz", [3.0, 20.0])
def test_the_first_request_carries_a_full_frame_window(
    monkeypatch: pytest.MonkeyPatch, hz: float
) -> None:
    sent = gateway(monkeypatch, [answer("s")])
    policy = DreamPolicy(session=DreamSession(offline_client()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        policy.bind(yam_info(control_hz=hz))
    policy.reset(SCENE)
    for _ in range(policy.preroll_steps + 1):
        policy.act(observation())
    policy.close()
    [first] = sent
    assert [len(first["observation"][f"image_window.{c}"]) for c in ("top", "left", "right")] == [
        9,
        9,
        9,
    ]


def test_playback_does_not_depend_on_dream_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    def plan(offset: float) -> Vector:
        rows = np.tile(LEFT, (48, 1))
        rows[:, 0] = offset + 0.01 * np.arange(48)
        return rows

    def targets(latency_s: float) -> Vector:
        envelopes = [answer("s", plan(k), np.tile(RIGHT, (48, 1))) for k in range(3)]
        gateway(monkeypatch, envelopes, latency_s)
        policy = DreamPolicy(
            session=DreamSession(offline_client()), rows_per_request=8, preroll_steps=0
        )
        policy.bind(yam_info())
        policy.reset(SCENE)
        state, played = STATE, []
        for _ in range(24):
            state = policy.act(observation(state)).actions[0].data
            played.append(state)
        policy.close()
        return np.array(played)

    fast, slow = targets(0.0), targets(0.15)
    np.testing.assert_allclose(fast, slow)
    assert fast[15, 0] > plan(0)[15, 0] + 0.1
