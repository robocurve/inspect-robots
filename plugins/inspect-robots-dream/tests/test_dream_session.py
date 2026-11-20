"""`DreamSession` against a fake Dream client. Nothing here needs the Dream client or Pillow."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from _dream_fakes import IMAGES, LEFT, RIGHT, STATE, Clock, DreamRefusal, FakeClient, encode

from inspect_robots_dream.session import (
    DEFAULT_MODEL,
    DreamSession,
    Episode,
    Refused,
    _dream_client,
)


def make_session(client: FakeClient, clock: Clock) -> DreamSession:
    return DreamSession(lambda: client, encode=encode, sleep=clock.sleep, clock=clock)


@pytest.mark.parametrize(
    "fields",
    [
        {"model": ""},
        {"model": 3},
        {"timeout_s": float("inf")},
        {"timeout_s": "60"},
        {"record": "no"},
    ],
)
def test_episode_rejects_bad_settings(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Episode(**fields)


def test_start_opens_the_model_for_a_paused_rollout(clock: Clock) -> None:
    client = FakeClient()
    make_session(client, clock).start(Episode(record=True))
    assert client.names() == ["get_models", "policy"]
    assert client.args_of("policy") == [
        {
            "checkpoint": DEFAULT_MODEL,
            "_record": True,
            "action_hz": 1e-3,
            "strict": False,
            "timeout_ms": 45_000,
            "obs_window": 9,
            "obs_fps": 5.0,
        }
    ]


def test_later_starts_reset_the_episode_until_settings_change(clock: Clock) -> None:
    client = FakeClient(models=(DEFAULT_MODEL, "dreamlabs/dream-1-beta-flash"))
    session = make_session(client, clock)
    session.start(Episode())
    session.start(Episode(timeout_s=60))
    session.start(Episode(record=True))
    session.start(Episode(model="dreamlabs/dream-1-beta-flash"))
    assert client.names() == ["get_models", "policy", "reset_episode", "policy", "policy"]


def test_check_asks_for_the_model_list_only_when_needed(clock: Clock) -> None:
    client = FakeClient()
    session = make_session(client, clock)
    session.check(Episode())
    session.check(Episode(timeout_s=60))
    session.start(Episode())
    assert client.names() == ["get_models", "policy"]
    with pytest.raises(ValueError, match="'dreamlabs/dream-2' is not a model"):
        session.check(Episode(model="dreamlabs/dream-2"))


def test_observe_sends_encoded_frames_and_station_keys_at_the_given_time(clock: Clock) -> None:
    client = FakeClient()
    session = make_session(client, clock)
    session.start(Episode())
    session.observe({**IMAGES, "wrist_cam": IMAGES["top_cam"]}, STATE, "stack the cups", 1.25)
    [(frames, keys, timestamp)] = client.args_of("set_observation")
    assert frames == {
        "image_window.top": "frame-1",
        "image_window.left": "frame-2",
        "image_window.right": "frame-3",
    }
    assert keys["prompt"] == "stack the cups" and keys["robot.arm"] == "yam"
    assert keys["state.arm_r.joint_pos"] == RIGHT[:6].tolist()
    assert timestamp == 1.25


def test_observe_names_a_camera_it_was_not_given(clock: Clock) -> None:
    client = FakeClient()
    session = make_session(client, clock)
    session.start(Episode())
    with pytest.raises(ValueError, match="right_cam"):
        session.observe({"top_cam": IMAGES["top_cam"]}, STATE, "x", 0.0)
    assert client.args_of("set_observation") == []


def test_act_returns_packed_rows_and_passes_the_last_row(clock: Clock) -> None:
    client = FakeClient()
    session = make_session(client, clock)
    session.start(Episode())
    chunk = session.act(STATE)
    [commanded] = client.args_of("get_action")
    assert commanded["q_l_chunk"].tolist() == LEFT.tolist()
    assert commanded["q_r_chunk"].tolist() == RIGHT.tolist()
    assert chunk.rows.shape == (48, 14) and chunk.rows[-1].tolist() == STATE.tolist()
    assert (chunk.attempts, chunk.infer_ms, chunk.session_id) == (1, 310, "session-1")
    client.last_timing = None
    assert session.act(None).infer_ms is None
    assert client.args_of("get_action")[-1] is None


@pytest.mark.parametrize(
    ("detail", "reason"),
    [
        (
            "unsafe instruction, refused: Pulling the tablecloth risks breaking the plates.",
            "Pulling the tablecloth risks breaking the plates.",
        ),
        ("unsafe instruction, refused", "DA-4000 BAD_REQUEST: unsafe instruction, refused"),
    ],
)
def test_act_raises_refused_with_dreams_reason_and_does_not_ask_again(
    clock: Clock, detail: str, reason: str
) -> None:
    client = FakeClient(errors=[DreamRefusal(detail)])
    session = make_session(client, clock)
    session.start(Episode())
    with pytest.raises(Refused) as raised:
        session.act(None)
    assert raised.value.reason.endswith(reason)
    assert isinstance(raised.value.__cause__, DreamRefusal)
    assert client.names().count("get_action") == 1 and clock.slept == []


@pytest.mark.parametrize(
    "error",
    [
        DreamRefusal("unsafe instruction, refused: too risky", code="DA-5000"),
        DreamRefusal("prompt 'say unsafe instruction, refused: x' exceeds 512 chars"),
        DreamRefusal("frames too large"),
    ],
)
def test_act_raises_any_other_error_as_it_came(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, error: DreamRefusal
) -> None:
    # Without the Dream client there is nothing to classify the error by, so it is not retried.
    monkeypatch.setitem(sys.modules, "dream", None)
    session = make_session(FakeClient(errors=[error]), clock)
    session.start(Episode())
    with pytest.raises(DreamRefusal) as raised:
        session.act(None)
    assert raised.value is error and clock.slept == []


def test_observe_and_act_need_an_episode(clock: Clock) -> None:
    session = make_session(FakeClient(), clock)
    with pytest.raises(RuntimeError, match="start an episode"):
        session.observe(IMAGES, STATE, "x", 0.0)
    with pytest.raises(RuntimeError, match="start an episode"):
        session.act(None)


def test_close_releases_the_client_and_a_later_start_reopens(clock: Clock) -> None:
    made: list[FakeClient] = []

    def make() -> FakeClient:
        made.append(FakeClient())
        return made[-1]

    session = DreamSession(make, sleep=clock.sleep, clock=clock)
    session.start(Episode())
    session.close()
    session.close()
    session.start(Episode())
    assert len(made) == 2
    assert made[0].args_of("close") == [None] and made[1].args_of("close") == []


@pytest.mark.parametrize("module", [None, types.ModuleType("dream")])
def test_default_client_explains_a_missing_or_wrong_dream_package(
    monkeypatch: pytest.MonkeyPatch, module: types.ModuleType | None
) -> None:
    monkeypatch.setitem(sys.modules, "dream", module)
    with pytest.raises(ImportError, match="pip install dreamlabs-client"):
        _dream_client()
