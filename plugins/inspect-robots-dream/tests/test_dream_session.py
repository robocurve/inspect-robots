from __future__ import annotations

import sys
import types
import warnings
from typing import Any

import dream
import numpy as np
import pytest
from _dream_fakes import FRAMES, LEFT, RIGHT, STATE, Clock, FakeClient

from inspect_robots_dream.session import (
    DEFAULT_MODEL,
    Chunk,
    DreamSession,
    Episode,
    _dream_client,
)


def make_session(client: FakeClient, clock: Clock) -> DreamSession:
    return DreamSession(lambda: client, sleep=clock.sleep, clock=clock)


def refusal(reason: str, retryable: bool, retry_after_s: float | None = None) -> Exception:
    return dream.DreamServerError(
        f"DA-2000 ({reason})", reason=reason, retryable=retryable, retry_after_s=retry_after_s
    )


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


def test_chunk_round_trips_through_json() -> None:
    chunk = Chunk(
        rows=np.tile(STATE, (3, 1)), rtt_s=0.4, attempts=2, infer_ms=300.0, session_id="s"
    )
    back = Chunk.from_json(chunk.to_json())
    assert back.rows.tolist() == chunk.rows.tolist()
    assert (back.rtt_s, back.attempts, back.infer_ms, back.session_id) == (0.4, 2, 300.0, "s")
    with pytest.raises(ValueError, match="finite"):
        Chunk.from_json({**chunk.to_json(), "rows": [[0.0] * 13]})


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


def test_observe_sends_station_keys_at_the_given_time(clock: Clock) -> None:
    client = FakeClient()
    session = make_session(client, clock)
    session.start(Episode())
    session.observe(FRAMES, STATE, "stack the cups", 1.25)
    [(frames, keys, timestamp)] = client.args_of("set_observation")
    assert frames == {"image_window.top": "T", "image_window.left": "L", "image_window.right": "R"}
    assert keys["prompt"] == "stack the cups" and keys["robot.arm"] == "yam"
    assert keys["state.arm_r.joint_pos"] == RIGHT[:6].tolist()
    assert timestamp == 1.25


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


def test_act_asks_again_after_refusals_worth_repeating(
    clock: Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    errors = [
        refusal("pool_warming", True, 30.0),
        refusal("rate_limited", True),
        dream.DreamSessionExpired("DA-0001", reason="expired"),
        dream.DreamTimeout("no answer within 45000 ms"),
        dream.DreamProtocolError("gateway unreachable"),
    ]
    session = make_session(FakeClient(errors=errors), clock)
    session.start(Episode())
    assert session.act(None).attempts == 6
    assert clock.slept == [30.0, 1.0, 0.0, 0.0, 1.0]
    assert capsys.readouterr().err.splitlines() == [
        "Dream: pool_warming; asking again in 30 s",
        "Dream: rate_limited; asking again in 1 s",
        "Dream: expired; asking again in 0 s",
        "Dream: DreamTimeout; asking again in 0 s",
        "Dream: DreamProtocolError; asking again in 1 s",
    ]


@pytest.mark.parametrize(
    "error", [refusal("max_serves", False), dream.DreamAuthError("DA-1000"), RuntimeError("bug")]
)
def test_act_raises_what_asking_again_cannot_fix(clock: Clock, error: Exception) -> None:
    session = make_session(FakeClient(errors=[error]), clock)
    session.start(Episode())
    with pytest.raises(type(error)):
        session.act(None)


def test_act_gives_up_when_the_next_wait_passes_the_timeout(clock: Clock) -> None:
    session = make_session(FakeClient(errors=[refusal("pool_warming", True, 50.0)] * 3), clock)
    session.start(Episode(timeout_s=120))
    with pytest.raises(dream.DreamServerError, match="pool_warming"):
        session.act(None)
    assert clock.slept == [50.0, 50.0]


def test_observe_and_act_need_an_episode(clock: Clock) -> None:
    session = make_session(FakeClient(), clock)
    with pytest.raises(RuntimeError, match="start an episode"):
        session.observe(FRAMES, STATE, "x", 0.0)
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


def test_default_client_is_the_dream_client_without_slow_network_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(_dream_client(), dream.DreamClient)
        warnings.warn("slow", dream.DreamSlowNetworkWarning, stacklevel=1)
    assert caught == []


@pytest.mark.parametrize("module", [None, types.ModuleType("dream")])
def test_default_client_explains_a_missing_or_wrong_dream_package(
    monkeypatch: pytest.MonkeyPatch, module: types.ModuleType | None
) -> None:
    monkeypatch.setitem(sys.modules, "dream", module)
    with pytest.raises(ImportError, match="pip install dreamlabs-client"):
        _dream_client()
