"""Which of the Dream client's errors `DreamSession` asks again after, by the client's own classes.

Skipped where the Dream client is not installed.
"""

from __future__ import annotations

import warnings

import pytest
from _dream_fakes import Clock, FakeClient, encode

from inspect_robots_dream.session import DreamSession, Episode, _dream_client

dream = pytest.importorskip("dream", reason="these tests raise the Dream client's own errors")


def make_session(client: FakeClient, clock: Clock) -> DreamSession:
    return DreamSession(lambda: client, encode=encode, sleep=clock.sleep, clock=clock)


def refusal(reason: str, retryable: bool, retry_after_s: float | None = None) -> Exception:
    error: Exception = dream.DreamServerError(
        f"DA-2000 ({reason})", reason=reason, retryable=retryable, retry_after_s=retry_after_s
    )
    return error


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
    "error",
    [
        refusal("max_serves", False),
        dream.DreamAuthError("DA-1000"),
        dream.DreamBadRequest("DA-4000 BAD_REQUEST (too_large)", code="DA-4000"),
        RuntimeError("bug"),
    ],
)
def test_act_raises_what_asking_again_cannot_fix(clock: Clock, error: Exception) -> None:
    session = make_session(FakeClient(errors=[error]), clock)
    session.start(Episode())
    with pytest.raises(type(error)) as raised:
        session.act(None)
    assert raised.value is error


def test_act_gives_up_when_the_next_wait_passes_the_timeout(clock: Clock) -> None:
    session = make_session(FakeClient(errors=[refusal("pool_warming", True, 50.0)] * 3), clock)
    session.start(Episode(timeout_s=120))
    with pytest.raises(dream.DreamServerError, match="pool_warming"):
        session.act(None)
    assert clock.slept == [50.0, 50.0]


def test_default_client_is_the_dream_client_without_slow_network_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(_dream_client(), dream.DreamClient)
        warnings.warn("slow", dream.DreamSlowNetworkWarning, stacklevel=1)
    assert caught == []
