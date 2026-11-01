from __future__ import annotations

import base64
import io
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
import pytest
from _dream_fakes import FRAMES, LEFT, STATE, Clock, FakeClient, FakeSession, serving
from PIL import Image

from inspect_robots_dream import bridge
from inspect_robots_dream.bridge import BridgeError, BridgeServer, BridgeSession
from inspect_robots_dream.session import Chunk, DreamSession, Episode


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def url(client: FakeClient, clock: Clock) -> Iterator[str]:
    session = DreamSession(lambda: client, sleep=clock.sleep, clock=clock)
    with serving(BridgeServer(session, "127.0.0.1", 0)) as address:
        yield address


def raw_post(url: str, data: bytes) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def answer_with(status: int, body: bytes) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return HTTPServer(("127.0.0.1", 0), Handler)


def test_a_bridged_episode_reaches_the_dream_client(url: str, client: FakeClient) -> None:
    session = BridgeSession(url + "/")
    session.check(Episode())
    session.start(Episode(timeout_s=60))
    session.observe(FRAMES, STATE, "put the red cube on the plate", 0.05)
    chunk = session.act(STATE)
    assert client.names()[:2] == ["get_models", "policy"]
    [(frames, keys, timestamp)] = client.args_of("set_observation")
    assert frames["image_window.top"] == "T" and keys["prompt"] == "put the red cube on the plate"
    assert timestamp == 0.05
    assert client.args_of("get_action")[0]["q_l_chunk"].tolist() == LEFT.tolist()
    assert chunk.rows.shape == (48, 14) and chunk.session_id == "session-1"
    session.close()


def test_check_reports_a_model_the_key_may_not_run(url: str) -> None:
    with pytest.raises(BridgeError, match="ValueError: 'dreamlabs/dream-2' is not a model"):
        BridgeSession(url).check(Episode(model="dreamlabs/dream-2"))


def test_only_the_latest_episode_may_observe_or_act(url: str) -> None:
    first, second = BridgeSession(url), BridgeSession(url)
    with pytest.raises(BridgeError, match="not the one the bridge is running"):
        first.act(None)
    first.start(Episode())
    second.start(Episode())
    second.observe(FRAMES, STATE, "x", 0.0)
    with pytest.raises(BridgeError, match="not the one the bridge is running"):
        first.observe(FRAMES, STATE, "x", 0.0)


def test_health_answers_while_another_connection_idles(url: str) -> None:
    port = int(url.rsplit(":", 1)[1])
    with (
        socket.create_connection(("127.0.0.1", port)),
        urllib.request.urlopen(url + "/health", timeout=5) as response,
    ):
        assert json.load(response) == {"ok": True}
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(url + "/nope")
    assert missing.value.code == 404
    with pytest.raises(BridgeError, match="NotFound: /nope/start"):
        BridgeSession(url + "/nope").start(Episode())


def test_malformed_requests_are_refused_by_name(url: str) -> None:
    _, started = raw_post(url + "/start", b"{}")
    tick = {"episode": started["episode"], "frames": FRAMES, "state": STATE.tolist(), "prompt": "x"}
    for body, error in [
        (b"[]", "TypeError"),
        (b"{not json", "JSONDecodeError"),
        (json.dumps({**tick, "frames": []}).encode(), "TypeError"),
        (json.dumps({**tick, "state": [0.0]}).encode(), "ValueError"),
        (json.dumps(tick).encode(), "KeyError"),
    ]:
        status, answer = raw_post(url + "/observe", body)
        assert (status, answer["error"]) == (500, error)


def test_failures_behind_the_bridge_are_reported_by_name() -> None:
    class Down(FakeSession):
        def act(self, commanded: np.ndarray | None) -> Chunk:
            raise RuntimeError("Dream is down")

    with serving(BridgeServer(Down(), "127.0.0.1", 0)) as url:
        session = BridgeSession(url)
        session.start(Episode())
        with pytest.raises(BridgeError, match="RuntimeError: Dream is down"):
            session.act(None)


def test_a_missing_bridge_is_a_connection_failure() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(ConnectionError, match="no Dream bridge answering"):
        BridgeSession(f"http://127.0.0.1:{port}").start(Episode())


def test_a_bridge_that_never_answers_is_a_connection_failure() -> None:
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        url = f"http://127.0.0.1:{silent.getsockname()[1]}/check"
        with pytest.raises(ConnectionError, match="timed out"):
            bridge._post(url, {}, 0.2)


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [(500, b"<html>oops</html>", "500: Internal Server Error"), (200, b"<html>ok</html>", "JSON")],
)
def test_a_foreign_server_is_reported(status: int, body: bytes, match: str) -> None:
    with serving(answer_with(status, body)) as url, pytest.raises(BridgeError, match=match):
        BridgeSession(url).check(Episode())


def test_act_waits_for_every_retry_the_bridge_may_make() -> None:
    sent: list[tuple[str, Mapping[str, Any], float]] = []

    def record(url: str, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        sent.append((url, payload, timeout_s))
        if url.endswith("/start"):
            return {"episode": "e1"}
        return Chunk(rows=np.tile(STATE, (2, 1)), rtt_s=0.3, attempts=1).to_json()

    session = BridgeSession("http://bridge", post=record)
    session.start(Episode(timeout_s=90))
    session.act(None)
    assert [(url, timeout) for url, _, timeout in sent] == [
        ("http://bridge/start", 60.0),
        ("http://bridge/act", 150.0),
    ]
    assert sent[0][1] == {"model": "dreamlabs/dream-1-beta", "record": False, "timeout_s": 90}
    assert sent[1][1] == {"commanded": None, "episode": "e1"}


def test_warm_asks_for_one_chunk_on_a_still_scene() -> None:
    session = FakeSession()
    bridge._warm(session, "dreamlabs/dream-1-beta-flash", 1800.0)
    [episode] = session.episodes
    assert (episode.model, episode.timeout_s) == ("dreamlabs/dream-1-beta-flash", 1800.0)
    [(frames, state, prompt, timestamp)] = session.ticks
    image = Image.open(io.BytesIO(base64.b64decode(frames["top_cam"])))
    assert image.size == (320, 176) and np.asarray(image.convert("L")).max() < 10
    assert state.tolist() == 2 * ([0.0] * 6 + [1.0])
    assert (prompt, timestamp, session.commanded) == ("hold still", 0.0, [None])


def test_main_warms_then_serves_until_interrupted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], clock: Clock
) -> None:
    client = FakeClient()
    monkeypatch.setattr(
        bridge, "DreamSession", lambda: DreamSession(lambda: client, sleep=clock.sleep, clock=clock)
    )

    def interrupt(self: BridgeServer, poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(BridgeServer, "serve_forever", interrupt)
    assert bridge.main(["--warm", "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert "dreamlabs/dream-1-beta answered in" in out and "listening on http://127.0.0.1:" in out
    assert client.args_of("close") == [None]
    assert bridge.main(["--port", "0"]) == 0
    assert "answered" not in capsys.readouterr().out
    assert len(client.args_of("get_action")) == 1
