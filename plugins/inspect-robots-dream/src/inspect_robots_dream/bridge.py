"""Serve the Dream client to the ``dream`` policy over loopback HTTP, from its own environment.

Isaac Sim 5.1 pins websockets 12 and the Dream client needs 13 or newer, so under Isaac Sim the
client runs in a separate venv as this bridge::

    inspect-robots-dream-bridge --warm                      # in the Dream venv
    inspect-robots run --policy dream -P server_url=http://127.0.0.1:8210 ...

JSON in and out: ``POST /check`` and ``POST /start`` (an episode), ``POST /observe`` (one tick),
``POST /act`` (one chunk) and ``GET /health``. Requests reach the Dream client one at a time, and
only the episode that started last may observe or act. The bridge has no authentication, so it
listens on loopback unless told otherwise.
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing, suppress
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import numpy as np

from inspect_robots_dream import _station
from inspect_robots_dream._station import Vector
from inspect_robots_dream.session import (
    DEFAULT_MODEL,
    REQUEST_TIMEOUT_S,
    Chunk,
    DreamSession,
    Episode,
    Session,
)

DEFAULT_PORT = 8210

_Post = Callable[[str, Mapping[str, Any], float], dict[str, Any]]
_Route = Callable[[dict[str, Any]], dict[str, Any]]


class BridgeError(RuntimeError):
    """The bridge answered, but refused the request."""


class BridgeSession:
    """A `Session` whose Dream client runs in a bridge process at ``url``.

    A bridge that does not answer raises `ConnectionError`; one that refuses raises `BridgeError`.
    """

    def __init__(self, url: str, *, post: _Post | None = None) -> None:
        self.url = url.rstrip("/")
        self._post = post or _post
        self._timeout_s = Episode.timeout_s
        self._episode = ""

    def check(self, episode: Episode) -> None:
        """Raise now if the bridge is down or cannot run ``episode``."""
        self._call("check", asdict(episode), 60.0)

    def start(self, episode: Episode) -> None:
        """Begin ``episode`` on the bridge's Dream client; it then belongs to this session."""
        self._timeout_s = episode.timeout_s
        self._episode = self._call("start", asdict(episode), 60.0)["episode"]

    def observe(
        self, frames: Mapping[str, str], state: Vector, prompt: str, timestamp: float
    ) -> None:
        """Pass one tick through the bridge to Dream."""
        tick = {
            "frames": dict(frames),
            "state": state.tolist(),
            "prompt": prompt,
            "timestamp": timestamp,
        }
        self._call("observe", tick, 30.0)

    def act(self, commanded: Vector | None) -> Chunk:
        """The bridge's next chunk, waiting as long as the bridge may keep retrying Dream."""
        payload = {"commanded": None if commanded is None else commanded.tolist()}
        wait_s = self._timeout_s + REQUEST_TIMEOUT_S + 15.0
        return Chunk.from_json(self._call("act", payload, wait_s))

    def close(self) -> None:
        """Nothing to release here: the bridge owns the Dream session."""

    def _call(self, route: str, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        body = {**payload, "episode": self._episode} if route in ("observe", "act") else payload
        return self._post(f"{self.url}/{route}", body, timeout_s)


def _post(url: str, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read())
        except (OSError, ValueError):
            detail = {}
        message = f"{detail.get('error', error.code)}: {detail.get('message', error.reason)}"
        raise BridgeError(message) from None
    except (OSError, http.client.HTTPException) as error:
        reason = getattr(error, "reason", None) or error
        raise ConnectionError(f"no Dream bridge answering at {url}: {reason}") from error
    try:
        body: dict[str, Any] = json.loads(raw)
    except ValueError:
        raise BridgeError(f"{url} did not answer with JSON") from None
    return body


class BridgeServer(ThreadingHTTPServer):
    """One `Session` served over HTTP, one Dream call at a time."""

    def __init__(self, session: Session, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        super().__init__((host, port), _Handler)
        self.lock = threading.Lock()
        self.routes = _routes(session)


def _routes(session: Session) -> dict[str, _Route]:
    current = ""

    def check(body: dict[str, Any]) -> dict[str, Any]:
        session.check(Episode(**body))
        return {}

    def start(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal current
        session.start(Episode(**body))
        current = uuid.uuid4().hex
        return {"episode": current}

    def observe(body: dict[str, Any]) -> dict[str, Any]:
        _own(body, current)
        frames, prompt = body["frames"], body["prompt"]
        if not isinstance(frames, dict) or not isinstance(prompt, str):
            raise TypeError("observe takes a frames object and a prompt string")
        session.observe(frames, _station.vector(body["state"]), prompt, float(body["timestamp"]))
        return {}

    def act(body: dict[str, Any]) -> dict[str, Any]:
        _own(body, current)
        commanded = body["commanded"]
        return session.act(None if commanded is None else _station.vector(commanded)).to_json()

    return {"/check": check, "/start": start, "/observe": observe, "/act": act}


def _own(body: Mapping[str, Any], current: str) -> None:
    if not current or body.get("episode") != current:
        raise RuntimeError("this episode is not the one the bridge is running: start it again")


class _Handler(BaseHTTPRequestHandler):
    timeout = 30.0

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"ok": True})
        else:
            self._reply(404, {"error": "NotFound", "message": self.path})

    def do_POST(self) -> None:
        server = cast(BridgeServer, self.server)
        route = server.routes.get(self.path)
        if route is None:
            self._reply(404, {"error": "NotFound", "message": self.path})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise TypeError("the request body must be a JSON object")
            with server.lock:
                reply = route(body)
        except Exception as error:
            self._reply(500, {"error": type(error).__name__, "message": str(error)})
        else:
            self._reply(200, reply)

    def log_message(self, format: str, *args: Any) -> None:
        """Stay quiet: a line per tick would bury the retries the session reports."""

    def _reply(self, status: int, body: Mapping[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _warm(session: Session, model: str, timeout_s: float) -> Chunk:
    """Ask ``model`` for one chunk on a still scene, so a stopped model starts before a run."""
    black = _station.jpeg(np.zeros((_station.FRAME_HEIGHT, _station.FRAME_WIDTH, 3), np.uint8))
    home = np.tile([0.0] * _station.ARM_JOINTS + [1.0], 2)
    session.start(Episode(model=model, timeout_s=timeout_s))
    session.observe(dict.fromkeys(_station.CAMERAS, black), home, "hold still", 0.0)
    return session.act(None)


def main(argv: Sequence[str] | None = None) -> int:
    """``inspect-robots-dream-bridge``: serve a `DreamSession` until interrupted."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-dream-bridge",
        description="Serve the Dream client to the inspect-robots dream policy over HTTP.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--warm", action="store_true", help="start the model before serving")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the checkpoint --warm starts")
    parser.add_argument("--warm-timeout-s", type=float, default=3600.0)
    args = parser.parse_args(argv)

    session = DreamSession()
    with closing(session):
        if args.warm:
            print(f"Starting {args.model}; a stopped model can take many minutes.", flush=True)
            chunk = _warm(session, args.model, args.warm_timeout_s)
            print(f"{args.model} answered in {chunk.rtt_s:.1f} s.", flush=True)
        with BridgeServer(session, args.host, args.port) as server, suppress(KeyboardInterrupt):
            print(f"Dream bridge listening on http://{args.host}:{server.server_port}", flush=True)
            server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
