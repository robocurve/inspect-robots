"""Serve the Actuate conference monitor and leaderboard.

Run with:  python examples/actuate/serve.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from _roster import ROSTER

HERE = Path(__file__).parent
STATE_DIR = HERE / "state"
LOGS_DIR = HERE / "logs"
STATUS_PATH = STATE_DIR / "status.json"
RESULTS_PATH = STATE_DIR / "results.jsonl"
MEDIA_DIR = STATE_DIR / "media"
_MEDIA_NAME = re.compile(r"^[A-Za-z0-9._-]+\.mp4$")


def _status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "roles": None,
        "models": None,
        "eval_index": None,
        "started_at": None,
        "task": None,
        "rubric": None,
        "cooling": None,
    }
    drawn_at: float | None = None
    with suppress(Exception):
        with STATUS_PATH.open(encoding="utf-8") as status_file:
            status = json.load(status_file)
        for key in ("roles", "models", "eval_index", "started_at", "cooling"):
            payload[key] = status.get(key)
        drawn_at = STATUS_PATH.stat().st_mtime

    with suppress(Exception):
        newest_log = max(LOGS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime)
        # A log older than the current draw belongs to a previous eval; keep
        # the "generating" state rather than showing a stale task.
        if drawn_at is not None and newest_log.stat().st_mtime < drawn_at:
            return payload
        with newest_log.open(encoding="utf-8") as log_file:
            log = json.load(log_file)
        sample = log["samples"][0]
        payload["task"] = sample.get("instruction")
        metadata = sample.get("scene_metadata") or {}
        payload["rubric"] = metadata.get("rubric")
    return payload


def _leaderboard_payload() -> list[dict[str, Any]]:
    observations: dict[str, list[float]] = {name: [] for name in ROSTER}
    try:
        lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    for line in lines:
        try:
            result = json.loads(line)
            name = result["roles"]["test_taker"]
            score = result.get("score")
            numeric_score = float(score) if score is not None else None
            if name in observations and numeric_score is not None and math.isfinite(numeric_score):
                observations[name].append(numeric_score)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return [
        {
            "name": name,
            "n": len(scores),
            "score": sum(scores) / len(scores) if scores else None,
        }
        for name, scores in observations.items()
    ]


def _recent_payload() -> list[dict[str, Any]]:
    try:
        lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    recent: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            result = json.loads(line)
            roles = result["roles"]
            models = result["models"]
            if not isinstance(roles, dict) or not isinstance(models, dict):
                continue
            score = result.get("score")
            numeric_score = float(score) if score is not None else None
            if numeric_score is not None and not math.isfinite(numeric_score):
                continue
            task = result.get("task")
            clips = result.get("clips")
            eval_index = result.get("eval_index")
            if not isinstance(eval_index, int) or isinstance(eval_index, bool) or eval_index < 1:
                eval_index = None
            recent.append(
                {
                    "eval_index": eval_index,
                    "roles": roles,
                    "models": models,
                    "task": task if isinstance(task, str) else None,
                    "score": numeric_score,
                    "clips": clips if isinstance(clips, dict) else {},
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(recent) == 3:
            break
    return recent


class DemoHandler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._send_json(_status_payload())
            return
        if path == "/api/leaderboard":
            self._send_json(_leaderboard_payload())
            return
        if path == "/api/recent":
            self._send_json(_recent_payload())
            return
        if path.startswith("/media/"):
            name = path[len("/media/") :]
            if not _MEDIA_NAME.fullmatch(name):
                self._send(b"Not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return
            try:
                body = (MEDIA_DIR / name).read_bytes()
            except OSError:
                self._send(b"Not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return
            self._send(body, "video/mp4")
            return
        if path in {"/", "/leaderboard"}:
            try:
                body = (HERE / "monitor.html").read_bytes()
            except OSError:
                self._send(
                    b"Page unavailable\n",
                    "text/plain; charset=utf-8",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send(body, "text/html; charset=utf-8")
            return
        self._send(b"Not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Actuate conference display.")
    parser.add_argument("--port", type=int, default=8377)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("", args.port), DemoHandler)
    print(f"Actuate display available at http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
