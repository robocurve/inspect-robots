"""Serve a phone-friendly view of both Actuate overnight campaign rigs.

Run with:  python examples/actuate/overnight.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import run
from _roster import ACCENTS

TRIALS_PER_RIG = 5
STALL_AFTER_MIN = 30
RIGS = {"rig1": ("state-rig1", "logs-rig1"), "rig2": ("state-rig2", "logs-rig2")}

HERE = Path(__file__).parent
_MEDIA_NAME = re.compile(r"^[A-Za-z0-9._-]+\.mp4$")
_STATUS_KEYS = ("eval_index", "started_at", "roles", "models", "cooling")


def _eligible_names() -> list[str]:
    try:
        return [name for name, _model in run._eligible("test_taker")]
    except Exception:
        return []


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(now: datetime, then: datetime | None) -> float | None:
    if then is None:
        return None
    return max(0.0, (now - then).total_seconds())


def _read_status(
    path: Path, now: datetime
) -> tuple[dict[str, Any] | None, float | None, datetime | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None, None, None
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except Exception:
        return None, None, None
    status = {key: raw.get(key) for key in _STATUS_KEYS}
    return status, _age_seconds(now, modified_at), modified_at


def _numeric_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _recent_result(result: Any) -> dict[str, Any] | None:
    try:
        if not isinstance(result, dict):
            return None
        roles = result["roles"]
        if not isinstance(roles, dict):
            return None
        raw_score = result.get("score")
        score = _numeric_score(raw_score)
        if raw_score is not None and score is None:
            return None
        eval_index = result.get("eval_index")
        if not isinstance(eval_index, int) or isinstance(eval_index, bool) or eval_index < 1:
            eval_index = None
        test_taker = roles.get("test_taker")
        task = result.get("task")
        ts = result.get("ts")
        clips = result.get("clips")
        return {
            "eval_index": eval_index,
            "test_taker": test_taker if isinstance(test_taker, str) else None,
            "score": score,
            "task": task if isinstance(task, str) else None,
            "ts": ts if isinstance(ts, str) else None,
            "clips": clips if isinstance(clips, dict) else {},
        }
    except Exception:
        return None


def _read_results(
    path: Path, roster: list[str]
) -> tuple[dict[str, list[float]], list[dict[str, Any]], datetime | None]:
    observations: dict[str, list[float]] = {name: [] for name in roster}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    parsed_rows: list[dict[str, Any]] = []
    last_result_at: datetime | None = None
    for line in lines:
        try:
            result = json.loads(line)
        except Exception:
            continue
        if not isinstance(result, dict):
            continue

        parsed_ts = _parse_iso(result.get("ts"))
        if parsed_ts is not None:
            last_result_at = parsed_ts

        try:
            name = result["roles"]["test_taker"]
            raw_score = result.get("score")
            score = _numeric_score(raw_score)
            if name in observations and raw_score is not None and score is not None:
                observations[name].append(score)
        except Exception:
            pass

        recent = _recent_result(result)
        if recent is not None:
            parsed_rows.append(recent)

    return observations, list(reversed(parsed_rows[-3:])), last_result_at


def _summaries(observations: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "n": len(scores),
            "mean": sum(scores) / len(scores) if scores else None,
        }
        for name, scores in observations.items()
    }


def _empty_rig_payload(roster: list[str]) -> dict[str, Any]:
    observations = {name: [] for name in roster}
    return {
        "status": None,
        "status_age_s": None,
        "results": _summaries(observations),
        "recent": [],
        "last_activity_s": None,
        "done": False,
        "stalled": False,
    }


def _rig_payload(rig: str, roster: list[str], now: datetime) -> dict[str, Any]:
    try:
        state_name, _logs_name = RIGS[rig]
        state_dir = HERE / state_name
        status, status_age_s, status_at = _read_status(state_dir / "status.json", now)
        observations, recent, result_at = _read_results(state_dir / "results.jsonl", roster)
        activity_at = max(
            (value for value in (status_at, result_at) if value is not None),
            default=None,
        )
        last_activity_s = _age_seconds(now, activity_at)
        done = bool(roster) and all(len(observations[name]) >= TRIALS_PER_RIG for name in roster)
        stalled = (
            not done and last_activity_s is not None and last_activity_s > STALL_AFTER_MIN * 60
        )
        return {
            "status": status,
            "status_age_s": status_age_s,
            "results": _summaries(observations),
            "recent": recent,
            "last_activity_s": last_activity_s,
            "done": done,
            "stalled": stalled,
        }
    except Exception:
        return _empty_rig_payload(roster)


def _overnight_payload() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    roster = _eligible_names()
    rigs = {rig: _rig_payload(rig, roster, now) for rig in RIGS}
    pooled_observations: dict[str, list[float]] = {name: [] for name in roster}
    for rig in RIGS:
        state_name, _logs_name = RIGS[rig]
        observations, _recent, _last_result_at = _read_results(
            HERE / state_name / "results.jsonl", roster
        )
        for name in roster:
            pooled_observations[name].extend(observations[name])
    pooled = _summaries(pooled_observations)
    return {
        "generated_at": now.isoformat(),
        "roster": roster,
        "accents": ACCENTS,
        "rigs": rigs,
        "pooled": {
            "results": pooled,
            "total_scored": sum(entry["n"] for entry in pooled.values()),
            "target": 2 * TRIALS_PER_RIG * len(roster),
        },
    }


def _empty_payload() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    roster = _eligible_names()
    empty_rigs = {rig: _empty_rig_payload(roster) for rig in RIGS}
    pooled = _summaries({name: [] for name in roster})
    return {
        "generated_at": now.isoformat(),
        "roster": roster,
        "accents": ACCENTS,
        "rigs": empty_rigs,
        "pooled": {
            "results": pooled,
            "total_scored": 0,
            "target": 2 * TRIALS_PER_RIG * len(roster),
        },
    }


class OvernightHandler(BaseHTTPRequestHandler):
    """Serve the overnight page, aggregate API, and per-rig video clips."""

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any) -> None:
        try:
            body = json.dumps(payload).encode("utf-8")
        except Exception:
            body = json.dumps(_empty_payload()).encode("utf-8")
        self._send(body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        """Route one GET request without retaining response state."""
        path = urlsplit(self.path).path
        if path == "/api/overnight":
            try:
                payload = _overnight_payload()
            except Exception:
                payload = _empty_payload()
            self._send_json(payload)
            return
        if path.startswith("/media/"):
            parts = path.split("/")
            if len(parts) != 4 or parts[2] not in RIGS or not _MEDIA_NAME.fullmatch(parts[3]):
                self._send(b"Not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return
            state_name, _logs_name = RIGS[parts[2]]
            try:
                body = (HERE / state_name / "media" / parts[3]).read_bytes()
            except OSError:
                self._send(b"Not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return
            self._send(body, "video/mp4")
            return
        if path == "/":
            try:
                body = (HERE / "overnight.html").read_bytes()
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
        """Suppress the base server's per-request console logging."""
        return


def main() -> None:
    """Serve the overnight campaign view on the selected port."""
    parser = argparse.ArgumentParser(description="Serve the Actuate overnight campaign view.")
    parser.add_argument("--port", type=int, default=8380)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("", args.port), OvernightHandler)
    print(f"Actuate overnight view available at http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
