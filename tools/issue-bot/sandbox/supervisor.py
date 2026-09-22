"""Fixed authenticated stage supervisor; no command, filesystem, or shell API."""

import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_REQUEST = 15_000_000
MAX_ARCHIVE = 10_000_000
MAX_OUTPUT = 1_500_000
SECRET_NAME = "ISSUE_CONTROL_SECRET"
STAGES = {"triage", "plan", "plan_review", "implement", "code_review"}
FIXED_COMMAND = ["/opt/issue-env/bin/python", "-I", "/opt/issue-runner.py"]
CHILD_ENV = {
    "PATH": "/opt/issue-env/bin:/opt/issue-js/node_modules/.bin:/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "UV_PYTHON_INSTALL_DIR": "/opt/python",
}


def failure(reason):
    """Return a capability-free terminal failure record."""
    return {
        "exitCode": 1,
        "failure": reason,
        "result": None,
        "files": [],
        "executions": [],
    }


def validate_payload(raw):
    """Validate the bounded fixed request before claiming any execution."""
    if len(raw) > MAX_REQUEST:
        raise ValueError("request_too_large")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"request", "archive"}:
        raise ValueError("invalid_request")
    request = data["request"]
    if not isinstance(request, dict):
        raise ValueError("invalid_request")
    if len(json.dumps(request).encode()) > 1_500_000:
        raise ValueError("request_too_large")
    for field in ("token", "checkpointToken", "inputDigest"):
        if not isinstance(request.get(field), str) or not re.fullmatch(
            r"[0-9a-f]{64}", request[field]
        ):
            raise ValueError("invalid_capability")
    if request["token"] == request["checkpointToken"]:
        raise ValueError("invalid_capability")
    if not re.fullmatch(r"[0-9a-f]{40}", str(request.get("base", ""))):
        raise ValueError("invalid_base")
    issue = request.get("issue")
    if (
        request.get("kind") not in STAGES
        or not isinstance(issue, dict)
        or issue.get("base") != request["base"]
        or type(issue.get("number")) is not int
        or issue["number"] < 1
        or not isinstance(request.get("schema"), dict)
        or not isinstance(request.get("files"), list)
        or len(request["files"]) > 80
    ):
        raise ValueError("invalid_request")
    archive_text = data["archive"]
    if not isinstance(archive_text, str) or len(archive_text) > 13_333_336:
        raise ValueError("archive_too_large")
    archive = base64.b64decode(archive_text, validate=True)
    if len(archive) > MAX_ARCHIVE or not archive.startswith(b"\x1f\x8b"):
        raise ValueError("invalid_archive")
    return request, archive


def exclusive_write(path, value):
    """Create a private file without following or replacing existing paths."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())


class Supervisor:
    """One root-owned claim per container, persisted before the fixed subprocess."""

    def __init__(self, root=Path("/run/issue-control"), staging=Path("/tmp")):
        self.root = root
        self.staging = staging
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.lock = threading.Lock()
        self.process = None
        self.output = None
        self.claim = self.root / "claim.json"
        if self.claim.exists():
            # A process-level restart cannot prove whether prior inference ran.
            # Never restart it or discard its reserved spending.
            self.output = failure("stage_launch_uncertain")

    def start(self, raw):
        """Persist one claim and launch only the fixed trusted runner."""
        request, archive = validate_payload(raw)
        identity = hashlib.sha256(raw).hexdigest()
        with self.lock:
            if self.claim.exists():
                saved = json.loads(self.claim.read_text())
                if saved["digest"] != identity:
                    raise ValueError("stage_request_changed")
                return {"accepted": True}
            exclusive_write(self.claim, json.dumps({"digest": identity}).encode())
            try:
                exclusive_write(self.staging / "issue-request.json", json.dumps(request).encode())
                exclusive_write(self.staging / "issue-base.tar.gz", archive)
                log = self.root / "output.json"
                descriptor = os.open(
                    log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                )
                with os.fdopen(descriptor, "wb") as target:
                    self.process = subprocess.Popen(
                        FIXED_COMMAND,
                        stdin=subprocess.DEVNULL,
                        stdout=target,
                        stderr=subprocess.DEVNULL,
                        env=CHILD_ENV,
                        start_new_session=True,
                    )
            except (OSError, ValueError):
                self.output = failure("stage_launch_failed")
                return {"accepted": True}
            threading.Thread(target=self._finish, daemon=True).start()
            return {"accepted": True}

    def _finish(self):
        try:
            code = self.process.wait(timeout=41 * 60)
            if code != 0:
                output = failure("stage_execution_failed")
            else:
                path = self.root / "output.json"
                if path.stat().st_size > MAX_OUTPUT:
                    raise ValueError("output_too_large")
                output = json.loads(path.read_bytes())
                if not isinstance(output, dict) or set(output) != {
                    "exitCode",
                    "failure",
                    "result",
                    "files",
                    "executions",
                }:
                    raise ValueError("invalid_output")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                self.process.kill()
            output = failure("stage_supervisor_failed")
        with self.lock:
            self.output = output

    def status(self):
        """Return bounded trusted stage state without exposing file access."""
        with self.lock:
            if self.output is not None:
                return {"state": "complete", "output": self.output}
            return {"state": "running" if self.claim.exists() else "idle"}


def handler(supervisor, secret):
    """Build the two-route authenticated HTTP handler."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Never log control headers, capabilities, code, or issue content.

        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, f"Bearer {secret}")

        def do_GET(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/status":
                return self.respond(404, {"error": "not_found"})
            self.respond(200, supervisor.status())

        def do_POST(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/start":
                return self.respond(404, {"error": "not_found"})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid_length")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST:
                    return self.respond(413, {"error": "invalid_length"})
                self.connection.settimeout(15)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete_body")
                result = supervisor.start(raw)
            except (ValueError, TypeError, OSError):
                return self.respond(400, {"error": "invalid_start"})
            self.respond(202, result)

    return Handler


def main():
    """Start the supervisor with a root-only controller capability."""
    secret = os.environ.pop(SECRET_NAME, "")
    if not re.fullmatch(r"[0-9a-f]{64}", secret) or os.geteuid() != 0:
        raise SystemExit("supervisor_configuration_invalid")
    os.umask(0o077)
    server = ThreadingHTTPServer(("0.0.0.0", 8080), handler(Supervisor(), secret))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
