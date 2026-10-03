"""Offline defensive contract tests for the fixed stage supervisor."""

# Test names state each invariant without repeated method docstrings.
# ruff: noqa: D101, D102

import base64
import io
import json
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import supervisor


def payload():
    """Build the smallest valid fixed supervisor request."""
    return json.dumps(
        {
            "request": {
                "id": "stage-1",
                "jobId": "job-1",
                "kind": "triage",
                "base": "a" * 40,
                "token": "b" * 64,
                "checkpointToken": "c" * 64,
                "inputDigest": "d" * 64,
                "issue": {"base": "a" * 40, "number": 401},
                "schema": {},
                "files": [],
            },
            "archive": base64.b64encode(b"\x1f\x8btest-fixture").decode(),
        }
    ).encode()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.root = base / "control"
        self.staging = base / "tmp"
        self.staging.mkdir()
        self.supervisor = supervisor.Supervisor(self.root, self.staging)

    def request(self, method, path, body=b"", auth=None):
        cls = supervisor.handler(self.supervisor, "a" * 64)
        endpoint = cls.__new__(cls)
        endpoint.path = path
        endpoint.headers = Message()
        endpoint.headers["Content-Length"] = str(len(body))
        if auth is not None:
            endpoint.headers["Authorization"] = auth
        endpoint.rfile = io.BytesIO(body)
        endpoint.connection = MagicMock()
        endpoint.respond = MagicMock()
        getattr(endpoint, f"do_{method}")()
        return endpoint.respond.call_args.args

    def test_unauthenticated_local_requests_cannot_launch_or_read_output(self):
        with patch.object(self.supervisor, "start") as start:
            self.assertEqual(self.request("POST", "/start", payload())[0], 401)
            self.assertEqual(self.request("GET", "/status")[0], 401)
            self.assertEqual(self.request("POST", "/start", payload(), "Bearer wrong")[0], 401)
            start.assert_not_called()
        self.assertFalse(self.supervisor.claim.exists())

    def test_no_generic_command_or_file_routes(self):
        for path in ("/execute", "/exec", "/files", "/terminal", "/start?exec=id"):
            self.assertEqual(
                self.request("POST", path, b'{"command":"id"}', "Bearer " + "a" * 64)[0],
                404,
            )

    @patch.object(supervisor.threading, "Thread")
    @patch.object(supervisor.subprocess, "Popen")
    def test_authenticated_start_is_one_shot_and_has_fixed_clean_environment(self, popen, _thread):
        with patch.dict(os.environ, {"ISSUE_CONTROL_SECRET": "never-forward"}):
            self.assertEqual(
                self.request("POST", "/start", payload(), "Bearer " + "a" * 64)[0], 202
            )
            self.supervisor.start(payload())
        popen.assert_called_once()
        args, kwargs = popen.call_args
        self.assertEqual(args[0], supervisor.FIXED_COMMAND)
        self.assertNotIn("ISSUE_CONTROL_SECRET", kwargs["env"])
        self.assertNotIn("shell", kwargs)
        self.assertEqual(kwargs["env"], supervisor.CHILD_ENV)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        for file in (
            self.supervisor.claim,
            self.staging / "issue-request.json",
            self.staging / "issue-base.tar.gz",
            self.root / "output.json",
        ):
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
        changed = json.loads(payload())
        changed["request"]["kind"] = "plan"
        with self.assertRaisesRegex(ValueError, "stage_request_changed"):
            self.supervisor.start(json.dumps(changed).encode())
        popen.assert_called_once()

    @patch.object(supervisor.threading, "Thread")
    @patch.object(supervisor.subprocess, "Popen")
    def test_lost_ack_and_supervisor_restart_never_repeat_execution(self, popen, _thread):
        self.supervisor.start(payload())
        replacement = supervisor.Supervisor(self.root, self.staging)
        replacement.start(payload())
        self.assertEqual(replacement.status()["output"]["failure"], "stage_launch_uncertain")
        popen.assert_called_once()

    @patch.object(supervisor.threading, "Thread")
    @patch.object(supervisor.subprocess, "Popen")
    def test_status_only_returns_bounded_trusted_launcher_output(self, popen, _thread):
        self.supervisor.start(payload())
        expected = supervisor.failure("fixture_error")
        (self.root / "output.json").write_text(json.dumps(expected))
        popen.return_value.wait.return_value = 0
        self.supervisor._finish()
        self.assertEqual(self.supervisor.status(), {"state": "complete", "output": expected})
        (self.root / "output.json").write_text("x" * (supervisor.MAX_OUTPUT + 1))
        self.supervisor._finish()
        self.assertEqual(self.supervisor.status()["output"]["failure"], "stage_supervisor_failed")

    def test_invalid_or_oversized_payload_never_claims_execution(self):
        invalid = json.loads(payload())
        invalid["request"]["checkpointToken"] = invalid["request"]["token"]
        with self.assertRaises(ValueError):
            self.supervisor.start(json.dumps(invalid).encode())
        invalid = json.loads(payload())
        invalid["archive"] = "not base64"
        with self.assertRaises(ValueError):
            self.supervisor.start(json.dumps(invalid).encode())
        invalid = json.loads(payload())
        invalid["command"] = "id"
        with self.assertRaises(ValueError):
            self.supervisor.start(json.dumps(invalid).encode())
        self.assertFalse(self.supervisor.claim.exists())

    def test_docker_uses_only_fixed_supervisor_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text()
        self.assertIn(
            'ENTRYPOINT ["/opt/issue-env/bin/python", "-I", "/opt/issue-supervisor.py"]', dockerfile
        )
        self.assertIn("CMD []", dockerfile)
        self.assertIn("!sandbox/supervisor.py", (root / ".dockerignore").read_text())


if __name__ == "__main__":
    unittest.main()
