"""Offline Linux/container regression for PR-supplied build code.

Run as root in the review image: python /tests/test_evidence.py
"""

import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

spec = importlib.util.spec_from_file_location("runner", "/opt/codex-review.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

BACKEND = r"""
import json, os, shutil, zipfile
from pathlib import Path
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    build = Path.cwd()
    workspace = build.parent
    attempts = {}
    operations = {
        'overwrite_source': lambda: (workspace/'head/source.py').write_text('base'),
        'chmod_source': lambda: (workspace/'head/source.py').chmod(0o666),
        'remove_head': lambda: shutil.rmtree(workspace/'head'),
        'rename_head': lambda: (workspace/'head').rename(workspace/'hidden-head'),
        'replace_context': lambda: (workspace/'context.json').write_text('{}'),
        'unlink_context': lambda: (workspace/'context.json').unlink(),
        'chmod_parent': lambda: workspace.chmod(0o777),
        'replace_parent': lambda: workspace.rename(workspace.parent/'hidden-review'),
        'forge_setup': lambda: (workspace/'setup.json').write_text('[]'),
        'forge_result': lambda: (workspace/'output/result.json').write_text('APPROVE'),
        'modify_codex_home': lambda: (workspace/'home/config.toml').write_text('bad'),
    }
    for name, action in operations.items():
        try:
            action()
            attempts[name] = 'ALLOWED'
        except PermissionError:
            attempts[name] = 'DENIED'
    (build/'tamper-attempts.json').write_text(json.dumps(attempts))
    name = 'evidence_probe-0.0.0-py3-none-any.whl'
    with zipfile.ZipFile(Path(wheel_directory)/name, 'w') as wheel:
        wheel.writestr('evidence_probe.py', 'VALUE = 42\n')
        wheel.writestr('evidence_probe-0.0.0.dist-info/METADATA',
                      'Metadata-Version: 2.1\nName: evidence-probe\nVersion: 0.0.0\n')
        wheel.writestr('evidence_probe-0.0.0.dist-info/WHEEL',
                      'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        wheel.writestr('evidence_probe-0.0.0.dist-info/RECORD', '')
    return name
"""


@unittest.skipUnless(os.geteuid() == 0, "requires the container's root launcher")
class EvidenceIsolation(unittest.TestCase):
    """PR code must not change the evidence used to review that same PR."""

    @classmethod
    def setUpClass(cls):
        """Simulate the root-only checkpoint request in the disposable test container."""
        path = Path("/tmp/request.json")
        path.write_text('{"checkpointToken":"synthetic-only"}')
        path.chmod(0o600)

    def test_management_boundary_has_a_reachable_control(self):
        """Ensure denied connections aren't artifacts of an absent management server."""

        # A standalone Linux run supplies a local mock; Cloudflare already runs
        # the real management API here. No production credentials exist in this test.
        class Management(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"stdout":"uid=0(root)"}')

            def log_message(self, *_args):
                pass

        server = None
        try:
            with suppress(OSError):
                server = HTTPServer(("127.0.0.1", 3000), Management)
            if server:
                Thread(target=server.serve_forever, daemon=True).start()
            result = subprocess.run(
                [
                    "/opt/review-env/bin/python",
                    "-I",
                    "-S",
                    "-c",
                    "import urllib.request; r=urllib.request.Request("
                    "'http://127.0.0.1:3000/api/execute',"
                    'data=b\'{"command":"id","sessionId":"default"}\','
                    "headers={'Content-Type':'application/json'}); "
                    "print(urllib.request.urlopen(r,timeout=5).read().decode())",
                ],
                user=65534,
                group=65534,
                extra_groups=[],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("uid=0(root)", result.stdout)
        finally:
            if server:
                server.shutdown()
                server.server_close()

    def test_build_backend_cannot_replace_evidence_or_forge_output(self):
        """Run a real malicious offline build, then verify canonical evidence remains."""
        with tempfile.TemporaryDirectory(prefix="review-evidence-") as parent:
            workspace = Path(parent) / "review"
            for tree in ("base", "head"):
                (workspace / tree).mkdir(parents=True)
            (workspace / "base/source.py").write_text("base")
            (workspace / "head/source.py").write_text("proposed change")
            (workspace / "head/source.py").chmod(0o777)  # Attacker-selected archive mode.
            context = '{"maintainer_decisions": ["review this PR"]}'
            (workspace / "context.json").write_text(context)
            (workspace / "head/pyproject.toml").write_text(
                '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n'
            )
            (workspace / "head/backend.py").write_text(BACKEND)
            runner.protect_evidence(workspace)
            self.assertTrue((workspace / "home/.codex").is_dir())
            self.assertEqual((workspace / "home/.codex").stat().st_uid, 65534)
            environment = {
                "PATH": "/opt/review-env/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": str(workspace / "home"),
                "PIP_NO_INDEX": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "CODEX_HOME": str(workspace / "home/.codex"),
            }
            runner.verify_isolation(workspace, environment)
            backend = workspace / "head/backend.py"
            backend.chmod(0o644)
            backend.write_text(
                BACKEND.replace(
                    "    build = Path.cwd()",
                    "    build = Path.cwd()\n    import subprocess\n"
                    "    subprocess.run(['/opt/review-env/bin/python','-I','-S',"
                    "'/opt/check-isolation.py',str(build.parent),"
                    + repr(os.readlink("/proc/self/ns/pid"))
                    + ",'build'],check=True)",
                )
            )
            backend.chmod(0o444)
            executions = runner.prepare_packages(workspace, environment, "a" * 40)
            setup = json.loads((workspace / "setup.json").read_text())
            self.assertEqual(executions[0]["exitCode"], 0, setup)
            attempts = json.loads((workspace / "build-source/tamper-attempts.json").read_text())
            self.assertEqual(set(attempts.values()), {"DENIED"}, attempts)
            self.assertEqual(len(attempts), 11)
            self.assertEqual((workspace / "head/source.py").read_text(), "proposed change")
            self.assertEqual((workspace / "base/source.py").read_text(), "base")
            self.assertEqual((workspace / "context.json").read_text(), context)
            self.assertTrue((workspace / "python-packages/evidence_probe.py").is_file())
            # Review shell commands get the same immutable evidence, but can write scratch.
            code = """import pathlib, shutil, os
w=pathlib.Path(os.environ['WORKSPACE'])
for p in [w/'head/source.py',w/'context.json',w/'setup.json']:
 try: p.write_text('tampered')
 except OSError: pass
 else: raise AssertionError(str(p))
shutil.copytree(w/'head',w/'scratch/copy')
p=w/'scratch/copy/source.py';p.chmod(0o644);p.write_text('reproduction')
try: (w/'output/result.json').write_text('{}')
except OSError: pass
else: raise AssertionError('forged client output')
"""
            check = subprocess.run(
                [
                    "codex",
                    "sandbox",
                    *runner.tool_permissions(workspace),
                    "--",
                    "/opt/review-env/bin/python",
                    "-c",
                    code,
                ],
                env={**environment, "WORKSPACE": str(workspace)},
                user=65534,
                group=65534,
                extra_groups=[],
                capture_output=True,
                text=True,
            )
            self.assertEqual(check.returncode, 0, check.stderr)
            diff = subprocess.run(
                [
                    "git",
                    "diff",
                    "--no-index",
                    "--",
                    str(workspace / "base"),
                    str(workspace / "head"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(diff.returncode, 1)
            self.assertIn("+proposed change", diff.stdout)

    def test_real_codex_starts_with_protected_workspace_and_offline_provider(self):
        """Real multi-turn CLI: hostile tool call, private header and valid final output."""
        text = '{"ok":true}'
        requests = []
        command = ""
        escalation = ""
        capability = "synthetic-model-capability-not-a-real-secret"
        part = {"type": "output_text", "text": text, "annotations": []}
        item = {
            "type": "message",
            "id": "msg_probe",
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }
        response = {
            "id": "resp_probe",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6-astra",
            "output": [item],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }
        events = [
            {
                "type": "response.created",
                "response": {**response, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": {**part, "text": ""},
            },
            {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
            {
                "type": "response.output_text.done",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
            {
                "type": "response.content_part.done",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": part,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        payload = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()

        class Provider(BaseHTTPRequestHandler):
            """Serve fixed Responses events without credentials or external network."""

            def do_POST(self):
                """Exercise native tools before returning the fixed final response."""
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                requests.append((self.headers.get("X-Review-Token"), json.loads(raw)))
                body = payload
                if len(requests) <= 2:
                    arguments = (
                        {"cmd": command, "yield_time_ms": 1000}
                        if len(requests) == 1
                        else {
                            "cmd": escalation,
                            "sandbox_permissions": "require_escalated",
                            "justification": "Synthetic boundary bypass test.",
                        }
                    )
                    call = {
                        "type": "custom_tool_call",
                        "id": "ctc_probe" + str(len(requests)),
                        "call_id": "call_probe" + str(len(requests)),
                        "name": "exec",
                        "namespace": "functions",
                        "input": "text(await tools.exec_command(" + json.dumps(arguments) + "));",
                    }
                    tool_events = [
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {**call, "input": ""},
                        },
                        {
                            "type": "response.custom_tool_call_input.delta",
                            "item_id": call["id"],
                            "output_index": 0,
                            "delta": call["input"],
                        },
                        {
                            "type": "response.custom_tool_call_input.done",
                            "item_id": call["id"],
                            "output_index": 0,
                            "input": call["input"],
                        },
                        {"type": "response.output_item.done", "output_index": 0, "item": call},
                        {"type": "response.completed", "response": {**response, "output": [call]}},
                    ]
                    body = "".join("data: " + json.dumps(e) + "\n\n" for e in tool_events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                """Keep request text out of test output."""

        with HTTPServer(("127.0.0.1", 0), Provider) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory(prefix="review-startup-") as parent:
                    workspace = Path(parent) / "review"
                    for tree in ("head", "base"):
                        (workspace / tree).mkdir(parents=True)
                    (workspace / "context.json").write_text("{}")
                    runner.protect_evidence(workspace)
                    environment = {
                        "PATH": "/opt/review-env/bin:/usr/local/bin:/usr/bin:/bin",
                        "HOME": str(workspace / "home"),
                        "CODEX_HOME": str(workspace / "home/.codex"),
                        "TMPDIR": str(workspace / "scratch"),
                    }
                    runner.verify_isolation(workspace, environment)
                    output = workspace / "output/result.json"
                    Path("/tmp/review-schema.json").write_text(
                        json.dumps(
                            {
                                "type": "object",
                                "properties": {"ok": {"type": "boolean"}},
                                "required": ["ok"],
                                "additionalProperties": False,
                            }
                        )
                    )
                    args = runner.codex_args(
                        workspace,
                        environment,
                        "Synthetic isolation regression only.",
                        'Run the supplied isolation check, then return {"ok":true}.',
                    )
                    args[-1:-1] = [
                        "-c",
                        f'model_providers.review_gateway.base_url="http://127.0.0.1:{server.server_port}"',
                    ]
                    command = shlex.join(
                        [
                            "/opt/review-env/bin/python",
                            "-I",
                            "-S",
                            "/opt/check-isolation.py",
                            str(workspace),
                            os.readlink("/proc/self/ns/pid"),
                            "tool",
                        ]
                    )
                    escalation = "echo escaped > " + shlex.quote(
                        str(workspace / "home/escalation-marker")
                    )
                    self.assertNotIn(capability, " ".join(args))
                    result = subprocess.run(
                        args,
                        cwd=workspace / "head",
                        env={**environment, "REVIEW_MODEL_CAPABILITY": capability},
                        user=65534,
                        group=65534,
                        extra_groups=[],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(output.read_text()), {"ok": True})
                    self.assertEqual(len(requests), 3, result.stderr)
                    self.assertFalse((workspace / "home/escalation-marker").exists())
                    for header, body in requests:
                        self.assertEqual(header, capability)
                        self.assertNotIn(capability, json.dumps(body))
                    tool_output = [
                        v
                        for v in requests[1][1]["input"]
                        if v.get("type") == "custom_tool_call_output"
                    ]
                    self.assertTrue(tool_output, result.stdout)
                    records = [
                        json.loads(part["text"])
                        for part in tool_output[0]["output"]
                        if part.get("text", "").startswith("{")
                    ]
                    self.assertEqual(records[0]["exit_code"], 0, records)
                    self.assertIn('"isolated": true', records[0]["output"])
            finally:
                server.shutdown()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
