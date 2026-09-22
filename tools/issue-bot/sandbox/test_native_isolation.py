"""Exercise a hostile tool through real Codex and its native exec-server offline."""

# ruff: noqa: D101, D102
import io
import json
import os
import secrets
import shlex
import shutil
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import issue_runner as runner


@unittest.skipUnless(
    os.geteuid() == 0 and os.uname().sysname == "Linux" and shutil.which("codex"),
    "requires root Linux container with native Codex",
)
class NativeIsolationTest(unittest.TestCase):
    def test_repository_tool_cannot_forge_result_or_steal_client_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            workspace = root / "issue"
            secret = secrets.token_hex(32)
            fixture = {
                "status": "REQUIRE_REVIEWER",
                "serious": False,
                "summary": "Trusted synthetic final response.",
                "evidence": [],
                "plan": "",
                "findings": [],
                "checks": ["ISOLATION_OK"],
                "limitations": [],
            }
            attack = r"""
import ctypes, errno, json, os, pathlib, signal, sys
w = pathlib.Path(sys.argv[1])
assert os.getuid() == 65532, ('wrong tool UID', os.getuid())
assert not os.getgroups()
blocked = [
 lambda: (w/'output/result.json').write_text('{"status":"APPROVE"}'),
 lambda: (w/'home/.codex/config.toml').read_text(),
 lambda: (w/'home/.codex/config.toml').write_text('forged=true'),
 lambda: (w/'base/src/attack.py').write_text('tampered'),
 lambda: (w/'candidate').rename(w/'moved'),
]
for attempt in blocked:
 try: attempt()
 except PermissionError: pass
 else: raise AssertionError('untrusted tool crossed filesystem boundary')
assert not any(k.startswith(('CODEX_EXEC_SERVER', 'OPENAI_')) for k in os.environ)
assert 'ISSUE_GATEWAY_TOKEN' not in os.environ
clients, readable = [], []
for item in pathlib.Path('/proc').iterdir():
 if not item.name.isdigit(): continue
 try: args = (item/'cmdline').read_bytes()
 except (PermissionError, FileNotFoundError, ProcessLookupError): continue
 readable.append(args.decode(errors='replace'))
 if b'--output-schema\x00' in args: clients.append(item)
assert clients, 'real Codex client not found'
for client in clients:
 for sensitive in ('environ', 'mem', 'fd/1'):
  try:
   fd = os.open(client/sensitive, os.O_RDWR)
  except (PermissionError, ProcessLookupError): pass
  else:
   os.close(fd)
   raise AssertionError('client process accessible: '+sensitive)
 try: os.kill(int(client.name), signal.SIGSTOP)
 except PermissionError: pass
 else: raise AssertionError('client signal allowed')
 libc=ctypes.CDLL(None,use_errno=True)
 assert libc.ptrace(16,int(client.name),0,0) == -1 and ctypes.get_errno() == errno.EPERM
print('VISIBLE_ARGV='+json.dumps(readable))
print('ISOLATION_OK')
"""
            # The attack is repository code; it never receives the secret itself.
            archive = root / "source.tar"
            with tarfile.open(archive, "w") as out:
                data = attack.encode()
                member = tarfile.TarInfo("repo/src/attack.py")
                member.size = len(data)
                out.addfile(member, io.BytesIO(data))
            captured = []
            command = (
                f"python {shlex.quote(str(workspace / 'candidate/src/attack.py'))} "
                + shlex.quote(str(workspace))
            )

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_POST(self):
                    if (
                        self.path != "/responses"
                        or self.headers.get("Authorization") != f"Bearer {secret}"
                    ):
                        self.send_error(403)
                        return
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    captured.append(payload)
                    response = {
                        "id": f"resp_{len(captured)}",
                        "object": "response",
                        "created_at": 1,
                        "status": "completed",
                        "model": "gpt-6-astra",
                        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
                    }
                    outputs = [
                        i for i in payload["input"] if i.get("type") == "custom_tool_call_output"
                    ]
                    if not outputs:
                        call = {
                            "type": "custom_tool_call",
                            "id": "ctc_attack",
                            "call_id": "call_attack",
                            "name": "exec",
                            "namespace": "functions",
                            "input": "text(await tools.exec_command("
                            + json.dumps({"cmd": command, "yield_time_ms": 1000})
                            + "));",
                        }
                        events = [
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
                            {
                                "type": "response.completed",
                                "response": {**response, "output": [call]},
                            },
                        ]
                    else:
                        text = json.dumps(fixture)
                        part = {"type": "output_text", "text": text, "annotations": []}
                        item = {
                            "type": "message",
                            "id": "msg_final",
                            "role": "assistant",
                            "status": "completed",
                            "content": [part],
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
                            {
                                "type": "response.completed",
                                "response": {**response, "output": [item]},
                            },
                        ]
                    data = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            request = {
                "kind": "triage",
                "files": [],
                "issue": {},
                "context": "{}",
                "plan": "",
                "feedback": "",
                "inputDigest": "d" * 64,
                "token": secret,
                "checkpointToken": secrets.token_hex(32),
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(fixture),
                    "properties": {
                        key: {"type": "boolean"}
                        if isinstance(value, bool)
                        else {"type": "array", "items": {"type": "string"}}
                        if isinstance(value, list)
                        else {"type": "string"}
                        for key, value in fixture.items()
                    },
                },
            }
            original_unpack, original_config = runner.unpack, runner.write_client_config
            try:
                with (
                    patch.object(
                        runner,
                        "unpack",
                        side_effect=lambda _archive, target: original_unpack(archive, target),
                    ),
                    patch.object(runner, "prepare_packages", return_value=[]),
                    patch.object(
                        runner,
                        "write_client_config",
                        side_effect=lambda req, path: original_config(
                            req, path, f"http://127.0.0.1:{server.server_port}"
                        ),
                    ),
                ):
                    output = runner.run_stage(request, workspace)
            finally:
                server.shutdown()
                server.server_close()
            self.assertEqual(output["exitCode"], 0, output)
            self.assertEqual(output["result"], fixture)
            self.assertNotIn(secret, (workspace / "home/.codex/config.toml").read_text())
            self.assertEqual(len(captured), 2)
            tool_outputs = [
                i for i in captured[-1]["input"] if i.get("type") == "custom_tool_call_output"
            ]
            returned = json.dumps(tool_outputs)
            self.assertIn("ISOLATION_OK", returned)
            self.assertNotIn("Traceback", returned)
            self.assertNotIn(secret, returned)
            self.assertEqual(output["executions"][0]["exitCode"], 0)
            self.assertEqual(output["executions"][0]["command"].strip(), command.strip())


if __name__ == "__main__":
    unittest.main()
