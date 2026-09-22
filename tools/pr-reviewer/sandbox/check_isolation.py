"""Fail-closed startup checks, executed inside each untrusted execution boundary."""

import json
import os
import socket
import sys
import urllib.request
from pathlib import Path


def check(workspace, host_pid_namespace, mode):
    """Verify namespace, network, filesystem and privilege separation."""
    workspace = Path(workspace)
    assert os.readlink("/proc/self/ns/pid") != host_pid_namespace, "host_processes_visible"
    status = Path("/proc/self/status").read_text()
    assert "NoNewPrivs:\t1" in status, "privilege_gain_possible"
    assert "CapEff:\t0000000000000000" in status, "capabilities_present"
    assert "REVIEW_MODEL_CAPABILITY" not in os.environ, "model_capability_in_tool_environment"
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            environment = (process / "environ").read_bytes()
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        assert b"REVIEW_MODEL_CAPABILITY=" not in environment, "client_environment_visible"
    assert os.getuid() == (65533 if mode == "build" else 65534)
    try:
        Path("/tmp/request.json").read_bytes()
    except (PermissionError, FileNotFoundError):
        pass
    else:
        raise AssertionError("checkpoint_request_readable")
    for host in ("127.0.0.1", "[::1]"):
        request = urllib.request.Request(
            f"http://{host}:3000/api/execute",
            data=b'{"command":"id","sessionId":"default"}',
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=1)
        except OSError:
            pass
        else:
            raise AssertionError("management_api_reachable")
    # Direct socket calls bypass language-level proxies and HTTP client settings.
    for family, address in (
        (socket.AF_INET, ("127.0.0.1", 3000)),
        (socket.AF_INET6, ("::1", 3000)),
    ):
        try:
            with socket.socket(family) as connection:
                connection.settimeout(1)
                connection.connect(address)
        except OSError:
            pass
        else:
            raise AssertionError("management_socket_reachable")
    for name in ("home/isolation-sentinel", "output/isolation-sentinel"):
        path = workspace / name
        for action in (path.read_bytes, lambda path=path: path.write_text("tampered")):
            try:
                action()
            except PermissionError:
                pass
            else:
                raise AssertionError("client_files_accessible:" + name)
    for name in ("head/isolation-evidence", "base/isolation-evidence", "context.json"):
        path = workspace / name
        if not path.exists():
            continue
        path.read_bytes()
        try:
            path.write_text("tampered")
        except OSError:
            pass
        else:
            raise AssertionError("canonical_evidence_writable")
    writable = workspace / ("build-home" if mode == "build" else "scratch") / "isolation-ok"
    writable.write_text("ok")
    writable.unlink()
    print(json.dumps({"isolated": True, "mode": mode, "uid": os.getuid()}))


if __name__ == "__main__":
    check(*sys.argv[1:])
