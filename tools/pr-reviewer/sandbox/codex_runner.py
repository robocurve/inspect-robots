"""Run the real Codex CLI in a disposable sandbox with no provider credentials."""

import json
import os
import selectors
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path


def tool_permissions(workspace):
    """Native Codex mount/PID/network sandbox; no unsandboxed fallback."""
    paths = {
        "/": "read",
        str(workspace / "scratch"): "write",
        str(workspace / "home"): "deny",
        str(workspace / "output"): "deny",
        "/tmp/request.json": "deny",
        "/tmp/review-output.json": "deny",
    }
    filesystem = "{" + ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in paths.items()) + "}"
    return [
        "-c",
        'default_permissions="review"',
        "-c",
        "features.use_linux_sandbox_bwrap=true",
        "-c",
        "permissions.review.filesystem=" + filesystem,
        "-c",
        "permissions.review.network.enabled=false",
        "-c",
        'approval_policy="never"',
    ]


def isolated_build(command):
    """New network and PID namespaces; drop all privilege before any PR code."""
    return [
        "/usr/bin/unshare",
        "--net",
        "--pid",
        "--fork",
        "--mount-proc",
        "--mount",
        "--ipc",
        "--uts",
        "--kill-child=KILL",
        "--",
        "/usr/bin/setpriv",
        "--reuid=65533",
        "--regid=65533",
        "--clear-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--",
        *command,
    ]


def verify_isolation(workspace, environment):
    """Abort before builds or paid inference if either boundary is unavailable."""
    for name in ("home/isolation-sentinel", "output/isolation-sentinel"):
        sentinel = workspace / name
        sentinel.write_text("client-only")
        os.chown(sentinel, 65534, 65534)
        sentinel.chmod(0o600)
    check = [
        "/opt/review-env/bin/python",
        "-I",
        "-S",
        "/opt/check-isolation.py",
        str(workspace),
        os.readlink("/proc/self/ns/pid"),
    ]
    subprocess.run(
        ["codex", "sandbox", *tool_permissions(workspace), "--", *check, "tool"],
        cwd=workspace,
        env=environment,
        user=65534,
        group=65534,
        extra_groups=[],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        isolated_build([*check, "build"]),
        cwd=workspace,
        env=environment,
        check=True,
        capture_output=True,
        timeout=30,
    )


def protect_evidence(workspace):
    """Keep snapshots and evidence immutable, including their parent directories.

    Only explicitly designated scratch locations belong to unprivileged users.
    Permissions must be normalized: archive modes are contributor-controlled.
    """
    for parent in (workspace.parent, workspace):
        os.chown(parent, 0, 0)
        parent.chmod(0o755)
    for root in (workspace / "head", workspace / "base"):
        for path in [root, *root.rglob("*")]:
            executable = path.stat().st_mode & 0o111
            os.chown(path, 0, 0)
            path.chmod(0o755 if path.is_dir() else 0o555 if executable else 0o444)
    context = workspace / "context.json"
    os.chown(context, 0, 0)
    context.chmod(0o444)
    for name, owner in (
        ("home", 65534),
        ("scratch", 65534),
        ("output", 65534),
        ("build-home", 65533),
        ("python-packages", 65533),
    ):
        path = workspace / name
        path.mkdir()
        os.chown(path, owner, owner)
        path.chmod(0o700 if name in ("home", "output", "build-home") else 0o755)
    codex_home = workspace / "home" / ".codex"
    codex_home.mkdir()
    os.chown(codex_home, 65534, 65534)
    codex_home.chmod(0o700)


def unpack(archive_path, root):
    """Extract bounded regular source files without links or path traversal."""
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 100_000_000:
            raise ValueError("archive_too_large")
        for member in members:
            parts = Path(member.name).parts
            if (
                member.name.startswith("/")
                or ".." in parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("unsupported_archive_entry")
            if len(parts) < 2:
                continue
            target = root.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as out:
                    out.write(source.read())
                target.chmod(member.mode & 0o777)


def prepare_packages(workspace, environment, revision):
    """Build changed local Python packages offline as a separate unprivileged user.

    Build a disposable copy so backend writes cannot contaminate the source diff.
    Failures remain visible to Codex; unsupported dependencies never block inspection.
    """
    head, base = workspace / "head", workspace / "base"
    projects = []
    for config in sorted(head.rglob("pyproject.toml")):
        relative = config.parent.relative_to(head)
        changed = (
            relative == Path(".")
            or any(
                not (base / f.relative_to(head)).is_file()
                or f.read_bytes() != (base / f.relative_to(head)).read_bytes()
                for f in config.parent.rglob("*")
                if f.is_file()
            )
            or any(
                not (head / f.relative_to(base)).is_file()
                for f in (base / relative).rglob("*")
                if f.is_file()
            )
        )
        if changed:
            projects.append(relative)
    build = workspace / "build-source"
    shutil.copytree(head, build)
    # This copy still contains only validated regular archive files. Never recurse
    # as root through a tree after a build backend has had a chance to add symlinks.
    for path in [build, *build.rglob("*")]:
        executable = path.stat().st_mode & 0o111
        os.chown(path, 65533, 65533)
        path.chmod(0o755 if path.is_dir() or executable else 0o644)
    deadline = time.monotonic() + 120
    results, executions = [], []
    for relative in projects:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append({"package": str(relative), "status": "setup_time_limit"})
            continue
        command = [
            "/opt/review-env/bin/python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--disable-pip-version-check",
            "--target",
            str(workspace / "python-packages"),
            str(build / relative),
        ]
        timed_out = False
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(
                isolated_build(command),
                cwd=build,
                env={
                    **environment,
                    "HOME": str(workspace / "build-home"),
                    "TMPDIR": str(workspace / "build-home"),
                    "PYTHONPATH": "/opt/review-env/lib/python3.11/site-packages",
                    "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=min(45, remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
                code = None
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 6000))
            output = log.read().decode(errors="replace")
        results.append({"package": str(relative), "exitCode": code, "output": output})
        executions.append(
            {
                "revision": revision,
                "command": " ".join(command),
                "exitCode": code,
                "limit": "setup_timeout" if timed_out else None,
            }
        )
    (workspace / "setup.json").write_text(json.dumps(results))
    return executions


def codex_args(workspace, environment, policy, prompt):
    """Trusted client configuration shared by production and real CLI regressions."""
    tool_environment = {**environment, "HOME": str(workspace / "scratch")}
    tool_environment.pop("CODEX_HOME", None)
    return [
        "codex",
        "exec",
        "--model",
        "gpt-6-astra",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        *tool_permissions(workspace),
        "--json",
        "--output-schema",
        "/tmp/review-schema.json",
        "-o",
        str(workspace / "output" / "result.json"),
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'model_provider="review_gateway"',
        "-c",
        'model_providers.review_gateway.name="Budgeted review gateway"',
        "-c",
        'model_providers.review_gateway.base_url="http://review-model.local"',
        "-c",
        'model_providers.review_gateway.env_http_headers={"X-Review-Token"="REVIEW_MODEL_CAPABILITY"}',
        "-c",
        'model_providers.review_gateway.wire_api="responses"',
        "-c",
        "model_providers.review_gateway.requires_openai_auth=false",
        "-c",
        "model_providers.review_gateway.request_max_retries=0",
        "-c",
        "model_providers.review_gateway.stream_max_retries=0",
        "-c",
        "model_providers.review_gateway.stream_idle_timeout_ms=900000",
        "-c",
        "model_auto_compact_token_limit=200000",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        "features.apps=false",
        "-c",
        "features.multi_agent=false",
        "-c",
        "features.shell_snapshot=false",
        "-c",
        'shell_environment_policy.inherit="none"',
        "-c",
        "shell_environment_policy.set={"
        + ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in tool_environment.items())
        + "}",
        "-c",
        'web_search="disabled"',
        "-c",
        "developer_instructions=" + json.dumps(policy),
        prompt,
    ]


def main():
    """Launch a fresh CLI session and return only validated-shape diagnostics."""
    deadline = time.monotonic() + 1200
    request = json.loads(Path("/tmp/request.json").read_text())
    workspace = Path("/workspace/review")
    unpack("/tmp/head.tar.gz", workspace / "head")
    unpack("/tmp/base.tar.gz", workspace / "base")
    home = workspace / "home"
    (workspace / "context.json").write_text(request["context"])
    Path("/tmp/review-schema.json").write_text(request["schema"])
    protect_evidence(workspace)
    environment = {
        "PATH": "/opt/review-env/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "CODEX_HOME": str(home / ".codex"),
        "PYTHONPATH": f"{workspace}/python-packages:{workspace}/head/src",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "UV_OFFLINE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CI": "1",
        "TMPDIR": str(workspace / "scratch"),
    }
    verify_isolation(workspace, environment)
    executions = prepare_packages(workspace, environment, request["head"])
    prompt = (
        "Review the PR described in /workspace/review/context.json using the supplied policy. "
        "The complete immutable head snapshot is /workspace/review/head and its merge-base "
        "snapshot is /workspace/review/base. Start with git diff --no-index --stat and "
        "--name-status between those directories (exit 1 means differences, not failure). "
        "Read /workspace/review/setup.json for offline package installation results. "
        "Snapshots, context and setup records are root-owned and read-only. Use "
        "/workspace/review/scratch for reproductions or writable source copies, and "
        "keep all review evidence anchored to the canonical head/base snapshots. "
        "Track changed-file coverage and inspect per-file diffs in bounded batches; do not "
        "dump the whole directory diff or context JSON. Read the PR description, relevant "
        "discussion and base CLAUDE.md, then prioritize changed code, tests and focused "
        "reproductions. Investigate correctness even if a product decision is pending. "
        "Assess usefulness and concrete maintenance tradeoffs without requiring a separate "
        "approval comment for routine in-scope work. Return the requested review JSON, "
        "naming any exact remaining checks and why they could not be completed. "
        "Do not modify GitHub or contact anyone."
    )
    args = codex_args(workspace, environment, request["policy"], prompt)
    process = subprocess.Popen(
        args,
        cwd=workspace / "head",
        env={**environment, "REVIEW_MODEL_CAPABILITY": request["token"]},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        user=65534,
        group=65534,
        extra_groups=[],
    )
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending = b""
    errors = bytearray()
    timed_out = False
    while streams.get_map():
        if time.monotonic() >= deadline:
            timed_out = True
            break
        for key, _ in streams.select(timeout=0.5):
            data = os.read(key.fd, 8192)
            if not data:
                streams.unregister(key.fileobj)
                continue
            if key.data == "stderr":
                errors.extend(data)
                del errors[:-4000]
                continue
            pending += data
            if len(pending) > 2_000_000:
                raise ValueError("codex_event_too_large")
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") in ("error", "turn.failed"):
                    errors.extend(json.dumps(event).encode())
                    del errors[:-4000]
                item = event.get("item", {})
                if (
                    event.get("type") == "item.completed"
                    and item.get("type") == "command_execution"
                    and len(executions) < 100
                ):
                    executions.append(
                        {
                            "revision": request["head"],
                            "command": str(item.get("command", ""))[:12000],
                            "exitCode": item.get("exit_code"),
                            "limit": None,
                        }
                    )
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    exit_code = process.wait(timeout=5)
    result = workspace / "output" / "result.json"
    review = (
        json.loads(result.read_text())
        if exit_code == 0 and result.is_file() and result.stat().st_size < 24000
        else None
    )
    # Only typed output leaves this container. Codex stderr and prompts may
    # include the scoped gateway token, so never publish/log them.
    failure = "model_timeout" if timed_out else ("codex_review_incomplete" if exit_code else None)
    if b"Review budget exhausted" in errors:
        failure = "budget_exhausted"
    elif b"Review billing hold" in errors:
        failure = "billing_hold"
    elif b"Review context limit reached" in errors:
        failure = "context_too_large"
    output = json.dumps(
        {"exitCode": exit_code, "failure": failure, "review": review, "executions": executions}
    )
    # Persist independently of the Workflow's RPC lifetime. Only delivery retries,
    # never inference. stdout remains a fallback for the trusted process poller.
    endpoint = f"http://review-model.local/{request['checkpointToken']}/checkpoint"
    for attempt in range(3):
        try:
            post = urllib.request.Request(
                endpoint, data=output.encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(post, timeout=10) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            if attempt < 2:
                time.sleep(1)
    print(output)


if __name__ == "__main__":
    main()
