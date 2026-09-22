"""Run agent plugin tests offline under the production tool sandbox boundary.

The credential-free isolation probe supplies immutable head/base archives.
No repository code executes as root, and no model or GitHub write is involved.
"""

import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

import tomllib
from packaging.requirements import Requirement

spec = importlib.util.spec_from_file_location("runner", "/opt/codex-review.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    """Check dependencies and retain collection, focused and full-suite outcomes."""
    workspace = Path("/workspace/review")
    runner.unpack("/tmp/head.tar.gz", workspace / "head")
    runner.unpack("/tmp/base.tar.gz", workspace / "base")
    (workspace / "context.json").write_text("{}")
    request = Path("/tmp/request.json")
    request.write_text('{"checkpointToken":"synthetic-dependency-probe-only"}')
    request.chmod(0o600)
    runner.protect_evidence(workspace)
    environment = {
        "PATH": "/opt/review-env/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workspace / "home"),
        "CODEX_HOME": str(workspace / "home/.codex"),
        "PYTHONPATH": f"{workspace}/python-packages:{workspace}/head/src",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "UV_OFFLINE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CI": "1",
        "TMPDIR": str(workspace / "scratch"),
    }
    runner.verify_isolation(workspace, environment)
    project = tomllib.loads(
        (workspace / "head/plugins/inspect-robots-agent/pyproject.toml").read_text()
    )["project"]
    dependencies = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
    versions = {}
    for declared in dependencies:
        requirement = Requirement(declared)
        if requirement.name == "inspect-robots":
            continue  # Installed from this immutable head with synthetic SCM metadata.
        if requirement.marker and not requirement.marker.evaluate():
            continue
        version = importlib.metadata.version(requirement.name)
        if not requirement.specifier.contains(version):
            raise AssertionError(f"{declared}: installed {version}")
        versions[requirement.name] = version
    print(json.dumps({"declared_dependencies": versions}), flush=True)
    runner.prepare_packages(workspace, environment, "a" * 40)
    setup = json.loads((workspace / "setup.json").read_text())
    if not setup or any(item.get("exitCode") != 0 for item in setup):
        raise AssertionError(json.dumps({"package_setup": setup}))
    environment["HOME"] = str(workspace / "scratch")
    environment.pop("CODEX_HOME")
    command = [
        "/opt/review-env/bin/python",
        "-m",
        "pytest",
        "-p",
        "pytest_cov",
        "-o",
        "addopts=",
        "-o",
        f"cache_dir={workspace}/scratch/pytest-cache",
        "-q",
        str(workspace / "head/plugins/inspect-robots-agent/tests"),
    ]
    for mode in ("collection", "affected_policy_tests", "full_suite"):
        if mode == "collection":
            args = [*command, "--collect-only"]
        elif mode == "affected_policy_tests":
            args = [*command[:-1], command[-1] + "/test_policy_e2e.py"]
        else:
            args = command
        result = subprocess.run(
            ["codex", "sandbox", *runner.tool_permissions(workspace), "--", *args],
            cwd=workspace / "scratch",
            env=environment,
            user=65534,
            group=65534,
            extra_groups=[],
            capture_output=True,
            text=True,
            timeout=150,
        )
        print(
            json.dumps(
                {
                    "phase": mode,
                    "exitCode": result.returncode,
                    "stdout": result.stdout[-12000:],
                    "stderr": result.stderr[-6000:],
                }
            ),
            flush=True,
        )
        if result.returncode and mode != "full_suite":
            raise SystemExit(result.returncode)
        # The complete suite includes localhost server tests. Retain their real
        # failure output without mistaking socket restrictions for missing deps.
        # A completed probe is not a claim that the full suite passed.


if __name__ == "__main__":
    main()
