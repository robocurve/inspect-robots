"""Offline artifact tests; Linux root additionally exercises real UID isolation."""

# Test names state their invariant; separate docstrings would repeat them.
# ruff: noqa: D101, D102

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import issue_runner as runner


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base, self.work = self.root / "base", self.root / "work"
        for tree in (self.base, self.work):
            (tree / "src").mkdir(parents=True)
            (tree / "src" / "a.py").write_text("before\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cumulative_diff_tracks_edits_additions_deletions_modes(self):
        (self.work / "src" / "a.py").unlink()
        added = self.work / "src" / "new.py"
        added.write_text("print('hello')\n")
        added.chmod(0o755)
        self.assertEqual(
            runner.capture_changes(self.base, self.work),
            [
                {"path": "src/a.py", "content": None, "mode": "100644"},
                {"path": "src/new.py", "content": "print('hello')\n", "mode": "100755"},
            ],
        )

    def test_prior_candidate_plus_new_edit_preserves_cumulative_artifact(self):
        runner.apply_changes(
            self.work, [{"path": "tests/regression.py", "content": "assert True", "mode": "100644"}]
        )
        (self.work / "src" / "a.py").write_text("fixed\n")
        self.assertEqual(
            [c["path"] for c in runner.capture_changes(self.base, self.work)],
            ["src/a.py", "tests/regression.py"],
        )

    def test_rejects_symlinks_without_reading_their_target(self):
        (self.work / "src" / "secret.py").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "unsafe_artifact_entry"):
            runner.capture_changes(self.base, self.work)

    def test_rejects_symlink_directory(self):
        (self.work / "tests").symlink_to(self.base / "src", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "unsafe_artifact_entry"):
            runner.capture_changes(self.base, self.work)

    def test_rejects_hardlinks(self):
        os.link(self.base / "src" / "a.py", self.work / "src" / "hard.py")
        with self.assertRaisesRegex(ValueError, "unsafe_artifact_entry"):
            runner.scan_tree(self.work)

    def test_rejects_fifo_without_hanging(self):
        os.mkfifo(self.work / "src" / "pipe")
        with self.assertRaisesRegex(ValueError, "unsafe_artifact_entry"):
            runner.scan_tree(self.work)

    def test_protected_and_traversal_paths(self):
        for path in (
            "../src/a.py",
            "src/../secret",
            "src/.env",
            "src/AGENTS.md",
            "plugins/x/pyproject.toml",
            "tests/a.key",
            ".github/workflows/test.yml",
            "tools/issue-bot/a.ts",
        ):
            with self.subTest(path=path):
                self.assertFalse(runner.safe_change(path))
        (self.work / "pyproject.toml").write_text("malicious")
        with self.assertRaisesRegex(ValueError, "unsafe_artifact_path"):
            runner.capture_changes(self.base, self.work)

    def test_binary_and_oversized_changes(self):
        for value in (b"null\x00byte", b"invalid\xff", b"a" * 200001):
            (self.work / "src" / "a.py").write_bytes(value)
            with self.assertRaises((ValueError, UnicodeError)):
                runner.capture_changes(self.base, self.work)

    def test_bounded_archive_rejects_links_and_traversal(self):
        for index, name in enumerate(("root/../../secret", "root/link")):
            archive = self.root / f"bad-{index}.tar"
            with tarfile.open(archive, "w") as out:
                member = tarfile.TarInfo(name)
                if index:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/etc/passwd"
                out.addfile(member)
            with self.assertRaisesRegex(ValueError, "unsupported_archive_entry"):
                runner.unpack(archive, self.root / f"extract-{index}")

    def test_archive_normalizes_untrusted_modes(self):
        archive = self.root / "source.tar"
        with tarfile.open(archive, "w") as out:
            member = tarfile.TarInfo("root/src/a.py")
            member.size, member.mode = 4, 0o4777
            out.addfile(member, io.BytesIO(b"pass"))
        target = self.root / "extract"
        runner.unpack(archive, target)
        self.assertEqual((target / "src" / "a.py").stat().st_mode & 0o7777, 0o755)

    def test_build_outputs_remain_readable_under_private_supervisor_umask(self):
        candidate = self.root / "candidate"
        (candidate / "src").mkdir(parents=True)
        (candidate / "src" / "a.py").write_text("before\n")
        destination = self.root / "python-packages"
        real_popen = subprocess.Popen
        observed = []

        def write_package(_command, **kwargs):
            observed.append(kwargs["umask"])
            # Exercise the real subprocess umask on a harmless local fixture;
            # UID switching and package execution are covered by Linux tests.
            for key in ("user", "group", "extra_groups"):
                kwargs.pop(key)
            return real_popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import sys; "
                        "p=Path(sys.argv[1]); p.mkdir(); "
                        "(p/'fixture.py').write_text('VALUE = 1\\n')"
                    ),
                    str(destination),
                ],
                **kwargs,
            )

        previous = os.umask(0o077)
        try:
            with (
                patch.object(runner, "own_tree"),
                patch.object(runner, "kill_user"),
                patch.object(runner.subprocess, "Popen", side_effect=write_package),
            ):
                records = runner.prepare_packages(self.root, {})
        finally:
            os.umask(previous)
        self.assertEqual(observed, [0o022])
        self.assertEqual(records[0]["exitCode"], 0)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o755)
        self.assertEqual((destination / "fixture.py").stat().st_mode & 0o777, 0o644)

    def test_each_role_is_fresh_native_codex_and_checkpoint_is_absent(self):
        for kind in runner.ROLES:
            request = {"kind": kind, "token": "a" * 64, "checkpointToken": "b" * 64}
            args = runner.codex_args(request, self.root)
            self.assertEqual(args[:2], ["codex", "exec"])
            self.assertIn("--ephemeral", args)
            self.assertIn("features.multi_agent=false", args)
            self.assertIn('model_reasoning_effort="high"', args)
            self.assertIn("gpt-6-astra", args)
            self.assertNotIn("resume", args)
            self.assertNotIn(request["checkpointToken"], " ".join(args))
            self.assertNotIn(request["token"], " ".join(args))
            self.assertIn('shell_environment_policy.inherit="none"', args)
            self.assertIn(
                runner.ROLES[kind],
                json.loads(
                    next(
                        x.split("=", 1)[1] for x in args if x.startswith("developer_instructions=")
                    )
                ),
            )

    def test_client_config_contains_only_constant_url_and_env_key(self):
        (self.root / "home").mkdir()
        request = {"token": "b" * 64, "checkpointToken": "c" * 64}
        with patch.object(runner.os, "chown"):
            runner.write_client_config(request, self.root)
        config = (self.root / "home/.codex/config.toml").read_text()
        self.assertIn('base_url = "http://issue-model.local"', config)
        self.assertIn('env_key = "ISSUE_GATEWAY_TOKEN"', config)
        self.assertNotIn(request["token"], config)
        self.assertNotIn(request["checkpointToken"], config)

    def test_capability_is_added_only_to_client_after_starting_tools(self):
        request = {"kind": "triage", "token": "b" * 64, "checkpointToken": "c" * 64}
        events = []
        server, process, selector = MagicMock(), MagicMock(), MagicMock()
        process.returncode = 0
        selector.get_map.return_value = {}

        def start_tools(_workspace, _source, environment):
            self.assertNotIn("ISSUE_GATEWAY_TOKEN", environment)
            self.assertNotIn(request["token"], json.dumps(environment))
            events.append("tools")
            return server, "ws://127.0.0.1:12345"

        def start_client(args, **kwargs):
            self.assertEqual(events, ["tools"])
            self.assertEqual(kwargs["env"]["ISSUE_GATEWAY_TOKEN"], request["token"])
            self.assertNotIn(request["token"], " ".join(args))
            self.assertNotIn(request["checkpointToken"], json.dumps(kwargs["env"]))
            events.append("client")
            return process

        with (
            patch.object(runner, "stage_workspace"),
            patch.object(runner, "prepare_packages", return_value=[]),
            patch.object(runner, "write_client_config"),
            patch.object(runner, "start_tool_server", side_effect=start_tools),
            patch.object(runner.subprocess, "Popen", side_effect=start_client),
            patch.object(runner.selectors, "DefaultSelector", return_value=selector),
            patch.object(runner, "read_result", return_value={}),
            patch.object(runner, "kill_user"),
        ):
            self.assertEqual(runner.run_stage(request, self.root)["exitCode"], 0)
        self.assertEqual(events, ["tools", "client"])

    def test_native_shell_wrappers_preserve_the_actual_script(self):
        commands = {
            "/bin/bash -c 'pytest tests/test_bug.py -q'": "pytest tests/test_bug.py -q",
            "/bin/bash -lc 'cd /workspace/issue/candidate\nuv run pytest -q'": (
                "cd /workspace/issue/candidate\nuv run pytest -q"
            ),
            "/bin/bash --noprofile --norc -ec 'pytest -q'": "pytest -q",
        }
        for native, expected in commands.items():
            with self.subTest(native=native):
                self.assertEqual(runner.normalize_command(native), expected)

    def test_command_normalization_never_turns_printed_prose_into_execution(self):
        commands = (
            "echo 'pytest tests/test_bug.py'",
            "printf '%s' 'pytest; npm test'",
            "/bin/bash -c 'echo pytest' && echo done",
            "unterminated 'pytest",
        )
        for command in commands:
            self.assertEqual(runner.normalize_command(command), command)
        self.assertEqual(runner.normalize_command("x" * 12001), "command_too_large_to_classify")

    def test_result_symlink_cannot_exfiltrate_a_root_file(self):
        (self.root / "output").mkdir()
        (self.root / "output" / "result.json").symlink_to("/etc/passwd")
        with self.assertRaises(OSError):
            runner.read_result(self.root)

    @unittest.skipUnless(
        os.geteuid() == 0 and os.uname().sysname == "Linux", "requires root Linux container"
    )
    def test_actual_agent_and_build_users_cannot_change_evidence_or_checkpoint(self):
        self.root.chmod(0o755)
        archive = self.root / "source.tar"
        with tarfile.open(archive, "w") as out:
            member = tarfile.TarInfo("root/src/a.py")
            member.size, member.mode = 4, 0o777
            out.addfile(member, io.BytesIO(b"pass"))
        original = runner.unpack
        workspace = self.root / "issue"
        request = {
            "kind": "implement",
            "files": [],
            "issue": {},
            "context": "",
            "plan": "",
            "feedback": "",
            "inputDigest": "a" * 64,
            "schema": {},
        }
        with patch.object(
            runner,
            "unpack",
            side_effect=lambda _archive, destination: original(archive, destination),
        ):
            runner.stage_workspace(request, workspace)
        secret = self.root / "request.json"
        secret.write_text("checkpoint-secret")
        secret.chmod(0o600)
        script = """
import os, pathlib, sys
w = pathlib.Path(sys.argv[1])
attempts = [lambda: (w/'base/src/a.py').write_text('tamper'),
 lambda: (w/'candidate/src/a.py').write_text('tamper'),
 lambda: (w/'base').rename(w/'moved'),
 lambda: (w/'issue.json').write_text('tamper'),
 lambda: (w/'schema.json').unlink(),
 lambda: pathlib.Path(sys.argv[2]).read_text()]
for attempt in attempts:
 try: attempt()
 except PermissionError: pass
 else: raise AssertionError('evidence or checkpoint accessible')
"""
        for uid in (runner.AGENT_UID, runner.BUILD_UID, runner.TOOL_UID):
            completed = subprocess.run(
                [sys.executable, "-c", script, str(workspace), str(secret)],
                user=uid,
                group=uid,
                extra_groups=[],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((workspace / "base" / "src" / "a.py").read_text(), "pass")
        # Native tool identity can implement; build user cannot mutate its work.
        command = [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('fixed')",
            str(workspace / "work" / "src" / "a.py"),
        ]
        self.assertEqual(
            subprocess.run(
                command,
                user=runner.TOOL_UID,
                group=runner.TOOL_UID,
                extra_groups=[],
                capture_output=True,
            ).returncode,
            0,
        )
        self.assertNotEqual(
            subprocess.run(
                command,
                user=runner.BUILD_UID,
                group=runner.BUILD_UID,
                extra_groups=[],
                capture_output=True,
            ).returncode,
            0,
        )


if __name__ == "__main__":
    unittest.main()
