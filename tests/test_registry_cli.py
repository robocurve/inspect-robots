"""Registry resolution, entry-point discovery, and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

import roboinspect.registry as reg
from roboinspect.cli import main
from roboinspect.mock import ScriptedPolicy
from roboinspect.registry import registered, resolve


def test_builtins_are_registered() -> None:
    assert "cubepick" in registered("embodiment")
    assert "scripted" in registered("policy")
    assert "success_at_end" in registered("scorer")
    assert "cubepick-reach" in registered("task")


def test_resolve_constructs_with_args() -> None:
    policy = resolve("policy", "scripted", chunk_size=6)
    assert isinstance(policy, ScriptedPolicy)
    assert policy.chunk_size == 6


def test_resolve_unknown_raises() -> None:
    with pytest.raises(KeyError, match="no policy named"):
        resolve("policy", "does-not-exist")


def test_entrypoint_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEP:
        name = "plugin_policy"

        def load(self) -> object:
            return ScriptedPolicy

    def fake_entry_points(*, group: str) -> list[object]:
        return [_FakeEP()] if group == "roboinspect.policies" else []

    # Reset discovery state and inject a fake installed plugin.
    monkeypatch.setattr(reg, "entry_points", fake_entry_points)
    monkeypatch.setattr(reg, "_loaded_entrypoints", False)
    assert "plugin_policy" in registered("policy")


def test_cli_list_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "policies"]) == 0
    out = capsys.readouterr().out
    assert "scripted" in out


def test_cli_list_all(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "embodiments:" in out and "tasks:" in out


def test_cli_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "-P",
            "chunk_size=6",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "status: success" in out
    assert "success_at_end" in out
    assert list(tmp_path.glob("*.json"))


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "RoboInspect" in capsys.readouterr().out
