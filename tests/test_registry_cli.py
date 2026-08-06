"""Registry resolution, entry-point discovery, and the CLI."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

import pytest

import inspect_robots.cli as cli
import inspect_robots.registry as reg
from inspect_robots._claims import claim_devices
from inspect_robots.cli import main
from inspect_robots.conformance import DeviceSlot
from inspect_robots.console import USAGE, USAGE_END_ONLY
from inspect_robots.defaults import (
    _ENV_EMBODIMENT as ENV_EMBODIMENT,
)
from inspect_robots.defaults import (
    _ENV_POLICY as ENV_POLICY,
)
from inspect_robots.defaults import (
    _ENV_SIM_EMBODIMENT as ENV_SIM_EMBODIMENT,
)
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
from inspect_robots.mock import ScriptedPolicy
from inspect_robots.registry import registered, resolve
from inspect_robots.session import OperatorSession


class _ConsolePolicy(ScriptedPolicy):
    """Opt into the framework's attended operator-message channel."""

    accepts_operator_messages = True


@pytest.fixture(autouse=True)
def _hermetic_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep CLI runs blind to the developer's real config file and env vars."""
    config_home = tmp_path / "config-home"
    config_home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv(ENV_POLICY, raising=False)
    monkeypatch.delenv(ENV_EMBODIMENT, raising=False)
    monkeypatch.delenv(ENV_SIM_EMBODIMENT, raising=False)
    monkeypatch.delenv("INSPECT_ROBOTS_CONFIG", raising=False)
    return config_home


def _write_config(config_home: Path, body: str) -> Path:
    path = config_home / "inspect-robots" / "config.ini"
    path.parent.mkdir()
    path.write_text(body, encoding="utf-8")
    return path


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
        return [_FakeEP()] if group == "inspect_robots.policies" else []

    # Reset discovery state and inject a fake installed plugin.
    monkeypatch.setattr(reg, "entry_points", fake_entry_points)
    monkeypatch.setattr(reg, "_loaded_entrypoints", False)
    assert "plugin_policy" in registered("policy")


def test_autoload_opt_out_skips_entrypoints_but_keeps_builtins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = "optout-skip-probe-policy"

    class _FakeEP:
        name = probe

        def load(self) -> object:  # pragma: no cover - must never run when opted out
            raise AssertionError("entry point loaded despite the autoload opt-out")

    def fake_entry_points(*, group: str) -> list[object]:
        return [_FakeEP()] if group == "inspect_robots.policies" else []

    monkeypatch.setattr(reg, "entry_points", fake_entry_points)
    monkeypatch.setattr(reg, "_loaded_entrypoints", False)
    monkeypatch.setenv(reg.DISABLE_AUTOLOAD_ENV, "1")

    try:
        policies = registered("policy")
    finally:
        reg._FACTORIES["policy"].pop(probe, None)  # keep the shared registry clean
    assert probe not in policies  # discovery skipped, load() never called
    assert "scripted" in policies  # in-tree builtins still resolve


def test_autoload_opt_out_is_not_latched(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = "optout-latch-probe-policy"

    class _FakeEP:
        name = probe

        def load(self) -> object:
            return ScriptedPolicy

    def fake_entry_points(*, group: str) -> list[object]:
        return [_FakeEP()] if group == "inspect_robots.policies" else []

    monkeypatch.setattr(reg, "entry_points", fake_entry_points)
    monkeypatch.setattr(reg, "_loaded_entrypoints", False)

    try:
        monkeypatch.setenv(reg.DISABLE_AUTOLOAD_ENV, "1")
        assert probe not in registered("policy")

        # Clearing the opt-out re-enables discovery in the same process: the
        # skip must not have marked entry points as already loaded.
        monkeypatch.delenv(reg.DISABLE_AUTOLOAD_ENV)
        assert probe in registered("policy")
    finally:
        reg._FACTORIES["policy"].pop(probe, None)  # keep the shared registry clean


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
    assert "run status: completed" in out
    assert "success_at_end" in out
    (written,) = tmp_path.glob("*.json")
    assert f"log: {written}" in out  # the CLI tells the user where the log went
    assert "error:" not in out
    # A clean run teaches both terminal and browser read-back commands.
    assert f"hint: inspect it with: inspect-robots inspect {written}" in out
    assert f"hint: HTML viewer: inspect-robots view {written}" in out
    assert f"hint: browse all logs: inspect-robots view {tmp_path}" in out
    assert out.count("hint:") == 3
    assert out.rstrip().endswith(f"hint: browse all logs: inspect-robots view {tmp_path}")


@pytest.mark.parametrize(
    ("path_state", "expected"),
    [
        ("existing", "partial log written"),
        ("none", "no log written"),
        ("missing", "no log written"),
    ],
)
def test_cli_cancelled_run_reports_partial_log_state(
    path_state: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots
    import inspect_robots.logging

    path = tmp_path / f"{path_state}.json"
    if path_state == "existing":
        path.write_text("{}", encoding="utf-8")

    class _CancelSink:
        def __init__(self, log_dir: str) -> None:
            del log_dir
            self.path: Path | None = None if path_state == "none" else path

    def interrupted_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(inspect_robots.logging, "JsonLogSink", _CancelSink)
    monkeypatch.setattr(inspect_robots, "eval", interrupted_eval)

    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "--log-dir",
            str(tmp_path),
        ]
    )

    assert rc == 130
    assert f"cancelled: {expected}" in capsys.readouterr().out


def test_cli_run_embodiment_fault_prints_error_scene_and_inspect_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.scene import Scene
    from inspect_robots.types import Observation

    class _FaultOnSecondScene(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self._resets = 0

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self._resets += 1
            if self._resets == 2:
                raise RuntimeError("reset exploded")
            return super().reset(scene, seed=seed)

    name = "fault-on-second-scene-for-cli-test"
    embodiment_decorator(name)(_FaultOnSecondScene)
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "-T",
                "num_scenes=2",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name)

    assert rc == 1
    out = capsys.readouterr().out
    assert "run status: error" in out
    assert "error: EmbodimentFault: reset exploded" in out
    assert "  [error] scene-1\n" in out
    assert "scene-0" not in out  # successful scenes are not failure context
    assert out.count("EmbodimentFault: reset exploded") == 1
    (written,) = tmp_path.glob("*.json")
    assert f"hint: inspect it with: inspect-robots inspect {written}" in out


def test_cli_run_prints_distinct_scene_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_robots.registry import policy as policy_decorator
    from inspect_robots.scene import Scene

    class _ResetFailurePolicy(ScriptedPolicy):
        def reset(self, scene: Scene) -> None:
            raise RuntimeError("policy reset exploded")

    name = "reset-failure-for-cli-test"
    policy_decorator(name)(_ResetFailurePolicy)
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "-T",
                "num_scenes=1",
                "--policy",
                name,
                "--embodiment",
                "cubepick",
                "--fail-on-error",
                "1",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["policy"].pop(name)

    assert rc == 1
    out = capsys.readouterr().out
    assert "error: fail_on_error threshold exceeded (1 errors)" in out
    assert "[error] scene-0: PolicyError: policy reset exploded" in out


def test_cli_all_errored_run_exits_nonzero_with_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Issue #73: a run in which every trial errored must not look healthy.
    from inspect_robots.registry import policy as policy_decorator
    from inspect_robots.scene import Scene

    class _AlwaysBoomPolicy(ScriptedPolicy):
        def reset(self, scene: Scene) -> None:
            raise RuntimeError("invalid API key")

    name = "always-boom-for-cli-test"
    policy_decorator(name)(_AlwaysBoomPolicy)
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "-T",
                "num_scenes=1",
                "--policy",
                name,
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["policy"].pop(name)

    assert rc == 1
    out = capsys.readouterr().out
    assert "run status: error" in out
    assert "error: all 1 trial(s) errored; nothing was scored" in out
    assert "[error] scene-0: PolicyError: invalid API key" in out
    assert "trials: 1 (1 errored)" in out
    (written,) = tmp_path.glob("*.json")
    assert f"hint: inspect it with: inspect-robots inspect {written}" in out
    # And `inspect` on the written log shows the same headline facts.
    assert main(["inspect", str(written)]) == 1
    out = capsys.readouterr().out
    assert "run status:  error" in out
    assert "trials: 1 (1 errored)" in out


def test_cli_partial_errors_stay_success_but_are_visible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Issue #73: errored trials in an overall-success run must still be legible.
    from inspect_robots.registry import policy as policy_decorator
    from inspect_robots.scene import Scene

    class _BoomOnSecondScenePolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self._resets = 0

        def reset(self, scene: Scene) -> None:
            self._resets += 1
            if self._resets == 2:
                raise RuntimeError("policy reset exploded")
            super().reset(scene)

    name = "boom-on-second-scene-for-cli-test"
    policy_decorator(name)(_BoomOnSecondScenePolicy)
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "-T",
                "num_scenes=2",
                "--policy",
                name,
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["policy"].pop(name)

    assert rc == 0  # data survived; library semantics unchanged for partials
    out = capsys.readouterr().out
    assert "run status: completed" in out
    assert "trials: 2 (1 errored)" in out
    assert "[error] scene-1: PolicyError: policy reset exploded" in out
    (written,) = tmp_path.glob("*.json")
    assert f"hint: inspect it with: inspect-robots inspect {written}" in out


def test_cli_run_epochs_fail_on_error_store_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "-T",
            "num_scenes=1",
            "--epochs",
            "2",
            "--fail-on-error",
            "1",
            "--store-frames",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "trials: 2" in out  # --epochs overrode the task's epoch count
    assert list((tmp_path / "frames").rglob("*.npy"))  # --store-frames streamed (per-run subdir)


@pytest.mark.parametrize("epochs_value", ["0", "-1", "-5"])
def test_cli_run_zero_epochs_exits_with_guided_error(epochs_value: str) -> None:
    """--epochs 0 / negative must produce a guided SystemExit, not a raw traceback (#145)."""
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--epochs",
                epochs_value,
            ]
        )
    message = str(excinfo.value)
    assert "--epochs" in message
    assert epochs_value in message


@pytest.mark.parametrize("epochs_value", ["0", "-1"])
def test_cli_eval_set_zero_epochs_exits_with_guided_error(epochs_value: str) -> None:
    """eval-set --epochs 0 / negative must produce a guided SystemExit naming the task (#145)."""
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--epochs",
                epochs_value,
            ]
        )
    message = str(excinfo.value)
    assert "--epochs" in message
    assert epochs_value in message
    assert "cubepick-reach" in message  # the failing task name is named


def _register_task(name: str, *, num_scenes: int = 1, max_steps: int = 20) -> None:
    from inspect_robots.registry import task as task_decorator
    from inspect_robots.scene import Scene
    from inspect_robots.scorer import success_at_end
    from inspect_robots.task import Task

    @task_decorator(name)
    def _factory() -> Task:
        return Task(
            name=name,
            scenes=[Scene(id=f"s{i}", instruction="reach", init_seed=i) for i in range(num_scenes)],
            scorer=success_at_end(),
            max_steps=max_steps,
        )


_DEVICE_CLAIM_SLOTS = (DeviceSlot(arg="channel", kind="can", label="bus"),)
_DEVICE_CLAIM_EMBODIMENT = "device-claim-test-embodiment"


def _register_device_claim_embodiment(monkeypatch: pytest.MonkeyPatch) -> str:
    from inspect_robots.mock import CubePickEmbodiment

    class _DeviceClaimEmbodiment(CubePickEmbodiment):
        DEVICE_SLOTS = _DEVICE_CLAIM_SLOTS

        def __init__(self, channel: str) -> None:
            super().__init__()
            self.channel = channel

    monkeypatch.setitem(
        reg._FACTORIES["embodiment"], _DEVICE_CLAIM_EMBODIMENT, _DeviceClaimEmbodiment
    )
    return _DEVICE_CLAIM_EMBODIMENT


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_run_claims_and_releases_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _register_device_claim_embodiment(monkeypatch)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    args = [
        "run",
        "--task",
        "cubepick-reach",
        "--policy",
        "scripted",
        "--embodiment",
        name,
        "-E",
        "channel=can9",
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    existing = claim_devices(_DEVICE_CLAIM_SLOTS, {"channel": "can9"}, os.environ)

    try:
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        assert "can9" in str(exc_info.value)
        assert "already claimed" in str(exc_info.value)
    finally:
        existing.release()

    assert main(args) == 0
    after = claim_devices(_DEVICE_CLAIM_SLOTS, {"channel": "can9"}, os.environ)
    after.release()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_run_without_device_slots_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    assert (
        main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    assert not (tmp_path / "inspect-robots" / "locks").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_claim_released_when_embodiment_construction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenDeviceClaimEmbodiment:
        DEVICE_SLOTS = _DEVICE_CLAIM_SLOTS

        def __init__(self, channel: str) -> None:
            raise TypeError(f"construction exploded for {channel}")

    monkeypatch.setitem(
        reg._FACTORIES["embodiment"], _DEVICE_CLAIM_EMBODIMENT, _BrokenDeviceClaimEmbodiment
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    with pytest.raises(SystemExit, match="construction exploded"):
        main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                _DEVICE_CLAIM_EMBODIMENT,
                "-E",
                "channel=can9",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )

    after = claim_devices(_DEVICE_CLAIM_SLOTS, {"channel": "can9"}, os.environ)
    after.release()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_eval_set_claims_once_for_the_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _register_device_claim_embodiment(monkeypatch)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    _register_task("claim/a")
    _register_task("claim/b")
    args = [
        "eval-set",
        "claim/a",
        "claim/b",
        "--policy",
        "scripted",
        "--embodiment",
        name,
        "-E",
        "channel=can9",
        "--log-dir",
        str(tmp_path / "logs"),
    ]

    try:
        existing = claim_devices(_DEVICE_CLAIM_SLOTS, {"channel": "can9"}, os.environ)
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(args)
            assert "can9" in str(exc_info.value)
            assert "already claimed" in str(exc_info.value)
        finally:
            existing.release()

        assert main(args) == 0
        after = claim_devices(_DEVICE_CLAIM_SLOTS, {"channel": "can9"}, os.environ)
        after.release()
    finally:
        reg._FACTORIES["task"].pop("claim/a", None)
        reg._FACTORIES["task"].pop("claim/b", None)


def test_cli_eval_set_runs_multiple_exact_tasks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_task("kb/a")
    _register_task("kb/b")
    try:
        rc = main(
            [
                "eval-set",
                "kb/a",
                "kb/b",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["task"].pop("kb/a", None)
        reg._FACTORIES["task"].pop("kb/b", None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks: kb/a, kb/b" in out
    assert "run status: completed" in out
    assert "[completed] kb/a" in out
    assert "[completed] kb/b" in out
    assert out.count("log dir:") == 1  # one shared line, not one per task
    assert f"hint: browse all logs: inspect-robots view {tmp_path}" in out
    assert "HTML viewer: inspect-robots view" not in out
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_cli_eval_set_glob_matches_by_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_task("kb/a")
    _register_task("kb/b")
    _register_task("other/c")
    try:
        rc = main(
            [
                "eval-set",
                "kb/*",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["task"].pop("kb/a", None)
        reg._FACTORIES["task"].pop("kb/b", None)
        reg._FACTORIES["task"].pop("other/c", None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks: kb/a, kb/b" in out
    assert "other/c" not in out


def test_cli_eval_set_dedups_overlapping_patterns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_task("kb/a")
    _register_task("kb/b")
    try:
        rc = main(
            [
                "eval-set",
                "kb/*",
                "kb/a",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["task"].pop("kb/a", None)
        reg._FACTORIES["task"].pop("kb/b", None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks: kb/a, kb/b" in out  # kb/a not repeated despite matching twice
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_cli_eval_set_unmatched_pattern_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "eval-set",
                "does-not-exist/*",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
            ]
        )
    message = str(excinfo.value)
    assert "no task matches 'does-not-exist/*'" in message
    assert "registered tasks: " in message


def test_cli_eval_set_epochs_override_applies_to_every_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_task("kb/a")
    _register_task("kb/b")
    try:
        rc = main(
            [
                "eval-set",
                "kb/a",
                "kb/b",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--epochs",
                "2",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["task"].pop("kb/a", None)
        reg._FACTORIES["task"].pop("kb/b", None)
    assert rc == 0
    from inspect_robots import read_eval_log

    logs = [read_eval_log(str(p)) for p in tmp_path.glob("*.json")]
    assert len(logs) == 2
    assert all(log.results.total_trials == 2 for log in logs)  # --epochs overrode both tasks


def test_cli_eval_set_sim_and_embodiment_conflict() -> None:
    with pytest.raises(SystemExit, match="drop one"):
        main(["eval-set", "cubepick-reach", "--sim", "--embodiment", "cubepick"])


def test_cli_eval_set_guardrail_flags_conflict() -> None:
    with pytest.raises(SystemExit, match="drop one"):
        main(
            [
                "eval-set",
                "cubepick-reach",
                "--disable-guardrails",
                "--max-action-delta",
                "0.1",
            ]
        )


def test_cli_eval_set_disable_guardrails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "eval-set",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "--disable-guardrails",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out, err = capsys.readouterr()
    assert "guardrails: disabled (--disable-guardrails)" in out
    assert "WARNING: guardrails disabled" in err


def test_cli_eval_set_degraded_guardrails_warn_but_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Same bare-space case as test_cli_degraded_guardrails_warn_but_run, via eval-set.
    from dataclasses import replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.spaces import Box

    class _BareSpaceEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, action_space=Box(shape=(2,)))

    name = "bare-cubepick-for-eval-set-test"
    embodiment_decorator(name)(_BareSpaceEmbodiment)
    try:
        rc = main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    captured = capsys.readouterr()
    assert "guardrails: none active" in captured.out
    assert "no guardrails" in captured.err


def test_cli_eval_set_sim_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_SIM_EMBODIMENT, "cubepick")
    rc = main(
        [
            "eval-set",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--sim",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"embodiment: cubepick (--sim, from ${ENV_SIM_EMBODIMENT})" in out


def test_eval_set_prompts_when_operator_ends_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace as dc_replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.types import OPERATOR_END, Action, StepResult

    class _OperatorEndsEmbodiment(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:  # every step: human ends the episode
            result = super().step(action)
            return dc_replace(
                result, terminated=True, truncated=False, termination_reason=OPERATOR_END
            )

    name = "operator-ends-cubepick-for-eval-set-test"
    embodiment_decorator(name)(_OperatorEndsEmbodiment)
    _tty_stdin(monkeypatch)
    answers = iter(["y", ""] * 4)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    log_dir = tmp_path / "logs"
    try:
        rc = main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(log_dir),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    log = _read_only_log(log_dir)
    assert [s.operator_judgements for s in log.samples] == [("y",)] * 4
    assert [s.operator_notes for s in log.samples] == [(None,)] * 4
    capsys.readouterr()


def test_eval_set_operator_end_does_not_prompt_without_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace as dc_replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.types import OPERATOR_END, Action, StepResult

    class _OperatorEndsEmbodiment(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:  # every step: human ends the episode
            result = super().step(action)
            return dc_replace(
                result, terminated=True, truncated=False, termination_reason=OPERATOR_END
            )

    name = "operator-ends-cubepick-for-eval-set-test"
    embodiment_decorator(name)(_OperatorEndsEmbodiment)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("non-TTY eval-set must not prompt")
    )
    log_dir = tmp_path / "logs"
    try:
        rc = main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(log_dir),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    log = _read_only_log(log_dir)
    assert [s.operator_judgements for s in log.samples] == [(None,)] * 4
    capsys.readouterr()


def test_eval_set_operator_end_respects_no_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace as dc_replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.types import OPERATOR_END, Action, StepResult

    class _OperatorEndsEmbodiment(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:  # every step: human ends the episode
            result = super().step(action)
            return dc_replace(
                result, terminated=True, truncated=False, termination_reason=OPERATOR_END
            )

    name = "operator-ends-cubepick-for-eval-set-test"
    embodiment_decorator(name)(_OperatorEndsEmbodiment)
    _tty_stdin(monkeypatch)
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("--no-prompt eval-set must not prompt")
    )
    log_dir = tmp_path / "logs"
    try:
        rc = main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--no-prompt",
                "--log-dir",
                str(log_dir),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    log = _read_only_log(log_dir)
    assert [s.operator_judgements for s in log.samples] == [(None,)] * 4
    capsys.readouterr()


def test_cli_eval_set_one_task_fails_aggregate_status_is_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.scene import Scene
    from inspect_robots.types import Observation

    class _FaultOnSecondTask(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self._resets = 0

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self._resets += 1
            if self._resets == 2:
                raise RuntimeError("reset exploded")
            return super().reset(scene, seed=seed)

    name = "fault-on-second-task-for-eval-set-test"
    embodiment_decorator(name)(_FaultOnSecondTask)
    _register_task("kb/a")
    _register_task("kb/b")
    try:
        rc = main(
            [
                "eval-set",
                "kb/a",
                "kb/b",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
        reg._FACTORIES["task"].pop("kb/a", None)
        reg._FACTORIES["task"].pop("kb/b", None)
    assert rc == 1
    out = capsys.readouterr().out
    assert "run status: error" in out
    assert "[completed] kb/a" in out
    assert "[error] kb/b" in out
    assert "reset exploded" in out


def test_cli_eval_set_policy_errors_every_reset_without_fail_on_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A task where every trial errors degrades to a top-level error log even
    without --fail-on-error (issue #73): there is no surviving data for
    fail_on_error's flaky-trial tolerance to protect, so eval-set surfaces it
    as [error] with the "all N trial(s) errored" detail, not a silent success.
    """
    from inspect_robots.registry import policy as policy_decorator
    from inspect_robots.scene import Scene

    class _AlwaysFailsPolicy(ScriptedPolicy):
        def reset(self, scene: Scene) -> None:
            raise RuntimeError("policy reset exploded")

    name = "always-fails-for-eval-set-test"
    policy_decorator(name)(_AlwaysFailsPolicy)
    _register_task("kb/a")
    try:
        rc = main(
            [
                "eval-set",
                "kb/a",
                "--policy",
                name,
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["policy"].pop(name, None)
        reg._FACTORIES["task"].pop("kb/a", None)
    assert rc == 1
    out = capsys.readouterr().out
    assert "run status: error" in out
    assert "[error] kb/a  all 1 trial(s) errored; nothing was scored" in out


def test_cli_eval_set_zero_scene_task_has_no_metric_or_error_detail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No metrics, no top-level error: a task with zero scenes succeeds
    trivially (nothing ran, nothing to reduce), so its eval-set summary row
    has neither a metric nor an error to show (see _print_eval_set_summary).
    """
    _register_task("kb/a", num_scenes=0)
    try:
        rc = main(
            [
                "eval-set",
                "kb/a",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["task"].pop("kb/a", None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "run status: completed" in out
    assert "[completed] kb/a\n" in out  # no trailing metric/error detail


def test_eval_set_summary_surfaces_seconds_horizon(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _step_limit_log(task="timed", max_seconds=120.0, control_hz=10.0)
    cli._print_eval_set_summary(True, [log], "logs")
    out = capsys.readouterr().out
    assert "[completed] timed [120s -> 1200 steps at 10 Hz]" in out


def test_cli_eval_set_ctrl_c_reports_partial_logs_and_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C mid-set points at the log dir and returns 130 instead of a traceback.

    eval_set writes per-task logs and eval() persists a cancelled log for the
    interrupted task (#118), but eval-set doesn't hold the sink paths, so the
    hint points at the shared dir.
    """
    import inspect_robots

    def interrupted_eval_set(*args: object, **kwargs: object) -> tuple[bool, list[EvalLog]]:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(inspect_robots, "eval_set", interrupted_eval_set)
    _register_task("kb/a")
    try:
        rc = main(
            [
                "eval-set",
                "kb/a",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["task"].pop("kb/a", None)
    assert rc == 130
    out = capsys.readouterr().out
    assert f"cancelled: partial logs are under {tmp_path}" in out
    assert "inspect-robots inspect" in out
    assert f"hint: browse all logs: inspect-robots view {tmp_path}" in out
    assert "HTML viewer: inspect-robots view" not in out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Inspect Robots" in capsys.readouterr().out


def test_cli_help_lists_setup(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "setup" in capsys.readouterr().out


def test_view_is_protected_by_instruction_sugar_guard() -> None:
    assert "view" in cli._SUBCOMMANDS


def test_setup_is_protected_by_instruction_sugar_guard() -> None:
    assert "setup" in cli._SUBCOMMANDS


def test_cli_setup_requires_an_interactive_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="setup is interactive"):
        main(["setup"])


def test_cli_setup_dispatches_to_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_run_setup(_env: object, *, input_fn: object, out: object, interactive: bool) -> int:
        del input_fn, out
        calls.append(interactive)
        return 7

    monkeypatch.setattr("inspect_robots._setup.run_setup", fake_run_setup)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert main(["setup"]) == 7
    assert calls == [True]


# --------------------------------------------------------------------------- #
# Zero-config CLI (plan 0005): instruction sugar, defaults chain, operator flow.
# --------------------------------------------------------------------------- #
def _read_only_log(log_dir: Path) -> EvalLog:
    from inspect_robots import read_eval_log

    (path,) = log_dir.glob("*.json")
    return read_eval_log(str(path))


def _step_limit_log(
    *,
    task: str = "adhoc",
    reasons: tuple[str | None, ...] = ("max_steps",),
    max_steps: int | None = 1200,
    max_seconds: float | None = None,
    control_hz: object = 10.0,
) -> EvalLog:
    return EvalLog(
        version=1,
        status="success",
        eval=EvalSpec(
            task=task,
            policy="p",
            embodiment="e",
            created="x",
            inspect_robots_version="0",
            embodiment_info={"control_hz": control_hz},
            max_steps=max_steps,
            max_seconds=max_seconds,
        ),
        results=EvalResults(
            total_scenes=1,
            total_trials=len(reasons),
            metrics={"success_at_end": 0.0},
        ),
        stats=EvalStats(started_at="a", completed_at="b", duration_s=0.0, total_steps=1),
        samples=(
            SceneResult(
                scene_id="s0",
                status="success",
                epochs=tuple({} for _ in reasons),
                termination_reasons=reasons,
            ),
        ),
    )


def _transcript_log(*, status: str = "success") -> EvalLog:
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at the workspace"},
                {"type": "image_url", "image_url": {"url": "omitted"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_move",
                    "type": "function",
                    "function": {"name": "move_by", "arguments": '{"dx": 0.1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_move", "content": "moved 2 steps"},
    ]
    return EvalLog(
        version=1,
        status=status,
        eval=EvalSpec(
            task="agent-task",
            policy="agent",
            embodiment="e",
            created="x",
            inspect_robots_version="0",
        ),
        results=EvalResults(total_scenes=1, total_trials=3, metrics={}),
        stats=EvalStats(started_at="a", completed_at="b", duration_s=0.0, total_steps=1),
        samples=(
            SceneResult(
                scene_id="s0",
                status=status,
                epochs=({}, {}, {}),
                policy_transcripts=(chat, None, {"custom": [1, 2]}),
            ),
        ),
        error="trial failed" if status == "error" else None,
    )


def _run_with_synthesized_log(log: EvalLog, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> int:
    import inspect_robots

    def synthesized_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args, kwargs
        return [log]

    monkeypatch.setattr(inspect_robots, "eval", synthesized_eval)
    return main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "--log-dir",
            str(tmp_path),
        ]
    )


def _write_log(log: EvalLog, tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")
    return path


def _write_wire_log(
    tmp_path: Path,
    trials: tuple[list[dict[str, object]], ...],
    *,
    pointer_override: str | None = None,
) -> tuple[Path, Path]:
    log = _step_limit_log(reasons=tuple("success" for _ in trials))
    pointers: list[dict[str, object]] = []
    for epoch, rows in enumerate(trials):
        pointer = (
            pointer_override
            if pointer_override is not None
            else f"wire/run/s0-e{epoch}/calls.jsonl"
        )
        pointers.append({"wire_capture": pointer})
        if pointer_override is None:
            calls_path = tmp_path / pointer
            calls_path.parent.mkdir(parents=True, exist_ok=True)
            calls_path.write_text(
                "".join(f"{json.dumps(row)}\n" for row in rows),
                encoding="utf-8",
            )
    scene = dataclasses.replace(log.samples[0], trial_metadata=tuple(pointers))
    log_path = _write_log(dataclasses.replace(log, samples=(scene,)), tmp_path, "wire.json")
    blob_dir = tmp_path / "wire" / "run" / "blobs"
    return log_path, blob_dir


def _view_frame_log(frames_dir: str) -> EvalLog:
    log = _transcript_log()
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "camera 'top_cam' (step 4):"},
                {"type": "text", "text": "[image omitted: streamed camera frame]"},
            ],
        }
    ]
    scene = dataclasses.replace(log.samples[0], epochs=({},), policy_transcripts=(chat,))
    return dataclasses.replace(
        log,
        stats=dataclasses.replace(log.stats, frames_dir=frames_dir),
        samples=(scene,),
    )


def _write_view_frame_fixture(tmp_path: Path) -> tuple[Path, Path]:
    import numpy as np

    log_dir = tmp_path / "logs"
    frames_dir = log_dir / "frames" / "run-stamp"
    frames_dir.mkdir(parents=True)
    np.save(
        frames_dir / "s0-e0_top_cam_000004.npy",
        np.zeros((3, 4, 3), dtype=np.uint8),
    )
    recorded = str(tmp_path / "old-machine" / "run-stamp")
    path = _write_log(_view_frame_log(recorded), log_dir, "run.json")
    return path, frames_dir


def _directory_view_log(
    *,
    created: str,
    instruction: str = "pick up the cube",
    status: str = "success",
    metrics: dict[str, float] | None = None,
    errored_trials: int = 0,
) -> EvalLog:
    log = _step_limit_log(reasons=("success",))
    scene = dataclasses.replace(
        log.samples[0],
        status=status,
        instruction=instruction,
        termination_reasons=("success",),
    )
    return dataclasses.replace(
        log,
        status=status,
        eval=dataclasses.replace(
            log.eval,
            created=created,
            policy_config={"model": "provider/models/claude-test"},
        ),
        results=dataclasses.replace(
            log.results,
            metrics={"success_at_end": 1.0} if metrics is None else metrics,
            errored_trials=errored_trials,
        ),
        samples=(scene,),
    )


@pytest.mark.parametrize("name", ["run.json", "run"])
def test_view_derives_default_html_path(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, name)
    expected = path.with_suffix(".html")

    assert main(["view", str(path)]) == 0

    document = expected.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert f"<title>adhoc - {name}</title>" in document
    assert capsys.readouterr().out == f"wrote {expected}\n"


def test_view_honors_output_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    output = tmp_path / "report.htm"

    assert main(["view", str(path), "-o", str(output)]) == 0

    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert capsys.readouterr().out == f"wrote {output}\n"


def test_view_stdout_contains_only_the_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")

    assert main(["view", str(path), "-o", "-"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("<!doctype html>")
    assert "wrote " not in out


def test_view_renders_null_metric_from_sanitized_non_finite_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression for #253: a single-file `view` must render a sanitized
    null metric as "n/a" instead of crashing on ``.4g`` formatting."""
    log = _step_limit_log(reasons=("success",))
    log = dataclasses.replace(
        log,
        results=dataclasses.replace(log.results, metrics={"min_distance_to_goal": None}),  # type: ignore[dict-item]
    )
    path = _write_log(log, tmp_path, "null-metric.json")

    assert main(["view", str(path)]) == 0

    document = path.with_suffix(".html").read_text(encoding="utf-8")
    assert '<div class="stat-value">n/a</div>' in document


def test_view_embeds_frames_resolved_from_log_relative_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _frames_dir = _write_view_frame_fixture(tmp_path)
    output = tmp_path / "report.html"

    assert main(["view", str(path), "-o", str(output)]) == 0

    document = output.read_text(encoding="utf-8")
    assert 'src="data:image/png;base64,' in document
    assert "[image omitted: streamed camera frame]" not in document
    capsys.readouterr()


def test_view_no_frames_keeps_placeholders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _frames_dir = _write_view_frame_fixture(tmp_path)
    output = tmp_path / "no-frames.html"

    assert main(["view", str(path), "-o", str(output), "--no-frames"]) == 0

    document = output.read_text(encoding="utf-8")
    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document
    capsys.readouterr()


def test_view_unresolvable_recorded_frames_degrade_without_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_view_frame_log(str(tmp_path / "missing" / "stamp")), tmp_path, "run.json")
    output = tmp_path / "missing.html"

    assert main(["view", str(path), "-o", str(output)]) == 0

    document = output.read_text(encoding="utf-8")
    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document
    capsys.readouterr()


def test_view_frames_budget_is_forwarded_as_decimal_megabytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    received: list[int] = []

    def fake_render_html(
        log: EvalLog,
        *,
        title: str,
        log_path: Path,
        frames_dir: Path | None,
        frames_budget_bytes: int,
    ) -> str:
        del log, title, frames_dir
        assert log_path == path
        received.append(frames_budget_bytes)
        return "<html></html>"

    monkeypatch.setattr(cli, "render_html", fake_render_html)

    assert main(["view", str(path), "--frames-budget", "1.25"]) == 0
    assert received == [1_250_000]
    capsys.readouterr()


@pytest.mark.parametrize(
    ("document_size", "suffix"),
    [(1_000_000, ""), (1_234_567, " (1.2 MB)")],
)
def test_view_wrote_line_reports_only_documents_over_one_megabyte(
    document_size: int,
    suffix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "render_html", lambda *args, **kwargs: "x" * document_size)

    assert main(["view", str(path), "-o", str(output)]) == 0

    assert capsys.readouterr().out == f"wrote {output}{suffix}\n"


def test_view_stdout_embeds_resolved_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _frames_dir = _write_view_frame_fixture(tmp_path)

    assert main(["view", str(path), "-o", "-"]) == 0

    out = capsys.readouterr().out
    assert 'src="data:image/png;base64,' in out
    assert "[image omitted: streamed camera frame]" not in out
    assert "wrote " not in out


@pytest.mark.parametrize("budget", ["-0.1", "nan", "inf"])
def test_view_rejects_non_finite_or_negative_frames_budget(budget: str, tmp_path: Path) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")

    with pytest.raises(SystemExit, match="--frames-budget must be a non-negative finite number"):
        main(["view", str(path), "--frames-budget", budget])


def test_view_rejects_directory_output_with_guidance(tmp_path: Path) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    output = tmp_path / "existing-directory"
    output.mkdir()

    with pytest.raises(SystemExit, match="is a directory; pass an HTML file path"):
        main(["view", str(path), "-o", str(output)])


def test_view_rejects_overwriting_the_input_log(tmp_path: Path) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")

    with pytest.raises(SystemExit, match="would overwrite the input log"):
        main(["view", str(path), "-o", str(path)])


def test_view_rejects_open_with_stdout(tmp_path: Path) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")

    with pytest.raises(SystemExit, match="no file to open"):
        main(["view", str(path), "-o", "-", "--open"])


def test_view_creates_missing_output_parents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    output = tmp_path / "new" / "nested" / "report.html"

    assert main(["view", str(path), "-o", str(output)]) == 0

    assert output.is_file()
    assert capsys.readouterr().out == f"wrote {output}\n"


@pytest.mark.parametrize("stdout_mode", [False, True])
def test_view_degrades_lone_surrogates_in_both_output_modes(
    stdout_mode: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _step_limit_log()
    scene = dataclasses.replace(log.samples[0], instruction="bad \ud800 data")
    path = _write_log(dataclasses.replace(log, samples=(scene,)), tmp_path, "hostile.json")

    if stdout_mode:
        assert main(["view", str(path), "-o", "-"]) == 0
        document = capsys.readouterr().out
    else:
        output = tmp_path / "hostile.html"
        assert main(["view", str(path), "-o", str(output)]) == 0
        capsys.readouterr()
        document = output.read_text(encoding="utf-8")

    assert "\ud800" not in document
    assert "bad ? data" in document


def test_view_open_receives_resolved_file_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    output = tmp_path / "report.html"
    opened: list[str] = []

    def open_browser(uri: str) -> bool:
        opened.append(uri)
        return True

    monkeypatch.setattr("webbrowser.open", open_browser)

    assert main(["view", str(path), "-o", str(output), "--open"]) == 0
    assert opened == [output.resolve().as_uri()]
    assert capsys.readouterr().err == ""


def test_view_false_browser_result_warns_without_changing_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    monkeypatch.setattr("webbrowser.open", lambda _uri: False)

    assert main(["view", str(path), "--open"]) == 0

    assert "warning: could not open browser" in capsys.readouterr().err


def test_view_browser_exception_warns_without_changing_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")

    def fail_to_open(_uri: str) -> bool:
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr("webbrowser.open", fail_to_open)

    assert main(["view", str(path), "--open"]) == 0

    err = capsys.readouterr().err
    assert "warning: could not open browser" in err
    assert "browser unavailable" in err


@pytest.mark.parametrize("status", ["success", "error"])
def test_view_exit_code_reports_artifact_production_not_eval_status(
    status: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_transcript_log(status=status), tmp_path, f"{status}.json")

    assert main(["view", str(path)]) == 0
    capsys.readouterr()


def test_view_directory_end_to_end_and_unreadable_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        _directory_view_log(created="2026-07-29T12:00:00Z"),
        logs,
        "older.json",
    )
    _write_log(
        _directory_view_log(
            created="2026-07-30T12:00:00Z",
            instruction="place the cube",
        ),
        logs,
        "newer.json",
    )
    (logs / "foreign.json").write_text("{not json", encoding="utf-8")

    assert main(["view", str(logs)]) == 0

    out = capsys.readouterr()
    html_dir = logs / "html"
    assert (html_dir / "older.html").is_file()
    assert (html_dir / "newer.html").is_file()
    assert not (html_dir / "foreign.html").exists()
    index = (html_dir / "index.html").read_text(encoding="utf-8")
    assert index.index("newer.json") < index.index("older.json")
    assert 'href="newer.html"' in index and 'href="older.html"' in index
    assert "foreign.json" in index and "unreadable:" in index
    assert out.out.startswith(f"index: {html_dir / 'index.html'} (3 logs, 2 pages, ")
    assert "[1/3] rendering foreign.json" not in out.err
    assert "warning: could not read or render foreign.json" in out.err
    assert out.err.count(" rendering ") == 2


def test_view_directory_includes_log_with_sanitized_null_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression for #253: a log the sink itself wrote (null metric from a
    non-finite score) reads fine and must appear in the directory index
    rather than being dropped as "could not read or render"."""
    logs = tmp_path / "logs"
    logs.mkdir()
    log = _directory_view_log(
        created="2026-07-30T12:00:00Z",
        metrics={"min_distance_to_goal": None},  # type: ignore[dict-item]
    )
    _write_log(log, logs, "null-metric.json")

    assert main(["view", str(logs)]) == 0

    out = capsys.readouterr()
    assert "could not read or render" not in out.err
    assert (logs / "html" / "null-metric.html").is_file()
    index = (logs / "html" / "index.html").read_text(encoding="utf-8")
    assert "null-metric.json" in index
    assert "min_distance_to_goal=n/a" in index
    assert out.out.startswith(f"index: {logs / 'html' / 'index.html'} (1 logs, 1 pages, ")


def test_view_directory_multi_scene_metrics_empty_samples_and_errored_trials(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    base = _directory_view_log(
        created="2026-07-30T12:00:00Z",
        metrics={"mean_score": 0.333333},
        errored_trials=1,
    )
    first = dataclasses.replace(
        base.samples[0],
        scene_id="first",
        instruction="shared instruction",
        termination_reasons=("success", None),
    )
    second = dataclasses.replace(
        base.samples[0],
        scene_id="second",
        instruction="shared instruction",
        termination_reasons=("max_steps", "success"),
    )
    multi = dataclasses.replace(
        base,
        results=dataclasses.replace(base.results, total_scenes=2, total_trials=4),
        samples=(first, second),
    )
    empty = dataclasses.replace(
        _directory_view_log(
            created="2026-07-29T12:00:00Z",
            status="cancelled",
            metrics={},
        ),
        results=EvalResults(total_scenes=0, total_trials=0, metrics={}),
        samples=(),
    )
    _write_log(multi, logs, "multi.json")
    _write_log(empty, logs, "empty.json")

    assert main(["view", str(logs)]) == 0

    index = (logs / "html" / "index.html").read_text(encoding="utf-8")
    assert "shared instruction" in index
    assert "mean_score=0.3333" in index
    assert "succeeded, hit step limit" in index
    assert '<span class="badge status-completed">completed</span>' in index
    assert '<span class="errored">(1 errored)</span>' in index
    assert "0 scenes" in index
    assert '<span class="badge status-cancelled">cancelled</span>' in index


def test_view_directory_incremental_mtime_and_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    first = _write_log(
        _directory_view_log(created="2026-07-29T12:00:00Z"),
        logs,
        "a.json",
    )
    _write_log(
        _directory_view_log(created="2026-07-30T12:00:00Z"),
        logs,
        "b.json",
    )

    assert main(["view", str(logs)]) == 0
    capsys.readouterr()

    assert main(["view", str(logs)]) == 0
    second = capsys.readouterr()
    assert " rendering " not in second.err
    assert "(2 logs, 0 pages, " in second.out

    from inspect_robots._html import render_html

    calls: list[str] = []

    def record_render(
        log: EvalLog,
        *,
        title: str,
        log_path: Path,
        frames_dir: Path | None,
        frames_budget_bytes: int,
    ) -> str:
        calls.append(log.eval.created)
        return render_html(
            log,
            title=title,
            log_path=log_path,
            frames_dir=frames_dir,
            frames_budget_bytes=frames_budget_bytes,
        )

    monkeypatch.setattr(cli, "render_html", record_render)
    page_mtime = (logs / "html" / "a.html").stat().st_mtime
    os.utime(first, (page_mtime + 10, page_mtime + 10))

    assert main(["view", str(logs)]) == 0
    refreshed = capsys.readouterr()
    assert calls == ["2026-07-29T12:00:00Z"]
    assert refreshed.err.count(" rendering ") == 1
    assert "rendering a.json" in refreshed.err

    calls.clear()
    assert main(["view", str(logs), "--force"]) == 0
    forced = capsys.readouterr()
    assert len(calls) == 2
    assert forced.err.count(" rendering ") == 2
    assert "(2 logs, 2 pages, " in forced.out


def test_view_directory_index_log_uses_collision_free_page(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        _directory_view_log(created="2026-07-30T12:00:00Z"),
        logs,
        "index.json",
    )

    assert main(["view", str(logs)]) == 0

    html_dir = logs / "html"
    assert (html_dir / "index.html").is_file()
    assert (html_dir / "index_log.html").is_file()
    assert 'href="index_log.html"' in (html_dir / "index.html").read_text(encoding="utf-8")


def test_view_directory_index_log_suffix_uses_run_targets_not_filesystem(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        _directory_view_log(created="2026-07-30T12:00:00Z"),
        logs,
        "index.json",
    )
    _write_log(
        _directory_view_log(created="2026-07-29T12:00:00Z"),
        logs,
        "index_log.json",
    )

    assert main(["view", str(logs)]) == 0
    assert (logs / "html" / "index_log_2.html").is_file()

    assert main(["view", str(logs)]) == 0
    assert not (logs / "html" / "index_log_3.html").exists()


def test_view_directory_rejects_default_html_regular_file(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    (logs / "html").write_text("occupied", encoding="utf-8")

    with pytest.raises(SystemExit, match="exists and is not a directory; move it or pass -o DIR"):
        main(["view", str(logs)])


def test_view_directory_rejects_stdout_output(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")

    with pytest.raises(SystemExit, match="cannot be used with a logs directory"):
        main(["view", str(logs), "-o", "-"])


def test_view_directory_rejects_existing_output_file(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    output = tmp_path / "reports"
    output.write_text("occupied", encoding="utf-8")

    with pytest.raises(SystemExit, match="is not a directory; pass an output directory"):
        main(["view", str(logs), "-o", str(output)])


def test_view_directory_out_dir_replaces_html_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    output = tmp_path / "reports" / "nested"

    assert main(["view", str(logs), "-o", str(output)]) == 0

    assert (output / "run.html").is_file()
    assert (output / "index.html").is_file()
    assert not (logs / "html").exists()
    assert capsys.readouterr().out.startswith(f"index: {output / 'index.html'} (1 logs, 1 pages, ")


def test_unreadable_index_entry_tolerates_vanished_file(tmp_path: Path) -> None:
    entry = cli._unreadable_index_entry(tmp_path / "gone.json", ValueError("boom"))

    assert entry.created == ""
    assert entry.error == "unreadable: boom"
    assert entry.page is None


def test_view_directory_empty_is_runtime_error(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()

    with pytest.raises(SystemExit, match=r"no top-level \*\.json") as excinfo:
        main(["view", str(logs)])

    assert excinfo.value.code
    assert not (logs / "html" / "index.html").exists()


def test_view_directory_open_targets_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    opened: list[str] = []

    def open_browser(uri: str) -> bool:
        opened.append(uri)
        return True

    monkeypatch.setattr("webbrowser.open", open_browser)

    assert main(["view", str(logs), "--open"]) == 0

    assert opened == [(logs / "html" / "index.html").resolve().as_uri()]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--port", "8300"),
        ("--host", "127.0.0.1"),
    ],
)
def test_view_serve_options_require_serve(
    flag: str,
    value: str,
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")

    with pytest.raises(SystemExit, match=rf"{flag} requires --serve"):
        main(["view", str(logs), flag, value])


def test_view_serve_requires_logs_directory(tmp_path: Path) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")

    with pytest.raises(SystemExit, match="--serve requires a logs directory"):
        main(["view", str(path), "--serve"])


def test_view_serve_end_to_end_uses_bound_port_and_refreshes_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    bound_ports: list[int] = []
    response_status: list[int] = []
    response_bodies: list[str] = []
    display_urls = cli._serve_display_urls

    def record_display_urls(bind_host: str, port: int) -> tuple[str, ...]:
        bound_ports.append(port)
        return display_urls(bind_host, port)

    def probe_server(seconds: float) -> None:
        assert seconds == cli._SERVE_RERENDER_SECONDS
        url = f"http://127.0.0.1:{bound_ports[0]}/index.html"
        with urllib.request.urlopen(url, timeout=2) as response:
            response_status.append(response.status)
            response_bodies.append(response.read().decode())
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_serve_display_urls", record_display_urls)
    monkeypatch.setattr(cli, "_serve_sleep", probe_server)

    assert main(["view", str(logs), "--serve", "--port", "0"]) == 0

    out = capsys.readouterr()
    assert response_status == [200]
    assert '<meta http-equiv="refresh" content="60">' in response_bodies[0]
    assert bound_ports[0] != 0
    assert f"serving logs at: http://127.0.0.1:{bound_ports[0]}/" in out.out
    assert "serving to this machine only; pass --host 0.0.0.0" in out.out
    assert out.out.index("index: ") < out.out.index("serving logs at: ")


@pytest.mark.parametrize(
    "rerender_error",
    [
        RuntimeError("directory changing"),
        SystemExit("no logs left"),
    ],
)
def test_view_serve_rerender_failure_warns_and_continues(
    rerender_error: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    bound_ports: list[int] = []
    sleep_calls = 0
    quiet_calls = 0
    pass_force_values: list[bool] = []
    render_directory = cli._render_view_directory
    display_urls = cli._serve_display_urls

    def record_display_urls(bind_host: str, port: int) -> tuple[str, ...]:
        bound_ports.append(port)
        return display_urls(bind_host, port)

    def flaky_render(
        args: Any,
        log_dir: Path,
        *,
        force: bool,
        quiet: bool,
        refresh_seconds: int | None,
    ) -> cli._DirectoryRenderResult:
        nonlocal quiet_calls
        pass_force_values.append(force)
        if quiet:
            quiet_calls += 1
            if quiet_calls == 1:
                raise rerender_error
        return render_directory(
            args,
            log_dir,
            force=force,
            quiet=quiet,
            refresh_seconds=refresh_seconds,
        )

    def advance_loop(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            url = f"http://127.0.0.1:{bound_ports[0]}/index.html"
            with urllib.request.urlopen(url, timeout=2) as response:
                assert response.status == 200
        elif sleep_calls == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_serve_display_urls", record_display_urls)
    monkeypatch.setattr(cli, "_render_view_directory", flaky_render)
    monkeypatch.setattr(cli, "_serve_sleep", advance_loop)

    assert main(["view", str(logs), "--serve", "--port", "0", "--force"]) == 0

    out = capsys.readouterr()
    assert f"warning: re-render failed: {rerender_error}" in out.err
    assert quiet_calls == 2
    assert pass_force_values == [True, False, False]
    assert out.out.count("index: ") == 1


def test_view_serve_keyboard_interrupt_mid_render_exits_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    render_directory = cli._render_view_directory

    def interrupted_render(
        args: Any,
        log_dir: Path,
        *,
        force: bool,
        quiet: bool,
        refresh_seconds: int | None,
    ) -> cli._DirectoryRenderResult:
        if quiet:
            raise KeyboardInterrupt
        return render_directory(
            args,
            log_dir,
            force=force,
            quiet=quiet,
            refresh_seconds=refresh_seconds,
        )

    monkeypatch.setattr(cli, "_render_view_directory", interrupted_render)
    monkeypatch.setattr(cli, "_serve_sleep", lambda _seconds: None)

    assert main(["view", str(logs), "--serve", "--port", "0"]) == 0

    out = capsys.readouterr()
    assert "re-render failed" not in out.err


def test_view_serve_interrupt_before_thread_start_skips_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")

    class InterruptedThread:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def start(self) -> None:
            # Interrupt delivered before serve_forever ever runs: shutdown()
            # must be skipped or this test hangs forever.
            raise KeyboardInterrupt

    monkeypatch.setattr(threading, "Thread", InterruptedThread)

    assert main(["view", str(logs), "--serve", "--port", "0"]) == 0


def test_view_serve_localhost_bind_prints_loopback_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_serve_sleep", interrupt)

    assert main(["view", str(logs), "--serve", "--port", "0", "--host", "localhost"]) == 0

    out = capsys.readouterr()
    assert "serving to this machine only" in out.out
    assert "serving to your network" not in out.out


def test_view_serve_wildcard_prints_hostname_localhost_and_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    bound_ports: list[int] = []
    display_urls = cli._serve_display_urls

    def record_display_urls(bind_host: str, port: int) -> tuple[str, ...]:
        bound_ports.append(port)
        return display_urls(bind_host, port)

    monkeypatch.setattr("socket.gethostname", lambda: "robot-host")
    monkeypatch.setattr(cli, "_serve_display_urls", record_display_urls)

    def stop_server(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_serve_sleep", stop_server)

    result = main(["view", str(logs), "--serve", "--host", "0.0.0.0", "--port", "0"])

    assert result == 0

    out = capsys.readouterr().out
    assert f"serving logs at: http://robot-host:{bound_ports[0]}/" in out
    assert f"http://localhost:{bound_ports[0]}/" in out
    assert "serving to your network: anyone who can reach this machine can view these logs" in out


def test_view_serve_url_helpers_follow_bind_address_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cli._serve_display_urls("127.0.0.1", 8300) == ("http://127.0.0.1:8300/",)
    monkeypatch.setattr("socket.gethostname", lambda: "robot-host")
    assert cli._serve_display_urls("0.0.0.0", 8300) == (
        "http://robot-host:8300/",
        "http://localhost:8300/",
    )
    assert cli._serve_open_url("127.0.0.1", 8300) == "http://127.0.0.1:8300/"
    assert cli._serve_open_url("0.0.0.0", 8300) == "http://127.0.0.1:8300/"
    assert cli._serve_open_url("192.0.2.10", 8300) == "http://192.0.2.10:8300/"


def test_view_serve_sigterm_handler_is_scoped_to_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    original_handler = signal.getsignal(signal.SIGTERM)
    serving_handlers: list[object] = []

    def record_handler_and_stop(_seconds: float) -> None:
        serving_handlers.append(signal.getsignal(signal.SIGTERM))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_serve_sleep", record_handler_and_stop)

    assert main(["view", str(logs), "--serve", "--port", "0"]) == 0

    assert serving_handlers == [cli._raise_keyboard_interrupt]
    assert signal.getsignal(signal.SIGTERM) == original_handler
    with pytest.raises(KeyboardInterrupt):
        cli._raise_keyboard_interrupt(signal.SIGTERM, None)


def test_view_serve_bind_failure_is_clean_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")

    def fail_to_bind(_address: tuple[str, int], _handler: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr(cli, "ThreadingHTTPServer", fail_to_bind)

    with pytest.raises(SystemExit, match="address already in use"):
        main(["view", str(logs), "--serve"])


def test_view_serve_open_url_rules_and_loopback_browser_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(_directory_view_log(created="2026-07-30T12:00:00Z"), logs, "run.json")
    opened: list[str] = []

    def open_browser(uri: str) -> bool:
        opened.append(uri)
        return True

    monkeypatch.setattr("webbrowser.open", open_browser)

    def stop_server(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_serve_sleep", stop_server)

    assert main(["view", str(logs), "--serve", "--port", "0", "--open"]) == 0

    assert len(opened) == 1
    assert opened[0].startswith("http://127.0.0.1:")
    assert opened[0].endswith("/")


def test_run_outcome_shows_timeout_without_a_count(
    _hermetic_defaults: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "max_steps = 3\n",
    )

    assert main(["reach the cube", "--log-dir", str(tmp_path / "logs")]) == 0

    out = capsys.readouterr().out
    assert "run status: completed" in out
    assert "outcome: hit step limit" in out


def test_run_outcome_groups_counts_and_orders_by_count_then_phrase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Insertion order (max_steps first) differs from the required order, so
    # this fails if the count-descending sort is dropped.
    log = _step_limit_log(reasons=("max_steps", "success", "success"))

    assert _run_with_synthesized_log(log, monkeypatch, tmp_path) == 0

    assert "outcome: 2 succeeded, 1 hit step limit" in capsys.readouterr().out


def test_run_outcome_breaks_count_ties_alphabetically_by_phrase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Insertion order ("succeeded" first) differs from alphabetical order, so
    # this fails if the tie-break is dropped.
    log = _step_limit_log(reasons=("success", "max_steps"))

    assert _run_with_synthesized_log(log, monkeypatch, tmp_path) == 0

    assert "outcome: 1 hit step limit, 1 succeeded" in capsys.readouterr().out


def test_inspect_renders_null_metric_from_sanitized_non_finite_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression for #253: json_log.py writes inf/nan metrics as JSON null
    so the log stays RFC 8259 valid; `inspect` must tolerate that null."""
    log = _step_limit_log(reasons=("success",))
    log = dataclasses.replace(
        log,
        results=dataclasses.replace(log.results, metrics={"min_distance_to_goal": None}),  # type: ignore[dict-item]
    )
    path = _write_log(log, tmp_path, "null-metric.json")

    assert main(["inspect", str(path)]) == 0

    assert "min_distance_to_goal: n/a" in capsys.readouterr().out


def test_inspect_outcome_maps_give_up_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_step_limit_log(reasons=("give_up",)), tmp_path, "give-up.json")

    assert main(["inspect", str(path)]) == 0

    assert "outcome:     gave up" in capsys.readouterr().out


def test_run_outcome_keeps_errored_trial_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _step_limit_log(reasons=(None,))
    scene = dataclasses.replace(
        log.samples[0], status="error", error="PolicyError: policy exploded"
    )
    log = dataclasses.replace(
        log,
        status="error",
        results=dataclasses.replace(log.results, errored_trials=1),
        samples=(scene,),
        error="all 1 trial(s) errored; nothing was scored",
    )

    assert _run_with_synthesized_log(log, monkeypatch, tmp_path) == 1

    out = capsys.readouterr().out
    assert "outcome: no reason recorded" in out
    assert "[error] s0: PolicyError: policy exploded" in out
    assert "trials: 1 (1 errored)" in out


def test_unmapped_outcome_degrades_lone_surrogate_in_run_and_inspect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _step_limit_log(reasons=("bad \ud800 reason",))

    assert _run_with_synthesized_log(log, monkeypatch, tmp_path / "run") == 0
    run_out = capsys.readouterr().out
    assert "\ud800" not in run_out
    # encode(errors="replace") substitutes ASCII "?" on the encode side.
    assert "outcome: bad ? reason" in run_out

    path = _write_log(log, tmp_path, "hostile-reason.json")
    assert main(["inspect", str(path)]) == 0
    inspect_out = capsys.readouterr().out
    assert "\ud800" not in inspect_out
    assert "outcome:     bad ? reason" in inspect_out


def test_outcome_is_omitted_without_recorded_reasons_in_run_and_inspect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _transcript_log()

    assert _run_with_synthesized_log(log, monkeypatch, tmp_path / "run") == 0
    assert "outcome:" not in capsys.readouterr().out

    path = _write_log(log, tmp_path, "old-log.json")
    assert main(["inspect", str(path)]) == 0
    assert "outcome:" not in capsys.readouterr().out


def test_inspect_cancelled_status_and_singular_no_reason_outcome_are_aligned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _step_limit_log(reasons=(None,))
    scene = dataclasses.replace(log.samples[0], status="cancelled")
    log = dataclasses.replace(log, status="cancelled", samples=(scene,))
    path = _write_log(log, tmp_path, "cancelled.json")

    assert main(["inspect", str(path)]) == 1

    out = capsys.readouterr().out
    assert "run status:  cancelled\n" in out
    assert "outcome:     no reason recorded\n" in out


def test_inspect_started_status_uses_raw_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = dataclasses.replace(_step_limit_log(reasons=("done",)), status="started")
    path = _write_log(log, tmp_path, "started.json")

    assert main(["inspect", str(path)]) == 1

    assert "run status:  started" in capsys.readouterr().out


def test_inspect_outcome_coerces_non_string_hand_edited_reasons(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _step_limit_log(reasons=("success", "success", "success")).to_dict()
    data["samples"][0]["termination_reasons"] = [3, True, ["nested"]]
    path = tmp_path / "non-string-reasons.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert main(["inspect", str(path)]) == 0

    assert "outcome:     1 3, 1 True, 1 ['nested']" in capsys.readouterr().out


def test_inspect_outcome_folds_empty_reason_into_no_reason_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_step_limit_log(reasons=("",)), tmp_path, "empty-reason.json")

    assert main(["inspect", str(path)]) == 0

    assert "outcome:     no reason recorded" in capsys.readouterr().out


def test_inspect_outcome_merges_phrase_collision_and_uses_degraded_print(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_log(_step_limit_log(reasons=("give_up", "gave up")), tmp_path, "collision.json")
    degraded_lines: list[str] = []
    original_print_degraded = cli._print_degraded

    def record_degraded(line: str) -> None:
        degraded_lines.append(line)
        original_print_degraded(line)

    monkeypatch.setattr(cli, "_print_degraded", record_degraded)

    assert main(["inspect", str(path)]) == 0

    assert "outcome:     2 gave up" in capsys.readouterr().out
    assert "outcome:     2 gave up" in degraded_lines


def test_inspect_transcript_renders_chat_and_unknown_shapes_after_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "transcripts.json"
    path.write_text(json.dumps(_transcript_log().to_dict()), encoding="utf-8")

    assert main(["inspect", str(path), "--transcript"]) == 0

    out = capsys.readouterr().out
    assert out.index("policy transcripts:") > out.index("[success] s0")
    assert "scene s0, trial 0:" in out
    assert "user: look at the workspace\n[image]" in out
    assert '-> move_by({"dx": 0.1})' in out
    assert "moved 2 steps" in out
    assert '  "custom": [' in out


def test_inspect_transcript_keeps_error_status_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "error.json"
    path.write_text(json.dumps(_transcript_log(status="error").to_dict()), encoding="utf-8")
    assert main(["inspect", str(path), "--transcript"]) == 1
    assert "policy transcripts:" in capsys.readouterr().out


def test_inspect_transcript_reports_when_none_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "none.json"
    path.write_text(json.dumps(_step_limit_log(reasons=("success",)).to_dict()), encoding="utf-8")
    assert main(["inspect", str(path), "--transcript"]) == 0
    assert "no policy transcripts recorded" in capsys.readouterr().out


def test_plain_inspect_mentions_recorded_policy_transcripts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "transcripts.json"
    path.write_text(json.dumps(_transcript_log().to_dict()), encoding="utf-8")
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "policy transcripts: recorded (--transcript to print)" in out
    assert f"hint: HTML viewer: inspect-robots view {path}" in out
    assert "scene s0, trial 0:" not in out


def test_inspect_wire_table_reports_attempts_images_and_new_blob_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sha = "a" * 64
    missing_sha = "c" * 64
    reference = f"data:image/png;base64,$blob:{sha}"
    rows: list[dict[str, object]] = [
        {
            "call": 0,
            "attempt": 0,
            "endpoint": "/chat/completions",
            "duration_s": 0.1251,
            "status": 429,
            "request": {
                "messages": [{"content": [reference, reference, f"$blob:{missing_sha}", 3]}]
            },
            "response": {"error": "retry"},
        },
        {
            "call": 0,
            "attempt": 1,
            "endpoint": "/chat/completions",
            "duration_s": True,
            "status": None,
            "request": {"messages": [{"content": reference}]},
            "response": None,
        },
    ]
    path, blob_dir = _write_wire_log(tmp_path, (rows,))
    blob_dir.mkdir(parents=True)
    (blob_dir / f"{sha}.png").write_bytes(b"blob")

    assert main(["inspect", str(path), "--wire"]) == 0

    out = capsys.readouterr().out
    assert "wire calls:" in out
    assert "trial  call  attempt  endpoint  status  duration  images  new blob bytes" in out
    assert "s0-e0  0  0  /chat/completions  429  0.125s  3  4" in out
    assert "s0-e0  0  1  /chat/completions  -  -  1  0" in out


def test_inspect_wire_call_dumps_every_retry_with_symbolic_blobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sha = "b" * 64
    rows: list[dict[str, object]] = [
        {
            "call": 3,
            "attempt": attempt,
            "endpoint": "/responses",
            "duration_s": 0.2,
            "status": status,
            "request": {"input": [{"image_url": f"$blob:{sha}"}]},
            "response": response,
        }
        for attempt, status, response in (
            (0, 500, {"error": "retry"}),
            (1, 200, "raw response"),
        )
    ]
    rows[0]["error"] = "temporary transport detail"
    path, _ = _write_wire_log(tmp_path, (rows,))

    assert main(["inspect", str(path), "--wire", "3"]) == 0

    out = capsys.readouterr().out
    assert "wire trial s0-e0, call 3:" in out
    assert out.count("attempt ") == 2
    assert f"$blob:{sha}" in out
    assert '"error": "retry"' in out
    assert '"raw response"' in out
    assert "endpoint: /responses" in out
    assert "status: 500" in out
    assert "error: temporary transport detail" in out


def test_inspect_wire_call_requires_trial_for_multiple_captures(
    tmp_path: Path,
) -> None:
    row = {
        "call": 0,
        "attempt": 0,
        "endpoint": "/messages",
        "duration_s": 0.1,
        "status": 200,
        "request": {},
        "response": {},
    }
    path, _ = _write_wire_log(tmp_path, ([row], [row]))

    with pytest.raises(
        SystemExit,
        match=r"--trial is required.*available trials: s0-e0, s0-e1",
    ):
        main(["inspect", str(path), "--wire", "0"])


def test_inspect_wire_trial_selects_one_capture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = {
        "call": 0,
        "attempt": 0,
        "endpoint": "/first",
        "duration_s": 0.1,
        "status": 200,
        "request": {},
        "response": {},
    }
    second = {**first, "endpoint": "/second"}
    path, _ = _write_wire_log(tmp_path, ([first], [second]))

    assert main(["inspect", str(path), "--wire", "0", "--trial", "s0-e1"]) == 0

    out = capsys.readouterr().out
    assert "wire trial s0-e1, call 0:" in out
    assert "/first" not in out


def test_inspect_wire_missing_capture_is_non_error_for_table_and_error_for_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(_step_limit_log(reasons=("success",)), tmp_path, "none-wire.json")

    assert main(["inspect", str(path), "--wire"]) == 0
    assert "no wire capture recorded" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="no wire capture recorded"):
        main(["inspect", str(path), "--wire", "0"])


def test_inspect_wire_hostile_pointer_is_treated_as_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = _write_wire_log(tmp_path, ([],), pointer_override="../outside/calls.jsonl")

    assert main(["inspect", str(path), "--wire"]) == 0
    assert "no wire capture recorded" in capsys.readouterr().out


def test_inspect_wire_keeps_rows_before_a_torn_tail_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    row = {"call": 0, "attempt": 0, "endpoint": "/chat/completions", "status": 200}
    path, _ = _write_wire_log(tmp_path, ([row],))
    calls_path = tmp_path / "wire/run/s0-e0/calls.jsonl"
    with calls_path.open("a", encoding="utf-8") as handle:
        handle.write('{"call": 1, "attempt": 0, "sta')

    assert main(["inspect", str(path), "--wire"]) == 0
    out = capsys.readouterr().out
    assert "/chat/completions" in out


def test_inspect_wire_shallow_pointer_is_treated_as_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "calls.jsonl").write_text('{"call": 0}\n', encoding="utf-8")
    path, _ = _write_wire_log(tmp_path, ([],), pointer_override="calls.jsonl")

    assert main(["inspect", str(path), "--wire"]) == 0
    assert "no wire capture recorded" in capsys.readouterr().out


def test_inspect_wire_guides_invalid_trial_call_and_trial_without_dump(
    tmp_path: Path,
) -> None:
    row = {
        "call": 1,
        "attempt": 0,
        "endpoint": "/responses",
        "duration_s": 0.1,
        "status": 200,
        "request": {},
        "response": {},
    }
    path, _ = _write_wire_log(tmp_path, ([row],))

    with pytest.raises(SystemExit, match=r"wire trial 'missing' not found.*s0-e0"):
        main(["inspect", str(path), "--wire", "1", "--trial", "missing"])
    with pytest.raises(SystemExit, match="wire call 2 not found in trial s0-e0"):
        main(["inspect", str(path), "--wire", "2"])
    with pytest.raises(SystemExit, match="--trial requires an integer --wire CALL"):
        main(["inspect", str(path), "--wire", "--trial", "s0-e0"])
    with pytest.raises(SystemExit, match="--trial requires --wire CALL"):
        main(["inspect", str(path), "--trial", "s0-e0"])


@pytest.mark.parametrize("contents", [None, "", "[]\n", "{\n"])
def test_inspect_wire_unreadable_or_invalid_sidecars_are_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str | None,
) -> None:
    path, _ = _write_wire_log(tmp_path, ([{}],))
    calls_path = tmp_path / "wire" / "run" / "s0-e0" / "calls.jsonl"
    if contents is None:
        calls_path.unlink()
    else:
        calls_path.write_text(contents, encoding="utf-8")

    assert main(["inspect", str(path), "--wire"]) == 0
    assert "no wire capture recorded" in capsys.readouterr().out


def test_run_summary_adds_agent_conversation_hint_when_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_run_summary(_transcript_log(), "run.json", is_adhoc=False)
    out = capsys.readouterr().out
    assert "hint: inspect it with: inspect-robots inspect run.json" in out
    assert "hint: HTML viewer: inspect-robots view run.json" in out
    assert "hint: agent conversation: inspect-robots inspect run.json --transcript" in out


def test_run_summary_shows_cancelled_scene_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _transcript_log(status="cancelled")
    cancelled_scene = dataclasses.replace(
        log.samples[0], status="cancelled", error="cancelled by user"
    )
    log = dataclasses.replace(log, samples=(cancelled_scene,))

    cli._print_run_summary(log, "run.json", is_adhoc=False)

    assert "[cancelled] s0: cancelled by user" in capsys.readouterr().out


def test_transcript_rendering_degrades_lone_surrogates_instead_of_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hostile/buggy model server can emit lone UTF-16 surrogates; they
    # survive the log's JSON round-trip (ensure_ascii escapes them on disk)
    # and must degrade at print time, not crash the forensic reader.
    log = _transcript_log()
    hostile = dataclasses.replace(
        log.samples[0],
        policy_transcripts=([{"role": "assistant", "content": "bad \ud800 data"}],),
        epochs=({},),
    )
    path = tmp_path / "hostile.json"
    path.write_text(
        json.dumps(dataclasses.replace(log, samples=(hostile,)).to_dict()), encoding="utf-8"
    )

    assert main(["inspect", str(path), "--transcript"]) == 0

    out = capsys.readouterr().out
    assert "\ud800" not in out
    assert "bad � data" in out or "bad ? data" in out


def test_inspect_shows_shared_instruction_in_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two scenes with the same instruction collapse to ONE header line: the
    # collapse is by value equality across scenes, not by scene count.
    log = _step_limit_log(reasons=("success",))
    first = dataclasses.replace(log.samples[0], instruction="wipe the table")
    second = dataclasses.replace(first, scene_id="s1")
    path = tmp_path / "shared.json"
    path.write_text(
        json.dumps(dataclasses.replace(log, samples=(first, second)).to_dict()),
        encoding="utf-8",
    )

    assert main(["inspect", str(path)]) == 0

    out = capsys.readouterr().out
    assert "instruction: wipe the table\n" in out
    # Run-level identity: directly under the task line, not repeated per scene.
    assert out.index("task:") < out.index("instruction:") < out.index("policy:")
    assert "      instruction:" not in out
    assert out.count("instruction:") == 1


def test_inspect_shows_differing_instructions_per_scene(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _step_limit_log(reasons=("success",))
    first = dataclasses.replace(log.samples[0], instruction="fold the towel")
    second = dataclasses.replace(log.samples[0], scene_id="s1", instruction="stack the cups")
    third = dataclasses.replace(log.samples[0], scene_id="s2")
    path = tmp_path / "differing.json"
    path.write_text(
        json.dumps(dataclasses.replace(log, samples=(first, second, third)).to_dict()),
        encoding="utf-8",
    )

    assert main(["inspect", str(path)]) == 0

    out = capsys.readouterr().out
    assert "      instruction: fold the towel\n" in out
    assert "      instruction: stack the cups\n" in out
    assert out.index("[success] s0") < out.index("      instruction:") < out.index("[success] s1")
    # No shared header line, and the instruction-less scene prints no sub-line.
    assert out.index("instruction:") > out.index("scenes:")
    assert out.count("instruction:") == 2


@pytest.mark.parametrize("instruction", [None, ""])
def test_inspect_stays_silent_without_a_real_instruction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], instruction: str | None
) -> None:
    # Logs written before SceneResult.instruction existed (None) and empty
    # strings must render exactly as today: no blank-value header line.
    log = _step_limit_log(reasons=("success",))
    scene = dataclasses.replace(log.samples[0], instruction=instruction)
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(dataclasses.replace(log, samples=(scene,)).to_dict()), encoding="utf-8"
    )
    assert main(["inspect", str(path)]) == 0
    assert "instruction:" not in capsys.readouterr().out


def test_inspect_degrades_lone_surrogates_in_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Instructions are foreign text and survive the JSON round-trip; a lone
    # UTF-16 surrogate must degrade at print time, not crash the reader.
    log = _step_limit_log(reasons=("success",))
    scene = dataclasses.replace(log.samples[0], instruction="wipe \ud800 the table")
    path = tmp_path / "hostile.json"
    path.write_text(
        json.dumps(dataclasses.replace(log, samples=(scene,)).to_dict()), encoding="utf-8"
    )

    assert main(["inspect", str(path)]) == 0

    out = capsys.readouterr().out
    assert "\ud800" not in out
    assert "wipe � the table" in out or "wipe ? the table" in out


def test_transcript_empty_list_falls_back_to_json_rendering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # all() is vacuously true on [], which must not classify as a chat
    # transcript: the JSON fallback at least prints the empty list.
    log = _transcript_log()
    empty = dataclasses.replace(log.samples[0], policy_transcripts=([],), epochs=({},))
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(dataclasses.replace(log, samples=(empty,)).to_dict()), encoding="utf-8"
    )
    assert main(["inspect", str(path), "--transcript"]) == 0
    assert "    []" in capsys.readouterr().out


def test_chat_renderer_tolerates_malformed_tool_call_entries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._render_chat_transcript(
        [
            "not a message",
            {
                "role": "assistant",
                "tool_calls": [
                    "not a call",
                    {"function": "not a function"},
                    {"function": {"name": "move", "arguments": {"dx": 1}}},
                ],
            },
        ]
    )
    assert '-> move({"dx": 1})' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("control_hz", "parenthetical"),
    [
        (10.0, " (max_steps=1200, ~120s at 10 Hz)"),
        (None, " (max_steps=1200)"),
        (0, " (max_steps=1200)"),
        # bool is an int subclass; a hand-edited log must not print "at 1 Hz".
        (True, " (max_steps=1200)"),
    ],
)
def test_run_summary_surfaces_step_limit_for_adhoc_runs(
    control_hz: object,
    parenthetical: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_run_summary(_step_limit_log(control_hz=control_hz), "run.json", is_adhoc=True)
    out = capsys.readouterr().out
    note = f"note: 1/1 trials hit the step limit before terminating{parenthetical}"
    assert note in out
    assert out.index(note) < out.index("success_at_end")
    assert ("hint: raise it with --max-steps N or: inspect-robots config set max_steps N") in out


def test_run_summary_uses_registered_task_hint(capsys: pytest.CaptureFixture[str]) -> None:
    cli._print_run_summary(_step_limit_log(task="registered-task"), "run.json", is_adhoc=False)
    out = capsys.readouterr().out
    assert "hint: task 'registered-task' defines its own max_steps" in out
    assert "--max-steps N" not in out


def test_run_summary_surfaces_seconds_horizon_for_registered_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _step_limit_log(task="timed-task", max_seconds=120.0, control_hz=10.0)
    cli._print_run_summary(log, "run.json", is_adhoc=False)
    out = capsys.readouterr().out
    assert "(max_seconds=120, resolved max_steps=1200 at 10 Hz)" in out
    assert "hint: task 'timed-task' defines its own max_seconds" in out


def test_run_summary_surfaces_seconds_horizon_without_logged_control_rate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _step_limit_log(task="timed-task", max_seconds=120.0, control_hz=None)
    cli._print_run_summary(log, "run.json", is_adhoc=False)
    out = capsys.readouterr().out
    assert "(max_seconds=120, resolved max_steps=1200)" in out
    assert "resolved max_steps=1200 at" not in out


def test_seconds_horizon_text_tolerates_missing_rate_and_malformed_limits() -> None:
    assert (
        cli._seconds_horizon_text(_step_limit_log(max_seconds=120.0, control_hz=None))
        == "120s -> 1200 steps"
    )
    assert cli._seconds_horizon_text(_step_limit_log(max_seconds=0.0)) is None
    assert cli._seconds_horizon_text(_step_limit_log(max_seconds=120.0, max_steps=None)) is None


def test_run_summary_omits_step_limit_note_without_truncation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_run_summary(_step_limit_log(reasons=("success",)), "run.json", is_adhoc=True)
    out = capsys.readouterr().out
    assert "step limit" not in out
    assert "raise it" not in out


def test_run_summary_omits_parenthetical_without_horizon(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_run_summary(_step_limit_log(max_steps=None), "run.json", is_adhoc=True)
    out = capsys.readouterr().out
    assert "note: 1/1 trials hit the step limit before terminating\n" in out
    assert "max_steps=None" not in out


def test_inspect_surfaces_step_limit_note_hint_and_scene_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _step_limit_log(reasons=("max_steps", "success"), control_hz=None)
    path = tmp_path / "timeout.json"
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")

    assert main(["inspect", str(path)]) == 0

    out = capsys.readouterr().out
    assert out.startswith(
        "note: 1/2 trials hit the step limit before terminating (max_steps=1200)\n"
    )
    assert "hint: raise it with --max-steps N" in out
    assert "[success] s0: (1/2 trials hit max_steps)" in out


def test_inspect_tolerates_non_numeric_max_steps_in_hand_edited_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # from_dict does no field validation, so a hand-edited log can carry a
    # string horizon; the note must degrade to no parenthetical, not crash.
    data = _step_limit_log().to_dict()
    data["eval"]["max_steps"] = "1200"
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert main(["inspect", str(path)]) == 0

    out = capsys.readouterr().out
    assert "note: 1/1 trials hit the step limit before terminating\n" in out
    assert "max_steps=1200" not in out


def test_inspect_prints_seconds_horizon(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_log(_step_limit_log(max_seconds=120.0, control_hz=10.0), tmp_path, "timed.json")
    assert main(["inspect", str(path)]) == 0
    assert "horizon:     120s -> 1200 steps at 10 Hz" in capsys.readouterr().out


def test_bare_instruction_runs_adhoc_task_from_env_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--scorer", "success_at_end", "--log-dir", str(log_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"policy: scripted (from ${ENV_POLICY})" in out
    assert f"embodiment: cubepick (from ${ENV_EMBODIMENT})" in out
    log = _read_only_log(log_dir)
    assert log.eval.task == "adhoc"
    assert log.samples[0].instruction == "reach the cube"
    assert log.results.metrics["success_at_end"] == 1.0


def test_config_file_supplies_defaults_and_adhoc_settings(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "max_steps = 50\n"
        "[policy.args]\n"
        "chunk_size = 6\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--log-dir", str(log_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"policy: scripted (from {config})" in out
    log = _read_only_log(log_dir)
    assert log.samples[0].instruction == "reach the cube"
    # The config's scorer (not the "operator" fallback) actually scored the run.
    assert log.results.metrics == {"success_at_end": 1.0}
    # [policy.args] chunk_size=6 reached the policy constructor (recorded in
    # the log as the policy's action_horizon).
    assert log.eval.policy_config["action_horizon"] == 6


def test_config_max_steps_truncates_adhoc_run(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "max_steps = 3\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--log-dir", str(log_dir)])
    assert rc == 0  # truncation is not an error; the eval itself succeeded
    # Three steps cannot reach the cube: the config horizon (not the 300
    # fallback, which would succeed) governed the rollout.
    assert _read_only_log(log_dir).results.metrics["success_at_end"] == 0.0
    capsys.readouterr()


def test_cli_flags_beat_config_defaults_and_args(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = noop\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "[policy.args]\n"
        "chunk_size = 6\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(
        [
            "run",
            "--instruction",
            "reach the cube",
            "--policy",
            "scripted",
            "-P",
            "chunk_size=4",
            "--log-dir",
            str(log_dir),
        ]
    )
    assert rc == 0
    assert "policy: scripted (--policy)" in capsys.readouterr().out
    # The -P flag's chunk_size=4 overrode the same-named [policy.args] key.
    assert _read_only_log(log_dir).eval.policy_config["action_horizon"] == 4


def test_config_args_do_not_leak_to_explicitly_selected_components(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Issue #44: the [*.args] sections belong to the configured defaults; before
    # the fix they followed *whatever* was selected, so a persisted rig arg
    # (rest_pose here) TypeErrored an unrelated --embodiment. Now the run is
    # green and each dropped section is noted on stderr.
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = noop\n"
        "embodiment = yam-arms\n"
        "scorer = success_at_end\n"
        "[policy.args]\n"
        "bogus_policy_knob = 1\n"
        "[embodiment.args]\n"
        "rest_pose = 0.5\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(
        [
            "run",
            "--instruction",
            "reach the cube",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "--log-dir",
            str(log_dir),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "note: ignoring [policy.args] for 'scripted': they apply to 'noop'" in err
    assert "note: ignoring [embodiment.args] for 'cubepick': they apply to 'yam-arms'" in err
    assert _read_only_log(log_dir).results.metrics["success_at_end"] == 1.0


def test_config_args_apply_when_flag_names_the_configured_default(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Selecting the configured default *explicitly* must keep its args: a lab
    # operator typing --policy for the component the file already names cannot
    # silently lose the configured calibration (gate on name, not provenance).
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "[policy.args]\n"
        "chunk_size = 6\n",
    )
    log_dir = tmp_path / "logs"
    args = ["run", "--instruction", "reach the cube", "--policy", "scripted"]
    rc = main([*args, "--log-dir", str(log_dir)])
    assert rc == 0
    assert "note: ignoring" not in capsys.readouterr().err
    assert _read_only_log(log_dir).eval.policy_config["action_horizon"] == 6


def test_env_selected_component_does_not_inherit_config_args(
    _hermetic_defaults: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An env var swaps the component name but the file's args stay owned by the
    # file's name — they must not follow the env-selected component.
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = noop\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "[policy.args]\n"
        "bogus_policy_knob = 1\n",
    )
    monkeypatch.setenv(ENV_POLICY, "scripted")
    log_dir = tmp_path / "logs"
    rc = main(["run", "--instruction", "reach the cube", "--log-dir", str(log_dir)])
    assert rc == 0
    assert "note: ignoring [policy.args] for 'scripted': they apply to 'noop'" in (
        capsys.readouterr().err
    )


def test_mistyped_subcommand_never_starts_a_rollout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Even with defaults fully configured, a single-token typo (no interior
    # whitespace) and a whitespace-padded subcommand both error out.
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    for argv in (["isnpect"], ["runs"], [" list "]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2  # argparse invalid-choice, not a run
    capsys.readouterr()


def test_run_requires_exactly_one_of_task_or_instruction() -> None:
    with pytest.raises(SystemExit, match="not both"):
        main(["run", "--task", "cubepick-reach", "--instruction", "reach it"])
    with pytest.raises(SystemExit, match="--task name or an --instruction"):
        main(["run", "--policy", "scripted", "--embodiment", "cubepick"])


def test_adhoc_only_flags_rejected_with_task() -> None:
    base = ["run", "--task", "cubepick-reach", "--policy", "scripted", "--embodiment", "cubepick"]
    with pytest.raises(SystemExit, match="--max-steps only applies"):
        main([*base, "--max-steps", "10"])
    with pytest.raises(SystemExit, match="--scorer only applies"):
        main([*base, "--scorer", "operator"])


def test_task_args_rejected_with_instruction() -> None:
    with pytest.raises(SystemExit, match="-T only applies"):
        main(["run", "--instruction", "reach it", "-T", "num_scenes=1"])


def test_missing_defaults_error_lists_registered_components() -> None:
    with pytest.raises(SystemExit, match=r"registered policies: .*scripted") as excinfo:
        main(["run", "--instruction", "reach the cube"])
    assert ENV_POLICY in str(excinfo.value)  # the message shows the remedies


def test_unknown_scorer_name_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    with pytest.raises(SystemExit, match="no scorer named 'nope'"):
        main(["run", "--instruction", "reach it", "--scorer", "nope"])


def test_adhoc_flags_override_config(
    _hermetic_defaults: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\nscorer = operator\nmax_steps = 7\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(
        [
            "run",
            "--instruction",
            "reach the cube",
            "--scorer",
            "success_at_end",
            "--max-steps",
            "60",
            "--log-dir",
            str(log_dir),
        ]
    )
    assert rc == 0
    log = _read_only_log(log_dir)
    # config max_steps=7 would truncate before success; the flag's 60 won.
    assert log.results.metrics["success_at_end"] == 1.0
    assert "operator" not in log.results.metrics  # --scorer replaced the config scorer
    capsys.readouterr()


def _tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)


@pytest.mark.parametrize(
    (
        "embodiment_kind",
        "accepts_messages",
        "platform",
        "console_on",
        "expected_output",
        "expected_connect_calls",
        "expected_defer_calls",
    ),
    [
        ("new", True, "linux", True, f"{USAGE}\n", 1, 0),
        ("new", False, "linux", True, f"{USAGE_END_ONLY}\n", 1, 0),
        (
            "new",
            True,
            "win32",
            False,
            "operator console unavailable: select cannot watch stdin on Windows; "
            "feedback typing stays off\n",
            0,
            0,
        ),
        (
            "new",
            False,
            "win32",
            False,
            "operator console unavailable: select cannot watch stdin on Windows; "
            "feedback typing stays off\n",
            0,
            0,
        ),
        ("defer", True, "linux", True, f"{USAGE}\n", 0, 1),
        ("defer", False, "win32", False, "", 0, 0),
        ("simulated", True, "linux", True, f"{USAGE}\n", 0, 0),
        ("simulated", False, "linux", False, "", 0, 0),
        (
            "bare",
            True,
            "linux",
            False,
            "operator console unavailable: this embodiment predates the operator console "
            "and still owns the end-of-episode keypress; feedback typing stays off\n",
            0,
            0,
        ),
        ("bare", False, "win32", False, "", 0, 0),
    ],
)
def test_build_operator_session_enablement_matrix(
    embodiment_kind: str,
    accepts_messages: bool,
    platform: str,
    console_on: bool,
    expected_output: str,
    expected_connect_calls: int,
    expected_defer_calls: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from inspect_robots.mock import CubePickEmbodiment

    class NewHookEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.connect_calls = 0
            self.connected_session: OperatorSession | None = None

        def connect_operator_session(self, session: OperatorSession) -> None:
            self.connect_calls += 1
            self.connected_session = session

    class DeferredEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = dataclasses.replace(self.info, is_simulated=False)
            self.defer_calls = 0

        def defer_operator_end(self) -> None:
            self.defer_calls += 1

    class BareEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = dataclasses.replace(self.info, is_simulated=False)

    policy = type("PolicyStub", (), {"accepts_operator_messages": accepts_messages})()
    embodiment: CubePickEmbodiment
    if embodiment_kind == "new":
        embodiment = NewHookEmbodiment()
    elif embodiment_kind == "defer":
        embodiment = DeferredEmbodiment()
    elif embodiment_kind == "simulated":
        embodiment = CubePickEmbodiment()
    else:
        embodiment = BareEmbodiment()
    monkeypatch.setattr(sys, "platform", platform)

    session, operator_input = cli._build_operator_session(policy, embodiment)

    assert isinstance(session, OperatorSession)
    assert (operator_input is session) is console_on
    assert getattr(embodiment, "connect_calls", 0) == expected_connect_calls
    assert getattr(embodiment, "defer_calls", 0) == expected_defer_calls
    if expected_connect_calls:
        assert getattr(embodiment, "connected_session", None) is session
    assert capsys.readouterr().out == expected_output


def test_new_session_hook_runs_before_eval_and_owns_prompt_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots
    from inspect_robots.mock import CubePickEmbodiment

    order: list[str] = []
    connected: list[OperatorSession] = []

    class SessionAwareEmbodiment(CubePickEmbodiment):
        def connect_operator_session(self, session: OperatorSession) -> None:
            order.append("connect")
            connected.append(session)

    embodiment_name = "session-aware-for-order-test"
    reg.embodiment(embodiment_name)(SessionAwareEmbodiment)

    def fake_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args
        order.append("eval")
        operator_input = kwargs.get("operator_input")
        before_scoring = kwargs.get("before_scoring")
        assert operator_input is connected[0]
        assert getattr(before_scoring, "__self__", None) is operator_input
        return [_step_limit_log(reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval", fake_eval)
    monkeypatch.setattr(sys, "platform", "linux")
    _tty_stdin(monkeypatch)
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                embodiment_name,
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(embodiment_name, None)

    assert rc == 0
    assert order == ["connect", "eval"]
    assert capsys.readouterr().out.count(USAGE_END_ONLY) == 1


def test_attended_opted_in_run_builds_console_calls_defer_and_prints_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots
    from inspect_robots.mock import CubePickEmbodiment

    class _DeferredHardware(CubePickEmbodiment):
        hook_calls: ClassVar[int] = 0

        def __init__(self) -> None:
            super().__init__()
            self.info = dataclasses.replace(self.info, is_simulated=False)

        def defer_operator_end(self) -> None:
            type(self).hook_calls += 1

    policy_name = "console-policy-for-run-test"
    embodiment_name = "deferred-hardware-for-console-test"
    reg.policy(policy_name)(_ConsolePolicy)
    reg.embodiment(embodiment_name)(_DeferredHardware)
    operator_inputs: list[object] = []

    def fake_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args
        operator_inputs.append(kwargs.get("operator_input"))
        return [_step_limit_log(reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval", fake_eval)
    monkeypatch.setattr(sys, "platform", "linux")
    _tty_stdin(monkeypatch)
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                policy_name,
                "--embodiment",
                embodiment_name,
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["policy"].pop(policy_name, None)
        reg._FACTORIES["embodiment"].pop(embodiment_name, None)

    assert rc == 0
    assert len(operator_inputs) == 1
    assert isinstance(operator_inputs[0], OperatorSession)
    assert _DeferredHardware.hook_calls == 1
    assert capsys.readouterr().out.count(USAGE) == 1


def test_attended_opted_in_eval_set_uses_simulated_console_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots

    policy_name = "console-policy-for-eval-set-test"
    reg.policy(policy_name)(_ConsolePolicy)
    operator_inputs: list[object] = []

    def fake_eval_set(*args: object, **kwargs: object) -> tuple[bool, list[EvalLog]]:
        del args
        operator_inputs.append(kwargs.get("operator_input"))
        return True, [_step_limit_log(task="cubepick-reach", reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval_set", fake_eval_set)
    monkeypatch.setattr(sys, "platform", "darwin")
    _tty_stdin(monkeypatch)
    try:
        rc = main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                policy_name,
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    finally:
        reg._FACTORIES["policy"].pop(policy_name, None)

    assert rc == 0
    assert len(operator_inputs) == 1
    assert isinstance(operator_inputs[0], OperatorSession)
    assert capsys.readouterr().out.count(USAGE) == 1


def test_attended_console_stays_off_for_legacy_real_hardware(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots
    from inspect_robots.mock import CubePickEmbodiment

    class _LegacyHardware(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = dataclasses.replace(self.info, is_simulated=False)

    policy_name = "console-policy-for-legacy-hardware-test"
    embodiment_name = "legacy-hardware-for-console-test"
    reg.policy(policy_name)(_ConsolePolicy)
    reg.embodiment(embodiment_name)(_LegacyHardware)
    operator_inputs: list[object] = []

    def fake_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args
        operator_inputs.append(kwargs.get("operator_input"))
        return [_step_limit_log(reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval", fake_eval)
    monkeypatch.setattr(sys, "platform", "linux")
    _tty_stdin(monkeypatch)
    try:
        assert (
            main(
                [
                    "run",
                    "--task",
                    "cubepick-reach",
                    "--policy",
                    policy_name,
                    "--embodiment",
                    embodiment_name,
                    "--log-dir",
                    str(tmp_path),
                ]
            )
            == 0
        )
    finally:
        reg._FACTORIES["policy"].pop(policy_name, None)
        reg._FACTORIES["embodiment"].pop(embodiment_name, None)

    assert operator_inputs == [None]
    out = capsys.readouterr().out
    assert out.count("predates the operator console") == 1
    assert "still owns the end-of-episode keypress" in out
    assert "feedback typing stays off" in out
    assert USAGE not in out


def test_attended_console_stays_off_on_windows_before_calling_defer_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots
    from inspect_robots.mock import CubePickEmbodiment

    class _DeferredHardware(CubePickEmbodiment):
        hook_calls: ClassVar[int] = 0

        def __init__(self) -> None:
            super().__init__()
            self.info = dataclasses.replace(self.info, is_simulated=False)

        def defer_operator_end(self) -> None:
            type(self).hook_calls += 1

    policy_name = "console-policy-for-windows-test"
    embodiment_name = "deferred-hardware-for-windows-test"
    reg.policy(policy_name)(_ConsolePolicy)
    reg.embodiment(embodiment_name)(_DeferredHardware)
    operator_inputs: list[object] = []

    def fake_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args
        operator_inputs.append(kwargs.get("operator_input"))
        return [_step_limit_log(reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval", fake_eval)
    monkeypatch.setattr(sys, "platform", "win32")
    _tty_stdin(monkeypatch)
    try:
        assert (
            main(
                [
                    "run",
                    "--task",
                    "cubepick-reach",
                    "--policy",
                    policy_name,
                    "--embodiment",
                    embodiment_name,
                    "--log-dir",
                    str(tmp_path),
                ]
            )
            == 0
        )
    finally:
        reg._FACTORIES["policy"].pop(policy_name, None)
        reg._FACTORIES["embodiment"].pop(embodiment_name, None)

    assert operator_inputs == [None]
    assert _DeferredHardware.hook_calls == 0
    out = capsys.readouterr().out
    assert out.count("select cannot watch stdin on Windows") == 1
    assert USAGE not in out


@pytest.mark.parametrize(("tty", "extra_args"), [(False, ()), (True, ("--no-prompt",))])
def test_operator_console_requires_attendance(
    tty: bool,
    extra_args: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots

    policy_name = f"console-policy-unattended-{tty}"
    reg.policy(policy_name)(_ConsolePolicy)
    operator_inputs: list[object] = []

    def fake_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args
        operator_inputs.append(kwargs.get("operator_input"))
        return [_step_limit_log(reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval", fake_eval)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    try:
        assert (
            main(
                [
                    "run",
                    "--task",
                    "cubepick-reach",
                    "--policy",
                    policy_name,
                    "--embodiment",
                    "cubepick",
                    "--log-dir",
                    str(tmp_path),
                    *extra_args,
                ]
            )
            == 0
        )
    finally:
        reg._FACTORIES["policy"].pop(policy_name, None)

    assert operator_inputs == [None]
    assert USAGE not in capsys.readouterr().out


def test_policy_without_opt_in_skips_console_silently_before_platform_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect_robots

    operator_inputs: list[object] = []

    def fake_eval(*args: object, **kwargs: object) -> list[EvalLog]:
        del args
        operator_inputs.append(kwargs.get("operator_input"))
        return [_step_limit_log(reasons=("success",))]

    monkeypatch.setattr(inspect_robots, "eval", fake_eval)
    monkeypatch.setattr(sys, "platform", "win32")
    _tty_stdin(monkeypatch)

    assert (
        main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert operator_inputs == [None]
    out = capsys.readouterr().out
    assert "operator console" not in out
    assert USAGE not in out


def test_operator_prompt_records_verdict_and_reprompts_on_typos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    _tty_stdin(monkeypatch)
    answers = iter(["yse", "y", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    log_dir = tmp_path / "logs"
    rc = main(
        ["reach the cube", "--max-steps", "3", "--log-dir", str(log_dir)]
    )  # default scorer: operator
    assert rc == 0
    assert "unrecognized answer 'yse'" in capsys.readouterr().out
    log = _read_only_log(log_dir)
    assert log.samples[0].operator_judgements == ("y",)
    assert log.results.metrics["operator"] == 1.0


def test_operator_prompt_persists_grader_note_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    _tty_stdin(monkeypatch)
    answers = iter(["n", "  Gripper Closed Early  "])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    log_dir = tmp_path / "logs"

    rc = main(["reach the cube", "--max-steps", "3", "--log-dir", str(log_dir)])

    assert rc == 0
    log = _read_only_log(log_dir)
    assert log.samples[0].operator_judgements == ("n",)
    assert log.samples[0].operator_notes == ("Gripper Closed Early",)
    capsys.readouterr()


def test_operator_prompt_adopts_self_confirming_embodiment_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    _tty_stdin(monkeypatch)
    prompts: list[str] = []

    def _answer(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", _answer)
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--log-dir", str(log_dir)])
    assert rc == 0
    assert prompts == []
    assert "operator verdict adopted from embodiment: success" in capsys.readouterr().out
    log = _read_only_log(log_dir)
    assert log.samples[0].operator_judgements == ("y",)
    assert log.results.metrics["operator"] == 1.0


def test_operator_prompt_suppressed_without_tty_or_with_no_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("operator prompt must not fire")
    )

    # Non-TTY stdin (the pytest default): never prompts.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    log_dir_a = tmp_path / "a"
    assert main(["reach the cube", "--log-dir", str(log_dir_a)]) == 0
    assert _read_only_log(log_dir_a).samples[0].operator_judgements == (None,)

    # TTY but --no-prompt: never prompts either.
    _tty_stdin(monkeypatch)
    log_dir_b = tmp_path / "b"
    assert main(["reach the cube", "--no-prompt", "--log-dir", str(log_dir_b)]) == 0
    log = _read_only_log(log_dir_b)
    assert log.samples[0].operator_judgements == (None,)
    assert log.results.metrics["operator"] == 0.0  # unjudged scores honestly as failure
    capsys.readouterr()


def test_registered_task_stays_silent_unless_operator_ends_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_robots.registry import task as task_decorator
    from inspect_robots.scene import Scene
    from inspect_robots.scorer import operator_scorer
    from inspect_robots.task import Task

    @task_decorator("operator-task-for-test")
    def _operator_task() -> Task:
        return Task(
            name="operator-task-for-test",
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            scorer=operator_scorer(),
            max_steps=40,
        )

    _tty_stdin(monkeypatch)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("R6: --task runs prompt only for operator_end trials"),
    )
    log_dir = tmp_path / "logs"
    try:
        rc = main(
            [
                "run",
                "--task",
                "operator-task-for-test",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(log_dir),
            ]
        )
    finally:
        # Don't leak the ad-hoc registration into later tests' registry views.
        reg._FACTORIES["task"].pop("operator-task-for-test", None)
    assert rc == 0
    assert _read_only_log(log_dir).samples[0].operator_judgements == (None,)
    capsys.readouterr()


def test_registered_task_prompts_when_operator_ends_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace as dc_replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.types import OPERATOR_END, Action, StepResult

    class _OperatorEndsEmbodiment(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:  # every step: human ends the episode
            result = super().step(action)
            return dc_replace(
                result, terminated=True, truncated=False, termination_reason=OPERATOR_END
            )

    name = "operator-ends-cubepick-for-test"
    embodiment_decorator(name)(_OperatorEndsEmbodiment)
    _tty_stdin(monkeypatch)
    answers = iter(["y", "smooth grasp"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    log_dir = tmp_path / "logs"
    try:
        rc = main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "-T",
                "num_scenes=1",
                "--policy",
                "scripted",
                "--embodiment",
                name,
                "--log-dir",
                str(log_dir),
            ]
        )
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    log = _read_only_log(log_dir)
    assert log.samples[0].operator_judgements == ("y",)
    assert log.samples[0].operator_notes == ("smooth grasp",)
    capsys.readouterr()


@pytest.mark.parametrize("suppress", ["no_tty", "no_prompt_flag"])
def test_registered_task_operator_end_suppressed_without_tty_or_with_no_prompt(
    suppress: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The run-command twin of the eval-set suppression tests: the gate is
    # evaluated per command, so each command needs its own negative coverage.
    from dataclasses import replace as dc_replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.types import OPERATOR_END, Action, StepResult

    class _OperatorEndsEmbodiment(CubePickEmbodiment):
        def step(self, action: Action) -> StepResult:  # every step: human ends the episode
            result = super().step(action)
            return dc_replace(
                result, terminated=True, truncated=False, termination_reason=OPERATOR_END
            )

    name = "operator-ends-cubepick-suppressed-for-test"
    embodiment_decorator(name)(_OperatorEndsEmbodiment)
    argv = [
        "run",
        "--task",
        "cubepick-reach",
        "-T",
        "num_scenes=1",
        "--policy",
        "scripted",
        "--embodiment",
        name,
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    if suppress == "no_tty":
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    else:
        _tty_stdin(monkeypatch)
        argv.insert(1, "--no-prompt")
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("suppressed run must not prompt")
    )
    try:
        rc = main(argv)
    finally:
        reg._FACTORIES["embodiment"].pop(name, None)
    assert rc == 0
    assert _read_only_log(tmp_path / "logs").samples[0].operator_judgements == (None,)
    capsys.readouterr()


def test_outcome_phrase_maps_operator_end() -> None:
    from inspect_robots.cli import _OUTCOME_PHRASES

    assert _OUTCOME_PHRASES["operator_end"] == "ended by operator"


# --------------------------------------------------------------------------- #
# --sim (plan 0006): swap the default embodiment for its sim counterpart.
# --------------------------------------------------------------------------- #
_SIM_SWAP_CONFIG = (
    "[defaults]\n"
    "policy = scripted\n"
    "embodiment = missing-real-arm\n"  # would explode if ever resolved
    "sim_embodiment = cubepick\n"
    "scorer = success_at_end\n"
    "[embodiment.args]\n"
    "port = 1\n"  # real-rig arg cubepick would reject
)


def test_sim_flag_swaps_embodiment_and_is_load_bearing(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_config(_hermetic_defaults, _SIM_SWAP_CONFIG)
    log_dir = tmp_path / "logs"

    # With --sim: runs green on cubepick — the real default (which does not
    # exist) was never resolved, and [embodiment.args] port=1 never leaked
    # into the sim constructor (it would TypeError).
    rc = main(["reach the cube", "--sim", "--log-dir", str(log_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"embodiment: cubepick (--sim, from {config})" in out
    assert _read_only_log(log_dir).results.metrics["success_at_end"] == 1.0

    # Without --sim the same command dies resolving the real default,
    # proving --sim was load-bearing.
    with pytest.raises(SystemExit, match="no embodiment named 'missing-real-arm'"):
        main(["reach the cube", "--log-dir", str(log_dir)])


def test_sim_embodiment_args_reach_constructor_and_e_flag_overrides(
    _hermetic_defaults: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "sim_embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "[sim_embodiment.args]\n"
        "max_step = 0.001\n",  # crawls: cube is >=0.5 away, 300 steps max
    )
    log_dir_a = tmp_path / "a"
    assert main(["reach the cube", "--sim", "--log-dir", str(log_dir_a)]) == 0
    assert _read_only_log(log_dir_a).results.metrics["success_at_end"] == 0.0

    # An explicit -E overrides the same-named [sim_embodiment.args] key.
    log_dir_b = tmp_path / "b"
    assert main(["reach the cube", "--sim", "-E", "max_step=0.1", "--log-dir", str(log_dir_b)]) == 0
    assert _read_only_log(log_dir_b).results.metrics["success_at_end"] == 1.0
    capsys.readouterr()


def test_env_selected_sim_drops_args_owned_by_configured_sim(
    _hermetic_defaults: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "sim_embodiment = other-sim\n"
        "scorer = success_at_end\n"
        "[sim_embodiment.args]\n"
        "bogus_sim_knob = 1\n",
    )
    monkeypatch.setenv(ENV_SIM_EMBODIMENT, "cubepick")
    log_dir = tmp_path / "logs"

    assert main(["reach the cube", "--sim", "--log-dir", str(log_dir)]) == 0
    assert (
        "note: ignoring [sim_embodiment.args] for 'cubepick': they apply to 'other-sim'"
    ) in capsys.readouterr().err
    assert _read_only_log(log_dir).results.metrics["success_at_end"] == 1.0


def test_sim_conflicts_with_explicit_embodiment() -> None:
    with pytest.raises(SystemExit, match="drop one"):
        main(["run", "--instruction", "reach it", "--sim", "--embodiment", "cubepick"])


def test_sim_without_configuration_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    with pytest.raises(SystemExit, match="no sim embodiment configured") as excinfo:
        main(["run", "--instruction", "reach it", "--sim"])
    message = str(excinfo.value)
    assert ENV_SIM_EMBODIMENT in message and "config set sim_embodiment NAME" in message


def test_sim_works_with_registered_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_SIM_EMBODIMENT, "cubepick")
    log_dir = tmp_path / "logs"
    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--sim",
            "--log-dir",
            str(log_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # env beats config for the sim chain, and the header says so.
    assert f"embodiment: cubepick (--sim, from ${ENV_SIM_EMBODIMENT})" in out
    assert "run status: completed" in out


def test_sim_ignores_real_embodiment_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A persistent real-embodiment env var must not break (or leak into)
    # --sim runs: the sim chain simply doesn't consult it.
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "bogus-real-arm")
    monkeypatch.setenv(ENV_SIM_EMBODIMENT, "cubepick")
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--sim", "--scorer", "success_at_end", "--log-dir", str(log_dir)])
    assert rc == 0
    assert "embodiment: cubepick" in capsys.readouterr().out


def test_cli_run_closes_the_embodiment_it_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI resolves the embodiment itself (so eval() does not own it); it
    must close what it opened — real-hardware embodiments release motor torque
    in close(), and skipping it leaves arms energized after the run."""
    from inspect_robots.mock import CubePickEmbodiment

    closed: list[bool] = []

    class _Tracked(CubePickEmbodiment):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setitem(reg._FACTORIES["embodiment"], "tracked-cubepick", _Tracked)
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "tracked-cubepick")
    rc = main(["reach the cube", "--scorer", "success_at_end", "--log-dir", str(tmp_path / "logs")])
    assert rc == 0
    assert closed == [True]


def test_cli_run_closes_embodiment_when_validation_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure between resolving the embodiment and eval() (here: --epochs 0
    raising ConfigError) must still close the embodiment — otherwise a bad flag
    leaves real arms energized.

    The fix converts the raw ConfigError into a guided SystemExit; the
    important invariant is that close() still fires via the outer try/finally.
    """
    from inspect_robots.mock import CubePickEmbodiment

    closed: list[bool] = []

    class _Tracked(CubePickEmbodiment):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setitem(reg._FACTORIES["embodiment"], "tracked-cubepick", _Tracked)
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "tracked-cubepick")
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "reach the cube",
                "--scorer",
                "success_at_end",
                "--epochs",
                "0",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
    assert "--epochs" in str(excinfo.value)
    assert closed == [True]


def test_config_store_frames_enables_frame_capture(
    _hermetic_defaults: Path, tmp_path: Path
) -> None:
    """store_frames = true in the config file captures frames with no CLI flag."""
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\n"
        "scorer = success_at_end\nstore_frames = true\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--log-dir", str(log_dir)])
    assert rc == 0
    assert list((log_dir / "frames").rglob("*.npy"))
    assert _read_only_log(log_dir).stats.frames_dir is not None


def test_no_store_frames_flag_overrides_config_default(
    _hermetic_defaults: Path, tmp_path: Path
) -> None:
    """--no-store-frames must win over store_frames = true in the config file."""
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\n"
        "scorer = success_at_end\nstore_frames = true\n",
    )
    log_dir = tmp_path / "logs"
    rc = main(["reach the cube", "--no-store-frames", "--log-dir", str(log_dir)])
    assert rc == 0
    assert not (log_dir / "frames").exists()
    assert _read_only_log(log_dir).stats.frames_dir is None


class _FakeRerunSink:
    """Stands in for RerunSink: records construction and step traffic."""

    instances: ClassVar[list[_FakeRerunSink]] = []

    def __init__(
        self,
        recording_path: str | None = None,
        *,
        spawn: bool = False,
        spawn_port: int = 9876,
        connect_url: str | None = None,
    ) -> None:
        self.spawn = spawn
        self.spawn_port = spawn_port
        self.connect_url = connect_url
        self.steps = 0
        _FakeRerunSink.instances.append(self)

    def on_eval_start(self, spec: object) -> None: ...

    def on_trial_start(self, scene_id: str, epoch: int) -> None: ...

    def log_step(self, t: int, observation: object, action: object, result: object) -> None:
        self.steps += 1

    def on_trial_end(self, record: object) -> None: ...

    def on_eval_end(self, log: object) -> None: ...


@pytest.fixture()
def _fake_rerun(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRerunSink]:
    import inspect_robots.logging.rerun_sink as rrs

    _FakeRerunSink.instances = []
    monkeypatch.setattr(rrs, "RerunSink", _FakeRerunSink)
    return _FakeRerunSink


def _run_adhoc(config_home: Path, tmp_path: Path, *extra: str) -> int:
    _write_config(
        config_home,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\n"
        "scorer = success_at_end\nrerun = true\n",
    )
    return main(["reach the cube", "--log-dir", str(tmp_path / "logs"), *extra])


def test_config_rerun_attaches_live_viewer_sink(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    assert _run_adhoc(_hermetic_defaults, tmp_path) == 0
    (sink,) = _fake_rerun.instances  # constructed exactly once
    assert sink.spawn is True  # live viewer, not just a recording
    assert sink.spawn_port == 9876
    assert sink.steps > 0  # actually received rollout traffic


def test_no_rerun_flag_overrides_config(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    assert _run_adhoc(_hermetic_defaults, tmp_path, "--no-rerun") == 0
    assert _fake_rerun.instances == []


def test_rerun_flag_enables_without_config(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\nscorer = success_at_end\n",
    )
    rc = main(["reach the cube", "--log-dir", str(tmp_path / "logs"), "--rerun"])
    assert rc == 0
    assert len(_fake_rerun.instances) == 1


def test_bare_rerun_connect_uses_default_url(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    """A bare --rerun-connect streams to the documented localhost URL."""
    assert _run_adhoc(_hermetic_defaults, tmp_path, "--rerun-connect") == 0
    (sink,) = _fake_rerun.instances
    assert sink.connect_url == cli.DEFAULT_RERUN_CONNECT_URL
    assert sink.spawn is False


def test_rerun_connect_honors_explicit_url(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    """An explicit --rerun-connect URL is passed through to RerunSink."""
    url = "rerun+http://viewer.example:9988/proxy"
    assert _run_adhoc(_hermetic_defaults, tmp_path, "--rerun-connect", url) == 0
    (sink,) = _fake_rerun.instances
    assert sink.connect_url == url


def test_rerun_connect_takes_precedence_over_rerun(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    """Remote connection wins when both --rerun modes are requested."""
    assert _run_adhoc(_hermetic_defaults, tmp_path, "--rerun", "--rerun-connect") == 0
    (sink,) = _fake_rerun.instances
    assert sink.connect_url == cli.DEFAULT_RERUN_CONNECT_URL
    assert sink.spawn is False


def test_rerun_port_flag_spawns_viewer_on_that_port(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\nscorer = success_at_end\n",
    )
    rc = main(
        [
            "reach the cube",
            "--log-dir",
            str(tmp_path / "logs"),
            "--rerun-port",
            "9877",
        ]
    )
    assert rc == 0
    (sink,) = _fake_rerun.instances
    assert sink.spawn is True
    assert sink.spawn_port == 9877


def test_config_rerun_port_reaches_spawned_viewer(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\n"
        "scorer = success_at_end\nrerun = true\nrerun_port = 9878\n",
    )
    assert main(["reach the cube", "--log-dir", str(tmp_path / "logs")]) == 0
    (sink,) = _fake_rerun.instances
    assert sink.spawn is True
    assert sink.spawn_port == 9878


def test_rerun_port_flag_beats_config(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\n"
        "scorer = success_at_end\nrerun_port = 9878\n",
    )
    rc = main(
        [
            "reach the cube",
            "--log-dir",
            str(tmp_path / "logs"),
            "--rerun-port",
            "9879",
        ]
    )
    assert rc == 0
    (sink,) = _fake_rerun.instances
    assert sink.spawn_port == 9879


def test_config_rerun_port_alone_does_not_enable_viewer(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\npolicy = scripted\nembodiment = cubepick\n"
        "scorer = success_at_end\nrerun_port = 9878\n",
    )
    assert main(["reach the cube", "--log-dir", str(tmp_path / "logs")]) == 0
    assert _fake_rerun.instances == []


def test_rerun_port_conflicts_with_rerun_connect(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    with pytest.raises(SystemExit, match=r"--rerun-port.*--rerun-connect"):
        _run_adhoc(
            _hermetic_defaults,
            tmp_path,
            "--rerun-port",
            "9877",
            "--rerun-connect",
        )
    assert _fake_rerun.instances == []


def test_rerun_port_conflicts_with_no_rerun(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    with pytest.raises(SystemExit, match=r"--no-rerun.*--rerun-port"):
        _run_adhoc(_hermetic_defaults, tmp_path, "--no-rerun", "--rerun-port", "9877")
    assert _fake_rerun.instances == []


def test_rerun_port_rejects_out_of_range(
    _hermetic_defaults: Path, tmp_path: Path, _fake_rerun: type[_FakeRerunSink]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_adhoc(_hermetic_defaults, tmp_path, "--rerun-port", "0")
    assert excinfo.value.code == 2
    assert _fake_rerun.instances == []


def test_styled_plain_when_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_robots.cli import _styled

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    assert _styled("policy:", "36") == "policy:"


def test_styled_emits_ansi_on_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_robots.cli import _styled

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert _styled("policy:", "36") == "\x1b[36mpolicy:\x1b[0m"


def test_styled_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_robots.cli import _styled

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert _styled("x", "1") == "x"


# --- guardrails by default (plan 0008 §3e) -----------------------------------


def _guard_space(**kwargs: object) -> object:
    from inspect_robots.spaces import Box

    return Box(**kwargs)  # type: ignore[arg-type]


def test_build_guardrails_full_chain_on_bounded_displacement_space() -> None:
    import numpy as np

    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.types import Action

    space = CubePickEmbodiment().info.action_space
    approver, active, warnings = _build_guardrails(space, None)
    assert active == ["clamp", "delta-limit"]
    assert warnings == []
    out = approver.review(Action(data=np.array([5.0, 5.0])), {})
    assert out.meta.get("clamped") is True  # box bounds enforced


def test_contributed_guardrail_follows_delta_limit_and_has_exact_banner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    import numpy as np

    from inspect_robots.approver import GuardrailContribution
    from inspect_robots.cli import _build_and_announce_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box
    from inspect_robots.types import Action

    seen: list[float] = []

    class _Observer:
        def review(self, action: Action, store: dict[str, Any]) -> Action:
            seen.append(float(action.data[0]))
            return action

    class _ContributingEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
            return GuardrailContribution(approvers=(("collision", _Observer()),))

    embodiment = _ContributingEmbodiment()
    args = argparse.Namespace(disable_guardrails=False, max_action_delta=0.05)
    approver = _build_and_announce_guardrails(args, embodiment.info.action_space, embodiment)
    assert approver is not None
    approver.review(Action(data=np.array([0.08, 0.0])), {})

    assert seen == [0.05]
    assert capsys.readouterr().out == "guardrails: clamp + delta-limit + collision\n"


def test_contributed_approvers_keep_declaration_order_and_warnings() -> None:
    import numpy as np

    from inspect_robots.approver import GuardrailContribution
    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box
    from inspect_robots.types import Action

    calls: list[str] = []

    class _Recorder:
        def __init__(self, name: str) -> None:
            self._name = name

        def review(self, action: Action, store: dict[str, Any]) -> Action:
            calls.append(self._name)
            return action

    class _ContributingEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
            return GuardrailContribution(
                approvers=(
                    ("first", _Recorder("first")),
                    ("second", _Recorder("second")),
                ),
                warnings=("collision model is running without table geometry",),
            )

    embodiment = _ContributingEmbodiment()
    approver, active, warnings = _build_guardrails(embodiment.info.action_space, None, embodiment)
    approver.review(Action(data=np.array([0.01, 0.01])), {})

    assert calls == ["first", "second"]
    assert active == ["clamp", "delta-limit", "first", "second"]
    assert warnings == ["collision model is running without table geometry"]


def test_warnings_only_contribution_keeps_the_generic_chain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    import numpy as np

    from inspect_robots.approver import GuardrailContribution
    from inspect_robots.cli import _build_and_announce_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box
    from inspect_robots.types import Action

    class _WarningEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
            return GuardrailContribution(warnings=("collision guardrail unavailable",))

    embodiment = _WarningEmbodiment()
    args = argparse.Namespace(disable_guardrails=False, max_action_delta=None)
    approver = _build_and_announce_guardrails(args, embodiment.info.action_space, embodiment)
    assert approver is not None
    action = Action(data=np.array([0.01, 0.01]))

    assert approver.review(action, {}) is action
    captured = capsys.readouterr()
    assert captured.out == "guardrails: clamp + delta-limit\n"
    assert captured.err == "guardrails warning: collision guardrail unavailable\n"


def test_non_callable_guardrail_hook_is_a_hard_error() -> None:
    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment

    class _BrokenEmbodiment(CubePickEmbodiment):
        contribute_guardrails = 7

    embodiment = _BrokenEmbodiment()
    with pytest.raises(SystemExit, match=r"cubepick.*int"):
        _build_guardrails(embodiment.info.action_space, None, embodiment)


def test_none_guardrail_hook_is_a_hard_error() -> None:
    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment

    class _BrokenEmbodiment(CubePickEmbodiment):
        contribute_guardrails = None

    embodiment = _BrokenEmbodiment()
    with pytest.raises(SystemExit, match=r"cubepick.*NoneType"):
        _build_guardrails(embodiment.info.action_space, None, embodiment)


def test_wrong_guardrail_contribution_type_is_a_hard_error() -> None:
    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box

    class _BrokenEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> object:
            return "not a contribution"

    embodiment = _BrokenEmbodiment()
    with pytest.raises(SystemExit, match=r"cubepick.*str"):
        _build_guardrails(embodiment.info.action_space, None, embodiment)


def test_contribution_is_the_only_gate_for_an_unbounded_space() -> None:
    import numpy as np

    from inspect_robots.approver import ChainApprover, GuardrailContribution
    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box
    from inspect_robots.types import Action

    calls: list[Action] = []

    class _OnlyGate:
        def review(self, action: Action, store: dict[str, Any]) -> Action:
            calls.append(action)
            return action

    class _ContributingEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
            return GuardrailContribution(approvers=(("only-gate", _OnlyGate()),))

    embodiment = _ContributingEmbodiment()
    space = Box(shape=(2,))
    approver, active, warnings = _build_guardrails(space, None, embodiment)
    action = Action(data=np.array([4.0, 5.0]))

    assert isinstance(approver, ChainApprover)
    assert approver.review(action, {}) is action
    assert calls == [action]
    assert active == ["only-gate"]
    assert not any("no guardrails" in warning for warning in warnings)


def test_disable_guardrails_does_not_invoke_contribution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    from inspect_robots.approver import GuardrailContribution
    from inspect_robots.cli import _build_and_announce_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box

    calls: list[Box] = []

    class _ContributingEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
            calls.append(action_space)
            return GuardrailContribution()

    embodiment = _ContributingEmbodiment()
    args = argparse.Namespace(disable_guardrails=True, max_action_delta=None)

    approver = _build_and_announce_guardrails(args, embodiment.info.action_space, embodiment)
    assert approver is None
    assert calls == []
    assert "guardrails: disabled" in capsys.readouterr().out


def test_embodiment_base_default_matches_an_absent_hook() -> None:
    from inspect_robots.approver import GuardrailContribution
    from inspect_robots.cli import _build_guardrails
    from inspect_robots.embodiment import EmbodimentBase
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.scene import Scene
    from inspect_robots.types import Action, Observation, StepResult

    class _BaseEmbodiment(EmbodimentBase):
        def __init__(self) -> None:
            self._delegate = CubePickEmbodiment()
            self.info = self._delegate.info

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            return self._delegate.reset(scene, seed=seed)

        def step(self, action: Action) -> StepResult:
            return self._delegate.step(action)

    absent = CubePickEmbodiment()
    base = _BaseEmbodiment()

    _, absent_active, absent_warnings = _build_guardrails(absent.info.action_space, None, absent)
    _, base_active, base_warnings = _build_guardrails(base.info.action_space, None, base)

    assert base.contribute_guardrails(base.info.action_space) == GuardrailContribution()
    assert (base_active, base_warnings) == (absent_active, absent_warnings)


def test_guardrail_contribution_conformance_passes_absent_and_valid_hooks() -> None:
    from inspect_robots.approver import GuardrailContribution
    from inspect_robots.conformance import (
        assert_guardrail_contribution_conformant,
        check_guardrail_contribution,
    )
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box

    class _ValidEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
            return GuardrailContribution()

    absent = CubePickEmbodiment()
    valid = _ValidEmbodiment()

    assert check_guardrail_contribution(absent, absent.info.action_space).ok
    assert check_guardrail_contribution(valid, valid.info.action_space).ok
    assert_guardrail_contribution_conformant(valid, valid.info.action_space)


def test_guardrail_contribution_conformance_reports_bad_hooks() -> None:
    from inspect_robots.conformance import (
        assert_guardrail_contribution_conformant,
        check_guardrail_contribution,
    )
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.spaces import Box

    class _NonCallableEmbodiment(CubePickEmbodiment):
        contribute_guardrails = 7

    class _WrongTypeEmbodiment(CubePickEmbodiment):
        def contribute_guardrails(self, action_space: Box) -> object:
            return "not a contribution"

    non_callable = _NonCallableEmbodiment()
    wrong_type = _WrongTypeEmbodiment()

    report = check_guardrail_contribution(non_callable, non_callable.info.action_space)
    assert not report.ok
    assert "int" in report.summary()

    report = check_guardrail_contribution(wrong_type, wrong_type.info.action_space)
    assert not report.ok
    assert "str" in report.summary()
    with pytest.raises(AssertionError, match="GuardrailContribution"):
        assert_guardrail_contribution_conformant(wrong_type, wrong_type.info.action_space)


def test_build_guardrails_threads_max_action_delta() -> None:
    import numpy as np

    from inspect_robots.cli import _build_guardrails
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.types import Action

    approver, _, _ = _build_guardrails(CubePickEmbodiment().info.action_space, 0.05)
    out = approver.review(Action(data=np.array([0.08, 0.0])), {})
    assert float(out.data[0]) == pytest.approx(0.05)
    assert out.meta.get("delta_clamped") is True


def test_build_guardrails_degrades_per_component() -> None:
    import numpy as np

    from inspect_robots.cli import _build_guardrails
    from inspect_robots.spaces import ActionSemantics, Box

    # Bounds-less absolute space (the isaacsim shape): nothing is applicable.
    bare = Box(shape=(2,), semantics=ActionSemantics("joint_pos"))
    _approver, active, warnings = _build_guardrails(bare, None)
    assert active == []
    assert any("no guardrails" in w for w in warnings)
    # ...but an explicit max delta re-enables the delta limiter.
    _, active, warnings = _build_guardrails(bare, 0.1)
    assert active == ["delta-limit"]
    assert not any("no guardrails" in w for w in warnings)

    # Semantics-less bounded space: clamp works, delta limiter refuses.
    blind = Box(shape=(2,), low=np.zeros(2), high=np.ones(2))
    _, active, warnings = _build_guardrails(blind, None)
    assert active == ["clamp"]
    assert any("semantics" in w for w in warnings)

    # Quat-repr pose space: the delta limiter names the rotation refusal.
    quat = Box(
        shape=(7,),
        low=np.full(7, -1.0),
        high=np.full(7, 1.0),
        semantics=ActionSemantics("eef_abs_pose", rotation_repr="quat_wxyz"),
    )
    _, active, warnings = _build_guardrails(quat, None)
    assert active == ["clamp"]
    assert any("rotation_repr" in w for w in warnings)


def test_cli_run_header_names_active_guardrails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert "guardrails: clamp + delta-limit" in capsys.readouterr().out


def test_cli_disable_guardrails_warns_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "cubepick",
            "--log-dir",
            str(tmp_path),
            "--disable-guardrails",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "guardrails: disabled" in captured.out
    assert "WARNING" in captured.err and "guardrails" in captured.err


def test_cli_max_action_delta_conflicts_with_disable(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="disable-guardrails"):
        main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
                "--disable-guardrails",
                "--max-action-delta",
                "0.1",
            ]
        )


@pytest.mark.parametrize("value", ["-1", "0", "inf", "nan"])
def test_cli_run_rejects_invalid_max_action_delta(value: str, tmp_path: Path) -> None:
    """An explicit malformed --max-action-delta fails fast (#154), not a soft warning:
    it is a CLI input error, distinct from a derived limit an embodiment's space
    cannot support (test_cli_degraded_guardrails_warn_but_run stays a warning)."""
    with pytest.raises(SystemExit, match="finite and > 0"):
        main(
            [
                "run",
                "--task",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
                "--max-action-delta",
                value,
            ]
        )


@pytest.mark.parametrize("value", ["-1", "0", "inf", "nan"])
def test_cli_eval_set_rejects_invalid_max_action_delta(value: str) -> None:
    """Same guard as run, exercised through eval-set's shared conflict check (#154)."""
    with pytest.raises(SystemExit, match="finite and > 0"):
        main(["eval-set", "cubepick-reach", "--max-action-delta", value])


def test_cli_degraded_guardrails_warn_but_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A semantics-less, bounds-less action space: every guardrail component
    # refuses, the CLI says so on stderr, and the run still proceeds.
    from dataclasses import replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.spaces import Box

    class _BareSpaceEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, action_space=Box(shape=(2,)))

    embodiment_decorator("bare-cubepick")(_BareSpaceEmbodiment)
    rc = main(
        [
            "run",
            "--task",
            "cubepick-reach",
            "--policy",
            "scripted",
            "--embodiment",
            "bare-cubepick",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "guardrails: none active" in captured.out
    assert "no guardrails" in captured.err


def test_guided_error_mentions_config_set() -> None:
    with pytest.raises(SystemExit, match="inspect-robots config set policy") as excinfo:
        main(["run", "--instruction", "reach the cube"])
    assert "inspect-robots setup" in str(excinfo.value)


# --- config set / show (plan 0008 §3e) ----------------------------------------


def test_config_show_honors_config_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", "placeholder")
    path = tmp_path / "rig-b.ini"
    path.write_text("[defaults]\npolicy = rig-b\n", encoding="utf-8")

    assert main(["config", "show", "--config", str(path)]) == 0

    out = capsys.readouterr().out
    assert "policy: rig-b" in out
    assert f"policy: rig-b  ({path})" in out


def test_config_set_honors_config_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", "placeholder")
    path = tmp_path / "rig-b.ini"
    xdg = tmp_path / "decoy"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    assert main(["config", "set", "policy", "x", "--config", str(path)]) == 0

    assert "policy = x" in path.read_text(encoding="utf-8")
    assert not (xdg / "inspect-robots" / "config.ini").exists()


def test_config_flag_beats_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", "placeholder")
    path_a = tmp_path / "rig-a.ini"
    path_b = tmp_path / "rig-b.ini"
    path_a.write_text("[defaults]\npolicy = rig-a\n", encoding="utf-8")
    path_b.write_text("[defaults]\npolicy = rig-b\n", encoding="utf-8")
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(path_a))

    assert main(["config", "show", "--config", str(path_b)]) == 0

    out = capsys.readouterr().out
    assert f"policy: rig-b  ({path_b})" in out
    assert "rig-a" not in out


def test_config_flag_expands_tilde(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", "placeholder")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    path = tmp_path / "rig.ini"
    path.write_text("[defaults]\npolicy = tilde-rig\n", encoding="utf-8")

    assert main(["config", "show", "--config", "~/rig.ini"]) == 0

    assert f"policy: tilde-rig  ({path})" in capsys.readouterr().out


def test_config_flag_anchors_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "rig.ini"
    path.write_text("[defaults]\npolicy = relative-rig\n", encoding="utf-8")
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", "placeholder")

    assert main(["config", "show", "--config", "rig.ini"]) == 0

    assert "policy: relative-rig" in capsys.readouterr().out
    assert os.environ["INSPECT_ROBOTS_CONFIG"] == str(path)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--config", "p"],
        ["eval-set", "t", "--config", "p"],
        ["doctor", "--config", "p"],
        ["setup", "--config", "p"],
        ["config", "set", "policy", "x", "--config", "p"],
        ["config", "show", "--config", "p"],
    ],
)
def test_config_flag_is_wired_to_every_config_reading_subcommand(argv: list[str]) -> None:
    args = cli.build_parser().parse_args(argv)

    assert args.config == "p"


def test_no_config_flag_leaves_environ_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSPECT_ROBOTS_CONFIG", raising=False)

    assert main(["list"]) == 0

    assert "INSPECT_ROBOTS_CONFIG" not in os.environ


def test_cli_config_set_writes_and_show_reads(
    _hermetic_defaults: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "set", "embodiment", "cubepick"]) == 0
    assert main(["config", "set", "policy", "scripted"]) == 0
    path = _hermetic_defaults / "inspect-robots" / "config.ini"
    body = path.read_text(encoding="utf-8")
    assert "embodiment = cubepick" in body and "policy = scripted" in body
    capsys.readouterr()
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert f"embodiment: cubepick  ({path})" in out
    assert "sim_embodiment: (unset)" in out


def test_cli_config_set_preserves_unknown_sections(_hermetic_defaults: Path) -> None:
    _write_config(
        _hermetic_defaults,
        "[defaults]\nscorer = success_at_end\n[embodiment.args]\nleft_channel = can2\n",
    )
    assert main(["config", "set", "embodiment", "cubepick"]) == 0
    body = (_hermetic_defaults / "inspect-robots" / "config.ini").read_text(encoding="utf-8")
    assert "[embodiment.args]" in body and "left_channel = can2" in body
    assert "scorer = success_at_end" in body
    assert "embodiment = cubepick" in body


def test_cli_config_set_validates_values(_hermetic_defaults: Path) -> None:
    with pytest.raises(SystemExit, match="max_steps"):
        main(["config", "set", "max_steps", "zero"])
    with pytest.raises(SystemExit, match="store_frames"):
        main(["config", "set", "store_frames", "maybe"])
    # Unknown keys are rejected by argparse itself (exit code 2).
    with pytest.raises(SystemExit) as excinfo:
        main(["config", "set", "frobnicate", "1"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit, match="rerun"):
        main(["config", "set", "rerun", "sometimes"])
    # Valid values round-trip.
    assert main(["config", "set", "max_steps", "50"]) == 0
    assert main(["config", "set", "store_frames", "true"]) == 0
    assert main(["config", "set", "rerun", "true"]) == 0


def test_cli_config_set_rejects_bad_rerun_port(
    _hermetic_defaults: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="rerun_port must be an integer in 1-65535"):
        main(["config", "set", "rerun_port", "sometimes"])

    assert main(["config", "set", "rerun_port", "9877"]) == 0
    capsys.readouterr()
    assert main(["config", "show"]) == 0
    assert "rerun_port: 9877" in capsys.readouterr().out


def test_component_config_error_exits_cleanly(tmp_path: Path) -> None:
    """A factory's guided ConfigError must exit cleanly, not print a traceback."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.registry import policy as policy_decorator

    @policy_decorator("misconfigured-policy")
    def _factory(**kwargs: object) -> object:
        raise ConfigError("no model configured.\nfix: set $SOME_KEY")

    with pytest.raises(SystemExit, match="no model configured") as excinfo:
        main(
            [
                "run",
                "--instruction",
                "reach it",
                "--policy",
                "misconfigured-policy",
                "--embodiment",
                "cubepick",
                "--log-dir",
                str(tmp_path),
            ]
        )
    assert "Traceback" not in str(excinfo.value)


def test_component_type_error_from_config_args_exits_cleanly(
    _hermetic_defaults: Path, tmp_path: Path
) -> None:
    """Invalid persisted kwargs must identify their component and both sources."""
    from inspect_robots.registry import policy as policy_decorator

    name = "strict-args-policy-for-cli-test"

    @policy_decorator(name)
    def _factory() -> ScriptedPolicy:
        return ScriptedPolicy()

    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        f"policy = {name}\n"
        "embodiment = cubepick\n"
        "scorer = success_at_end\n"
        "[policy.args]\n"
        "typoed_option = 1\n",
    )
    try:
        with pytest.raises(SystemExit) as excinfo:
            main(["reach it", "--log-dir", str(tmp_path)])
    finally:
        reg._FACTORIES["policy"].pop(name)

    message = str(excinfo.value)
    assert f"invalid arguments for policy {name!r}" in message
    assert "unexpected keyword argument 'typoed_option'" in message
    assert "[policy.args]" in message and "-P k=v" in message
    assert "Traceback" not in message


def test_sim_embodiment_type_error_names_sim_args_section(
    _hermetic_defaults: Path, tmp_path: Path
) -> None:
    """Invalid sim kwargs must identify the sim-specific config section."""
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator

    name = "strict-args-sim-embodiment-for-cli-test"

    @embodiment_decorator(name)
    def _factory() -> CubePickEmbodiment:
        return CubePickEmbodiment()

    _write_config(
        _hermetic_defaults,
        "[defaults]\n"
        "policy = scripted\n"
        "scorer = success_at_end\n"
        f"sim_embodiment = {name}\n"
        "[sim_embodiment.args]\n"
        "typoed_option = 1\n",
    )
    try:
        with pytest.raises(SystemExit) as excinfo:
            main(["reach it", "--sim", "--log-dir", str(tmp_path)])
    finally:
        reg._FACTORIES["embodiment"].pop(name)

    message = str(excinfo.value)
    assert f"invalid arguments for embodiment {name!r}" in message
    assert "unexpected keyword argument 'typoed_option'" in message
    assert "check [sim_embodiment.args]" in message
    assert "-E k=v" in message
    assert "check [embodiment.args]" not in message
    assert "Traceback" not in message


# --- doctor (adapter conformance) ---------------------------------------------


def test_doctor_passes_on_conformant_embodiment(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--embodiment", "cubepick"]) == 0
    assert "conformant" in capsys.readouterr().out


def test_doctor_reports_missing_runtime_requirement_before_conformance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator

    name = "missing-runtime-doctor-cubepick"

    class _MissingRuntimeCubePick(CubePickEmbodiment):
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "definitely_missing_xyz": "uv pip install thing"
        }

    embodiment_decorator(name)(_MissingRuntimeCubePick)
    assert main(["doctor", "--embodiment", name]) == 1
    out = capsys.readouterr().out
    error = "[error] runtime-requirement: definitely_missing_xyz missing → uv pip install thing"
    assert error in out
    assert out.index(error) < out.index("conformant")


def test_doctor_accepts_present_runtime_requirements(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator

    name = "present-runtime-doctor-cubepick"

    class _PresentRuntimeCubePick(CubePickEmbodiment):
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {"os": "install os"}

    embodiment_decorator(name)(_PresentRuntimeCubePick)
    assert main(["doctor", "--embodiment", name]) == 0
    assert "runtime-requirement" not in capsys.readouterr().out


def test_doctor_unknown_embodiment_keeps_guided_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = "definitely-missing-doctor-embodiment"

    with pytest.raises(SystemExit, match=f"no embodiment named '{name}'") as excinfo:
        main(["doctor", "--embodiment", name])

    message = str(excinfo.value)
    assert "available:" in message
    assert "cubepick" in message
    assert "Traceback" not in message
    assert f"embodiment: {name} (--embodiment)" in capsys.readouterr().out


def test_doctor_fails_on_nonconformant_embodiment(capsys: pytest.CaptureFixture[str]) -> None:
    from dataclasses import replace

    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator
    from inspect_robots.spaces import Box

    class _UndeclaredEmbodiment(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, action_space=Box(shape=(2,)))

    embodiment_decorator("undeclared-cubepick")(_UndeclaredEmbodiment)
    assert main(["doctor", "--embodiment", "undeclared-cubepick"]) == 1
    out = capsys.readouterr().out
    assert "[error] semantics" in out and "[error] bounds" in out


def test_doctor_uses_default_embodiment_and_guides_when_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="no embodiment given"):
        main(["doctor"])
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    assert main(["doctor"]) == 0
    assert "cubepick" in capsys.readouterr().out


def test_doctor_ignores_config_args_for_a_different_embodiment(
    _hermetic_defaults: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The issue-#44 repro: a lab machine's persisted rig args must not crash a
    # conformance check of an unrelated, explicitly-selected embodiment.
    _write_config(
        _hermetic_defaults,
        "[defaults]\nembodiment = yam-arms\n[embodiment.args]\nrest_pose = 0.5\n",
    )
    assert main(["doctor", "--embodiment", "cubepick"]) == 0
    captured = capsys.readouterr()
    assert "conformant" in captured.out
    assert "note: ignoring [embodiment.args] for 'cubepick': they apply to 'yam-arms'" in (
        captured.err
    )


def test_doctor_config_args_without_a_default_owner_never_apply(
    _hermetic_defaults: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An [embodiment.args] section with no [defaults] embodiment has no owner,
    # so it applies to nothing (and says so).
    _write_config(_hermetic_defaults, "[embodiment.args]\nrest_pose = 0.5\n")
    assert main(["doctor", "--embodiment", "cubepick"]) == 0
    err = capsys.readouterr().err
    assert "no default embodiment is configured" in err


def test_doctor_closes_the_embodiment_it_constructs() -> None:
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots.registry import embodiment as embodiment_decorator

    closed: list[str] = []

    class _ClosableCubePick(CubePickEmbodiment):
        def close(self) -> None:
            closed.append("closed")

    embodiment_decorator("closable-doctor-cubepick")(_ClosableCubePick)
    assert main(["doctor", "--embodiment", "closable-doctor-cubepick"]) == 0
    assert closed == ["closed"]
