"""Registry resolution, entry-point discovery, and the CLI."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

import inspect_robots.cli as cli
import inspect_robots.registry as reg
from inspect_robots._defaults import ENV_EMBODIMENT, ENV_POLICY, ENV_SIM_EMBODIMENT
from inspect_robots.cli import main
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
from inspect_robots.mock import ScriptedPolicy
from inspect_robots.registry import registered, resolve


@pytest.fixture(autouse=True)
def _hermetic_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep CLI runs blind to the developer's real config file and env vars."""
    config_home = tmp_path / "config-home"
    config_home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv(ENV_POLICY, raising=False)
    monkeypatch.delenv(ENV_EMBODIMENT, raising=False)
    monkeypatch.delenv(ENV_SIM_EMBODIMENT, raising=False)
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
    assert out.count("hint:") == 2


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
    assert "hint: HTML viewer: inspect-robots view" in out
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
    assert "inspect-robots view" in out


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


def test_view_rejects_directory_output_with_guidance(tmp_path: Path) -> None:
    path = _write_log(_step_limit_log(), tmp_path, "run.json")
    output = tmp_path / "existing-directory"
    output.mkdir()

    with pytest.raises(SystemExit, match="is a directory; pass an HTML file path"):
        main(["view", str(path), "-o", str(output)])


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


def test_operator_prompt_records_verdict_and_reprompts_on_typos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "cubepick")
    _tty_stdin(monkeypatch)
    answers = iter(["yse", "y"])
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


def test_registered_task_never_prompts_even_with_operator_scorer_on_tty(
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
        "builtins.input", lambda _prompt: pytest.fail("R6: --task runs must not block")
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


@pytest.mark.parametrize(
    ("termination_reason", "expected_verdict"),
    [("success", "y"), ("failure", "n")],
)
def test_prompt_operator_adopts_definitive_embodiment_verdict(
    termination_reason: str,
    expected_verdict: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots.cli import _prompt_operator
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene

    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=True,
        termination_reason=termination_reason,
    )
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("operator prompt must not fire")
    )

    _prompt_operator(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == expected_verdict
    (event,) = record.events
    assert event.kind == "operator"
    assert event.t == 0
    assert event.data == {"verdict": expected_verdict, "source": "embodiment"}


@pytest.mark.parametrize(
    ("terminated", "termination_reason"),
    [
        (False, None),
        (False, "max_steps"),
        (False, "policy_stop"),
        (True, None),
    ],
)
def test_prompt_operator_still_prompts_without_definitive_verdict(
    terminated: bool,
    termination_reason: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots.cli import _prompt_operator
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene

    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=terminated,
        termination_reason=termination_reason,
    )
    prompts: list[str] = []

    def _answer(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", _answer)
    _prompt_operator(record, Scene(id="s0", instruction="reach"))

    assert len(prompts) == 1
    assert record.operator_judgement == "y"
    (event,) = record.events
    assert event.data == {"verdict": "y", "source": "prompt"}


def test_prompt_operator_prompts_for_truncated_success_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots.cli import _prompt_operator
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene

    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=False,
        truncated=True,
        termination_reason="success",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    _prompt_operator(record, Scene(id="s0", instruction="reach"))

    assert record.operator_judgement == "n"
    (event,) = record.events
    assert event.data == {"verdict": "n", "source": "prompt"}


def test_prompt_operator_warns_before_judging_step_limited_trial(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspect_robots.cli import _prompt_operator
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene

    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        truncated=True,
        termination_reason="max_steps",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    _prompt_operator(record, Scene(id="s0", instruction="reach"))

    assert "trial hit the step limit before terminating" in capsys.readouterr().out
    assert record.operator_judgement == "n"


@pytest.mark.parametrize(
    ("termination_reason", "expected_score"),
    [("success", True), ("failure", False)],
)
def test_operator_scorer_reads_adopted_embodiment_verdict(
    termination_reason: str,
    expected_score: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_robots.cli import _prompt_operator
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene
    from inspect_robots.scorer import operator_scorer

    record = TrialRecord(
        scene_id="s0",
        epoch=0,
        seed=0,
        terminated=True,
        termination_reason=termination_reason,
    )
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("operator prompt must not fire")
    )

    _prompt_operator(record, Scene(id="s0", instruction="reach"))

    assert operator_scorer()(record, None).value is expected_score


def test_prompt_operator_unit_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_robots.cli import _prompt_operator
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene

    scene = Scene(id="s0", instruction="reach")

    def _record() -> TrialRecord:
        record = TrialRecord(scene_id="s0", epoch=0, seed=0)
        record.steps = [None, None, None]  # type: ignore[list-item]
        return record

    # A verdict is recorded verbatim with an operator event at the final step.
    record = _record()
    monkeypatch.setattr("builtins.input", lambda _prompt: "Partial")
    _prompt_operator(record, scene)
    assert record.operator_judgement == "partial"
    (event,) = record.events
    assert event.kind == "operator"
    assert event.t == 3
    assert event.data["verdict"] == "partial"

    # skip: no judgement, no event.
    record = _record()
    monkeypatch.setattr("builtins.input", lambda _prompt: "skip")
    _prompt_operator(record, scene)
    assert record.operator_judgement is None
    assert record.events == []

    # EOF (operator hit Ctrl-D): treated as skip.
    def _eof(_prompt: str) -> str:
        raise EOFError

    record = _record()
    monkeypatch.setattr("builtins.input", _eof)
    _prompt_operator(record, scene)
    assert record.operator_judgement is None
    assert record.events == []


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
    leaves real arms energized."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.mock import CubePickEmbodiment

    closed: list[bool] = []

    class _Tracked(CubePickEmbodiment):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setitem(reg._FACTORIES["embodiment"], "tracked-cubepick", _Tracked)
    monkeypatch.setenv(ENV_POLICY, "scripted")
    monkeypatch.setenv(ENV_EMBODIMENT, "tracked-cubepick")
    with pytest.raises(ConfigError):
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
        connect_url: str | None = None,
    ) -> None:
        self.spawn = spawn
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
