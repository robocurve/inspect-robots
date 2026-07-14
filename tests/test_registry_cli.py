"""Registry resolution, entry-point discovery, and the CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

import inspect_robots.cli as cli
import inspect_robots.registry as reg
from inspect_robots._defaults import ENV_EMBODIMENT, ENV_POLICY, ENV_SIM_EMBODIMENT
from inspect_robots.cli import main
from inspect_robots.log import EvalLog
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
    assert "status: success" in out
    assert "success_at_end" in out
    (written,) = tmp_path.glob("*.json")
    assert f"log: {written}" in out  # the CLI tells the user where the log went
    assert "error:" not in out
    assert "hint:" not in out


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
    assert "status: error" in out
    assert "error: EmbodimentFault: reset exploded" in out
    assert "  [error] scene-1\n" in out
    assert "scene-0" not in out  # successful scenes are not failure context
    assert out.count("EmbodimentFault: reset exploded") == 1
    (written,) = tmp_path.glob("*.json")
    assert f"hint: inspect-robots inspect {written}" in out


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


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Inspect Robots" in capsys.readouterr().out


def test_cli_help_lists_setup(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "setup" in capsys.readouterr().out


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
    rc = main(["reach the cube", "--log-dir", str(log_dir)])  # default scorer: operator
    assert rc == 0
    assert "unrecognized answer 'yse'" in capsys.readouterr().out
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
    assert "status: success" in out


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

    def __init__(self, recording_path: str | None = None, *, spawn: bool = False) -> None:
        self.spawn = spawn
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


# --- doctor (adapter conformance) ---------------------------------------------


def test_doctor_passes_on_conformant_embodiment(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--embodiment", "cubepick"]) == 0
    assert "conformant" in capsys.readouterr().out


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
