"""Registry resolution, entry-point discovery, and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

import inspect_robots.registry as reg
from inspect_robots._defaults import ENV_EMBODIMENT, ENV_POLICY
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
    assert list((tmp_path / "frames").glob("*.npy"))  # --store-frames streamed


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Inspect Robots" in capsys.readouterr().out


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
    assert log.samples[0].operator_judgements == ["y"]
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
    assert _read_only_log(log_dir_a).samples[0].operator_judgements == [None]

    # TTY but --no-prompt: never prompts either.
    _tty_stdin(monkeypatch)
    log_dir_b = tmp_path / "b"
    assert main(["reach the cube", "--no-prompt", "--log-dir", str(log_dir_b)]) == 0
    log = _read_only_log(log_dir_b)
    assert log.samples[0].operator_judgements == [None]
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
    assert _read_only_log(log_dir).samples[0].operator_judgements == [None]
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
