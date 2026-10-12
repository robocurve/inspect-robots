"""CLI task ownership preserves halts through failed cleanup and bounded dispatch."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ALL_COMPLETED, wait
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest

import inspect_robots
import inspect_robots._parallel as parallel
import inspect_robots.cli as cli
import inspect_robots.registry as registry
from inspect_robots._claims import DeviceClaim
from inspect_robots.errors import EmbodimentFault, SafetyAbort
from inspect_robots.log import EvalLog
from inspect_robots.logging import JsonLogSink, LiveLogSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task


def _task(name: str) -> Task:
    return Task(
        name=name,
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_steps=1,
    )


@pytest.mark.parametrize("halt_type", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
@pytest.mark.parametrize("failed_cleanup", [False, True])
def test_parallel_cli_halt_closes_every_resource_without_replacement_admission(
    halt_type: type[BaseException],
    failed_cleanup: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.registered("task")
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    barrier = Barrier(2)
    started: list[str] = []
    policies: list[Policy] = []
    embodiments: list[Embodiment] = []
    claims: list[Claim] = []
    halt = halt_type("primary CLI halt")

    class Policy(ScriptedPolicy):
        close_count = 0
        fail_close = False

        def __init__(self) -> None:
            super().__init__()
            policies.append(self)

        def close(self) -> None:
            self.close_count += 1
            if self.fail_close:
                raise RuntimeError("policy cleanup failed")

    class Embodiment(CubePickEmbodiment):
        close_count = 0
        fail_close = False

        def __init__(self) -> None:
            super().__init__()
            embodiments.append(self)

        def close(self) -> None:
            self.close_count += 1
            super().close()
            if self.fail_close:
                raise RuntimeError("embodiment cleanup failed")

    class Claim(DeviceClaim):
        release_count = 0

        def __init__(self) -> None:
            super().__init__([])
            claims.append(self)

        def release(self) -> None:
            self.release_count += 1

    names = ["cli-halt", "cli-active", "cli-replacement"]
    for name in names:
        monkeypatch.setitem(registry._FACTORIES["task"], name, lambda name=name: _task(name))
    monkeypatch.setitem(registry._FACTORIES["policy"], "cli-cleanup-policy", Policy)
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "cli-cleanup-sim", Embodiment)
    monkeypatch.setattr(cli, "claim_devices", lambda *args: Claim())

    def run_task(
        tasks: Sequence[Task], policy: object, embodiment: object, **kwargs: object
    ) -> tuple[bool, list[EvalLog]]:
        del kwargs
        name = tasks[0].name
        started.append(name)
        if name == "cli-halt":
            cast(Policy, policy).fail_close = failed_cleanup
            cast(Embodiment, embodiment).fail_close = failed_cleanup
        barrier.wait(timeout=5)
        if name == "cli-halt":
            raise halt
        return True, []

    monkeypatch.setattr(inspect_robots, "eval_set", run_task)
    # Complete both initial futures before collection, so even a successful
    # sibling finishing first cannot hide premature replacement admission.
    monkeypatch.setattr(
        parallel, "wait", lambda futures, **kwargs: wait(futures, return_when=ALL_COMPLETED)
    )
    argv = [
        "eval-set",
        *names,
        "--policy",
        "cli-cleanup-policy",
        "--embodiment",
        "cli-cleanup-sim",
        "--no-prompt",
        "--max-workers",
        "2",
        "--log-dir",
        str(tmp_path),
    ]
    with warnings.catch_warnings(record=True) as reported:
        warnings.simplefilter("always")
        if halt_type is KeyboardInterrupt:
            assert cli.main(argv) == 130
            assert "cancelled: partial logs" in capsys.readouterr().out
        else:
            with pytest.raises(halt_type) as raised:
                cli.main(argv)
            assert raised.value is halt
    assert sorted(started) == sorted(names[:2])
    assert len(policies) == len(embodiments) == len(claims) == 2
    assert all(policy.close_count == 1 for policy in policies)
    assert all(embodiment.close_count == 1 for embodiment in embodiments)
    assert all(claim.release_count == 1 for claim in claims)
    assert len(reported) == (2 if failed_cleanup else 0)
    if failed_cleanup:
        messages = [str(warning.message) for warning in reported]
        assert all(
            halt_type.__name__ in message and "RuntimeError" in message for message in messages
        )
        assert any("policy cleanup failed" in message for message in messages)
        assert any("embodiment cleanup failed" in message for message in messages)


@pytest.mark.parametrize("halt_type", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_serial_cli_eval_set_preserves_halt_through_embodiment_cleanup_failure(
    halt_type: type[BaseException], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.registered("task")
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    closed: list[bool] = []
    released: list[bool] = []
    halt = halt_type("primary serial halt")

    class Embodiment(CubePickEmbodiment):
        def close(self) -> None:
            closed.append(True)
            raise RuntimeError("serial embodiment cleanup failed")

    class Claim(DeviceClaim):
        def release(self) -> None:
            released.append(True)

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "cli-cleanup-sim", Embodiment)
    monkeypatch.setattr(cli, "claim_devices", lambda *args: Claim([]))

    def run_task(*args: object, **kwargs: object) -> tuple[bool, list[EvalLog]]:
        raise halt

    monkeypatch.setattr(inspect_robots, "eval_set", run_task)
    argv = [
        "eval-set",
        "cubepick-reach",
        "--policy",
        "scripted",
        "--embodiment",
        "cli-cleanup-sim",
        "--no-prompt",
        "--log-dir",
        str(tmp_path),
    ]
    with pytest.warns(RuntimeWarning, match="serial embodiment cleanup failed"):
        if halt_type is KeyboardInterrupt:
            assert cli.main(argv) == 130
        else:
            with pytest.raises(halt_type) as raised:
                cli.main(argv)
            assert raised.value is halt
    assert closed == [True]
    assert released == [True]


@pytest.mark.parametrize("halt_type", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_parallel_cli_constructor_halt_survives_policy_and_claim_cleanup_failures(
    halt_type: type[BaseException], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.registered("task")
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    barrier = Barrier(2)
    started: list[str] = []
    policies: list[ConstructorPolicy] = []
    claims: list[ConstructorClaim] = []

    class ConstructorPolicy(ScriptedPolicy):
        close_count = 0

        def __init__(self) -> None:
            super().__init__()
            policies.append(self)

        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("constructor policy cleanup failed")

    class ConstructorClaim(DeviceClaim):
        release_count = 0

        def __init__(self) -> None:
            super().__init__([])
            claims.append(self)

        def release(self) -> None:
            self.release_count += 1
            raise RuntimeError("constructor claim cleanup failed")

    def construct_embodiment() -> CubePickEmbodiment:
        barrier.wait(timeout=5)
        raise halt_type("primary constructor halt")

    def construct_task(name: str) -> Task:
        started.append(name)
        return _task(name)

    names = ["cli-halt", "cli-active", "cli-replacement"]
    for name in names:
        monkeypatch.setitem(
            registry._FACTORIES["task"], name, lambda name=name: construct_task(name)
        )
    monkeypatch.setitem(registry._FACTORIES["policy"], "cli-cleanup-policy", ConstructorPolicy)
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "cli-cleanup-sim", construct_embodiment)
    monkeypatch.setattr(cli, "claim_devices", lambda *args: ConstructorClaim())
    monkeypatch.setattr(
        parallel, "wait", lambda futures, **kwargs: wait(futures, return_when=ALL_COMPLETED)
    )
    argv = [
        "eval-set",
        *names,
        "--policy",
        "cli-cleanup-policy",
        "--embodiment",
        "cli-cleanup-sim",
        "--no-prompt",
        "--max-workers",
        "2",
        "--log-dir",
        str(tmp_path),
    ]
    with pytest.warns(RuntimeWarning, match="constructor .* cleanup failed") as reported:
        if halt_type is KeyboardInterrupt:
            assert cli.main(argv) == 130
        else:
            with pytest.raises(halt_type, match="primary constructor halt"):
                cli.main(argv)
    assert sorted(started) == sorted(names[:2])
    assert len(policies) == len(claims) == 2
    assert all(policy.close_count == 1 for policy in policies)
    assert all(claim.release_count == 1 for claim in claims)
    assert len(reported) == 4
    assert all(halt_type.__name__ in str(warning.message) for warning in reported)


@pytest.mark.parametrize("failed_write", [False, True])
def test_cli_eval_set_live_snapshot_survives_only_a_failed_canonical_write(
    failed_write: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    snapshot = tmp_path / "partial.live.json"

    def interrupt(*args: object, **kwargs: object) -> tuple[bool, list[EvalLog]]:
        sinks = kwargs["sinks"]
        assert isinstance(sinks, list)
        canonical = next(sink for sink in sinks if isinstance(sink, JsonLogSink))
        live = next(sink for sink in sinks if isinstance(sink, LiveLogSink))
        canonical.write_failed = failed_write
        live.path = snapshot
        snapshot.write_text("partial log")
        raise KeyboardInterrupt

    monkeypatch.setattr(inspect_robots, "eval_set", interrupt)
    assert (
        cli.main(
            [
                "eval-set",
                "cubepick-reach",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--no-prompt",
                "--log-dir",
                str(tmp_path),
            ]
        )
        == 130
    )
    assert snapshot.exists() is failed_write
