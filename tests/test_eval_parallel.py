"""Parallel evaluations preserve task identity, resource ownership and bounded admission."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import Any, cast

import pytest

from inspect_robots import eval_set, read_eval_log
from inspect_robots._parallel import run_parallel
from inspect_robots.cli import main
from inspect_robots.errors import ConfigError, EmbodimentFault, SafetyAbort
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.registry import registered
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Observation

# --- Fixture helpers ---


def _task(name: str) -> Task:
    return Task(
        name=name,
        scenes=[Scene(id="s", instruction="reach", init_seed=7)],
        scorer=success_at_end(),
        max_steps=3,
    )


@pytest.fixture
def components(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[ScriptedPolicy], list[CubePickEmbodiment]]:
    import inspect_robots.registry as registry

    registered("task")
    policies: list[ScriptedPolicy] = []
    embodiments: list[CubePickEmbodiment] = []

    class Policy(ScriptedPolicy):
        closed = 0

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            policies.append(self)

        def close(self) -> None:
            self.closed += 1

    class Embodiment(CubePickEmbodiment):
        closed = 0

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            embodiments.append(self)

        def close(self) -> None:
            self.closed += 1
            super().close()

    for name in ("parallel-a", "parallel-b", "parallel-c"):
        monkeypatch.setitem(registry._FACTORIES["task"], name, lambda name=name: _task(name))
    monkeypatch.setitem(registry._FACTORIES["policy"], "parallel-policy", Policy)
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "parallel-sim", Embodiment)
    return policies, embodiments


# --- End fixture helpers ---


def test_bounded_dispatch_and_order() -> None:
    barrier = Barrier(2)
    release_first = Event()
    lock = Lock()
    active = 0
    peak = 0

    def run(index: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if index < 2:
            barrier.wait(timeout=5)
            if index == 0:
                assert release_first.wait(timeout=5)
            else:
                release_first.set()
        with lock:
            active -= 1
        return index

    assert run_parallel(list(range(7)), run, 2) == list(range(7))
    assert peak == 2
    assert active == 0
    assert run_parallel([], run, 2) == []


@pytest.mark.parametrize("error", [KeyboardInterrupt, SafetyAbort, EmbodimentFault])
def test_halt_drains_active_tasks_without_admitting_more(error: type[BaseException]) -> None:
    barrier = Barrier(2)
    started: list[int] = []
    finished: list[int] = []

    def run(index: int) -> int:
        started.append(index)
        try:
            barrier.wait(timeout=5)
            if index == 0:
                raise error("halt")
            return index
        finally:
            finished.append(index)

    # A completed sibling can race the halt; block result collection until both
    # first-batch futures complete so this tests the admission boundary exactly.
    from concurrent.futures import ALL_COMPLETED, wait
    from unittest.mock import patch

    import inspect_robots._parallel as parallel

    with (
        patch.object(
            parallel, "wait", side_effect=lambda fs, **kw: wait(fs, return_when=ALL_COMPLETED)
        ),
        pytest.raises(error),
    ):
        run_parallel(list(range(5)), run, 2)
    assert sorted(started) == [0, 1]
    assert sorted(finished) == [0, 1]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_invalid_worker_count(workers: Any, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        eval_set(
            "cubepick-reach", "scripted", "cubepick", max_workers=workers, log_dir=str(tmp_path)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tasks": _task("live")},
        {"policy": ScriptedPolicy()},
        {"embodiment": CubePickEmbodiment()},
        {"grader": object()},
        {"sinks": []},
        {"controller": object()},
        {"approver": object()},
        {"operator_input": object()},
        {"before_scoring": lambda *_: None},
    ],
)
def test_parallel_rejects_shared_objects(kwargs: dict[str, Any], tmp_path: Path) -> None:
    options: dict[str, Any] = {
        "tasks": "cubepick-reach",
        "policy": "scripted",
        "embodiment": "cubepick",
        "max_workers": 2,
        "log_dir": str(tmp_path),
    }
    options.update(kwargs)
    with pytest.raises(ConfigError, match="parallel eval_set"):
        eval_set(**options)


def test_parallel_empty_tasks(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="empty sequence"):
        eval_set([], "scripted", "cubepick", max_workers=2, log_dir=str(tmp_path))


def test_parallel_equivalence_logs_and_resources(
    tmp_path: Path,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    names = ["parallel-a", "parallel-b", "parallel-c"]
    _, serial = eval_set(names, "scripted", "cubepick", log_dir=str(tmp_path / "serial"), seed=19)
    success, parallel = eval_set(
        names,
        "parallel-policy",
        "parallel-sim",
        max_workers=2,
        log_dir=str(tmp_path / "parallel"),
        seed=19,
        store_frames=True,
    )
    assert success
    assert [log.eval.task for log in parallel] == names
    for expected, actual in zip(serial, parallel, strict=True):
        assert actual.results == expected.results
        assert [replace(sample, trial_metadata=()) for sample in actual.samples] == [
            replace(sample, trial_metadata=()) for sample in expected.samples
        ]
        assert actual.eval.seed == expected.eval.seed
    saved = [read_eval_log(str(path)) for path in (tmp_path / "parallel").glob("*.json")]
    assert {log.eval.task for log in saved} == set(names)
    assert len({log.stats.frames_dir for log in parallel}) == 3
    policies, embodiments = components
    assert len(policies) == len(embodiments) == 3
    assert all(cast(Any, resource).closed == 1 for resource in [*policies, *embodiments])


def test_task_failure_keeps_other_results(tmp_path: Path) -> None:
    success, logs = eval_set(
        ["missing-parallel-task", "cubepick-reach"],
        "scripted",
        "cubepick",
        max_workers=2,
        log_dir=str(tmp_path),
    )
    assert not success
    assert [log.status for log in logs] == ["error", "success"]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_parallel_api_preserves_provenance_in_success_and_error_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry

    def fail() -> Task:
        raise RuntimeError("task construction failed")

    monkeypatch.setitem(registry._FACTORIES["task"], "parallel-b", fail)
    success, logs = eval_set(
        ["parallel-a", "parallel-b"],
        "parallel-policy",
        "parallel-sim",
        max_workers=2,
        log_dir=str(tmp_path),
        environment_id="parallel-environment",
        environment_revision="parallel-revision",
        policy_checkpoint="parallel-checkpoint",
    )
    assert not success
    assert [log.status for log in logs] == ["success", "error"]
    assert "task construction failed" in str(logs[1].error)
    saved = [read_eval_log(str(path)) for path in tmp_path.glob("*.json")]
    assert len(saved) == 1
    for log in [*logs, *saved]:
        assert log.eval.environment_id == "parallel-environment"
        assert log.eval.environment_revision == "parallel-revision"
        assert log.eval.policy_checkpoint == "parallel-checkpoint"
    assert all(len(group) == 2 for group in components)
    assert all(cast(Any, resource).closed == 1 for group in components for resource in group)


def test_real_embodiment_never_resets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect_robots.registry as registry

    registered("task")
    closed: list[bool] = []

    class Hardware(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, is_simulated=False)

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            pytest.fail("hardware must never reset")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "parallel-hardware", Hardware)
    success, logs = eval_set(
        "cubepick-reach", "scripted", "parallel-hardware", max_workers=2, log_dir=str(tmp_path)
    )
    assert not success
    assert "simulated embodiment" in str(logs[0].error)
    assert closed == [True]


@pytest.mark.parametrize("error", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_api_propagates_halts_and_closes_resources(
    error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry

    def fail() -> Task:
        raise error("stop")

    monkeypatch.setitem(registry._FACTORIES["task"], "parallel-fail", fail)
    with pytest.raises(error):
        eval_set(
            "parallel-fail", "parallel-policy", "parallel-sim", max_workers=2, log_dir=str(tmp_path)
        )
    assert all(cast(Any, resource).closed == 1 for group in components for resource in group)


@pytest.mark.parametrize("error", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_api_cleanup_failures_preserve_halt_and_stop_admission(
    error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ALL_COMPLETED, wait
    from unittest.mock import patch

    import inspect_robots._parallel as parallel
    import inspect_robots.registry as registry

    registered("task")
    halt = error("stop before another task starts")
    barrier = Barrier(2)
    started: list[int] = []
    policies: list[CleanupPolicy] = []
    embodiments: list[CleanupEmbodiment] = []

    class CleanupPolicy(ScriptedPolicy):
        closed = 0

        def __init__(self) -> None:
            super().__init__()
            policies.append(self)

        def close(self) -> None:
            self.closed += 1
            raise RuntimeError("policy cleanup failed")

    class CleanupEmbodiment(CubePickEmbodiment):
        closed = 0

        def __init__(self) -> None:
            super().__init__()
            embodiments.append(self)

        def close(self) -> None:
            self.closed += 1
            super().close()
            raise RuntimeError("embodiment cleanup failed")

    def make_task(index: int) -> Task:
        started.append(index)
        if index < 2:
            barrier.wait(timeout=5)
        if index == 0:
            raise halt
        return _task(f"cleanup-task-{index}")

    names = [f"cleanup-task-{index}" for index in range(4)]
    for index, name in enumerate(names):
        monkeypatch.setitem(registry._FACTORIES["task"], name, lambda index=index: make_task(index))
    monkeypatch.setitem(registry._FACTORIES["policy"], "failing-cleanup-policy", CleanupPolicy)
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "failing-cleanup-sim", CleanupEmbodiment)

    # Collect the entire first batch together, even if the sibling completes
    # first, to make the scheduler's admission decision deterministic.
    with warnings.catch_warnings(record=True) as cleanup_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        with (
            patch.object(
                parallel, "wait", side_effect=lambda fs, **kw: wait(fs, return_when=ALL_COMPLETED)
            ),
            pytest.raises(error) as raised,
        ):
            eval_set(
                names,
                "failing-cleanup-policy",
                "failing-cleanup-sim",
                max_workers=2,
                log_dir=str(tmp_path),
            )

    assert raised.value is halt
    assert sorted(started) == [0, 1]
    assert len(policies) == len(embodiments) == 2
    assert all(policy.closed == 1 for policy in policies)
    assert all(embodiment.closed == 1 for embodiment in embodiments)
    messages = [str(item.message) for item in cleanup_warnings]
    assert any("RuntimeError: policy cleanup failed" in message for message in messages)
    assert any("RuntimeError: embodiment cleanup failed" in message for message in messages)


@pytest.mark.parametrize("error", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_api_construction_halt_survives_failed_policy_cleanup(
    error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_robots.registry as registry

    registered("task")
    halt = error("embodiment construction halted")
    policies: list[ConstructionPolicy] = []

    class ConstructionPolicy(ScriptedPolicy):
        closed = 0

        def __init__(self) -> None:
            super().__init__()
            policies.append(self)

        def close(self) -> None:
            self.closed += 1
            raise RuntimeError("policy construction cleanup failed")

    def fail() -> CubePickEmbodiment:
        raise halt

    monkeypatch.setitem(
        registry._FACTORIES["policy"], "construction-cleanup-policy", ConstructionPolicy
    )
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "construction-halt-sim", fail)
    with (
        pytest.warns(RuntimeWarning, match="RuntimeError: policy construction cleanup failed"),
        pytest.raises(error) as raised,
    ):
        eval_set(
            "cubepick-reach",
            "construction-cleanup-policy",
            "construction-halt-sim",
            max_workers=2,
            log_dir=str(tmp_path),
        )
    assert raised.value is halt
    assert len(policies) == 1
    assert policies[0].closed == 1


def test_cli_parallel_preserves_options_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    assert (
        main(
            [
                "eval-set",
                "parallel-a",
                "parallel-b",
                "--policy",
                "parallel-policy",
                "--embodiment",
                "parallel-sim",
                "--max-workers",
                "2",
                "--no-prompt",
                "--epochs",
                "2",
                "--seed",
                "31",
                "--store-frames",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    logs = [read_eval_log(str(path)) for path in (tmp_path / "logs").glob("*.json")]
    assert len(logs) == 2
    assert all(log.eval.seed == 31 and log.results.total_trials == 2 for log in logs)
    assert all(log.stats.frames_dir is not None for log in logs)
    assert all(cast(Any, resource).closed == 1 for group in components for resource in group)


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--max-workers", "0"], "positive integer"),
        (["--max-workers", "2"], "requires --no-prompt"),
        (["--max-workers", "2", "--no-prompt", "--voice"], "does not support --voice"),
    ],
)
def test_cli_parallel_rejects_invalid_options(flags: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        main(["eval-set", "cubepick-reach", *flags])


def test_parallel_api_really_overlaps_and_owns_graders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry
    from inspect_robots.rollout import TrialRecord

    barrier = Barrier(2)
    graders: list[object] = []

    class Grader:
        name = "parallel-grader"

        def __init__(self) -> None:
            graders.append(self)

        def grade(self, record: TrialRecord, scene: Scene) -> None:
            barrier.wait(timeout=5)

    monkeypatch.setitem(registry._FACTORIES["grader"], "parallel-grader", Grader)
    success, logs = eval_set(
        ["parallel-a", "parallel-b"],
        "parallel-policy",
        "parallel-sim",
        grader="parallel-grader",
        max_workers=2,
        log_dir=str(tmp_path),
    )
    assert success
    assert len(graders) == 2 and graders[0] is not graders[1]
    assert all(log.eval.grader == "parallel-grader" for log in logs)


@pytest.mark.parametrize("failure_stage", ["factory", "preflight"])
def test_parallel_grader_rejection_stops_before_constructing_components(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry
    from inspect_robots.rollout import TrialRecord

    barrier = Barrier(2)
    factory_calls: list[bool] = []
    preflight_calls: list[object] = []
    task_constructions: list[str] = []

    class RejectingParallelGrader:
        name = "rejecting-parallel-grader"

        def preflight(self) -> None:
            preflight_calls.append(self)
            raise ConfigError("parallel grader preflight rejected")

        def grade(self, record: TrialRecord, scene: Scene) -> None:
            pytest.fail("a rejecting grader must never grade")

    def make_grader() -> RejectingParallelGrader:
        factory_calls.append(True)
        barrier.wait(timeout=5)
        if failure_stage == "factory":
            raise ConfigError("parallel grader factory rejected")
        return RejectingParallelGrader()

    def make_task(name: str) -> Task:
        task_constructions.append(name)
        return _task(name)

    for name in ("parallel-a", "parallel-b", "parallel-c"):
        monkeypatch.setitem(registry._FACTORIES["task"], name, lambda name=name: make_task(name))
    monkeypatch.setitem(registry._FACTORIES["grader"], "rejecting-parallel-grader", make_grader)
    with pytest.raises(ConfigError, match=f"parallel grader {failure_stage} rejected"):
        eval_set(
            ["parallel-a", "parallel-b", "parallel-c", "parallel-a"],
            "parallel-policy",
            "parallel-sim",
            grader="rejecting-parallel-grader",
            max_workers=2,
            log_dir=str(tmp_path),
        )
    assert len(factory_calls) == 2
    assert len(preflight_calls) == (2 if failure_stage == "preflight" else 0)
    assert components == ([], [])
    assert task_constructions == []
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("workers", [1, 2])
def test_unknown_grader_propagates_before_constructing_components(
    workers: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry

    task_constructions: list[str] = []

    def make_task(name: str) -> Task:
        task_constructions.append(name)
        return _task(name)

    for name in ("parallel-a", "parallel-b"):
        monkeypatch.setitem(registry._FACTORIES["task"], name, lambda name=name: make_task(name))
    with pytest.raises(KeyError, match="no grader named 'missing-parallel-grader'"):
        eval_set(
            ["parallel-a", "parallel-b"],
            "parallel-policy",
            "parallel-sim",
            grader="missing-parallel-grader",
            max_workers=workers,
            log_dir=str(tmp_path),
        )
    assert components == ([], [])
    assert task_constructions == []
    assert not list(tmp_path.glob("*.json"))


def test_policy_cleanup_when_embodiment_factory_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry

    def fail() -> CubePickEmbodiment:
        raise ConfigError("cannot construct sim")

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "parallel-sim", fail)
    success, logs = eval_set(
        "parallel-a", "parallel-policy", "parallel-sim", max_workers=2, log_dir=str(tmp_path)
    )
    assert not success and "cannot construct sim" in str(logs[0].error)
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    with pytest.raises(SystemExit, match="cannot construct sim"):
        main(
            [
                "eval-set",
                "parallel-a",
                "--policy",
                "parallel-policy",
                "--embodiment",
                "parallel-sim",
                "--max-workers",
                "2",
                "--no-prompt",
                "--log-dir",
                str(tmp_path),
            ]
        )
    assert len(components[0]) == 2
    assert all(cast(Any, policy).closed == 1 for policy in components[0])


def test_cli_parallel_real_embodiment_rejected_before_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry

    closed: list[bool] = []

    class Hardware(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, is_simulated=False)

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            pytest.fail("hardware must never reset")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "parallel-hardware", Hardware)
    with pytest.raises(SystemExit, match="simulated embodiment"):
        main(
            [
                "eval-set",
                "parallel-a",
                "--policy",
                "parallel-policy",
                "--embodiment",
                "parallel-hardware",
                "--max-workers",
                "2",
                "--no-prompt",
                "--log-dir",
                str(tmp_path),
            ]
        )
    assert closed == [True]
    assert cast(Any, components[0][0]).closed == 1


def test_cli_parallel_policy_without_close_and_failed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    components: tuple[list[ScriptedPolicy], list[CubePickEmbodiment]],
) -> None:
    import inspect_robots.registry as registry

    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "missing.ini"))
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scorer import Score

    class FailingScorer:
        name = "failing"

        def __call__(self, record: TrialRecord, target: Target | None) -> Score:
            raise ValueError("scorer failed")

    monkeypatch.setitem(
        registry._FACTORIES["task"],
        "parallel-b",
        lambda: Task(
            name="bad",
            scenes=[Scene(id="s", instruction="reach")],
            scorer=FailingScorer(),
            max_steps=1,
        ),
    )
    assert (
        main(
            [
                "eval-set",
                "parallel-a",
                "parallel-b",
                "--policy",
                "scripted",
                "--embodiment",
                "cubepick",
                "--max-workers",
                "2",
                "--no-prompt",
                "--no-live-log",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 1
    )
    logs = [read_eval_log(str(path)) for path in (tmp_path / "logs").glob("*.json")]
    assert sorted(log.status for log in logs) == ["error", "success"]
