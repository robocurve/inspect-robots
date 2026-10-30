"""Benchmarks: named, versioned task sequences, run end to end and recorded in each log."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from inspect_robots import Scene, Task
from inspect_robots.analyzer import AnnotatedLog
from inspect_robots.benchmark import Benchmark, eval_benchmark, resolve_benchmark
from inspect_robots.cli import main
from inspect_robots.errors import ConfigError
from inspect_robots.log import read_eval_log
from inspect_robots.registry import _FACTORIES, resolve
from inspect_robots.scorer import success_at_end


def _task(name: str) -> Task:
    return Task(
        name=name,
        scenes=[Scene(id="s", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=20,
    )


@pytest.fixture
def registered_benchmark() -> Iterator[Benchmark]:
    bench = Benchmark(
        name="reach-suite",
        tasks=["cubepick-reach", _task("reach-short")],
        version="2",
        description="Two reaching tasks.",
    )
    from inspect_robots.registry import benchmark

    @benchmark("reach-suite")
    def reach_suite() -> Benchmark:
        return bench

    yield bench
    _FACTORIES["benchmark"].pop("reach-suite", None)


# -- the type ----------------------------------------------------------------


def test_a_benchmark_names_its_tasks_in_order() -> None:
    bench = Benchmark(name="b", tasks=["cubepick-reach", _task("x")])
    assert bench.task_names == ["cubepick-reach", "x"]
    assert isinstance(bench.tasks, tuple)
    assert bench.describe() == {"name": "b", "version": "1"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": " ", "tasks": ["a"]}, "non-empty name"),
        ({"name": "b", "tasks": ["a"], "version": ""}, "version must be"),
        ({"name": "b", "tasks": []}, "has no tasks"),
        ({"name": "b", "tasks": [3]}, "Task objects or registered task names"),
        ({"name": "b", "tasks": ["a", _task("a")]}, "lists task\\(s\\) twice: a"),
    ],
)
def test_benchmarks_are_validated(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Benchmark(**kwargs)


def test_resolving_by_name(registered_benchmark: Benchmark) -> None:
    assert resolve_benchmark("reach-suite") is registered_benchmark
    assert resolve_benchmark(registered_benchmark) is registered_benchmark
    with pytest.raises(ConfigError, match="unknown benchmark 'nope'; registered benchmarks: "):
        resolve_benchmark("nope")


def test_a_factory_must_return_a_benchmark() -> None:
    _FACTORIES["benchmark"]["broken"] = lambda: "not a benchmark"
    try:
        with pytest.raises(ConfigError, match="returned str; expected Benchmark"):
            resolve_benchmark("broken")
    finally:
        _FACTORIES["benchmark"].pop("broken")


# -- running -----------------------------------------------------------------


def test_eval_benchmark_runs_every_task_and_stamps_each_log(
    tmp_path: Path, registered_benchmark: Benchmark
) -> None:
    success, logs = eval_benchmark("reach-suite", "scripted", "cubepick", log_dir=str(tmp_path))
    assert success
    assert [log.eval.task for log in logs] == ["cubepick-reach", "reach-short"]
    assert all(log.eval.benchmark == {"name": "reach-suite", "version": "2"} for log in logs)
    reread = [read_eval_log(str(path)) for path in sorted(tmp_path.glob("*.json"))]
    assert all(log.eval.benchmark == {"name": "reach-suite", "version": "2"} for log in reread)


def test_a_task_that_fails_to_start_is_still_stamped(tmp_path: Path) -> None:
    bench = Benchmark(name="b", tasks=["no-such-task"])
    success, (log,) = eval_benchmark(bench, "scripted", "cubepick", log_dir=str(tmp_path))
    assert not success and log.status == "error"
    assert log.eval.benchmark == {"name": "b", "version": "1"}


def test_parallel_benchmark_runs_are_stamped(tmp_path: Path) -> None:
    bench = Benchmark(name="b", tasks=["cubepick-reach"], version="3")
    success, (log,) = eval_benchmark(
        bench, "scripted", "cubepick", log_dir=str(tmp_path), max_workers=2
    )
    assert success and log.eval.benchmark == {"name": "b", "version": "3"}


def test_parallel_errors_are_stamped(tmp_path: Path) -> None:
    bench = Benchmark(name="b", tasks=["no-such-task"])
    success, (log,) = eval_benchmark(
        bench, "scripted", "cubepick", log_dir=str(tmp_path), max_workers=2
    )
    assert not success and log.eval.benchmark == {"name": "b", "version": "1"}


def test_the_benchmarks_analyzer_groups_results(
    tmp_path: Path, registered_benchmark: Benchmark
) -> None:
    eval_benchmark("reach-suite", "scripted", "cubepick", log_dir=str(tmp_path / "bench"))
    eval_benchmark(
        Benchmark(name="reach-suite", tasks=["cubepick-reach"], version="3"),
        "scripted",
        "cubepick",
        log_dir=str(tmp_path / "bench"),
    )
    paths = sorted((tmp_path / "bench").glob("*.json"))
    analysis = resolve("analyzer", "benchmarks").analyze([AnnotatedLog.load(p) for p in paths])
    assert set(analysis.data) == {"reach-suite v2", "reach-suite v3"}
    assert len(analysis.data["reach-suite v2"]["tasks"]) == 2
    assert "reach-suite v2: 2 task(s)" in analysis.text
    assert "cubepick-reach [success] success_at_end" in analysis.text

    from inspect_robots import eval as ir_eval

    (plain,) = ir_eval("cubepick-reach", "scripted", "cubepick", log_dir=str(tmp_path / "plain"))
    (path,) = (tmp_path / "plain").glob("*.json")
    empty = resolve("analyzer", "benchmarks").analyze([AnnotatedLog.load(path)])
    assert empty.text == "no benchmark runs among these logs\n" and plain.eval.benchmark is None


# -- the CLI -----------------------------------------------------------------


def _cli(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(tmp_path / "none.ini"))
    return main([*argv, "--policy", "scripted", "--embodiment", "cubepick", "--no-prompt"])


def test_cli_run_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    registered_benchmark: Benchmark,
) -> None:
    assert _cli(["run", "--benchmark", "reach-suite"], tmp_path, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "benchmark: reach-suite v2" in out
    assert "tasks: cubepick-reach, reach-short" in out
    logs = [read_eval_log(str(p)) for p in (tmp_path / "logs").glob("*.json")]
    assert len(logs) == 2 and all(log.eval.benchmark for log in logs)


def test_cli_eval_set_benchmark_and_its_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_benchmark: Benchmark,
) -> None:
    assert _cli(["eval-set", "--benchmark", "reach-suite"], tmp_path, monkeypatch) == 0
    with pytest.raises(SystemExit, match="task names or --benchmark, not both"):
        _cli(["eval-set", "cubepick-reach", "--benchmark", "reach-suite"], tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="eval-set needs task names or --benchmark NAME"):
        _cli(["eval-set"], tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="unknown benchmark 'nope'"):
        _cli(["eval-set", "--benchmark", "nope"], tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="--benchmark runs its own tasks"):
        _cli(
            ["run", "--benchmark", "reach-suite", "--task", "cubepick-reach"], tmp_path, monkeypatch
        )


def test_cli_parallel_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_benchmark: Benchmark
) -> None:
    argv = ["eval-set", "--benchmark", "reach-suite", "--max-workers", "2"]
    assert _cli(argv, tmp_path, monkeypatch) == 0


def test_cli_lists_benchmarks(
    capsys: pytest.CaptureFixture[str], registered_benchmark: Benchmark
) -> None:
    assert main(["list", "benchmarks"]) == 0
    assert "reach-suite" in capsys.readouterr().out
