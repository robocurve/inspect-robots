"""Benchmarks: a named, versioned sequence of tasks.

The paper defines a benchmark as a sequence of tasks. A
[`Benchmark`][inspect_robots.benchmark.Benchmark] adds the identity a result
needs to be cited: a name and a version, recorded in every log it produces as
``EvalSpec.benchmark``. Benchmarks are a registry kind (``@benchmark`` from
``inspect_robots.registry``, entry-point group ``inspect_robots.benchmarks``) and
ship in their own packages, outside the core.

Run one with [`eval_benchmark`][inspect_robots.benchmark.eval_benchmark] or
``inspect-robots run --benchmark NAME``; it is
[`eval_set`][inspect_robots.eval.eval_set] over the benchmark's tasks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from inspect_robots.errors import ConfigError
from inspect_robots.task import Task

if TYPE_CHECKING:
    from inspect_robots.embodiment import Embodiment
    from inspect_robots.log import EvalLog
    from inspect_robots.policy import Policy


@dataclass(frozen=True)
class Benchmark:
    """A named, versioned sequence of tasks, each a ``Task`` or a registered task name.

    Bump ``version`` whenever a task, scene, rubric or budget changes, so
    results from different versions are never pooled. ``metadata`` is free-form
    (authors, a paper, a license) and is not recorded in logs.
    """

    name: str
    tasks: Sequence[Task | str]
    version: str = "1"
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("a benchmark needs a non-empty name")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ConfigError(f"benchmark {self.name!r}: version must be a non-empty string")
        tasks = tuple(self.tasks)
        if not tasks:
            raise ConfigError(f"benchmark {self.name!r} has no tasks")
        for task in tasks:
            if not isinstance(task, (Task, str)):
                raise ConfigError(
                    f"benchmark {self.name!r}: tasks must be Task objects or registered "
                    f"task names, got {task!r}"
                )
        names = self.task_names
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ConfigError(f"benchmark {self.name!r} lists task(s) twice: {', '.join(repeated)}")
        object.__setattr__(self, "tasks", tasks)

    @property
    def task_names(self) -> list[str]:
        """Each task's name, in benchmark order."""
        return [task if isinstance(task, str) else task.name for task in self.tasks]

    def describe(self) -> dict[str, str]:
        """The identity recorded in each log: ``{"name", "version"}``."""
        return {"name": self.name, "version": self.version}


def resolve_benchmark(benchmark: Benchmark | str) -> Benchmark:
    """``benchmark`` itself, or the registered benchmark of that name."""
    if isinstance(benchmark, Benchmark):
        return benchmark
    from inspect_robots.registry import registered, resolve

    if benchmark not in registered("benchmark"):
        known = ", ".join(sorted(registered("benchmark"))) or "(none installed)"
        raise ConfigError(f"unknown benchmark {benchmark!r}; registered benchmarks: {known}")
    resolved = resolve("benchmark", benchmark)
    if not isinstance(resolved, Benchmark):
        raise ConfigError(
            f"benchmark {benchmark!r} factory returned {type(resolved).__name__}; "
            "expected Benchmark"
        )
    return resolved


def eval_benchmark(
    benchmark: Benchmark | str,
    policy: Policy | str,
    embodiment: Embodiment | str,
    **kwargs: Any,
) -> tuple[bool, list[EvalLog]]:
    """Run every task of ``benchmark`` with ``eval_set``; each log records the benchmark.

    ``kwargs`` are [`eval_set`][inspect_robots.eval.eval_set]'s keyword
    arguments. Returns ``(success, logs)``, one log per task.
    """
    from inspect_robots.eval import eval_set

    resolved = resolve_benchmark(benchmark)
    return eval_set(list(resolved.tasks), policy, embodiment, benchmark=resolved, **kwargs)
