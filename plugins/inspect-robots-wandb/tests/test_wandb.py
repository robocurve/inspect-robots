"""Tests for the Weights & Biases logging sink."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import NoReturn

import pytest

from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats
from inspect_robots_wandb import WandbSink


class _FakeRun:
    """Record the W&B calls made by one sink lifecycle."""

    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.finished_with: list[int | None] = []

    def log(self, data: dict[str, object]) -> None:
        """Record one payload sent to W&B."""
        self.logged.append(data)

    def finish(self, *, exit_code: int | None = None) -> None:
        """Record the exit status used to finish the W&B run."""
        self.finished_with.append(exit_code)


class _FakeWandb(ModuleType):
    """Provide the small W&B module surface exercised by the sink."""

    def __init__(self) -> None:
        super().__init__("wandb")
        self.init_calls: list[dict[str, object]] = []
        self.runs: list[_FakeRun] = []

    def init(self, **kwargs: object) -> _FakeRun:
        """Record initialization arguments and return a fake run."""
        self.init_calls.append(kwargs)
        run = _FakeRun()
        self.runs.append(run)
        return run


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _FakeWandb:
    """Install a fake W&B module without importing the external SDK."""
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _spec() -> EvalSpec:
    """Build a representative immutable evaluation specification."""
    return EvalSpec(
        task="cubepick-reach",
        policy="scripted",
        embodiment="cubepick",
        created="2026-09-02T12:00:00+00:00",
        inspect_robots_version="0.57.1",
        git_commit="abc123",
        policy_config={"temperature": 0.2},
        embodiment_info={"control_hz": 10.0},
        seed=7,
        max_steps=80,
        max_seconds=None,
    )


def _log(status: str = "success") -> EvalLog:
    """Build a representative final evaluation log."""
    return EvalLog(
        version=1,
        status=status,
        eval=_spec(),
        results=EvalResults(
            total_scenes=2,
            total_trials=3,
            metrics={"success_at_end": 0.5, "episode_length": 11.0},
            errored_trials=1,
        ),
        stats=EvalStats(
            started_at="2026-09-02T12:00:00+00:00",
            completed_at="2026-09-02T12:00:12+00:00",
            duration_s=12.5,
            total_steps=27,
        ),
    )


def test_sink_initializes_wandb_with_spec_and_logs_final_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sink sends reproducibility config and aggregate scalar metrics."""
    fake = _install_fake_wandb(monkeypatch)

    sink = WandbSink(
        project="robot-evals",
        entity="robotics-team",
        name="reach-run",
        group="nightly",
        tags=("cube", "nightly"),
        mode="offline",
        dir="wandb-data",
    )
    sink.on_eval_start(_spec())
    sink.on_eval_end(_log())

    assert fake.init_calls == [
        {
            "project": "robot-evals",
            "entity": "robotics-team",
            "name": "reach-run",
            "group": "nightly",
            "tags": ["cube", "nightly"],
            "mode": "offline",
            "dir": "wandb-data",
            "config": {
                "task": "cubepick-reach",
                "policy": "scripted",
                "embodiment": "cubepick",
                "created": "2026-09-02T12:00:00+00:00",
                "inspect_robots_version": "0.57.1",
                "git_commit": "abc123",
                "policy_config": {"temperature": 0.2},
                "embodiment_info": {"control_hz": 10.0},
                "seed": 7,
                "max_steps": 80,
                "max_seconds": None,
            },
        }
    ]
    assert fake.runs[0].logged == [
        {
            "eval/status": "success",
            "eval/total_scenes": 2,
            "eval/total_trials": 3,
            "eval/errored_trials": 1,
            "eval/total_steps": 27,
            "eval/duration_s": 12.5,
            "metric/success_at_end": 0.5,
            "metric/episode_length": 11.0,
        }
    ]
    assert fake.runs[0].finished_with == [0]


def test_sink_finishes_failed_evaluation_with_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled or errored evaluation is marked failed in W&B."""
    fake = _install_fake_wandb(monkeypatch)

    sink = WandbSink(mode="disabled")
    sink.on_eval_start(_spec())
    sink.on_eval_end(_log("cancelled"))

    assert fake.runs[0].logged[0]["eval/status"] == "cancelled"
    assert fake.runs[0].finished_with == [1]


def test_sink_explains_how_to_install_missing_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing SDK produces an actionable error at evaluation start."""
    import inspect_robots_wandb._sink as sink_module

    def missing_import(name: str) -> NoReturn:
        """Simulate an environment without the W&B SDK."""
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(sink_module, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="pip install inspect-robots-wandb"):
        WandbSink().on_eval_start(_spec())


def test_sink_rejects_an_empty_project_name() -> None:
    """A blank project cannot produce a useful W&B run."""
    with pytest.raises(ValueError, match="project must not be empty"):
        WandbSink(project=" ")


def test_sink_ignores_eval_end_before_start() -> None:
    """A sink with no active run remains safe to finalize."""
    WandbSink().on_eval_end(_log())


def test_sink_can_be_reused_for_sequential_eval_set_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each eval-set task gets an independent W&B run and finalization."""
    fake = _install_fake_wandb(monkeypatch)

    sink = WandbSink(mode="offline")
    sink.on_eval_start(_spec())
    sink.on_eval_end(_log())
    sink.on_eval_start(_spec())
    sink.on_eval_end(_log())

    assert len(fake.init_calls) == 2
    assert [run.finished_with for run in fake.runs] == [[0], [0]]


def test_wandb_sink_entry_point_loads_factory() -> None:
    """Installing the plugin exposes the sink through the framework registry."""
    from importlib.metadata import entry_points

    matches = [
        entry for entry in entry_points(group="inspect_robots.sinks") if entry.name == "wandb"
    ]

    assert len(matches) == 1
    assert matches[0].value == "inspect_robots_wandb:WandbSink"
    assert matches[0].load() is WandbSink
