"""Unit tests for WandbSink."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats
from inspect_robots_wandb.sink import WandbSink, wandb_sink


def _make_spec() -> EvalSpec:
    return EvalSpec(
        task="cube_pick",
        policy="scripted",
        embodiment="cubepick",
        created="2026-01-01T00:00:00Z",
        inspect_robots_version="0.30.0",
        max_steps=10,
    )


def _make_log() -> EvalLog:
    return EvalLog(
        version=1,
        status="success",
        eval=_make_spec(),
        results=EvalResults(
            total_scenes=1,
            total_trials=1,
            metrics={"success": 1.0},
            errored_trials=0,
        ),
        stats=EvalStats(
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:05Z",
            duration_s=5.0,
            total_steps=10,
        ),
        samples=(),
    )


def test_wandb_sink_absent_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify WandbSink warns and becomes a no-op when wandb is missing."""

    def _mock_find_spec(name: str) -> None:
        if name == "wandb":
            return None
        return importlib.util.find_spec(name)  # type: ignore[return-value]

    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec)
    sink = WandbSink()
    with pytest.warns(RuntimeWarning, match="wandb is not installed"):
        sink.on_eval_start(_make_spec())
    sink.on_eval_end(_make_log())


def test_wandb_sink_import_wandb_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify WandbSink imports and caches real/mocked wandb module."""
    fake_mod = ModuleType("wandb")
    mock_run = MagicMock()
    fake_mod.init = MagicMock(return_value=mock_run)  # type: ignore[attr-defined]

    def _find_wandb(name: str) -> MagicMock | None:
        return MagicMock() if name == "wandb" else None

    monkeypatch.setitem(sys.modules, "wandb", fake_mod)
    monkeypatch.setattr(importlib.util, "find_spec", _find_wandb)

    sink = WandbSink()
    sink.on_eval_start(_make_spec())
    assert sink._ensure_wandb() is fake_mod
    sink.on_eval_end(_make_log())
    mock_run.log.assert_called_once()
    mock_run.finish.assert_called_once()


def test_wandb_sink_mocked_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify WandbSink initializes, logs metrics, and finishes run."""
    mock_run = MagicMock()
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    sink = WandbSink(project="test-proj")
    monkeypatch.setattr(sink, "_ensure_wandb", lambda: mock_wandb)

    spec = _make_spec()
    log = _make_log()

    sink.on_eval_start(spec)
    mock_wandb.init.assert_called_once()
    config = mock_wandb.init.call_args.kwargs["config"]
    assert config["task"] == "cube_pick"

    sink.on_eval_end(log)
    mock_run.log.assert_called_once()
    mock_run.finish.assert_called_once()


def test_wandb_sink_factory() -> None:
    """Verify factory entry point constructs WandbSink."""
    sink = wandb_sink(project="my-proj", name="my-run")
    assert isinstance(sink, WandbSink)
    assert sink.project == "my-proj"
    assert sink.name == "my-run"
