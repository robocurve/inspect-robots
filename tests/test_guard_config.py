"""Guards as configuration: the guard registry kind, specs, the CLI and config, and the record."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots import eval_set
from inspect_robots.cli import _build_guardrails, main
from inspect_robots.defaults import load_defaults
from inspect_robots.errors import ConfigError
from inspect_robots.guard import (
    ClampGuard,
    DeltaLimitGuard,
    GuardChain,
    PassThroughGuard,
    build_guard,
    guard_name,
    parse_guard_spec,
)
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import registered
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation

_SPACE = CubePickEmbodiment().info.action_space


class _Overshooter:
    """Asks for a target far outside CubePick's bounds, every step."""

    def __init__(self) -> None:
        self.info = PolicyInfo(name="overshooter", action_space=_SPACE)
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.array([5.0, -5.0]))])


def _task(max_steps: int = 3) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


# -- specs and the registry ----------------------------------------------------


def test_the_built_in_guards_are_registered() -> None:
    assert {"clamp", "delta-limit"} <= set(registered("guard"))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("clamp", ("clamp", {})),
        ("delta-limit:max_delta=0.02", ("delta-limit", {"max_delta": 0.02})),
        (" box : low=-1, high=1 ,", ("box", {"low": -1, "high": 1})),
        ("x:label='none'", ("x", {"label": "none"})),
    ],
)
def test_guard_specs_parse_like_cli_values(spec: str, expected: tuple[str, Any]) -> None:
    assert parse_guard_spec(spec) == expected


@pytest.mark.parametrize(
    ("spec", "match"),
    [(":k=v", "has no name"), ("clamp:oops", "is not key=value"), ("clamp:=1", "not key=value")],
)
def test_malformed_guard_specs_say_how_to_write_one(spec: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_guard_spec(spec)


def test_build_guard_resolves_specs_and_keeps_objects_in_order() -> None:
    passthrough = PassThroughGuard()
    chain = build_guard(["clamp", "delta-limit:max_delta=0.01", passthrough], _SPACE)
    assert isinstance(chain, GuardChain)
    assert chain.names == ("clamp", "delta-limit", "pass_through")
    assert isinstance(chain.guards[0], ClampGuard)
    assert isinstance(chain.guards[1], DeltaLimitGuard)
    assert chain.guards[2] is passthrough


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("teleport", "unknown guard 'teleport'; registered guards: .*clamp"),
        ("clamp:speed=2", "guard 'clamp:speed=2'"),
        ("delta-limit:max_delta=-1", "guard 'delta-limit:max_delta=-1'"),
    ],
)
def test_build_guard_names_what_went_wrong(spec: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        build_guard([spec], _SPACE)


def test_a_factory_that_returns_no_guard_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_robots import registry

    monkeypatch.setitem(registry._FACTORIES["guard"], "broken", lambda action_space: object())
    with pytest.raises(ConfigError, match="has no review"):
        build_guard(["broken"], _SPACE)


def test_guard_names_fall_back_to_the_class() -> None:
    class Custom:
        def review(self, action: Action, store: dict[str, Any]) -> Action:
            return action

    assert guard_name(Custom()) == "Custom"
    assert guard_name(ClampGuard(_SPACE)) == "clamp"
    with pytest.raises(ValueError, match="1 names for 2 guards"):
        GuardChain(Custom(), Custom(), names=["a"])


# -- the record --------------------------------------------------------------


def test_the_record_names_each_guard_that_changed_an_action(tmp_path: Path) -> None:
    (log,) = ir_eval(
        _task(max_steps=1),
        _Overshooter(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        guard=["clamp", "delta-limit:max_delta=0.01"],
    )
    (events,) = log.samples[0].events
    (approval,) = [e for e in events if e["kind"] == "approval"]
    clamp, delta = approval["data"]["guards"]
    assert clamp["guard"] == "clamp"
    assert clamp["before"] == [5.0, -5.0]
    assert clamp["after"] == [float(v) for v in np.clip([5.0, -5.0], _SPACE.low, _SPACE.high)]
    assert delta["guard"] == "delta-limit"
    assert delta["before"] == clamp["after"]
    assert max(abs(a) for a in delta["after"]) <= 0.01 + 1e-12


def test_a_single_guard_is_named_in_the_record_too(tmp_path: Path) -> None:
    (log,) = ir_eval(
        _task(max_steps=1),
        _Overshooter(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        guard=ClampGuard(_SPACE),
    )
    (events,) = log.samples[0].events
    (approval,) = [e for e in events if e["kind"] == "approval"]
    assert [change["guard"] for change in approval["data"]["guards"]] == ["clamp"]


def test_parallel_eval_set_accepts_guard_specs(tmp_path: Path) -> None:
    success, logs = eval_set(
        ["cubepick-reach"],
        "scripted",
        "cubepick",
        log_dir=str(tmp_path),
        max_workers=2,
        guard=["clamp"],
    )
    assert success and len(logs) == 1
    with pytest.raises(ConfigError, match="guard objects"):
        eval_set(
            ["cubepick-reach"],
            "scripted",
            "cubepick",
            log_dir=str(tmp_path),
            max_workers=2,
            guard=[ClampGuard(_SPACE)],
        )


# -- the CLI and the config file ------------------------------------------------


def test_configured_guards_follow_the_defaults() -> None:
    guard, active, warnings = _build_guardrails(_SPACE, None, None, ["delta-limit:max_delta=0.01"])
    assert active == ["clamp", "delta-limit", "delta-limit"]
    assert isinstance(guard, GuardChain) and guard.names == tuple(active)
    assert warnings == []


def test_without_the_defaults_only_configured_guards_run() -> None:
    guard, active, _ = _build_guardrails(_SPACE, None, None, ["clamp"], defaults=False)
    assert active == ["clamp"]
    guard, active, warnings = _build_guardrails(_SPACE, None, None, [], defaults=False)
    assert isinstance(guard, PassThroughGuard) and active == []
    assert "no guardrails are active" in warnings[0]


def test_a_bad_configured_guard_stops_the_cli() -> None:
    with pytest.raises(SystemExit, match="unknown guard 'nope'"):
        _build_guardrails(_SPACE, None, None, ["nope"])


def test_the_guards_config_section_lists_specs_in_order(tmp_path: Path) -> None:
    config = tmp_path / "config.ini"
    config.write_text("[guards]\ndelta-limit = max_delta=0.02\nclamp =\n")
    defaults = load_defaults({"INSPECT_ROBOTS_CONFIG": str(config)})
    assert defaults.guards == ("delta-limit:max_delta=0.02", "clamp")


def _run(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str) -> None:
    path = tmp_path / "config.ini"
    path.write_text(config)
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(path))
    monkeypatch.chdir(tmp_path)
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
                "--no-prompt",
                *argv,
            ]
        )
        == 0
    )


def test_cli_guards_from_flags_and_config_are_announced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(
        ["--guard", "delta-limit:max_delta=0.05"],
        tmp_path,
        monkeypatch,
        "[guards]\nclamp =\n",
    )
    assert "guardrails: clamp + delta-limit + clamp + delta-limit" in capsys.readouterr().out


def test_cli_disabling_the_defaults_keeps_configured_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(["--disable-guardrails", "--guard", "clamp"], tmp_path, monkeypatch, "")
    captured = capsys.readouterr()
    assert "guardrails: clamp\n" in captured.out
    assert "only the configured guards below check actions" in captured.err


def test_cli_lists_guards(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "guards"]) == 0
    out = capsys.readouterr().out
    assert "clamp" in out and "delta-limit" in out
