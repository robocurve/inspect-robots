"""Targeted tests closing the remaining coverage gaps to reach 100%.

Each test names the behavior it exercises; together with the rest of the suite
these drive line + branch coverage of ``robolens`` to 100% (enforced by
``--cov-fail-under=100``).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest

import robolens.controller as controller_mod
import robolens.registry as reg
from robolens import eval
from robolens.approver import AutoApprover, ClampApprover
from robolens.cli import _parse_kvs, _parse_value, main
from robolens.compat import check_compatibility
from robolens.controller import DefaultController, EnsemblingController, SmoothingController
from robolens.embodiment import EmbodimentInfo
from robolens.errors import EmbodimentFault, PolicyError
from robolens.eval import _git_commit, _should_fail
from robolens.log import EvalLog, EvalResults, EvalSpec, EvalStats
from robolens.logging.sink import NullSink
from robolens.mock import CubePickEmbodiment, NoopPolicy, ScriptedPolicy
from robolens.policy import PolicyConfig, PolicyInfo
from robolens.registry import register, registered, resolve
from robolens.rollout import StepRecord, TrialRecord, _effective_control_hz, rollout
from robolens.scene import ListSceneDataset, Scene
from robolens.scorer import min_distance_to_goal, success_at_end, value_to_float
from robolens.spaces import ActionSemantics, Box, ObservationSpace
from robolens.task import Task
from robolens.transcript import approval_event, operator_event
from robolens.types import Action, ActionChunk, Observation, StepResult

_SCENE = Scene(id="s", instruction="reach", init_seed=0)
_CUBE_SEM = ActionSemantics(control_mode="eef_delta_pos", frame="world")


def _task(*scenes: Scene, max_steps: int = 10) -> Task:
    return Task(
        name="t",
        scenes=list(scenes) or [_SCENE],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


# --------------------------------------------------------------------------- #
# approver
# --------------------------------------------------------------------------- #
def test_clamp_approver_passthrough_when_unbounded() -> None:
    approver = ClampApprover(Box(shape=(2,)))  # no low/high
    action = Action(data=np.array([9.0, 9.0]))
    assert approver.review(action, {}) is action


def test_clamp_approver_passthrough_when_in_bounds() -> None:
    approver = ClampApprover(Box(shape=(2,), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])))
    action = Action(data=np.array([0.5, -0.5]))
    assert approver.review(action, {}) is action


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def test_parse_value_variants() -> None:
    assert _parse_value("true") is True
    assert _parse_value("false") is False
    assert _parse_value("none") is None
    assert _parse_value("null") is None
    assert _parse_value("7") == 7
    assert _parse_value("1.5") == 1.5
    assert _parse_value("hello") == "hello"


def test_parse_kvs_rejects_bad_pair() -> None:
    with pytest.raises(SystemExit):
        _parse_kvs(["no-equals-sign"])


def test_cli_list_empty_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("robolens.registry.registered", lambda kind: {})
    assert main(["list", "sinks"]) == 0
    assert "(none)" in capsys.readouterr().out


def test_cli_inspect_error_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = EvalLog(
        version=1,
        status="error",
        eval=EvalSpec(task="t", policy="p", embodiment="e", created="x", robolens_version="0"),
        results=EvalResults(total_scenes=0, total_trials=0),
        stats=EvalStats(started_at="a", completed_at="b", duration_s=0.0, total_steps=0),
        samples=[],
        error="boom",
    )
    path = tmp_path / "err.json"
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")
    assert main(["inspect", str(path)]) == 1
    assert "error: boom" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# controller
# --------------------------------------------------------------------------- #
def test_controller_validation_raises() -> None:
    with pytest.raises(ValueError):
        DefaultController(replan_interval=0)
    with pytest.raises(ValueError):
        SmoothingController(DefaultController(), alpha=0.0)
    with pytest.raises(ValueError):
        EnsemblingController(Box(shape=(2,), semantics=_CUBE_SEM), m=-1.0)


def test_ensembling_warns_per_instance() -> None:
    # No global warn-once state: every instance constructed without semantics
    # warns, independent of construction order elsewhere in the process.
    with pytest.warns(RuntimeWarning):
        EnsemblingController(Box(shape=(2,)))
    with pytest.warns(RuntimeWarning):
        EnsemblingController(Box(shape=(2,)))


def test_default_controller_records_actual_buffered_count() -> None:
    # replan_interval larger than the chunk: the recorded per-inference count is
    # the number of actions actually buffered, not the requested interval.
    policy = ScriptedPolicy(chunk_size=4)
    embodiment = CubePickEmbodiment()
    obs = embodiment.reset(_SCENE, seed=0)
    store: dict[str, object] = {}
    DefaultController(replan_interval=10).next_action(policy, obs, 0, store)
    inferences = store[controller_mod._INFER_KEY]
    assert inferences[-1][1] == 4  # type: ignore[index]


# --------------------------------------------------------------------------- #
# eval helpers + halt path
# --------------------------------------------------------------------------- #
def test_git_commit_handles_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> object:
        raise OSError("no git here")

    monkeypatch.setattr("subprocess.run", boom)
    assert _git_commit() is None


def test_git_commit_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Result())
    assert _git_commit() is None


def test_should_fail_branches() -> None:
    assert _should_fail(True, 1, 1) is True
    assert _should_fail(2, 2, 5) is True
    assert _should_fail(2, 1, 5) is False


class _FaultEmbodiment(CubePickEmbodiment):
    def step(self, action: Action) -> StepResult:
        raise EmbodimentFault("motor stalled")


def test_eval_halts_on_embodiment_fault(tmp_path: Path) -> None:
    task = _task(
        Scene(id="s0", instruction="x", init_seed=0),
        Scene(id="s1", instruction="x", init_seed=1),
    )
    (log,) = eval(task, ScriptedPolicy(), _FaultEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "error"
    assert log.results.metrics == {}  # no scene was scored before the halt


# --------------------------------------------------------------------------- #
# mock world / policies
# --------------------------------------------------------------------------- #
def test_cubepick_close() -> None:
    CubePickEmbodiment().close()


def test_noop_policy_never_succeeds(tmp_path: Path) -> None:
    (log,) = eval(_task(max_steps=8), NoopPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.results.metrics["success_at_end"] == 0.0


# --------------------------------------------------------------------------- #
# scorer
# --------------------------------------------------------------------------- #
def test_value_to_float_string_paths() -> None:
    assert value_to_float("3.5") == 3.5
    assert value_to_float("not-a-number") == 0.0


def test_min_distance_without_signal() -> None:
    record = TrialRecord(
        scene_id="s",
        epoch=0,
        seed=0,
        steps=[
            StepRecord(
                t=0,
                observation=Observation(),
                action=Action(data=np.zeros(2)),
                result=StepResult(observation=Observation(), info={}),
            )
        ],
    )
    assert min_distance_to_goal()(record, None).value == float("inf")


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def test_effective_control_hz_precedence() -> None:
    # First non-None of chunk -> task -> embodiment (R1).
    assert _effective_control_hz(None, None, None) is None
    assert _effective_control_hz(30.0, 20.0, 10.0) == 30.0
    assert _effective_control_hz(None, 20.0, 10.0) == 20.0
    assert _effective_control_hz(None, None, 10.0) == 10.0


class _PolicyErrorPolicy:
    def __init__(self) -> None:
        self.info = PolicyInfo(name="perr", action_space=Box(shape=(2,), semantics=_CUBE_SEM))
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        raise PolicyError("typed failure")


def _rollout(policy: object, embodiment: object):  # type: ignore[no-untyped-def]
    return rollout(
        policy,  # type: ignore[arg-type]
        embodiment,  # type: ignore[arg-type]
        _SCENE,
        max_steps=20,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
    )


def test_rollout_reraises_typed_policy_error() -> None:
    with pytest.raises(PolicyError, match="typed failure"):
        _rollout(_PolicyErrorPolicy(), CubePickEmbodiment())


def test_rollout_reraises_typed_embodiment_fault() -> None:
    with pytest.raises(EmbodimentFault, match="motor stalled"):
        _rollout(ScriptedPolicy(), _FaultEmbodiment())


class _TruncatingEmbodiment(CubePickEmbodiment):
    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=False,
            truncated=True,
            info=result.info,
        )


def test_rollout_truncation_path() -> None:
    record = _rollout(ScriptedPolicy(), _TruncatingEmbodiment())
    assert record.truncated is True
    assert record.termination_reason == "truncated"


# --------------------------------------------------------------------------- #
# scene dataset
# --------------------------------------------------------------------------- #
def test_list_scene_dataset() -> None:
    ds = ListSceneDataset([Scene(id="a", instruction="x"), Scene(id="b", instruction="y")])
    assert len(ds) == 2
    assert [s.id for s in ds] == ["a", "b"]


# --------------------------------------------------------------------------- #
# transcript helpers
# --------------------------------------------------------------------------- #
def test_transcript_event_helpers() -> None:
    assert approval_event(1, modified=True, detail="clamped").kind == "approval"
    assert operator_event(2, "success").data["verdict"] == "success"


# --------------------------------------------------------------------------- #
# logging sink
# --------------------------------------------------------------------------- #
def test_null_sink_lifecycle() -> None:
    sink = NullSink()
    obs = Observation()
    record = TrialRecord(scene_id="s", epoch=0, seed=0)
    sink.on_eval_start(None)  # type: ignore[arg-type]
    sink.on_trial_start("s", 0)
    sink.log_step(0, obs, Action(data=np.zeros(2)), StepResult(observation=obs))
    sink.on_trial_end(record)
    sink.on_eval_end(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# registry error paths
# --------------------------------------------------------------------------- #
def test_register_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown registry kind"):
        register("bogus")


def test_register_without_resolvable_name() -> None:
    nameless = functools.partial(lambda: None)  # partials have no __name__
    with pytest.raises(ValueError, match="cannot determine"):
        register("policy")(nameless)


def test_registered_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown registry kind"):
        registered("bogus")


def test_entrypoint_load_failure_is_skipped_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadEP:
        name = "broken"

        def load(self) -> object:
            raise RuntimeError("import blew up")

    monkeypatch.setattr(
        reg, "entry_points", lambda *, group: [_BadEP()] if group == "robolens.policies" else []
    )
    monkeypatch.setattr(reg, "_loaded_entrypoints", False)
    # A broken plugin must not crash discovery — but must not vanish silently.
    with pytest.warns(RuntimeWarning, match="failed to load robolens plugin 'broken'"):
        names = registered("policy")
    assert "broken" not in names


def test_resolve_forwards_kwargs() -> None:
    policy = resolve("policy", "scripted", chunk_size=9)
    assert policy.chunk_size == 9


# --------------------------------------------------------------------------- #
# compatibility: every issue branch
# --------------------------------------------------------------------------- #
class _CfgPolicy:
    def __init__(
        self,
        action_space: Box,
        *,
        obs: ObservationSpace | None = None,
        control_hz: float | None = None,
    ) -> None:
        self.info = PolicyInfo(
            name="cfg",
            action_space=action_space,
            observation_space=obs or ObservationSpace(),
            control_hz=control_hz,
        )
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.zeros(self.info.action_space.dim))])


def test_compat_semantics_unknown_warns() -> None:
    report = check_compatibility(_CfgPolicy(Box(shape=(2,))), CubePickEmbodiment())
    assert any(i.code == "action_semantics_unknown" for i in report.warnings)


def test_compat_rotation_repr_mismatch() -> None:
    box = Box(
        shape=(2,), semantics=ActionSemantics("eef_delta_pos", rotation_repr="rot6d", frame="world")
    )
    report = check_compatibility(_CfgPolicy(box), CubePickEmbodiment())
    assert any(i.code == "rotation_repr" for i in report.errors)


def test_compat_gripper_and_frame_warnings() -> None:
    gripper_box = Box(
        shape=(2,), semantics=ActionSemantics("eef_delta_pos", gripper="continuous", frame="world")
    )
    assert any(
        i.code == "gripper"
        for i in check_compatibility(_CfgPolicy(gripper_box), CubePickEmbodiment()).warnings
    )
    frame_box = Box(shape=(2,), semantics=ActionSemantics("eef_delta_pos", frame="base"))
    assert any(
        i.code == "frame"
        for i in check_compatibility(_CfgPolicy(frame_box), CubePickEmbodiment()).warnings
    )


def test_compat_control_rate_warning() -> None:
    box = Box(shape=(2,), semantics=_CUBE_SEM)
    report = check_compatibility(_CfgPolicy(box, control_hz=5.0), CubePickEmbodiment())
    assert any(i.code == "control_rate" for i in report.warnings)


def test_compat_scene_setup_realizability() -> None:
    emb = CubePickEmbodiment()
    emb.info = EmbodimentInfo(
        name=emb.info.name,
        action_space=emb.info.action_space,
        observation_space=emb.info.observation_space,
        control_hz=emb.info.control_hz,
        is_simulated=True,
        supported_setups=frozenset({"layout_a"}),
    )
    task = _task(Scene(id="s", instruction="x", setup="layout_b"))
    report = check_compatibility(ScriptedPolicy(), emb, task)
    assert any(i.code == "scene_setup" for i in report.errors)


# --------------------------------------------------------------------------- #
# scorer reducers (median, pass@k edge cases)
# --------------------------------------------------------------------------- #
def test_reduce_median() -> None:
    from robolens.scorer import Score, reduce_scores

    assert (
        reduce_scores("median", [Score(value=1.0), Score(value=3.0), Score(value=9.0)]).value == 3.0
    )


def test_pass_at_k_edge_cases() -> None:
    from robolens.scorer import Score, get_reducer, pass_at_k, reduce_scores

    with pytest.raises(ValueError, match="k must be"):
        pass_at_k(0)
    with pytest.raises(ValueError, match="needs at least"):
        reduce_scores("pass_at_5", [Score(value=True), Score(value=False)])
    with pytest.raises(ValueError, match="invalid pass@k"):
        get_reducer("pass_at_notanint")


# --------------------------------------------------------------------------- #
# RerunSink — exercise the rerun-present logging path with fake backends
# --------------------------------------------------------------------------- #
def _fake_rerun_module(calls: list[str], paths: list[str], *, new_api: bool) -> object:
    import types

    fake = types.ModuleType("rerun")
    fake.init = lambda *a, **k: calls.append("init")  # type: ignore[attr-defined]
    fake.save = lambda p: calls.append("save")  # type: ignore[attr-defined]
    fake.log = lambda path, *a, **k: paths.append(path)  # type: ignore[attr-defined]
    fake.Image = lambda img: ("Image",)  # type: ignore[attr-defined]
    fake.TextLog = lambda t: ("TextLog", t)  # type: ignore[attr-defined]
    if new_api:  # rerun-sdk >= 0.23 surface
        fake.set_time = lambda *a, **k: calls.append("time")  # type: ignore[attr-defined]
        fake.Scalars = lambda v: ("Scalars", v)  # type: ignore[attr-defined]
    else:  # older SDK surface
        fake.set_time_sequence = lambda *a: calls.append("time")  # type: ignore[attr-defined]
        fake.Scalar = lambda v: ("Scalar", v)  # type: ignore[attr-defined]
    return fake


def test_rerun_sink_logs_with_fake_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    calls: list[str] = []
    paths: list[str] = []
    monkeypatch.setitem(sys.modules, "rerun", _fake_rerun_module(calls, paths, new_api=False))

    from robolens.logging.rerun_sink import RerunSink

    sink = RerunSink(str(tmp_path / "run.rrd"))
    assert sink.available is True  # imports the fake backend
    assert sink.available is True  # cached self._rr path

    (log,) = eval(_task(max_steps=40), ScriptedPolicy(), CubePickEmbodiment(), sinks=[sink])
    assert log.status == "success"
    assert "init" in calls and "save" in calls and "time" in calls
    # Entities are namespaced per trial so trials never overwrite each other.
    assert paths and all(p.startswith("trial/s/e0/") for p in paths)

    # Exercise the empty-observation and reward-is-None branches directly.
    sink.log_step(
        0,
        Observation(),  # no images, no state
        Action(data=np.zeros(2)),
        StepResult(observation=Observation(), reward=None),
    )

    # A sink with no recording path skips rr.save (the other on_eval_start branch).
    RerunSink().on_eval_start(None)  # type: ignore[arg-type]


def test_rerun_sink_supports_new_sdk_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    calls: list[str] = []
    paths: list[str] = []
    monkeypatch.setitem(sys.modules, "rerun", _fake_rerun_module(calls, paths, new_api=True))

    from robolens.logging.rerun_sink import RerunSink

    sink = RerunSink()
    (log,) = eval(_task(max_steps=40), ScriptedPolicy(), CubePickEmbodiment(), sinks=[sink])
    assert log.status == "success"
    assert "time" in calls  # rr.set_time (>=0.23) was used
    assert paths and all(p.startswith("trial/s/e0/") for p in paths)
