"""The embodiment evaluation interface: health, prepare_trial, home and park around trials."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inspect_robots import HealthCheck, HealthReport, Readiness
from inspect_robots import eval as ir_eval
from inspect_robots.conformance import check_evaluation_interface, check_policy_interfaces
from inspect_robots.errors import EmbodimentFault, PolicyError, SafetyAbort
from inspect_robots.log import read_eval_log
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box, ControlMode
from inspect_robots.task import Task
from inspect_robots.types import ActionChunk, Observation


def _task(scenes: int = 1, epochs: int = 1) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id=f"s{i}", instruction="reach", init_seed=i) for i in range(scenes)],
        scorer=success_at_end(),
        max_steps=40,
        epochs=epochs,
    )


class _Lifecycle(CubePickEmbodiment):
    """CubePick that logs every evaluation-interface call next to its resets."""

    def __init__(self, **overrides: Any) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.overrides = overrides

    def _answer(self, name: str, default: Any) -> Any:
        self.calls.append(name)
        answer = self.overrides.get(name, default)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def health(self) -> HealthReport:
        result: HealthReport = self._answer("health", HealthReport())
        return result

    def prepare_trial(self, scene: Scene) -> Readiness:
        result: Readiness = self._answer("prepare_trial", Readiness(ready=True))
        return result

    def home(self) -> None:
        self._answer("home", None)

    def park(self) -> None:
        self._answer("park", None)

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.calls.append("reset")
        return super().reset(scene, seed=seed)


# -- the types ---------------------------------------------------------------


def test_health_report_is_ok_only_when_every_check_passed() -> None:
    assert HealthReport().ok
    assert HealthReport().summary() == "ok"
    report = HealthReport(
        checks=(
            HealthCheck("motors", ok=True),
            HealthCheck("camera right", ok=False, detail="no frame for 2.0 s"),
            HealthCheck("thermal", ok=False),
        )
    )
    assert not report.ok
    assert [check.name for check in report.failures] == ["camera right", "thermal"]
    assert report.summary() == "camera right: no frame for 2.0 s; thermal"


# -- eval --------------------------------------------------------------------


def test_hooks_run_in_order_around_every_trial(tmp_path: Path) -> None:
    embodiment = _Lifecycle()
    (log,) = ir_eval(_task(scenes=2), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    trial = ["health", "prepare_trial", "home", "reset", "park"]
    assert embodiment.calls == trial * 2
    assert log.status == "success"
    assert log.eval.embodiment_info["evaluation_interface"] == [
        "health",
        "prepare_trial",
        "home",
        "park",
    ]


def test_embodiments_without_hooks_run_unchanged(tmp_path: Path) -> None:
    (log,) = ir_eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.eval.embodiment_info["evaluation_interface"] == []
    assert log.eval.embodiment_info["morphology"] is None


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"health": HealthReport((HealthCheck("camera right", False, "no frames"),))},
            "embodiment health check failed before scene 's0' epoch 0: camera right: no frames",
        ),
        (
            {"health": RuntimeError("bus down")},
            "embodiment.health() failed before scene 's0' epoch 0: RuntimeError: bus down",
        ),
        (
            {"health": "fine"},
            "embodiment.health() returned str; expected HealthReport",
        ),
        (
            {"prepare_trial": Readiness(ready=False, reason="the drawer is missing")},
            "embodiment not ready for scene 's0' epoch 0: the drawer is missing",
        ),
        (
            {"prepare_trial": Readiness(ready=False)},
            "embodiment not ready for scene 's0' epoch 0: no reason given",
        ),
        (
            {"prepare_trial": EmbodimentFault("scene sensor offline")},
            "embodiment.prepare_trial() failed before scene 's0' epoch 0: "
            "EmbodimentFault: scene sensor offline",
        ),
        (
            {"prepare_trial": True},
            "embodiment.prepare_trial() returned bool; expected Readiness",
        ),
        (
            {"home": EmbodimentFault("joint 3 stalled")},
            "embodiment.home() failed before scene 's0' epoch 0: EmbodimentFault: joint 3 stalled",
        ),
    ],
    ids=[
        "unhealthy",
        "health-raises",
        "health-wrong-type",
        "not-ready",
        "not-ready-no-reason",
        "prepare-raises",
        "prepare-wrong-type",
        "home-raises",
    ],
)
def test_a_failed_pre_trial_hook_stops_the_run_before_the_trial(
    tmp_path: Path, overrides: dict[str, Any], error: str
) -> None:
    embodiment = _Lifecycle(**overrides)
    log = _run_or_halt(_task(scenes=2), embodiment, tmp_path, overrides)
    assert log.status == "error"
    assert log.error == error
    assert "reset" not in embodiment.calls
    # Only a failed home() may have moved the platform, so only it is parked.
    assert embodiment.calls.count("park") == (1 if "home" in overrides else 0)
    assert log.results.total_trials == 0
    (sample,) = log.samples
    assert sample.error == error
    assert sample.epochs == ()


def _run_or_halt(task: Any, embodiment: Any, tmp_path: Path, overrides: dict[str, Any]) -> Any:
    """Run ``task``; a hook that raised a halt signal re-raises it with the written log."""
    halts = [v for v in overrides.values() if isinstance(v, (SafetyAbort, EmbodimentFault))]
    if not halts:
        (log,) = ir_eval(task, ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
        return log
    with pytest.raises(type(halts[0])) as raised:
        ir_eval(task, ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert raised.value is halts[0]
    log = getattr(raised.value, "log")  # noqa: B009 - set on the instance by eval()
    (written,) = tmp_path.glob("*.json")
    assert read_eval_log(str(written)) == log
    return log


def test_a_failed_home_reports_a_failed_park_too(tmp_path: Path) -> None:
    overrides: dict[str, Any] = {
        "home": EmbodimentFault("joint 3 stalled"),
        "park": RuntimeError("brake"),
    }
    embodiment = _Lifecycle(**overrides)
    log = _run_or_halt(_task(), embodiment, tmp_path, overrides)
    assert log.error == (
        "embodiment.home() failed before scene 's0' epoch 0: EmbodimentFault: joint 3 stalled; "
        "embodiment.park() failed: RuntimeError: brake"
    )


def test_a_later_unhealthy_check_keeps_the_trials_already_run(tmp_path: Path) -> None:
    class _FailsSecondTime(_Lifecycle):
        def health(self) -> HealthReport:
            self.calls.append("health")
            if self.calls.count("health") == 2:
                return HealthReport((HealthCheck("thermal", False, "left motor 3 at 71 C"),))
            return HealthReport()

    embodiment = _FailsSecondTime()
    (log,) = ir_eval(_task(epochs=3), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert log.results.total_trials == 1
    assert log.error is not None and "thermal: left motor 3 at 71 C" in log.error
    assert embodiment.calls.count("park") == 1


def test_a_park_failure_after_a_clean_trial_stops_the_run(tmp_path: Path) -> None:
    overrides: dict[str, Any] = {"park": EmbodimentFault("ramp stalled")}
    embodiment = _Lifecycle(**overrides)
    log = _run_or_halt(_task(scenes=2), embodiment, tmp_path, overrides)
    assert log.status == "error"
    assert log.error == "embodiment.park() failed: EmbodimentFault: ramp stalled"
    assert log.results.total_trials == 1
    assert embodiment.calls.count("reset") == 1
    (sample,) = log.samples
    assert sample.reduced  # the trial that ran is still scored


def test_park_failure_joins_an_existing_scene_error(tmp_path: Path) -> None:
    class _BadPolicy(ScriptedPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            raise PolicyError("model unreachable")

    embodiment = _Lifecycle(park=RuntimeError("no rest pose"))
    (log,) = ir_eval(_task(), _BadPolicy(), embodiment, log_dir=str(tmp_path))
    (sample,) = log.samples
    assert sample.error is not None
    assert sample.error.endswith("; embodiment.park() failed: RuntimeError: no rest pose")
    assert log.error == "embodiment.park() failed: RuntimeError: no rest pose"


def test_a_halted_trial_still_parks_and_a_park_failure_only_warns(tmp_path: Path) -> None:
    class _Faulting(_Lifecycle):
        def step(self, action: Any) -> Any:
            raise EmbodimentFault("motor control loop stopped")

    embodiment = _Faulting()
    (log,) = ir_eval(_task(scenes=2), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert log.error == "EmbodimentFault: motor control loop stopped"
    assert embodiment.calls[-1] == "park"

    embodiment = _Faulting(park=EmbodimentFault("driver gone"))
    with pytest.warns(RuntimeWarning, match="embodiment.park\\(\\) failed: .*driver gone"):
        (log,) = ir_eval(_task(), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert log.error == "EmbodimentFault: motor control loop stopped"


def test_a_cancelled_trial_is_not_parked(tmp_path: Path) -> None:
    class _Interrupted(_Lifecycle):
        def step(self, action: Any) -> Any:
            raise KeyboardInterrupt

    embodiment = _Interrupted()
    with pytest.raises(KeyboardInterrupt):
        ir_eval(_task(), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert "park" not in embodiment.calls


def test_a_policy_that_fails_to_start_is_parked_after_homing(tmp_path: Path) -> None:
    class _NoStart(ScriptedPolicy):
        def on_trial_start(self, *args: object) -> None:
            raise RuntimeError("no transcript dir")

    embodiment = _Lifecycle()
    ir_eval(_task(), _NoStart(), embodiment, log_dir=str(tmp_path))
    assert embodiment.calls == ["health", "prepare_trial", "home", "park"]


def test_morphology_is_logged(tmp_path: Path) -> None:
    embodiment = CubePickEmbodiment()
    embodiment.info = replace(embodiment.info, morphology="cube-gripper")
    (log,) = ir_eval(_task(), ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert log.eval.embodiment_info["morphology"] == "cube-gripper"


# -- conformance -------------------------------------------------------------


def _info(name: str, morphology: str | None, control_mode: ControlMode = "eef_delta_pos") -> Any:
    info = CubePickEmbodiment().info
    space = Box(
        shape=info.action_space.shape,
        low=info.action_space.low,
        high=info.action_space.high,
        semantics=ActionSemantics(control_mode=control_mode, frame="world"),
    )
    return replace(info, name=name, morphology=morphology, action_space=space)


def test_embodiments_on_one_morphology_need_distinct_policy_interfaces() -> None:
    report = check_policy_interfaces(
        [
            _info("arm-delta", "arm"),
            _info("arm-delta-again", "arm"),
            _info("arm-joint", "arm", control_mode="joint_delta"),
            _info("other-delta", "other"),
            _info("loose", None),
            _info("loose-too", None),
        ]
    )
    assert not report.ok
    (issue,) = report.issues
    assert issue.code == "policy_interface"
    assert "'arm-delta' and 'arm-delta-again' share morphology 'arm'" in issue.message
    assert check_policy_interfaces([_info("a", "arm"), _info("a", "arm")]).ok


def test_policy_interface_tells_missing_semantics_and_state_apart() -> None:
    bare = CubePickEmbodiment().info
    bare = replace(
        bare,
        morphology="arm",
        action_space=Box(shape=bare.action_space.shape),
        observation_space=replace(bare.observation_space, state=None),
    )
    assert check_policy_interfaces([bare, _info("arm-delta", "arm")]).ok


def test_rotation_and_gripper_encodings_are_distinct_interfaces() -> None:
    def pose_info(name: str, rotation: Any, gripper: Any = "none") -> Any:
        info = CubePickEmbodiment().info
        space = Box(
            shape=(7,),
            semantics=ActionSemantics(
                control_mode="eef_delta_pose", rotation_repr=rotation, gripper=gripper
            ),
        )
        return replace(info, name=name, morphology="arm", action_space=space)

    assert check_policy_interfaces(
        [pose_info("axis-angle", "axis_angle"), pose_info("euler", "euler_xyz")]
    ).ok
    assert check_policy_interfaces(
        [
            pose_info("binary", "axis_angle", gripper="binary"),
            pose_info("continuous", "axis_angle", gripper="continuous"),
        ]
    ).ok
    assert not check_policy_interfaces(
        [pose_info("one", "axis_angle"), pose_info("two", "axis_angle")]
    ).ok


def test_evaluation_interface_hooks_are_checked_without_being_called() -> None:
    embodiment = _Lifecycle()
    assert check_evaluation_interface(embodiment).ok
    assert embodiment.calls == []
    assert check_evaluation_interface(CubePickEmbodiment()).issues == ()


def test_evaluation_interface_rejects_wrong_signatures_and_non_callables() -> None:
    class _Broken(CubePickEmbodiment):
        health = "yes"

        def prepare_trial(self) -> Readiness:
            return Readiness(ready=True)

        def park(self, speed: float) -> None:
            return None

    report = check_evaluation_interface(_Broken())
    messages = [issue.message for issue in report.issues]
    assert messages[0] == "health is not callable (got str)"
    assert messages[1] == "prepare_trial() cannot be called with the scene"
    assert messages[2] == "park(speed) cannot be called with no arguments"
    assert report.issues[3].severity == "warning"
    assert "park() is offered without health()" in messages[3]


def test_uninspectable_hooks_pass() -> None:
    class _Builtin(CubePickEmbodiment):
        park = staticmethod(max)  # a builtin whose signature cannot be read

        def health(self) -> HealthReport:
            return HealthReport()

    assert check_evaluation_interface(_Builtin()).ok


@pytest.mark.parametrize("hook", ["health", "prepare_trial", "home", "park"])
@pytest.mark.parametrize("halt", [EmbodimentFault("bus fault"), SafetyAbort("e-stop")])
def test_a_hook_halt_stops_the_task_set(tmp_path: Path, hook: str, halt: Exception) -> None:
    from inspect_robots import eval_set

    embodiment = _Lifecycle(**{hook: halt})
    tasks = [replace(_task(), name="first"), replace(_task(), name="second")]
    with pytest.raises(type(halt)):
        eval_set(tasks, ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    # The first task's log is written; the second never touched the platform again.
    (written,) = tmp_path.glob("*.json")
    assert read_eval_log(str(written)).eval.task == "first"
    assert embodiment.calls.count("reset") == (1 if hook == "park" else 0)


def test_a_hook_failure_that_is_not_a_halt_lets_the_set_continue(tmp_path: Path) -> None:
    from inspect_robots import eval_set

    class _OnceBroken(_Lifecycle):
        def health(self) -> HealthReport:
            self.calls.append("health")
            if self.calls.count("health") == 1:
                raise RuntimeError("flaky probe")
            return HealthReport()

    tasks = [replace(_task(), name="first"), replace(_task(), name="second")]
    success, logs = eval_set(tasks, ScriptedPolicy(), _OnceBroken(), log_dir=str(tmp_path))
    assert not success
    assert [log.status for log in logs] == ["error", "success"]


@pytest.mark.parametrize("raised", [RuntimeError("grader crashed"), SafetyAbort("judge saw smoke")])
def test_a_grading_exception_still_parks_the_robot(tmp_path: Path, raised: Exception) -> None:
    def grade(record: Any, scene: Scene) -> None:
        raise raised

    embodiment = _Lifecycle()
    with pytest.raises(type(raised)):
        ir_eval(_task(), ScriptedPolicy(), embodiment, log_dir=str(tmp_path), before_scoring=grade)
    assert embodiment.calls[-1] == "park"


def test_a_scorer_halt_parks_and_a_failed_park_only_warns(tmp_path: Path) -> None:
    from inspect_robots.scorer import Score

    class _HaltingScorer:
        name = "halting"

        def __call__(self, record: Any, target: Any) -> Score:
            raise SafetyAbort("scorer saw a collision")

    task = replace(_task(), scorer=_HaltingScorer())
    embodiment = _Lifecycle(park=RuntimeError("brake"))
    with (
        pytest.raises(SafetyAbort),
        pytest.warns(RuntimeWarning, match="park\\(\\) failed: RuntimeError: brake"),
    ):
        ir_eval(task, ScriptedPolicy(), embodiment, log_dir=str(tmp_path))
    assert embodiment.calls[-1] == "park"


def test_observe_parked_halts_park_before_escaping(tmp_path: Path) -> None:
    class _BadParkedView(_Lifecycle):
        def observe_parked(self) -> Observation:
            raise EmbodimentFault("camera bus fault")

    def grade(record: Any, scene: Scene) -> None:
        record.operator_judgement = "yes"

    embodiment = _BadParkedView()
    unfinished = replace(_task(), max_steps=1)  # not a definitive success, so it is graded
    with pytest.raises(EmbodimentFault):
        ir_eval(
            unfinished, ScriptedPolicy(), embodiment, log_dir=str(tmp_path), before_scoring=grade
        )
    assert embodiment.calls[-1] == "park"


def test_a_ctrl_c_while_grading_is_not_parked(tmp_path: Path) -> None:
    def grade(record: Any, scene: Scene) -> None:
        raise KeyboardInterrupt

    embodiment = _Lifecycle()
    with pytest.raises(KeyboardInterrupt):
        ir_eval(_task(), ScriptedPolicy(), embodiment, log_dir=str(tmp_path), before_scoring=grade)
    assert "park" not in embodiment.calls


def test_a_hook_halt_survives_a_failed_final_log_write(tmp_path: Path) -> None:
    from inspect_robots import eval_set
    from inspect_robots.logging import NullSink

    class _FullDisk(NullSink):
        critical = True

        def on_eval_end(self, log: Any) -> None:
            raise OSError("disk full")

    embodiment = _Lifecycle(health=SafetyAbort("e-stop"))
    tasks = [replace(_task(), name="first"), replace(_task(), name="second")]
    with pytest.raises(SafetyAbort) as raised:
        eval_set(tasks, ScriptedPolicy(), embodiment, log_dir=str(tmp_path), sinks=[_FullDisk()])
    assert isinstance(raised.value.__cause__, OSError)
    assert embodiment.calls.count("health") == 1
