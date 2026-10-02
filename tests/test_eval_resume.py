"""Resumable evaluation set behavior with the deterministic CubePick world."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from inspect_robots import eval, eval_set
from inspect_robots.errors import EmbodimentFault, PolicyError, SafetyAbort
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Epochs, Task
from inspect_robots.types import ActionChunk, Observation

if TYPE_CHECKING:
    from inspect_robots.approver import Approver
    from inspect_robots.controller import Controller


class _OneTransientFailure(ScriptedPolicy):
    """Fail one inference transiently, then follow the scripted trajectory."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def act(self, observation: Observation) -> ActionChunk:
        if not self.failed:
            self.failed = True
            raise PolicyError("server restarted", retryable=True)
        return super().act(observation)


def test_real_attempt_records_scene_local_retryability_and_error_count(tmp_path: Path) -> None:
    """A failed epoch must retain its retry marker even if the next epoch scores."""
    task = Task(
        name="two-epochs",
        scenes=[Scene(id="s0", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
        epochs=Epochs(count=2),
    )

    (log,) = eval(task, _OneTransientFailure(), CubePickEmbodiment(), log_dir=str(tmp_path))

    assert log.samples[0].status == "error"
    assert log.samples[0].errored_trials == 1
    assert log.samples[0].retryable_error is True
    assert log.results.errored_trials == 1
    assert len(log.samples[0].epochs) == 2


def test_real_attempt_marks_frame_source_on_scene(tmp_path: Path) -> None:
    """A scene retains the attempt frame root needed after aggregate merging."""
    task = Task(
        name="one-scene",
        scenes=[Scene(id="s0", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )

    (log,) = eval(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        store_frames=True,
    )

    assert log.samples[0].frames_dir == log.stats.frames_dir
    assert log.samples[0].frames_dir is not None


def _task() -> Task:
    """Use a stable two-scene task for checkpoint identity tests."""
    return Task(
        name="resume-demo",
        scenes=[Scene(id="s0", instruction="reach"), Scene(id="s1", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )


def test_checkpoint_reopens_with_attempt_and_rejects_changed_identity(tmp_path: Path) -> None:
    """A resumed call may reuse only a matching scene declaration and attempt log."""
    from inspect_robots._eval_set_checkpoint import _identity, _open_checkpoint
    from inspect_robots.errors import ConfigError

    task = _task()
    policy = ScriptedPolicy()
    embodiment = CubePickEmbodiment()
    log_dir = tmp_path / "logs"
    checkpoint = tmp_path / "run.checkpoint.json"
    identity = _identity([task], policy, embodiment, seed=17, log_dir=str(log_dir), options={})
    (attempt,) = eval(task, policy, embodiment, log_dir=str(log_dir), seed=17)
    attempt_path = next(log_dir.glob("*.json"))
    assert attempt.samples[0].status == "success"

    with _open_checkpoint(checkpoint, identity) as manifest:
        manifest.add_attempt(0, ["s0", "s1"], attempt_path)
    with _open_checkpoint(checkpoint, identity) as manifest:
        assert manifest.attempts[0]["scene_ids"] == ["s0", "s1"]
        assert manifest.attempt_log_path(manifest.attempts[0]) == attempt_path

    changed = Task(
        name="resume-demo",
        scenes=[Scene(id="s0", instruction="different"), Scene(id="s1", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    changed_identity = _identity(
        [changed], policy, embodiment, seed=17, log_dir=str(log_dir), options={}
    )
    with (
        pytest.raises(ConfigError, match="checkpoint identity"),
        _open_checkpoint(checkpoint, changed_identity),
    ):
        pass


def test_checkpoint_rejects_non_json_scene_before_creation(tmp_path: Path) -> None:
    """Opaque scene metadata cannot produce a reliable resume identity."""
    from inspect_robots._eval_set_checkpoint import _identity
    from inspect_robots.errors import ConfigError

    task = Task(
        name="opaque",
        scenes=[Scene(id="s", instruction="reach", metadata={"opaque": object()})],
        scorer=success_at_end(),
        max_steps=30,
    )
    with pytest.raises(ConfigError, match="JSON"):
        _identity(
            [task],
            ScriptedPolicy(),
            CubePickEmbodiment(),
            seed=17,
            log_dir=str(tmp_path),
            options={},
        )


def test_checkpoint_scorer_hook_and_opaque_scorer_identity(tmp_path: Path) -> None:
    """A scorer can expose stable settings; opaque scorers fail before rollout."""
    from dataclasses import replace

    from inspect_robots._eval_set_checkpoint import _identity
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Target
    from inspect_robots.scorer import Score

    class _HookScorer:
        __slots__ = ()
        name = "hook"

        def checkpoint_identity(self) -> dict[str, float]:
            return {"threshold": 0.5}

        def __call__(self, record: TrialRecord, target: Target | None) -> Score:
            del record, target
            return Score(value=1.0)

    class _OpaqueScorer:
        __slots__ = ()
        name = "opaque"

        def __call__(self, record: TrialRecord, target: Target | None) -> Score:
            del record, target
            return Score(value=1.0)

    identity = _identity(
        [replace(_task(), scorer=_HookScorer())],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        seed=17,
        log_dir=str(tmp_path),
        options={},
    )
    assert identity["tasks"][0]["scorers"][0]["name"] == "hook"  # type: ignore[index]
    with pytest.raises(ConfigError, match="checkpoint_identity"):
        _identity(
            [replace(_task(), scorer=_OpaqueScorer())],
            ScriptedPolicy(),
            CubePickEmbodiment(),
            seed=17,
            log_dir=str(tmp_path),
            options={},
        )


def test_checkpoint_rejects_closure_held_scorer_settings(tmp_path: Path) -> None:
    """Hidden scorer closure values cannot silently reuse earlier scores."""
    from dataclasses import replace

    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Target
    from inspect_robots.scorer import Score

    def scorer_with_value(value: float):  # type: ignore[no-untyped-def]
        class _ClosureScorer:
            name = "closure"

            def __call__(self, record: TrialRecord, target: Target | None) -> Score:
                del record, target
                return Score(value=value)

        return _ClosureScorer()

    checkpoint = tmp_path / "run.json"
    with pytest.raises(ConfigError, match="checkpoint_identity"):
        eval_set(
            replace(_task(), scorer=scorer_with_value(0.0)),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert not checkpoint.exists()


def _execution_components(case: str, setting: int) -> tuple[Controller | None, Approver | None]:
    """Vary effective built-in settings while retaining each component type."""
    import numpy as np

    from inspect_robots.approver import (
        AutoApprover,
        ChainApprover,
        ClampApprover,
        DeltaLimitApprover,
    )
    from inspect_robots.controller import (
        DefaultController,
        EnsemblingController,
        SmoothingController,
    )
    from inspect_robots.spaces import ActionSemantics, Box

    space_scale = (
        setting if case in {"clamp", "delta-displacement", "chain-clamp", "ensembling-space"} else 1
    )
    space = Box(
        shape=(2,),
        low=np.full(2, -0.1 / space_scale),
        high=np.full(2, 0.1 / space_scale),
        semantics=ActionSemantics(control_mode="eef_delta_pos"),
    )
    absolute_space = Box(shape=(2,), semantics=ActionSemantics(control_mode="joint_pos"))
    controllers: dict[str, Controller] = {
        "default": DefaultController(replan_interval=setting),
        "smoothing": SmoothingController(DefaultController(), alpha=0.5 / setting),
        "smoothing-inner": SmoothingController(DefaultController(replan_interval=setting)),
        "ensembling": EnsemblingController(space, m=0.1 * setting),
        "ensembling-space": EnsemblingController(space),
    }
    clamp = ClampApprover(space)
    delta = DeltaLimitApprover(space)
    approvers: dict[str, Approver] = {
        "clamp": clamp,
        "delta-absolute": DeltaLimitApprover(absolute_space, max_delta=0.1 / setting),
        "delta-displacement": delta,
        "chain-clamp": ChainApprover(AutoApprover(), clamp),
        "chain-order": (
            ChainApprover(clamp, delta) if setting == 1 else ChainApprover(delta, clamp)
        ),
    }
    return controllers.get(case), approvers.get(case)


@pytest.mark.parametrize(
    "case",
    [
        "default",
        "smoothing",
        "smoothing-inner",
        "ensembling",
        "ensembling-space",
        "clamp",
        "delta-absolute",
        "delta-displacement",
        "chain-clamp",
        "chain-order",
    ],
)
def test_checkpoint_matches_effective_execution_component_settings(
    tmp_path: Path, case: str
) -> None:
    """Equal settings reuse attempts, while changed action constraints reject reuse."""
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    controller, approver = _execution_components(case, 1)
    first_success, first_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        controller=controller,
        approver=approver,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    controller, approver = _execution_components(case, 1)
    same_success, same_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        controller=controller,
        approver=approver,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert first_success and same_success
    assert first_logs[0].source_logs == same_logs[0].source_logs
    controller, approver = _execution_components(case, 2)
    policy = ScriptedPolicy()
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            policy,
            CubePickEmbodiment(),
            controller=controller,
            approver=approver,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert policy.num_inferences == 0


def test_checkpoint_rejects_tightened_clamp_before_reusing_success(tmp_path: Path) -> None:
    """A changed API action gate cannot reuse scores from unrestricted motion."""
    import numpy as np

    from inspect_robots.approver import ClampApprover
    from inspect_robots.errors import ConfigError
    from inspect_robots.spaces import Box

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    success, first_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        approver=ClampApprover(Box(shape=(2,))),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success and first_logs[0].results.metrics == {"success_at_end": 1.0}
    tightened = ClampApprover(Box(shape=(2,), low=np.zeros(2), high=np.zeros(2)))
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            approver=tightened,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    _, fresh_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        approver=tightened,
        log_dir=str(log_dir),
    )
    assert fresh_logs[0].results.metrics == {"success_at_end": 0.0}


@pytest.mark.parametrize("kind", ["controller", "approver"])
def test_checkpoint_requires_identity_for_custom_execution_components(
    tmp_path: Path, kind: str
) -> None:
    """Subclass settings stay opaque even when a built-in supplies their behavior."""
    from inspect_robots.approver import AutoApprover
    from inspect_robots.controller import DefaultController
    from inspect_robots.errors import ConfigError

    class _CustomController(DefaultController):
        pass

    class _CustomApprover(AutoApprover):
        pass

    controller = _CustomController() if kind == "controller" else None
    approver = _CustomApprover() if kind == "approver" else None
    checkpoint = tmp_path / "run.json"
    with pytest.raises(ConfigError, match=f"{kind}.*checkpoint_identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            controller=controller,
            approver=approver,
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert not checkpoint.exists()
    assert not (tmp_path / "logs").exists()
    success, _ = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        controller=controller,
        approver=approver,
        retry_attempts=1,
        log_dir=str(tmp_path / "logs"),
    )
    assert success


@pytest.mark.parametrize("kind", ["controller", "approver"])
def test_checkpoint_compares_custom_execution_identity(tmp_path: Path, kind: str) -> None:
    """Custom component declarations reject revisions and non-JSON settings."""
    from inspect_robots.approver import AutoApprover
    from inspect_robots.controller import DefaultController
    from inspect_robots.errors import ConfigError

    revision: list[object] = ["original"]

    class _CustomController(DefaultController):
        def checkpoint_identity(self) -> dict[str, object]:
            return {"revision": revision[0]}

    class _CustomApprover(AutoApprover):
        def checkpoint_identity(self) -> dict[str, object]:
            return {"revision": revision[0]}

    controller = _CustomController() if kind == "controller" else None
    approver = _CustomApprover() if kind == "approver" else None
    checkpoint = tmp_path / "run.json"
    for expected_error in [None, "checkpoint identity", "JSON"]:
        if expected_error is None:
            success, _ = eval_set(
                _task(),
                ScriptedPolicy(),
                CubePickEmbodiment(),
                controller=controller,
                approver=approver,
                checkpoint_path=str(checkpoint),
                log_dir=str(tmp_path / "logs"),
            )
            assert success
            revision[0] = "changed"
        else:
            with pytest.raises(ConfigError, match=expected_error):
                eval_set(
                    _task(),
                    ScriptedPolicy(),
                    CubePickEmbodiment(),
                    controller=controller,
                    approver=approver,
                    checkpoint_path=str(checkpoint),
                    log_dir=str(tmp_path / "logs"),
                )
            revision[0] = object()


def test_checkpoint_rejects_grading_callback_without_declared_identity(tmp_path: Path) -> None:
    """Opaque grading settings cannot enter a checkpoint that may reuse scores."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord

    def grade(record: TrialRecord, scene: Scene) -> None:
        del scene
        record.operator_judgement = "yes"

    checkpoint = tmp_path / "run.json"
    with pytest.raises(ConfigError, match=r"before_scoring.*checkpoint_identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            before_scoring=grade,
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert not checkpoint.exists()
    assert not (tmp_path / "logs").exists()
    success, _ = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=grade,
        retry_attempts=1,
        log_dir=str(tmp_path / "logs"),
    )
    assert success


def test_checkpoint_rejects_distinct_grading_functions(tmp_path: Path) -> None:
    """Changing the grading function cannot reuse an earlier operator score."""
    from dataclasses import replace

    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scorer import operator_scorer

    def grade_success(record: TrialRecord, scene: Scene) -> None:
        del scene
        record.operator_judgement = "yes"

    def grade_failure(record: TrialRecord, scene: Scene) -> None:
        del scene
        record.operator_judgement = "no"

    for grade in [grade_success, grade_failure]:
        grade.checkpoint_identity = lambda: {"revision": 1}  # type: ignore[attr-defined]
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    task = replace(_task(), scorer=operator_scorer())
    success, logs = eval_set(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=grade_success,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success and logs[0].results.metrics == {"operator": 1.0}
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            task,
            ScriptedPolicy(),
            CubePickEmbodiment(),
            before_scoring=grade_failure,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    _, fresh_logs = eval_set(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=grade_failure,
        log_dir=str(log_dir),
    )
    assert fresh_logs[0].results.metrics == {"operator": 0.0}


@pytest.mark.parametrize("bound_method", [False, True])
def test_checkpoint_rejects_changed_declared_callback_settings(
    tmp_path: Path, bound_method: bool
) -> None:
    """A callable's hidden grading configuration must participate in identity."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord

    class _Grade:
        def __init__(self, verdict: str) -> None:
            self.verdict = verdict

        def checkpoint_identity(self) -> dict[str, str]:
            return {"verdict": self.verdict}

        def __call__(self, record: TrialRecord, scene: Scene) -> None:
            self.grade(record, scene)

        def grade(self, record: TrialRecord, scene: Scene) -> None:
            del scene
            record.operator_judgement = self.verdict

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    grade = _Grade("yes")
    callback = grade.grade if bound_method else grade
    first_success, first_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=callback,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    same_success, same_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=callback,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert first_success and same_success
    assert first_logs[0].source_logs == same_logs[0].source_logs
    grade.verdict = "no"
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            before_scoring=callback,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )


def test_checkpoint_uses_identity_attached_to_bound_callable(tmp_path: Path) -> None:
    """A method's declared revision governs reuse when its owner has no hook."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord

    class _Grade:
        def grade(self, record: TrialRecord, scene: Scene) -> None:
            del scene
            record.operator_judgement = "yes"

    revision = [1]
    _Grade.grade.checkpoint_identity = lambda: {"revision": revision[0]}  # type: ignore[attr-defined]
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    callback = _Grade().grade
    first_success, first_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=callback,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    same_success, same_logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        before_scoring=callback,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert first_success and same_success
    assert first_logs[0].source_logs == same_logs[0].source_logs
    revision[0] = 2
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            before_scoring=callback,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )


def test_checkpoint_rejects_non_json_callback_identity(tmp_path: Path) -> None:
    """Invalid declared callback settings fail before a checkpoint is created."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord

    def grade(record: TrialRecord, scene: Scene) -> None:
        del record, scene

    grade.checkpoint_identity = lambda: object()  # type: ignore[attr-defined]
    checkpoint = tmp_path / "run.json"
    with pytest.raises(ConfigError, match=r"before_scoring checkpoint_identity.*JSON"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            before_scoring=grade,
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert not checkpoint.exists()


def test_checkpoint_lock_blocks_second_writer(tmp_path: Path) -> None:
    """A live writer must prevent another process from mutating the manifest."""
    from inspect_robots._eval_set_checkpoint import _identity, _open_checkpoint
    from inspect_robots.errors import ConfigError

    identity = _identity(
        [_task()],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        seed=17,
        log_dir=str(tmp_path / "logs"),
        options={},
    )
    checkpoint = tmp_path / "run.json"
    with (
        _open_checkpoint(checkpoint, identity),
        pytest.raises(ConfigError, match="lock"),
        _open_checkpoint(checkpoint, identity),
    ):
        pass


def _saved_attempt(tmp_path: Path, *, seed: int = 17) -> Path:
    """Write a real two-scene EvalLog for manifest validation."""
    log_dir = tmp_path / "logs"
    eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(log_dir), seed=seed)
    return next(log_dir.glob("*.json"))


def _checkpoint_identity(tmp_path: Path) -> dict[str, object]:
    """Build the matching identity for a real saved attempt."""
    from inspect_robots._eval_set_checkpoint import _identity

    return _identity(
        [_task()],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        seed=17,
        log_dir=str(tmp_path / "logs"),
        options={},
    )


def test_checkpoint_seed_reads_existing_and_rejects_invalid(tmp_path: Path) -> None:
    """The recorded seed survives restart, while corrupt seed data fails closed."""
    from inspect_robots._eval_set_checkpoint import _checkpoint_seed, _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    assert _checkpoint_seed(checkpoint) is None
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)):
        pass
    assert _checkpoint_seed(checkpoint) == 17
    payload = json.loads(checkpoint.read_text())
    payload["identity"]["seed"] = True
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="seed"):
        _checkpoint_seed(checkpoint)
    checkpoint.write_text("not JSON")
    with pytest.raises(ConfigError, match="seed"):
        _checkpoint_seed(checkpoint)


def test_checkpoint_rejects_missing_or_untrusted_attempt_log(tmp_path: Path) -> None:
    """A manifest cannot point outside the run log directory or to a missing log."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        with pytest.raises(ConfigError, match="attempt"):
            manifest.add_attempt(0, ["s0", "s1"], tmp_path / "missing.json")
        with pytest.raises(ConfigError, match="attempt"):
            manifest.add_attempt(0, ["s0", "s1"], tmp_path / "outside.json")
        manifest.add_attempt(0, ["s0", "s1"], log_path)
    log_path.unlink()
    with (
        pytest.raises(ConfigError, match="attempt"),
        _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)),
    ):
        pass


def test_checkpoint_rejects_tampered_attempt_metadata(tmp_path: Path) -> None:
    """A forged scene, seed, task, or path cannot be reused as prior work."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        manifest.add_attempt(0, ["s0", "s1"], log_path)
        entry = manifest.attempts[0]
        bad_entries = [
            {**entry, "task_index": True},
            {**entry, "task_index": -1},
            {**entry, "task_index": 1},
            {**entry, "scene_ids": "s0"},
            {**entry, "scene_ids": [0]},
            {**entry, "scene_ids": ["other"]},
            {**entry, "log": 0},
            {**entry, "log": str(log_path)},
            {"scene_ids": ["s0", "s1"], "log": entry["log"]},
        ]
        for bad in bad_entries:
            with pytest.raises(ConfigError, match="attempt"):
                manifest.attempt_log_path(bad)

        data = json.loads(log_path.read_text())
        for field, value in [("task", "other"), ("seed", 999)]:
            changed = json.loads(json.dumps(data))
            changed["eval"][field] = value
            log_path.write_text(json.dumps(changed))
            with pytest.raises(ConfigError, match="attempt"):
                manifest.attempt_log_path(entry)
        log_path.write_text(json.dumps(data))


def test_checkpoint_atomic_failure_retains_previous_manifest(tmp_path: Path) -> None:
    """Failed publication rolls back memory and leaves the old JSON readable."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        before = checkpoint.read_text()
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.add_attempt(0, ["s0", "s1"], log_path)
        assert manifest.attempts == []
        assert checkpoint.read_text() == before
        assert not list(tmp_path.glob("*.tmp"))
        manifest.add_attempt(0, ["s0", "s1"], log_path)
        manifest.set_aggregate(0, log_path)
        previous_aggregate = manifest.aggregates["0"]
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.set_aggregate(0, tmp_path / "other.json")
        assert manifest.aggregates["0"] == previous_aggregate
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.set_aggregate(1, tmp_path / "other.json")
        assert "1" not in manifest.aggregates
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        assert len(manifest.attempts) == 1
        assert manifest.aggregates["0"] == previous_aggregate


def test_checkpoint_start_failure_does_not_move_robot_or_mark_in_flight(tmp_path: Path) -> None:
    """A failed pre-attempt manifest publication leaves the old checkpoint usable."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint

    checkpoint = tmp_path / "run.json"
    log_path = _saved_attempt(tmp_path)
    with _open_checkpoint(checkpoint, _checkpoint_identity(tmp_path)) as manifest:
        before = checkpoint.read_text()
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.start_attempt()
        assert manifest.in_flight is False
        assert checkpoint.read_text() == before
        manifest.start_attempt()
        assert json.loads(checkpoint.read_text())["in_flight"] is True
        with (
            patch("inspect_robots._eval_set_checkpoint.os.replace", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            manifest.add_attempt(0, ["s0", "s1"], log_path)
        assert json.loads(checkpoint.read_text())["in_flight"] is True
        manifest.add_attempt(0, ["s0", "s1"], log_path)
        assert manifest.in_flight is False


def test_checkpoint_invalid_schema_and_entries_fail_closed(tmp_path: Path) -> None:
    """Corrupt or unknown manifests fail before they can schedule work."""
    from inspect_robots._eval_set_checkpoint import _open_checkpoint
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    identity = _checkpoint_identity(tmp_path)
    with _open_checkpoint(checkpoint, identity):
        pass
    original = json.loads(checkpoint.read_text())
    variants = [
        {**original, "version": 999},
        {**original, "attempts": "bad"},
        {**original, "aggregates": []},
        {**original, "in_flight": "bad"},
        {**original, "attempts": [{}]},
        {"version": 1, "identity": identity},
    ]
    for variant in variants:
        checkpoint.write_text(json.dumps(variant))
        with pytest.raises(ConfigError, match="checkpoint"), _open_checkpoint(checkpoint, identity):
            pass
        assert not checkpoint.with_name("run.json.lock").exists()
    checkpoint.write_text(json.dumps(original))


def test_merge_keeps_first_complete_zero_score_and_task_order(tmp_path: Path) -> None:
    """A completed scene is retained even if its score is zero and later logs differ."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task, _pending_scenes

    task = _task()
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    zero = replace(
        original.samples[0],
        reduced={"success_at_end": 0.0},
        epochs=({"success_at_end": 0.0},),
    )
    first = replace(
        original,
        samples=(original.samples[1], zero),
        stats=replace(original.stats, duration_s=1.25, total_steps=10),
    )
    second = replace(original, stats=replace(original.stats, duration_s=2.5, total_steps=20))

    merged = _merge_task(task, [(first, "attempt-1.json"), (second, "attempt-2.json")])

    assert [sample.scene_id for sample in merged.samples] == ["s0", "s1"]
    assert merged.samples[0].reduced == {"success_at_end": 0.0}
    assert merged.results.metrics == {"success_at_end": 0.5}
    assert merged.results.total_trials == 2
    assert merged.stats.duration_s == 3.75
    assert merged.stats.total_steps == 30
    assert merged.stats.mean_inference_latency_s is None
    assert merged.stats.frames_dir is None
    assert merged.source_logs == ("attempt-1.json", "attempt-2.json")
    assert merged.status == "success"
    assert _pending_scenes(task, merged) == []


def test_merge_replaces_partial_scene_and_keeps_unreached_pending(tmp_path: Path) -> None:
    """Incomplete scenes use the latest record and stay pending until fully rerun."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task, _pending_scenes

    task = _task()
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    failed = replace(
        original.samples[0],
        status="error",
        epochs=({},),
        reduced={},
        error="PolicyError: offline",
        errored_trials=1,
        retryable_error=True,
    )
    first = replace(original, status="error", samples=(failed,), error="offline")
    incomplete = _merge_task(task, [(first, "first.json")])
    assert incomplete.status == "error"
    assert incomplete.results.errored_trials == 1
    assert [scene.id for scene in _pending_scenes(task, incomplete)] == ["s0", "s1"]

    second = replace(original, samples=(original.samples[0],), error=None)
    merged = _merge_task(task, [(first, "first.json"), (second, "second.json")])
    assert merged.samples[0].status == "success"
    assert merged.results.errored_trials == 0
    assert [scene.id for scene in _pending_scenes(task, merged)] == ["s1"]


def test_merge_uses_attempt_frame_root_for_selected_scene(tmp_path: Path) -> None:
    """A selected scene remembers the source even when only the log has the root."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task

    task = _task()
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    first = replace(
        original,
        samples=(original.samples[0],),
        stats=replace(original.stats, frames_dir="first-frames"),
    )
    second = replace(
        original,
        samples=(original.samples[1],),
        stats=replace(original.stats, frames_dir="second-frames"),
    )
    merged = _merge_task(task, [(first, "first.json"), (second, "second.json")])
    assert [sample.frames_dir for sample in merged.samples] == [
        "first-frames",
        "second-frames",
    ]


def test_merge_requires_real_attempts_and_rejects_foreign_scenes(tmp_path: Path) -> None:
    """A merge cannot fabricate provenance or accept a scene outside the task."""
    from dataclasses import replace

    from inspect_robots._eval_set_merge import _merge_task
    from inspect_robots.errors import ConfigError

    task = _task()
    with pytest.raises(ConfigError, match="without attempt"):
        _merge_task(task, [])
    (original,) = eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), seed=17)
    foreign = replace(original.samples[0], scene_id="foreign")
    with pytest.raises(ConfigError, match="unknown scene"):
        _merge_task(task, [(replace(original, samples=(foreign,)), "foreign.json")])


class _RecordingTransientPolicy(_OneTransientFailure):
    """Record scene resets to prove completed work is skipped."""

    def __init__(self) -> None:
        super().__init__()
        self.resets: list[str] = []

    def reset(self, scene: Scene) -> None:
        self.resets.append(scene.id)
        super().reset(scene)


def test_eval_set_retries_marked_scene_without_replaying_completed_scene(tmp_path: Path) -> None:
    """One retry repairs only the transient scene, with a durable attempt trail."""
    from inspect_robots.log import read_eval_log

    policy = _RecordingTransientPolicy()
    checkpoint = tmp_path / "run.checkpoint.json"
    log_dir = tmp_path / "logs"
    success, logs = eval_set(
        _task(),
        policy,
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        retry_attempts=1,
        log_dir=str(log_dir),
    )

    assert success is True
    assert policy.resets.count("s0") == 2
    assert policy.resets.count("s1") == 1
    assert len(logs[0].source_logs) == 2
    assert len(json.loads(checkpoint.read_text())["attempts"]) == 2
    assert logs[0].results.errored_trials == 0
    assert logs[0].stats.total_steps > 0
    assert all(read_eval_log(path).eval.seed == 0 for path in logs[0].source_logs)

    again = _RecordingTransientPolicy()
    success_again, resumed = eval_set(
        _task(),
        again,
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        retry_attempts=1,
        log_dir=str(log_dir),
    )
    assert success_again is True
    assert again.resets == []
    assert resumed[0].source_logs == logs[0].source_logs
    assert len(json.loads(checkpoint.read_text())["attempts"]) == 2


def test_eval_set_manual_resume_retries_only_unfinished_scene(tmp_path: Path) -> None:
    """A later invocation retries an incomplete scene even with no automatic budget."""
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    first_policy = _RecordingTransientPolicy()
    success, logs = eval_set(
        _task(),
        first_policy,
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success is False
    assert len(logs[0].source_logs) == 1
    assert [scene.id for scene in _task().scenes] == ["s0", "s1"]

    second_policy = _RecordingTransientPolicy()
    second_policy.failed = True
    resumed_success, resumed = eval_set(
        _task(),
        second_policy,
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert resumed_success is True
    assert second_policy.resets == ["s0"]
    assert len(resumed[0].source_logs) == 2


def test_eval_set_does_not_retry_unmarked_policy_failure(tmp_path: Path) -> None:
    """An ordinary policy error gets one attempt despite a retry budget."""

    class _NonRetryPolicy(_RecordingTransientPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            if not self.failed:
                self.failed = True
                raise PolicyError("malformed response")
            return ScriptedPolicy.act(self, observation)

    policy = _NonRetryPolicy()
    success, logs = eval_set(
        _task(), policy, CubePickEmbodiment(), retry_attempts=3, log_dir=str(tmp_path)
    )
    assert success is False
    assert policy.resets == ["s0", "s1"]
    assert len(logs[0].source_logs) == 1


def test_eval_set_seed_none_stays_fixed_across_retry_and_resume(tmp_path: Path) -> None:
    """Unseeded calls draw once, then reuse that seed from the checkpoint."""
    from inspect_robots.log import read_eval_log

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    success, logs = eval_set(
        _task(),
        _RecordingTransientPolicy(),
        CubePickEmbodiment(),
        retry_attempts=1,
        seed=None,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success is True
    seeds = {read_eval_log(path).eval.seed for path in logs[0].source_logs}
    assert len(seeds) == 1
    recorded_seed = seeds.pop()
    assert isinstance(recorded_seed, int)
    assert json.loads(checkpoint.read_text())["identity"]["seed"] == recorded_seed

    again_success, again_logs = eval_set(
        _task(),
        _RecordingTransientPolicy(),
        CubePickEmbodiment(),
        seed=None,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert again_success is True
    assert again_logs[0].eval.seed == recorded_seed


def test_eval_set_checkpoint_mismatch_fails_before_robot_reset(tmp_path: Path) -> None:
    """A changed scene declaration must be detected before moving hardware."""
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )

    class _ResetSpy(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.resets = 0

        def reset(self, scene: Scene, *, seed: int | None = None):  # type: ignore[no-untyped-def]
            self.resets += 1
            return super().reset(scene, seed=seed)

    changed = Task(
        name="resume-demo",
        scenes=[Scene(id="s0", instruction="changed"), Scene(id="s1", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    embodiment = _ResetSpy()
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            changed,
            ScriptedPolicy(),
            embodiment,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert embodiment.resets == 0


def test_checkpoint_rejects_changed_spaces_before_robot_reset(tmp_path: Path) -> None:
    """A reused scene needs the same robot limits and policy observations."""
    from dataclasses import replace

    from inspect_robots.errors import ConfigError
    from inspect_robots.spaces import ObservationSpace

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )

    class _ResetSpy(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.resets = 0

        def reset(self, scene: Scene, *, seed: int | None = None):  # type: ignore[no-untyped-def]
            self.resets += 1
            return super().reset(scene, seed=seed)

    changed_robot = _ResetSpy()
    assert changed_robot.info.action_space.low is not None
    changed_robot.info = replace(
        changed_robot.info,
        action_space=replace(
            changed_robot.info.action_space,
            low=changed_robot.info.action_space.low * 2,
        ),
    )
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            changed_robot,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert changed_robot.resets == 0

    changed_policy = ScriptedPolicy()
    changed_policy.info = replace(
        changed_policy.info,
        observation_space=ObservationSpace(state_keys=frozenset({"eef_pos"})),
    )
    same_robot = _ResetSpy()
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            changed_policy,
            same_robot,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert same_robot.resets == 0


def test_checkpoint_rejects_changed_scorer_settings(tmp_path: Path) -> None:
    """A scorer's threshold is part of the saved result's meaning."""
    from dataclasses import replace

    from inspect_robots.errors import ConfigError
    from inspect_robots.scorer import reached_goal_state

    task = replace(_task(), scorer=reached_goal_state(0.05))
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    eval_set(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            replace(task, scorer=reached_goal_state(0.001)),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )


def test_checkpoint_resolves_string_component_identity_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed registered factory cannot reuse old scenes under the same name."""
    from dataclasses import replace

    from inspect_robots import registry
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    success, _ = eval_set(
        _task(),
        "scripted",
        "cubepick",
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success
    original_factory = registry.registered("embodiment")["cubepick"]

    class _ChangedRobot(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            self.info = replace(self.info, environment_revision="new-rig-revision")

        def close(self) -> None:
            self.closed = True

    made: list[_ChangedRobot] = []

    def changed_robot_factory() -> _ChangedRobot:
        robot = _ChangedRobot()
        made.append(robot)
        return robot

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "cubepick", changed_robot_factory)
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            "scripted",
            "cubepick",
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert len(made) == 1 and made[0].closed

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "cubepick", original_factory)
    original_policy_factory = registry.registered("policy")["scripted"]

    def changed_policy_factory() -> ScriptedPolicy:
        policy = ScriptedPolicy()
        policy.info = replace(policy.info, checkpoint="new-model-revision")
        return policy

    monkeypatch.setitem(registry._FACTORIES["policy"], "scripted", changed_policy_factory)
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            "scripted",
            "cubepick",
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    monkeypatch.setitem(registry._FACTORIES["policy"], "scripted", original_policy_factory)

    # A registered policy can also run against a caller-owned robot instance.
    mixed_success, _ = eval_set(
        _task(),
        "scripted",
        CubePickEmbodiment(),
        checkpoint_path=str(tmp_path / "mixed.json"),
        log_dir=str(log_dir),
    )
    assert mixed_success


def test_checkpoint_fingerprints_adaptive_policy_after_bind(tmp_path: Path) -> None:
    """A policy that adopts the robot's spaces has one effective identity."""
    from dataclasses import replace

    from inspect_robots.embodiment import EmbodimentInfo
    from inspect_robots.spaces import Box

    class _AdaptivePolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.info = replace(self.info, action_space=Box(shape=(1,)))

        def bind(self, embodiment_info: EmbodimentInfo) -> None:
            self.info = replace(self.info, action_space=embodiment_info.action_space)

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    policy = _AdaptivePolicy()
    first_success, first_logs = eval_set(
        _task(),
        policy,
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    same_success, same_logs = eval_set(
        _task(),
        policy,
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    fresh_success, fresh_logs = eval_set(
        _task(),
        _AdaptivePolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert first_success and same_success and fresh_success
    assert same_logs[0].source_logs == first_logs[0].source_logs
    assert fresh_logs[0].source_logs == first_logs[0].source_logs


def test_checkpoint_binds_policy_once_across_retry_attempts(tmp_path: Path) -> None:
    """Identity preflight covers retries without repeating a one-time bind hook."""
    from inspect_robots.embodiment import EmbodimentInfo

    class _OneBindPolicy(_OneTransientFailure):
        def __init__(self) -> None:
            super().__init__()
            self.bind_calls = 0

        def bind(self, embodiment_info: EmbodimentInfo) -> None:
            del embodiment_info
            self.bind_calls += 1
            if self.bind_calls != 1:
                raise RuntimeError("bind called twice")

    policy = _OneBindPolicy()
    success, logs = eval_set(
        _task(),
        policy,
        CubePickEmbodiment(),
        checkpoint_path=str(tmp_path / "run.json"),
        log_dir=str(tmp_path / "logs"),
        retry_attempts=1,
    )
    assert success
    assert policy.bind_calls == 1
    assert len(logs[0].source_logs) == 2


def test_new_checkpoint_preserves_unknown_string_component_error_row(tmp_path: Path) -> None:
    """An unresolved name on a new run is still reported as an eval-set row."""
    checkpoint = tmp_path / "run.json"
    success, logs = eval_set(
        _task(),
        "missing-policy",
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(tmp_path / "logs"),
    )
    assert success is False
    assert logs[0].status == "error"
    assert "missing-policy" in (logs[0].error or "")
    assert not checkpoint.exists()


def test_existing_checkpoint_rejects_unresolved_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved checkpoint cannot be reused after its component factory vanishes."""
    from inspect_robots import registry
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    success, _ = eval_set(
        _task(),
        "scripted",
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success
    monkeypatch.delitem(registry._FACTORIES["policy"], "scripted")
    with pytest.raises(ConfigError, match="checkpoint component could not resolve"):
        eval_set(
            _task(),
            "scripted",
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )


@pytest.mark.parametrize("error_type", [SafetyAbort, EmbodimentFault])
def test_checkpoint_component_safety_halt_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[SafetyAbort] | type[EmbodimentFault],
) -> None:
    """Factory safety failures propagate before a checkpoint or rollout is made."""
    from inspect_robots import registry

    def halt_factory() -> ScriptedPolicy:
        raise error_type("stop")

    registry.registered("policy")
    monkeypatch.setitem(registry._FACTORIES["policy"], "halting", halt_factory)
    checkpoint = tmp_path / "run.json"
    with pytest.raises(error_type):
        eval_set(
            _task(),
            "halting",
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert not checkpoint.exists()


def test_checkpoint_blocks_ambiguous_attempt_after_grading_failure(tmp_path: Path) -> None:
    """A post-step hook failure cannot let a later call replay hidden robot work."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord

    class _ResetSpy(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.resets = 0

        def reset(self, scene: Scene, *, seed: int | None = None):  # type: ignore[no-untyped-def]
            self.resets += 1
            return super().reset(scene, seed=seed)

    def failing_grade(_record: TrialRecord, _scene: Scene) -> None:
        raise RuntimeError("grader disconnected")

    failing_grade.checkpoint_identity = lambda: {"revision": 1}  # type: ignore[attr-defined]
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    first_robot = _ResetSpy()
    success, _ = eval_set(
        _task(),
        ScriptedPolicy(),
        first_robot,
        before_scoring=failing_grade,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert success is False
    assert first_robot.resets == 1
    assert json.loads(checkpoint.read_text())["in_flight"] is True
    assert json.loads(checkpoint.read_text())["attempts"] == []

    second_robot = _ResetSpy()
    with pytest.raises(ConfigError, match="unfinished attempt"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            second_robot,
            before_scoring=failing_grade,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert second_robot.resets == 0


def test_checkpoint_unsaved_attempt_stops_later_tasks(tmp_path: Path) -> None:
    """Later tasks cannot clear the recovery marker for unrecorded robot motion."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.rollout import TrialRecord

    class _ResetSpy(CubePickEmbodiment):
        def __init__(self) -> None:
            super().__init__()
            self.resets: list[str] = []

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            self.resets.append(scene.id)
            return super().reset(scene, seed=seed)

    def failing_grade(_record: TrialRecord, scene: Scene) -> None:
        if scene.id == "first":
            raise RuntimeError("grader disconnected after motion")

    failing_grade.checkpoint_identity = lambda: {"revision": 1}  # type: ignore[attr-defined]
    tasks = [
        Task(
            name=name,
            scenes=[Scene(id=name, instruction="reach")],
            scorer=success_at_end(),
            max_steps=30,
        )
        for name in ["first", "later"]
    ]
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    first_robot = _ResetSpy()
    success, logs = eval_set(
        tasks,
        ScriptedPolicy(),
        first_robot,
        before_scoring=failing_grade,
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )

    assert success is False
    assert first_robot.resets == ["first"]
    assert len(logs) == 1 and logs[0].status == "error"
    manifest = json.loads(checkpoint.read_text())
    assert manifest["in_flight"] is True
    assert manifest["attempts"] == []

    second_robot = _ResetSpy()
    with pytest.raises(ConfigError, match="unfinished attempt"):
        eval_set(
            tasks,
            ScriptedPolicy(),
            second_robot,
            before_scoring=failing_grade,
            checkpoint_path=str(checkpoint),
            log_dir=str(log_dir),
        )
    assert second_robot.resets == []


def test_checkpoint_resume_handles_nonfinite_saved_score(tmp_path: Path) -> None:
    """Strict JSON null scores omit an invalid aggregate metric on every call."""
    from dataclasses import replace

    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Target
    from inspect_robots.scorer import Score

    class _NonfiniteScorer:
        name = "nonfinite"

        def checkpoint_identity(self) -> dict[str, object]:
            return {}

        def __call__(self, record: TrialRecord, target: Target | None) -> Score:
            del record, target
            return Score(value=float("inf"))

    task = replace(_task(), scorer=_NonfiniteScorer())
    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    first_success, first_logs = eval_set(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    again_success, again_logs = eval_set(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(log_dir),
    )
    assert first_success and again_success
    assert "nonfinite" not in first_logs[0].results.metrics
    assert "nonfinite" not in again_logs[0].results.metrics
    assert again_logs[0].source_logs == first_logs[0].source_logs


@pytest.mark.parametrize("attempts", [-1, True, 1.5])
def test_eval_set_rejects_invalid_retry_budget(attempts: object, tmp_path: Path) -> None:
    """A malformed retry budget cannot silently alter hardware run length."""
    from inspect_robots.errors import ConfigError

    with pytest.raises(ConfigError, match="retry_attempts"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            retry_attempts=attempts,  # type: ignore[arg-type]
            log_dir=str(tmp_path),
        )


def test_eval_set_retry_budget_is_bounded_for_repeated_transient_failures(tmp_path: Path) -> None:
    """One additional attempt means exactly two rollouts for an always failing scene."""

    class _AlwaysTransient(_RecordingTransientPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            raise PolicyError("offline", retryable=True)

    task = Task(
        name="bounded",
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    policy = _AlwaysTransient()
    success, logs = eval_set(
        task, policy, CubePickEmbodiment(), retry_attempts=1, log_dir=str(tmp_path)
    )
    assert success is False
    assert policy.resets == ["s", "s"]
    assert len(logs[0].source_logs) == 2
    assert logs[0].results.errored_trials == 1


def test_controller_timeout_does_not_trigger_policy_retry(tmp_path: Path) -> None:
    """Only a failure from policy inference may be retried automatically."""
    from typing import Any

    from inspect_robots.policy import Policy
    from inspect_robots.types import Action

    class _TimeoutController:
        def next_action(
            self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
        ) -> Action:
            del policy, observation, t, store
            raise TimeoutError("controller scheduler stalled")

    policy = _RecordingTransientPolicy()
    success, logs = eval_set(
        _task(),
        policy,
        CubePickEmbodiment(),
        controller=_TimeoutController(),
        retry_attempts=1,
        log_dir=str(tmp_path),
    )
    assert success is False
    assert policy.resets == ["s0", "s1"]
    assert all(sample.retryable_error is False for sample in logs[0].samples)
    assert len(logs[0].source_logs) == 1


@pytest.mark.parametrize("halt", ["safety", "fault"])
def test_eval_set_never_retries_halt_class_failure(halt: str, tmp_path: Path) -> None:
    """A safety abort or hardware fault must end the attempt without auto-advancing."""
    from inspect_robots.errors import EmbodimentFault, SafetyAbort

    class _HaltPolicy(_RecordingTransientPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            if halt == "safety":
                raise SafetyAbort("stop")
            raise EmbodimentFault("motor fault")

    task = Task(
        name="halt",
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    policy = _HaltPolicy()
    success, logs = eval_set(
        task, policy, CubePickEmbodiment(), retry_attempts=3, log_dir=str(tmp_path)
    )
    assert success is False
    assert policy.resets == ["s"]
    assert len(logs[0].source_logs) == 1


@pytest.mark.parametrize("error_type", [SafetyAbort, EmbodimentFault])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_eval_set_halt_suppresses_retries_of_earlier_scenes(
    tmp_path: Path,
    error_type: type[SafetyAbort] | type[EmbodimentFault],
    checkpointed: bool,
) -> None:
    """A later halt blocks robot resets for earlier recoverable scene errors."""
    from inspect_robots.log import read_eval_log

    class _MixedHaltPolicy(_RecordingTransientPolicy):
        def reset(self, scene: Scene) -> None:
            self.scene_id = scene.id
            super().reset(scene)

        def act(self, observation: Observation) -> ActionChunk:
            if self.scene_id == "halt":
                raise error_type("stop")
            return super().act(observation)

    task = Task(
        name="mixed-halt",
        scenes=[Scene(id="early", instruction="reach"), Scene(id="halt", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    policy = _MixedHaltPolicy()
    success, logs = eval_set(
        task,
        policy,
        CubePickEmbodiment(),
        retry_attempts=1,
        checkpoint_path=str(tmp_path / "run.json") if checkpointed else None,
        log_dir=str(tmp_path / "logs"),
    )

    assert success is False
    assert policy.resets == ["early", "halt"]
    assert len(logs[0].source_logs) == 1
    assert logs[0].halted is True
    assert read_eval_log(logs[0].source_logs[0]).halted is True
    assert logs[0].samples[0].retryable_error is True


@pytest.mark.parametrize("error_type", [SafetyAbort, EmbodimentFault])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("phase", ["start", "end"])
def test_eval_set_lifecycle_halt_stops_epochs_scenes_and_retries(
    tmp_path: Path,
    error_type: type[SafetyAbort] | type[EmbodimentFault],
    checkpointed: bool,
    phase: str,
) -> None:
    """Lifecycle halts block later trials and retries of earlier transient errors."""
    from inspect_robots.log import read_eval_log
    from inspect_robots.rollout import TrialRecord

    class _LifecycleHaltPolicy(_RecordingTransientPolicy):
        def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_stamp: str) -> None:
            del epoch, log_dir, run_stamp
            if scene_id == "halt" and phase == "start":
                raise error_type("stop in lifecycle")

        def on_trial_end(self, record: TrialRecord, log_dir: str, run_stamp: str) -> None:
            del log_dir, run_stamp
            if record.scene_id == "halt" and phase == "end":
                raise error_type("stop in lifecycle")

    task = Task(
        name="lifecycle-halt",
        scenes=[Scene(id=name, instruction="reach") for name in ["early", "halt", "after"]],
        scorer=success_at_end(),
        max_steps=30,
        epochs=Epochs(count=2),
    )
    policy = _LifecycleHaltPolicy()
    success, logs = eval_set(
        task,
        policy,
        CubePickEmbodiment(),
        retry_attempts=1,
        checkpoint_path=str(tmp_path / "run.json") if checkpointed else None,
        log_dir=str(tmp_path / "logs"),
    )

    assert success is False
    assert policy.resets == ["early", "early"] + (["halt"] if phase == "end" else [])
    assert len(logs[0].source_logs) == 1
    assert logs[0].halted is True
    attempt = read_eval_log(logs[0].source_logs[0])
    assert attempt.halted is True
    assert [sample.scene_id for sample in attempt.samples] == ["early", "halt"]
    assert len(attempt.samples[1].epochs) == 1
    assert attempt.samples[1].status == "error"
    assert attempt.samples[0].retryable_error is True
    assert attempt.error is not None and "stop in lifecycle" in attempt.error


@pytest.mark.parametrize("error_type", [SafetyAbort, EmbodimentFault])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("phase", ["begin", "poll", "end"])
def test_eval_set_operator_input_halt_stops_scenes_and_retries(
    tmp_path: Path,
    error_type: type[SafetyAbort] | type[EmbodimentFault],
    checkpointed: bool,
    phase: str,
) -> None:
    """An input e-stop must preserve its partial trial and suppress all replay."""
    from dataclasses import replace

    from inspect_robots.console import ConsolePoll
    from inspect_robots.log import read_eval_log

    class _Policy(_RecordingTransientPolicy):
        def reset(self, scene: Scene) -> None:
            self.scene_id = scene.id
            super().reset(scene)

        def act(self, observation: Observation) -> ActionChunk:
            return replace(super().act(observation), inference_latency_s=0.1)

        def transcript(self) -> dict[str, str]:
            return {"scene": self.scene_id}

    policy = _Policy()

    class _Input:
        def begin_trial(self) -> None:
            self.poll_count = 0
            if policy.scene_id == "halt" and phase == "begin":
                raise error_type("input e-stop")

        def poll(self) -> ConsolePoll:
            self.poll_count += 1
            if policy.scene_id == "halt" and phase == "poll" and self.poll_count == 2:
                raise error_type("input e-stop")
            return ConsolePoll()

        def end_trial(self) -> None:
            if policy.scene_id == "halt" and phase == "end":
                raise error_type("input e-stop")

    task = Task(
        name="operator-halt",
        scenes=[Scene(id=name, instruction="reach") for name in ["early", "halt", "after"]],
        scorer=success_at_end(),
        max_steps=30,
    )
    success, logs = eval_set(
        task,
        policy,
        CubePickEmbodiment(),
        operator_input=_Input(),
        retry_attempts=1,
        checkpoint_path=str(tmp_path / "run.json") if checkpointed else None,
        log_dir=str(tmp_path / "logs"),
    )

    assert success is False
    assert policy.resets == ["early", "halt"]
    assert len(logs[0].source_logs) == 1
    assert logs[0].halted is True
    attempt = read_eval_log(logs[0].source_logs[0])
    assert attempt.halted is True
    assert [sample.scene_id for sample in attempt.samples] == ["early", "halt"]
    assert attempt.samples[1].status == "error"
    assert len(attempt.samples[1].epochs) == 1
    assert attempt.samples[1].policy_transcripts == ({"scene": "halt"},)
    assert attempt.error is not None and "input e-stop" in attempt.error
    if phase == "begin":
        assert attempt.stats.total_steps == 0
        assert attempt.stats.mean_inference_latency_s is None
    elif phase == "poll":
        assert attempt.stats.total_steps == 1
        assert attempt.stats.mean_inference_latency_s == pytest.approx(0.1)
    else:
        assert attempt.stats.total_steps > 1
        assert attempt.stats.mean_inference_latency_s == pytest.approx(0.1)


def test_eval_set_interrupt_records_partial_attempt_for_manual_resume(tmp_path: Path) -> None:
    """Ctrl-C publishes the cancelled attempt, then propagates immediately."""
    from inspect_robots.errors import _CancelledTrial
    from inspect_robots.log import read_eval_log

    class _InterruptPolicy(_RecordingTransientPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            raise KeyboardInterrupt("stop")

    checkpoint = tmp_path / "run.json"
    task = Task(
        name="interrupt",
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
    )
    with pytest.raises(_CancelledTrial):
        eval_set(
            task,
            _InterruptPolicy(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            retry_attempts=2,
            log_dir=str(tmp_path / "logs"),
        )
    attempts = json.loads(checkpoint.read_text())["attempts"]
    assert len(attempts) == 1
    path = (checkpoint.parent / attempts[0]["log"]).resolve()
    assert read_eval_log(str(path)).status == "cancelled"


def test_eval_set_string_tasks_and_preflight_errors_keep_rows(tmp_path: Path) -> None:
    """Retry-only mode retains the legacy synthetic row for a bad task name."""
    from inspect_robots.errors import ConfigError

    success, logs = eval_set(
        ["missing-task", "cubepick-reach"],
        "scripted",
        "cubepick",
        retry_attempts=1,
        log_dir=str(tmp_path / "logs"),
    )
    assert success is False
    assert logs[0].status == "error"
    assert logs[0].eval.task == "missing-task"
    assert logs[1].status == "success"
    with pytest.raises(ConfigError, match="could not resolve"):
        eval_set(
            "missing-task",
            "scripted",
            "cubepick",
            checkpoint_path=str(tmp_path / "run.json"),
            log_dir=str(tmp_path / "logs"),
        )


def test_eval_set_checkpoint_handles_multiple_task_slots(tmp_path: Path) -> None:
    """Attempt references stay attached to their ordered task slot on resume."""
    from dataclasses import replace

    tasks = [_task(), replace(_task(), name="second-task")]
    checkpoint = tmp_path / "run.json"
    first_success, first_logs = eval_set(
        tasks,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(tmp_path / "logs"),
    )
    second_success, second_logs = eval_set(
        tasks,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(tmp_path / "logs"),
    )
    assert first_success and second_success
    assert [log.eval.task for log in second_logs] == ["resume-demo", "second-task"]
    assert [len(log.source_logs) for log in first_logs] == [1, 1]
    assert [len(log.source_logs) for log in second_logs] == [1, 1]
    assert [entry["task_index"] for entry in json.loads(checkpoint.read_text())["attempts"]] == [
        0,
        1,
    ]


def test_eval_set_reuses_caller_json_sink_for_attempt(tmp_path: Path) -> None:
    """A supplied canonical sink writes one attempt and remains observable."""
    from inspect_robots.approver import AutoApprover
    from inspect_robots.logging import JsonLogSink

    log_dir = tmp_path / "logs"
    sink = JsonLogSink(str(log_dir))
    success, logs = eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(tmp_path / "run.json"),
        log_dir=str(log_dir),
        sinks=[sink],
        approver=AutoApprover(),
    )
    assert success is True
    assert sink.path is not None and sink.path.exists()
    assert len(list(log_dir.glob("*.json"))) == 2
    assert len(logs[0].source_logs) == 1


def test_eval_set_resumable_preflight_error_does_not_lose_good_task(tmp_path: Path) -> None:
    """A reducer configuration error becomes one row while later tasks still run."""
    good = _task()
    bad = Task(
        name="bad-reducer",
        scenes=[Scene(id="s", instruction="reach")],
        scorer=success_at_end(),
        max_steps=30,
        epochs=Epochs(count=1, reducer="bogus"),
    )
    success, logs = eval_set(
        [bad, good],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        retry_attempts=1,
        log_dir=str(tmp_path),
    )
    assert success is False
    assert logs[0].status == "error"
    assert logs[0].source_logs == ()
    assert logs[1].status == "success"


def test_eval_set_rejects_empty_checkpoint_path(tmp_path: Path) -> None:
    """An empty checkpoint path cannot silently become the working directory."""
    from inspect_robots.errors import ConfigError

    with pytest.raises(ConfigError, match="checkpoint_path"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            checkpoint_path="",
            log_dir=str(tmp_path),
        )


@pytest.mark.parametrize("error", [KeyboardInterrupt("stop"), SafetyAbort("unsafe")])
def test_eval_set_pre_rollout_halt_does_not_publish_attempt(
    error: BaseException, tmp_path: Path
) -> None:
    """A halt before log creation propagates and leaves the checkpoint empty."""
    from inspect_robots.task import TaskEnvelope

    class _PreflightHalt(ScriptedPolicy):
        def bind_task(self, envelope: TaskEnvelope) -> None:
            del envelope
            raise error

    checkpoint = tmp_path / "run.json"
    with pytest.raises(type(error), match=str(error)):
        eval_set(
            _task(),
            _PreflightHalt(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert json.loads(checkpoint.read_text())["attempts"] == []


def test_eval_set_rejects_missing_durable_attempt_log(tmp_path: Path) -> None:
    """An eval return without a JSON sink write cannot advance a checkpoint."""
    from inspect_robots.errors import ConfigError
    from inspect_robots.logging import JsonLogSink

    checkpoint = tmp_path / "run.json"
    with (
        patch.object(JsonLogSink, "on_eval_end", lambda self, log: None),
        pytest.raises(ConfigError, match="durable attempt log"),
    ):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            log_dir=str(tmp_path / "logs"),
        )
    assert json.loads(checkpoint.read_text())["attempts"] == []


@pytest.mark.parametrize("include_later_task", [False, True])
def test_eval_set_persists_attempt_when_later_sink_fails(
    tmp_path: Path, include_later_task: bool
) -> None:
    """A saved attempt survives a sink error and permits remaining tasks to run."""
    from dataclasses import replace

    from inspect_robots.log import EvalLog
    from inspect_robots.logging import JsonLogSink
    from inspect_robots.logging.sink import NullSink

    class _FailingSink(NullSink):
        def on_eval_end(self, log: EvalLog) -> None:
            if log.eval.task == "resume-demo":
                raise RuntimeError("viewer offline")

    checkpoint = tmp_path / "run.json"
    tasks = [_task()]
    if include_later_task:
        tasks.append(replace(_task(), name="later-task"))
    success, logs = eval_set(
        tasks,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        log_dir=str(tmp_path / "logs"),
        sinks=[JsonLogSink(str(tmp_path / "logs")), _FailingSink()],
    )
    assert success is False
    assert logs[0].error is not None and "viewer offline" in logs[0].error
    manifest = json.loads(checkpoint.read_text())
    assert len(manifest["attempts"]) == (2 if include_later_task else 1)
    assert manifest["in_flight"] is False
    if include_later_task:
        assert logs[1].status == "success"


def test_eval_set_checkpoint_inputs_detect_changed_external_setting(tmp_path: Path) -> None:
    """A caller supplied adapter setting participates in checkpoint matching."""
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    log_dir = tmp_path / "logs"
    eval_set(
        _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(checkpoint),
        checkpoint_inputs={"rig_calibration": "revision-a"},
        log_dir=str(log_dir),
    )
    assert "revision-a" not in checkpoint.read_text()
    with pytest.raises(ConfigError, match="checkpoint identity"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            checkpoint_inputs={"rig_calibration": "revision-b"},
            log_dir=str(log_dir),
        )


def test_eval_set_rejects_unserializable_checkpoint_inputs_before_rollout(tmp_path: Path) -> None:
    """Opaque caller settings cannot enter a reproducible checkpoint identity."""
    from inspect_robots.errors import ConfigError

    checkpoint = tmp_path / "run.json"
    with pytest.raises(ConfigError, match="checkpoint_inputs"):
        eval_set(
            _task(),
            ScriptedPolicy(),
            CubePickEmbodiment(),
            checkpoint_path=str(checkpoint),
            checkpoint_inputs={"opaque": object()},
            log_dir=str(tmp_path / "logs"),
        )
    assert not checkpoint.exists()


def test_resumed_aggregate_json_retains_mixed_attempt_frames(tmp_path: Path) -> None:
    """A saved aggregate reads back with two selected frame roots and source logs."""
    from inspect_robots.log import read_eval_log

    log_dir = tmp_path / "logs"
    success, logs = eval_set(
        _task(),
        _RecordingTransientPolicy(),
        CubePickEmbodiment(),
        checkpoint_path=str(tmp_path / "run.json"),
        retry_attempts=1,
        store_frames=True,
        log_dir=str(log_dir),
    )
    assert success is True
    roots = {sample.frames_dir for sample in logs[0].samples}
    assert len(roots) == 2
    assert None not in roots
    assert all(Path(root).is_dir() for root in roots if root is not None)
    saved_aggregates = [
        read_eval_log(str(path))
        for path in log_dir.glob("*.json")
        if read_eval_log(str(path)).source_logs
    ]
    assert len(saved_aggregates) == 1
    assert saved_aggregates[0].source_logs == logs[0].source_logs
    assert {sample.frames_dir for sample in saved_aggregates[0].samples} == roots
