"""``EvalSpec`` records what actually graded a run, not what was asked for.

The companion to ``judgement_sources`` (plan 0080): that field says which path
produced each verdict, these say what configuration that path ran under.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inspect_robots import eval as ir_eval
from inspect_robots import eval_set as ir_eval_set
from inspect_robots.grader import _DEFAULT_RUBRIC, Grader, vlm_grader
from inspect_robots.log import read_eval_log
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

_KEY = "grader-config-test-key"


class _CapturePost:
    """An HttpPost double that records each request body and returns a verdict."""

    def __init__(self) -> None:
        content = json.dumps({"choices": [{"message": {"content": "GRADE: success"}}]})
        self.response: tuple[int, bytes] = (200, content.encode())
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        del url, headers
        self.bodies.append(json.loads(body))
        return self.response


class _PlainGrader:
    """A conforming grader with no ``config`` hook, like every plugin today."""

    name = "plain"

    def grade(self, record: TrialRecord, scene: Scene) -> None:
        """Stamp a verdict without exposing any configuration."""
        del scene
        record.operator_judgement = "success"


def _task(name: str = "grader-config") -> Task:
    # max_steps=1 keeps every trial off the definitive-termination path, so the
    # grader actually spends a model call and the recorded config governs it.
    return Task(
        name=name,
        scenes=[Scene(id="s0", instruction="reach")],
        scorer=success_at_end(),
        max_steps=1,
    )


def _vlm(post: _CapturePost, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Grader:
    monkeypatch.setenv("GRADER_CONFIG_KEY", _KEY)
    return vlm_grader(
        "judge-model",
        api_key_env="GRADER_CONFIG_KEY",
        http_post=post,
        **kwargs,
    )


def _run(grader: Grader | None, tmp_path: Path, task: Task | None = None) -> Any:
    (log,) = ir_eval(
        task or _task(),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        grader=grader,
        log_dir=str(tmp_path),
    )
    return log


def test_spec_records_the_grader_name_and_its_effective_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = _CapturePost()
    log = _run(_vlm(post, monkeypatch, base_url="https://example.invalid/v1"), tmp_path)

    assert log.eval.grader == "vlm"
    assert log.eval.grader_config["model"] == "judge-model"
    assert log.eval.grader_config["base_url"] == "https://example.invalid/v1"
    assert log.eval.grader_config["max_cameras"] == 4
    # The grader really did spend a call under exactly that model.
    assert [body["model"] for body in post.bodies] == ["judge-model"]


def test_recorded_rubric_is_the_resolved_text_not_the_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rubric_file is read at construction; the log must hold what was sent."""
    rubric_path = tmp_path / "rubric.md"
    rubric_path.write_text("Succeed only if the cube is lifted clear.", encoding="utf-8")
    post = _CapturePost()

    log = _run(_vlm(post, monkeypatch, rubric_file=str(rubric_path)), tmp_path)

    recorded = log.eval.grader_config["rubric"]
    assert recorded == "Succeed only if the cube is lifted clear."
    assert str(rubric_path) not in recorded
    # The invariant that matters: the recorded rubric is the one in the prompt.
    prompt = post.bodies[0]["messages"][0]["content"][0]["text"]
    assert recorded in prompt


def test_recorded_rubric_is_the_default_when_none_was_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omitted rubric still governs grading, so it must not record as None."""
    post = _CapturePost()

    log = _run(_vlm(post, monkeypatch), tmp_path)

    assert log.eval.grader_config["rubric"] == _DEFAULT_RUBRIC
    prompt = post.bodies[0]["messages"][0]["content"][0]["text"]
    assert _DEFAULT_RUBRIC in prompt


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, None),
        ({"effort": None}, "none"),
        ({"effort": "low"}, "low"),
        ({"effort": 0.25}, 0.25),
    ],
)
def test_recorded_effort_is_what_rides_the_request(
    kwargs: dict[str, Any],
    expected: str | float | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset records ``None`` and sends nothing; ``-G effort=none`` records ``"none"``."""
    post = _CapturePost()

    log = _run(_vlm(post, monkeypatch, **kwargs), tmp_path)

    assert log.eval.grader_config["effort"] == expected
    sent = post.bodies[0].get("reasoning_effort")
    if expected is None:
        assert "reasoning_effort" not in post.bodies[0]
    else:
        assert sent == expected


def test_a_scene_rubric_overrides_grading_without_rewriting_the_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec keeps the run-level fallback; the override stays with its scene."""
    post = _CapturePost()
    task = Task(
        name="scene-rubric",
        scenes=[
            Scene(
                id="s0",
                instruction="reach",
                metadata={"rubric": "Succeed only if the gripper is closed."},
            )
        ],
        scorer=success_at_end(),
        max_steps=1,
    )

    log = _run(_vlm(post, monkeypatch, rubric="run-level fallback"), tmp_path, task)

    # The spec records the run-level rubric, which is what it is.
    assert log.eval.grader_config["rubric"] == "run-level fallback"
    # The scene's own rubric is what actually reached the prompt, and it is
    # already persisted with the scene rather than duplicated into the spec.
    prompt = post.bodies[0]["messages"][0]["content"][0]["text"]
    assert "Succeed only if the gripper is closed." in prompt
    assert "run-level fallback" not in prompt
    assert log.samples[0].scene_metadata["rubric"] == "Succeed only if the gripper is closed."


def test_api_key_never_reaches_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential is not provenance; it must appear nowhere on disk."""
    post = _CapturePost()

    log = _run(_vlm(post, monkeypatch), tmp_path)

    assert _KEY not in json.dumps(log.to_dict())
    written = next(Path(tmp_path).glob("*.json"))
    assert _KEY not in written.read_text(encoding="utf-8")


def test_a_grader_without_a_config_hook_is_still_named(tmp_path: Path) -> None:
    """Out-of-tree graders predate the hook and must not be forced to adopt it."""
    log = _run(_PlainGrader(), tmp_path)

    assert log.eval.grader == "plain"
    assert log.eval.grader_config == {}


def test_an_ungraded_run_records_no_grader(tmp_path: Path) -> None:
    log = _run(None, tmp_path)

    assert log.eval.grader is None
    assert log.eval.grader_config == {}


def test_grader_provenance_survives_the_json_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = _CapturePost()
    log = _run(_vlm(post, monkeypatch, effort="low"), tmp_path)

    restored = read_eval_log(str(next(Path(tmp_path).glob("*.json"))))

    assert restored.eval.grader == "vlm"
    assert restored.eval.grader_config == log.eval.grader_config
    assert restored.eval.grader_config["effort"] == "low"


def test_eval_set_records_the_grader_for_every_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval_set resolves the grader once; each task's own spec must still say so."""
    post = _CapturePost()

    success, logs = ir_eval_set(
        [_task("set-a"), _task("set-b")],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        grader=_vlm(post, monkeypatch),
        log_dir=str(tmp_path),
    )

    assert success
    assert len(logs) == 2
    assert [log.eval.grader for log in logs] == ["vlm", "vlm"]
    assert all(log.eval.grader_config["model"] == "judge-model" for log in logs)


def test_eval_set_without_a_grader_records_none(tmp_path: Path) -> None:
    success, logs = ir_eval_set(
        [_task("set-c")],
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    assert success
    assert logs[0].eval.grader is None
    assert logs[0].eval.grader_config == {}
