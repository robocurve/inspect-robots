"""Rubrics: tasks measure trials against named rubrics, and results are keyed by rubric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.errors import ConfigError
from inspect_robots.log import read_eval_log
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import Record
from inspect_robots.rubric import SAFETY, Rubric, merge
from inspect_robots.scene import Scene, Target
from inspect_robots.scorer import Score, episode_length, success_at_end
from inspect_robots.session import OperatorSession
from inspect_robots.task import Task


class _Constant:
    """A scorer with a fixed value, to tell rubrics apart in results."""

    def __init__(self, value: float, name: str = "constant") -> None:
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, record: Record, target: Target | None) -> Score:
        return Score(value=self._value)


def _scene(scene_id: str = "s0", **kwargs: Any) -> Scene:
    return Scene(id=scene_id, instruction="reach the cube", init_seed=0, **kwargs)


# -- the type ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "has space", "-leading", "a/b", 3])
def test_rubric_names_are_simple_identifiers(name: object) -> None:
    with pytest.raises(ConfigError, match="not a simple identifier"):
        Rubric(name=name, scorer="success_at_end")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"text": None}, "text must be a string"),
        ({"kind": " "}, "kind must be a non-empty string"),
        ({"scorer": " "}, "scorer name must not be empty"),
    ],
)
def test_rubric_fields_are_validated(kwargs: dict[str, Any], match: str) -> None:
    fields: dict[str, Any] = {"name": "success", "scorer": "success_at_end", **kwargs}
    with pytest.raises(ConfigError, match=match):
        Rubric(**fields)


def test_rubric_resolves_registry_scorers_and_describes_itself() -> None:
    by_name = Rubric(name="success", scorer="success_at_end", text="The cube is reached.")
    assert by_name.resolve_scorer().name == "success_at_end"
    assert by_name.describe() == {
        "name": "success",
        "kind": "capability",
        "text": "The cube is reached.",
        "scorer": "success_at_end",
        "metadata": {},
    }
    by_object = Rubric(name="safe", scorer=_Constant(1.0), kind=SAFETY, metadata={"v": 2})
    assert by_object.describe()["scorer"] == "constant"
    assert by_object.describe()["metadata"] == {"v": 2}


def test_merge_replaces_by_name_and_appends_new_rubrics() -> None:
    a, b = Rubric("a", "success_at_end"), Rubric("b", "success_at_end")
    b2, c = Rubric("b", "episode_length", text="new"), Rubric("c", "success_at_end")
    assert merge((a, b), (c, b2)) == (a, b2, c)
    assert merge((a,), ()) == (a,)


def test_duplicate_or_foreign_rubrics_are_rejected() -> None:
    rubric = Rubric("success", "success_at_end")
    with pytest.raises(ConfigError, match="duplicate rubric name 'success'"):
        _scene(rubrics=(rubric, rubric))
    with pytest.raises(ConfigError, match="rubrics must be Rubric objects"):
        Task(name="t", scenes=[_scene()], max_steps=5, rubrics=["success"])  # type: ignore[list-item]


# -- tasks -------------------------------------------------------------------


def test_a_task_declares_exactly_one_of_rubrics_or_scorer() -> None:
    with pytest.raises(ConfigError, match="exactly one of rubrics or scorer"):
        Task(name="t", scenes=[_scene()], max_steps=5)
    with pytest.raises(ConfigError, match="exactly one of rubrics or scorer"):
        Task(
            name="t",
            scenes=[_scene()],
            scorer="success_at_end",
            max_steps=5,
            rubrics=[Rubric("success", "success_at_end")],
        )


def test_scorer_shorthand_names_each_rubric_after_its_scorer() -> None:
    task = Task(
        name="t", scenes=[_scene()], scorer=[success_at_end(), episode_length()], max_steps=5
    )
    assert [r.name for r in task.task_rubrics] == ["success_at_end", "episode_length"]
    assert [s.name for s in task.scorers] == ["success_at_end", "episode_length"]


def test_scene_rubrics_override_the_tasks() -> None:
    override = Rubric("success", "episode_length", text="scene-specific")
    scene = _scene(rubrics=(override,))
    task = Task(
        name="t", scenes=[scene], max_steps=5, rubrics=[Rubric("success", "success_at_end")]
    )
    assert task.rubrics_for(scene) == (override,)
    assert [s.name for s in task.scorers] == ["success_at_end"]


# -- eval --------------------------------------------------------------------


class _RubricRecordingGrader:
    name = "recording"

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def grade(self, record: Record, scene: Scene) -> None:
        self.seen.append(tuple(r.name for r in scene.rubrics))


def test_results_are_keyed_by_rubric_and_rubrics_are_logged(tmp_path: Path) -> None:
    task = Task(
        name="t",
        scenes=[
            _scene("plain"),
            _scene(
                "special",
                rubrics=(
                    Rubric("safe", _Constant(0.5), text="Nothing falls.", kind=SAFETY),
                    Rubric("tidy", _Constant(0.25)),
                ),
            ),
        ],
        max_steps=40,
        rubrics=[
            Rubric("reached", _Constant(1.0), text="The cube is reached."),
            Rubric("safe", _Constant(0.0), text="No contact.", kind=SAFETY),
        ],
    )
    grader = _RubricRecordingGrader()
    (log,) = ir_eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), grader=grader
    )

    plain, special = log.samples
    assert plain.reduced == {"reached": 1.0, "safe": 0.0}
    assert special.reduced == {"reached": 1.0, "safe": 0.5, "tidy": 0.25}
    assert log.results.metrics == {"reached": 1.0, "safe": 0.25, "tidy": 0.25}
    assert [r["name"] for r in log.eval.rubrics] == ["reached", "safe"]
    assert [(r["name"], r["text"]) for r in special.rubrics] == [
        ("reached", "The cube is reached."),
        ("safe", "Nothing falls."),
        ("tidy", ""),
    ]
    assert grader.seen == [("reached", "safe"), ("reached", "safe", "tidy")]

    (log_file,) = tmp_path.glob("*.json")
    reread = read_eval_log(str(log_file))
    assert reread.version == 2
    assert reread.samples[1].rubrics == special.rubrics
    assert reread.eval.rubrics == log.eval.rubrics


def test_a_named_rubric_names_itself_when_its_scorer_fails(tmp_path: Path) -> None:
    class _Boom(_Constant):
        def __call__(self, record: Record, target: Target | None) -> Score:
            raise RuntimeError("scorer bug")

    task = Task(name="t", scenes=[_scene()], max_steps=40, rubrics=[Rubric("safe", _Boom(0))])
    (log,) = ir_eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert log.error == "rubric 'safe' (scorer 'constant') failed: scorer bug"


def test_schema_one_logs_still_read(tmp_path: Path) -> None:
    task = Task(name="t", scenes=[_scene()], scorer="success_at_end", max_steps=40)
    (log,) = ir_eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    data = log.to_dict()
    data["version"] = 1
    del data["eval"]["rubrics"]
    for sample in data["samples"]:
        del sample["rubrics"]
        del sample["termination_details"]
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(data))
    old = read_eval_log(str(path))
    assert old.version == 1
    assert old.eval.rubrics == []
    assert old.samples[0].rubrics == ()
    assert old.samples[0].reduced == log.samples[0].reduced

    data["version"] = 3
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"reads versions \[1, 2\]"):
        read_eval_log(str(path))


def test_the_operator_sees_every_rubric_with_text() -> None:
    written: list[str] = []
    session = OperatorSession(input_fn=lambda _prompt: "y", write=written.append)
    scene = _scene(
        rubrics=(
            Rubric("reached", "success_at_end", text="The cube is reached."),
            Rubric("quiet", "success_at_end"),
            Rubric("safe", "success_at_end", text="Nothing falls.\nNo contact.", kind=SAFETY),
        )
    )
    session.prompt_verdict(Record(scene_id="s0", epoch=0, seed=0), scene)
    assert written[:5] == [
        "rubric reached (capability):\n",
        "  The cube is reached.\n",
        "rubric safe (safety):\n",
        "  Nothing falls.\n",
        "  No contact.\n",
    ]


def test_a_seconds_budget_hit_is_noted_as_a_time_limit() -> None:
    written: list[str] = []
    session = OperatorSession(input_fn=lambda _prompt: "n", write=written.append)
    record = Record(
        scene_id="s0", epoch=0, seed=0, truncated=True, termination_reason="max_seconds"
    )
    session.prompt_verdict(record, _scene())
    assert written == ["note: this trial hit the time limit before terminating\n"]


def test_a_safety_rubric_warns_when_the_policy_cannot_give_up() -> None:
    from inspect_robots.compat import check_compatibility
    from inspect_robots.policy import PolicyInfo

    task = Task(
        name="t",
        scenes=[_scene(rubrics=(Rubric("harm", _Constant(0.0), kind=SAFETY),))],
        max_steps=5,
        rubrics=[Rubric("reached", _Constant(1.0))],
    )
    policy = ScriptedPolicy()
    report = check_compatibility(policy, CubePickEmbodiment(), task)
    (warning,) = [issue for issue in report.warnings if issue.code == "refusal_path"]
    assert "['harm']" in warning.message

    policy.info = PolicyInfo(
        name="refuser",
        action_space=policy.info.action_space,
        observation_space=policy.info.observation_space,
        terminal_actions=frozenset({"give_up"}),
    )
    report = check_compatibility(policy, CubePickEmbodiment(), task)
    assert not [issue for issue in report.warnings if issue.code == "refusal_path"]
