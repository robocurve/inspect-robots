"""Record editors and analyzers: annotations beside raw logs, re-scoring, read-only analysis."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inspect_robots import Rubric, Scene, Task
from inspect_robots import eval as ir_eval
from inspect_robots.analyzer import Analysis, AnnotatedLog, rescore
from inspect_robots.annotation import (
    GRADING_ERROR,
    JUDGEMENT,
    NOTE,
    Annotation,
    annotations_path,
    append_annotations,
    current,
    lift_annotations,
    read_annotations,
    trial_key,
)
from inspect_robots.cli import main
from inspect_robots.log import EvalLog, read_eval_log
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import Record
from inspect_robots.scorer import Score, operator_scorer, success_at_end

_VERDICTS = ["yes", "no", "partial", "yes"]


def _judged_log(tmp_path: Path, *, epochs: int = 2, reducer: str = "mean") -> tuple[Path, EvalLog]:
    """A two-scene run graded by a scripted operator, with an operator and an automatic rubric."""
    verdicts = iter(_VERDICTS)

    def grade(record: Record, scene: Scene) -> None:
        record.operator_judgement = next(verdicts)
        record.operator_note = f"note for {scene.id}"

    from inspect_robots.task import Epochs

    task = Task(
        name="judged",
        scenes=[Scene(id=f"s{i}", instruction="reach", init_seed=i) for i in range(2)],
        max_steps=40,
        epochs=Epochs(epochs, reducer),
        rubrics=[
            Rubric("stacked", operator_scorer(), text="Bowls stacked."),
            Rubric("reached", success_at_end()),
        ],
    )
    (log,) = ir_eval(
        task,
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        before_scoring=grade,
    )
    (path,) = tmp_path.glob("*.json")
    return path, log


# -- the format --------------------------------------------------------------


def test_annotations_carry_provenance_and_round_trip() -> None:
    annotation = Annotation(
        record_id="abc",
        kind=JUDGEMENT,
        value="yes",
        author="reviewer-1",
        source="editor",
        rubric="stacked",
        metadata={"tool": "x"},
    )
    data = annotation.to_dict()
    assert data["created"] and len(data["annotation_id"]) == 32
    assert Annotation.from_dict({**data, "future_field": 1}) == annotation


@pytest.mark.parametrize("field", ["record_id", "kind", "author", "source"])
def test_annotations_need_identity_and_provenance(field: str) -> None:
    fields: dict[str, Any] = {
        "record_id": "abc",
        "kind": NOTE,
        "value": "x",
        "author": "a",
        "source": "editor",
    }
    fields[field] = " "
    with pytest.raises(ValueError, match=f"annotation {field} must be a non-empty string"):
        Annotation(**fields)


def test_current_prefers_rubric_specific_then_latest_and_drops_superseded() -> None:
    def make(value: str, rubric: str | None = None, **kw: Any) -> Annotation:
        return Annotation(
            record_id="r", kind=JUDGEMENT, value=value, author="a", source="s", rubric=rubric, **kw
        )

    general = make("no")
    later = make("partial")
    specific = make("yes", rubric="stacked")
    assert current([general, later], "r", JUDGEMENT) is later
    assert current([general, specific, later], "r", JUDGEMENT, "stacked") is specific
    assert current([general, specific], "r", JUDGEMENT, "other") is general
    retract = make("no", rubric="stacked", supersedes=specific.annotation_id)
    assert current([specific, retract], "r", JUDGEMENT, "stacked") is retract
    assert current([general], "other-record", JUDGEMENT) is None


# -- lifting the run's judgements ---------------------------------------------


def test_the_runs_judgements_are_lifted_as_annotations(tmp_path: Path) -> None:
    path, log = _judged_log(tmp_path)
    lifted = lift_annotations(log)
    judgements = [a for a in lifted if a.kind == JUDGEMENT]
    assert [a.value for a in judgements] == _VERDICTS
    first = judgements[0]
    assert first.record_id == log.samples[0].record_ids[0]
    assert first.note == "note for s0"
    assert first.author == "operator"
    assert first.annotation_id == f"{first.record_id}:run:judgement"
    assert lift_annotations(log) == lifted
    assert read_annotations(path) == lifted  # no edits yet


def test_older_logs_key_trials_by_position_and_lift_notes_and_grading_errors() -> None:
    from inspect_robots.log import EvalResults, EvalSpec, EvalStats, SceneResult

    sample = SceneResult(
        scene_id="s",
        status="success",
        epochs=({"operator": 0.0}, {"operator": None}, {}),
        operator_judgements=(None, None),
        judgement_sources=(None, "vlm"),
        operator_notes=("looked fine",),
        trial_metadata=({}, {"grading_error": "judge timed out"}),
    )
    log = EvalLog(
        version=1,
        status="success",
        eval=EvalSpec(
            task="t", policy="p", embodiment="e", created="c", inspect_robots_version="0"
        ),
        results=EvalResults(total_scenes=1, total_trials=3),
        stats=EvalStats(started_at="a", completed_at="b", duration_s=1.0, total_steps=3),
        samples=(sample,),
    )
    lifted = lift_annotations(log)
    assert [(a.record_id, a.kind, a.value, a.source) for a in lifted] == [
        ("s/e0", NOTE, "looked fine", "run"),
        ("s/e1", GRADING_ERROR, "judge timed out", "vlm"),
    ]
    assert trial_key(sample, 2) == "s/e2"
    assert trial_key(replace(sample, record_ids=(None,)), 0) == "s/e0"


# -- re-scoring (T5) ---------------------------------------------------------


@pytest.mark.parametrize("reducer", ["mean", "max", "pass_at_1"])
def test_an_unedited_log_rescores_to_exactly_its_stored_results(
    tmp_path: Path, reducer: str
) -> None:
    path, _ = _judged_log(tmp_path, reducer=reducer)
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations) == item.log
    assert item.log.eval.reducer == reducer


def test_an_edited_judgement_changes_only_what_it_should(tmp_path: Path) -> None:
    path, log = _judged_log(tmp_path)
    assert log.samples[0].epochs[1]["stacked"] == 0.0  # "no"
    append_annotations(
        path,
        [
            Annotation(
                record_id=trial_key(log.samples[0], 1),
                kind=JUDGEMENT,
                value="yes",
                author="reviewer",
                source="editor",
            )
        ],
    )
    item = AnnotatedLog.load(path)
    rescored = rescore(item.log, item.annotations)
    assert rescored.samples[0].epochs[1]["stacked"] == 1.0
    assert rescored.samples[0].reduced["stacked"] == 1.0
    assert rescored.samples[1] == log.samples[1]
    assert rescored.results.metrics["stacked"] == 0.75
    assert rescored.results.metrics["reached"] == log.results.metrics["reached"]
    assert read_eval_log(str(path)) == log  # the raw log is never rewritten


def test_swapped_epoch_judgements_rescore_even_when_the_mean_is_unchanged(
    tmp_path: Path,
) -> None:
    path, log = _judged_log(tmp_path)
    sample = log.samples[0]
    assert [e["stacked"] for e in sample.epochs] == [1.0, 0.0]
    append_annotations(
        path,
        [
            Annotation(
                record_id=trial_key(sample, index),
                kind=JUDGEMENT,
                value=value,
                author="reviewer",
                source="editor",
            )
            for index, value in enumerate(["no", "yes"])
        ],
    )
    item = AnnotatedLog.load(path)
    rescored = rescore(item.log, item.annotations)
    assert [e["stacked"] for e in rescored.samples[0].epochs] == [0.0, 1.0]
    assert rescored.samples[0].reduced == sample.reduced
    assert rescored.results.metrics == log.results.metrics


def test_a_judged_abstention_clears_even_when_the_mean_is_unchanged(tmp_path: Path) -> None:
    _, log = _judged_log(tmp_path)
    sample = log.samples[0]
    # Stored: the first trial abstained, so the scene reduces over the second ("no") alone.
    abstaining = replace(
        sample,
        epochs=({"stacked": None, "reached": sample.epochs[0]["reached"]}, sample.epochs[1]),
        operator_judgements=(None, sample.operator_judgements[1]),
        trial_metadata=({"grading_error": "judge down"}, sample.trial_metadata[1]),
        reduced={**sample.reduced, "stacked": 0.0},
    )
    stored = replace(
        log,
        samples=(abstaining, log.samples[1]),
        results=replace(log.results, abstentions={"stacked": 1}),
    )
    judged = Annotation(
        record_id=trial_key(sample, 0), kind=JUDGEMENT, value="no", author="r", source="editor"
    )
    fixed = rescore(stored, [*lift_annotations(stored), judged])
    assert fixed.samples[0].epochs[0]["stacked"] == 0.0
    assert fixed.samples[0].reduced["stacked"] == 0.0
    assert fixed.results.abstentions == {}


def test_a_reducer_that_cannot_reduce_drops_the_scene_and_warns(tmp_path: Path) -> None:
    _, log = _judged_log(tmp_path, reducer="pass_at_2")
    sample = log.samples[0]
    # The second trial abstained, so pass_at_2 has one voting epoch for this scene.
    abstaining = replace(
        sample,
        epochs=(sample.epochs[0], {"stacked": None, "reached": sample.epochs[1]["reached"]}),
        operator_judgements=(sample.operator_judgements[0], None),
        trial_metadata=(sample.trial_metadata[0], {"grading_error": "judge down"}),
    )
    stored = replace(log, samples=(abstaining, log.samples[1]))
    edit = Annotation(
        record_id=trial_key(sample, 0), kind=JUDGEMENT, value="no", author="r", source="editor"
    )
    with pytest.warns(RuntimeWarning, match="pass_at_2.*'stacked'.*'s0'"):
        rescored = rescore(stored, [*lift_annotations(stored), edit])
    assert "stacked" not in rescored.samples[0].reduced
    assert rescored.results.metrics["stacked"] == log.samples[1].reduced["stacked"]


def test_a_rubric_specific_edit_leaves_other_rubrics_alone(tmp_path: Path) -> None:
    path, log = _judged_log(tmp_path)
    key = trial_key(log.samples[1], 0)
    append_annotations(
        path,
        [
            Annotation(
                record_id=key,
                kind=JUDGEMENT,
                value="no",
                author="r",
                source="editor",
                rubric="other",
            )
        ],
    )
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations) == log


def test_grading_errors_abstain_until_someone_judges_the_trial(tmp_path: Path) -> None:
    _, log = _judged_log(tmp_path)
    sample = log.samples[0]
    abstaining = replace(
        sample,
        epochs=({"stacked": None, "reached": sample.epochs[0]["reached"]}, sample.epochs[1]),
        operator_judgements=(None, sample.operator_judgements[1]),
        trial_metadata=({"grading_error": "judge down"}, sample.trial_metadata[1]),
        reduced={**sample.reduced, "stacked": 0.0},
    )
    stored = replace(
        log,
        samples=(abstaining, log.samples[1]),
        results=replace(log.results, abstentions={"stacked": 1}),
    )
    lifted = lift_annotations(stored)
    assert rescore(stored, lifted).samples[0].epochs[0]["stacked"] is None
    fixed = rescore(
        stored,
        [
            *lifted,
            Annotation(
                record_id=trial_key(sample, 0),
                kind=JUDGEMENT,
                value="yes",
                author="r",
                source="editor",
            ),
        ],
    )
    assert fixed.samples[0].epochs[0]["stacked"] == 1.0
    assert fixed.results.abstentions == {}

    # Editing another trial recomputes the rubric; the ungraded trial still abstains.
    other = Annotation(
        record_id=trial_key(sample, 1), kind=JUDGEMENT, value="yes", author="r", source="editor"
    )
    still_missing = rescore(stored, [*lifted, other])
    assert still_missing.samples[0].epochs[0]["stacked"] is None
    assert still_missing.results.abstentions == {"stacked": 1}


def test_the_operator_scorer_scores_annotations_as_it_scores_records() -> None:
    scorer = operator_scorer()
    for verdict in ["yes", "Pass", " no ", "partial", "skip"]:
        record = Record(scene_id="s", epoch=0, seed=0, operator_judgement=verdict)
        annotation = Annotation(
            record_id=record.record_id, kind=JUDGEMENT, value=verdict, author="a", source="s"
        )
        from_record = scorer(record, None)
        from_annotations: Score = scorer.score_annotations(  # type: ignore[attr-defined]
            [annotation], record.record_id, "operator"
        )
        assert from_annotations == from_record
    empty = Record(scene_id="s", epoch=0, seed=0)
    assert scorer.score_annotations([], empty.record_id, "operator") == scorer(empty, None)  # type: ignore[attr-defined]


def _two_judgement_rubric_log(tmp_path: Path) -> tuple[Path, EvalLog]:
    verdicts = iter(["yes"])

    def grade(record: Record, scene: Scene) -> None:
        record.operator_judgement = next(verdicts)

    task = Task(
        name="two-judged",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        max_steps=40,
        rubrics=[
            Rubric("reached", operator_scorer(), text="Reached."),
            Rubric("safe", operator_scorer(), text="Nothing hit."),
        ],
    )
    (log,) = ir_eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), before_scoring=grade
    )
    (path,) = tmp_path.glob("*.json")
    return path, log


def test_an_edit_without_rubric_corrects_the_rubric_the_run_judged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, log = _two_judgement_rubric_log(tmp_path)
    assert log.samples[0].epochs[0] == {"reached": 1.0, "safe": None}
    edit = ["edit", str(path), "--trial", "s0-e0", "--judgement", "no", "--author", "r"]
    assert main(edit) == 0
    assert "the run judged rubric 'reached'" in capsys.readouterr().out
    (correction,) = [a for a in read_annotations(path) if a.source == "editor"]
    assert correction.rubric == "reached"
    assert correction.supersedes == f"{log.samples[0].record_ids[0]}:run:judgement"
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations).samples[0].epochs[0] == {
        "reached": 0.0,
        "safe": None,
    }


def test_a_grader_hint_for_a_single_judgement_rubric_is_not_kept(tmp_path: Path) -> None:
    def grade(record: Record, scene: Scene) -> None:
        record.operator_judgement = "yes"
        record.metadata["judged_rubric"] = "goal"  # e.g. a grader that judged another rubric

    task = Task(
        name="one-judged",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        max_steps=40,
        rubrics=[Rubric("success", operator_scorer()), Rubric("goal", success_at_end(), text="G")],
    )
    (log,) = ir_eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), before_scoring=grade
    )
    assert "judged_rubric" not in log.samples[0].trial_metadata[0]
    (path,) = tmp_path.glob("*.json")
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations) == item.log


def test_a_rubric_edit_never_supersedes_the_runs_general_judgement(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    args = ["edit", str(path), "--trial", "s0-e1", "--judgement", "yes", "--author", "r"]
    assert main([*args, "--rubric", "stacked"]) == 0
    (edit,) = [a for a in read_annotations(path) if a.source == "editor"]
    assert edit.supersedes is None


def test_a_renamed_operator_scorer_still_rescores_from_corrections(tmp_path: Path) -> None:
    verdicts = iter(["yes"])

    def grade(record: Record, scene: Scene) -> None:
        record.operator_judgement = next(verdicts)

    task = Task(
        name="renamed",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        max_steps=40,
        scorer=operator_scorer(name="success"),
    )
    (log,) = ir_eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), before_scoring=grade
    )
    (path,) = tmp_path.glob("*.json")
    assert log.samples[0].epochs[0] == {"success": 1.0}
    assert log.samples[0].rubrics[0]["rescored_by"] == "operator"
    edit = ["edit", str(path), "--trial", "s0-e0", "--judgement", "no", "--author", "r"]
    assert main(edit) == 0
    item = AnnotatedLog.load(path)
    rescored = rescore(item.log, item.annotations)
    assert rescored.samples[0].epochs[0] == {"success": 0.0}
    assert rescored.results.metrics["success"] == 0.0


def test_unresolvable_and_automatic_scorers_keep_their_stored_scores(tmp_path: Path) -> None:
    _, log = _judged_log(tmp_path)
    renamed = replace(
        log,
        samples=tuple(
            replace(
                s,
                rubrics=tuple(
                    {k: v for k, v in r.items() if k != "rescored_by"}
                    | {"scorer": "no-such-scorer"}
                    if r["name"] == "stacked"
                    else r
                    for r in s.rubrics
                ),
            )
            for s in log.samples
        ),
    )
    flipped = [
        Annotation(record_id=key, kind=JUDGEMENT, value="yes", author="r", source="editor")
        for s in log.samples
        for key in s.record_ids
        if key is not None
    ]
    assert rescore(renamed, flipped) == renamed

    no_rubrics = replace(renamed, samples=tuple(replace(s, rubrics=()) for s in renamed.samples))
    assert rescore(no_rubrics, flipped) == no_rubrics

    weird = replace(
        renamed,
        samples=tuple(
            replace(s, rubrics=tuple({**r, "scorer": 7} for r in s.rubrics))
            for s in renamed.samples
        ),
    )
    assert rescore(weird, flipped) == weird


def test_logs_without_a_recorded_reducer_rescore_with_the_mean(tmp_path: Path) -> None:
    _, log = _judged_log(tmp_path)
    old = replace(log, eval=replace(log.eval, reducer=None))
    edit = Annotation(
        record_id=trial_key(log.samples[0], 1), kind=JUDGEMENT, value="yes", author="r", source="e"
    )
    assert rescore(old, [*lift_annotations(old), edit]).samples[0].reduced["stacked"] == 1.0


# -- the annotations file ----------------------------------------------------


def test_the_annotations_file_sits_beside_the_log_and_is_only_appended(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    assert annotations_path(path) == path.with_name(path.stem + ".annotations.jsonl")
    first = Annotation(record_id="a", kind=NOTE, value="1", author="x", source="editor")
    second = Annotation(record_id="b", kind=NOTE, value="2", author="x", source="editor")
    append_annotations(path, [first])
    append_annotations(path, [second])
    lines = annotations_path(path).read_text().splitlines()
    assert [json.loads(line)["value"] for line in lines] == ["1", "2"]
    assert read_annotations(path)[-2:] == (first, second)


def test_a_malformed_annotations_line_names_the_file_and_line(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    annotations_path(path).write_text('\n{"kind": "note"}\n')
    with pytest.raises(ValueError, match=r"annotations.jsonl:2: not a valid annotation"):
        read_annotations(path)


# -- analyzers ---------------------------------------------------------------


def test_builtin_analyzers_read_logs_without_changing_them(tmp_path: Path) -> None:
    from inspect_robots.registry import resolve

    path, _ = _judged_log(tmp_path)
    before = path.read_bytes()
    items = [AnnotatedLog.load(path)]
    for name in ["scores", "terminations", "digest", "html"]:
        analysis = resolve("analyzer", name).analyze(items)
        assert isinstance(analysis, Analysis) and analysis.name == name
        assert analysis.text
    assert path.read_bytes() == before
    assert not annotations_path(path).exists()

    terminations = resolve("analyzer", "terminations").analyze(items)
    assert (
        terminations.data["counts"] == {"success": 4}
        or sum(terminations.data["counts"].values()) == 4
    )
    html = resolve("analyzer", "html").analyze(items)
    assert list(html.files) == [f"{path.stem}.html"]


def test_the_terminations_analyzer_lists_stated_reasons(tmp_path: Path) -> None:
    from inspect_robots.registry import resolve

    path, log = _judged_log(tmp_path)
    sample = replace(
        log.samples[0],
        termination_reasons=("give_up", "done"),
        termination_details=("cannot see the drawer",),
    )
    item = AnnotatedLog(path=path, log=replace(log, samples=(sample,)))
    analysis = resolve("analyzer", "terminations").analyze([item])
    assert analysis.data["counts"] == {"give_up": 1, "done": 1}
    assert analysis.data["stated"][0]["detail"] == "cannot see the drawer"
    assert "[give_up] s0: cannot see the drawer" in analysis.text
    unknown = replace(sample, termination_reasons=(None,), termination_details=())
    counts = resolve("analyzer", "terminations").analyze(
        [replace(item, log=replace(log, samples=(unknown,)))]
    )
    assert counts.data["counts"] == {"unknown": 1}


# -- the CLI -----------------------------------------------------------------


def test_cli_edit_then_score_shows_the_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, log = _judged_log(tmp_path)
    assert (
        main(
            [
                "edit",
                str(path),
                "--trial",
                "s0-e1",
                "--judgement",
                "yes",
                "--author",
                "reviewer",
                "--reason",
                "stack held after release",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "supersedes" in out
    (edit,) = [a for a in read_annotations(path) if a.source == "editor"]
    assert edit.supersedes == f"{log.samples[0].record_ids[1]}:run:judgement"
    assert edit.note == "stack held after release"

    out_path = tmp_path / "rescored.json"
    assert main(["score", str(path), "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "stacked: 0.500 -> 0.750  (changed)" in out
    assert "reached:" in out and f"wrote {out_path}" in out
    assert read_eval_log(str(out_path)).results.metrics["stacked"] == 0.75

    record_id = log.samples[1].record_ids[0]
    assert (
        main(
            [
                "edit",
                str(path),
                "--record",
                str(record_id),
                "--note",
                "bowl chipped",
                "--author",
                "reviewer",
            ]
        )
        == 0
    )
    assert "supersedes" not in capsys.readouterr().out


def test_cli_rubric_edit_leaves_other_judgement_rubrics_alone(tmp_path: Path) -> None:
    from inspect_robots.task import Epochs

    verdicts = iter(["yes"])

    def grade(record: Record, scene: Scene) -> None:
        record.operator_judgement = next(verdicts)

    task = Task(
        name="two-judged",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        max_steps=40,
        epochs=Epochs(1, "mean"),
        rubrics=[
            Rubric("safety", operator_scorer(), text="Nothing hit."),
            Rubric("capability", operator_scorer(), text="Bowls stacked."),
        ],
    )
    (log,) = ir_eval(
        task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), before_scoring=grade
    )
    (path,) = tmp_path.glob("*.json")
    # One verdict per trial: it counts for the first judgement rubric; the
    # other abstains instead of inheriting it.
    assert log.samples[0].epochs[0] == {"safety": 1.0, "capability": None}
    assert log.samples[0].trial_metadata[0]["judged_rubric"] == "safety"
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations) == item.log
    args = ["edit", str(path), "--trial", "s0-e0", "--author", "r"]
    assert main([*args, "--judgement", "yes", "--rubric", "capability"]) == 0
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations).samples[0].epochs[0] == {
        "safety": 1.0,
        "capability": 1.0,
    }
    assert main([*args, "--judgement", "no", "--rubric", "safety"]) == 0
    safety_edit = [a for a in read_annotations(path) if a.rubric == "safety"][-1]
    assert safety_edit.supersedes == f"{log.samples[0].record_ids[0]}:run:judgement"
    item = AnnotatedLog.load(path)
    assert rescore(item.log, item.annotations).samples[0].epochs[0] == {
        "safety": 0.0,
        "capability": 1.0,
    }


def test_a_cross_rubric_supersedes_does_not_retire_a_general_judgement() -> None:
    general = Annotation(record_id="r", kind=JUDGEMENT, value="yes", author="a", source="run")
    scoped = Annotation(
        record_id="r",
        kind=JUDGEMENT,
        value="no",
        author="b",
        source="editor",
        rubric="safety",
        supersedes=general.annotation_id,
    )
    assert current([general, scoped], "r", JUDGEMENT, "capability") == general
    assert current([general, scoped], "r", JUDGEMENT, "safety") == scoped


def test_cli_score_export_carries_its_corrections_so_rescoring_reproduces_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = _judged_log(tmp_path)
    edit = ["edit", str(path), "--trial", "s0-e1", "--judgement", "yes", "--author", "r"]
    assert main(edit) == 0
    out_path = tmp_path / "exported.json"
    assert main(["score", str(path), "--out", str(out_path)]) == 0
    assert f"wrote {annotations_path(out_path)}" in capsys.readouterr().out

    exported = AnnotatedLog.load(out_path)
    assert exported.log.results.metrics["stacked"] == 0.75
    again = rescore(exported.log, exported.annotations)
    assert again.results.metrics["stacked"] == 0.75
    assert again.samples == exported.log.samples


def test_cli_score_refuses_an_export_outside_the_logs_directory(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "log.json"
    with pytest.raises(SystemExit, match="must be in the input log's directory"):
        main(["score", str(path), "--out", str(elsewhere)])
    assert not elsewhere.exists()


def test_cli_score_export_without_edits_clears_a_stale_sidecar(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    out_path = tmp_path / "exported.json"
    annotations_path(out_path).write_text("stale\n", encoding="utf-8")
    assert main(["score", str(path), "--out", str(out_path)]) == 0
    assert not annotations_path(out_path).exists()
    AnnotatedLog.load(out_path)
    assert main(["score", str(path), "--out", str(out_path)]) == 0  # nothing to clear
    assert not annotations_path(out_path).exists()


def test_cli_edit_rejects_unknown_trials(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    with pytest.raises(SystemExit, match="--trial 's9-e0' is not a trial in this log"):
        main(["edit", str(path), "--trial", "s9-e0", "--note", "x", "--author", "a"])
    with pytest.raises(SystemExit, match="--record 'nope' is not a trial in this log"):
        main(["edit", str(path), "--record", "nope", "--note", "x", "--author", "a"])


def test_cli_score_refuses_to_overwrite_the_log(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    with pytest.raises(SystemExit, match="would overwrite the input log"):
        main(["score", str(path), "--out", str(path)])


def test_cli_score_marks_nothing_when_unedited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = _judged_log(tmp_path)
    assert main(["score", str(path)]) == 0
    assert "(changed)" not in capsys.readouterr().out


def test_cli_analyze_prints_text_json_and_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = _judged_log(tmp_path)
    assert main(["analyze", "scores", str(path)]) == 0
    assert "0 edit(s) after the run" in capsys.readouterr().out
    assert main(["analyze", "terminations", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]
    out_dir = tmp_path / "pages"
    assert main(["analyze", "html", str(path), "--out", str(out_dir)]) == 0
    assert (out_dir / f"{path.stem}.html").exists()


def test_cli_analyze_writes_files_to_the_working_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _judged_log(tmp_path / "logs")
    monkeypatch.chdir(tmp_path)
    assert main(["analyze", "html", str(path)]) == 0
    assert (tmp_path / f"{path.stem}.html").exists()


def test_cli_analyze_names_registered_analyzers_on_a_typo(tmp_path: Path) -> None:
    path, _ = _judged_log(tmp_path)
    with pytest.raises(SystemExit, match=r"unknown analyzer 'scorez'; registered: .*scores"):
        main(["analyze", "scorez", str(path)])


def test_cli_analyze_rejects_an_analyzer_that_returns_the_wrong_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_robots import registry

    class _Bad:
        name = "bad"

        def analyze(self, logs: object) -> str:
            return "oops"

    path, _ = _judged_log(tmp_path)
    monkeypatch.setitem(registry._FACTORIES["analyzer"], "bad", _Bad)
    with pytest.raises(SystemExit, match="returned str; expected Analysis"):
        main(["analyze", "bad", str(path)])


def test_a_registered_annotation_aware_scorer_rescores_by_its_own_name(tmp_path: Path) -> None:
    from inspect_robots.registry import _FACTORIES, scorer

    class _PartialCredit:
        name = "partial_credit"

        def __call__(self, record: Record, target: object) -> Score:
            return Score(value=0.5)

        def score_annotations(self, annotations: object, record_id: str, rubric: str) -> Score:
            return Score(value=0.5)

    scorer("partial_credit")(_PartialCredit)
    try:
        task = Task(
            name="partial",
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            max_steps=5,
            rubrics=[Rubric("credit", "partial_credit")],
        )
        (log,) = ir_eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
        assert "rescored_by" not in log.samples[0].rubrics[0]
        (path,) = tmp_path.glob("*.json")
        item = AnnotatedLog.load(path)
        assert rescore(item.log, item.annotations) == item.log
        assert item.log.samples[0].epochs[0] == {"credit": 0.5}
    finally:
        _FACTORIES["scorer"].pop("partial_credit", None)
