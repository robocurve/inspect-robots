"""Scorer builtins, epoch reducers, and the operator-event scorer."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scorer import (
    Score,
    VLMScorer,
    episode_length,
    get_reducer,
    is_affirmative_verdict,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    reduce_scores,
    success_at_end,
)
from inspect_robots.types import Action, Observation, StepResult


def _record(distances: list[float], *, success: bool, operator: str | None = None) -> TrialRecord:
    steps = []
    for t, d in enumerate(distances):
        last = t == len(distances) - 1
        steps.append(
            StepRecord(
                t=t,
                observation=Observation(),
                action=Action(data=np.zeros(2)),
                result=StepResult(
                    observation=Observation(),
                    terminated=last and success,
                    termination_reason="success" if (last and success) else None,
                    info={"distance": d},
                ),
            )
        )
    rec = TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)
    rec.terminated = success
    rec.termination_reason = "success" if success else None
    rec.operator_judgement = operator
    return rec


def test_success_at_end() -> None:
    assert success_at_end()(_record([0.5, 0.0], success=True), None).value is True
    assert success_at_end()(_record([0.5, 0.3], success=False), None).value is False


def test_episode_length() -> None:
    assert episode_length()(_record([1.0, 0.5, 0.0], success=True), None).value == 3


def test_min_distance_to_goal() -> None:
    assert min_distance_to_goal()(_record([0.9, 0.2, 0.4], success=False), None).value == 0.2


def test_reached_goal_state() -> None:
    assert reached_goal_state(0.05)(_record([0.5, 0.02], success=True), None).value is True
    assert reached_goal_state(0.05)(_record([0.5, 0.2], success=False), None).value is False


def test_operator_scorer_reads_recorded_verdict() -> None:
    assert operator_scorer()(_record([0.5], success=False, operator="success"), None).value is True
    assert operator_scorer()(_record([0.5], success=False, operator="fail"), None).value is False
    # No verdict recorded (unattended run): defaults to not-successful.
    assert operator_scorer()(_record([0.5], success=False), None).value is False


@pytest.mark.parametrize("verdict", ["success", "pass", "yes", "y", "1", "true"])
def test_is_affirmative_verdict_accepts_the_recognized_vocabulary(verdict: str) -> None:
    assert is_affirmative_verdict(verdict) is True


@pytest.mark.parametrize("verdict", ["YES", "Success", "  y  ", "\tPASS\n"])
def test_is_affirmative_verdict_ignores_case_and_surrounding_whitespace(verdict: str) -> None:
    # Operators type free-form text; the comparison rules are part of the
    # contract, not incidental to it.
    assert is_affirmative_verdict(verdict) is True


@pytest.mark.parametrize("verdict", ["fail", "no", "n", "0", "partial", "", "   ", "yes please"])
def test_is_affirmative_verdict_rejects_everything_else(verdict: str) -> None:
    assert is_affirmative_verdict(verdict) is False


def test_is_affirmative_verdict_treats_no_judgement_as_not_affirmative() -> None:
    # Unattended runs record no verdict at all; absence is not assent.
    assert is_affirmative_verdict(None) is False


def test_operator_scorer_agrees_with_the_public_predicate() -> None:
    # The scorer must not carry a second copy of the contract.
    for verdict in ("  Yes ", "fail"):
        score = operator_scorer()(_record([0.5], success=False, operator=verdict), None)
        assert score.value is is_affirmative_verdict(verdict)


def test_reducers_numeric() -> None:
    scores = [Score(value=True), Score(value=False), Score(value=True), Score(value=True)]
    assert reduce_scores("mean", scores).value == 0.75
    assert reduce_scores("max", scores).value == 1.0
    assert reduce_scores("min", scores).value == 0.0


def test_reducer_mode_categorical() -> None:
    scores = [Score(value="a"), Score(value="b"), Score(value="a")]
    assert reduce_scores("mode", scores).value == "a"


def test_mean_over_nonnumeric_string_raises() -> None:
    scores = [Score(value="left"), Score(value="right")]
    with pytest.raises(TypeError, match="non-numeric"):
        reduce_scores("mean", scores)


def test_pass_at_k() -> None:
    # 4 epochs, 1 success: pass@1 = 1/4, pass@4 = 1.0
    scores = [Score(value=True), Score(value=False), Score(value=False), Score(value=False)]
    assert reduce_scores("pass_at_1", scores).value == pytest.approx(0.25)
    assert reduce_scores("pass_at_4", scores).value == pytest.approx(1.0)


def test_unknown_reducer_raises() -> None:
    with pytest.raises(ValueError, match="unknown epoch reducer"):
        get_reducer("nope")


def test_vlm_scorer_stub_points_at_the_vlm_grader() -> None:
    with pytest.raises(NotImplementedError, match=r"'vlm' grader \(--grader vlm\)"):
        VLMScorer()(_record([], success=False), None)
