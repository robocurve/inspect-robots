"""Run uncertainty and paired comparison (plan 0086).

The fixtures are built so every number a test asserts can be checked by hand: per-scene values are
chosen, not sampled, and the statistics are compared against closed forms where one exists.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from inspect_robots import (
    Comparison,
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    MetricEvidence,
    SceneResult,
    compare_logs,
    evidence,
    metric_evidence,
)
from inspect_robots._compare import _final_verdict, run_compare
from inspect_robots.cli import main


def make_log(
    per_scene: Mapping[str, Sequence[float | None]],
    *,
    scorer: str = "success",
    task: str = "t",
    policy: str = "p",
    embodiment: str = "mock",
    max_steps: int | None = 50,
    extra: dict[str, float | None] | None = None,
) -> EvalLog:
    """A log whose scene ``s`` has one epoch per value; ``None`` is an errored (empty) trial."""
    samples = []
    for scene_id, values in per_scene.items():
        epochs: tuple[dict[str, float | None], ...] = tuple(
            {} if v is None else {scorer: float(v), **(extra or {})} for v in values
        )
        samples.append(SceneResult(scene_id=scene_id, status="success", epochs=epochs))
    total = sum(len(v) for v in per_scene.values())
    errored = sum(1 for v in per_scene.values() for x in v if x is None)
    return EvalLog(
        version=EvalLog.SCHEMA_VERSION,
        status="success",
        eval=EvalSpec(
            task=task,
            policy=policy,
            embodiment=embodiment,
            created="now",
            inspect_robots_version="test",
            max_steps=max_steps,
        ),
        results=EvalResults(len(per_scene), total, {}, errored),
        stats=EvalStats(started_at="a", completed_at="b", duration_s=0.0, total_steps=0),
        samples=tuple(samples),
    )


def write(log: EvalLog, path: Path) -> str:
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")
    return str(path)


# -- metric_evidence ---------------------------------------------------------


def test_mean_is_trial_weighted_and_interval_brackets_it() -> None:
    log = make_log({f"s{i}": [i / 10, i / 10 + 0.05] for i in range(8)})
    ev = metric_evidence(log)["success"]
    assert isinstance(ev, MetricEvidence)
    assert ev.mean == pytest.approx(np.mean([i / 10 + d for i in range(8) for d in (0, 0.05)]))
    assert ev.ci_low < ev.mean < ev.ci_high
    assert (ev.n_scenes, ev.scored_trials, ev.attempted_trials) == (8, 16, 16)
    assert ev.coverage == 1.0


def test_interval_is_deterministic_for_a_seed() -> None:
    log = make_log({f"s{i}": [i % 3 / 2] for i in range(10)})
    first = metric_evidence(log, seed=3)["success"]
    again = metric_evidence(log, seed=3)["success"]
    assert (first.ci_low, first.ci_high) == (again.ci_low, again.ci_high)


def test_adding_epochs_does_not_manufacture_precision() -> None:
    """Scenes are the unit: repeating each scene's value 10x must not shrink the interval."""
    base = {f"s{i}": [i / 10] for i in range(6)}
    many = {s: v * 10 for s, v in base.items()}
    one = metric_evidence(make_log(base))["success"]
    ten = metric_evidence(make_log(many))["success"]
    assert ten.ci_high - ten.ci_low == pytest.approx(one.ci_high - one.ci_low, rel=0.05)


def test_errored_trials_lower_coverage_not_the_mean() -> None:
    log = make_log({"s0": [1.0, None], "s1": [0.0, None], "s2": [1.0, 1.0]})
    ev = metric_evidence(log)["success"]
    assert (ev.scored_trials, ev.attempted_trials) == (4, 6)
    assert ev.coverage == pytest.approx(4 / 6)
    assert ev.mean == pytest.approx(2 / 3)  # scenes weigh equally, as in results.metrics


def test_single_scene_has_no_interval_and_missing_scorer_is_reported() -> None:
    log = make_log({"s0": [0.5, 0.7]})
    out = metric_evidence(log, ["success", "absent"])
    assert out["success"].n_scenes == 1
    assert math.isnan(out["success"].ci_low) and math.isnan(out["success"].ci_high)
    assert out["absent"].n_scenes == 0 and math.isnan(out["absent"].mean)


def test_non_numeric_and_non_finite_values_are_ignored() -> None:
    log = make_log({"s0": [1.0]})
    sample = log.samples[0]
    raw = ({"success": 1.0, "label": "x", "flag": True, "bad": float("nan"), "abstained": None},)
    epochs = cast("tuple[dict[str, float | None], ...]", raw)
    patched = EvalLog(
        log.version, log.status, log.eval, log.results, log.stats,
        (SceneResult(sample.scene_id, "success", epochs=epochs),),
    )  # fmt: skip
    assert list(metric_evidence(patched)) == ["success"]


def test_empty_log_has_nan_coverage() -> None:
    ev = metric_evidence(make_log({}), ["success"])["success"]
    assert math.isnan(ev.coverage)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_alpha_out_of_range_is_refused(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        metric_evidence(make_log({"s": [1.0]}), alpha=alpha)


@pytest.mark.parametrize("n", [0, -1, True, 2.5])
def test_bad_resample_count_is_refused(n: Any) -> None:
    with pytest.raises(ValueError, match="n_boot"):
        metric_evidence(make_log({"s": [1.0]}), n_boot=n)


# -- compare_logs ------------------------------------------------------------


def paired(a: list[float], b: list[float], **kw: Any) -> Comparison:
    log_a = make_log({f"s{i}": [v] for i, v in enumerate(a)}, policy="A")
    log_b = make_log({f"s{i}": [v] for i, v in enumerate(b)}, policy="B")
    return compare_logs(log_a, log_b, "success", **kw)


def test_consistent_winner_is_named_with_exact_statistics() -> None:
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.9, 0.8, 0.75, 0.88, 0.92]
    b = [x - 0.2 for x in a]
    c = paired(a, b)
    assert c.verdict == "a_better"
    assert c.delta == pytest.approx(0.2)
    assert (c.wins, c.losses, c.ties) == (10, 0, 0)
    assert c.p_sign == pytest.approx(2 / 2**10)
    assert c.p_permutation == pytest.approx(2 / 2**10)
    assert c.ci_low == pytest.approx(0.2) and c.ci_high == pytest.approx(0.2)


def test_b_better_and_not_separated() -> None:
    worse = paired([0.1] * 8, [0.5] * 8)
    assert worse.verdict == "b_better"
    mixed = paired([0.5, 0.6, 0.4, 0.55], [0.55, 0.5, 0.45, 0.5])
    assert mixed.verdict == "not_separated"
    assert mixed.ties == 0


def test_one_hard_scene_cannot_carry_the_sign_count() -> None:
    """A difference of means driven by one scene: the sign count exposes it."""
    a = [0.5] * 9 + [1.0]
    b = [0.51] * 9 + [0.0]
    c = paired(a, b)
    assert c.delta > 0
    assert (c.wins, c.losses) == (1, 9)
    assert c.p_sign == pytest.approx(22 / 1024)


def test_low_coverage_blocks_a_verdict_and_says_why() -> None:
    log_a = make_log({f"s{i}": [0.9, None] for i in range(6)})
    log_b = make_log({f"s{i}": [0.1, 0.1] for i in range(6)})
    c = compare_logs(log_a, log_b, "success")
    assert c.verdict == "insufficient_coverage"
    assert any("6 of 12" in w and "survivors" in w for w in c.warnings)


def test_different_tasks_are_refused() -> None:
    with pytest.raises(ValueError, match="different tasks"):
        compare_logs(make_log({"s": [1.0]}, task="x"), make_log({"s": [1.0]}, task="y"), "success")


def test_unpaired_scenes_and_condition_drift_are_reported() -> None:
    log_a = make_log({"s0": [1.0], "s1": [0.5], "only_a": [0.2]})
    log_b = make_log(
        {"s0": [0.5, 0.5], "s1": [0.2, 0.2], "only_b": [0.1, 0.1]},
        embodiment="other",
        max_steps=99,
    )
    c = compare_logs(log_a, log_b, "success")
    assert c.unpaired_scenes == ("only_a", "only_b")
    assert c.n_paired_scenes == 2
    joined = " ".join(c.warnings)
    for fragment in ("embodiment differs", "max_steps differs", "epochs per scene", "only one"):
        assert fragment in joined


def test_too_few_scenes() -> None:
    none = compare_logs(make_log({"a": [1.0]}), make_log({"b": [1.0]}), "success")
    assert none.verdict == "insufficient_scenes" and none.n_paired_scenes == 0
    assert math.isnan(none.mde) and none.scenes_needed(0.1) is None
    one = paired([0.9], [0.1])
    assert one.verdict == "insufficient_scenes"
    assert math.isnan(one.ci_low) and math.isnan(one.sd_difference)


def test_mde_and_scene_planning_follow_the_normal_approximation() -> None:
    a = [0.5, 0.7, 0.6, 0.8, 0.4, 0.9, 0.3, 0.6]
    b = [0.45, 0.6, 0.62, 0.7, 0.42, 0.8, 0.35, 0.5]
    c = paired(a, b)
    z = 1.959963984540054 + 0.8416212335729143
    assert c.mde == pytest.approx(z * c.sd_difference / math.sqrt(8))
    needed = c.scenes_needed(0.01)
    assert needed == math.ceil((z * c.sd_difference / 0.01) ** 2)
    assert c.scenes_needed(-0.01) == needed
    assert c.scenes_needed(10.0) == 2


def test_identical_differences_need_minimal_scenes() -> None:
    c = paired([0.5] * 4, [0.4] * 4)
    assert c.sd_difference == pytest.approx(0.0, abs=1e-12)
    assert c.scenes_needed(0.05) == 2


@pytest.mark.parametrize(("delta", "power"), [(0.0, 0.8), (float("inf"), 0.8), (0.1, 1.0)])
def test_scene_planning_rejects_bad_inputs(delta: float, power: float) -> None:
    c = paired([0.5, 0.6], [0.4, 0.4])
    with pytest.raises(ValueError):
        c.scenes_needed(delta, power=power)


def test_compare_argument_validation() -> None:
    log = make_log({"s": [1.0]})
    with pytest.raises(ValueError, match="min_coverage"):
        compare_logs(log, log, "success", min_coverage=1.5)
    with pytest.raises(ValueError, match="n_perm"):
        compare_logs(log, log, "success", n_perm=0)


# -- the tests underneath ----------------------------------------------------


def test_sign_test_closed_forms() -> None:
    assert evidence.sign_test(0, 0) == 1.0
    assert evidence.sign_test(5, 5) == 1.0
    assert evidence.sign_test(9, 0) == pytest.approx(2 / 512)
    with pytest.raises(ValueError):
        evidence.sign_test(-1, 2)


def test_permutation_exact_and_sampled_branches_agree() -> None:
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.1, 0.2, size=16)
    exact = evidence.paired_permutation_p(diffs)
    sampled = evidence.paired_permutation_p(
        np.concatenate([diffs, [0.0]]), n_perm=40000, rng=np.random.default_rng(2)
    )
    assert exact == pytest.approx(sampled, abs=0.02)
    assert evidence.paired_permutation_p(np.array([])) == 1.0
    assert 0 < evidence.paired_permutation_p(rng.normal(size=20)) <= 1.0


def test_holm_step_down() -> None:
    adj = evidence.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "b": pytest.approx(0.06), "c": pytest.approx(0.06)}
    assert evidence.holm({}) == {}


# -- the CLI -----------------------------------------------------------------


def two_logs(tmp_path: Path, coverage_gap: bool = False) -> tuple[str, str]:
    a = {f"s{i}": [0.8, None if coverage_gap else 0.8] for i in range(8)}
    b = {f"s{i}": [0.3, 0.3] for i in range(8)}
    return (
        write(make_log(a, policy="A", extra={"steps": 10.0}), tmp_path / "a.json"),
        write(make_log(b, policy="B", extra={"steps": 20.0}), tmp_path / "b.json"),
    )


def test_cli_compare_prints_holm_adjusted_verdicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path_a, path_b = two_logs(tmp_path)
    assert main(["compare", path_a, path_b]) == 0
    out = capsys.readouterr().out
    assert "paired scenes:  8" in out
    assert "A:     A  (" in out
    assert "A better" in out and "B better" in out
    assert "p holm" in out
    assert "warning" not in out


def test_cli_compare_prints_warnings_as_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path_a, path_b = two_logs(tmp_path, coverage_gap=True)
    assert main(["compare", path_a, path_b]) == 0
    out = capsys.readouterr().out
    assert "no verdict: coverage" in out
    assert out.count("warning: a scored 8 of 16 trials") == 1


def test_cli_compare_json_and_scorer_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path_a, path_b = two_logs(tmp_path, coverage_gap=True)
    assert main(["compare", path_a, path_b, "--scorer", "success", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    row = data["comparisons"][0]
    assert row["final_verdict"] == "insufficient_coverage"
    assert row["a"]["coverage"] == pytest.approx(0.5)
    assert "p_holm" in row and "mde" in row


def test_cli_compare_reports_incomparable_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write(make_log({"s": [1.0]}, task="x"), tmp_path / "a.json")
    b = write(make_log({"s": [1.0]}, task="y"), tmp_path / "b.json")
    assert main(["compare", a, b]) == 2
    assert "different tasks" in capsys.readouterr().out
    c = write(make_log({"s": [1.0]}, scorer="other"), tmp_path / "c.json")
    d = write(make_log({"s": [1.0]}), tmp_path / "d.json")
    assert run_compare(c, d) == 2
    assert "share no numeric scorer" in capsys.readouterr().out


def test_cli_compare_prints_na_for_undefined_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write(make_log({"s": [1.0]}), tmp_path / "a.json")
    b = write(make_log({"s": [0.0]}), tmp_path / "b.json")
    assert run_compare(a, b) == 0
    out = capsys.readouterr().out
    assert "n/a" in out and "too few scenes" in out


def test_final_verdict_rereads_separation_against_adjusted_p() -> None:
    c = paired([0.9] * 10, [0.1] * 10)
    assert _final_verdict(c, 0.2) == "not_separated"
    assert _final_verdict(c, 0.001) == "a_better"
    low = paired([0.1] * 10, [0.9] * 10)
    assert _final_verdict(low, 0.001) == "b_better"


def test_policy_label_names_the_configured_model() -> None:
    from inspect_robots._compare import _policy_label

    log = make_log({"s": [1.0]}, policy="agent")
    spec = log.eval
    with_model = EvalLog(
        log.version,
        log.status,
        EvalSpec(
            spec.task,
            spec.policy,
            spec.embodiment,
            spec.created,
            spec.inspect_robots_version,
            policy_config={"model": "m-1"},
        ),
        log.results,
        log.stats,
        log.samples,
    )
    assert _policy_label(with_model) == "agent (m-1)"
    assert _policy_label(log) == "agent"


def test_design_floor_is_known_before_the_data() -> None:
    assert evidence.min_attainable_p(5) == pytest.approx(0.0625)
    assert evidence.min_attainable_p(1) == 1.0
    assert evidence.scenes_to_reach(0.05) == 6
    assert evidence.scenes_to_reach(0.01) == 8
    with pytest.raises(ValueError):
        evidence.min_attainable_p(0)


def test_a_clean_sweep_on_five_scenes_cannot_separate_and_says_so() -> None:
    c = paired([0.9] * 5, [0.1] * 5)
    assert (c.wins, c.losses) == (5, 0)
    assert c.verdict == "not_separated"
    assert any("cannot reach p < 0.05" in w and "at least 6 scenes" in w for w in c.warnings)
    six = paired([0.9] * 6, [0.1] * 6)
    assert six.verdict == "a_better"
    assert not any("cannot reach" in w for w in six.warnings)


# -- anytime-valid sequential test --------------------------------------------


@pytest.mark.parametrize("statistic", ["sign", "mean"])
def test_sequential_test_holds_its_error_under_optional_stopping(statistic: str) -> None:
    """Looking after every one of 100 scenes must still stop falsely at most alpha of the time."""
    rng = np.random.default_rng(0)
    stops = sum(
        evidence.anytime_valid_test(
            np.clip(rng.normal(0.0, 0.2, 100), -1, 1),
            statistic=statistic,  # type: ignore[arg-type]
        ).stopped_at
        is not None
        for _ in range(1500)
    )
    assert stops / 1500 <= 0.05


def test_sequential_test_stops_early_on_a_consistent_gap() -> None:
    by_sign = evidence.anytime_valid_test([0.3] * 60)
    assert by_sign.stopped_at == 8 and by_sign.direction == "a_better"
    assert by_sign.e_value == by_sign.e_values[7] >= 20.0
    by_mean = evidence.anytime_valid_test([-0.3] * 60, statistic="mean")
    assert by_mean.stopped_at == 20 and by_mean.direction == "b_better"


def test_sequential_test_without_a_stop_reports_the_final_e_value() -> None:
    result = evidence.anytime_valid_test([0.1, -0.1, 0.0, 0.1])
    assert result.stopped_at is None and result.direction is None
    assert result.e_value == result.e_values[-1]
    assert evidence.anytime_valid_test([]).e_value == 1.0


def test_sequential_test_argument_validation() -> None:
    with pytest.raises(ValueError, match="statistic"):
        evidence.anytime_valid_test([0.1], statistic="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bound"):
        evidence.anytime_valid_test([0.1], bound=0.0)
    with pytest.raises(ValueError, match="exceeds the bound"):
        evidence.anytime_valid_test([2.0], statistic="mean")


def test_compare_carries_the_sequential_result_only_for_bounded_scores() -> None:
    c = paired([0.9] * 12, [0.1] * 12)
    assert c.stopped_at == 8 and c.e_value >= 20.0
    log_a = make_log({f"s{i}": [30.0] for i in range(6)}, scorer="steps")
    log_b = make_log({f"s{i}": [10.0] for i in range(6)}, scorer="steps")
    unbounded = compare_logs(log_a, log_b, "steps")
    assert math.isnan(unbounded.e_value) and unbounded.stopped_at is None


def test_cli_prints_the_stop_column(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = write(make_log({f"s{i}": [0.9] for i in range(10)}), tmp_path / "a.json")
    b = write(make_log({f"s{i}": [0.1] for i in range(10)}), tmp_path / "b.json")
    assert run_compare(a, b) == 0
    out = capsys.readouterr().out
    assert "stop@" in out and "8/10" in out
    from inspect_robots._compare import _fmt_e

    assert _fmt_e(21.5) == "21.5"
    assert _fmt_e(123456.0) == "1.2e+05"
    assert _fmt_e(float("nan")) == "n/a"


# -- review regressions: per-scorer coverage, reducers, direction, scale, strict JSON ---------


def _raw_log(
    scenes: Mapping[str, tuple[Sequence[dict[str, float | None]], dict[str, float | None]]],
) -> EvalLog:
    """A log from explicit ``(epochs, reduced)`` per scene, for cases ``make_log`` cannot build."""
    base = make_log({"x": [1.0]})
    samples = tuple(
        SceneResult(scene_id=sid, status="success", epochs=tuple(epochs), reduced=reduced)
        for sid, (epochs, reduced) in scenes.items()
    )
    return EvalLog(base.version, base.status, base.eval, base.results, base.stats, samples)


def test_coverage_counts_only_trials_this_scorer_scored() -> None:
    # Ten trials per scene; "success" has a value on one of them, "other" on all ten.
    def side(value: float) -> EvalLog:
        epochs: list[dict[str, float | None]] = [{"success": value, "other": 1.0}]
        epochs += [{"success": None, "other": 1.0}] * 9
        return _raw_log({f"s{i}": (epochs, {}) for i in range(8)})

    a, b = side(1.0), side(0.0)
    ev = metric_evidence(a)

    assert (ev["success"].scored_trials, ev["success"].attempted_trials) == (8, 80)
    assert ev["other"].coverage == 1.0
    c = compare_logs(a, b, "success")
    assert c.verdict == "insufficient_coverage"
    assert any("a scored 8 of 80 trials" in w for w in c.warnings)


def test_scene_scores_follow_the_saved_reducer() -> None:
    # Under a max reducer A solves every scene once in ten tries; B scores 0.6 every time.
    a_epochs: list[dict[str, float | None]] = [
        {"success": 1.0 if k == 0 else 0.0} for k in range(10)
    ]
    b_epochs: list[dict[str, float | None]] = [{"success": 0.6}] * 10
    a = _raw_log({f"s{i}": (a_epochs, {"success": 1.0}) for i in range(8)})
    b = _raw_log({f"s{i}": (b_epochs, {"success": 0.6}) for i in range(8)})

    assert metric_evidence(a)["success"].mean == pytest.approx(1.0)
    c = compare_logs(a, b, "success")
    assert c.delta == pytest.approx(0.4)
    assert c.verdict == "a_better"


def test_an_abstained_reduced_value_drops_the_scene() -> None:
    log = _raw_log(
        {
            "s0": ([{"success": 1.0}], {"success": 1.0}),
            "s1": ([{"success": 0.0}], {"success": None}),
            "s2": ([{"success": 0.0}], {}),
        }
    )

    ev = metric_evidence(log)["success"]
    assert ev.n_scenes == 2  # s1 abstained at the reducer; s2 falls back to its epochs
    assert ev.mean == pytest.approx(0.5)


def test_lower_is_better_scorers_name_the_smaller_value_the_winner() -> None:
    a = make_log({f"s{i}": [0.9] for i in range(8)}, scorer="min_distance_to_goal")
    b = make_log({f"s{i}": [0.1] for i in range(8)}, scorer="min_distance_to_goal")

    builtin = compare_logs(a, b, "min_distance_to_goal")
    assert builtin.lower_is_better
    assert builtin.delta == pytest.approx(0.8)
    assert (builtin.wins, builtin.losses) == (0, 8)
    assert builtin.verdict == "b_better"
    assert _final_verdict(builtin, 0.001) == "b_better"
    assert _final_verdict(replace(builtin, delta=-0.8), 0.001) == "a_better"

    custom_a = make_log({f"s{i}": [0.9] for i in range(8)}, scorer="time_s")
    custom_b = make_log({f"s{i}": [0.1] for i in range(8)}, scorer="time_s")
    assert compare_logs(custom_a, custom_b, "time_s").verdict == "a_better"
    assert compare_logs(custom_a, custom_b, "time_s", lower_is_better=True).verdict == "b_better"


def test_cli_lower_is_better_flag_and_coverage_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write(make_log({f"s{i}": [0.9] for i in range(8)}, scorer="time_s"), tmp_path / "a.json")
    b = write(make_log({f"s{i}": [0.1] for i in range(8)}, scorer="time_s"), tmp_path / "b.json")

    assert main(["compare", a, b, "--lower-is-better", "time_s"]) == 0
    out = capsys.readouterr().out
    assert "time_s (lower)" in out
    assert "B better" in out
    assert "1.00/1.00" in out


def test_large_scene_counts_do_not_overflow() -> None:
    assert evidence.min_attainable_p(1100) == 0.0  # 2 / 2**1100 underflows instead of raising
    n = 1100
    a = make_log({f"s{i}": [1.0 if i % 3 else 0.0] for i in range(n)})
    b = make_log({f"s{i}": [0.0] for i in range(n)})

    c = compare_logs(a, b, "success", n_boot=50, n_perm=200)
    assert c.n_paired_scenes == n
    assert c.verdict == "a_better"


def test_json_output_is_strict_when_statistics_are_undefined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write(make_log({"s0": [1.0]}), tmp_path / "a.json")
    b = write(make_log({"s0": [0.0]}), tmp_path / "b.json")

    assert main(["compare", a, b, "--json"]) == 0
    out = capsys.readouterr().out

    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON token {token}")

    payload = json.loads(out, parse_constant=reject)
    row = payload["comparisons"][0]
    assert row["ci_low"] is None
    assert row["mde"] is None
    assert row["unpaired_scenes"] == []
