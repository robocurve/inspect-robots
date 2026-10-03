"""``inspect-robots compare``: two saved runs, scene-paired, with the evidence behind the verdict.

Rendering and multiplicity handling live here so ``cli.py`` only parses and dispatches. The
statistics themselves are :mod:`inspect_robots.evidence`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from inspect_robots.evidence import (
    DEFAULT_MIN_COVERAGE,
    Comparison,
    compare_logs,
    holm,
    metric_evidence,
)
from inspect_robots.log import EvalLog, read_eval_log

_VERDICT_TEXT = {
    "a_better": "A better",
    "b_better": "B better",
    "not_separated": "not separated",
    "insufficient_coverage": "no verdict: coverage",
    "insufficient_scenes": "no verdict: too few scenes",
}


def run_compare(
    path_a: str,
    path_b: str,
    *,
    scorers: Sequence[str] | None = None,
    alpha: float = 0.05,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    seed: int = 0,
    as_json: bool = False,
    lower_is_better: Sequence[str] = (),
) -> int:
    """Compare two logs and print the result; return the process exit code.

    Every scorer common to both logs is compared unless ``scorers`` names some. With more than
    one scorer the verdicts are read against Holm-adjusted p-values, so scanning many scorers
    does not manufacture a winner. Scorers named in ``lower_is_better`` are read with smaller
    values winning, on top of the built-ins :data:`~inspect_robots.evidence.LOWER_IS_BETTER`
    already marks. Exit code 0 on success, 2 when the logs cannot be compared.
    """
    log_a = read_eval_log(path_a)
    log_b = read_eval_log(path_b)
    names = list(scorers) if scorers else _common_scorers(log_a, log_b)
    if not names:
        print("error: the two logs share no numeric scorer")
        return 2
    try:
        results = [
            compare_logs(
                log_a,
                log_b,
                name,
                alpha=alpha,
                min_coverage=min_coverage,
                seed=seed,
                lower_is_better=True if name in lower_is_better else None,
            )
            for name in names
        ]
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    adjusted = holm({c.scorer: c.p_permutation for c in results})
    if as_json:
        payload = _finite_or_null(_as_json(path_a, path_b, results, adjusted))
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    _print_report(path_a, path_b, log_a, log_b, results, adjusted)
    return 0


def _common_scorers(log_a: EvalLog, log_b: EvalLog) -> list[str]:
    """Scorers with a numeric value in both logs, in log A's order."""
    names_b = set(metric_evidence(log_b, n_boot=1))
    return [name for name in metric_evidence(log_a, n_boot=1) if name in names_b]


def _final_verdict(comparison: Comparison, p_adjusted: float) -> str:
    """The verdict with the separation decision re-read against the adjusted p-value."""
    if comparison.verdict in ("insufficient_coverage", "insufficient_scenes"):
        return comparison.verdict
    if p_adjusted < comparison.alpha:
        a_ahead = (
            (comparison.delta < 0.0) if comparison.lower_is_better else (comparison.delta > 0.0)
        )
        return "a_better" if a_ahead else "b_better"
    return "not_separated"


def _policy_label(log: EvalLog) -> str:
    """The policy name, with the configured model when the policy records one (LLM agents)."""
    model = log.eval.policy_config.get("model")
    return f"{log.eval.policy} ({model})" if isinstance(model, str) and model else log.eval.policy


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_e(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}" if value < 1000 else f"{value:.1e}"


def _fmt_stop(c: Comparison) -> str:
    return f"{c.stopped_at}/{c.n_paired_scenes}" if c.stopped_at is not None else "-"


def _fmt_cov(c: Comparison) -> str:
    return f"{_fmt(c.a.coverage, 2)}/{_fmt(c.b.coverage, 2)}"


def _fmt_p(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.4f}"


def _print_report(
    path_a: str,
    path_b: str,
    log_a: EvalLog,
    log_b: EvalLog,
    results: Sequence[Comparison],
    adjusted: dict[str, float],
) -> None:
    first = results[0]
    print(f"task:  {log_a.eval.task}")
    print(f"A:     {_policy_label(log_a)}  ({path_a})")
    print(f"B:     {_policy_label(log_b)}  ({path_b})")
    print(f"trials attempted:  A {first.a.attempted_trials}   B {first.b.attempted_trials}")
    print(f"paired scenes:  {first.n_paired_scenes}")
    level = round((1 - first.alpha) * 100)
    print()
    header = (
        f"{'scorer':24s} {'A mean':>7s} {'B mean':>7s} {'A-B':>8s} "
        f"{f'{level}% CI':>18s} {'cov A/B':>9s} {'W-L-T':>9s} {'p':>8s} {'p holm':>8s} {'MDE':>7s} "
        f"{'e':>7s} {'stop@':>7s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for c in results:
        ci = f"[{_fmt(c.ci_low)}, {_fmt(c.ci_high)}]"
        wlt = f"{c.wins}-{c.losses}-{c.ties}"
        verdict = _VERDICT_TEXT[_final_verdict(c, adjusted[c.scorer])]
        name = f"{c.scorer} (lower)" if c.lower_is_better else c.scorer
        print(
            f"{name:24s} {_fmt(c.a.mean):>7s} {_fmt(c.b.mean):>7s} {_fmt(c.delta):>8s} "
            f"{ci:>18s} {_fmt_cov(c):>9s} {wlt:>9s} {_fmt_p(c.p_permutation):>8s} "
            f"{_fmt_p(adjusted[c.scorer]):>8s} {_fmt(c.mde):>7s} {_fmt_e(c.e_value):>7s} "
            f"{_fmt_stop(c):>7s}  {verdict}"
        )
    print()
    print(
        "Intervals and tests resample and permute whole scenes; W-L-T counts scenes where A did "
        "better, worse, or level, and (lower) marks a scorer where smaller wins. cov is each "
        "side's share of attempted trials that this scorer scored. "
        "MDE is the difference this design had an 80% chance to detect. e is the anytime-valid "
        "e-value and stop@ the scene at which checking after every scene could have stopped."
    )
    warnings = dict.fromkeys(w for c in results for w in c.warnings)
    for warning in warnings:
        print(f"warning: {warning}")


def _as_json(
    path_a: str, path_b: str, results: Sequence[Comparison], adjusted: dict[str, float]
) -> dict[str, Any]:
    rows = []
    for c in results:
        row = asdict(c)
        row["mde"] = c.mde
        row["a"]["coverage"] = c.a.coverage
        row["b"]["coverage"] = c.b.coverage
        row["p_holm"] = adjusted[c.scorer]
        row["final_verdict"] = _final_verdict(c, adjusted[c.scorer])
        rows.append(row)
    return {"a": path_a, "b": path_b, "comparisons": rows}


def _finite_or_null(value: Any) -> Any:
    """``value`` with every non-finite float replaced by ``None``, so the JSON is strict."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_or_null(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_null(v) for v in value]
    return value
