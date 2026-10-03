"""Uncertainty and paired comparison for saved eval logs.

A run's ``results.metrics`` is a bare mean per scorer. On a robot that mean usually rests on a few
dozen trials drawn from a handful of scenes, so on its own it cannot say whether a second run that
scored higher is actually better. This module reads a finished :class:`~inspect_robots.log.EvalLog`
and answers two questions without touching the log format:

- :func:`metric_evidence`: how precisely is each metric known, and how many of the trials the run
  set out to score actually produced a score?
- :func:`compare_logs`: given two runs of the same task, is one better, by how much, and how sure?
- :func:`anytime_valid_test`: may a head-to-head stop now? Safe to ask after every scene.

Three rules are built in, because each one was learned from a real benchmark going wrong:

**Scenes are the unit of evidence, not trials.** Epochs of one scene share a world, so they are
not independent draws. Every interval resamples whole scenes, and every test permutes whole scenes.
Adding epochs narrows nothing that more scenes would not narrow further.

**Coverage travels with every number.** A trial that errors, or that a scorer abstains on, is
recorded but never scored, and the run-level mean averages whatever survived. When the lost trials
are not a random subset (a rate limit, a spend cap, a crash that hits long rollouts first) the
survivor mean is biased in a direction the log cannot reveal. Every summary here reports, per
scorer, scored against attempted trials, and a comparison refuses to call a winner when either side
falls below ``min_coverage``.

**Respect the saved metric.** Each scene contributes its saved reduced score
(``SceneResult.reduced``, from the task's epoch reducer) and scenes weigh equally, exactly as
``results.metrics`` was computed, so a comparison never contradicts the metric the task reports.

**Pair by scene.** Two runs of the same task see the same scenes, so the comparison is made scene
by scene rather than mean against mean. A single hard scene can dominate a difference of means
while the two policies disagree on almost nothing else; the paired statistics, the sign count in
particular, show that directly.

NumPy only. Results are deterministic for a given ``seed``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Literal, TypeGuard

import numpy as np
import numpy.typing as npt

from inspect_robots.log import EvalLog, SceneResult

__all__ = [
    "Comparison",
    "MetricEvidence",
    "SequentialTest",
    "anytime_valid_test",
    "compare_logs",
    "holm",
    "metric_evidence",
    "min_attainable_p",
    "paired_permutation_p",
    "scenes_to_reach",
    "sign_test",
]

#: Bootstrap resamples used when the caller does not choose.
DEFAULT_N_BOOT = 4000

#: Coverage below which a comparison is reported but no winner is named.
DEFAULT_MIN_COVERAGE = 0.95

#: Cap on the fraction of wealth staked per scene in :func:`anytime_valid_test`. Any value below 1
#: keeps the test valid (wealth stays positive); 0.75 is the upper end of what Waudby-Smith and
#: Ramdas recommend, trading a little robustness for faster stopping on consistent gaps.
MAX_STAKE = 0.75

#: Largest scene count for which the paired permutation test enumerates every sign pattern
#: exactly (2**16 = 65,536 patterns). Above it the test samples ``n_perm`` patterns.
EXACT_PERMUTATION_MAX_SCENES = 16

#: Built-in scorers for which a smaller value is the better outcome. ``compare_logs`` reads these
#: in that direction unless told otherwise; any other scorer is read as higher-is-better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"min_distance_to_goal"})

Verdict = Literal[
    "a_better",
    "b_better",
    "not_separated",
    "insufficient_coverage",
    "insufficient_scenes",
]


@dataclass(frozen=True)
class MetricEvidence:
    """One scorer's mean with a scene-resampled interval and the trial accounting behind it.

    ``mean`` is the mean over scenes of each scene's saved reduced score, the quantity
    ``results.metrics`` reports. ``ci_low``/``ci_high`` resample whole scenes and are ``nan`` when
    fewer than two scenes produced a score, since between-scene variation is then unidentifiable.
    ``coverage`` is ``scored_trials / attempted_trials`` for this scorer: a trial counts as scored
    only when it carries a finite value for it, so errors and abstentions both lower it.
    """

    scorer: str
    mean: float
    ci_low: float
    ci_high: float
    n_scenes: int
    scored_trials: int
    attempted_trials: int
    alpha: float

    @property
    def coverage(self) -> float:
        """Fraction of attempted trials scored by this scorer; ``nan`` for an empty run."""
        if self.attempted_trials == 0:
            return float("nan")
        return self.scored_trials / self.attempted_trials


@dataclass(frozen=True)
class Comparison:
    """Log ``a`` against log ``b`` on one scorer, paired scene by scene.

    ``delta`` is ``mean(a) - mean(b)`` over the paired scenes' reduced scores, each scene weighted
    equally, with a scene-resampled interval; it keeps that sign whatever the scorer's direction.
    ``wins``/``losses``/``ties`` count scenes where ``a`` did better, worse, or level with ``b``,
    reading "better" as lower when ``lower_is_better``. ``p_sign`` is the exact two-sided sign
    test on that count and ``p_permutation`` the two-sided paired sign-flip test on the scene
    differences; ``verdict`` reads the latter against ``alpha``. ``mde`` is the difference this
    comparison had an 80% chance of detecting at ``alpha``, and :meth:`scenes_needed` turns a
    target difference into a scene count, both from a normal approximation to the paired
    differences.
    """

    scorer: str
    delta: float
    ci_low: float
    ci_high: float
    n_paired_scenes: int
    wins: int
    losses: int
    ties: int
    p_sign: float
    p_permutation: float
    alpha: float
    verdict: Verdict
    a: MetricEvidence
    b: MetricEvidence
    sd_difference: float
    unpaired_scenes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)
    #: The anytime-valid e-value after the last paired scene, in log A's scene order, and the
    #: first scene count at which it crossed ``1 / alpha``. ``nan`` and ``None`` when a per-scene
    #: mean falls outside [0, 1], since the test needs bounded scores. See
    #: :func:`anytime_valid_test`.
    e_value: float = float("nan")
    stopped_at: int | None = None
    #: Whether a smaller value of this scorer is the better outcome. Verdicts, win counts and the
    #: sequential test all read the difference in this direction.
    lower_is_better: bool = False

    @property
    def mde(self) -> float:
        """Smallest difference with 80% power at this comparison's ``alpha`` and scene count."""
        if self.n_paired_scenes < 2 or not math.isfinite(self.sd_difference):
            return float("nan")
        return _z_sum(self.alpha, 0.8) * self.sd_difference / math.sqrt(self.n_paired_scenes)

    def scenes_needed(self, delta: float, *, power: float = 0.8) -> int | None:
        """Paired scenes needed to detect ``delta`` with ``power``, or ``None`` if unestimable.

        Uses this comparison's observed spread of per-scene differences as the planning value, so
        it is a pilot-based estimate: read it as an order of magnitude, not a guarantee.
        """
        if not 0.0 < power < 1.0:
            raise ValueError(f"power must be in (0, 1), got {power!r}")
        if delta == 0.0 or not math.isfinite(delta):
            raise ValueError(f"delta must be finite and non-zero, got {delta!r}")
        if not math.isfinite(self.sd_difference) or self.n_paired_scenes < 2:
            return None
        if self.sd_difference == 0.0:
            return 2
        n = (_z_sum(self.alpha, power) * self.sd_difference / abs(delta)) ** 2
        return max(2, math.ceil(n))


# -- per-log accounting ------------------------------------------------------


def _scene_scores(log: EvalLog, scorer: str) -> dict[str, float]:
    """Each scene's score for ``scorer``, scenes without one omitted.

    The saved reduced value when it is finite, so the task's epoch reducer (``max``,
    ``pass_at_k``, ...) is respected. Logs written without one fall back to the mean of the
    scene's finite per-trial values.
    """
    out: dict[str, float] = {}
    for sample in log.samples:
        value = _scene_value(sample, scorer)
        if value is not None:
            out[sample.scene_id] = value
    return out


def _scene_value(sample: SceneResult, scorer: str) -> float | None:
    reduced = sample.reduced.get(scorer)
    if _is_number(reduced):
        return float(reduced)
    if scorer in sample.reduced:
        return None  # the reducer ran and abstained (or failed) for this scene
    raw = (epoch.get(scorer) for epoch in sample.epochs)
    values = [float(value) for value in raw if _is_number(value)]
    return float(np.mean(values)) if values else None


def _trial_accounting(log: EvalLog, scorer: str) -> tuple[int, int]:
    """``(scored, attempted)`` trials for ``scorer``: scored means a finite value for it."""
    attempted = sum(len(sample.epochs) for sample in log.samples)
    scored = sum(
        1 for sample in log.samples for epoch in sample.epochs if _is_number(epoch.get(scorer))
    )
    return scored, attempted


def _scorers(log: EvalLog) -> list[str]:
    """Every scorer name that appears in any epoch, in first-seen order."""
    seen: dict[str, None] = {}
    for sample in log.samples:
        for epoch in sample.epochs:
            for name, value in epoch.items():
                if _is_number(value):
                    seen.setdefault(name, None)
    return list(seen)


def metric_evidence(
    log: EvalLog,
    scorers: Sequence[str] | None = None,
    *,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, MetricEvidence]:
    """Each scorer's mean with a scene-resampled interval and that scorer's trial coverage.

    Args:
        log: A finished eval log.
        scorers: Which scorers to summarize; every numeric scorer in the log by default.
        alpha: Two-sided level of the interval (0.05 gives a 95% interval).
        n_boot: Scene-level bootstrap resamples.
        seed: Seed for the bootstrap, so a repeated call returns the same interval.

    Returns:
        ``{scorer: MetricEvidence}`` in the order requested. A scorer with no finite value in the
        log is reported with ``nan`` statistics and ``n_scenes == 0`` rather than omitted.
    """
    _check_alpha(alpha)
    _check_n(n_boot, "n_boot")
    names = list(scorers) if scorers is not None else _scorers(log)
    rng = np.random.default_rng(seed)
    out: dict[str, MetricEvidence] = {}
    for name in names:
        scored, attempted = _trial_accounting(log, name)
        values = np.array(list(_scene_scores(log, name).values()), dtype=np.float64)
        if values.size == 0:
            nan = float("nan")
            out[name] = MetricEvidence(name, nan, nan, nan, 0, scored, attempted, alpha)
            continue
        low, high = _scene_interval(values, alpha, n_boot, rng)
        out[name] = MetricEvidence(
            name, float(values.mean()), low, high, int(values.size), scored, attempted, alpha
        )
    return out


def _scene_interval(
    values: npt.NDArray[np.float64], alpha: float, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile interval of the scene-weighted mean, resampling whole scenes."""
    if values.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    low, high = np.quantile(values[idx].mean(axis=1), [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


# -- paired comparison -------------------------------------------------------


def compare_logs(
    log_a: EvalLog,
    log_b: EvalLog,
    scorer: str,
    *,
    alpha: float = 0.05,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = 20000,
    seed: int = 0,
    lower_is_better: bool | None = None,
) -> Comparison:
    """Compare two runs of the same task on one scorer, pairing their scenes.

    Scenes are matched by ``scene_id``; a scene scored in only one log is listed in
    ``unpaired_scenes`` and left out of every paired statistic. The verdict names a winner only
    when both logs clear ``min_coverage`` and at least two scenes pair. ``lower_is_better`` sets
    which direction wins; ``None`` reads the built-ins in :data:`LOWER_IS_BETTER` as lower is
    better and every other scorer as higher is better.

    Raises:
        ValueError: If the two logs are of different tasks, or an argument is out of range.
    """
    _check_alpha(alpha)
    _check_n(n_boot, "n_boot")
    _check_n(n_perm, "n_perm")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage!r}")
    if log_a.eval.task != log_b.eval.task:
        raise ValueError(
            f"the logs are of different tasks ({log_a.eval.task!r} and {log_b.eval.task!r}); "
            "a paired comparison needs the same scenes on both sides"
        )

    lower = scorer in LOWER_IS_BETTER if lower_is_better is None else lower_is_better
    rng = np.random.default_rng(seed)
    ev_a = metric_evidence(log_a, [scorer], alpha=alpha, n_boot=n_boot, seed=seed)[scorer]
    ev_b = metric_evidence(log_b, [scorer], alpha=alpha, n_boot=n_boot, seed=seed)[scorer]
    scenes_a = _scene_scores(log_a, scorer)
    scenes_b = _scene_scores(log_b, scorer)
    paired = [s for s in scenes_a if s in scenes_b]
    unpaired = tuple(sorted(set(scenes_a) ^ set(scenes_b)))
    diffs = np.array([scenes_a[s] - scenes_b[s] for s in paired], dtype=np.float64)
    # Oriented so that positive always means "a did better"; every test and count reads these.
    gains = -diffs if lower else diffs

    warnings = list(_comparability_warnings(log_a, log_b))
    if unpaired:
        warnings.append(
            f"{len(unpaired)} scene(s) scored in only one log were left out of the pairing"
        )
    low_coverage = [
        f"{label} scored {ev.scored_trials} of {ev.attempted_trials} trials"
        for label, ev in (("a", ev_a), ("b", ev_b))
        if ev.attempted_trials and ev.coverage < min_coverage
    ]
    warnings.extend(
        f"{msg}, below the {min_coverage:.0%} coverage bar; its mean is over survivors"
        for msg in low_coverage
    )

    n = len(diffs)
    if n >= 1 and min_attainable_p(n) >= alpha:
        warnings.append(
            f"with {n} paired scene(s) the exact test cannot reach p < {alpha:g} whatever the "
            f"data (smallest attainable p is {min_attainable_p(n):.4f}); at least "
            f"{scenes_to_reach(alpha)} scenes are needed before any pair can separate, and the "
            "percentile interval is optimistic at this size"
        )
    wins = int(np.sum(gains > 0.0))
    losses = int(np.sum(gains < 0.0))
    ties = n - wins - losses
    if n == 0:
        nan = float("nan")
        return Comparison(
            scorer, nan, nan, nan, 0, 0, 0, 0, 1.0, 1.0, alpha,
            "insufficient_scenes", ev_a, ev_b, nan, unpaired, tuple(warnings),
            lower_is_better=lower,
        )  # fmt: skip

    delta = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = diffs[idx].mean(axis=1)
        ci_low, ci_high = (float(q) for q in np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0]))
    else:
        ci_low = ci_high = float("nan")
    p_sign = sign_test(wins, losses)
    p_perm = paired_permutation_p(gains, n_perm=n_perm, rng=rng)
    bounded = all(0.0 <= side[s] <= 1.0 for side in (scenes_a, scenes_b) for s in paired)
    sequential = anytime_valid_test(gains, alpha=alpha) if bounded else None

    verdict: Verdict
    if low_coverage:
        verdict = "insufficient_coverage"
    elif n < 2:
        verdict = "insufficient_scenes"
    elif p_perm < alpha:
        verdict = "a_better" if float(gains.mean()) > 0.0 else "b_better"
    else:
        verdict = "not_separated"
    return Comparison(
        scorer=scorer,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired_scenes=n,
        wins=wins,
        losses=losses,
        ties=ties,
        p_sign=p_sign,
        p_permutation=p_perm,
        alpha=alpha,
        verdict=verdict,
        a=ev_a,
        b=ev_b,
        sd_difference=sd,
        unpaired_scenes=unpaired,
        warnings=tuple(warnings),
        e_value=sequential.e_value if sequential else float("nan"),
        stopped_at=sequential.stopped_at if sequential else None,
        lower_is_better=lower,
    )


def _comparability_warnings(log_a: EvalLog, log_b: EvalLog) -> list[str]:
    """Differences in run conditions that a paired comparison silently depends on."""
    out: list[str] = []
    for label, left, right in (
        ("embodiment", log_a.eval.embodiment, log_b.eval.embodiment),
        ("max_steps", log_a.eval.max_steps, log_b.eval.max_steps),
        ("environment_id", log_a.eval.environment_id, log_b.eval.environment_id),
        ("environment_revision", log_a.eval.environment_revision, log_b.eval.environment_revision),
    ):
        if left != right:
            out.append(f"{label} differs between the runs ({left!r} vs {right!r})")
    epochs_a = {len(s.epochs) for s in log_a.samples}
    epochs_b = {len(s.epochs) for s in log_b.samples}
    if epochs_a != epochs_b:
        out.append(
            f"epochs per scene differ ({sorted(epochs_a)} vs {sorted(epochs_b)}); "
            "scenes are still weighted equally"
        )
    return out


# -- tests and corrections ---------------------------------------------------


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test at p = 0.5; ties must already be excluded."""
    if wins < 0 or losses < 0:
        raise ValueError(f"counts must be non-negative, got {wins!r} and {losses!r}")
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_permutation_p(
    diffs: npt.NDArray[np.float64], *, n_perm: int = 20000, rng: np.random.Generator | None = None
) -> float:
    """Two-sided paired sign-flip test on the mean of ``diffs``.

    With at most :data:`EXACT_PERMUTATION_MAX_SCENES` differences every sign pattern is
    enumerated and ``p`` is the exact enumerated fraction, which already counts the observed
    pattern and so needs no add-one. Above that, ``n_perm`` patterns are sampled and the
    add-one correction keeps the sampled test valid.
    """
    values = np.asarray(diffs, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 1.0
    observed = abs(float(values.mean()))
    tol = 1e-12 * max(1.0, observed)
    if n <= EXACT_PERMUTATION_MAX_SCENES:
        patterns = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        signs = 1.0 - 2.0 * patterns
        stats = np.abs((signs * values).mean(axis=1))
        return float(np.mean(stats >= observed - tol))
    gen = rng if rng is not None else np.random.default_rng(0)
    signs = gen.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    stats = np.abs((signs * values).mean(axis=1))
    hits = int(np.sum(stats >= observed - tol))
    return (hits + 1) / (n_perm + 1)


def min_attainable_p(n_scenes: int) -> float:
    """Smallest two-sided p the paired sign-flip or sign test can return on ``n_scenes`` scenes.

    Every scene agreeing in sign is the most extreme outcome, and it and its mirror are two of the
    ``2**n`` equally likely sign patterns under the null, so the floor is ``2 / 2**n``. It is a
    property of the design, known before any data is collected: at 5 scenes it is 0.0625, above
    0.05, so a 5-scene comparison cannot separate two policies however far apart they are.
    """
    if n_scenes < 1:
        raise ValueError(f"n_scenes must be >= 1, got {n_scenes!r}")
    return min(1.0, math.ldexp(1.0, 1 - n_scenes))


def scenes_to_reach(alpha: float) -> int:
    """Fewest paired scenes whose :func:`min_attainable_p` is below ``alpha``."""
    _check_alpha(alpha)
    n = 1
    while min_attainable_p(n) >= alpha:
        n += 1
    return n


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed as given; controls family-wise error at alpha."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


# -- anytime-valid sequential testing ----------------------------------------


@dataclass(frozen=True)
class SequentialTest:
    """A betting e-process over paired scene differences, read after every scene.

    ``e_values[k]`` is the evidence against "no difference" after ``k + 1`` scenes. By Ville's
    inequality the chance it *ever* reaches ``1 / alpha`` under the null is at most ``alpha``, so
    the test may be consulted after every scene and stopped the first time it crosses, and the
    error guarantee still holds. ``stopped_at`` is that first scene count (1-based), or ``None``
    if it never crossed. ``direction`` says which side the evidence favours at the stop.
    """

    e_values: tuple[float, ...]
    stopped_at: int | None
    direction: Literal["a_better", "b_better"] | None
    alpha: float

    @property
    def e_value(self) -> float:
        """The e-value at the stop, or after the last scene if it never stopped."""
        if not self.e_values:
            return 1.0
        index = (self.stopped_at if self.stopped_at is not None else len(self.e_values)) - 1
        return self.e_values[index]


def anytime_valid_test(
    diffs: Sequence[float] | npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
    statistic: Literal["sign", "mean"] = "sign",
    bound: float = 1.0,
) -> SequentialTest:
    """Anytime-valid two-sided test that the mean paired scene difference is zero.

    With ``statistic="sign"`` (the default) each scene contributes only who won it, +1, -1, or 0
    for a tie, and the null is that a win and a loss are equally likely. With ``"mean"`` the
    difference itself is bet on, scaled by ``bound``, and the null is a zero mean difference.
    The sign form stops sooner whenever differences are small against ``bound`` (a consistent
    0.3 gap on [0, 1] scores stops in 8 scenes by sign against 20 by mean) and, as with the
    scene win counts, one extreme scene cannot carry it.

    Two test martingales bet on the next scene, one on ``a`` being better and one on
    ``b``, each with a stake chosen from the scenes already seen (the predictable plug-in of
    Waudby-Smith and Ramdas, capped at :data:`MAX_STAKE`), and their average is the
    reported e-process. Differences must lie in ``[-bound, bound]``; for scores in [0, 1] the
    default ``bound`` of 1 always holds for the mean form.

    The fixed-design tests in :func:`compare_logs` are valid only if the number of scenes was fixed
    in advance. An operator who looks after every scene and stops once the result looks good
    breaks that assumption and inflates the error. This test is built to be read that way.

    Raises:
        ValueError: If ``alpha`` or ``bound`` is out of range, or a difference exceeds ``bound``.
    """
    _check_alpha(alpha)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"bound must be finite and positive, got {bound!r}")
    if statistic not in ("sign", "mean"):
        raise ValueError(f"statistic must be 'sign' or 'mean', got {statistic!r}")
    raw = np.asarray(diffs, dtype=np.float64)
    if statistic == "sign":
        x = np.sign(raw)
    else:
        x = raw / bound
        if x.size and float(np.max(np.abs(x))) > 1.0 + 1e-12:
            raise ValueError(
                f"a difference exceeds the bound {bound!r}; the test needs bounded data"
            )
    threshold = 1.0 / alpha
    up = down = 1.0
    total = 0.0
    total_sq = 0.0
    values: list[float] = []
    stopped: int | None = None
    direction: Literal["a_better", "b_better"] | None = None
    for count, xi in enumerate(x.tolist()):
        # Predictable stakes from the scenes before this one, shrunk toward a zero mean and a
        # variance of 0.25 by one pseudo-observation so the first bet is small.
        mean = total / (count + 1)
        var = (0.25 + total_sq - count * mean * mean) / (count + 1)
        denom = max(var + mean * mean, 1e-12)
        stake_up = min(max(mean / denom, 0.0), MAX_STAKE)
        stake_down = min(max(-mean / denom, 0.0), MAX_STAKE)
        up *= 1.0 + stake_up * xi
        down *= 1.0 - stake_down * xi
        e_value = 0.5 * (up + down)
        values.append(e_value)
        total += xi
        total_sq += xi * xi
        if stopped is None and e_value >= threshold:
            stopped = count + 1
            direction = "a_better" if up >= down else "b_better"
    return SequentialTest(tuple(values), stopped, direction, alpha)


# -- helpers -----------------------------------------------------------------


def _z_sum(alpha: float, power: float) -> float:
    """``z(1 - alpha/2) + z(power)`` for a two-sided test."""
    unit = NormalDist()
    return unit.inv_cdf(1.0 - alpha / 2.0) + unit.inv_cdf(power)


def _is_number(value: object) -> TypeGuard[float]:
    """A finite int or float that is not a bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_n(n: int, name: str) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {n!r}")
