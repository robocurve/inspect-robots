# 0086: Uncertainty and paired comparison for saved runs

Follow-up to #440 (a run where every scene errored can still report `status: success`), closed by
#450, which warns when no scene completed cleanly. That fix covers one symptom. This plan addresses
what the symptom exposed: a finished run carries a mean per scorer and nothing that says how far to
trust it, and there is no supported way to ask whether one run beat another.

## Problem

`EvalResults.metrics` is `dict[str, float]`, a bare mean per scorer. Three things a reader needs
are missing, and each has already produced a wrong conclusion in practice.

1. **No uncertainty.** A real-robot run is usually 5 to 25 scenes at a few epochs each. Two
   policies scoring 0.42 and 0.25 may or may not differ, and nothing in the log says which. Inspect
   AI ships `stderr()` (with a `cluster` argument) and `bootstrap_stderr()` as metrics; Inspect
   Robots has no equivalent.
2. **No coverage next to the number.** Errored trials are recorded but never scored, and the mean
   averages whatever survived. `errored_trials` exists, but the metric does not carry it. In a
   27-cell LLM benchmark run against rate-limited providers, 9 cells were re-run at full coverage
   and the survivor means had been off by -0.092 to +0.087 (median absolute 0.060) on a 0-to-1
   score, in both directions, which is the size of the between-model differences being measured.
3. **No comparison.** There is no command that takes two logs of the same task and says which is
   better. People compare `metrics` by eye, which compares means of different trial sets, ignores
   that both runs saw the same scenes, and cannot tell a consistent win from one hard scene. On a
   published three-way tie among bimanual policies, a scene-paired sign test separated two of them
   (18 of 20 anchors, p = 0.0004) because one anchor carried 87% of the squared difference.

A fourth problem is invisible until someone does the arithmetic: **the scene count sets a floor on
significance that no amount of data can cross.** A paired test over `n` scenes cannot return a
two-sided p below `2 / 2**n`. At 5 scenes that is 0.0625, so a 5-scene comparison cannot separate
two policies at 0.05 even if one wins every scene. Adding epochs does not move the floor. Only
scenes do.

## Design

A new public module `inspect_robots.evidence`, NumPy only, that reads a finished `EvalLog`. **No
schema change.** Everything is computed from `SceneResult.epochs`, so it works on every log ever
written, including logs from before this plan.

```python
metric_evidence(log, scorers=None, *, alpha=0.05, n_boot=4000, seed=0) -> dict[str, MetricEvidence]
compare_logs(log_a, log_b, scorer, *, alpha=0.05, min_coverage=0.95, n_boot=4000,
             n_perm=20000, seed=0) -> Comparison
```

**`MetricEvidence`:** the mean over scenes of each scene's saved reduced score (as in
`results.metrics`), a percentile interval that resamples **whole scenes**, `n_scenes`, and
per-scorer `scored_trials`, `attempted_trials` and `coverage` (a trial counts as scored only with
a finite value for that scorer). Lower-is-better scorers (`LOWER_IS_BETTER`, or
`lower_is_better=True`) flip which side wins.

**`Comparison`:** scenes matched by `scene_id`. `delta` is the mean per-scene difference with a
scene-resampled interval, `wins`/`losses`/`ties` count scenes, `p_sign` is the exact sign test,
`p_permutation` the two-sided paired sign-flip test (exact up to 16 scenes, sampled with the add-one
correction above that), and `verdict` is one of `a_better`, `b_better`, `not_separated`,
`insufficient_coverage`, `insufficient_scenes`. It also reports `mde`, the difference the design had
an 80% chance to detect, and `scenes_needed(delta)` for planning the next run.

Refusals and warnings, each one a case seen in practice:

- different `task`: `ValueError`, since there is nothing to pair.
- either side below `min_coverage`: the numbers are printed, no winner is named, and the warning
  says the mean is over survivors.
- scenes scored on one side only: left out of the pairing and listed.
- `embodiment`, `max_steps`, `environment_id`, `environment_revision`, or epochs per scene differ:
  warned, since the pairing silently depends on them.
- the scene count cannot reach `alpha`: warned with the scene count that would, before anyone
  reads the p-value as evidence of a tie.

**CLI:** `inspect-robots compare A.json B.json [--scorer NAME]... [--alpha] [--min-coverage]
[--seed] [--json]`. With several scorers the verdict is read against Holm-adjusted p-values, so
scanning scorers does not manufacture a winner. Exit code 0, or 2 when the logs cannot be compared.

```
scorer                    A mean  B mean      A-B             95% CI     W-L-T        p   p holm     MDE  verdict
bb_binary_success          0.933   0.067    0.867     [0.667, 1.000]    13-0-2   0.0002   0.0002   0.255  A better
bb_graded_success          0.979   0.380    0.599     [0.470, 0.708]    15-0-0   0.0001   0.0002   0.176  A better
```

## Stopping a head-to-head early

On a robot every scene costs operator time, so the natural protocol is to watch the result and
stop once it is clear. Done with the fixed-design tests above, that is invalid: they assume the
scene count was fixed in advance. Simulated on scene-difference noise measured in a real
three-model benchmark run, an operator who checks after every scene from 6 to 60 and stops at
p < 0.05 declares a winner under no true difference in **35 to 36%** of runs with a z-test and
**18%** with the exact sign test, against a nominal 5%.

`anytime_valid_test(diffs, alpha=0.05, statistic="sign")` is built to be read that way: a betting
e-process (Waudby-Smith and Ramdas) whose chance of ever crossing `1 / alpha` under the null is at
most `alpha` by Ville's inequality. `compare_logs` runs it on the paired scenes in log order and
reports the e-value and the scene at which checking after every scene could have stopped (`e` and
`stop@` in the CLI). On the same noise:

| true gap | fixed 20 scenes, power | fixed 60, power | anytime (sign), power within 60 | mean scenes used |
| --- | --- | --- | --- | --- |
| 0.30 | 99.9% | 100% | 100% | 14.7 |
| 0.20 | 92.2% | 100% | 87.5% | 32.6 |
| 0.10 | 40.3% | 83.9% | 28.1% | 52.4 |

False winners under peeking: 1.9% for the anytime test. On clear gaps it stops after a quarter of
a 60-scene budget; on small, noisy gaps a fixed 60-scene design is more powerful. The honest
protocol is sequential stopping with a fixed-budget backstop, and the defaults say so: the sign
statistic is the default because the mean form, which must assume differences can span the whole
[-1, 1] range, stops later on the small differences robot scores produce.

## Why scenes, and why post hoc

Epochs of a scene share its world, so they are correlated draws, and in a benchmark that
re-randomizes the world per epoch they are still conditioned on the same scene. Resampling and
permuting scenes is the conservative choice that stays valid either way. Measured on a 5-scene
task: doubling trials per arm moved permutation-test power from 0.130 to 0.135, while going from 5
to 25 scenes at the same trial count moved it to 1.000.

Computing from the log rather than at eval time keeps `eval()` unchanged, keeps the schema at its
current version, and lets a benchmark maintainer re-evaluate every published log without re-running
a robot.

## Out of scope, and next

- Writing intervals into `EvalResults` at eval time (a later, additive schema change once the
  statistics have settled).
- A live `eval --stop-when-separated` mode that runs the anytime test between scenes and ends the
  run itself. The test here is the decision rule; wiring it into the rollout loop is separate.
- Sequential designs with a minimum sample (STEP, arXiv 2503.10966) and SAVI-based comparison of
  graded metrics (arXiv 2603.13616), both natural extensions of the same per-scene differences.
- `eval-set`-level comparison across many tasks with one family-wise correction.

## Tests

`tests/test_evidence.py`: closed-form sign-test values, exact against sampled permutation branches,
the design floor (`min_attainable_p(5) == 0.0625`, `scenes_to_reach(0.05) == 6`), a 5-0 sweep that
correctly does not separate, a one-hard-scene case the sign count exposes, coverage refusal, pairing
and drift warnings, Holm, planning against the normal approximation, and the CLI in text and JSON.
100% coverage of both new modules.
