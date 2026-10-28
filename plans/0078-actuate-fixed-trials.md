# 0078: Actuate demo fixed-trials runner

> Amended 2026-08-19 (commit on demo branch): selection is now a randomized
> block design. Fewest-completed still bounds every model to one trial per
> round, but ties are drawn randomly instead of in roster order, so a
> position-in-round effect cannot align with any one model. Everything else
> in this plan stands.

Demo-branch only. New sibling script; `run.py` and `start.sh` are not
modified.

## Goal

A deterministic campaign for the show format: every test-taker in the
roster completes exactly `TRIALS_PER_MODEL = 10` scored evals (7 models,
70 scored evals total), tasker and grader pinned as the roster already
pins them (single-member pools). No randomization anywhere.

## New file: `examples/actuate/run_trials.py`

Imports `run` (same directory; `run.py` is `__main__`-guarded and
import-safe) and reuses its machinery unchanged: `_command`,
`_write_status`, `_extra_args`, `_docs_extra_args` (via `_command`),
`_final_log_paths`, `_newest_final_log`, `_log_outcome`, `_render_clips`,
`_append_result`, `_next_eval_index`, `_thermal_gate`, `_eligible`, and the
constants `RIG_CONFIG`, `PAUSE_S`, `FAILURE_PAUSE_S`, `FAILURE_STREAK_STOP`,
`RESULTS_PATH`, plus `config_channels` from `_thermal`.

Booth-editable constant at top: `TRIALS_PER_MODEL = 10`. A second guard
constant `UNSCORED_STREAK_STOP = 10`.

Behavior (mirrors `run.py`'s loop; differences only where stated):

1. Startup: same `RIG_CONFIG` check, channels via
   `config_channels(RIG_CONFIG)`, state/log dirs, extra args, gate state.
   Role pools via `run._eligible`: tasker and grader pools must have
   exactly one member each; the test-taker pool order is the roster order.
   `SystemExit` with a clear message otherwise (the pinned-roles roster is
   a precondition, not something this script re-implements).
2. Trial accounting, recomputed from `RESULTS_PATH` before every eval
   (restart-safe): a completed trial for model name N is a results.jsonl
   row with `roles.test_taker == N` and a non-null finite `score`. Rows
   from earlier demo sessions count; the README tells the operator to
   clear state for a fresh campaign.
3. Selection, fully deterministic: among test-takers with fewer than
   `TRIALS_PER_MODEL` completed trials, pick the one with the fewest;
   ties break by roster order. (With no failures this yields strict
   round-robin interleaving — cycle 1 all seven, cycle 2 all seven, ... —
   and after a failure the lagging model is retried first.)
4. When every model has `>= TRIALS_PER_MODEL`: print a per-model summary
   (n, mean score, matching the leaderboard's scored-evals-only
   semantics) and exit 0.
5. Each eval: identical to `run.py` — thermal gate first, status write
   (same shape, so `serve.py`/`monitor.html` work unchanged), launch via
   `run._command(tasker, test_taker, grader, extra_args)`, outcome
   parsing, clip rendering, results append, pause logic. Console line
   additionally prints trial progress: `trial 4/10 for GPT-5`.
6. Failure guards: `run.py`'s log-less `FAILURE_STREAK_STOP` behavior is
   retained verbatim. New: a consecutive streak of evals that produce a
   log but **no score** (grader broken) halts after
   `UNSCORED_STREAK_STOP` with a message naming the grader — without
   this, a broken grader would loop forever because unscored evals never
   count as trials. Any scored eval resets both streaks.
7. Ctrl-C: same `KeyboardInterrupt` exit as `run.py`.

## README

New short subsection after "Running it": fixed-trials campaign, run
manually in the same tmux arrangement (`start.sh` stays wired to the
rotating demo; the trials runner is
`tmux new -s actuate 'python examples/actuate/run_trials.py -- ...'` with
the serve session as usual), passthrough args after `--` identical, start
from a fresh leaderboard for a clean campaign (existing scored evals count
toward the 10), resumes where it left off after a restart, exits when
every test-taker has ten scored evals.

## Out of scope

- Any change to `run.py`, `start.sh`, `_roster.py`, `serve.py`,
  `monitor.html`.
- Per-trial scene control (task authorship stays with the pinned tasker).

## Validation

- ruff gates pass.
- Offline simulation with stubbed `RESULTS_PATH` content: selection is
  round-robin from empty state; resumes correctly from partial counts
  (e.g. one model at 10, others at 9 → picks the laggards in roster
  order); unscored rows do not count; exits at 10/10 for all seven.
- On-rig: covered by the existing booth checklist (force-one-eval items).
