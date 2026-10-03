# 0079: Dual-rig fixed-trials campaign

Demo-branch only. Splits the plan-0078 campaign across the two physical
rigs so every test-taker runs 5 scored trials on rig-1 and 5 on rig-2
(8 models x 5 x 2 = 80 evals; measuring every model on both rigs keeps
rig differences out of the model comparison). The rigs run concurrently
on this machine, so each rig gets fully separate state and log
directories: the shared dirs would race the newest-final-log attribution
and interleave results.jsonl.

## Changes (all in `examples/actuate/`)

### `run_trials.py`: extract a parameterized engine

- New `run_campaign(rig_config: Path, state_dir: Path, logs_dir: Path,
  trials_per_model: int, argv: list[str]) -> None` holding the existing
  `main()` body. It assigns the rig-dependent module globals before the
  loop: `run.RIG_CONFIG`, `run.STATE_DIR`, `run.LOGS_DIR`,
  `run.STATUS_PATH` (`state_dir / "status.json"`), `run.RESULTS_PATH`
  (`state_dir / "results.jsonl"`), `run.MEDIA_DIR` (`state_dir /
  "media"`), plus the local trials target; passthrough args come from
  `argv` (same `--` convention via `run._extra_args`).
- `main()` becomes a thin call with today's defaults (rig-1 config,
  `state/`, `logs/`, `TRIALS_PER_MODEL`), keeping the single-rig
  invocation byte-compatible.
- The invariant to restate at the point of modification: the engine
  keeps the plan-0078 semantics unchanged — fewest-completed selection
  with roster-order ties, scored = non-null finite, both streak guards
  with the 3ad5612d reset rules, completion check after append (no dead
  final sleep), and the roles-clearing final status write.

### New: `run_trials_rig1.py`, `run_trials_rig2.py`

Thin booth scripts, `_roster.py`-style docstrings, each:

```python
TRIALS_PER_RIG = 5   # booth-editable; combined target is 2x this
RIG_CONFIG = Path.home() / "robocurve" / "rig-1" / "config.ini"  # or rig-2
```

calling `run_campaign(RIG_CONFIG, HERE / "state-rig1", HERE /
"logs-rig1", TRIALS_PER_RIG, sys.argv[1:])` (rig-2 mirrors with
`-rig2`). Thermal-gate channels resolve per rig automatically because
`run_campaign` reads them from the rig config (rig-2 uses can1/can0;
verified present in its config.ini).

### `serve.py`: per-rig monitor

`--state-dir` and `--logs-dir` flags (defaults: today's `state/` and
`logs/`), applied to the module paths before serving, so each rig can
run its own monitor: rig-1 on the default port 8377, rig-2 via
`--port 8378 --state-dir .../state-rig2 --logs-dir .../logs-rig2`.

### New: `combine_results.py`

Stdlib one-shot: takes any number of results.jsonl paths, prints the
pooled per-model n/mean (same scored = non-null finite rule) plus a
per-file breakdown, so the booth gets the combined leaderboard of both
rigs in one command.

### `.gitignore` (repo root)

Add `examples/actuate/logs-rig*/` and `examples/actuate/state-rig*/`
beside the existing entries.

### `README.md`

Extend the fixed-trials section: the dual-rig split (5+5 rationale),
the two run commands in separate tmux sessions (`actuate-rig1`,
`actuate-rig2`; each rig needs its own attended terminal for Esc), the
two serve commands with ports 8377/8378, `combine_results.py` for the
pooled leaderboard, the reminder that each rig's own `.env`/key setup
and the flock claim guard already handle concurrent runs, and that
fresh-campaign resets now clear the per-rig dirs.

## Out of scope

- run.py, start.sh (still wired to the rotating single-rig demo).
- Merging the two monitors into one page.

## Validation

- ruff gates; offline sims against `run_campaign` with per-rig temp
  dirs proving: default `main()` path unchanged; two campaigns with
  separate dirs do not interact; 5-trial target honored; per-rig
  channels resolved from the given config path.
- `combine_results.py` pooled means against hand-computed values.
