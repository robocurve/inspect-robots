# 0067 — Durable per-trial action log

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** Close #369. Commanded actions currently reach disk through exactly one
path: the RerunSink `.rrd` tee (plan 0059), which deliberately drop-sheds under
pressure — the `.rrd` is "what the viewer saw", not a guaranteed-complete
record. The durable stores don't cover the gap: `JsonLogSink` persists only the
final `EvalLog` (no per-step data by design) and `FrameStore` persists frames
only. After this change, every `eval()` run also writes one small JSONL file
per trial containing the executed action sequence, default-on, alongside the
run's other side-car artifacts — so a finished run can be replayed at exact
control-step resolution instead of interpolated between inference boundaries.

**Critique status:** R1 (2026-08-11, vs main @ aeefa002) found 7 substantive
issues (sanitization must reuse `frames._safe` — the described character rule
was wrong and collision-prone; the labels attribute is
`ActionSemantics.dim_labels`, not `.labels`, and may be unset with semantics
present; `allow_nan=False` is required by the repo's strict-JSON convention
and the NaN-can't-happen rationale was false under `AutoApprover`; default-on
is a behavior change for custom-sink callers and six existing test call sites
would start writing `logs/` into the checkout; side-car placement relative to
the `policy_start_failed` guard was ambiguous and the "this trial ran"
wording false for synthetic records; the zero-step test had no implementable
route; the `eval_set` signature-level test verifies nothing and
cancelled-trial/epochs>1 outcomes were untested) plus 5 nits (`_run_eval` not
`_eval_one`; header `run_id` is `run_stamp`; the scoring sentence; the
`eval()` docstring surface; the `resolve_log_pointer` convention note) —
all folded in below.

## Problem

`rollout()` hands every executed (post-approval) action to `sink.log_step` and
appends it to `TrialRecord.steps`, but nothing persists the sequence durably:

- `JsonLogSink` — final log only ("Per-step data lives in the
  `TrialRecord`/`FrameStore`, not here").
- `FrameStore` — frames only (R5).
- `RerunSink` — logs action vectors, but sheds under pressure by contract.

Anyone reconstructing trajectories from a finished run (offline `.rrd`
rebuilds, report analysis) must interpolate between LLM-call boundaries. The
GPT-5.6 Sol eval report hit exactly this.

The data already exists in memory at trial end: `TrialRecord.steps` holds every
`StepRecord` with its action, and errored trials carry the partial record on
`exc.record`, which the eval loop already recovers and delivers to sinks. So a
one-shot write at trial end is complete for every outcome except a hard process
kill, and puts zero I/O in the control-rate loop — no drop-shedding concern.

## Design

### Ownership: eval-owned side-car, not a sink

Per-step artifacts are owned by the run, not by sinks (R5 precedent: frames
live in a rollout-owned `FrameStore`, never in a sink). The closest sibling is
the agent plugin's transcript side-car
(`transcripts/<run_stamp>/<scene>-e<epoch>.jsonl`, relative path recorded in
`record.metadata["transcript"]`), written at trial end from the policy's
`on_trial_end(record, log_dir, run_id)` hook. Actions are policy-agnostic, so
the writer lives in core `eval()`, which already knows `log_dir`, `run_stamp`,
and receives the (possibly partial) record in its per-trial block.

A sink cannot do this cleanly: sinks never learn `run_stamp` (only the
duck-typed `bind_frames_dir` hints at it), and adding a binding hook for one
consumer grows protocol surface for no benefit.

### File layout and format

One file per trial: `<log_dir>/actions/<run_stamp>/<trial_id>.jsonl` where
`trial_id = f"{scene_id}-e{epoch}"` and `scene_id` is sanitized by **reusing
`inspect_robots.frames._safe`** (module-private but intra-package; promote to
a shared underscore-helper import rather than duplicating). `_safe` replaces
unsafe runs with `-` and appends a crc32 suffix whenever it changed anything,
so distinct hostile ids cannot collide — do not re-derive a character rule
here. (The transcripts side-car writes scene ids raw; new code should not
copy that latent bug.)

Line 1 is a header, then one line per step, in step order:

```json
{"kind": "header", "run_id": "<run_stamp>", "scene_id": "...", "epoch": 0, "action_dim": 14, "labels": ["left_j0", ...]}
{"t": 0, "action": [0.0, ...]}
{"t": 1, "action": [0.01, ...]}
```

- The header's `run_id` value is `eval()`'s `run_stamp` — the same identifier
  that names the `frames/`, `wire/`, and `transcripts/` subdirectories; there
  is no other run identifier.
- `action` is the **executed** action — `StepRecord.action` is post-approval
  (`action = reviewed` precedes both `sink.log_step` and the record append in
  `rollout()`), which is what replay fidelity requires. Approver
  modifications remain visible in the log's approval events.
- `labels` comes from `embodiment.info.action_space.semantics.dim_labels`;
  `null` when semantics is absent **or** `dim_labels` is unset (it defaults
  to `None` even with semantics present).
- `Action.meta` is deliberately excluded: unbounded, caller-defined, not
  JSON-safe by contract. Out of scope.
- A zero-step trial writes a header-only file: presence means "this trial was
  delivered" (see placement below — synthetic records get files too).
- Floats serialize via `float()` on each element of
  `np.asarray(step.action.data).ravel()` — plain `json.dumps` with
  **`allow_nan=False`**, matching the repo's strict-JSON convention for
  durable artifacts (`json_log.py` docstring, `tests/test_strict_json.py`).
  A non-finite action is reachable in programmatic runs (`eval()`'s default
  approver is `AutoApprover`, which reviews nothing; the NaN gate lives in
  `ClampApprover`, which only the CLI wires by default), and sanitizing it to
  `null` would corrupt a replay artifact — so a non-finite value trips the
  documented degrade path loudly instead: warning, no file, no pointer.

### Wiring in `eval()`

New keyword `store_actions: bool = True` on both `eval()` and `eval_set()`.
`eval_set` forwards every keyword **explicitly** in its internal `eval(...)`
call — the forwarding line `store_actions=store_actions` must be added there,
and its absence is invisible to a signature check (hence the behavioral test
below).

In the per-trial block of `_run_eval`, the write sits **after and outside**
the `if not policy_start_failed:` guard that wraps the `policy.on_trial_end`
hook, and **before** `trial_metadatas.append(record.metadata)`. Every record
that reaches `bus.on_trial_end` gets a side-car — including the synthetic
record built when `policy.on_trial_start` fails (rollout never ran;
header-only file) and the cancelled-trial record recovered from
KeyboardInterrupt (the log is written before the interrupt re-raises, so the
side-car must exist by then too).

- When `store_actions` and the write succeeds, set
  `record.metadata["actions"] = f"actions/{run_stamp}/{trial_id}.jsonl"` —
  relative and beneath the log dir, deliberately matching the
  `resolve_log_pointer` convention that makes the existing `transcript` and
  `wire_capture` pointers consumable later. The mutation precedes
  `bus.on_trial_end(record)`, so sinks observe the pointer.
- The write is best-effort: on `OSError`/`ValueError` (non-finite refusal)
  emit a `RuntimeWarning` (mirroring the live-transcript-stream degrade
  style) and skip the pointer — a side-car failure must never fail the eval
  or mask the trial's own status.
- The directory is created lazily on first write
  (`mkdir(parents=True, exist_ok=True)`), so `store_actions=False` and
  zero-trial runs leave no empty `actions/` litter.

The writer is a module-private helper `_write_action_log(record, log_dir,
run_stamp, action_space) -> str | None` in `eval.py` (returns the relative
path or `None` on degrade) so the per-trial block stays readable and the
helper is unit-testable.

### Behavior change: default-on writes for custom-sink callers

Today `eval(..., sinks=[custom])` touches no disk (`JsonLogSink` is only
constructed when `sinks is None`; frames only when `store_frames=True`).
After this change such calls write `actions/` beneath the default
`log_dir="logs"` unless they pass `store_actions=False`. This is deliberate —
the artifact must not silently vanish because a caller brought their own
sinks — but it must be called out in the changelog and docs, and the existing
test call sites that would start writing `logs/` into the repo checkout must
be updated to a tmp `log_dir` or `store_actions=False`:
`tests/test_coverage_completion.py:509, :539`,
`tests/test_eval_orchestration.py:447`,
`tests/test_rerun_sink.py:419, :428, :1046`.

### What does not change

- `EvalLog` schema: untouched — the pointer rides the existing free-form
  per-trial `metadata`, exactly like `transcript` and `wire_capture`.
- The `LogSink` protocol, `rollout()`, `RerunSink`, and the CLI: untouched.
  The CLI inherits default-on behavior through `eval()`; no new flag (the
  files are ~100 KB/trial; programmatic callers can pass
  `store_actions=False`).
- `TrialRecord` / `StepRecord`: no new fields.

## Tests

`tests/test_eval_action_log.py`, driven end-to-end through `eval()` with the
`CubePick` mock world (no hardware), plus targeted helper tests:

- [ ] Happy path: `eval(..., log_dir=tmp)` writes
  `actions/<run_stamp>/<trial_id>.jsonl`; header parses with correct
  `action_dim`/`labels`/identity; step lines are contiguous `t = 0..N-1`; each
  `action` round-trips equal to the corresponding `record`-side step action;
  `metadata["actions"]` holds the relative path and joins to an existing file.
- [ ] Executed-action fidelity: with a rewriting approver (e.g. `ClampApprover`
  and an out-of-bounds scripted policy), the logged vector equals the clamped
  action, not the emitted one.
- [ ] Errored trial: a policy that raises mid-trial still yields a side-car
  with the steps walked before the failure, and the pointer in that trial's
  metadata.
- [ ] Cancelled trial: KeyboardInterrupt mid-trial (`_CancelledTrial` path) —
  the delivered record's side-car and pointer exist even though the interrupt
  re-raises after the log write.
- [ ] `store_actions=False`: no `actions/` directory, no metadata key.
- [ ] Zero-step trial: header-only file, via an operator input whose first
  `poll()` ends the episode at t=0 (precedent: `_EndingOperatorInput`,
  `tests/test_rollout_observation_step.py`; eval-level operator fixtures in
  `tests/test_eval_orchestration.py`).
- [ ] Start-failed trial: `policy.on_trial_start` raises → synthetic record →
  header-only file and pointer.
- [ ] `epochs > 1`: two trials of the same scene produce distinct `-e0`/`-e1`
  files under one `run_stamp` directory.
- [ ] Sanitization: a scene id with a path-hostile character produces the same
  name `frames._safe` yields (collision-suffixed); no directory traversal.
- [ ] Labels: `null` when semantics is absent; `null` when semantics present
  but `dim_labels` unset; populated when set.
- [ ] Non-finite degrade: a policy emitting a NaN action under the default
  approver → `RuntimeWarning`, no file, no metadata key, eval status
  unaffected.
- [ ] I/O degrade: patch the helper's open to raise `OSError`; run completes,
  `RuntimeWarning` observed, no metadata key, eval status unaffected.
- [ ] `eval_set` forwards behaviorally: `eval_set(..., store_actions=False)`
  → no `actions/` directory (precedent:
  `test_eval_set_forwards_before_scoring`).
- [ ] Existing call sites with custom sinks and default `log_dir` updated so
  the suite leaves no `logs/` litter in the checkout (list above).

Core coverage stays at 100% (`inspect_robots` scope); `mypy --strict` stays
clean.

## Docs & changelog

- [ ] `eval()` docstring: a `store_actions` paragraph alongside the existing
  keyword docs (`store_frames`, `operator_input`, …).
- [ ] `docs/guide/logging-and-rerun.md`: in the plan 0059 drop-shedding
  caveat, point to the action side-car as the guaranteed-complete record of
  commanded actions ("the `.rrd` is what the viewer saw; the actions JSONL is
  what the robot was told").
- [ ] `src/inspect_robots/CLAUDE.md` module table (`eval.py` row) and the
  repo-level artifact list mention the `actions/` side-car.
- [ ] `CHANGELOG.md` entry under Unreleased: default-on durable action log,
  `store_actions=` opt-out, and the custom-sink behavior change called out
  explicitly.
