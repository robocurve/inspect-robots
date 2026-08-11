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
(`transcripts/<run_id>/<scene>-e<epoch>.jsonl`, relative path recorded in
`record.metadata["transcript"]`), written at trial end from the policy's
`on_trial_end(record, log_dir, run_id)` hook. Actions are policy-agnostic, so
the writer lives in core `eval()`, which already knows `log_dir`, `run_stamp`,
and receives the (possibly partial) record in its per-trial block.

A sink cannot do this cleanly: sinks never learn `run_stamp` (only the
duck-typed `bind_frames_dir` hints at it), and adding a binding hook for one
consumer grows protocol surface for no benefit.

### File layout and format

One file per trial: `<log_dir>/actions/<run_stamp>/<trial_id>.jsonl` where
`trial_id = f"{scene_id}-e{epoch}"`, with `scene_id` sanitized by the same
character rule `FrameStore` uses (`[^A-Za-z0-9._-]` → `_`) — scene ids are
task-author input; the transcripts side-car writes them raw, but new code
should not copy that latent bug.

Line 1 is a header, then one line per step, in step order:

```json
{"kind": "header", "run_id": "...", "scene_id": "...", "epoch": 0, "action_dim": 14, "labels": ["left_j0", ...]}
{"t": 0, "action": [0.0, ...]}
{"t": 1, "action": [0.01, ...]}
```

- `action` is the **executed** action — `StepRecord.action` is post-approval
  (`action = reviewed` precedes both `sink.log_step` and the record append in
  `rollout()`), which is what replay fidelity requires. Approver
  modifications remain visible in the log's approval events.
- `labels` comes from `embodiment.info.action_space.semantics.labels`; `null`
  when the space carries no semantics.
- `Action.meta` is deliberately excluded: unbounded, caller-defined, not
  JSON-safe by contract. Out of scope.
- A zero-step trial writes a header-only file: presence means "this trial
  ran".
- Floats serialize via `float()` on each element of
  `np.asarray(step.action.data).ravel()` — plain `json.dumps`, `allow_nan`
  left at default (actions passed the approver chain; a NaN action would have
  faulted the embodiment long before scoring).

### Wiring in `eval()`

New keyword `store_actions: bool = True` on both `eval()` and `eval_set()`
(passed through like `store_frames`). In the per-trial block of `_eval_one`,
immediately after the `policy.on_trial_end` hook and **before**
`trial_metadatas.append(record.metadata)`:

- when `store_actions` and the write succeeds, set
  `record.metadata["actions"] = f"actions/{run_stamp}/{trial_id}.jsonl"`
  (relative, like the transcript pointer);
- the write is best-effort: on `OSError`/serialization failure, emit a
  `RuntimeWarning` (mirroring the live-transcript-stream degrade style) and
  skip the pointer — a side-car failure must never fail the eval or mask the
  trial's own status.

The write happens for **every** delivered trial, errored ones included — the
partial record is already in hand on that path. The directory is created
lazily on first write (`mkdir(parents=True, exist_ok=True)`), so
`store_actions=False` and zero-trial runs leave no empty `actions/` litter.

The writer is a module-private helper `_write_action_log(record, log_dir,
run_stamp, action_space) -> str | None` in `eval.py` (returns the relative
path or `None` on degrade) so the per-trial block stays readable and the
helper is unit-testable.

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
- [ ] `store_actions=False`: no `actions/` directory, no metadata key.
- [ ] Zero-step trial: header-only file.
- [ ] Sanitization: a scene id with a path-hostile character maps to the same
  name `FrameStore` would produce; no directory traversal.
- [ ] No-semantics space: header `labels` is `null`.
- [ ] Degrade: patch the helper's open to raise `OSError`; run completes,
  `RuntimeWarning` observed, no metadata key, eval status unaffected.
- [ ] `eval_set` passes `store_actions` through (signature-level test is
  sufficient given `eval_set` delegates).

Core coverage stays at 100% (`inspect_robots` scope); `mypy --strict` stays
clean.

## Docs & changelog

- [ ] `docs/guide/logging-and-rerun.md`: in the plan 0059 drop-shedding
  caveat, point to the action side-car as the guaranteed-complete record of
  commanded actions ("the `.rrd` is what the viewer saw; the actions JSONL is
  what the robot was told").
- [ ] `src/inspect_robots/CLAUDE.md` module table (`eval.py` row) and the
  repo-level artifact list mention the `actions/` side-car.
- [ ] `CHANGELOG.md` entry under Unreleased: default-on durable action log,
  `store_actions=` opt-out.
