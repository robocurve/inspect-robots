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
all folded in below. R2 (2026-08-11, vs main @ aeefa002) found 2 more (the
call-site list was six sites short — the miss was systematic, hitting exactly
the cancelled/synthetic/halted paths this plan adds files for, so the list is
now twelve plus a re-scan checkbox and a clean-`git status` gate; the write
strategy was unspecified and a streaming writer would contradict the
"no file on degrade" contract, so the helper now serializes fully in memory
and lands the file atomically via temp + `os.replace`, matching
`json_log.py`) plus 5 nits (`eval_set` docstring line; `metadata["actions"]`
is framework-reserved; a narrower degrade-test seam than patching
`Path.open` globally; a Windows reserved-device-name acknowledgement for
`_safe` parity; a follow-up-issue checkbox for the transcripts side-car's
raw-scene-id bug) — all folded in below. R3 (2026-08-11, vs main @ aeefa002)
found 3 more, all test-implementability (the end-to-end NaN test was
platform-dependent — NaN→int64 casting yields 0 on ARM64 but INT64_MIN on
x86-64, where CubePick's render indexing then raises and the trial dies as
`EmbodimentFault` before the NaN ever reaches the record, so the test now
uses a NaN-tolerant embodiment subclass plus a helper-seam `ValueError`
test, with `pytest.warns(match=)` because numpy's own cast warning escapes;
both prior I/O-degrade seams were unimplementable — `run_stamp` is not
knowable before `eval()` and `Path.mkdir` can only be patched class-wide —
replaced with pre-creating `<log_dir>/actions` itself as a file; the
clean-`git status` gate was vacuous because `logs/` is gitignored, replaced
with asserting no `logs/` dir exists after the suite) plus 3 nits (the
re-scan must cover `plugins/*/tests` and import aliases like `ir_eval(`;
fsync before `os.replace` stated explicitly; header `scene_id` is raw) —
all folded in below. R4 (2026-08-11, vs main @ aeefa002) independently
re-verified the twelve-site list as exact and every citation/platform claim
correct; its one substantive finding was that the suite-litter gate was
vacuously implementable as an ordinary test (pytest collects
alphabetically, so `test_eval_action_log.py` runs before every litterer) —
the gate is now a `pytest_sessionfinish` hook in `tests/conftest.py` —
plus 4 nits (the zero-step precedent is
`FakeOperatorInput([ConsolePoll(end=EndRequest(...))])`, not
`_EndingOperatorInput`, which is the `end_trial()` teardown fixture; the
degrade catch+warn lives inside the helper, callers only check `None`; a
stranded `*.tmp` on disk-full mirrors `json_log.py` and is not a contract
violation; the follow-up issue covers the wire-capture side-car's raw ids
too) — all folded in below. R5 (2026-08-11, vs main @ aeefa002) verified
every structural element correct and implementable; its one substantive
finding was a false sentence — approval events do NOT survive into the
durable `EvalLog` (no events field; the `clamped` flags live on
`Action.meta`, which this artifact excludes), so the executed-action bullet
now states plainly that durable artifacts record only the executed action —
plus 3 nits (test_coverage_completion collects before the new test file, so
"every litterer" overstated; the `eval()` → `_run_eval` forwarding line
named alongside `eval_set`'s; the sessionfinish hook derives the repo root
from conftest `__file__`, not cwd) — all folded in below. R6 (2026-08-11,
vs main @ aeefa002) re-verified all citations and the twelve-site list;
its one substantive finding was that an existence-based litter gate
false-positives on any checkout that ever ran the quickstart (`logs/` is a
normal dev artifact) — the gate is now delta-based (sessionstart snapshot,
fail only if the dir appeared) — plus 4 nits (the "repo-level artifact
list" had no referent, retargeted to `docs/guide/logging-and-rerun.md`;
the sessionfinish failure mechanism pinned to `session.exitstatus = 1`,
raising there is an INTERNALERROR; plugin suites don't load
`tests/conftest.py`, gate-scope note added; the numpy-cast-warning
justification for `match=` softened) — all folded in below. R7 (2026-08-11,
vs main @ aeefa002) found NO SUBSTANTIVE ISSUES — every citation, platform
claim, call-site count, test seam, and gate mechanism independently
re-verified — with 3 optional nits (the delta gate is one-shot locally, so
the hook removes the suite-created `logs/` after failing; the header
example is illustrative, not a fixture spec; the `trial_metadatas.append`
ordering is an over-constraint since the append stores a reference —
harmless, kept for readability) — folded in below. **READY.**

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

Line 1 is a header, then one line per step, in step order (values
illustrative — tests run 2-dim CubePick with `dx`/`dy` labels):

```json
{"kind": "header", "run_id": "<run_stamp>", "scene_id": "...", "epoch": 0, "action_dim": 14, "labels": ["left_j0", ...]}
{"t": 0, "action": [0.0, ...]}
{"t": 1, "action": [0.01, ...]}
```

- The header's `run_id` value is `eval()`'s `run_stamp` — the same identifier
  that names the `frames/`, `wire/`, and `transcripts/` subdirectories; there
  is no other run identifier. The header's `scene_id` value is the **raw**
  scene id (sanitization applies to the filename only), so the original
  identity survives inside the artifact.
- `action` is the **executed** action — `StepRecord.action` is post-approval
  (`action = reviewed` precedes both `sink.log_step` and the record append in
  `rollout()`), which is what replay fidelity requires. Approver
  modifications are flagged in the in-memory record's approval events and
  the RerunSink stream, both non-durable; the durable artifacts record only
  the executed action. (Persisting approval events durably is a separate
  feature — out of scope.)
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
- **Write strategy (not streaming):** the helper serializes **all** lines in
  memory first, then writes to a temp path in the destination directory
  (flush + fsync, as `json_log.py` does) and `os.replace`s it into place —
  the full `json_log.py` atomic-write precedent.
  This makes "no file" literally true on the non-finite `ValueError` path
  (serialization fails before anything touches disk) and leaves no partial
  file **at the final path** on a mid-write `OSError` (a stranded `*.tmp`
  on disk-full mirrors `json_log.py` and is acceptable). A streaming
  line-by-line writer is explicitly ruled out: it would strand a
  header-plus-prefix partial file on exactly the failures the degrade
  contract promises leave nothing.
- **Degrade lives inside the helper:** `_write_action_log` itself catches
  `OSError`/`ValueError`, emits the `RuntimeWarning`, and returns `None`;
  the caller only checks the return value. (This is the only split
  consistent with the helper-seam test asserting the helper returns `None`
  and leaves no file.)
- `_safe` parity note: like `FrameStore`, Windows reserved device names
  (`CON`, `NUL`) pass through untouched. Accepted — parity with frames, and
  the Windows CI tier is non-blocking.

### Wiring in `eval()`

New keyword `store_actions: bool = True` on both `eval()` and `eval_set()`.
Two explicit forwarding lines are required — `eval()` → `_run_eval(...)`
(explicit keywords) and `eval_set`'s internal `eval(...)` call — and the
absence of either is invisible to a signature check (hence the behavioral
test below).

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
  `bus.on_trial_end(record)`, so sinks observe the pointer. The `"actions"`
  metadata key is hereby framework-reserved (alongside `transcript` and
  `wire_capture`); the write happens after `policy.on_trial_end`, so a
  policy-set key of the same name would be clobbered — document, don't
  defend.
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
be updated to a tmp `log_dir` or `store_actions=False`. Twelve sites found
as of aeefa002 (`tests/conftest.py` does no chdir isolation, and the
cancelled/errored/halted ones deliver records too — the side-car covers
exactly those paths):
`tests/test_coverage_completion.py:509, :539`,
`tests/test_eval_orchestration.py:248, :293, :334, :447, :475, :845`,
`tests/test_rerun_sink.py:419, :428, :1046`,
`tests/test_tracer_eval.py:103`.
Line numbers drift; re-run the scan at implementation time — every `eval(`/
`eval_set(` call lacking an explicit `log_dir`, covering `tests/` **and**
`plugins/*/tests` **and** import aliases (the agent plugin calls it as
`ir_eval(`; every plugin site passes `log_dir` today, but the scan must not
assume that). The gate cannot be `git status`: `.gitignore` covers `logs/`,
so a littering suite still shows clean. Gate on the directory itself —
assert no `logs/` exists under the repo root after the full suite.

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
  `poll()` ends the episode at t=0 — precedent:
  `FakeOperatorInput([ConsolePoll(end=EndRequest(...))])` in
  `tests/test_rollout_observation_step.py` (not `_EndingOperatorInput`,
  which is the `end_trial()` teardown fixture); eval-level operator
  fixtures in `tests/test_eval_orchestration.py`.
- [ ] Start-failed trial: `policy.on_trial_start` raises → synthetic record →
  header-only file and pointer.
- [ ] `epochs > 1`: two trials of the same scene produce distinct `-e0`/`-e1`
  files under one `run_stamp` directory.
- [ ] Sanitization: a scene id with a path-hostile character produces the same
  name `frames._safe` yields (collision-suffixed); no directory traversal.
- [ ] Labels: `null` when semantics is absent; `null` when semantics present
  but `dim_labels` unset; populated when set.
- [ ] Non-finite degrade (end-to-end): a policy emitting a NaN action, run
  against a **NaN-tolerant embodiment** — a `CubePickEmbodiment` subclass
  whose `step()` bypasses render indexing (precedent: `_NoDistanceEmbodiment`
  in `tests/test_strict_json.py`). Stock CubePick cannot carry this test
  portably: NaN→int64 casting is 0 on ARM64 but INT64_MIN on x86-64, where
  the render index raises and the trial dies as `EmbodimentFault` before the
  NaN reaches the record. Assert with
  `pytest.warns(RuntimeWarning, match=<degrade message>)` — a bare
  `RuntimeWarning` match is fragile (numpy cast warnings can escape
  NaN-handling runs depending on the embodiment's arithmetic; `match=` is
  robust regardless). Expect: warning, no file, no metadata key, eval
  status unaffected.
- [ ] Non-finite degrade (helper seam): `_write_action_log` on a hand-built
  record containing a NaN action returns `None` and leaves no file — covers
  the `ValueError` branch without an embodiment in the loop.
- [ ] I/O degrade: pre-create `<log_dir>/actions` **itself as a file** — the
  helper's `mkdir(parents=True, exist_ok=True)` of `actions/<run_stamp>`
  then raises `FileExistsError`/`NotADirectoryError` (both `OSError`)
  without the test needing to know `run_stamp` (generated inside `_run_eval`
  from wall clock + uuid4, so unknowable in advance), while `JsonLogSink`'s
  own mkdir of `log_dir` still succeeds. Run completes, `RuntimeWarning`
  observed, no metadata key, eval status unaffected. (Do not patch
  `Path.mkdir`/`Path.open` — class-wide, would break `JsonLogSink` and muddy
  the status assertion.)
- [ ] `eval_set` forwards behaviorally: `eval_set(..., store_actions=False)`
  → no `actions/` directory (precedent:
  `test_eval_set_forwards_before_scoring`).
- [ ] Existing call sites with custom sinks and default `log_dir` updated so
  the suite leaves no `logs/` litter in the checkout — re-run the scan (list
  above is a snapshot; include `plugins/*/tests` and the `ir_eval(` alias).
  The no-litter gate is **delta-based**: `pytest_sessionstart` in
  `tests/conftest.py` snapshots whether `<repo_root>/logs` exists, and
  `pytest_sessionfinish` fails the run (set `session.exitstatus = 1` and
  print the reason — raising there produces an INTERNALERROR exit) only if
  the directory **appeared during the session** — and then removes the
  suite-created directory after failing, so the gate is not one-shot on a
  local checkout (without cleanup, the second run would pass silently
  because the dir now pre-exists). Existence alone is not
  litter: a repo-root `logs/` is a normal artifact of developing here
  (quickstart and the docs default `log_dir="logs"`), and an
  existence-based gate would fail every local run forever after one
  quickstart while CI (clean checkout) stays exactly as strict either way.
  Derive the repo root from conftest's `__file__`, not cwd
  (`test_examples.py` chdirs). It cannot be an ordinary test in
  `test_eval_action_log.py`: pytest collects alphabetically, so that file
  runs before most of the potential litterers and the assertion would pass
  vacuously (`git status` is equally blind — `logs/` is gitignored). CI
  runs plain sequential pytest, so the hook is deterministic. Scope note:
  plugin test suites run as separate pytest invocations that never load
  `tests/conftest.py`, so plugin-side litter is protected by the call-site
  scan only, not by this gate.

Core coverage stays at 100% (`inspect_robots` scope); `mypy --strict` stays
clean.

## Docs & changelog

- [ ] `eval()` docstring: a `store_actions` paragraph alongside the existing
  keyword docs (`store_frames`, `operator_input`, …); one sentence in
  `eval_set()`'s docstring too (its keyword docs defer to `eval()`'s
  contract).
- [ ] File a follow-up issue for the raw-scene-id bug in the agent plugin's
  side-cars — both the transcripts path and the wire-capture path
  (`on_trial_start` passes `f"{scene_id}-e{epoch}"` raw to `begin_trial`);
  one issue covers both.
- [ ] `docs/guide/logging-and-rerun.md`: in the plan 0059 drop-shedding
  caveat, point to the action side-car as the guaranteed-complete record of
  commanded actions ("the `.rrd` is what the viewer saw; the actions JSONL is
  what the robot was told").
- [ ] `src/inspect_robots/CLAUDE.md` module table (`eval.py` row) mentions
  the `actions/` side-car; add it to the side-car coverage in
  `docs/guide/logging-and-rerun.md` (alongside its "Frame side-cars"
  material — there is no other enumerated artifact list in the repo).
- [ ] `CHANGELOG.md` entry under Unreleased: default-on durable action log,
  `store_actions=` opt-out, and the custom-sink behavior change called out
  explicitly.
