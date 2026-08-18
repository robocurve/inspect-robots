# live view: instruction and rubric on running pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live `--serve` pages show each scene's instruction and rubric while
the run is still going (issue #392). Today `LiveLogSink` builds every
snapshot sample with `instruction=None` and no `scene_metadata` (its
`on_trial_start` receives only `(scene_id, epoch)`), so running pages show
neither until the completed log replaces the snapshot. Plan 0072 flagged
the gap and scoped the sink change out; this plan is that change.

**Architecture:** a duck-typed `bind_scenes(scenes)` hook offered by
`eval()`'s sink bus at run start, exactly like the existing
`bind_spaces`/`bind_frames_dir` hooks. `LiveLogSink` implements it by
storing a scene-id map of `(instruction, json_safe_metadata)` and fills
both fields in `_samples()`. The JSON-safe per-key deep copy `eval()`
already applies to final samples moves to a shared helper both call. The
renderer needs no change (`_scene_section` is state-agnostic for the
instruction and the plan-0072 rubric dropdown).

**Tech stack:** existing dependency-free logging machinery; no new deps;
mypy strict; pytest at 100% coverage.

**Spec:** issue #392 + this plan (the plan is the spec, per repo convention).

**Critique:** round 1 (fresh-context subagent) verified every claim
against the code and found 4 minors, all fixed in this revision: bus
class named correctly (`_Broadcast`) with the `callable` guard stated;
the reuse test respecified with a colliding scene id (distinct ids
cannot observe a stale map); helper renamed `_json_safe_scene_metadata`
with the exception tuple pinned; docs wording keyed on the rubric, not
auto-task mode. The critic blessed the plan as implementable.

## Global constraints

- Gates: `ruff check .`, `ruff format --check .`, `mypy` (strict,
  src + tests), `pytest --cov` at 100%.
- No public API change: `bind_scenes` is a duck-typed optional hook like
  `bind_spaces`, never added to the `LogSink` Protocol; `__all__` and
  `tests/test_api_snapshot.py` stay untouched.
- No log-schema change: both fields already exist on `SceneResult`.
- User-facing text follows the repo writing-style rule: no em dashes in
  prose.
- D1 docstrings state contracts, never restate names.

## Binding decisions

1. **The hook and its ordering mirror `bind_frames_dir` exactly.**
   `_Broadcast` (eval.py, the sink fan-out bus) gains
   `bind_scenes(scenes: Sequence[Scene])` offering
   `getattr(sink, "bind_scenes", None)` to every sink best-effort with
   the same `callable(hook)` guard its siblings have (the orchestration
   test pins that non-callable attributes are skipped); `eval()` calls `bus.bind_scenes(task.scenes)` alongside
   the existing two bind calls, **before** `bus.on_eval_start(spec)`.
   `LiveLogSink.bind_scenes` stores the map on the instance where
   `on_eval_start`'s reset does not clear it (the `_frames_dir` pattern:
   every run re-binds before starting, so reuse across sequential runs
   overwrites rather than leaks; the existing frames-dir reuse test is
   the model). Third-party sinks without the hook are unaffected.
2. **Shared JSON-safety.** The per-key `json.loads(json.dumps(value))`
   filter-and-deep-copy loop currently inline in `eval.py` (~line 466)
   moves to a module-level helper `_json_safe_scene_metadata(metadata:
   Mapping[str, Any]) -> dict[str, Any]` in `log.py` (underscore-private
   per repo convention, like `_sanitize`/`_slug`; a D1 docstring stating
   the filter+deep-copy contract). The helper preserves the exact
   `except (TypeError, ValueError, OverflowError)` tuple from eval.py:
   `ValueError` is what catches circular references, pinned by the
   existing mixed-metadata test. `eval.py` calls it for final samples;
   `LiveLogSink.bind_scenes` calls it once per scene at bind time (not
   per snapshot write: the metadata is frozen scene input, and
   `_samples()` runs on every throttled write). Bind-time copying means
   a caller that mutates `scene.metadata` mid-run can make the live page
   differ from the final log; the `bind_scenes` docstring says so (the
   final log replaces the snapshot, so the divergence is transient).
3. **Snapshot fill.** `LiveLogSink.bind_scenes` stores
   `{scene.id: (scene.instruction, _json_safe_scene_metadata(scene.metadata))}`.
   `_samples()` looks each `scene_id` up and fills
   `instruction=<stored or None>` and `scene_metadata=<a fresh dict copy
   of the stored dict, or {}>` — a fresh shallow copy per snapshot so the
   shallow-immutability stance of a written log is not undermined by
   sharing one mutable dict across every snapshot and the final log
   consumer. Unknown scene ids (a sink driven directly without a bind)
   keep today's `None`/`{}` behavior.
4. **Plan-0072 test wording.** `tests/test_html_view.py`'s
   started-log robustness test docstring says production started pages
   never carry `scene_metadata`; after this change live snapshots do
   carry it. Reword that docstring to drop the claim (the test itself
   stays: it pins renderer state-agnosticism).
5. **Docs.** `docs/guide/live-view.md` gains one sentence: running pages
   show each scene's instruction and, for scenes that carry a rubric,
   the rubric dropdown, live (the dropdown is keyed on the metadata, not
   on auto-task mode). The `docs/guide/cli.md` view-section sentence from
   plan 0072 stays correct as written (scene cards include the dropdown)
   and needs no edit. `src/inspect_robots/CLAUDE.md`: update the
   `logging/` row (LiveLogSink carries bound scene identity) and the
   `eval.py` row (`bind_scenes` joins the sink extension binds); `log.py`
   row mentions the shared JSON-safety helper.

## Tasks

- [ ] **1. Helper extraction.** Decision 2: `_json_safe_scene_metadata` in
  `log.py`, `eval.py` switched to call it. Tests: the existing
  mixed-metadata eval test keeps passing unmodified (it pins the
  behavior); add one direct unit test in `tests/test_eval_log.py` for the
  helper (drops non-JSON keys, deep-copies nested values).
- [ ] **2. `bind_scenes` + snapshot fill.** Decisions 1 and 3: the bus
  method, the `eval()` call, `LiveLogSink.bind_scenes`, `_samples()`
  fill. Tests in `tests/test_live_log.py`: a driven-sink test binds two
  scenes (one with a rubric plus a non-JSON metadata key, one without
  metadata) and asserts mid-run snapshots carry the instruction and only
  the JSON-safe metadata keys, with the unknown-id fallback intact; the
  eval-integration probe asserts a mid-run snapshot carries the scene's
  instruction (extend the existing probe test); sink reuse across two
  sequential runs replaces the map rather than merging: two tasks
  sharing one scene id with different instructions and rubrics, run 2's
  snapshot carries task 2's text (distinct ids cannot observe a stale
  map entry, so the colliding id is the point); a sinks-list entry without the hook is
  unaffected (the bus offer is best-effort — extend the existing bus
  hook test in the eval tests if one models `bind_spaces`, else add a
  minimal stub-sink test).
- [ ] **3. Renderer test wording + docs.** Decisions 4 and 5.
- [ ] **4. Gates + PR.** `ruff check .`, `ruff format --check .`,
  `uv run mypy`, `uv run pytest --cov` (100%). Push, CI green, fresh-eye
  review loop, merge (closes #392).

## Out of scope

- Streaming per-trial frames or any other live-page content change (the
  renderer is untouched).
- Adding `bind_scenes` to the `LogSink` Protocol (duck-typed like its
  siblings).
- `RerunSink` or other sinks adopting the hook (nothing there renders
  scene text today).
