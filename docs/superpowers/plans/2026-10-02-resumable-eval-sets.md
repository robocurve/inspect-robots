# Resumable Evaluation Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse completed scenes across explicit evaluation-set checkpoints and automatically retry only marked transient policy failures within a bounded attempt count.

**Architecture:** Keep `eval()` as the source of real attempt logs. A new checkpoint module owns identity, locking, and atomic manifest writes; a new merge module composes an aggregate `EvalLog` from immutable attempts. `eval_set()` schedules scene subsets, while the CLI exposes an explicit checkpoint path and existing retry count.

**Tech Stack:** Python 3.10+, NumPy-only core, stdlib JSON/filesystem, pytest, Ruff, strict mypy.

**Spec:** `docs/superpowers/specs/2026-10-02-resumable-eval-sets-design.md`

## Global Constraints

- Keep the core NumPy-only and preserve legacy `eval_set()` behavior when `retry_attempts=0` and no checkpoint is supplied.
- A completed zero-score scene is reusable when its status is success and it has all planned epochs.
- Only marked transient policy failures retry automatically; interrupts and halt-class errors never do.
- Old schema-v1 logs must continue to read and render.
- End with `ruff check .`, `ruff format --check .`, strict `mypy`, and `pytest --cov` at 100% coverage.

---

## File structure

- `src/inspect_robots/errors.py`, `rollout.py`, `eval.py`, `log.py`: typed retry marker and exact per-scene status/provenance recorded by real attempts.
- `src/inspect_robots/_eval_set_checkpoint.py`: manifest schema, identity normalization, exclusive lock, atomic updates, and attempt paths.
- `src/inspect_robots/_eval_set_merge.py`: choose one record per scene, recompute metrics and counts, and keep attempt provenance.
- `src/inspect_robots/eval.py`, `cli.py`: schedule scene subsets and expose `--checkpoint` with bounded retries.
- `src/inspect_robots/_html.py`, `_video.py`, `cli.py`: resolve per-scene frame sources in aggregate logs.
- `tests/test_eval_resume.py`, plus focused existing test modules: cover behavior through CubePick and temporary directories.
- `docs/guide/cli.md`, `CHANGELOG.md`: document the new CLI and API contract.

### Task 1: Record retry and per-scene provenance

**Files:** Modify `src/inspect_robots/errors.py`, `rollout.py`, `eval.py`, `log.py`; test `tests/test_eval_resume.py` and `tests/test_eval_log.py`.

**Interfaces:** `PolicyError(message, retryable=False)` exposes `.retryable`; `SceneResult` gains `frames_dir`, `errored_trials`, and `retryable_error`; `EvalLog` gains `source_logs`. All new log fields have defaults so legacy JSON remains valid.

- [ ] **Step 1: Write failing tests** for a default nonretryable `PolicyError`, an opted-in retryable one, a wrapped connection/timeout failure, a malformed action that remains nonretryable, exact scene error counts, and old log readback.

```python
assert PolicyError("bad action").retryable is False
assert PolicyError("temporary", retryable=True).retryable is True
assert SceneResult(scene_id="s", status="success").retryable_error is False
assert EvalLog.from_dict(old_schema_v1).source_logs == ()
```

- [ ] **Step 2: Run the focused tests and observe failures.**

```bash
uv run pytest tests/test_eval_resume.py tests/test_eval_log.py -q
```

- [ ] **Step 3: Implement the marker and serialization.** Preserve the `PolicyError` class name in existing error strings. In `_policy_error`, mark recognized connection and timeout exception chains retryable. In `_run_eval`, track scene-local error count and a conservative retryable flag; set it false on scorer, reducer, hook, cancellation, safety, and embodiment failures. Record frame root on each `SceneResult`. Coerce `source_logs` back to a tuple in `EvalLog.from_dict`.

```python
class PolicyError(InspectRobotsError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
```

- [ ] **Step 4: Run focused tests, Ruff, and mypy.**

```bash
uv run pytest tests/test_eval_resume.py tests/test_eval_log.py -q
uv run ruff check src/inspect_robots/errors.py src/inspect_robots/rollout.py src/inspect_robots/eval.py src/inspect_robots/log.py tests/test_eval_resume.py
uv run mypy
```

- [ ] **Step 5: Commit the self-contained log contract.**

```bash
git add src/inspect_robots/errors.py src/inspect_robots/rollout.py src/inspect_robots/eval.py src/inspect_robots/log.py tests/test_eval_resume.py tests/test_eval_log.py
git commit -m "feat(eval): record retryable scene errors and media provenance"
```

### Task 2: Persist and validate checkpoint manifests

**Files:** Create `src/inspect_robots/_eval_set_checkpoint.py`; test `tests/test_eval_resume.py`.

**Interfaces:** `_open_checkpoint(path: Path, identity: dict[str, object])` yields a manifest object with `attempts` and `add_attempt(task_index: int, scene_ids: list[str], log_path: Path)`. `_identity(tasks: Sequence[Task], policy: Policy | str, embodiment: Embodiment | str, *, seed: int, log_dir: str, options: Mapping[str, object]) -> dict[str, object]` normalizes the selected tasks, components, seed, and scoring/artifact options. The manifest stores relative attempt paths and rejects a changed identity or schema version.

- [ ] **Step 1: Write failing tests** for new manifest creation, exact-match reopen, changed scene/epoch/policy mismatch, invalid non-JSON scene declarations, a live sibling lock, and failure during `os.replace` leaving the prior manifest readable.

```python
with _open_checkpoint(path, identity) as manifest:
    manifest.add_attempt(0, ["scene-a"], attempt_path)
with _open_checkpoint(path, identity) as manifest:
    assert manifest.attempts[0]["scene_ids"] == ["scene-a"]
with pytest.raises(ConfigError, match="checkpoint identity"):
    with _open_checkpoint(path, changed_identity):
        pass
```

- [ ] **Step 2: Run the focused checkpoint tests and observe failures.**

```bash
uv run pytest tests/test_eval_resume.py -q
```

- [ ] **Step 3: Implement the module.** Use `os.open(lock, O_CREAT | O_EXCL | O_WRONLY, 0o600)` and release the lock in `finally`. Write a unique temporary JSON file, flush and `fsync`, then `os.replace`. Read a referenced attempt only after validating its path and schema. The scheduler resolves `seed=None` once before calling `_identity`.

```python
@dataclass
class _Manifest:
    path: Path
    identity: dict[str, object]
    attempts: list[dict[str, object]]

    def add_attempt(self, task_index: int, scene_ids: list[str], log_path: Path) -> None:
        self.attempts.append({"task_index": task_index, "scene_ids": scene_ids,
                              "log": os.path.relpath(log_path, self.path.parent)})
        self.write_atomic()
```

- [ ] **Step 4: Run focused tests, Ruff, and mypy.**

```bash
uv run pytest tests/test_eval_resume.py -q
uv run ruff check src/inspect_robots/_eval_set_checkpoint.py tests/test_eval_resume.py
uv run mypy
```

- [ ] **Step 5: Commit the checkpoint storage.**

```bash
git add src/inspect_robots/_eval_set_checkpoint.py tests/test_eval_resume.py
git commit -m "feat(eval): add atomic evaluation-set checkpoints"
```

### Task 3: Merge immutable attempt logs

**Files:** Create `src/inspect_robots/_eval_set_merge.py`; test `tests/test_eval_resume.py`.

**Interfaces:** `_merge_task(task: Task, attempts: Sequence[tuple[EvalLog, str]]) -> EvalLog` selects the first full successful result for each scene, otherwise the latest result, then recomputes metrics and counts. `_pending_scenes(task: Task, merged: EvalLog) -> list[Scene]` returns scenes whose record is absent or incomplete.

- [ ] **Step 1: Write failing tests** for a zero-score success retained across a retry, a partial scene replaced by a full retry, task-order output, exact error counts, source log list, and aggregate stats that include all attempts.

```python
merged = _merge_task(task, [(first_log, "attempt-1.json"), (second_log, "attempt-2.json")])
assert [sample.scene_id for sample in merged.samples] == ["scene-a", "scene-b"]
assert merged.samples[0].reduced["success_at_end"] == 0.0
assert merged.source_logs == ("attempt-1.json", "attempt-2.json")
```

- [ ] **Step 2: Run the merge tests and observe failures.**

```bash
uv run pytest tests/test_eval_resume.py -q
```

- [ ] **Step 3: Implement aggregation.** Use `dataclasses.replace` to attach each selected scene's frame root without changing an attempt log. Calculate `metrics` with the existing per-scene mean rule, `total_trials` from selected epoch lengths, `errored_trials` from selected scene counts, and total time/steps from all attempts. Set aggregate `mean_inference_latency_s=None` and `frames_dir=None`. Return status success only when every planned scene is complete.

```python
def _complete(scene: SceneResult, epochs: int) -> bool:
    return scene.status == "success" and len(scene.epochs) == epochs
```

- [ ] **Step 4: Run focused tests, Ruff, and mypy.**

```bash
uv run pytest tests/test_eval_resume.py -q
uv run ruff check src/inspect_robots/_eval_set_merge.py tests/test_eval_resume.py
uv run mypy
```

- [ ] **Step 5: Commit aggregation.**

```bash
git add src/inspect_robots/_eval_set_merge.py tests/test_eval_resume.py
git commit -m "feat(eval): merge resumed scene results"
```

### Task 4: Schedule resumable attempts through the API and CLI

**Files:** Modify `src/inspect_robots/eval.py`, `cli.py`; test `tests/test_eval_resume.py`, `tests/test_registry_cli.py`.

**Interfaces:** `eval_set` gains `checkpoint_path: str | None = None` and uses its existing `retry_attempts: int = 0`; the function keeps the old path when both features are off. With a checkpoint or retry budget, it evaluates the pending scene subset via `dataclasses.replace(task, scenes=tuple(pending_scenes))`, saves each real attempt with a leading `JsonLogSink`, updates the manifest after the attempt file exists, and calls `_merge_task`. `--checkpoint PATH` passes the path through; `--retry-attempts` remains available on its own.

- [ ] **Step 1: Write integration tests** using a CubePick policy that raises `PolicyError("flap", retryable=True)` once. Assert that `retry_attempts=1` completes without replaying a prior completed scene, `retry_attempts=0` leaves an error, and a second call with the same checkpoint skips completed scenes. Add one test for each nonretryable/halt/cancellation class, a fixed `seed=None` test, a zero-feature behavior control, and one CLI test that reuses an explicit checkpoint.

```python
success, logs = eval_set(tasks, flaky_policy, embodiment,
                         checkpoint_path=str(path), retry_attempts=1,
                         log_dir=str(tmp_path))
assert success
assert len(logs) == len(tasks)
assert path.exists()
```

- [ ] **Step 2: Run focused integration tests and observe failures.**

```bash
uv run pytest tests/test_eval_resume.py tests/test_registry_cli.py -q
```

- [ ] **Step 3: Implement scheduling and CLI wiring.** Preserve the current `eval_set` loop as the zero-feature path. In resumable mode, resolve task strings once, draw one seed when `seed=None`, validate checkpoint identity before rollout, and keep a per-scene automatic retry count. On Ctrl-C, reference the partial log written by the leading sink in the manifest before re-raising. On ordinary pre-rollout errors, retain the existing synthetic error-row behavior. The CLI must print the checkpoint path and reused/attempted scene counts.

```python
attempt_task = replace(task, scenes=tuple(pending_scenes))
assert attempt_task.epoch_spec == task.epoch_spec
```

- [ ] **Step 4: Run focused tests, Ruff, and mypy.**

```bash
uv run pytest tests/test_eval_resume.py tests/test_registry_cli.py -q
uv run ruff check src/inspect_robots/eval.py src/inspect_robots/cli.py tests/test_eval_resume.py tests/test_registry_cli.py
uv run mypy
```

- [ ] **Step 5: Commit the API and CLI integration.**

```bash
git add src/inspect_robots/eval.py src/inspect_robots/cli.py tests/test_eval_resume.py tests/test_registry_cli.py
git commit -m "feat(eval): resume incomplete scenes and retry transient failures"
```

### Task 5: Render mixed-source media and document the contract

**Files:** Modify `src/inspect_robots/_html.py`, `_video.py`, `cli.py`, `docs/guide/cli.md`, `CHANGELOG.md`; test `tests/test_html_view.py`, `tests/test_video.py`, `tests/test_eval_resume.py`.

**Interfaces:** A scene's `frames_dir` overrides the log-level frame root in aggregate logs. Legacy logs with no override retain their existing viewer and video behavior. The video command scans each selected scene root and emits only that scene's trial streams.

- [ ] **Step 1: Write failing viewer and video tests** with two selected scenes whose frames live under different attempt directories. Check both frame values in HTML and both MP4 stream inputs through the existing fake ffmpeg seam. Keep a legacy one-directory control.

```python
scene_a = replace(scene_a, frames_dir=str(first_frames))
scene_b = replace(scene_b, frames_dir=str(second_frames))
html = render_html(merged_log, title="resumed", log_path=log_path)
assert "scene-a" in html and "scene-b" in html
```

- [ ] **Step 2: Run focused media tests and observe failures.**

```bash
uv run pytest tests/test_html_view.py tests/test_video.py tests/test_eval_resume.py -q
```

- [ ] **Step 3: Resolve each scene's frame root** with `resolve_frames_dir`, reuse the shared HTML byte budgets, and let the video command enumerate selected streams across the distinct roots. Explain `--checkpoint`, retry semantics, matching limits, and safety exclusions in the CLI guide and changelog.

```python
source = scene.frames_dir or log.stats.frames_dir
root = None if source is None else resolve_frames_dir(source, log_path)
```

- [ ] **Step 4: Run all required gates and fix any concrete failures.**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov --cov-report=term-missing:skip-covered -q
uv run python scripts/gen_api_docs.py
cd website && npm ci && npm run build
```

- [ ] **Step 5: Commit the media and documentation slice, then inspect the branch diff.**

```bash
git add src/inspect_robots/_html.py src/inspect_robots/_video.py src/inspect_robots/cli.py docs/guide/cli.md CHANGELOG.md tests/test_html_view.py tests/test_video.py tests/test_eval_resume.py
git commit -m "feat(view): show media from resumed scene attempts"
git diff origin/main...HEAD --check
```
