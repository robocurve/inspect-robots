# 0080: Persist which path produced each operator judgement

Closes #413.

## Problem

`_VLMGrader.grade` writes `operator_judgement` from two sources: when the
trial ended with a definitive termination reason it adopts the embodiment's
verdict without a model call, otherwise it samples the VLM. Both paths append
an `operator_event` carrying `source="embodiment"` or `source="vlm"`
(`transcript.py`), as do the console path (`rollout.py`, `source="console"`)
and the prompted operator grader (`session.py`, `source="prompt"`), but
`eval.py` persists only `kind == "operator_message"` events and
`logging/live_log.py` mirrors that, so the source never reaches either log.
The `source` and `note` fields on `kind == "operator"` events are dead data
today. Reproduced on `main`
(`f4b6f03f`): a four-scene run with two adopted and two VLM-graded verdicts
writes indistinguishable `operator_judgements`.

## Design

One additive, schema-compatible field on `SceneResult`
(`src/inspect_robots/log.py`), strictly parallel to `epochs` like its
siblings:

```python
# Strictly parallel to ``epochs``: which path produced each recorded
# operator judgement ("console", "prompt", "embodiment", "vlm"), ``None``
# when the trial has no judgement (errored trials, no grader, a skipped
# prompt) or the grader recorded one without an operator event. The default
# keeps older logs readable.
judgement_sources: tuple[str | None, ...] = ()
```

- Value rule: `None` whenever `record.operator_judgement is None`;
  otherwise the `source` string of the **last** `kind == "operator"` event
  in `record.events`, verbatim, or `None` if there is no such event (a
  grader that sets `operator_judgement` without appending an event, as the
  test helper `_RecordingGrader` does). Tying the source to the judgement
  matters for the skip-with-note case: `session.py` appends an
  `operator` event with `verdict="skip"`, `source="prompt"` while the
  judgement stays `None`, and a source for a judgement that was never
  recorded would contradict the field's definition. No closed enum: the
  event already carries the vocabulary and normalising would force every
  future grader through a core change.
- One shared helper, `judgement_source(record: TrialRecord) -> str | None`,
  implementing the rule above, placed in `transcript.py` next to
  `operator_event` (it is the counterpart reader) and called from both
  writers below so live and final logs agree. Private to the package (no
  `__all__` change). `rollout.py` imports `transcript` at module top, so
  the `TrialRecord` annotation must come from a `TYPE_CHECKING`-guarded
  import (the `live_log.py` precedent); return `event.data.get("source")`
  directly, no narrowing needed under mypy strict.
- `eval.py` (`_run_eval`): initialise a `judgement_sources` list beside
  `termination_reasons`; append `None` on the `record.status != "success"`
  branch (errored and cancelled trials, not under the inner
  `status == "error"` check); on the
  scored branch append `judgement_source(record)` at the same instant as
  `judgements.append(record.operator_judgement)` (the comment there
  explains why the capture point matters); pass
  `judgement_sources=tuple(...)` to the `SceneResult` constructor.
- `logging/live_log.py` keeps its own parallel pipeline and needs all four
  touch points: a field on `_LiveScene`, the placeholder append in
  `on_trial_start`, the fill from the record in `on_trial_end` (via the
  shared helper), and the snapshot `SceneResult(...)` constructor.
  `tests/test_live_log.py::_parallel_lengths` guards the invariant.
- No `SCHEMA_VERSION` bump: the field defaults to `()` and `EvalLog.from_dict`
  gains the same `tuple(sample.get("judgement_sources", ()))` coercion that
  `termination_reasons` has (`log.py` around line 164).
- These are the only two `SceneResult(...)` constructors in `src/`
  (`eval.py` and `live_log.py`); `_html.py` / the report get no rendering
  change in this PR.

## Tests

- `tests/test_vlm_grader.py` (it already has `_CapturePost` and the
  `_vlm(post, monkeypatch)` helper): an end-to-end `eval()` on four
  `CubePick` scenes (`init_seed` 0..3) with `ScriptedPolicy` and
  `max_steps=8`, which yields
  `termination_reasons == ("max_steps", "success", "success", "max_steps")`;
  assert `judgement_sources == ("vlm", "embodiment", "embodiment", "vlm")`,
  that it is parallel to `termination_reasons`, and that `http_post` was
  called exactly twice (guards against the field changing the adopt/sample
  decision).
- `tests/test_grader.py`, using the existing `_scripted_session` helper
  through `eval(..., grader=operator_grader(session))` with `max_steps=1`
  (at the file's default `max_steps=60` the scripted policy succeeds and the
  session adopts the embodiment verdict without consuming the scripted
  answers): a prompted verdict yields `("prompt",)`;
  `_scripted_session(["skip", "a note"])` yields `(None,)` beside
  `operator_judgements == (None,)`; the `_RecordingGrader` path (judgement
  set, no event) yields `(None,)`.
- Console: `tests/test_eval_orchestration.py::test_eval_does_not_observe_parked_after_operator_judgement`
  already runs `eval()` with `EndRequest(verdict="y")` and discards the
  return; capture `(log,) = eval(...)` and assert
  `log.samples[0].judgement_sources == ("console",)`. Record-level:
  `tests/test_rollout_observation_step.py::test_verdict_console_end_records_judgement_note_and_operator_event`
  can assert `judgement_source(record) == "console"`.
- An errored trial yields `None` at its position.
- `tests/test_eval_log.py`: add a non-default `judgement_sources` to
  `_golden_log()` so `to_dict`/`from_dict` round-trips a real value; in
  `test_v1_log_without_additive_fields_reads_back` add
  `del sample["judgement_sources"]` and assert it reads back as `()`.
- `tests/test_live_log.py`: add `len(sample.judgement_sources)` to
  `_parallel_lengths`; add `operator_event(0, "y", source="console")` to
  the `_record()` helper's events and assert
  `sample.judgement_sources == ("console",)` in a snapshot after
  `on_trial_end` (the helper currently carries only an
  `operator_message_event`, for which the source is `None`).

Coverage stays at 100%.

## Docs

- No docs page enumerates `SceneResult` fields (verified). Add one sentence
  where adoption is described ("adopted without spending a model call",
  `docs/guide/cli.md` around line 202): the log records which path produced
  each verdict in `judgement_sources`.
- `CHANGELOG.md` under `## [Unreleased]` / `### Added`, `**Core:**` prefix,
  linking `[#413](https://github.com/robocurve/inspect-robots/issues/413)`
  in the neighbouring entries' style.

## Out of scope

Recording grader configuration (model, rubric, effort) in `EvalSpec`: the
issue raises it as a second, separate ask; leave a note on the issue.
Unifying the adopted-verdict vocabulary between the operator grader
(`"y"/"n"`) and the VLM grader (`"success"/"failure"`). Issue #7 (whether
every operator path emits `operator_event`) is independent of this field.
