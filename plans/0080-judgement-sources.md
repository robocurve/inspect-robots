# 0080: Persist which path produced each operator judgement

Closes #413.

## Problem

`_VLMGrader.grade` writes `operator_judgement` from two sources: when the
trial ended with a definitive termination reason it adopts the embodiment's
verdict without a model call, otherwise it samples the VLM. Both paths append
an `operator_event` carrying `source="embodiment"` or `source="vlm"`
(`transcript.py`), but `eval.py` persists only `kind == "operator_message"`
events, so the source never reaches the log. `eval.py` is the only reader of
`record.events` in `src/`, which makes the `source` and `note` fields on
`kind == "operator"` events dead data today. Reproduced on `main`
(`f4b6f03f`): a four-scene run with two adopted and two VLM-graded verdicts
writes indistinguishable `operator_judgements`.

## Design

One additive, schema-compatible field on `SceneResult`
(`src/inspect_robots/log.py`), strictly parallel to `epochs` like its
siblings:

```python
# Strictly parallel to ``epochs``: which path produced each recorded
# operator judgement ("embodiment", "vlm", "prompt", ...), ``None`` when no
# operator event was recorded (console verdicts, errored trials). The default
# keeps older logs readable.
judgement_sources: tuple[str | None, ...] = ()
```

- Value: the `source` string of the **last** `kind == "operator"` event in
  `record.events`, verbatim; `None` when there is none. No closed enum: the
  event already carries the vocabulary, a console-set verdict honestly has
  no event (#7), and normalising would force every future grader through a
  core change.
- No `SCHEMA_VERSION` bump: the field defaults to `()` and `EvalLog.from_dict`
  gains the same `tuple(sample.get("judgement_sources", ()))` coercion that
  `termination_reasons` has (`log.py` around line 164).
- `eval.py` (`_run_eval`): initialise a `judgement_sources` list beside
  `termination_reasons`; append `None` on the errored-trial branch; on the
  scored branch append the last operator event's source (extract it with a
  small private helper next to the existing operator_messages
  comprehension); pass `judgement_sources=tuple(...)` to the `SceneResult`
  constructor.
- Every other `SceneResult(...)` constructor in `src/` must pass the field
  so the parallel-tuple invariant holds everywhere: grep for
  `termination_reasons=` and mirror each site (the live-log snapshot path in
  `logging/live_log.py` is the known one; `tests/test_live_log.py::_parallel_lengths`
  guards it).
- `_html.py` / the report: no rendering change in this PR.

## Tests

- `tests/test_grader.py`: an end-to-end `eval()` on a task whose scenes mix
  definitive and non-definitive terminations, with `vlm_grader(...,
  http_post=<canned "GRADE: success">)`; assert `judgement_sources` is
  parallel to `termination_reasons` and equals the expected
  `("vlm", "embodiment", ...)` pattern, and that the VLM was called exactly
  once per non-definitive trial (guards against the field changing the
  adopt/sample decision).
- `tests/test_eval_log.py`: round-trip through `to_dict`/`from_dict`; a
  legacy dict without the key reads back as `()`; an errored trial yields
  `None` at its position (mirror the `termination_reasons` tests around
  lines 162-174).
- `tests/test_live_log.py`: add `len(sample.judgement_sources)` to
  `_parallel_lengths`.
- A trial with no operator event (no grader, or console verdict path)
  yields `None`.

Coverage stays at 100%.

## Docs

- Wherever `SceneResult` fields / the JSON log schema are documented under
  `docs/guide` (grep for `termination_reasons`), add `judgement_sources`
  with the one-line meaning above.
- `CHANGELOG.md` under `## [Unreleased]` / `### Added`, `**Core:**` prefix,
  referencing #413.

## Out of scope

Recording grader configuration (model, rubric, effort) in `EvalSpec`: the
issue raises it as a second, separate ask; leave a note on the issue.
Unifying the adopted-verdict vocabulary between the operator grader
(`"y"/"n"`) and the VLM grader (`"success"/"failure"`). Emitting
`operator_event` from the console path (#7).
