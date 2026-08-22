# 0076: Parked final frames for grading

Closes #399.

## Problem

The vlm grader judges a trial from its first and last frames, and the final
frames come from `record.steps[-1].result.observation` (`grader.py`,
`_phase_frames`): the camera images captured wherever the arm happened to be
when the episode ended. Parking happens later (embodiment teardown or the
next reset), so end-of-episode poses routinely occlude the workspace in
exactly the frames being graded — worst on overhead cameras and for
timeout episodes that end mid-reach.

## Design

A duck-typed embodiment hook, following the existing `bind`, `bind_task`,
`on_trial_end`, and `transcript` precedents (no Protocol change, no
`__all__` change):

- **Hook contract** (documented in `eval()`'s docstring): when a trial is
  about to be graded, the eval loop calls `embodiment.observe_parked()` if
  the embodiment defines it. The embodiment moves itself to its parked/rest
  pose so cameras see the scene unobstructed and returns one fresh
  `Observation`; returning `None` declines (for example, when the task's
  success state involves the robot holding an object, parking would alter
  the state being graded — the decision belongs to the embodiment/config,
  not the framework). Failures degrade — one
  `warnings.warn(RuntimeWarning)` (the `eval.py` degrade precedent,
  assertable with `pytest.warns`), grading proceeds on the last step's
  frames — with one carve-out: `SafetyAbort` and `EmbodimentFault`
  re-raise. The repo invariant is that those two classes always halt the
  eval, and the park is real robot motion; reducing an embodiment-raised
  halt to a note would let the run drive hardware again at the next
  trial's reset.
- **Call site and conditions** (`eval.py`, in the scored-trial branch,
  immediately before `before_scoring(record, scene)`): call only when
  `before_scoring is not None` (no grader, no pointless motion), when
  `record.operator_judgement is None` (a console verdict already recorded
  means `grade()` will return without looking at frames), and when the
  trial is not definitively terminated — mirror `grade()`'s shortcut with
  the identical check (`record.terminated and record.termination_reason in
  _DEFINITIVE_REASONS`, imported from `session`), because an
  embodiment-definitive termination would otherwise trigger a real park
  motion whose observation is guaranteed unused. The attribute is fetched
  with `getattr` and guarded by `callable()` (the `bind_task` precedent,
  including its non-callable-attribute test). A return value that is
  neither `Observation` nor `None` is rejected at the call site with a
  precise degrade note (`isinstance` check) instead of surfacing later as
  a misleading `AttributeError` inside `grade()`'s broad except.
- **Record carriage**: new `TrialRecord` field
  `parked_observation: Observation | None = None` (`rollout.py`).
  In-memory only: `EvalLog` embeds `SceneResult` aggregates, never
  `TrialRecord`, so no `asdict` path reaches the numpy images and the log
  schema is untouched. Comment on the field states exactly that invariant.
- **Grader preference** (`grader.py`): a `_parked_frames(record,
  max_cameras)` helper returns `[(camera, image), ...]` from
  `record.parked_observation.images` with the same sorted-name order and
  camera cap as `_phase_frames`, or `[]` when the field is `None` or has no
  images. In `grade()`, the final frames become `_parked_frames(...) or
  _phase_frames(record.steps[-1], final=True, ...)`, and the initial frames
  become `_phase_frames(record.steps[0], ...) if record.steps else []` —
  parked frames make a zero-step scored trial (operator Esc at t=0)
  gradeable through the existing final-frames-only prompt path for the
  first time, and without the guard `record.steps[0]` would `IndexError`
  into the broad except and degrade. The existing "no final frames
  recorded" `ConfigError` still fires only when both final sources are
  empty. Prompt text unchanged ("after the trial ended" remains true).
- **Audit marker**: nothing persists the grading request, and the parked
  frames are deliberately not serialized, so the HTML/rerun views can show
  last-step frames that contradict a verdict judged on parked ones. When
  the grader uses parked frames it records the JSON-safe marker
  `record.metadata["graded_frames"] = "parked"`, which flows into
  `SceneResult.trial_metadata` and makes the provenance visible in the
  log.

## Changes

- `src/inspect_robots/rollout.py`: the `TrialRecord` field plus its
  invariant comment.
- `src/inspect_robots/eval.py`: the guarded hook call plus RuntimeWarning
  degrade and the SafetyAbort/EmbodimentFault re-raise; document the hook
  contract in the `eval()` docstring beside the other duck-typed hooks.
- `src/inspect_robots/grader.py`: `_parked_frames` and the preference in
  `grade()`.
- Docs: no `docs/` page enumerates the duck-typed embodiment hooks
  (verified), so the `eval()` docstring is the documentation of record.
  `src/inspect_robots/CLAUDE.md`'s `embodiment.py` row does enumerate
  `bind_task` and gains `observe_parked` beside it.

## Tests (100% coverage gate)

- Eval loop: hook called exactly once for a scored trial with a grader;
  not called without a grader; not called when an operator judgement
  already exists; not called on a definitive termination; a raising hook
  warns (pytest.warns RuntimeWarning), leaves `parked_observation` as
  `None`, and the trial still grades and scores; `SafetyAbort` and
  `EmbodimentFault` from the hook halt the eval; a `None` return leaves
  the field `None` without a warning; a non-Observation return degrades
  with a precise warning; a non-callable `observe_parked` attribute is
  ignored.
- Grader: with a parked observation present, the request contains the
  parked images (assert via the injected `http_post`) and not the last
  step's, and `record.metadata["graded_frames"] == "parked"`; the camera
  cap and sorted order match `_phase_frames`; with the field `None` or
  images empty, the last step's frames are used and no marker is written;
  both empty still raises the existing `ConfigError` path; a zero-step
  trial with parked frames grades through the final-frames-only prompt.
- `TrialRecord`: field defaults to `None`; existing construction sites
  unaffected.
- API snapshot: unchanged (`__all__` untouched); mypy strict and ruff D1
  clean.

## Out of scope

- The YAM implementation (inspect-robots-yam #126, plan 0023 there): ramp
  to the close()-style park target, capture via `_observe`, gated by a
  `park_before_grade` config flag. Ships independently — the hook is
  duck-typed, so either side without the other is a no-op.
- Serializing the parked frames into the EvalLog or FrameStore.
- Any change to scorers other than the vlm grader.
