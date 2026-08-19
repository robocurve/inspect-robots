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
  not the framework). Exceptions degrade: one stderr note, grading proceeds
  on the last step's frames.
- **Call site and conditions** (`eval.py`, in the scored-trial branch,
  immediately before `before_scoring(record, scene)`): call only when
  `before_scoring is not None` (no grader, no pointless motion) and
  `record.operator_judgement is None` (a console verdict already recorded
  means `grade()` will return without looking at frames). The
  definitive-termination shortcut inside `grade()` cannot be seen from the
  eval loop; that rarer case pays one harmless park-and-capture.
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
  _phase_frames(record.steps[-1], final=True, ...)`. The existing
  "no final frames recorded" `ConfigError` still fires only when both
  sources are empty. Prompt text unchanged ("after the trial ended" remains
  true). The grader transcript (the chat request) already embeds whatever
  frames were sent, so the judged images stay auditable without any log
  change.

## Changes

- `src/inspect_robots/rollout.py`: the `TrialRecord` field plus its
  invariant comment.
- `src/inspect_robots/eval.py`: the guarded hook call plus stderr degrade;
  document the hook contract in the `eval()` docstring beside the other
  duck-typed hooks.
- `src/inspect_robots/grader.py`: `_parked_frames` and the preference in
  `grade()`.
- Docs: if a docs page enumerates the embodiment's optional duck-typed
  hooks, add `observe_parked` there; otherwise the `eval()` docstring is
  the documentation of record.

## Tests (100% coverage gate)

- Eval loop: hook called exactly once for a scored trial with a grader;
  not called without a grader; not called when an operator judgement
  already exists; a raising hook prints the stderr note, leaves
  `parked_observation` as `None`, and the trial still grades and scores; a
  `None` return leaves the field `None` without a note.
- Grader: with a parked observation present, the request contains the
  parked images (assert via the injected `http_post`) and not the last
  step's; the camera cap and sorted order match `_phase_frames`; with the
  field `None` or images empty, the last step's frames are used; both
  empty still raises the existing `ConfigError` path.
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
