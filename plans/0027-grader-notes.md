# 0027 — grader notes: one optional free-text line after the operator verdict

Issue: #174. Status: draft (revised after critique round 1).

## Problem

An ad-hoc interactive run ends each trial at `_prompt_operator`
(`cli.py:528`), which asks `did the robot succeed? [y/n/partial/skip]` and
stores the answer on `TrialRecord.operator_judgement`. That token is the whole
of what the log keeps from the human who watched the robot.

Everything else the grader saw is dropped: the gripper closed two frames early,
the cube had already rolled out of frame, the arm clipped the table on
approach, camera 2 was unplugged so the trial is unusable rather than failed.
That qualitative read is the part a human is uniquely good at producing, and it
is what someone reading the log later needs in order to interpret an `n`.

Scope limit, stated up front so the Design is not read as promising more than
it does: this plan gives that context a place to live. It does not give it
teeth. A trial the grader considers unusable still scores exactly as it does
today (`skip` → `Score(value=False, explanation="no operator judgement
recorded")`, `scorer.py:243-249`) and still counts in the `operator` metric.
Excluding a trial from the numbers is a scoring change, needs its own argument
about what an eval's denominator means, and is out of scope here.

The verdict has a full persistence path already (`TrialRecord` → parallel
`SceneResult.operator_judgements` → JSON log → HTML view). Notes have none, so
"let the grader type something" is only half the work: without the same path,
the text dies with the terminal session.

## Design

One optional prompt, immediately after the verdict prompt, on the same code
path. Typing text records it; pressing Enter on an empty line records nothing.
Notes ride the existing verdict plumbing, field for field.

```
did the robot succeed? [y/n/partial/skip] (partial scores as failure) n
grader notes (Enter to skip): gripper closed early, cube still in frame
```

### Prompt contract

- `_NOTES_PROMPT = "grader notes (Enter to skip): "`, asked once, after the
  verdict loop accepts an answer.
- The answer is `.strip()`ed. Empty (bare Enter) or whitespace-only means no
  notes: the field stays `None`. Any other content is stored verbatim after
  stripping, case preserved — unlike the verdict, notes are never lowercased.
- One line. `input()` reads to the first newline, so a multi-line editor,
  raw-key capture, and any termios/tty handling are out of scope. The prompt is
  reached only under `sys.stdin.isatty()` (`cli.py:911`), so line editing comes
  from the terminal for free.
- Whitespace-only input collapsing to "no notes" is a deliberate narrowing of
  the request's literal "any char aside from Enter". A lone space is a slip,
  not a note, and storing `" "` would put an empty chip in the HTML view for
  no one's benefit.
- `EOFError` on the notes prompt yields `None` — the same defensive treatment
  the verdict prompt already has, for a stdin that closes mid-run.
- No re-prompting and no validation: every string is a legal note.

### When the prompt appears

| Verdict path | Verdict prompt | Notes prompt |
|---|---|---|
| `y` / `n` / `partial` | asked | asked |
| `skip` | asked | asked |
| `EOFError` on the verdict prompt | closed stdin | not asked |
| Adopted from a definitive embodiment verdict | not asked | not asked |

Two of those rows are deliberate and worth stating outright.

**`skip` still gets a notes prompt, and a note on a skipped trial is
recorded.** Skip means "record no judgement", not "record nothing" — "camera
unplugged, rerun this" is the single most valuable note a grader writes, and it
belongs to exactly the trial they refused to grade. `operator_judgement` stays
`None` on skip, unchanged; only `operator_note` is populated. Prompting and
then dropping what was typed would be silent data loss, which is the one
outcome this feature must not have.

**The adopted-verdict path stays prompt-free.** When the embodiment terminated
with a definitive `success`/`failure`, `_prompt_operator` announces the adoption
and returns without asking anything (`cli.py:537-544`). The distinction is not
"one keypress is too many" — it is that this branch takes *zero* input, so a
notes prompt would convert a run that needs no human at the keyboard into one
that does, trial after trial, including when the operator has walked away. On
the prompted path the human is already at the keyboard answering, so one more
Enter is an increment on an interaction that exists, not a new obligation. A
grader who wants to flag a mistaken adoption is not served by this plan; if
that need turns out to be real, it is a separate change with its own argument,
not a rider on this one.

### Why a second prompt and not one line

The alternative is folding notes into the verdict line and splitting on the
first space (`n gripper closed early`), which costs no extra keypress. Rejected
for two reasons:

- It changes the verdict contract. `_PROMPT_ANSWERS` is an exact-match set and
  anything outside it re-prompts (`cli.py:552-554`); splitting the line means a
  typo'd verdict with trailing words (`ye gripper...`) has to be diagnosed
  against a note the parser cannot distinguish from a mistake. The strict
  matcher is the reason the current prompt is hard to answer wrong.
- Nobody would find it. A trailing free-text field on an existing line is
  invisible unless the prompt string advertises it, and the prompt is already
  carrying `[y/n/partial/skip] (partial scores as failure)`. `grader notes
  (Enter to skip):` teaches the feature exists to every operator who runs one
  trial, which for an opt-in habit like note-taking is the whole ballgame.

### EOF on the verdict prompt

Today the `EOFError` handler assigns `answer = "skip"` and the `skip` branch
then returns without recording. With notes in the picture that path must not
fall through to a second `input()` on an already-closed stdin, so the handler
returns immediately instead. Behavior is identical to today (no judgement, no
event, no exception); only the route to it is shorter.

### Persistence, mirroring the verdict exactly

Singular on the trial, plural on the scene, matching every sibling pair in the
codebase (`operator_judgement`/`operator_judgements`,
`termination_reason`/`termination_reasons`,
`policy_transcript`/`policy_transcripts`). Reusing one name for both a `str`
and a `tuple[str | None, ...]` would make `eval.py`'s collector read as a type
error.

| Layer | Change |
|---|---|
| `rollout.TrialRecord` | new `operator_note: str | None = None`, documented as qualitative-only, next to `operator_judgement` |
| `transcript.operator_event` | new keyword `notes: str | None = None`, always present in `data` |
| `eval.py` trial loop | collect `record.operator_note` into a `notes` list beside `judgements`, `None` for errored trials (they are never scored and never prompted) |
| `log.SceneResult` | new `operator_notes: tuple[str | None, ...] = ()`, strictly parallel to `epochs`, documented like its siblings |
| `log.EvalLog.from_dict` | `sample["operator_notes"] = tuple(sample.get("operator_notes", ()))` — the `.get` is what keeps logs written before this field readable |
| `_html.py` | render the notes for a scene under the existing judgement/reason blocks |

`SCHEMA_VERSION` does not change. A defaulted field read through `.get` is the
same additive move `operator_judgements`, `termination_reasons`, and
`policy_transcripts` each made, and it buys the guarantee the log module
actually states: newer code reads older logs (`log.py:4-6`). It does not buy
the reverse — `from_dict` ends in `SceneResult(**sample)`, so an *older* build
handed a log with this key raises `TypeError`. That has always been true of
this schema's additive fields and is not a reason to bump the version.

### Event recording

`operator_event` gains `notes` as a keyword that is always present in `data`,
even when `None`. That is this module's existing convention:
`approval_event(t, modified, detail=None)` writes `"detail": None` rather than
varying the dict's keys (`transcript.py:47-49`). The cost is three exact-dict
assertions in the CLI tests, listed under Testing.

The event is appended when there is a verdict to record **or** a note to
record:

- `y`/`n`/`partial`: appended as today, now carrying `notes` (possibly `None`).
- `skip` with a note: appended with `verdict="skip"` and the note.
  `operator_judgement` stays `None`.
- `skip` with no note: nothing appended, exactly as today.

The middle case deliberately breaks a current invariant: today an operator
event exists if and only if a judgement was recorded. After this change an
event can carry `verdict="skip"` for a trial whose judgement is `None`. That is
the honest encoding of "the human said something about this trial but declined
to grade it", and a consumer that reads `verdict` without checking for `"skip"`
was already wrong (`skip` has never been a judgement).

Worth being precise about what this event buys, since the plan should not
oversell it: `TrialRecord.events` is not persisted anywhere. `SceneResult` and
`EvalLog` carry no events field, `JsonLogSink.on_trial_end` is a no-op, and
`RerunSink.on_trial_end` only flushes. The event is for in-memory consumers and
custom sinks, and it is there for symmetry with the verdict. The note reaches
disk solely through `SceneResult.operator_notes`, which is why the end-to-end
test below is the one that matters.

### HTML view

`_scene_section` already emits chip rows for trial scores, termination reasons,
and operator judgements (`_html.py:475`, judgement block at `:508-517`). Notes
are free text, not tokens,
so a chip row is the wrong shape: they get their own block under the judgement
row, one entry per trial that has a note, labelled with its trial index so a
note is readable against the parallel chip rows above it. Scenes with no notes
at all emit nothing, the same way an empty judgement tuple emits nothing.

Text goes through the module's `_escape` at interpolation, like every other
foreign string on the page. No truncation: a note is one operator-typed line,
and the page already carries a shared payload budget for the parts that can
actually get large (frames, transcripts).

### What does not change

- **Scoring.** `_OperatorScorer` reads `operator_judgement` and nothing else
  (`scorer.py:246`). Notes are qualitative; a note must never move a number,
  and no scorer, metric, or reducer learns about this field.
- **Unattended runs.** The prompt is behind `is_adhoc and not --no-prompt and
  sys.stdin.isatty()` (`cli.py:907-916`); CI and headless runs still see
  neither prompt.
- **`inspect-robots inspect`.** It does not print operator judgements today, so
  it does not gain notes here.
- **Core dependencies.** Nothing new; `input()` is stdlib.

## Testing

The repo gates at 100% coverage, so every branch below is a test.

### Existing tests this breaks

A second `input()` call per prompted trial invalidates fakes that were written
for exactly one. These are updates, not new cases, and each must be made
deliberately rather than by making assertions looser:

- `test_operator_prompt_records_verdict_and_reprompts_on_typos`
  (`tests/test_registry_cli.py:2047`) drives `answers = iter(["yse", "y"])`.
  The notes prompt draws a third value and raises `StopIteration`, which
  escapes through `before_scoring` (`eval.py:387`, outside the trial error
  handlers) and fails the run. Extend the iterator with the note answer.
- `test_prompt_operator_still_prompts_without_definitive_verdict`
  (`tests/test_registry_cli.py:2202`) asserts `len(prompts) == 1`. It becomes
  2, and asserting both prompt strings is the point of the test now.
- Three exact-dict event assertions gain the `notes` key:
  `tests/test_registry_cli.py:2188` (`source="embodiment"` — `notes` is `None`
  there), `:2229`, and the parametrized adoption test's counterpart.
- `test_prompt_operator_unit_semantics` (`tests/test_registry_cli.py:2318-2344`)
  patches `input` to a constant lambda, so the notes prompt would return
  `"Partial"` and `"skip"` as note text and the skip case's
  `assert record.events == []` would fail for a reason that looks like correct
  behavior. Convert each case to an explicit answer sequence.

### New tests

CLI prompt (`tests/test_registry_cli.py`, next to the tests above):

- verdict `y` plus a typed note: judgement `"y"`, `operator_note` set, one
  operator event carrying both.
- bare Enter at the notes prompt: `operator_note is None`, event `notes` key
  present and `None`.
- whitespace-only note: treated as empty.
- note text is not lowercased and is stripped at both ends.
- `skip` plus a note: `operator_judgement is None`, `operator_note` set, event
  appended with `verdict="skip"`.
- `skip` plus bare Enter: no judgement, no note, no event (today's behavior).
- `EOFError` on the verdict prompt: nothing recorded, notes prompt never
  reached (assert the fake input saw exactly one call).
- `EOFError` on the notes prompt: verdict recorded, `operator_note is None`.
- adopted-verdict path: unchanged, and the fake input is never called.
- end to end, mirroring `tests/test_registry_cli.py:2062`: run `main([...])`
  through the ad-hoc path answering verdict and note, then read the log back
  and assert `log.samples[0].operator_notes == ("<the note>",)`. This is the
  only test that catches a `SceneResult(...)` constructed without the new
  tuple (`eval.py:443-455`), which is the single most likely implementation
  slip.

Log, orchestration, view:

- `tests/test_eval_log.py`: round-trip a log with notes; add `operator_notes`
  to the field-deletion test at `:106-119` so a dict without the key still
  reads back as `()`.
- `tests/test_eval_orchestration.py`: `operator_notes` is parallel to `epochs`,
  with `None` for an errored trial.
- `tests/test_strict_json.py`: the new key survives the strict JSON sink.
- `tests/test_html_view.py`: a scene with notes renders them escaped; a scene
  with only `None` entries renders no notes block.

Docs: `docs/guide/cli.md` gains the notes line in the transcript sample and a
sentence on the Enter-to-skip contract and the skip-plus-note case.
`CHANGELOG.md` gets an `Added` entry under `[Unreleased]`. Both are
public-facing text under the `CLAUDE.md` writing-style rule: no em dashes in
prose, no mid-sentence bold. Do not copy this plan's voice into them.

## Rollout

One PR. Additive at every layer, defaulted everywhere, no schema bump, no
behavior change for unattended runs — nothing to stage or flag.
