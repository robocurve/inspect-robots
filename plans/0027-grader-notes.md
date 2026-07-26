# 0027 — grader notes: one optional free-text line after the operator verdict

Issue: #174. Status: draft (revised after critique rounds 1 and 2).

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
- `EOFError` on the notes prompt yields `None`: the verdict already accepted is
  still recorded, only the note is dropped. (This is deliberately *not* what
  EOF on the verdict prompt does — see below — because by then there is a
  verdict worth keeping.)
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
| `rollout.TrialRecord` | new `operator_note: str | None = None`, next to `operator_judgement`, with its own comment stating it is qualitative and read by nothing that scores (the block at `rollout.py:91-93` is the model) |
| `transcript.operator_event` | new trailing keyword `note: str | None = None` (after `source`, so the positional call at `tests/test_coverage_completion.py:309` still binds), always present in `data` |
| `cli._prompt_operator` | docstring extended: it currently states the verdict contract only (`cli.py:529-533`) and must state the notes contract too |
| `eval.py` trial loop | three edits, all three required: declare `notes: list[str | None] = []` at `eval.py:316-319` beside `judgements`; append `record.operator_note` beside each `judgements.append(...)` inside both arms of the status split (`None` in the non-success arm); pass `operator_notes=tuple(notes)` in the `SceneResult(...)` at `eval.py:443-455` |
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

`operator_event` gains `note` as a keyword that is always present in `data`,
even when `None`. Singular, because an event describes one trial, matching the
`verdict` key it sits beside and `TrialRecord.operator_note`. Always present,
because that is this module's existing convention:
`approval_event(t, modified, detail=None)` writes `"detail": None` rather than
varying the dict's keys (`transcript.py:47-49`). The cost is three exact-dict
assertions in the CLI tests, listed under Testing.

The event is kept rather than cut, even though nothing persists it (see below).
An event stream that records "the operator said `y`" while the note the same
human typed in the same breath is invisible to it would be a transcript that
lies by omission, and the transcript is the thing custom sinks and any future
event persistence read. The symmetry is worth three test edits.

The event is appended when there is a verdict to record **or** a note to
record:

- `y`/`n`/`partial`: appended as today, now carrying `note` (possibly `None`).
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
and operator judgements (`_html.py:475`, judgement block at `:507-517`). Notes
are free text, not tokens, so a chip row is the wrong shape: they get their own
block under the judgement row, one entry per trial that has a note, labelled
with its trial index so a note is readable against the parallel chip rows above
it.

The skip condition is where this block must **not** copy its neighbours. The
judgement block skips only on an empty tuple (`if not judgements`), and an
all-`None` tuple still renders a row of `n/a` chips. `operator_notes` is
strictly parallel to `epochs`, so the ordinary no-notes run produces `(None,)`
or `(None, None, ...)`, never `()`. A `if not scene.operator_notes` test would
therefore emit an empty "Grader notes" heading on nearly every page. The
predicate is "has at least one non-`None` entry", and trials whose entry is
`None` contribute no row at all — a missing note is not worth an `n/a` when
most trials will not have one.

Markup and style are specified here rather than left to taste, because
`_html.py` carries one fixed inline stylesheet (`_STYLES`, `:49-189`) and an
invented class renders unstyled. Emit `<h3>Grader notes</h3>` followed by one
`<div class="grader-note"><span class="note-label">trial {index}</span>{note}</div>`
per non-`None` entry, split across two implicitly concatenated f-strings so the
literal stays under ruff's 100-column limit at the indentation it lands at
(`pyproject.toml:89`; `ruff format` will not split a long string for you). The
`<h3>` says "Grader notes" rather than "Operator notes" even though the field
is `operator_note`: the heading mirrors the wording the human actually saw at
the prompt. Add a `.grader-note` rule to that stylesheet beside
`.agent-note`: neutral, using the existing `--line`/`--muted`/`--bg` custom
properties, with `overflow-wrap: anywhere` so a long note cannot widen the
card. The existing `.note-label` (`:179-183`) is reused for the trial index.

Do not reuse `.agent-note` itself (`:175-178`, emitted at `:295`). It is the
amber callout for an LLM policy's own tool-call notes; rendering a human
grader's note in it, under a label reading "agent note", would attribute the
sentence to the wrong author.

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
  (`tests/test_registry_cli.py:2047`) drives `answers = iter(["yse", "y"])` at
  `:2053`. The notes prompt draws a third value and raises `StopIteration`,
  which escapes through `before_scoring` (`eval.py:386`, outside the trial
  error handlers, and `main()` catches only `KeyboardInterrupt` around `eval()`
  at `cli.py:948`) and fails the run. Extend the iterator with the note answer.
- `test_prompt_operator_still_prompts_without_definitive_verdict` (`:2200`)
  asserts `len(prompts) == 1` (`:2225`) and the exact event dict (`:2228`). Its
  `_answer` fake (`:2218-2222`) returns `"y"` unconditionally, so leaving it
  alone would silently record `operator_note == "y"` and force the dict to
  `{"verdict": "y", "source": "prompt", "note": "y"}` — fixture nonsense that
  reads like intended behavior. Drive it from the sequence `["y", ""]`, assert
  both prompt strings, and assert `note` is `None`.
- `test_prompt_operator_prompts_for_truncated_success_reason` (`:2231`) patches
  `input` to the constant `lambda _prompt: "n"` (`:2246`) and asserts the exact
  event dict (`:2252`). Same failure: `"n"` becomes the note. Give it a
  sequence.
- `test_prompt_operator_warns_before_judging_step_limited_trial` (`:2255`,
  fake at `:2269`) is the quiet one. It asserts only `operator_judgement`, so
  it keeps passing while recording `operator_note == "n"` for a trial it never
  meant to annotate. Convert it too: a fake that answers a prompt the test does
  not know exists is a fake that will mislead the next reader.
- The three exact-dict event assertions that gain the `note` key are therefore
  `:2188` (the parametrized adoption test, where `note` is `None` and no prompt
  fires), `:2228`, and `:2252`.
- `test_prompt_operator_unit_semantics` (`:2307-2344`) patches `input` to
  constant lambdas at `:2321` (`"Partial"`) and `:2331` (`"skip"`), so the
  notes prompt would return those strings as note text and the skip case's
  `assert record.events == []` (`:2334`) would fail for a reason that looks
  like correct behavior. Convert those two cases to explicit answer sequences.
  Its third case (`:2336-2344`) raises `EOFError` on every call and needs no
  sequence; it is the natural home for the "notes prompt never reached" call
  counter listed below.

### New tests

CLI prompt (`tests/test_registry_cli.py`, next to the tests above):

- verdict `y` plus a typed note: judgement `"y"`, `operator_note` set, one
  operator event carrying both.
- bare Enter at the notes prompt: `operator_note is None`, event `note` key
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
  through the ad-hoc path answering verdict and note, reusing that test's
  `--max-steps 3` shape so `cubepick` does not terminate with a definitive
  verdict and get adopted without ever prompting, then read the log back
  and assert `log.samples[0].operator_notes == ("<the note>",)`. This is the
  only test that catches a `SceneResult(...)` constructed without the new
  tuple (`eval.py:443-455`), which is the single most likely implementation
  slip.

Log, orchestration, view:

- `tests/test_eval_log.py`: round-trip a log with notes; add `operator_notes`
  to the field-deletion test whose `del`s sit at `:107-113`, so a dict without
  the key still reads back as `()`.
- `tests/test_eval_orchestration.py`: `operator_notes` is parallel to `epochs`,
  with `None` for an errored trial.
- `tests/test_strict_json.py`: the new key survives the strict JSON sink.
- `tests/test_html_view.py`, with the fixtures pinned deliberately, because the
  100% gate cannot tell the two candidate skip conditions apart on its own: a
  `"".join(...)` over all-`None` entries is `""` either way. The "renders
  notes" case must mix one real note with one `None` so the per-entry filter is
  exercised in both directions, and the "no notes block" case must use
  `operator_notes=(None, None)` rather than the empty tuple that
  `test_absent_optional_fields_and_empty_scene_sequences_are_omitted` uses at
  `:162`. An empty tuple would pass against the wrong implementation.
- `tests/test_html_view.py:182-209`
  (`test_every_foreign_text_surface_is_escaped_exactly_once`) is the repo's
  guard for the escaping claim above, and its `dataclasses.replace` fixture
  (`:194-200`) does not set `operator_notes` — so it stays green against an
  implementation that interpolates a note raw, and the coverage gate does not
  care because the unescaped line still runs. Add `operator_notes=(attack,
  None)` to that fixture. The `>= 8` count at `:208` is a lower bound and needs
  no edit.

Docs and public text:

- `docs/guide/cli.md` gains the notes line in the transcript sample and a
  sentence on the Enter-to-skip contract. It also has a line to *correct*:
  `:101` currently reads that `skip` records nothing, which stops being true
  once a skipped trial can carry a note. An added sentence next to an
  uncorrected one leaves the page contradicting itself.
- `--no-prompt`'s help string (`cli.py:235`, "never ask the terminal operator
  for a success verdict") no longer describes everything the flag suppresses.
  Behavior is unchanged; the sentence is not.
- `CHANGELOG.md` gets an `Added` entry under `[Unreleased]`.
- All of the above is public-facing text under the `CLAUDE.md` writing-style
  rule: no em dashes in prose, no mid-sentence bold. Do not copy this plan's
  voice into them.

## Reversals after code review

Two decisions above were changed once the code existed, recorded here so the
plan does not describe something the repo no longer does:

- **`eval.py` collection point.** The draft put the `notes` append at the
  fall-through point beside `termination_reasons`, arguing for the smaller
  diff. Review pointed out that this captures the note *after*
  `policy.on_trial_end` runs while the judgement is captured *before* it, so a
  policy hook that set `operator_note` would be honoured while the same hook
  setting `operator_judgement` would be silently dropped. Two fields documented
  as strictly parallel must be captured by the same rule, so both appends now
  sit together inside the status split.
- **Notes prompt wording.** `grader notes (Enter to skip): ` became
  `grader notes (Enter for none): `. "Skip" is a literal verdict token one
  prompt earlier, and an operator who typed `skip` at the notes prompt would
  have recorded the word rather than skipping anything.

Kept unchanged under review, with the reasoning strengthened rather than the
behavior: the `skip`-with-note event (`operator_event`'s docstring now states
that `verdict` may be the non-verdict `"skip"`, which is what its consumers
need), and no control-character filtering of the note (rewriting what a human
typed is a worse failure than a stray glyph the browser already replaces).

## Rollout

One PR. Additive at every layer, defaulted everywhere, no schema bump, no
behavior change for unattended runs — nothing to stage or flag.
