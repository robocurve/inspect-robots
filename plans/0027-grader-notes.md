# 0027 — grader notes: one optional free-text line after the operator verdict

Issue: #174. Status: draft.

## Problem

An ad-hoc interactive run ends each trial at `_prompt_operator`
(`cli.py:528`), which asks `did the robot succeed? [y/n/partial/skip]` and
stores the answer on `TrialRecord.operator_judgement`. That token is the whole
of what the log keeps from the human who watched the robot.

Everything else the grader saw is dropped: the gripper closed two frames early,
the cube had already rolled out of frame, the arm clipped the table on
approach, camera 2 was unplugged so the trial is unusable rather than failed.
That qualitative read is the part a human is uniquely good at producing, and it
is what someone reading the log later needs in order to interpret an `n` — or
to decide the trial should not have counted at all.

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
`None` on skip, unchanged; only `operator_notes` is populated. Prompting and
then dropping what was typed would be silent data loss, which is the one
outcome this feature must not have.

**The adopted-verdict path stays prompt-free.** When the embodiment terminated
with a definitive `success`/`failure`, `_prompt_operator` announces the adoption
and returns without asking anything (`cli.py:537-544`). Its whole point is that
an attended run of a self-scoring embodiment costs zero keypresses per trial;
adding a notes prompt there would reintroduce the keypress this branch exists to
remove. A grader who wants to flag a mistaken adoption is not served by this
plan. If that need turns out to be real, it is a separate change with its own
argument, not a rider on this one.

### EOF on the verdict prompt

Today the `EOFError` handler assigns `answer = "skip"` and the `skip` branch
then returns without recording. With notes in the picture that path must not
fall through to a second `input()` on an already-closed stdin, so the handler
returns immediately instead. Behavior is identical to today (no judgement, no
event, no exception); only the route to it is shorter.

### Persistence, mirroring the verdict exactly

| Layer | Change |
|---|---|
| `rollout.TrialRecord` | new `operator_notes: str | None = None`, documented as qualitative-only, next to `operator_judgement` |
| `transcript.operator_event` | new keyword `notes: str | None = None`, always present in `data` so the event shape is uniform |
| `eval.py` trial loop | collect `record.operator_notes` into a `notes` list beside `judgements`, `None` for errored trials (they are never scored and never prompted) |
| `log.SceneResult` | new `operator_notes: tuple[str | None, ...] = ()`, strictly parallel to `epochs`, documented like its siblings |
| `log.EvalLog.from_dict` | `sample["operator_notes"] = tuple(sample.get("operator_notes", ()))` — the `.get` is what keeps logs written before this field readable |
| `_html.py` | render the notes for a scene under the existing judgement/reason blocks |

`SCHEMA_VERSION` does not change. A defaulted field read through `.get` is the
same additive move `operator_judgements`, `termination_reasons`, and
`policy_transcripts` each made; an older reader ignores the key and a newer
reader supplies the default.

### Event recording

The operator event is appended when there is a verdict to record **or** a note
to record:

- `y`/`n`/`partial`: appended as today, now carrying `notes` (possibly `None`).
- `skip` with a note: appended with `verdict="skip"` and the note, so the
  transcript explains the gap. `operator_judgement` stays `None`.
- `skip` with no note: nothing appended, exactly as today.

### HTML view

`_render_scene` already emits chip rows for trial scores, termination reasons,
and operator judgements (`_html.py:495-517`). Notes are free text, not tokens,
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

CLI prompt (`tests/test_registry_cli.py`, alongside the existing
`_prompt_operator` tests):

- verdict `y` plus a typed note: judgement `"y"`, `operator_notes` set, one
  operator event carrying both.
- bare Enter at the notes prompt: `operator_notes is None`, event `notes` key
  present and `None`.
- whitespace-only note: treated as empty.
- note text is not lowercased and is stripped at both ends.
- `skip` plus a note: `operator_judgement is None`, `operator_notes` set, event
  appended with `verdict="skip"`.
- `skip` plus bare Enter: no judgement, no notes, no event (today's behavior).
- `EOFError` on the verdict prompt: nothing recorded, notes prompt never
  reached (assert the fake input saw exactly one call).
- `EOFError` on the notes prompt: verdict recorded, `operator_notes is None`.
- adopted-verdict path: unchanged, and the fake input is never called.

Log and orchestration:

- `tests/test_eval_log.py`: round-trip a log with notes; a dict with the key
  deleted still reads back with `operator_notes == ()`.
- `tests/test_eval_orchestration.py`: `operator_notes` is parallel to `epochs`,
  with `None` for an errored trial.
- `tests/test_strict_json.py`: the new key survives the strict JSON sink.
- `tests/test_html_view.py`: a scene with notes renders them escaped; a scene
  with only `None` entries renders no notes block.

Docs: `docs/guide/cli.md` gains the notes line in the transcript sample and a
sentence on the Enter-to-skip contract and the skip-plus-note case. `CHANGELOG.md`
gets an `Added` entry under `[Unreleased]`.

## Rollout

One PR. Additive at every layer, defaulted everywhere, no schema bump, no
behavior change for unattended runs — nothing to stage or flag.
