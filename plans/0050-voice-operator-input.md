# Operator voice mode: `--voice` streams spoken feedback to the policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attended runs already deliver typed operator feedback to the policy
(`OperatorConsole.poll()` → transcript `operator_message` event → delivered-once
`observation.extra["operator_messages"]` at the next inference). On a rig the operator's
hands are busy, so feedback arrives late or never. This plan adds an operator voice mode:
`inspect-robots run --voice` keeps the microphone open for the whole run, transcribes
speech locally (faster-whisper), and feeds each accepted utterance through the **existing**
operator-message path. Silence sends nothing — utterances are segmented by an energy gate,
filtered by VAD and Whisper-hallucination guards, and dropped unless the operator actually
said something. Voice is **feedback-only**: episode end and verdicts stay on the keyboard,
so a mistranscription can never terminate a trial. (Issue #313.)

**Shape:** one PR, two packages.

1. **Core seam (small, 100% covered):** `OperatorSession` learns to merge additional
   `OperatorInput` sources into its poll; `ConsolePoll` gains a per-message `sources`
   provenance tuple; `operator_message_event` gains a `source` field; the registry gains
   an `operator_input` kind with entry-point group `inspect_robots.operator_inputs`; the
   CLI gains `--voice` and repeatable `-V k=v` with loud pre-run guards.
2. **New workspace plugin `plugins/inspect-robots-voice/`:** package
   `inspect_robots_voice`, deps `sounddevice` + `faster-whisper` + NumPy (lazily imported
   at construction, never at module top), registered under the new group as `voice`.
   Capture thread → pure-NumPy energy-gate segmentation → faster-whisper transcription
   gauntlet → thread-safe queue drained by a non-blocking `poll()`.

**Tech Stack:** core stays stdlib + NumPy (no new core deps). Plugin: `sounddevice`
(PortAudio capture), `faster-whisper` (CTranslate2 Whisper + bundled Silero VAD), NumPy.
Python 3.10+.

## Global Constraints

- Core gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
  Plugin gates mirror the other plugin jobs: ruff + ruff format + mypy (plugin config) +
  pytest, own coverage scope (reported, not gated at 100%).
- D1 docstrings on public defs; state the contract, not the name. Line length 100.
- Core must never import `sounddevice`/`faster_whisper` (the `core-only-import` job
  guards this); the plugin must not import them at module top either, so its test suite
  runs on CI runners without PortAudio system libraries.
- Public API is fenced by `inspect_robots.__all__` **and** `tests/test_api_snapshot.py`
  — update both together (`operator_input` decorator becomes public).
- No behavior change without `--voice`: every existing run takes today's exact paths;
  existing tests pass untouched.
- `uv lock` after touching any pyproject; CI installs `--locked`.
- Writing-style rules for public-facing text (README, docs): no em dashes in prose, no
  decorative emoji, headers use colons.

## Reference: current wiring (main @ 6c022ead)

- `console.py`: `ConsolePoll(messages: tuple[str, ...], end: EndRequest | None)`;
  `OperatorInput` Protocol = `poll()` + `begin_trial()`; `OperatorConsole` is the
  threadless fd-level stdin poller.
- `session.py`: `OperatorSession` composes one `OperatorConsole`; `poll()`/`begin_trial()`
  delegate; owns `status()`, `write_line()`, `gate()`, verdict prompts.
- `rollout.py` (~line 290): polls `operator_input` once per control step; each message
  becomes `operator_message_event(t, text)` plus a `{"t": t, "text": text}` dict in the
  store; the undelivered tail rides `observation.extra["operator_messages"]` into the
  next inference; a poll exception disables the channel for the trial with a warning.
- `transcript.py:52`: `operator_message_event(t, text)` → `Event(kind="operator_message",
  data={"text": text})`.
- `cli.py:696`: `_attended()` = TTY and not `--no-prompt`. `cli.py:705`
  `_build_operator_session(policy, embodiment)` returns `(session, operator_input | None)`
  — `None` on win32, on policies without `accepts_operator_messages` (when the embodiment
  also lacks `connect_operator_session`), and on legacy hardware embodiments.
- `registry.py`: `KINDS`/`_GROUPS`/`register()`/per-kind decorators; entry points load
  factories directly (`ep.load()`), `resolve(kind, name, **kwargs)`.
- Agent plugin `policy.py` `_operator_lines()`: reads only `t` and `text` keys from each
  message dict — additive keys are ignored safely.
- Plugin CI pattern (`ci.yml`): one ubuntu/py3.11 job per plugin — `uv sync --locked
  --all-packages --extra dev`, then ruff/format/mypy/pytest scoped to the plugin; every
  job is listed in `ci-ok`'s `needs`.

## Design decisions (and why)

1. **Voice is a second `OperatorInput` source merged inside `OperatorSession`, not a
   composite built in the CLI and not policy-internal capture.** PR #309 made the session
   the single owner of attended-run operator I/O; a second input channel belongs to that
   owner. The session can echo each accepted utterance (`write_line("voice: ...")`) so
   the terminal shows exactly what the model will receive, and it can degrade
   **per source**: rollout's existing catch around `operator_input.poll()` disables the
   whole channel, so the session must swallow a voice-source failure itself (one yellow
   warning line, source detached, typed console stays alive). Policy-internal capture was
   rejected: it would bypass transcript events and work for one policy only.
2. **Feedback-only: `end` requests from attached sources are ignored** (documented in
   `attach_input`). The blast radius of a bad transcription is one extra message in the
   model's context. Episode end and verdicts remain keyboard-only.
3. **Provenance is an additive parallel tuple, `ConsolePoll.sources`, defaulting to `()`
   (= all console).** `messages: tuple[str, ...]` keeps its type, so `OperatorConsole`,
   rollout iteration, and every existing consumer are untouched; a frozen dataclass gains
   one defaulted field. Rollout tags each message's event and store dict with
   `sources[i]` when present, else `"console"`. The event payload becomes
   `{"text": ..., "source": ...}` and the policy-facing dict becomes
   `{"t": ..., "text": ..., "source": ...}` — additive keys, no `EvalLog` schema bump;
   readers of old logs see no `source` and treat it as console.
4. **New registry kind `operator_input`, entry-point group
   `inspect_robots.operator_inputs`.** Costs a few lines (KINDS, _GROUPS, one decorator)
   and keeps core honest: core never names the plugin, `inspect-robots list` shows what
   is installed, and third parties can ship alternative sources (push-to-talk pedal, a
   phone app) with zero core changes. The CLI resolves `resolve("operator_input",
   "voice", **voice_args)`.
5. **CLI surface: `--voice` plus repeatable `-V k=v`,** mirroring `-P`/`-E`/`-T`. Keys
   are passed through to the factory as strings; the factory coerces and **rejects
   unknown keys loudly** (typo protection). Supported keys: `model` (Whisper size or
   path, default `small`), `device` (sounddevice index or name substring, default system
   default), `language` (default `en`), `compute` (CTranslate2 compute type, default
   `auto`). Both `run` and `eval-set` get the flags (shared-args helper).
6. **All `--voice` guards fire before the run starts, loudly:** requires attended mode
   (TTY and no `--no-prompt`); requires the resolved live input channel
   (`_build_operator_session` returned a non-`None` operator input — this simultaneously
   covers win32, `accepts_operator_messages=False` policies on hook-less embodiments, and
   legacy hardware embodiments, reusing the session's own notices); requires the plugin
   installed (error carries `pip install inspect-robots-voice`). `-V` without `--voice`
   is an error. A voice *startup* failure (no mic, unknown device, model download
   failure) aborts before any trial, printing the sounddevice device list on device
   errors.
7. **Plugin internals: three seams and an orchestrator, all injectable.**
   - `_segmenter.py` — pure function of arrays, no I/O, the unit with real logic.
     Adaptive RMS energy gate: utterance opens when short-window energy exceeds
     `noise_floor x open_ratio` for >= 100 ms, closes after 700 ms below, 300 ms pre-roll
     prepended, 30 s hard cap (force-close and continue), noise floor adapts (EMA) during
     non-speech only.
   - `_transcriber.py` — faster-whisper wrapper plus the rejection gauntlet:
     `vad_filter=True` (bundled Silero VAD); reject empty/whitespace text,
     `no_speech_prob > 0.6`, `avg_logprob < -1.0`, utterances shorter than 0.4 s, and a
     blocklist of Whisper silence hallucinations ("Thank you.", "you", "Thanks for
     watching!", ...) when they are the entire transcription. Rejections vanish silently.
   - `_capture.py` — `sounddevice.InputStream` wrapper (16 kHz mono float32 blocks into a
     bounded queue; PortAudio callback thread). Device resolution: integer index or
     case-insensitive name substring; ambiguous or missing raises with the device table.
   - `_input.py` — `VoiceInput`: one worker thread drains the capture queue through the
     segmenter, closed utterances through the transcriber, accepted text into a
     thread-safe output deque; `poll()` drains non-blocking and returns
     `ConsolePoll(messages=..., sources=("voice", ...))` with `end=None`;
     `begin_trial()` clears queued utterances (pre-trial chatter never leaks in, same
     semantics as the console); `start()` loads the model, opens the stream, and returns
     a human-readable "listening on <device> (model=<m>)" line for the CLI to print;
     `close()` is idempotent, stops thread and stream, never raises.
   - Worker exceptions are captured and re-raised from the **next `poll()`** — that is
     precisely the path the session already handles (decision 1), so a dead voice
     pipeline degrades to one warning line and a detached source.
   - Backpressure: the capture queue is bounded (~30 s of audio); when full the oldest
     audio is dropped and a `warnings.warn` fires once per run. The control loop never
     waits on ASR (same contract as `RerunSink`: drop under pressure, never delay
     control).
8. **Lazy heavy imports.** `sounddevice` and `faster_whisper` are real dependencies of
   the plugin (users get a working install) but are imported inside `_capture`/
   `_transcriber` construction, never at module top. CI runners have no PortAudio system
   library; the plugin test suite injects fakes and never touches audio hardware or
   model downloads.
9. **Lifecycle owned by the CLI.** Construct after the guards pass; `start()` before the
   first trial (model load is one-time, seconds; the listening line prints via
   `session.write_line`); `close()` in the same `finally` that releases device claims and
   closes the embodiment. The mic stays open across trials and verdict prompts; whatever
   is said during a prompt is discarded by the next `begin_trial()`.
10. **Versioning:** plugin `inspect-robots-voice` 0.1.0 (static version, like all
    plugins) with `inspect-robots>=0.43` floor (the core release carrying the seam).
    Core bump is minor. The plugin needs a `publish-voice` job in `release.yml` and a
    PyPI trusted-publisher environment (repo-owner action; `skip-existing` makes it a
    no-op until configured).

## Out of scope (YAGNI)

Voice-triggered episode end/verdicts, wake words, push-to-talk, speaker diarization,
streaming/partial transcriptions, TTS back to the operator, per-utterance timestamps in
the model payload beyond the existing step index, Windows support beyond what
sounddevice provides (the `--voice` guard already excludes win32 via the live-channel
requirement).

## Tasks

### Task 1: transcript + rollout provenance (core)

- [ ] `transcript.py`: `operator_message_event(t: int, text: str, source: str =
  "console")` → `data={"text": text, "source": source}`. Update its docstring ("typed at
  the console or spoken in voice mode").
- [ ] `console.py`: add `sources: tuple[str, ...] = ()` to `ConsolePoll` (docstring:
  parallel to `messages` when non-empty; empty means all console).
- [ ] `rollout.py`: when appending messages, compute `source = poll.sources[i] if i <
  len(poll.sources) else "console"`; pass it to the event and include it in the store
  dict (`{"t": t, "text": text, "source": source}`).
- [ ] Tests: event carries source; default stays `"console"`; rollout round-trip with a
  fake `OperatorInput` returning `sources`; `EvalLog` write/read preserves the key;
  old-log read (no `source`) still works.

### Task 2: `OperatorSession.attach_input` (core)

- [ ] `session.py`: `attach_input(source: OperatorInput, *, label: str)` registers an
  additional source. `poll()` polls the console first, then each attached source; for
  each attached-source message, `write_line(f"{label}: {text}")` then append to
  `messages`/`sources` (console messages get source `"console"`); attached-source `end`
  is ignored (feedback-only, documented); a source whose `poll()`/`begin_trial()` raises
  is detached with one `write_line` warning (`"{label} input disabled after
  {type}: {exc}"`) and the console keeps working. `begin_trial()` fans out.
- [ ] Tests: merge order and sources tuple; echo lines; end ignored; per-source
  detachment on poll and on begin_trial; console-only behavior unchanged when nothing is
  attached (sources stays `()`).

### Task 3: registry kind `operator_input` (core)

- [ ] `registry.py`: add `"operator_input"` to `KINDS`, `"inspect_robots.
  operator_inputs"` to `_GROUPS`, and an `operator_input()` decorator; module docstring
  group list updated.
- [ ] `__init__.py` `__all__` + `tests/test_api_snapshot.py`: export `operator_input`.
- [ ] Confirm `inspect-robots list` renders the new kind (it iterates `KINDS`
  generically; adjust the test fixture list if it enumerates kinds).

### Task 4: CLI `--voice` / `-V` (core)

- [ ] Shared eval args: `--voice` (store_true) and `-V` (`dest="voice_args"`, append,
  `metavar="k=v"`), help text stating attended-only and the plugin requirement.
- [ ] Guard + build helper (shared by `run`/`eval-set`): error paths (each a
  `SystemExit` with a one-line reason) — `-V` without `--voice`; `--voice` without
  attended mode; `--voice` when the operator input channel is `None`; `--voice` when
  `resolve("operator_input", "voice", ...)` raises unknown-name (message includes
  `pip install inspect-robots-voice`).
- [ ] Wiring: parse `-V` with the existing k=v parser; `resolve(...)` the factory;
  `start()` before `eval()` and print the returned listening line via
  `session.write_line`; `session.attach_input(voice, label="voice")`; `close()` in the
  existing `finally`. Startup exceptions abort the run with the message as-is.
- [ ] Tests: full guard matrix; lifecycle (start printed, attach called, close called on
  success and on eval failure) with a fake registered operator input; `-V` parsing.

### Task 5: plugin package `plugins/inspect-robots-voice/` (scaffolding + segmenter)

- [ ] `pyproject.toml`: name `inspect-robots-voice`, version `0.1.0`, deps
  `inspect-robots>=0.43`, `numpy>=1.24`, `sounddevice>=0.4.6`, `faster-whisper>=1.0`;
  entry point `[project.entry-points."inspect_robots.operator_inputs"] voice =
  "inspect_robots_voice:voice_input"`; mypy/ruff config mirroring the agent plugin;
  README (writing-style rules apply).
- [ ] Add to the uv workspace; `uv lock`.
- [ ] `_segmenter.py`: `EnergyGate` per decision 7 (constructor takes sample rate,
  open ratio, hangover, pre-roll, min-open, max-utterance, EMA alpha; `push(block) ->
  ndarray | None`).
- [ ] Tests (pure NumPy, no hardware): silence yields nothing; a speech burst yields one
  utterance including pre-roll; a burst shorter than min-open yields nothing; hangover
  closes; 30 s cap force-closes and the stream continues; noise floor adapts on silence
  and freezes during speech.

### Task 6: plugin transcriber + capture

- [ ] `_transcriber.py`: `WhisperTranscriber(model, compute, language)`; lazy
  `faster_whisper` import at construction; `transcribe(audio) -> str | None` applying the
  full gauntlet of decision 7. The model call is a thin injectable seam so the gauntlet
  is tested against canned segment/info objects.
- [ ] `_capture.py`: `MicrophoneCapture(device, sample_rate, queue)`; lazy `sounddevice`
  import; device resolution (index, name substring, ambiguity/missing errors listing the
  device table); bounded queue with drop-oldest and once-per-run `warnings.warn`.
- [ ] Tests: gauntlet accept/reject matrix (empty, no_speech_prob, avg_logprob, too
  short, each blocklist phrase, a normal sentence); device resolution against a fake
  device table; drop-oldest and single warning.

### Task 7: plugin `VoiceInput` orchestration + factory

- [ ] `_input.py`: `VoiceInput` per decision 7 — worker thread, output deque, `poll()`
  (returns `ConsolePoll` with `sources`), `begin_trial()`, `start()` (loads model, opens
  stream, returns the listening line), `close()` (idempotent), worker-error capture
  re-raised on next `poll()`.
- [ ] `__init__.py`: `voice_input(**kwargs)` factory — string coercion, unknown-key
  rejection, docstring documenting the `-V` keys.
- [ ] Tests with fake capture + fake transcriber: utterances flow to `poll()`; silence
  sends nothing; `begin_trial()` clears; worker error surfaces on next poll; `close()`
  idempotent and joins the thread; factory coercion and unknown-key error.

### Task 8: CI + release plumbing

- [ ] `ci.yml`: `test-plugin-voice` job cloned from the xpolicylab pattern (ubuntu,
  py3.11, `uv sync --locked --all-packages --extra dev`, ruff/format/mypy/pytest scoped
  to the plugin); add it to `ci-ok.needs`.
- [ ] `release.yml`: `publish-voice` job cloned from the existing plugin publish jobs
  (`skip-existing: true`). Note in the PR body that the PyPI trusted-publisher
  environment must be created by the repo owner before the first publish takes effect.

### Task 9: docs + changelog + module maps

- [ ] `docs/`: a voice-mode page (what it does, install, `-V` keys, the silence
  gauntlet, feedback-only rationale); run CLI docs mention `--voice`/`-V`.
- [ ] Root `README.md` plugins list + root `CLAUDE.md` Layout bullet; package module map
  `src/inspect_robots/CLAUDE.md` rows for the touched modules; plugin `CLAUDE.md`.
- [ ] `CHANGELOG.md`: core minor entry (seam + CLI) and plugin 0.1.0 entry, linking
  issue #313 and this plan.
