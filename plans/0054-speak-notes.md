# `run --speak`: broadcast agent policy notes through the rig speaker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent-policy runs require a `note` on every move/capture tool call — one or two
operator-facing sentences about what the model sees and decides — and end trials with
`done(summary=...)` or `give_up(reason=...)`. Today that narration is visible only on
screens (transcript echo, the Rerun llm tab, saved logs). On a rig the operator's eyes
are on the robot, not the terminal. This plan adds `inspect-robots run --speak`: each
note (and each terminal `summary`/`reason`) is spoken through the machine's audio output
as it streams, so the operator hears the policy narrate itself while hands-busy or across
the room. Works unattended (no TTY requirement). **`run` only** — `eval-set` passes no
sinks into `eval_set()` (its signature takes none), and adding that plumbing would grow
public core API for no current need. (Issue #327.)

**Shape:** one PR, two packages.

1. **Core (CLI wiring only, 100% covered):** `--speak` plus repeatable `-S k=v` on `run`
   (declared on the run parser directly, NOT in `_add_shared_eval_args` — eval-set must
   not accept them); resolve the sink registered as `speaker` in the existing
   `inspect_robots.sinks` entry-point group; call its duck-typed `start()`; append it to
   the sinks list the eval loop already fans `log_policy_messages` to; `close()` on the
   way out. Missing-plugin errors hint `pip install inspect-robots-voice`. No new public
   API symbols, no rollout/eval changes.
2. **Voice plugin (`inspect-robots-voice` 0.4.0):** new `SpeakerSink` — extracts spoken
   text from assistant `tool_calls` in the live transcript delta, bounded drop-oldest
   queue, one daemon worker thread, local Kokoro TTS behind a small engine seam,
   playback through the already-depended-on `sounddevice`.

**Tech Stack:** core stays stdlib + NumPy. Plugin adds `kokoro-onnx` (pulls `onnxruntime`
CPU). `sounddevice` and NumPy are already plugin deps. Python 3.10+ (see design
decision 4 for the `<3.14` marker).

## Global Constraints

- Core gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
  Plugin gates mirror the existing `plugin-voice` CI job: ruff + ruff format + mypy
  (plugin config) + pytest, own coverage scope (reported, not gated at 100%).
- D1 docstrings on public defs; state the contract, not the name. Line length 100.
- Core must never import `kokoro_onnx`/`sounddevice` (the `core-only-import` job guards
  this); the plugin must not import them at module top either, so its test suite runs on
  CI runners without PortAudio system libraries (`kokoro-onnx` still installs in the
  locked env; importing it stays confined to `start()`-time code paths that tests replace
  with fakes).
- Public API is fenced by `inspect_robots.__all__` **and** `tests/test_api_snapshot.py` —
  this plan adds **no** public core symbol, so neither file changes; if an implementation
  detail forces one, stop and re-plan.
- No behavior change without `--speak`: every existing run takes today's exact paths;
  existing tests pass untouched.
- `uv lock` after touching any pyproject; CI installs `--locked`. Note: all `plugin-*`
  CI jobs sync `--all-packages`; `onnxruntime` is already in every plugin job via the
  voice plugin's `onnx-asr[cpu,hub]` dep, so `kokoro-onnx` adds only itself and its
  small phonemization deps — accepted cost, no CI changes.
- Writing-style rules for public-facing text (README, docs): no em dashes in prose, no
  decorative emoji, headers use colons.

## Reference: current wiring (main @ d32337f7)

- `logging/sink.py`: `LogSink` protocol (`on_eval_start` / `on_trial_start` / `log_step` /
  `on_trial_end` / `on_eval_end`) + `NullSink` base. Duck-typed extension
  `log_policy_messages(t, messages)` is called at most once per control step, only when
  the policy performed an inference; messages are plain-JSON-type dicts shaped like
  `TrialRecord.policy_transcript` entries; **core does not enforce the shape on this live
  path, so sinks must render defensively and must not mutate** (see the module docstring).
- `eval.py` `_Broadcast` (~line 112): fans `log_policy_messages` to every sink that
  defines it — a second consumer costs nothing. `on_eval_end` fires on errored,
  fail-on-error-halted, and even Ctrl-C-mid-rollout paths too (`eval.py` reaches
  `bus.on_eval_end` at ~line 607 before re-raising a cancellation); only Ctrl-C
  *outside* the rollout window can skip it — so an end-of-run drain must gate on
  `log.status` (terminal literals: `"success"`, `"error"`, `"cancelled"`), not on the
  hook firing.
  `on_trial_end` is the bounded-loss flush point (see the Rerun sink's docstring). The
  rollout delivers the final delta (the `done`/`give_up` message) and
  then breaks out of the trial; `bus.on_trial_end` follows within moments in unattended
  runs, and on the last trial the clean path reaches `bus.on_eval_end` shortly after —
  the speaker's lifecycle must not treat those hooks as "abort now" (see design
  decision 8).
- `rollout.py` (~line 253): `stream_ok = callable(policy.transcript_delta) and
  callable(sink.log_policy_messages)`; the agent policy's `transcript_delta` returns
  sanitized OpenAI-style dicts appended since the previous call.
- Spoken-text sources (agent plugin `_tools.py`): every move/capture tool schema requires
  a `note` string argument; `done` requires `["summary", "hindsight"]` (~line 214) and
  `give_up` requires `["reason", "hindsight"]` (~line 232) — **neither has a `note`
  key**. In transcript messages these appear as `{"role": "assistant", "tool_calls":
  [{"function": {"name": ..., "arguments": "<JSON string>"}}]}` —
  `rerun_sink.py::_render_message` is the defensive-parsing precedent.
- `registry.py`: kind `"sink"` already exists with entry-point group
  `inspect_robots.sinks` (line ~39); nothing to add in the registry. Unknown-name
  message shape: `no sink named ...` (registry.py ~line 136).
- `cli.py`: `--voice` / repeatable `-V k=v` precedent (lines ~197-206), `_parse_kvs`
  coercion, `_resolve_or_exit` + kind→flag hint map (~line 652), missing-plugin hint for
  operator_input (~line 643), `_build_voice_input` / `_start_voice_input` /
  `_close_voice_input` lifecycle helpers (~lines 793-844), the single sinks-assembly
  site in `_cmd_run` (~lines 1474-1494), embodiment claim inside `_resolve_components`
  (~line 1439) **before** the try/finally that closes voice.
- Voice plugin: `_input.py` `VoiceInput` shows the `start()`/`close()` duck-typed
  lifecycle and lazy heavy imports; `__init__.py::voice_input` shows the strict
  factory-kwarg validation pattern (allowed-set + per-key `TypeError`).
- CI: `plugin-voice` job and `publish-voice` release job already exist (PR #316/#325) —
  no new CI jobs needed.
- CLI tests precedent for `--voice`: `tests/test_registry_cli.py` (~line 3360+).

## Design decisions (and why)

1. **The speaker is a `LogSink`, not an agent-policy feature and not console scraping.**
   `log_policy_messages` is the existing live seam that already carries exactly the
   payload we need, policy-agnostically (any policy with `transcript_delta` benefits;
   today that is the agent policy). Baking TTS into the agent plugin would tie audio to
   one policy and drag audio deps into the wire package; speaking `[agent]` stderr echo
   lines would scrape unstructured text that is off by default.
2. **Home is the voice plugin, not a third package.** Both directions of rig audio live
   together: `sounddevice` is already a dep, the lazy-import and lifecycle conventions
   exist, and the CI/release plumbing is built. `--speak` and `--voice` stay independent:
   either flag works alone.
3. **Engine: Kokoro-82M via `kokoro-onnx`, behind an internal seam — chosen on quality
   at CPU speed, with eyes open on licensing.** Verified 2026-08: Kokoro-82M is the
   consensus best open-weight TTS that still runs faster than real-time on CPU (~6x on
   laptop-class hardware; rig hosts are far beefier), and narration quality matters when
   an operator listens for a whole session. The lighter alternative, `piper-tts`, was
   rejected on voice quality, not license: **both** trees contain GPL-3.0 code
   (`piper-tts` 1.6.0 is itself GPL-3.0-or-later; `kokoro-onnx` hard-depends on
   GPL-3.0 `phonemizer-fork` and `espeakng-loader`). The plugin stays MIT and vendors
   nothing: GPL code arrives only as pip-installed dependencies in the user's
   environment, the same posture either engine forces. State this plainly in the PR
   body. API: `Kokoro(model_path, voices_path)`;
   `create(text, voice=..., speed=..., lang=...)` returns `(float32 samples,
   sample_rate)`. The seam is a private protocol — `_tts.py::TtsEngine` with
   `synthesize(text) -> tuple[npt.NDArray[np.float32], int]` — so a future engine swap or
   `-S engine=` never touches the sink. Not exposed as an option now (YAGNI: one
   engine).
4. **Dependency carries an environment marker: `kokoro-onnx>=0.4; python_version <
   '3.14'`.** `kokoro-onnx` declares `requires-python >=3.10,<3.14`; an unconditional
   dep would force a `<3.14` cap onto the whole uv workspace lock (every member today is
   an unbounded `>=3.10`). With the marker, `uv lock` keeps the workspace range, and on
   a 3.14+ interpreter `start()`'s guarded import raises a clear message (`--speak
   needs kokoro-onnx, which does not yet support this Python`). Rigs run 3.12; CI
   plugin jobs run 3.11. Note in the plugin pyproject comment (and accept) that
   `kokoro-onnx` floors `numpy>=2.0.2` where installed, while the plugin's own floor
   stays `numpy>=1.24`.
5. **Model acquisition: pinned release files, cached, overridable.** `start()` looks for
   `kokoro-v1.0.onnx` + `voices-v1.0.bin` in `$XDG_CACHE_HOME/inspect-robots-voice/`
   (respecting the env var, defaulting to `~/.cache/`), downloading each from the
   `thewh1teagle/kokoro-onnx` `model-files-v1.0` GitHub release URLs on first use
   (~310MB + ~27MB, one loud progress line each, stdlib `urllib`), then **verifying a
   sha256 pinned in `_tts.py`**. Pins are cross-checked before hardcoding: take the
   digest from the release page if published, otherwise download from two independent
   networks (rig + a GitHub Actions scratch job or any second host) and require
   agreement — never trust a single first download. A mismatch at runtime deletes the
   temp file and raises with expected/actual digests. Download goes to a `.part` temp
   path and renames into place so a killed run never leaves a truncated model behind.
   `-S model=/path -S voices=/path` bypasses cache and download entirely (offline
   rigs). Precedent: faster-whisper auto-downloads on first use.
6. **CLI surface: `--speak` plus repeatable `-S k=v`,** mirroring `--voice`/`-V` and
   parsed by `_parse_kvs` (values arrive coerced). Declared on the **run parser only**
   (see Goal). Factory validates the allowed set and raises `TypeError` on unknown or
   mis-typed keys (caught by `_resolve_or_exit`, same as voice). Keys: `voice` (Kokoro
   voice id, default `af_sarah` — the id used by kokoro-onnx's own examples), `speed`
   (float multiplier, default `1.0`, passed to `create`), `volume` (float `0..1` gain
   applied to samples, default `1.0`), `device` (sounddevice output index or name
   substring, default system default), `lang` (default `en-us`), `model` / `voices`
   (file path overrides). `-S` without `--speak` is a `SystemExit`, like `-V` without
   `--voice`. The kind→flag hint map gains `"sink": "-S"`. **No attended-mode guard**:
   unattended narration is a primary use case, so `--speak` must not require a TTY
   (contrast `--voice`, which gates on attended mode for operator-consent reasons that
   do not apply to output).
7. **Lifecycle mirrors voice input exactly: start after component resolution, close in
   the same `finally`.** The embodiment is claimed inside `_resolve_components`, before
   the try block whose `finally` closes voice — so "start before any hardware claim" is
   not achievable at this wiring point, and voice already accepts that trade (its
   Whisper model download also happens post-claim). `--speak` follows suit: build the
   sink right after resolution, `start()` it next to `_start_voice_input` (loud
   failures abort the run before the first trial), append to the sinks list, `close()`
   in the same `finally` that closes voice. Resolution failures exit before `start()`,
   so nothing leaks on those paths. Docs warn that the **first** `--speak` run downloads
   ~340MB at startup and show how to pre-seed the cache (or pass `-S model=/voices=`).
   After the run starts, the worker catches per-utterance synthesis/playback
   exceptions, prints **one** stderr warning (`speaker: disabled after <type>: <msg>`),
   and the sink goes permanently inert for the rest of the run — a dead speaker must
   never kill or stall an eval. (Raw stderr from a sink coexisting with the attended
   operator footer follows the Rerun sink's existing precedent; unattended runs — the
   primary use case — have no footer at all.)
8. **The control loop is never blocked, and terminal speech gets flush semantics, not
   abort semantics.** `log_policy_messages` only walks the delta, parses
   tool-call arguments defensively (accepted as a dict, or as a JSON string parsing to
   a dict — today's sanitized transcript always sends the string form, so the dict
   branch is defensive-only; anything else, malformed JSON, missing/empty text, and
   non-assistant roles are all skipped silently), and appends to a bounded queue
   (`maxlen=4`, drop-oldest). Synthesis and playback happen on one daemon worker
   thread. Stale narration is worse than skipped narration, hence drop-oldest; drops
   are counted at enqueue. Lifecycle, hook by hook:
   - `on_trial_start` clears anything left from the previous trial (notes must not leak
     across trials) and, when the drop counter is nonzero, prints one line
     (`speaker: dropped N stale note(s)`) and resets it — visible but not spammy,
     matching the Rerun sink's drop-report culture. Clearing at trial **start** rather
     than trial end is deliberate: the rollout enqueues the `done`/`give_up` message
     and returns almost immediately, so an `on_trial_end` clear would routinely delete
     the just-enqueued summary while the worker finishes the previous note. Between
     trials (scoring, grading, operator gates) the summary plays out naturally.
   - `on_eval_end(log)` **drains only when `log.status == "success"`**: wait up to ~15s
     for the queue to empty and the in-flight utterance to finish, then `close()`. The
     flagship unattended single-trial run must end with the summary spoken, not
     clipped. The drain bound keeps torque-holding time finite; scoring/grading already
     spends comparable time between rollout and exit. For any other status
     (`"cancelled"` or `"error"` — the only other terminal `EvalLog.status` literals,
     and `on_eval_end` fires on those paths too, see the Reference), it
     hard-abort-closes immediately: a Ctrl-C or SafetyAbort must not be followed by 15s
     of narration while the robot holds torque.
   - `close()` without a prior drain (the CLI `finally` on error/interrupt paths, which
     runs before `embodiment.close()` releases torque) is a hard abort: set the stop
     event, current playback cuts within ~100ms, join the worker (~5s ceiling), close
     the stream. Error paths should not narrate. `close()` is idempotent, so the
     `finally` after a drained `on_eval_end` is a no-op.
   - The worker checks the stop event between queue pop and synthesis, immediately
     after synthesis, and between playback chunks (frames written to a
     `sounddevice.OutputStream` in small chunks), so the stop event bounds
     everything except one in-flight `Kokoro.create()`, which is waited out (~1-2s for
     two sentences) — the ~5s join ceiling is a true worst case, and playback is never
     waited on.
9. **What is spoken, verbatim: move/capture `note`, `done` `summary`, `give_up`
   `reason`.** `hindsight` is retrospective self-critique addressed to future attempts,
   not operator narration — never spoken. Assistant free text, observations, tool
   results, and operator/grader text are not spoken. No text massaging beyond
   `str.strip()`.
10. **Known limitation, documented, not solved here: `--speak` + `--voice` echo.** The
    microphone can pick up the TTS and feed it back as operator speech; the voice
    input's energy gate and hallucination guards are tuned to *accept* speech, so they
    will not filter it. Docs tell the operator to separate mic and speaker (or use a
    headset), and a follow-up issue for playback-aware muting gets filed at PR time.
    Cross-wiring the two components through core for echo suppression is out of scope.

## Out of scope (YAGNI)

- `eval-set` support: `eval_set()` takes no sinks; adding that parameter is public-API
  growth this feature does not need. Revisit if someone asks for spoken eval-sets.
- Speaking anything besides the three fields above (goals, verdicts, rewards, trial
  banners, hindsight).
- `-S engine=` selection, streaming/incremental synthesis, SSML, per-note voices.
- Echo suppression between `--speak` and `--voice` (follow-up issue).
- A generic `--sink NAME` flag (nothing else needs it; revisit if a third sink appears).
- Windows/macOS audio validation (sounddevice/PortAudio should work, but rigs are Linux;
  no platform-specific code paths).

## Tasks

### Task 1: plugin engine seam `_tts.py`

- [x] `plugins/inspect-robots-voice/src/inspect_robots_voice/_tts.py`: private module.
  `TtsEngine` Protocol: `synthesize(text: str) -> tuple[npt.NDArray[np.float32], int]`
  (float32 mono samples, sample rate — bare `np.ndarray` fails the plugin's strict
  mypy `disallow_any_generics`; every existing plugin module uses `npt.NDArray`).
  Add `kokoro_onnx.*` to the plugin's `[[tool.mypy.overrides]]` ignore list alongside
  `sounddevice.*`/`faster_whisper.*`/`onnx_asr.*`. `KokoroEngine` implements it: constructor takes resolved `model` and
  `voices` paths plus `voice`, `speed`, `lang`; **imports `kokoro_onnx` inside
  `__init__`** behind a guarded try (ImportError message covers both "not installed" and
  the `python_version >= 3.14` marker case, per design decision 4), holds the `Kokoro`
  instance, `synthesize` calls `create(text, voice=..., speed=..., lang=...)`.
- [x] Same module: `resolve_model_files(model: str | None, voices: str | None) ->
  tuple[str, str]` — explicit paths pass through (missing file → `FileNotFoundError`
  naming the path); otherwise cache-dir logic + download + sha256 verify + atomic rename
  as per design decision 5. Module-level constants: URLs, filenames, pinned sha256s
  (cross-checked per design decision 5 before hardcoding).
- [x] Tests (`tests/test_tts.py`): explicit-path passthrough and missing-file error;
  cache hit skips download (monkeypatched fetch asserts not-called); download path writes
  `.part` then renames; sha256 mismatch deletes and raises with both digests; no
  module-top import of `kokoro_onnx` (mirror the existing lazy-import test pattern);
  guarded-import error message. `KokoroEngine` construction/synthesis covered via a
  stubbed `kokoro_onnx` module injected into `sys.modules`.

### Task 2: plugin `_speaker.py` — `SpeakerSink`

- [ ] `extract_speech(messages: Sequence[Any]) -> list[str]` (module-level, pure):
  defensive walk per design decisions 8-9 — per assistant tool call, speak `note`
  (move/capture), or `summary` when `name == "done"`, or `reason` when
  `name == "give_up"`; returns stripped non-empty strings in order.
- [ ] `SpeakerSink(NullSink)`: constructor takes validated options plus two injectable
  factories for tests — `engine_factory` (defaults to building `KokoroEngine` via
  `resolve_model_files`) and `playback_factory` (defaults to a thin
  `sounddevice.OutputStream` wrapper, lazy import inside it). `start()` builds engine +
  playback (loud failures propagate); `log_policy_messages` enqueues per design
  decision 8 and is a no-op before `start()` or after worker death; worker loop:
  pop → (stop check) → synthesize → (stop check) → apply `volume` gain → chunked write
  with stop-event checks; error handling per design decision 7; `on_trial_start` clears
  the queue + prints the drop report when `dropped > 0`, then resets the counter;
  `on_eval_end(log)` drains (queue empty + in-flight done, ~15s bound) then closes when
  `log.status == "success"`, hard-abort-closes for any other status; `close()` alone is
  a hard abort — idempotent, sets stop event, joins worker (timeout ~5s), closes
  playback.
- [ ] Tests (`tests/test_speaker.py`), all with fake engine/playback, no real audio:
  extraction table (move note; multiple tool_calls in one message; `done` summary spoken
  and hindsight NOT spoken; `give_up` reason; dict already parsed vs JSON string
  arguments; malformed JSON; missing/blank fields; non-assistant role; non-dict
  message); enqueue→spoken order; overflow drops oldest and the next `on_trial_start`
  report prints once with the right count then resets; queue cleared at trial start,
  NOT at trial end (a summary enqueued just before `on_trial_end` survives and gets
  spoken); `on_eval_end` with a success log drains — a queued utterance finishes before
  close (and the drain respects its time bound when the fake worker stalls);
  `on_eval_end` with a `"cancelled"` or `"error"` log aborts instead of draining; worker
  exception → one
  warning + inert (later `log_policy_messages` calls do nothing, eval hooks still no-op
  safely); bare `close()` idempotent + aborts mid-utterance (stop event cuts chunked
  playback); `log_policy_messages` before `start()` is a safe no-op.

### Task 3: plugin factory + entry point + packaging

- [ ] `__init__.py`: `speaker_sink(**kwargs: ScalarValue) -> SpeakerSink` factory
  mirroring `voice_input`'s allowed-set validation (`voice`, `speed`, `volume`,
  `device`, `lang`, `model`, `voices`); numeric coercions accept int-or-float where
  sensible; `volume` outside `[0, 1]` and non-positive `speed` are `TypeError`s (the
  registry surfaces `TypeError` cleanly; `ValueError` would escape as a traceback).
  Export `SpeakerSink` in `__all__`; bump `__version__` to `0.4.0`.
- [ ] `pyproject.toml`: version 0.4.0; add `kokoro-onnx>=0.4; python_version < '3.14'`
  to deps (comment: marker keeps the workspace lock's python range; kokoro floors
  numpy 2 where installed); new entry point
  `[project.entry-points."inspect_robots.sinks"] speaker = "inspect_robots_voice:speaker_sink"`.
  Core floor stays `inspect-robots>=0.44` (the sink seam predates it; the new flag lives
  in core, not here). Run `uv lock`; verify the workspace `requires-python` range is
  unchanged in the lockfile.
- [ ] Tests (`tests/test_factory.py` additions): unknown key, each mis-typed key, range
  violations; defaults; entry point resolves via `importlib.metadata` in the dev install.

### Task 4: core CLI `--speak` / `-S` (run only)

- [ ] `cli.py`: add `--speak` (store_true) and repeatable `-S` (`dest="speak_args"`,
  `metavar="k=v"`) to the **run parser only** — NOT `_add_shared_eval_args`, so
  `eval-set` rejects them as unknown flags. `_build_speaker_sink(args)`: `-S` without
  `--speak` → `SystemExit("-S requires --speak")`; resolve via
  `_resolve_or_exit("sink", "speaker", **_parse_kvs(args.speak_args))`; unknown-name
  message gains the `pip install inspect-robots-voice` hint (extend the existing hint
  branch at ~line 643 to cover `kind == "sink"` + `no sink named "speaker"`); add
  `"sink": "-S"` to the kind→flag map. `_start_speaker_sink` / `_close_speaker_sink`
  helpers mirror the voice ones (duck-typed `start`/`close`); start next to
  `_start_voice_input`, close in the same `finally`; append the started sink to the
  sinks list at the single assembly site in `_cmd_run` (~line 1474).
- [ ] Tests (mirror the `--voice` approach in `tests/test_registry_cli.py`): `-S`
  without `--speak` exits; missing plugin exits with the pip hint; a fake registered
  sink receives coerced `-S` kwargs, gets `start()`ed before eval and `close()`d after
  (including when eval raises); sink lands in the sinks list; no `--speak` → sinks list
  unchanged; `eval-set` rejects `--speak`. Coverage of every new branch, explicitly:
  `start()` raising → `SystemExit`; a resolved sink with no callable `start`; a sink
  with no callable `close`; and the false arm of the new hint condition (a `KeyError`
  escaping a sink factory must NOT get the pip hint — the hint-on-registry-KeyError
  precedent is tests/test_registry_cli.py ~3436 and the adjacent no-hint TypeError
  test ~3458; the factory-KeyError case has no precedent and is written fresh). Keep core coverage at 100%.

### Task 5: docs + changelog + module maps

- [ ] `docs/guide/voice-mode.md`: new "Speaking policy notes: `--speak`" section — what
  is spoken (note/summary/reason, never hindsight), install, `-S` keys with defaults,
  first-run model download (size, cache path, pre-seeding, offline override), unattended
  use, run-only scope, the `--speak`+`--voice` echo caveat. `docs/guide/cli.md`: mention
  `--speak`/`-S` next to `--voice`/`-V`. `docs/guide/plugins.md` (~lines 106-108): the
  voice plugin entry now covers both directions.
- [ ] Root `README.md`: voice plugin bullet now covers both directions. Root `CLAUDE.md`
  Layout bullet likewise; `src/inspect_robots/CLAUDE.md` cli row; plugin `CLAUDE.md`
  gains the speaker module map rows.
- [ ] `CHANGELOG.md`: core Unreleased "Added" entry (`--speak`/`-S`, hint-map, lifecycle)
  and voice plugin 0.4.0 entry (SpeakerSink, Kokoro engine, model cache), both linking
  issue #327 and this plan (0054).
- [ ] File the follow-up issue for `--speak`+`--voice` playback-aware muting; link it
  from the PR body.
