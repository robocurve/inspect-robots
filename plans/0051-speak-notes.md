# `run --speak`: broadcast agent policy notes through the rig speaker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent-policy runs require a `note` on every move/capture/done/give_up tool call —
one or two operator-facing sentences about what the model sees and decides. Today those
notes are visible only on screens (transcript echo, the Rerun llm tab, saved logs). On a
rig the operator's eyes are on the robot, not the terminal. This plan adds
`inspect-robots run --speak`: each note is spoken through the machine's audio output as it
streams, so the operator hears the policy narrate itself while hands-busy or across the
room. Works unattended (no TTY requirement). (Issue #327.)

**Shape:** one PR, two packages.

1. **Core (CLI wiring only, 100% covered):** `--speak` plus repeatable `-S k=v` on `run`
   and `eval-set`; resolve the sink registered as `speaker` in the existing
   `inspect_robots.sinks` entry-point group; call its duck-typed `start()`; append it to
   the sinks list the eval loop already fans `log_policy_messages` to; `close()` on the
   way out. Missing-plugin errors hint `pip install inspect-robots-voice`. No new public
   API symbols, no rollout/eval changes.
2. **Voice plugin (`inspect-robots-voice` 0.3.0):** new `SpeakerSink` — extracts `note`
   arguments from assistant `tool_calls` in the live transcript delta, bounded
   drop-oldest queue, one daemon worker thread, local Kokoro TTS behind a small engine
   seam, playback through the already-depended-on `sounddevice`.

**Tech Stack:** core stays stdlib + NumPy. Plugin adds `kokoro-onnx` (MIT wrapper;
Kokoro-82M model is Apache-2.0; pulls `onnxruntime` CPU). `sounddevice` and NumPy are
already plugin deps. Python 3.10+.

## Global Constraints

- Core gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% coverage**.
  Plugin gates mirror the existing `plugin-voice` CI job: ruff + ruff format + mypy
  (plugin config) + pytest, own coverage scope (reported, not gated at 100%).
- D1 docstrings on public defs; state the contract, not the name. Line length 100.
- Core must never import `kokoro_onnx`/`sounddevice` (the `core-only-import` job guards
  this); the plugin must not import them at module top either, so its test suite runs on
  CI runners without PortAudio or onnxruntime loaded (`kokoro-onnx` still installs in the
  locked env; importing it stays confined to `start()`-time code paths that tests replace
  with fakes).
- Public API is fenced by `inspect_robots.__all__` **and** `tests/test_api_snapshot.py` —
  this plan adds **no** public core symbol, so neither file changes; if an implementation
  detail forces one, stop and re-plan.
- No behavior change without `--speak`: every existing run takes today's exact paths;
  existing tests pass untouched.
- `uv lock` after touching any pyproject; CI installs `--locked`.
- Writing-style rules for public-facing text (README, docs): no em dashes in prose, no
  decorative emoji, headers use colons.

## Reference: current wiring (main @ 0ce79eb1)

- `logging/sink.py`: `LogSink` protocol (`on_eval_start` / `on_trial_start` / `log_step` /
  `on_trial_end` / `on_eval_end`) + `NullSink` base. Duck-typed extension
  `log_policy_messages(t, messages)` is called at most once per control step, only when
  the policy performed an inference; messages are plain-JSON-type dicts shaped like
  `TrialRecord.policy_transcript` entries; **core does not enforce the shape on this live
  path, so sinks must render defensively and must not mutate** (see the module docstring).
- `eval.py` `_SinkFan` (~line 115): fans `log_policy_messages` to every sink that defines
  it — a second consumer costs nothing.
- `rollout.py` (~line 253): `stream_ok = callable(policy.transcript_delta) and
  callable(sink.log_policy_messages)`; the agent policy's `transcript_delta` returns
  sanitized OpenAI-style dicts appended since the previous call.
- Note shape (agent plugin `_tools.py`): every move/capture tool schema requires a `note`
  string argument; `done`/`give_up` require `note` too. In transcript messages these
  appear as `{"role": "assistant", "tool_calls": [{"function": {"name": ...,
  "arguments": "<JSON string containing note>"}}]}` — `rerun_sink.py::_render_message`
  is the defensive-parsing precedent.
- `registry.py`: kind `"sink"` already exists with entry-point group
  `inspect_robots.sinks` (line ~39); nothing to add in the registry.
- `cli.py`: `--voice` / repeatable `-V k=v` precedent (lines ~197-206), `_parse_kvs`
  coercion, `_resolve_or_exit` + kind→flag hint map (~line 652), missing-plugin hint for
  operator_input (~line 643), `_build_voice_input` / `_start_voice_input` /
  `_close_voice_input` lifecycle helpers (~lines 791-830), sinks list assembly
  (~lines 1472-1500).
- Voice plugin: `_input.py` `VoiceInput` shows the `start()`/`close()` duck-typed
  lifecycle and lazy heavy imports; `__init__.py::voice_input` shows the strict
  factory-kwarg validation pattern (allowed-set + per-key `TypeError`).
- CI: `plugin-voice` job and `publish-voice` release job already exist (PR #316/#325) —
  no new CI jobs needed.

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
3. **Engine: Kokoro-82M via `kokoro-onnx`, behind an internal seam.** Verified 2026-08:
   Kokoro-82M is the consensus best open-weight TTS that runs faster than real-time on
   CPU (~6x on laptop-class hardware); `kokoro-onnx` is MIT and the model Apache-2.0,
   whereas current `piper-tts` (1.6.0) is GPL-3.0-or-later. API:
   `Kokoro(model_path, voices_path)`; `create(text, voice=..., speed=..., lang=...)`
   returns `(float32 samples, sample_rate)`. The seam is a private protocol —
   `_tts.py::TtsEngine` with `synthesize(text) -> tuple[np.ndarray, int]` — so a future
   engine swap or `-S engine=` never touches the sink. Not exposed as an option now
   (YAGNI: one engine).
4. **Model acquisition: pinned release files, cached, overridable.** `start()` looks for
   `kokoro-v1.0.onnx` + `voices-v1.0.bin` in `$XDG_CACHE_HOME/inspect-robots-voice/`
   (respecting the env var, defaulting to `~/.cache/`), downloading each from the
   `thewh1teagle/kokoro-onnx` `model-files-v1.0` GitHub release URLs on first use
   (~310MB + ~27MB, one loud progress line each, stdlib `urllib`), then **verifying a
   sha256 pinned in `_tts.py`** (compute the pins once during implementation; a mismatch
   deletes the temp file and raises with the expected/actual digests). Download goes to a
   `.part` temp path and renames into place so a killed run never leaves a truncated
   model behind. `-S model=/path -S voices=/path` bypasses cache and download entirely
   (offline rigs). Precedent: faster-whisper auto-downloads on first use.
5. **CLI surface: `--speak` plus repeatable `-S k=v`,** mirroring `--voice`/`-V` and
   parsed by `_parse_kvs` (values arrive coerced). Factory validates the allowed set and
   raises `TypeError` on unknown or mis-typed keys (caught by `_resolve_or_exit`, same as
   voice). Keys: `voice` (Kokoro voice id, default `af_sarah` — the id used by
   kokoro-onnx's own examples), `speed` (float multiplier, default `1.0`, passed to
   `create`), `volume` (float `0..1` gain applied to samples, default `1.0`), `device`
   (sounddevice output index or name substring, default system default), `lang` (default
   `en-us`), `model` / `voices` (file path overrides). `-S` without `--speak` is a
   `SystemExit`, like `-V` without `--voice`. The kind→flag hint map gains
   `"sink": "-S"`. **No attended-mode guard**: unattended narration is a primary use
   case, so `--speak` must not require a TTY (contrast `--voice`, which gates on
   attended mode for operator-consent reasons that do not apply to output).
6. **Fail loudly before the run, stay inert after it starts.** The CLI calls the sink's
   duck-typed `start()` right after resolving it — before any hardware claim or trial —
   so a missing/corrupt model, an unloadable onnxruntime, or a bad output device aborts
   the run with a clear message (mirror `_start_voice_input`; reuse its
   PortAudio-guidance precedent for output-device errors). After the run starts, the
   worker catches per-utterance synthesis/playback exceptions, prints **one** stderr
   warning (`speaker: disabled after <type>: <msg>`), and the sink goes permanently
   inert for the rest of the run — a dead speaker must never kill or stall an eval.
7. **The control loop is never blocked.** `log_policy_messages` only walks the delta,
   `json.loads`es tool-call arguments defensively (malformed or non-dict arguments,
   missing/empty `note`, non-assistant roles are all skipped silently), and appends to a
   bounded queue (`maxlen=4`, drop-oldest). Synthesis and playback happen on one daemon
   worker thread. Stale narration is worse than skipped narration, hence drop-oldest;
   drops are counted and reported in one line at trial end (`speaker: dropped N stale
   note(s)`) — visible but not spammy, matching the Rerun sink's drop-report culture.
   `on_trial_end` also clears the queue (notes must not leak across trials); the
   utterance already playing finishes (bounded by one note's length). Playback writes
   frames to a `sounddevice.OutputStream` in small chunks, checking a stop event between
   chunks, so `close()` aborts within ~100ms; `close()` (also called by `on_eval_end`)
   stops the worker and joins with a short timeout.
8. **Notes only, spoken verbatim.** Assistant free text, observations, tool results, and
   operator/grader text are not spoken. No text massaging beyond `str.strip()`.
9. **Known limitation, documented, not solved here: `--speak` + `--voice` echo.** The
   microphone can pick up the TTS and feed it back as operator speech; the voice input's
   energy gate and hallucination guards are tuned to *accept* speech, so they will not
   filter it. Docs tell the operator to separate mic and speaker (or use a headset), and
   a follow-up issue for playback-aware muting gets filed at PR time. Cross-wiring the
   two components through core for echo suppression is out of scope.

## Out of scope (YAGNI)

- Speaking anything besides tool-call notes (goals, verdicts, rewards, trial banners).
- `-S engine=` selection, streaming/incremental synthesis, SSML, per-note voices.
- Echo suppression between `--speak` and `--voice` (follow-up issue).
- A generic `--sink NAME` flag (nothing else needs it; revisit if a third sink appears).
- Windows/macOS audio validation (sounddevice/PortAudio should work, but rigs are Linux;
  no platform-specific code paths).

## Tasks

### Task 1: plugin engine seam `_tts.py`

- [ ] `plugins/inspect-robots-voice/src/inspect_robots_voice/_tts.py`: private module.
  `TtsEngine` Protocol: `synthesize(text: str) -> tuple[Any, int]` (float32 mono samples,
  sample rate). `KokoroEngine` implements it: constructor takes resolved `model` and
  `voices` paths plus `voice`, `speed`, `lang`; **imports `kokoro_onnx` inside
  `__init__`**, holds the `Kokoro` instance, `synthesize` calls
  `create(text, voice=..., speed=..., lang=...)`.
- [ ] Same module: `resolve_model_files(model: str | None, voices: str | None) ->
  tuple[str, str]` — explicit paths pass through (missing file → `FileNotFoundError`
  naming the path); otherwise cache-dir logic + download + sha256 verify + atomic rename
  as per design decision 4. Module-level constants: URLs, filenames, pinned sha256s
  (compute during implementation and hardcode).
- [ ] Tests (`tests/test_tts.py`): explicit-path passthrough and missing-file error;
  cache hit skips download (monkeypatched fetch asserts not-called); download path writes
  `.part` then renames; sha256 mismatch deletes and raises with both digests; no
  module-top import of `kokoro_onnx` (mirror the existing lazy-import test pattern).
  `KokoroEngine` construction/synthesis covered via a stubbed `kokoro_onnx` module
  injected into `sys.modules`.

### Task 2: plugin `_speaker.py` — `SpeakerSink`

- [ ] `extract_notes(messages: Sequence[Any]) -> list[str]` (module-level, pure):
  defensive walk per design decision 7; returns stripped non-empty notes in order.
- [ ] `SpeakerSink(NullSink)`: constructor takes validated options plus two injectable
  factories for tests — `engine_factory` (defaults to building `KokoroEngine` via
  `resolve_model_files`) and `playback_factory` (defaults to a thin
  `sounddevice.OutputStream` wrapper, lazy import inside it). `start()` builds engine +
  playback (loud failures propagate); `log_policy_messages` enqueues per design
  decision 7 and is a no-op before `start()` or after worker death; worker loop:
  pop → synthesize → apply `volume` gain → chunked write with stop-event checks; error
  handling per design decision 6; `on_trial_end` clears queue + prints the drop report
  when `dropped > 0`, then resets the counter; `on_eval_end` → `close()`; `close()`
  idempotent, sets stop event, joins worker (timeout ~5s), closes playback.
- [ ] Tests (`tests/test_speaker.py`), all with fake engine/playback, no real audio:
  extraction table (valid note; multiple tool_calls in one message; dict already parsed
  vs JSON string arguments; malformed JSON; missing/blank note; non-assistant role;
  non-dict message); enqueue→spoken order; overflow drops oldest and trial-end report
  prints once with the right count then resets; queue cleared at trial end; worker
  exception → one warning + inert (later `log_policy_messages` calls do nothing, eval
  hooks still no-op safely); `close()` idempotent + aborts mid-utterance (stop event cuts
  chunked playback); `log_policy_messages` before `start()` is a safe no-op.

### Task 3: plugin factory + entry point + packaging

- [ ] `__init__.py`: `speaker_sink(**kwargs: ScalarValue) -> SpeakerSink` factory
  mirroring `voice_input`'s allowed-set validation (`voice`, `speed`, `volume`,
  `device`, `lang`, `model`, `voices`); numeric coercions accept int-or-float where
  sensible; `volume` outside `[0, 1]` and non-positive `speed` are `TypeError`s (the
  registry surfaces `TypeError` cleanly; `ValueError` would escape as a traceback).
  Export `SpeakerSink` in `__all__`; bump `__version__` to `0.3.0`.
- [ ] `pyproject.toml`: version 0.3.0; add `kokoro-onnx>=0.4` to deps; new entry point
  `[project.entry-points."inspect_robots.sinks"] speaker = "inspect_robots_voice:speaker_sink"`.
  Core floor stays `inspect-robots>=0.44` (the sink seam predates it; the new flag lives
  in core, not here). Run `uv lock`.
- [ ] Tests (`tests/test_factory.py` additions): unknown key, each mis-typed key, range
  violations; defaults; entry point resolves via `importlib.metadata` in the dev install.

### Task 4: core CLI `--speak` / `-S`

- [ ] `cli.py`: add `--speak` (store_true) and repeatable `-S` (`dest="speak_args"`,
  `metavar="k=v"`) wherever `--voice`/`-V` are declared (run and eval-set share the
  helper). `_build_speaker_sink(args)`: `-S` without `--speak` → `SystemExit("-S
  requires --speak")`; resolve via `_resolve_or_exit("sink", "speaker",
  **_parse_kvs(args.speak_args))`; unknown-name message gains the
  `pip install inspect-robots-voice` hint (extend the existing hint branch at ~line 643
  to cover `kind == "sink"` + `no sink named "speaker"`); add `"sink": "-S"` to the
  kind→flag map. `_start_speaker_sink` / `_close_speaker_sink` helpers mirror the voice
  ones (duck-typed `start`/`close`, `close` in the same `finally` that closes voice);
  append the started sink to the sinks list next to the Rerun wiring (~line 1472) in
  **both** command paths that assemble sinks.
- [ ] Tests: mirror the `--voice` CLI test approach — `-S` without `--speak` exits;
  missing plugin exits with the pip hint; a fake registered sink receives coerced `-S`
  kwargs, gets `start()`ed before eval and `close()`d after (including on eval failure);
  sink lands in the sinks list; no `--speak` → sinks list unchanged. Keep core coverage
  at 100%.

### Task 5: docs + changelog + module maps

- [ ] `docs/guide/voice-mode.md`: new "Speaking policy notes: `--speak`" section — what
  is spoken, install, `-S` keys with defaults, first-run model download (size, cache
  path, offline override), unattended use, the `--speak`+`--voice` echo caveat.
  `docs/guide/cli.md`: mention `--speak`/`-S` next to `--voice`/`-V`.
- [ ] Root `README.md`: voice plugin bullet now covers both directions. Root `CLAUDE.md`
  Layout bullet likewise; `src/inspect_robots/CLAUDE.md` cli row; plugin `CLAUDE.md`
  gains the speaker module map rows.
- [ ] `CHANGELOG.md`: core Unreleased "Added" entry (`--speak`/`-S`, hint-map, lifecycle)
  and voice plugin 0.3.0 entry (SpeakerSink, Kokoro engine, model cache), both linking
  issue #327 and this plan.
- [ ] File the follow-up issue for `--speak`+`--voice` playback-aware muting; link it
  from the PR body.
