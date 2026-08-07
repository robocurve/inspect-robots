# Voice: Parakeet TDT 0.6B v3 backend via onnx-asr, as the new default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the voice plugin's default transcription model. faster-whisper `small`
was the conservative launch choice; published 2026 numbers place NVIDIA Parakeet TDT
0.6B v3 ahead on every axis that matters for spoken operator feedback: ~6.3% average WER
on the Open ASR Leaderboard (better than Whisper large-v3, far better than `small`),
roughly 30x real-time on CPU (vs ~8x), no fixed 30-second window (short utterances cost
their real length), and a transducer decoder that cannot confabulate text from silence
(the Whisper hallucination class our blocklist patches). Decision basis is the published
numbers **by explicit owner decision (no local benchmark)**; the residual risk (rig-noise
behavior) is mitigated by keeping Whisper one flag away. (Issue #324.)

**Shape:** one PR, plugin-only (`plugins/inspect-robots-voice/` → 0.3.0). No core changes.

**Tech Stack:** `onnx-asr[cpu,hub]>=0.12,<1` — pure Python, runs on onnxruntime + NumPy,
no torch and no NeMo. `onnxruntime` and `huggingface_hub` are already in the plugin's
tree transitively via faster-whisper, so the marginal dependency weight is one small
pure-Python package. Model weights are CC-BY-4.0 (attribution required, see Task 4);
int8 ONNX download is ~640 MB one-time via the HF hub cache.

## Global Constraints

- Plugin gates: `ruff check` + `ruff format --check` + mypy (plugin config, strict, src
  and tests) + pytest, scoped to `plugins/inspect-robots-voice`; core gates must stay
  green untouched (`pytest --cov` at 100% remains core-only).
- Lazy imports: `onnx_asr` joins `sounddevice`/`faster_whisper` under the existing
  invariant — never imported at module top, never imported by tests (fakes only), so CI
  needs no model downloads and `core-only-import` is unaffected.
- D1 docstrings, line length 100, repo writing style for README/docs text (no em dashes
  in prose, no dangling header colons).
- `uv lock` after the pyproject change.
- No behavior change for anyone passing `-V model=<whisper name>`: the Whisper path,
  its gauntlet, and its defaults are untouched.

## Reference: current state (main @ 0ce79eb1, voice 0.2.0)

- `_transcriber.py`: `WhisperTranscriber(model, compute, language, asr_device="cpu")`
  with injectable `_model_factory`; gauntlet = empty-text, `no_speech_prob > 0.6`,
  `avg_logprob < -1.0`, `< 0.4 s`, hallucination blocklist; `vad_filter=True`.
- `_input.py`: `TranscriberFactory = Callable[[str, str, str, str], _Transcriber]`;
  `VoiceInput(model="small", device=None, language="en", compute="auto",
  asr_device="cpu")`; `start()` builds the transcriber via the factory seam.
- `__init__.py`: `voice_input(**kwargs)` validates the scalar union per key; allowed =
  {model, device, language, compute, asr_device}; unknown/mis-typed → `TypeError`.
- Probed `onnx-asr` 0.12.0 API (exact, verified against the installed package):
  - `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8",
    providers=["CPUExecutionProvider"]) -> TextResultsAsrAdapter`
  - `.with_timestamps() -> TimestampedResultsAsrAdapter` (its results carry logprobs;
    the adapter passes `need_logprobs="yes"` internally)
  - `.recognize(waveform: npt.NDArray[np.float32] | ..., *, sample_rate=16000)` accepts
    a mono float32 array directly and returns `TimestampedResult` for a single input
  - `TimestampedResult` dataclass fields: `text: str`, `timestamps: list[float] | None`,
    `tokens: list[str] | None`, `logprobs: list[float] | None`

## Design decisions (and why)

1. **Backend is inferred from the model name; no new flag.** Precedence, evaluated on
   the raw string:
   - contains a path separator (`/` or `\`) → whisper (filesystem CT2 paths always
     mean whisper, even a path like `/models/parakeet-ct2`);
   - case-insensitive exact alias: `"parakeet"` or `"parakeet-tdt-0.6b-v3"` →
     canonical `"nemo-parakeet-tdt-0.6b-v3"`, parakeet backend;
   - case-insensitive prefix `"nemo-"` → parakeet backend, name passed through
     lowercased (onnx-asr's registry is lowercase);
   - any other string containing `"parakeet"` (e.g. `"parakeet-tdt-1.1b"`) →
     `TypeError` listing the supported parakeet names, so typos fail at the factory,
     not minutes later inside `start()`;
   - everything else → whisper, exactly as today.
   The default `model` becomes `"parakeet-tdt-0.6b-v3"`. Escape hatch stays one flag:
   `-V model=small` (or any Whisper size/path) restores 0.2.0 behavior exactly.
2. **Selection lives in `_transcriber.py` as `resolve_transcriber(model, compute,
   language, asr_device) -> _Transcriber`,** and `_input.py`'s default
   `_transcriber_factory` simply calls it. The `TranscriberFactory` seam, `VoiceInput`,
   the worker, and every orchestration invariant are untouched — this PR changes which
   object comes out of an existing seam, nothing else. **Import topology (no cycle):**
   `_parakeet.py` imports the shared private pre-check helpers from `_transcriber.py`
   at module top (one direction), and `resolve_transcriber` imports
   `ParakeetTranscriber` with a function-local
   `from inspect_robots_voice._parakeet import ParakeetTranscriber` — never at
   `_transcriber.py` module top. `resolve_transcriber` does selection and
   canonicalization only; all cross-backend option policy lives in `voice_input`
   (decision 5).
3. **`ParakeetTranscriber` lives in a new `_parakeet.py`** (module map updated), lazy
   `import onnx_asr` at construction, `load_model(name, quantization="int8",
   providers=["CPUExecutionProvider"])` then `.with_timestamps()`. `transcribe()`
   mirrors the Whisper contract (`(audio) -> str | None`): reshape to mono float32,
   apply the shared pre-checks, call `recognize(samples, sample_rate=16000)`, apply the
   Parakeet gauntlet, return stripped text or `None`. int8 quantization is fixed (not
   configurable): it is the published CPU-deployment shape and a 4x smaller download;
   a knob can come later if anyone asks.
4. **Parakeet gauntlet: architecture does most of the work.** Keep: min-duration
   pre-check (shared constant), empty/whitespace-text rejection, and the hallucination
   blocklist (free belt-and-suspenders; harmless on a transducer). Add: mean token
   logprob rejection, `mean(result.logprobs) < -2.5` → `None`, but **only when
   `logprobs` is truthy** (`if logprobs:` — `None` AND the empty list both skip the
   check; `mean([])` is NaN/raises depending on the implementation, and a missing list
   deliberately fails open) — a deliberately loose, uncalibrated threshold (we chose
   not to bench; document it as a coarse garbage-catch, not a tuned filter). Drop:
   `no_speech_prob` and `avg_logprob` (Whisper-specific; onnx-asr exposes neither), and
   `vad_filter` (no equivalent; the energy gate plus transducer behavior covers it).
5. **Cross-backend option semantics are validated loudly at the factory, not silently
   ignored — by VALUE, uniformly, in `voice_input` only.** The rule for every
   whisper-only option is the same: with a parakeet model, a value other than the
   backend-neutral default raises `TypeError`; the neutral default itself is always
   accepted, whether typed or omitted. This makes the CLI's `_parse_kvs` coercion a
   non-issue (`-V language=none` coerces to Python `None`, which IS the neutral value,
   so a user typing the parakeet-correct thing is accepted, never scolded) and needs no
   key-presence machinery (which `resolve_transcriber` could not perform anyway, since
   it receives resolved values).
   - `language` changes type to `str | None`, default `None`. Whisper: `None` resolves
     to `"en"` inside `WhisperTranscriber` (today's behavior preserved); explicit
     strings pass through. Parakeet: v3 auto-detects among 25 languages, so `None` is
     correct; a non-`None` language with a parakeet model raises
     `TypeError("parakeet models auto-detect language; drop -V language or pick a
     whisper model")`.
   - `asr_device`: parakeet runs CPU-only in this plan (CUDA needs `onnxruntime-gpu`,
     which we do not ship); a value other than `"cpu"` with a parakeet model raises
     `TypeError("the parakeet backend runs on the CPU; use a whisper model for
     -V asr_device=cuda")`.
   - `compute`: whisper-only (CTranslate2 compute type); a value other than `"auto"`
     with a parakeet model raises `TypeError("compute applies to whisper models; the
     parakeet backend is fixed to int8")`.
   So `-V asr_device=cpu`, `-V compute=auto`, and `-V language=none` are all accepted
   no-ops with parakeet — consistent by construction.
6. **Version and licensing:** plugin 0.3.0 (default-model change is behavior-visible;
   0.x minor). README and the voice-mode docs page gain a one-line attribution:
   Parakeet TDT 0.6B v3 weights by NVIDIA, licensed CC-BY-4.0, downloaded from the
   Hugging Face hub on first use (~640 MB int8). Docs table updates: `model` default,
   which `-V` keys apply to which backend, and the escape hatch.
7. **Startup line unchanged in shape:** `listening on <device>
   (model=parakeet-tdt-0.6b-v3)` — the alias the user typed (or defaulted to) is shown,
   not the canonical `nemo-` name, matching what they would pass to `-V model=`.

## Out of scope (YAGNI)

GPU execution providers for Parakeet, quantization knobs, onnx-asr's VAD adapter,
canary/other onnx-asr model families, batch recognition, streaming partials, removing
the faster-whisper dependency (Whisper stays the multilingual-explicit and GPU option).

## Tasks

### Task 1: dependency + scaffolding

- [ ] `plugins/inspect-robots-voice/pyproject.toml`: add `onnx-asr[cpu,hub]>=0.12,<1`
  to dependencies; bump version to `0.3.0`; `uv lock`. Expect the lock to move
  onnxruntime on the py3.10 resolution only (onnx-asr pins `onnxruntime<1.24` there;
  the advisory py3.10 CI tier covers it) — resolver-checked compatible with
  faster-whisper 1.2.1's `onnxruntime>=1.14,<2`.
- [ ] Bump `__version__` in `__init__.py` and the version-pinning test in
  `tests/test_factory.py` (it asserts the literal).

### Task 2: `_parakeet.py` + shared gauntlet pieces

- [ ] Extract the shared pre-check constants/helpers used by both backends
  (min-duration, blocklist membership) so `_parakeet.py` imports them from
  `_transcriber.py` rather than duplicating (keep them private; no API change).
- [ ] `_parakeet.py`: `ParakeetTranscriber(model: str)` per decisions 3-4, with an
  injectable `_model_factory` seam mirroring `WhisperTranscriber`'s (the factory
  returns an object with `recognize(samples, *, sample_rate) -> TimestampedResult`-
  shaped results; type it with a local Protocol).
- [ ] Tests (fake model objects only; never import onnx_asr): accept/reject matrix —
  normal sentence accepted; empty text; whitespace; blocklist phrase as entire text;
  short-audio pre-check (no model call at all); mean-logprob below threshold rejected;
  `logprobs=None` accepted when text is fine; `logprobs=[]` accepted (fail-open, no
  NaN warning); sample_rate=16000 passed through; audio reshaped to mono float32.

### Task 3: backend selection + factory validation

- [ ] `_transcriber.py`: a single shared private classifier `_classify_model(model)
  -> str | None` implementing the full decision-1 precedence (path separator →
  whisper i.e. `None`; alias map; `nemo-` passthrough lowercased; unknown
  parakeet-ish name → the decision-1 `TypeError`; else whisper i.e. `None`), returning
  the canonical parakeet name or `None` for whisper. `resolve_transcriber(model,
  compute, language, asr_device)` calls it and constructs `ParakeetTranscriber` or
  `WhisperTranscriber`; it performs no decision-5 option validation (that lives in
  `voice_input`, next bullets).
- [ ] `_input.py`: default `_transcriber_factory` delegates to `resolve_transcriber`;
  `VoiceInput` default `model` becomes `"parakeet-tdt-0.6b-v3"`; `language` default
  becomes `None` (threaded through the factory type and `WhisperTranscriber`'s
  `None -> "en"` resolution).
- [ ] `__init__.py`: `voice_input` — default `model` `"parakeet-tdt-0.6b-v3"`;
  `language` accepts `str | None` (default `None`): the existing
  `isinstance(language, str)` check (`__init__.py:46-47`) widens to `str | None`;
  cross-backend `TypeError`s per decision 5 are value-based (non-`None` language,
  non-`"auto"` compute, non-`"cpu"` asr_device with a parakeet model), all raised
  here, none in `resolve_transcriber`. **`voice_input` gates them on the shared
  `_classify_model` helper** — never a naive substring check, which would misclassify
  a whisper CT2 path like `/models/parakeet-ct2` — and calling the helper here also
  surfaces the decision-1 unknown-parakeet-name `TypeError` at factory time (the CLI
  resolves `voice_input(**kvs)` long before `start()`), satisfying decision 1's
  "fail at the factory" promise. When both could apply (unknown parakeet-ish name AND
  a whisper-only option), the decision-1 name error wins — the classifier runs first.
- [ ] Tests: alias table (`parakeet`, `parakeet-tdt-0.6b-v3`, `nemo-parakeet-tdt-0.6b-v3`,
  mixed case) → ParakeetTranscriber with the canonical name; unknown parakeet-ish name
  (`parakeet-tdt-1.1b`) → `TypeError` listing supported names; `small`/
  `distil-small.en`/a filesystem path (including one whose basename contains
  "parakeet", e.g. `/models/parakeet-ct2`) → WhisperTranscriber with today's exact
  arguments; factory defaults (`model`, `language=None`) updated; each cross-backend
  `TypeError` (message asserted) plus the accepted no-ops (`asr_device=cpu`,
  `compute=auto`, `language=None` with parakeet); whisper-only option values with a
  path model whose basename contains "parakeet" (e.g.
  `voice_input(model="/models/parakeet-ct2", language="fr", asr_device="cuda")`) are
  **accepted** (the classifier, not a substring, gates the rules); unknown
  parakeet-ish name + whisper-only option raises the decision-1 name error, not the
  decision-5 one; whisper `language=None` resolves to `"en"`; explicit whisper
  language passes through. **Delete the now-inverted
  rejection-matrix row** `({"language": None}, "language must be a string")` at
  tests/test_factory.py:52.
- [ ] `tests/test_input.py`: the two listening-line literals asserting
  `(model=small)` (lines ~293 and ~325, defaults-built `VoiceInput`) become
  `(model=parakeet-tdt-0.6b-v3)`; factory-override lambdas pick up the
  `language: str | None` type.

### Task 4: docs, attribution, changelog, module maps

- [ ] `docs/guide/voice-mode.md`: default model row and backend column in the `-V`
  table (which keys apply to parakeet vs whisper), download size note, escape hatch
  (`-V model=small`), CC-BY-4.0 attribution line. **Rewrite the prose that becomes
  false for the new default:** the "Silence filtering" section (lines ~76-79)
  currently presents Silero VAD, `no_speech_prob`, and `avg_logprob` as the pipeline —
  restate it per backend (parakeet: energy gate + duration + blocklist + coarse mean
  token logprob; whisper: today's text); line ~28's "Named faster-whisper models may
  be downloaded" becomes backend-neutral.
- [ ] Plugin `README.md`: default-model paragraph + attribution line, and reword the
  faster-whisper/VAD framing at lines ~5, ~9, ~46 so the description matches the new
  default.
- [ ] Plugin `CLAUDE.md`: module map row for `_parakeet.py`; reword the
  `_transcriber.py` row (it is now whisper wrapper + backend selection + shared
  gauntlet pieces); invariant list gains `onnx_asr` in the lazy-import rule.
- [ ] Root `CLAUDE.md`: the plugins blurb says "microphone capture and faster-whisper
  transcription" — becomes backend-neutral ("local transcription", parakeet default).
- [ ] `CHANGELOG.md`: voice 0.3.0 entry (default flip, why, escape hatch, attribution),
  linking issue #324 and this plan, and **naming the breaking change**: with the new
  default model, explicit whisper-only options (`-V language=fr`, `-V compute=int8`,
  `-V asr_device=cuda`) now require also selecting a whisper model, where 0.2.0
  accepted them bare.

### Task 5: gates

- [ ] `uv run --no-sync ruff check plugins/inspect-robots-voice` and
  `ruff format --check`.
- [ ] `uv run --no-sync mypy --config-file plugins/inspect-robots-voice/pyproject.toml
  plugins/inspect-robots-voice/src/inspect_robots_voice
  plugins/inspect-robots-voice/tests`.
- [ ] `uv run --no-sync python -m pytest plugins/inspect-robots-voice/tests -q`.
- [ ] Full core suite untouched and green: `uv run --no-sync pytest --cov -q` (100%).
