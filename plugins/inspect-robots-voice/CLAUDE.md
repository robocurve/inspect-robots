# `inspect-robots-voice` plugin agent guide

This workspace package adds local spoken operator feedback and local policy narration to Inspect
Robots runs. It registers the `voice` factory in `inspect_robots.operator_inputs` and the
`speaker` factory in `inspect_robots.sinks`; core owns both CLI lifecycles.

## Module map

| Module | Responsibility |
|--------|----------------|
| `src/inspect_robots_voice/__init__.py` | public factories, `-V` and `-S` scalar type validation, and package exports |
| `src/inspect_robots_voice/_capture.py` | sounddevice input-device resolution and bounded PortAudio callback queue with drop-oldest backpressure |
| `src/inspect_robots_voice/_segmenter.py` | pure NumPy adaptive energy gate with pre-roll, hangover, and hard utterance cap |
| `src/inspect_robots_voice/_transcriber.py` | faster-whisper wrapper, backend selection, and shared duration and hallucination rejection |
| `src/inspect_robots_voice/_parakeet.py` | onnx-asr Parakeet wrapper with timestamped confidence rejection |
| `src/inspect_robots_voice/_input.py` | worker-thread orchestration, generation-safe trial resets, output polling, and lifecycle hooks |
| `src/inspect_robots_voice/_tts.py` | lazy Kokoro engine seam plus pinned, verified, atomic model-file caching |
| `src/inspect_robots_voice/_speaker.py` | defensive note extraction, bounded narration queue, worker lifecycle, and lazy chunked playback |
| `tests/` | deterministic unit tests using fake devices, models, capture, playback, engines, transcribers, segmenters, and threading events |

## Invariants

- `sounddevice`, `faster_whisper`, `onnx_asr`, and `kokoro_onnx` are lazy imports. Never import
  them at module scope. Tests must run without PortAudio, audio hardware, model downloads, or
  importing those packages.
- `EnergyGate` is pure NumPy and has no I/O. The worker is the sole owner of its state and of
  capture-queue draining during normal operation.
- `begin_trial()` changes the generation and clears accepted output under the shared lock, then
  drains only the thread-safe capture queue. It never mutates segmenter internals.
- The generation check after transcription and the output append stay in one critical section
  under the same lock used by `begin_trial()`.
- `poll()` is non-blocking. Worker failures surface on the next poll so core can detach only the
  voice source while leaving typed console input alive.
- Voice is feedback-only. Polls always return `end=None`; spoken text cannot terminate or score a
  trial.
- `close()` is idempotent and never raises. It stops capture and joins the worker.
- `SpeakerSink.log_policy_messages()` never synthesizes or plays audio on the control thread. Its
  bounded queue drops the oldest stale note, and only a successful eval end drains before close.

## Working here

- Sync the workspace with `uv sync --locked --all-packages --extra dev`.
- Lint with `uv run --no-sync ruff check plugins/inspect-robots-voice`.
- Check formatting with `uv run --no-sync ruff format --check plugins/inspect-robots-voice`.
- Type-check source and tests with `uv run --no-sync mypy --config-file
  plugins/inspect-robots-voice/pyproject.toml plugins/inspect-robots-voice/src/inspect_robots_voice
  plugins/inspect-robots-voice/tests`.
- Run tests with `uv run --no-sync python -m pytest plugins/inspect-robots-voice/tests -q
  --cov=inspect_robots_voice --cov-report=term-missing --cov-fail-under=0`.

Prefer injected fakes and explicit threading events over sleeps. Segmenter tests use synthetic
NumPy blocks; capture and transcription tests call their injectable seams directly.
