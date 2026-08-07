# `inspect-robots-voice` plugin agent guide

This workspace package adds local spoken operator feedback to attended Inspect Robots runs. It
registers the `voice` factory in `inspect_robots.operator_inputs`; core owns CLI lifecycle and
merges its non-blocking polls with the typed console.

## Module map

| Module | Responsibility |
|--------|----------------|
| `src/inspect_robots_voice/__init__.py` | public factory, `-V` scalar type validation, and package exports |
| `src/inspect_robots_voice/_capture.py` | sounddevice input-device resolution and bounded PortAudio callback queue with drop-oldest backpressure |
| `src/inspect_robots_voice/_segmenter.py` | pure NumPy adaptive energy gate with pre-roll, hangover, and hard utterance cap |
| `src/inspect_robots_voice/_transcriber.py` | faster-whisper wrapper, backend selection, and shared duration and hallucination rejection |
| `src/inspect_robots_voice/_parakeet.py` | onnx-asr Parakeet wrapper with timestamped confidence rejection |
| `src/inspect_robots_voice/_input.py` | worker-thread orchestration, generation-safe trial resets, output polling, and lifecycle hooks |
| `tests/` | deterministic unit tests using fake devices, models, capture, transcribers, segmenters, and threading events |

## Invariants

- `sounddevice`, `faster_whisper`, and `onnx_asr` are lazy imports. Never import them at module
  scope. Tests must run without PortAudio, audio hardware, model downloads, or importing those
  packages.
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
