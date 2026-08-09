# Voice: subprocess capture fallback when PortAudio is absent — Design Sketch

> Status: SKETCH — deliberately unimplemented until the first voice-mode rig shakedown
> finishes and confirms the friction is worth this complexity (decision owner: Jay).
> The cheap layers already exist on this branch: the PortAudio load failure names
> per-OS install commands, and the docs carry a prerequisites section.

**Problem:** `pip install inspect-robots-voice` is not sufficient on Linux; sounddevice
needs the PortAudio system library and end users hit `PortAudio library not found` at
startup (observed on the omen rig, 2026-08-06). The friendly error helps, but the truly
frictionless path is to not need the library at all.

**Idea:** when the sounddevice import fails (and only then), fall back to spawning a
system audio CLI and reading raw PCM from its stdout, feeding the existing block queue:

1. `pw-record --format=f32 --rate=16000 --channels=1 -` (PipeWire; present on omen)
2. `parecord --raw --format=float32le --rate=16000 --channels=1` (PulseAudio)
3. `arecord -f FLOAT_LE -r 16000 -c 1 -t raw -` (ALSA; also present on omen)

First binary found wins. A `SubprocessCapture` class implements the same contract as
`MicrophoneCapture` (`start()`, `close()`, `device_name`, blocks into the bounded queue
from a stdout-reader thread), selected inside a small `open_capture()` factory that
`VoiceInput` calls instead of constructing `MicrophoneCapture` directly.

**Open questions to resolve before implementing (with rig data):**

- Device selection: `-V device=` semantics differ per tool (`--target` for pw-record,
  `--device` for parecord, `-D` for arecord). Probably: pass-through string, document
  per-tool meaning, keep index-based selection PortAudio-only.
- Backpressure: stdout pipe buffering replaces the PortAudio callback; the reader thread
  must apply the same drop-oldest policy so a stalled transcriber cannot grow the pipe.
- Failure modes: tool exits (device unplugged, server down) must surface through the
  existing worker-error-on-next-poll path, not hang the reader.
- Windows: none of these tools exist; fallback stays POSIX-only, error text unchanged.
- Testing: fake `Popen` seam mirroring the fake-sounddevice pattern; no real audio in CI.

**Explicitly rejected for now:** switching the primary backend away from sounddevice
(macOS and Windows wheels bundle PortAudio, so the primary path is zero-setup there),
and vendoring a PortAudio binary in the wheel (manylinux audio-stack linkage risk).
