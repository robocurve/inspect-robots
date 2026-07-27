# `inspect_robots` package — module map

Read `plans/0001-foundation-design.md` (§9–§11 are binding) before changing core
interfaces. The package is `mypy --strict` clean and ships `py.typed`.

## Modules

| Module | Responsibility |
|--------|----------------|
| `types.py` | `Observation`, `Action`, `ActionChunk`, `StepResult` (frozen, NumPy-native) |
| `spaces.py` | `Box`, `ObservationSpace`, `ActionSemantics`, `StateSpec` + canonical state vocab |
| `policy.py` | `Policy` Protocol + `PolicyBase` ABC, `PolicyInfo`, `PolicyConfig`; optional duck-typed `bind(embodiment_info)` hook for embodiment-adaptive policies plus `transcript()` and `transcript_delta()` hooks for complete and live per-trial audit records |
| `embodiment.py` | `Embodiment` Protocol + `EmbodimentBase` ABC, `EmbodimentInfo`, capability flags; optional duck-typed `bind_task(envelope)` hook for horizon-aware adapters (called by `eval()` after compat with a resolved step envelope; optional input — never fires on direct `rollout()`, keep a fallback) |
| `scene.py` | `Scene` (the Inspect `Sample` analog), `Target`, `ListSceneDataset` |
| `task.py` | `Task` (scenes + scorer + exactly one `max_steps`/`max_seconds` horizon), `Epochs`, `TaskEnvelope` (`resolve_envelope(control_hz)` — the adapter-safe identity+resolved-step limits view passed to `bind_task` hooks) |
| `scorer.py` | `Score`/`Scorer`, epoch reducers, builtin scorers (incl. operator/VLM) |
| `controller.py` | `Controller` middleware: `DefaultController` (open-loop chunking), `SmoothingController` |
| `approver.py` | `Approver` safety gate: `AutoApprover`, `ClampApprover`, `DeltaLimitApprover` (semantics-aware no-wild-swings limit), `ChainApprover` |
| `rollout.py` | `rollout()` closed loop, `TrialRecord`/`StepRecord`, per-trial seeding, best-effort normalized policy-transcript capture, and the duck-typed `transcript_delta()` to sink `log_policy_messages()` live-stream bridge; honors a policy-requested stop via pre-review `action.meta["request_stop"]` (truncation; embodiment termination wins; not preserved under ensembling) |
| `frames.py` | `FrameStore`/`FrameRef` — stream camera frames to disk (R5) |
| `transcript.py` | typed event stream (reset/inference/step/approval/operator/error) |
| `compat.py` | `check_compatibility`/`assert_compatible` — fail-fast before rollout |
| `conformance.py` | adapter conformance kit: `check_embodiment`/`assert_embodiment_conformant` for declarative guardrail/agent readiness; `missing_runtime_requirements` provides runtime-dependency preflight; `DeviceSlot`/`device_slots` declare and defensively read embodiment device slots |
| `errors.py` | error taxonomy (continue vs halt) |
| `eval.py` | `eval()` / `eval_set()` orchestration |
| `log.py` | immutable, schema-versioned `EvalLog` + `read_eval_log`, including per-trial policy transcripts parallel to epochs |
| `logging/` | `LogSink` protocol and optional duck-typed `log_policy_messages()` hook, `JsonLogSink` (atomic), optional `RerunSink` (non-blocking worker thread for steps and transcript rows; drops under pressure, never delays control) |
| `registry.py` | decorators + entry-point discovery; `_builtins.py` registers in-tree components |
| `cli.py` | `inspect-robots list` / `run` / `inspect` (with `--transcript` policy-audit rendering) / `view` (self-contained HTML report with optional stored-frame embedding via `_html.py`; `-o -` for stdout, `--open` for a browser) / `video` (frames-to-MP4 via `_video.py`) / `config set|show` / `setup` (first-run wizard) / `doctor` (adapter conformance), plus the zero-config form `inspect-robots "<instruction>"` (ad-hoc single-scene task; operator prompt on TTY). Every run wires guardrails (Clamp + DeltaLimit) by default; `--disable-guardrails` is the loud opt-out and the chain degrades per component with stderr warnings |
| `_html.py` | `render_html()`: a saved `EvalLog` as one self-contained HTML page, with exact-match stored-frame correlation and a shared payload budget (everything escaped once at interpolation, bounded JSON fallback); owns the shared chat-transcript predicates and status display map that `cli.py` imports |
| `_pngenc.py` | strict-uint8, NumPy plus stdlib PNG and data-URL encoding for stored camera frames |
| `_video.py` | `inspect-robots video`: reunite a log with its `FrameStore` side-cars and pipe them to the ffmpeg binary, one MP4 per (trial, camera) stream (plan 0016: stderr temp file not pipe, per-stream failure isolation, strict uint8) |
| `_defaults.py` | user default policy/embodiment (+ `--sim` counterpart) for the zero-config CLI: env vars > `~/.config/inspect-robots/config.ini` (INI — py3.10 has no tomllib; deliberately no project-local file); `set_default` backs `config set` |
| `_dotenv.py` | dependency-free `.env` parsing and working-directory auto-loading with real environment variables taking precedence |
| `_setup.py` | the `inspect-robots setup` wizard (plans 0009 and 0011): IO-injected prompts for `[defaults]`, plugin-declared V4L2/CAN/serial device slots with unplug-to-identify and CAN udev guidance, fallback camera discovery, headless-rerun warning; renders config.ini itself (comments survive) and carries unmanaged sections/keys through raw |
| `mock/` | dependency-free `CubePick` world + scripted/random/noop policies |

## Key invariants

- The rollout loop is **one control-rate loop** calling `Controller.next_action`;
  inference/replanning is controller-internal (so ensembling composes — R3).
- Frames live in a rollout-owned `FrameStore`, never in a sink (R5).
- Action *semantics* live on the action `Box`, not on every `Action` (R8).
- Generic policy/embodiment exceptions (incl. from `reset`) are wrapped into
  `PolicyError` / `EmbodimentFault`; a crashing approver becomes `SafetyAbort`;
  `SafetyAbort`/`EmbodimentFault` always halt the eval. Every error raised from
  inside a trial carries the partial `TrialRecord` on `exc.record`.
- `eval()` must always return/persist an `EvalLog` once rollouts have started —
  scorer/reducer failures degrade to an error log, never a crash. Errored
  trials are recorded (and delivered to sinks) but **never scored**. A run in
  which every trial errored ends with `status == "error"` even under the
  default `fail_on_error=False`.
- `eval()` closes embodiments it resolved from registry names ("close what we
  open"); caller-constructed objects are caller-owned.
- `mock/` and core must never import `rerun`/`torch` at module top.
