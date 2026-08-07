# `inspect_robots` package — module map

Read `plans/0001-foundation-design.md` (§9–§11 are binding) before changing core
interfaces. The package is `mypy --strict` clean and ships `py.typed`.

## Modules

| Module | Responsibility |
|--------|----------------|
| `types.py` | `Observation`, `Action`, `ActionChunk`, `StepResult` (frozen, NumPy-native); operator-message extras carry `t`, `text`, and `source` provenance |
| `console.py` | frozen poll results with optional per-message source labels, the `OperatorInput` Protocol, and the threadless fd-level stdin polling primitive; the line grammar ends episodes on Esc (the `END_SENTINEL` line) or `/stop [note]` (note rides the message path), never bare Enter, which prints the constructor-selected usage reminder at most once per poll |
| `session.py` | attended-run owner that polls the console first, merges and stamps attached feedback-only input sources, permanently detaches a failing source, and owns verdict prompts, readiness gates, scrollback output, and the attended-TTY two-row footer with its owned input line, in-place status, and per-trial cbreak window; the footer's line editor resolves a bare Esc keypress into the end-of-episode sentinel, using a time-floored (150 ms, `_ESC_GRACE_S`) grace on quiet polls so split arrow-key sequences arriving within the floor never end the run, and confirms messages in an ending poll as `[noted]` |
| `spaces.py` | `Box`, `ObservationSpace`, `ActionSemantics`, `StateSpec` + canonical state vocab |
| `policy.py` | `Policy` Protocol + `PolicyBase` ABC, `PolicyInfo`, `PolicyConfig`; optional duck-typed `bind(embodiment_info)` hook for embodiment-adaptive policies plus `transcript()` and `transcript_delta()` hooks for complete and live per-trial audit records |
| `embodiment.py` | `Embodiment` Protocol + `EmbodimentBase` ABC, `EmbodimentInfo`, capability flags; optional duck-typed `bind_task(envelope)` hook for horizon-aware adapters (called by `eval()` after compat with a resolved step envelope; optional input — never fires on direct `rollout()`, keep a fallback) |
| `scene.py` | `Scene` (the Inspect `Sample` analog), `Target`, `ListSceneDataset` |
| `task.py` | `Task` (scenes + scorer + exactly one `max_steps`/`max_seconds` horizon), `Epochs`, `TaskEnvelope` (`resolve_envelope(control_hz)` — the adapter-safe identity+resolved-step limits view passed to `bind_task` hooks) |
| `scorer.py` | `Score`/`Scorer`, epoch reducers, builtin scorers (incl. operator/VLM) |
| `grader.py` | `Grader` Protocol — judgement *capture*, the other half of R6 (plan 0049): runs once per scored trial via the `before_scoring` seam, mutates the record, scorers stay pure readers; builtin `operator` grader wraps `OperatorSession.prompt_verdict` with a lazy default session and a `connect_session(session)` hook |
| `controller.py` | `Controller` middleware: `DefaultController` (open-loop chunking), `SmoothingController` |
| `approver.py` | `Approver` safety gate: `AutoApprover`, `ClampApprover`, `DeltaLimitApprover` (semantics-aware no-wild-swings limit), `ChainApprover` |
| `rollout.py` | `rollout()` closed loop, `TrialRecord`/`StepRecord`, per-trial seeding, delivered-once sourced operator input, best-effort normalized policy-transcript capture, the duck-typed `transcript_delta()` to sink `log_policy_messages()` live-stream bridge, and a duck-typed best-effort `end_trial()` hook in the per-trial `finally`; honors a policy-requested stop via pre-review `action.meta["request_stop"]` (truncation; embodiment termination wins; not preserved under ensembling) |
| `frames.py` | `FrameStore`/`FrameRef` — stream camera frames to disk (R5) |
| `transcript.py` | typed event stream (reset/inference/step/approval/sourced operator_message/operator/error) |
| `compat.py` | `check_compatibility`/`assert_compatible` — fail-fast before rollout |
| `conformance.py` | adapter conformance kit: `check_embodiment`/`assert_embodiment_conformant` for declarative guardrail/agent readiness; `missing_runtime_requirements` provides runtime-dependency preflight; `DeviceSlot`/`device_slots` and `OptionSlot`/`option_slots` declare and defensively read embodiment device and boolean option slots |
| `errors.py` | error taxonomy (continue vs halt) |
| `eval.py` | `eval()` / `eval_set()` orchestration, including source-preserving operator-message persistence with console fallback for old events; `grader=` (object or registry name) and `before_scoring=` are the same pre-scoring seam, mutually exclusive, validated against the `Grader` protocol |
| `log.py` | immutable, schema-versioned `EvalLog` + `read_eval_log`, including operator messages and policy transcripts parallel to epochs |
| `logging/` | `LogSink` protocol and optional duck-typed `log_policy_messages()` hook, `JsonLogSink` (atomic final log), `LiveLogSink` (atomic throttled schema-valid running snapshots, plan 0055), optional `RerunSink` (non-blocking worker thread for steps and transcript rows; per-trial arm-grouped blueprints from bound spaces, plan 0041; configurable live-viewer spawn port, plan 0044; drops under pressure, never delays control) |
| `registry.py` | decorators + entry-point discovery, including the `grader` and `operator_input` kinds; `_builtins.py` registers in-tree components |
| `cli.py` | `inspect-robots list` / `run` / `inspect` (with `--transcript` policy-audit rendering) / `summarize` (markdown learnings from a saved log via `_summarize.py`) / `view` (self-contained static or live-refreshing HTML reports with optional stored-frame embedding via `_html.py`; `-o -` for stdout, `--open` for a browser, `--serve` for the plan 0055 live cadence) / `video` (frames-to-MP4 via `_video.py`) / `config set|show` / `setup` (first-run wizard) / `doctor` (adapter conformance), with `--config` selection for config-reading commands, plus the zero-config form `inspect-robots "<instruction>"` (ad-hoc single-scene task; attended runs construct and connect one `OperatorSession`). Attended `run` and `eval-set` can resolve and attach duck-typed operator input through `--voice` and repeatable `-V`. `run` alone can resolve, start, attach, and close the `speaker` sink through `--speak` and repeatable `-S`. Grading is a per-run grader (`--grader` > config `grader` > `operator` when attended, plan 0049): every attended run is graded by default (registered tasks and eval-set included), `--grader none` opts out, `--no-prompt` suppresses the operator grader, and a config-sourced `operator` downgrades with a stderr note when the run cannot be attended. Every run wires guardrails (Clamp + DeltaLimit) by default; `--disable-guardrails` is the loud opt-out and the chain degrades per component with stderr warnings |
| `_html.py` | `render_html()`: a saved or running `EvalLog` as one self-contained HTML page, with live refresh/banner and pending-score rendering, exact-match stored-frame correlation, and a shared payload budget (everything escaped once at interpolation, bounded JSON fallback); owns the shared chat-transcript predicates and status display map that `cli.py` imports |
| `_summarize.py` | `inspect-robots summarize`: load inline or sidecar policy transcripts per trial, build an offline markdown digest, and optionally request a grounded learnings document over the OpenAI-compatible chat wire |
| `_pngenc.py` | strict-uint8, NumPy plus stdlib PNG and data-URL encoding for stored camera frames |
| `_video.py` | `inspect-robots video`: reunite a log with its `FrameStore` side-cars and pipe them to the ffmpeg binary, one MP4 per (trial, camera) stream (plan 0016: stderr temp file not pipe, per-stream failure isolation, strict uint8) |
| `defaults.py` | public reader for user default policy/embodiment (+ `--sim` counterpart): component env vars > selected config file; `INSPECT_ROBOTS_CONFIG` overrides its XDG/HOME-derived location (INI, py3.10 has no tomllib; deliberately no project-local file); `_set_default` backs `config set` |
| `_claims.py` | per-user advisory `flock` claims for normalized embodiment device-slot values, with lazy no-op behavior without `fcntl`, process-lifetime locks, and idempotent release |
| `_dotenv.py` | dependency-free `.env` parsing and working-directory auto-loading with real environment variables taking precedence |
| `_setup.py` | the `inspect-robots setup` wizard (plans 0009, 0011, 0032, 0040, and 0043): IO-injected prompts for `[defaults]`, plugin-declared V4L2/CAN/serial device slots with serial-pinned or port-pinned CAN udev rules, boolean option interviews, color-probed camera inventory grouped by sysfs USB device, trust-ladder naming, device-level unplug identify, headless-rerun warning; renders config.ini itself (comments survive) and carries unmanaged sections/keys through raw |
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
- The CLI releases device claims in the same `finally` that closes the
  embodiment.
- `mock/` and core must never import `rerun`/`torch` at module top.
