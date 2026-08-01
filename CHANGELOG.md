# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is
`0.x`, breaking changes may occur on any minor release.

## [Unreleased]

### Added

- **Agent plugin (0.20.0):** `-P wire=gemini-live` now serves
  `google/gemini-robotics-er-2-streaming-preview` through Google's stateful
  v1beta Live API. The sync websocket client streams only new observations,
  preserves exact tool-response ordering, recovers expired or dropped
  sessions from the image-free transcript, and captures each socket attempt
  with content-addressed frame blobs (plan 0039, #252).

- `inspect-robots view --serve` now serves a rendered logs directory, refreshes
  it incrementally as new runs arrive, and supports explicit host/port binding
  for local or remote browsing (plan 0037, #241)

- `inspect-robots view` now accepts a logs directory and incrementally renders
  self-contained per-run reports plus a searchable browsable index, with
  unreadable logs surfaced in place instead of aborting the whole directory
  ([plan 0035](plans/0035-view-log-directory.md), #234).

- Embodiments can contribute specialized approvers to the CLI's default
  guardrail chain through the new public `GuardrailContribution` API. Generic
  clamp and delta-limit gates remain first, contribution warnings stay visible,
  and `DeltaLimitApprover.rewind_reference` safely supports hold substitutions.
  The new public API is the rationale for the 0.31.0 minor release
  ([plan 0034](plans/0034-embodiment-contributed-guardrails.md), #232).

- Per-dimension `ActionSemantics.max_step` declarations let embodiments set
  absolute-control pacing and safety ceilings in native units. Default delta
  approval, agent-tool interpolation, and CaP-X motion queues honor mixed
  declared/range-derived limits while policy compatibility deliberately ignores
  embodiment-only declarations ([plan 0033](plans/0033-per-dim-max-step.md), #223).

- `OptionSlot` / `OPTION_SLOTS` (plan 0032): embodiment plugins can declare
  boolean behavior toggles that `inspect-robots setup` interviews as yes/no
  questions and writes into `[embodiment.args]`. First consumer:
  inspect-robots-yam's `auto_start` (yam#87).

- Policy connection failures now include an actionable action-server
  remediation hint in the recorded error message (#219).

- Wire capture: eval logs now record 100% of what the LLM saw. The agent
  policy captures every request/response attempt at the wire-client
  serialization point (tool schemas, evicted view, depth composites,
  `cache_control` breakpoints, retries) into `wire/<run_id>/` sidecars with
  content-addressed image blobs, on by default (`-P wire_capture=false` to
  opt out). New core `on_trial_start` policy hook (fail-safe: a raising
  hook errors the trial, never the eval), HTML report **Wire** section, and
  `inspect --wire` call-table/dump. Format contract in
  `inspect_robots_agent._capture`; guide in *Logging & Rerun* (#206, #207).

- Rerun sink: transcript `TextLog` rows at `trial/<scene>/e<epoch>/llm` are now
  paired with a markdown `TextDocument` at `…/llm/latest` — add a Text Document
  view for a wrapped, timeline-synced transcript reading pane (#203).

- `OPERATOR_END` termination-reason constant (`"operator_end"`): the standard
  reason for "a human ended this episode by keypress, verdict pending". Attended
  runs now prompt for exactly those trials — registered tasks and `eval-set`
  included (#194).

### Changed

- The `llm/latest` Rerun pane now shows only the step's assistant message(s);
  the full conversation remains in the `llm` TextLog stream
  ([plan 0037](plans/0037-rerun-latest-assistant-only.md), #243).

### Removed

- **`Task.control_hz`** (breaking). `rollout()` never actually paced the
  control loop to it — `_effective_control_hz` was dead code and the loop
  never slept — so the field only misled adapter authors and eval
  reproducibility records. The rollout now documents plainly that it applies
  no wall-clock pacing of its own; an embodiment that needs real-time cadence
  paces itself in `step()` and declares the `"self_paced"` capability. Plan
  0001 §9 R1 is updated to match (see the inline reversal note there for
  rationale). `compat`'s policy/embodiment rate-mismatch warning is
  unaffected.

### Fixed

- `inspect-robots setup` camera slots (#261, plan 0040): the wizard now lists
  and unplug-identifies cameras as physical USB devices. A camera whose color
  node lost udev's by-id name race (multi-interface cameras such as the
  RealSense D435) or whose by-path name is duplicated by systemd
  `usbv2-`/`usbv3-` aliases no longer vanishes from the listing or defeats
  `u`; shared-serial cameras are listed by port-stable by-path names, and a
  saved-but-dead by-id path now points the operator at the camera's current
  location by serial.

- **Agent plugin (0.19.1):** the chat wire now round-trips Gemini's
  `tool_calls[].extra_content` (`google.thought_signature`) into conversation
  history. Dropping it made Gemini reject any request ending on a tool
  message — e.g. after an `eef_pos` workspace-bounds rejection — with HTTP 400
  "Function call is missing a thought_signature", erroring the trial (#229,
  #230). Non-Gemini requests are unchanged.

- **Task validation rejects boolean `max_steps` values** — `Task(max_steps=True)`
  now raises `ConfigError` instead of silently converting `True` to a 1-step
  horizon (`bool` is a subclass of `int` in Python).

### Changed

- Camera frames in HTML reports now render as captioned responsive grid rows,
  with click-to-expand cells for closer inspection (plan 0036, #239).
- **Task horizon binding now follows compatibility checking.** An
  embodiment's optional `bind_task()` hook receives the resolved step envelope
  only after the policy/embodiment/task triple is known to be compatible. This
  prevents adapters from acting on a seconds-derived budget built from an
  invalid control rate (#160).
- **Agent plugin:** the `LLMAgentPolicy` constructor now raises `ConfigError`
  (not `ValueError`) for invalid `wire`, `speed`, `effort`, `max_output_tokens`,
  `max_llm_calls` and `max_speed_frac`, so `_resolve_or_exit` renders a guided
  message instead of letting the traceback reach the user. This matches the
  wire-gated checks added alongside them (#168).
- **Agent plugin:** runtime camera dropouts in `images=on_demand` now reject
  `take_pic` without treating a well-formed call as a tool error, so a single
  dropout no longer errors the trial. The first world-state rejection in one
  `act()` is free and later rejections escalate to the three-strike guard,
  bounding repeated capture refusals (#173).
- **Agent plugin:** tool results in `images=always` mode now follow the model's
  tool-call order, and extra calls are still never executed. Two things change:
  their result ordering relative to the executed call, and the reason string
  when the executed call itself failed, which is now
  `ignored: an earlier call in this turn failed` rather than
  `ignored: one tool call per turn`. Extras behind a successful call keep the
  original wording (#173).
- **Docs site migrated from MkDocs Material to Docusaurus.** The site at
  inspectrobots.org now builds from `website/` (Docusaurus 3) while the
  Markdown source stays in `docs/`; every existing URL, `llms.txt`, and
  `llms-full.txt` are preserved. The API reference is generated at build
  time by `scripts/gen_api_docs.py` (griffe) into a gitignored
  `docs/api/index.md`, guide pages link to it with anchor-checked
  `/api/#...` links, and the site now carries the project logo, favicon,
  and teal-on-cream branding. PR validation moved to a merge-blocking
  `docs-build` job in `ci.yml`; `docs.yml` deploys on pushes to main.
- **Agent plugin:** move tool calls now require a note describing the current
  observation and why the agent chose the motion, so users can follow its
  perception and decisions live and in saved transcripts (#130). This tightens
  the tool contract: a model that persistently omits the note errors the trial
  (unscored) after three consecutive failures, and each correction turn spends
  one `max_llm_calls` unit.

### Changed

- **Agent plugin:** outgoing requests now retain camera frames only from the
  newest two image-bearing messages by default, bounding long-episode
  payloads that previously grew until the API's request-size ceiling (HTTP
  413). Stored history, transcripts, and frame side-cars are unchanged.
  Restore the old unbounded behavior with `-P image_horizon=none` (#188).

### Added

- **Public user-defaults API:** `inspect_robots.defaults` lets plugin CLIs read
  the configuration written by `inspect-robots setup`, including config-file
  source paths and the args-owner metadata needed to apply hardware settings
  safely (#197).
- **Agent plugin:** per-camera metric depth from observation extras now renders
  as near-bright grayscale beside its RGB camera in automatic observations and
  `take_pic` reveals. Labels anchor the render with the 2nd–98th percentile
  bright/dim distances, valid-pixel percentage, and optional center depth;
  `-P depth=off` is the payload-cost kill-switch (#190).
- **Agent plugin:** the native Anthropic wire adds automatic prompt-cache
  breakpoints (system prompt, eviction boundary, final message) and records
  per-trial token/cache totals in `record.metadata["llm_usage"]`, making
  cache savings directly observable via `cache_read_input_tokens` (#188).
- **Seconds-based benchmark horizons:** `Task(max_seconds=...)` gives every
  compatible embodiment the same physical-time budget. `eval()` resolves it
  with `ceil(max_seconds * embodiment.info.control_hz)`, rejects missing or
  invalid control rates before `bind_task()` or rollout, and records both the
  declared seconds and resolved steps in eval logs, CLI summaries, inspection,
  and HTML reports (#160).
- **Grader notes:** a prompted operator verdict is now followed by one optional
  line of free text. Bare Enter records nothing, so a grader with nothing to add
  pays a single keypress. Notes reach the JSON log and the HTML report, a note
  is kept even on a trial the grader answered `skip`, and no note ever moves a
  score (#174).
- **Agent plugin:** `-P images=on_demand` lets the model request camera frames
  with `take_pic` instead of attaching every frame to every observation. A
  capture may follow one motion in the same assistant turn and is delivered
  from the post-motion observation; its narration reports observed playout,
  any missing cameras, and the measured remaining offset from absolute targets
  when proprioception is available. `images=always` remains the default
  (#173).
- **Policy lifecycle hook: `on_trial_end`** — policies can now hook into
  the end of a trial to persist state or artifacts. The orchestrator calls
  `policy.on_trial_end(record, log_dir, run_id)` and any metadata the policy
  attaches to `record.metadata` is persisted in the final `EvalLog`. Hook
  failures are caught and logged as trial errors, preventing them from
  crashing the overall evaluation (#40).
- **Agent plugin transcript persistence** — `LLMAgentPolicy` now implements
  `on_trial_end` to persist its full conversation transcript (tool calls,
  observations, system prompts) to a JSONL file per trial under
  `<log-dir>/transcripts/<run_id>/<scene_id>-e<epoch>.jsonl`. Camera images
  are stripped from the transcript to save space, as they are already
  recorded in the frame store. The relative path to the transcript is
  stored in the trial's metadata for easy post-hoc analysis (#40).
- **Agent plugin:** `-P wire=anthropic` selects Anthropic's native Messages
  API instead of its OpenAI-compat endpoint, which is the only way to reach
  fast mode: `-P speed=fast` runs Claude Opus 5 and Opus 4.8 at up to 2.5x
  higher output tokens per second, at roughly double the standard price. Robot
  control is latency-sensitive, so the arm spends less time waiting on the
  model. The wire also carries `-P max_output_tokens=` (the Messages API
  requires an output cap), replays thinking blocks so multi-turn trials hold
  together, and turns refusals and truncated responses into errors that name
  their own cause instead of looking like a missing tool call. Absent an
  explicit `-P base_url=...`, a model that resolves to any endpoint other than
  Anthropic's own is refused up front with the fix named, rather than 404ing
  on the first call (#165).
- **Agent plugin:** `-P wire=responses` selects the OpenAI Responses API wire,
  so reasoning effort works together with function tools on recent OpenAI
  models (Chat Completions rejects the combination, observed on
  `gpt-5.6-sol`). The chat-wire rejection now names the fix in its error
  message (#131).
- **`inspect-robots view LOG.json`**: render a saved eval log as a
  self-contained HTML report with run metadata, scores, scene results,
  collapsible policy conversations, highlighted agent notes, and the camera
  frames the model saw in `--store-frames` runs. `--no-frames` keeps
  placeholders and `--frames-budget` controls the inline payload limit (#132,
  #141).
- **`inspect-robots eval-set TASK [TASK ...]`**: run several registered tasks
  against one resolved policy/embodiment pair in a single invocation, matching
  task names exactly or by shell-quoted `fnmatch` glob (e.g.
  `'kitchenbench/*'`). Thin CLI wrapper over
  [`eval_set`][inspect_robots.eval.eval_set] that resolves the embodiment once
  for the whole set rather than once per task, and prints one status line plus
  a compact per-task row instead of a full summary per task (#45).
- Live agent-policy transcript rows on the Rerun `step` timeline, with
  best-effort non-blocking streaming and complete eval-log persistence (#124).
- Remote Rerun streaming via `inspect-robots run --rerun-connect [URL]`, so
  headless evaluations can connect over gRPC to a viewer on another machine
  (including through an SSH reverse tunnel) (#86).
- Plugin-declared embodiment device slots for V4L2 cameras, SocketCAN
  interfaces, and serial devices. `inspect-robots setup` probes and interviews
  declared slots, enforces grouped all-or-none assignments, and suggests udev
  serial pinning for order-dependent USB-CAN names (#61).
- Runtime-requirement declarations for registered component factories, with
  missing-import preflight checklists in `inspect-robots setup` and
  `inspect-robots doctor` (#59).
- isaacsim plugin: `_ensure_env`'s cfg-wiring contract (`parse_env_cfg`'s
  args, `gym.make(cfg=...)`, the `headless` → `_disable_debug_vis` gate, and
  the named-obs-terms request) is now exercised in CI via stubbed
  `gymnasium`/`isaaclab_tasks` modules. Previously only the fake-env-injected
  `step()`/`reset()` translation was covered, so a regression in `_ensure_env`
  itself (e.g. #15's missing `cfg=`) would only have failed live (#25).
- **`inspect-robots setup`**: an interactive first-run wizard that prompts
  for the `[defaults]` keys with suggested values, discovers camera devices
  under `/dev/v4l/by-id` (with unplug-to-identify and a `/dev/v4l/by-path`
  fallback for serial-less cameras that collide in by-id), and writes
  `~/.config/inspect-robots/config.ini`. An existing file is backed up to
  `config.ini.bak` and unmanaged sections/keys are carried through
  unchanged. Warns before writing `rerun = true` in a headless session
  (part of #50).
- Public-docstring coverage gate via Ruff's D1 rules, with a full backfill of
  missing public docstrings.

### Fixed

- **`DeltaLimitApprover` refuses `rot6d` rotation deltas in displacement pose
  modes** (#150, breaking for any embodiment currently declaring them). A
  `rot6d` delta's identity is `(1, 0, 0, 0, 1, 0)`, not the zero vector, so
  per-dimension clamping toward a symmetric `±max_delta` box drags it away
  from identity; the Gram-Schmidt re-normalization every consumer applies can
  then amplify the rotation instead of limiting it — the same failure class
  as clamping an absolute quaternion, and pre-existing behavior rather than a
  regression (`eef_delta_pose` + `rot6d` already reached the displacement
  clamp path before #143/#144). `euler_xyz` and `axis_angle` deltas have no
  such problem and remain guardrail-conformant.
- **An explicit invalid `--max-action-delta` now fails fast instead of silently
  running with weaker guardrails** (#154). Non-finite or non-positive values
  were previously caught by `_build_guardrails`'s degrade-per-component path
  (meant for derived limits an embodiment's space can't support) and
  downgraded to a stderr warning, so the run proceeded with clamp-only
  guardrails despite the operator explicitly asking for a tighter limit. Both
  `run` and `eval-set` now reject a malformed explicit value in the shared
  conflict check, before anything resolves or energizes. Derived-limit
  degradation (no explicit flag, an embodiment declaring no bounds) is
  unaffected — it still warns and continues.
- **`--epochs 0` or a negative value now exits with a guided error instead of a
  raw traceback** (#145). Both `inspect-robots run` and `inspect-robots eval-set`
  catch the `ConfigError` raised by `Task`'s epoch validation and surface it
  through the existing `_resolve_or_exit` pattern, matching how invalid
  constructor kwargs are handled for config-file components (#47).
- **`DeltaLimitApprover` no longer rejects displacement pose modes whose
  rotation deltas are safe to clamp per dimension** (#143). The per-dimension
  rotation-repr refusal now fires for absolute pose modes (`eef_abs_pose`,
  where clamping an absolute orientation has wraparound and axis-coupling
  problems) and, separately, for quaternion deltas in displacement pose modes
  (`eef_delta_pose` + `quat_wxyz`/`quat_xyzw`, whose identity is not the zero
  vector, so per-dimension clamping distorts the rotation instead of limiting
  it). Euler and axis-angle deltas have no such problem and clamp fine, so an
  euler-delta embodiment (e.g. BridgeData V2's 7-D xyz+euler deltas) is now
  guardrail-ready: `doctor` reports it conformant, and CLI runs keep delta
  limiting instead of silently degrading to clamp-only.
- **Operator scoring no longer prompts twice for self-confirming embodiments**
  (#53). On interactive ad-hoc runs, definitive `success` or `failure`
  termination verdicts are adopted as the operator judgement, announced on the
  terminal, and identified as embodiment-sourced in the in-memory transcript.
- **Literal percent signs in config values now round-trip unchanged** (#54).
  Config reads no longer treat `%` as interpolation syntax, so values such as
  `policy = 50%off` work with `config set`, `config show`, and normal runs.
- **Component argument mistakes now fail cleanly and stale args are flagged**
  (#47). Changing a configured component warns when its non-empty args section
  still belongs to the old name, and invalid constructor kwargs exit with
  guidance to check the config section or CLI args flag instead of a traceback.
- **`inspect-robots run` now surfaces evaluation failures in its summary**:
  top-level errors, per-scene failure context, and a ready-to-run postmortem
  `inspect` command are printed after unsuccessful runs (#57).
- **Config `[*.args]` sections no longer follow a differently-selected
  component** (#44). `[policy.args]` / `[embodiment.args]` /
  `[sim_embodiment.args]` now apply only when the selected component matches
  the `[defaults]` name they were configured alongside; selecting another
  component (by flag or env var) ignores them with a stderr note instead of
  crashing its constructor with foreign kwargs. Selecting the configured
  default explicitly (e.g. `--embodiment` naming the config default) still
  applies its args.

## [0.6.0] - 2026-07-10

### Added

- **New plugin: `inspect-robots-agent`** — frontier LLMs (Claude, GPT,
  anything behind an OpenAI-compatible API) drive any registered embodiment
  through tool calls, as the first-class policy `agent`
  (`--policy agent -P model=anthropic/claude-fable-5`). Each tool call becomes
  one smooth, approver-checked action chunk (`move_joints` with named partial
  targets for absolute control, `move_by` for displacement control;
  `done`/`give_up` end the trial). One `httpx` client speaks the wire format;
  keys resolve from `$ANTHROPIC_API_KEY` / `$OPENAI_API_KEY` /
  `$OPENROUTER_API_KEY` or a custom `base_url` (plan 0008).
- Safety approvers: `DeltaLimitApprover` (semantics-aware "no wild swings"
  per-step limiting) and `ChainApprover` (sequential composition) join
  `ClampApprover` in `inspect_robots.approver`.
- **CLI guardrails on by default**: every `run`/ad-hoc invocation wires
  `ChainApprover(ClampApprover, DeltaLimitApprover)` from the embodiment's
  action space; `--disable-guardrails` is the explicit, loudly-warned opt-out
  and `--max-action-delta` tunes the per-step limit. The chain degrades per
  component with stderr warnings (never blocking, never silent).
- CLI: `inspect-robots config set KEY VALUE` / `config show` persist and
  display `[defaults]` config keys; guided errors now point at `config set`.
- `ActionSemantics.dim_labels` names action dimensions (validated against the
  owning `Box`); `ControlMode` gains `"joint_delta"` for joint-space
  displacement control.
- Policies may define an optional `bind(embodiment_info)` hook — `eval()`
  calls it after resolution and before the compatibility check, so
  embodiment-adaptive policies (like the LLM agent) can adopt the
  embodiment's spaces.
- Adapter conformance kit (`inspect_robots.conformance`):
  `check_embodiment` / `assert_embodiment_conformant` verify an embodiment's
  declared spaces are guardrail-ready and agent-ready (semantics, finite
  bounds, unique `dim_labels`, aligned `StateSpec` for absolute modes,
  limitable rotation reps). Adapter repos enforce it with one CI test; the
  new `inspect-robots doctor --embodiment NAME` command audits installed
  adapters the same way. The `CubePick` mock now labels its dims (`dx`/`dy`)
  and passes its own kit. See the new adapter authoring guide
  (`docs/guide/adapters.md`) for the non-mechanical half (honest control
  modes, per-step delta bounds, hold-behavior verification).
- Rollout honors a policy-requested stop via the pre-review action's
  `meta["request_stop"]` (ends the trial as a truncation; embodiment
  termination wins; not preserved under ensembling).

### Fixed

- The CLI exits with the guided message (not a traceback) when a component
  factory raises `ConfigError` during resolution.

## [0.5.0] - 2026-07-10

Backfilled: this version was released tag-only; the entries were reconstructed
from the merged PRs.

### Added

- CLI: `--rerun` flag and `rerun` config default open a live Rerun viewer
  streaming cameras, state, and actions for each run (#36).
- CLI: `store_frames` config default and per-run frame directories under
  `<log-dir>/frames`; `--store-frames` became tri-state so `--no-store-frames`
  overrides the config (#30).
- CLI: minimal ANSI styling on interactive terminals; plain output when piped
  or `NO_COLOR` is set (#37). `inspect-robot` is accepted as an alias for the
  common typo (#34).

### Fixed

- The CLI closes the embodiment it resolves, even when `eval()` raises: a
  real robot never stays energized after a crashed run (#30).

## [0.4.0] - 2026-07-09

Plugin releases alongside this version: `inspect-robots-xpolicylab` 0.1.0
(first release) and `inspect-robots-isaacsim` 0.1.1 (ships the env-creation
fix below).

### Added

- **New plugin: `inspect-robots-xpolicylab`** — a `Policy` adapter for
  [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) policy servers,
  making its zoo of 40+ served VLAs (π0/π0.5, GR00T, OpenVLA-OFT, RDT-1B,
  SmolVLA, ACT, …) evaluable with any Inspect Robots embodiment
  (`--policy xpolicylab -P url=ws://host:19000`). Speaks XPolicyLab's
  msgpack-over-websocket protocol directly — no `xpolicylab` install needed
  on the eval side.
- CLI: `inspect-robots run` gained `--epochs`, `--fail-on-error`, and
  `--store-frames`; the written log's path is printed at the end of a run.
- Tests are now type-checked under strict mypy (`files = ["src/inspect_robots",
  "tests"]`).
- CI: a blocking `test-rerun` job installs the real `rerun-sdk` and runs
  `test_rerun_sink.py` against it — previously `RerunSink` was only exercised
  against a fake `rerun` module, so a real SDK API change would go unnoticed
  (#6). The `plugin-isaacsim` CI job gained a `ruff format --check` step and
  now reports (but does not gate on) its own test coverage.

### Changed

- **Documentation site moved to a custom domain:** <https://inspectrobots.org/>.

### Fixed

- **`EvalLog` and friends are now actually immutable.** `EvalLog`, `EvalSpec`,
  `EvalStats`, `EvalResults`, and `SceneResult` are frozen dataclasses, and
  `SceneResult.epochs`/`operator_judgements` and `EvalLog.samples` are tuples
  instead of lists — previously nothing stopped e.g. `log.samples.clear()`
  despite the "immutable EvalLog" documentation (#4). `read_eval_log` coerces
  older on-disk logs (whose JSON arrays deserialize as lists) back into tuples,
  so the read-back guarantee is unaffected.
- **isaacsim plugin: real env creation was broken.** `_ensure_env` called
  `gym.make(task_id)` without the mandatory Isaac Lab `cfg` object, so every
  live run failed with `missing 1 required positional argument: 'cfg'`; the
  config is now resolved via Isaac Lab's own `parse_env_cfg`. Alongside it:
  observation groups are requested as *named* dicts (`concatenate_terms=False`
  — a flat tensor left `Observation.state` empty; a warning fires when the
  request can't be honored), and headless runs disable every `debug_vis` flag
  (markers exist for a viewport nobody has, and their material machinery can
  hang env creation on hosts with a broken render stack).
- **Eval logs are strict RFC 8259 JSON.** Non-finite floats (e.g. an inf
  `min_distance_to_goal` when no distance was ever recorded) are mapped to
  `null` at the JSON boundary, so `jq` and other conforming parsers accept the
  file; `json.dump(..., allow_nan=False)` stays on as a regression backstop.
  In-memory scores keep the inf sentinel.
- **`ClampApprover` hardening:** a NaN action raises `SafetyAbort` (a NaN has
  no meaningful clamp and must never reach hardware) while `±inf` clamps to the
  finite bound like any out-of-range value; one-sided boxes (`low`-only /
  `high`-only) are honored instead of ignored; an unmodified action is returned
  as the *same* object so the rollout's identity-based `approval_event` stays
  accurate.
- **Never lose the log.** `eval()` always produces and persists an `EvalLog`
  once rollouts have started: scorer/reducer failures degrade the run to an
  error log instead of crashing; `policy.reset`/`embodiment.reset` failures are
  wrapped into the error taxonomy; every error raised from inside a trial
  carries the partial `TrialRecord` on `exc.record` (recorded and delivered to
  sinks — errored trials are never scored); `on_trial_end` fires for halted
  trials too.
- **A crashing approver now halts the eval as `SafetyAbort`** — an approver that
  crashed cannot vouch for safety — and approved-but-modified actions emit an
  `approval_event`.
- **`eval()` owns what it opens:** an embodiment resolved from a registry name
  is closed when the run finishes (even on a halt); caller-constructed
  embodiments stay caller-owned.
- **`fail_on_error` is checked after every trial** (Inspect semantics:
  `True` = first error, `0<x<1` = proportion, `x>1` = count), not just at the
  end of the run.
- `derive_seed`: `seed=None` no longer aliases `seed=0` — unseeded runs draw a
  fresh OS seed and record it in the log.
- `Task`/`Epochs`/`Box`/`ObservationSpace` validate their configuration at
  construction (`max_steps`/`epochs` must be positive, `Box` bounds must be
  elementwise ordered, `state_keys` must agree with `StateSpec`), raising
  `ConfigError`/`ValueError` instead of failing mid-eval. `Task.scorer` also
  accepts registry names.
- Inference events no longer overstate `chunk_len` when `replan_interval`
  exceeds the chunk; the ensembling no-semantics warning fires per instance
  (at construction) instead of once per process.
- Collision-safe frame-file slugs (camera names and trial ids are fully
  sanitized); broken plugin entry points warn loudly instead of being silently
  skipped.
- Rerun sink: per-trial namespacing, new-SDK (`>=0.23`) compatibility, and a
  correct install hint.

## [0.3.0] - 2026-07-01

### Changed

- **Renamed the framework RoboInspect → Inspect Robots.** The import package is
  now `inspect_robots`, the distribution/CLI `inspect-robots`, the error base
  class `InspectRobotsError`, the log field `inspect_robots_version`, and the
  plugin entry-point groups `inspect_robots.*`. The Isaac Sim plugin follows as
  `inspect-robots-isaacsim` (import package `inspect_robots_isaacsim`, entry
  point group `inspect_robots.embodiments`).

## [0.2.0] - 2026-06-30

### Added

- **Isaac Sim / Isaac Lab plugin** as an in-repo uv-workspace package
  (`plugins/`): an `Embodiment` adapter backed by an Isaac Lab physics
  simulation (default profile: 7-DoF Franka Panda under joint-position control
  with a binary gripper), registered via entry point, with Isaac imported
  lazily so the plugin installs anywhere and the core stays NumPy-only.
  First-party plugins live as their own packages with their own pyproject,
  tests, and coverage scope; `uv sync --all-packages --extra dev` installs
  core + plugins editable.

### Changed

- Renamed the package RoboLens → RoboInspect (superseded by the 0.3.0 rename).

## [0.1.0] - 2026-06-27

### Added

- **Widened the public API for plugin authors.** `inspect_robots.__all__` now exports
  the authoring primitives directly — `Task`/`Epochs`, `Scene`/`Target`,
  `Scorer`/`Score` and the builtin scorers, `Policy`/`PolicyBase`/`PolicyInfo`/
  `PolicyConfig`, `Embodiment`/`EmbodimentBase`/`EmbodimentInfo`, the
  `types`/`spaces` dataclasses, `TrialRecord`, and the `@task`/`@policy`/
  `@embodiment`/`@scorer`/`@sink` registry decorators plus `registered`/`resolve`.
  Out-of-tree benchmarks (e.g. KitchenBench) and adapters can now `from inspect_robots
  import Task, Scene, task, ...` against a stable surface.

- **Core framework foundation.** The two-input model for robotics evals:
  `Policy` (VLA) and `Embodiment` (real robot or simulator), with a benchmark
  `Task` defined independently of both.
- **Types & spaces:** `Observation`, `Action`, `ActionChunk` (open-loop chunked
  execution), `StepResult`; `Box`/`ObservationSpace`, `ActionSemantics`, and a
  canonical proprioception `StateSpec` vocabulary.
- **Scenes & scoring:** `Scene`/`Target` datasets; `Scorer`/`Score` with an
  epoch-reducer split (`mean`/`median`/`max`/`min`/`mode`/`pass_at_k`); builtin
  scorers including `success_at_end`, `min_distance_to_goal`, `reached_goal_state`,
  and an operator-verdict scorer; reserved `VLMScorer` interface.
- **Rollout engine:** open-loop chunk execution via a composable `Controller`
  middleware layer (`DefaultController`, `SmoothingController`,
  `EnsemblingController` for ACT/ALOHA temporal ensembling); an `Approver`
  safety gate (`AutoApprover`, `ClampApprover`); an error taxonomy
  (`PolicyError` continue vs `EmbodimentFault`/`SafetyAbort` halt); a typed
  transcript; per-trial seeding; and a `FrameStore` that streams frames to disk.
- **Compatibility checking:** fail-fast action/observation/semantics checks with
  key remapping, control-rate reconciliation, and scene realizability.
- **`eval()` / `eval_set()`:** Inspect-style orchestration returning immutable,
  schema-versioned `EvalLog`s with `fail_on_error` semantics; atomic JSON logs
  with a read-back guarantee; optional frame side-cars.
- **Registry & plugins:** decorators and `importlib.metadata` entry-point
  discovery so out-of-tree backends register without being imported.
- **Logging sinks:** canonical `JsonLogSink`; optional, lazily-imported
  `RerunSink` for [Rerun](https://github.com/rerun-io/rerun) visualization.
- **CLI:** `inspect-robots list`, `inspect-robots run`, and `inspect-robots inspect <log>`.
- **String resolution:** `eval()`/`eval_set()` accept registry names
  (`eval("cubepick-reach", "scripted", "cubepick")`) in addition to objects.
- Dependency-free `CubePick` mock world and scripted/random/noop policies.
- **Documentation site** (MkDocs + Material + mkdocstrings) auto-generated from
  docstrings, deployed to GitHub Pages, with guides, an API reference, and
  `llms.txt` / `llms-full.txt` for LLM consumers. Homepage-style README.
- **100% test coverage**, enforced by `--cov-fail-under=100` in CI (a blocking PR
  check). Genuinely unexecutable lines (Protocol stubs, `__main__` guards,
  defensive branches) are excluded via `tool.coverage.report`.
- **Pre-commit hooks** (`.pre-commit-config.yaml`): ruff (lint + format) and
  strict mypy on commit, the 100% coverage gate on push. Install with
  `uv run pre-commit install`. Documented in `CONTRIBUTING.md`.

[Unreleased]: https://github.com/robocurve/inspect-robots/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/robocurve/inspect-robots/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robocurve/inspect-robots/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robocurve/inspect-robots/releases/tag/v0.1.0
