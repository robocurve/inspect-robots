# 0008 — `LLMAgentPolicy`: frontier LLMs as first-class policies

## 1. Goal

Let a frontier LLM (Claude, GPT, anything behind an OpenAI-compatible API)
control a robot by emitting tool calls that become smooth, approver-checked
action chunks — as a **registered `Policy`**, so the same agent that runs your
ad-hoc "place the fork on the plate" also scores on real `Task`s next to VLAs.
Pilot target: the bimanual YAM arms.

Works out of the box, no hardware (`cubepick` is core's mock world):

```bash
pip install inspect-robots inspect-robots-agent
export ANTHROPIC_API_KEY=sk-ant-...

inspect-robots "pick up the cube" --policy agent -P model=anthropic/claude-fable-5 \
    --embodiment cubepick
```

YAM pilot, after the yam-repo work in §6 lands (registry name is `yam_arms`):

```bash
pip install inspect-robots inspect-robots-agent "inspect-robots-yam[hardware]"
inspect-robots config set embodiment yam_arms   # once, per machine
inspect-robots config set policy agent          # once, per machine
export ANTHROPIC_API_KEY=sk-ant-...
export INSPECT_ROBOTS_MODEL=anthropic/claude-fable-5

inspect-robots "place the fork on the plate"
```

Every flag still works for one-off overrides (`-P model=openai/gpt-5.2`,
`--embodiment cubepick`), and the same policy drops into scored evals
unchanged:

```bash
inspect-robots run --task kitchenbench/tidy --policy agent -P model=anthropic/claude-fable-5
```

That last line is the point of doing this as a `Policy`: LLMs become an
evaluable policy family, comparable on any Task against fine-tuned VLAs,
reusing the entire rollout/approver/transcript/`EvalLog` stack instead of
growing a parallel vertical.

**Feasibility precedent:** multi-second inference between chunks is already how
this stack runs — the yam repo's MolmoAct client ships `timeout_s = 30.0`. An
LLM thinking for seconds and returning an open-loop `ActionChunk` is the same
shape, just slower and smarter.

Out of scope: an interactive chat/REPL mode (later sugar on top), cartesian/IK
primitives, and any non-OpenAI-compatible wire protocol.

## 2. Placement: what goes where

Follows the family convention (hardware adapters in satellite repos, policy
adapters as in-repo plugins; `plugins/inspect-robots-xpolicylab` is the
precedent — plan 0007).

| Piece | Where | Why |
|---|---|---|
| Safety approvers (`DeltaLimitApprover`, `ChainApprover`) | core `approver.py` | NumPy-only; the `Approver` seam already sits between every action and the embodiment, and its docstring reserves exactly this hardening |
| `dim_labels` on `ActionSemantics` | core `spaces.py` | NumPy-only; lets the agent (and future logging/viz) name action dimensions |
| `Policy.bind()` hook | core `policy.py` / `eval.py` | generic support for embodiment-adaptive policies |
| Policy-requested stop | core `rollout.py` | generic: a policy may conclude "done" before the horizon |
| `config set/show` subcommand | core `cli.py` / `_defaults.py` | dependency-free onboarding sugar; formalizes the config file `_defaults.py` already reads |
| CLI guardrails-by-default + embodiment cleanup | core `cli.py` | the user-facing safety switch must live where runs are launched |
| `LLMAgentPolicy`, LLM client, tool loop, motion primitives | new `plugins/inspect-robots-agent` | needs an HTTP client dep, so it cannot live in the NumPy-only core |
| YAM pilot enablement | `inspect-robots-yam` repo | its bimanual `Embodiment` exists (two CAN channels, `can0`/`can1` defaults, flat 14-D action space) but needs the §6 items before the pilot can run |

No new run-mode subcommand and no plugin-subcommand mechanism: the zero-config
ad-hoc instruction path (plan 0005) is already the right front door. The only
CLI addition is the `config` utility subcommand.

## 3. Core changes

All NumPy-only, all inside the 100% gate, each with an API-snapshot update.

### 3a. Safety approvers (`approver.py`)

- **`DeltaLimitApprover(action_space, max_delta=None)`** — the "no wild
  swings" gate, **semantics-aware** via the space's
  `ActionSemantics.control_mode`. The classification is total over the
  vocabulary (§3b adds `joint_delta`):
  - **Absolute-target modes** — `joint_pos`, `eef_abs_pose`: clamps each
    dimension to at most `max_delta[i]` away from the **last approved
    action** of the trial. The first action of a trial passes through
    un-delta-limited (documented: there is no trustworthy reference yet;
    bounds clamping and the plugin's interpolation cover it — see §8).
    Derived default: 5% of `(high - low)` per step — unit-agnostic (radians,
    meters, and normalized grippers all get proportionate limits).
  - **Displacement/rate modes** — `eef_delta_pos`, `eef_delta_pose`,
    `joint_delta`, `joint_vel`: the action *is* the per-step displacement
    (or rate), so limiting needs no reference. With an explicit `max_delta`,
    each dimension clamps to the **intersection** of the box and
    `[-max_delta[i], +max_delta[i]]` (well-defined for asymmetric boxes like
    a `[0, 1]` gripper dim). The derived default is the box alone — i.e.
    without an explicit override the limiter adds nothing beyond
    `ClampApprover`. Deliberately: core cannot assume displacement bounds
    are per-step-sized (that is an embodiment convention — true for
    `cubepick`, whose box is ±max-step and whose scripted oracle must keep
    passing; false for YAM's delta config today, which reuses absolute
    joint limits — a §6 fix). The guardrail value in these modes therefore
    depends on the embodiment declaring honest per-step bounds, or on
    `--max-action-delta`; the docs say so.
  - Constructing raises with a clear message when `semantics is None`, when
    a needed bound is missing/non-finite and no explicit `max_delta` (scalar
    or per-dimension array) is supplied, or when a pose mode
    (`eef_abs_pose`, `eef_delta_pose`) carries a
    `rotation_repr ∉ {none, rot6d}` — per-dimension clamping is invalid on
    quaternion/axis-angle/euler components (it can move the rotation axis
    arbitrarily), the same restriction `EnsemblingController` already
    enforces for per-dimension averaging. This approver never guesses; the
    CLI decides what to do about unlimitable spaces (§3e).
  - NaN → `SafetyAbort` (same contract as `ClampApprover`). A modified action
    is flagged `action.meta["delta_clamped"]` and returned as a new object so
    the rollout's identity check records an approval event. Per-trial
    reference state lives under a namespaced key in the rollout's `store`
    (created fresh per trial, so nothing leaks across trials/epochs).
- **`ChainApprover(*approvers)`** — runs approvers in sequence, feeding each
  the previous result. Gives "guardrails" one composable name:
  `ChainApprover(ClampApprover(space), DeltaLimitApprover(space))`.
- Rollout's approval-event detail extraction currently hardwires
  `meta.get("clamped")`; it is generalized to also surface `delta_clamped`
  (same step, §8 step 1).

Both approvers harden every policy — a fine-tuned VLA can wild-swing at least
as hard as an LLM. Python-API behavior is unchanged (`eval()` still defaults
to `AutoApprover`); the default flips at the CLI (§3e).

### 3b. `spaces.py` vocabulary: `dim_labels` + `joint_delta`

**`ControlMode` gains a `"joint_delta"` literal.** The current vocabulary
(`joint_pos`, `joint_vel`, `eef_delta_pose`, `eef_abs_pose`, `eef_delta_pos`)
cannot express joint-space displacement control — yet YAM's
`joints_are_delta=True` config *is* that mode while its semantics constant
declares `joint_pos` unconditionally. Misclassification is exactly the
dangerous case (an absolute-branch limiter and "hold = repeat state" applied
to a delta rig means `done()` commands a large motion), so the literal is
added here and YAM's semantics fix is scheduled in §6.

**`dim_labels`**: optional `dim_labels: tuple[str, ...] | None = None` on `ActionSemantics`.
When an action `Box` carries semantics with labels, `Box.__post_init__`
validates `len(dim_labels) == box.dim` (the pairing is only visible there).
YAM's repo will set
`("left_j0", …, "left_j5", "left_gripper", "right_j0", …, "right_gripper")`.
Embodiments without labels still work — the agent falls back to index-keyed
targets (`"3": 0.4`).

### 3c. `Policy.bind()` — embodiment-adaptive policies (`policy.py`, `eval.py`)

Compat checking compares `policy.info.action_space` against the embodiment's,
but a generic LLM policy has no fixed space — it adapts to whatever it drives.
New optional hook:

```python
def bind(self, embodiment_info: EmbodimentInfo) -> None: ...
```

`eval()` (and therefore the ad-hoc CLI path, which calls `eval()`) invokes
`bind` **after resolving both components and before `check_compatibility`**,
only when the policy defines it (`hasattr` duck-typing, same spirit as the
`Embodiment` Protocol; mypy narrows `hasattr` fine). `PolicyBase` gains a
no-op default. `LLMAgentPolicy.bind` copies the embodiment's action space
(with labels and semantics) and observation space into its own `PolicyInfo`,
so compat passes by construction and the tool schema is generated from the
true target.

### 3d. Policy-requested stop (`rollout.py`)

Today only the embodiment (`StepResult.terminated/truncated`) or the horizon
ends a trial; a policy that knows it is done can only burn steps. New,
policy-agnostic channel: after stepping an action, rollout checks the
**pre-review** action's `meta["request_stop"]` (the policy's intent must
survive any approver rewrite; the pre-review action is still in scope at that
point in the loop); if truthy, the trial ends with `truncated=True,
termination_reason=str(action.meta.get("stop_reason", "policy_stop"))`.
Embodiment-reported termination on the same step wins (it is ground truth).

Known limitation, documented on the feature: `EnsemblingController` rebuilds
emitted actions with the *chunk's* meta, dropping per-action meta — so
`request_stop` is honored under `DefaultController`/`SmoothingController`
(which preserve action identity/meta) but not under ensembling. Ensembling a
conversational agent policy is unsupported anyway; the docstring says so.

The LLM's `done(summary)` / `give_up(reason)` tools map to a single
hold-still action flagged `request_stop` (§4c defines hold-still per control
mode) — no empty-chunk special case, no `ActionChunk` change, no `Controller`
change. Scoring stays the scorer's job: `done()` ends the trial, it does not
declare success.

### 3e. CLI: guardrails by default, cleanup, `config` (`cli.py`, `_defaults.py`)

- **Guardrails on by default at the CLI**: `run` and ad-hoc instruction runs
  construct `ChainApprover(ClampApprover(space), DeltaLimitApprover(space))`
  from the resolved embodiment's action space unless `--disable-guardrails`
  is passed (which prints a prominent stderr warning and uses `AutoApprover`).
  `--max-action-delta` overrides the derived per-step limit (scalar,
  interpreted in the space's native units). The chain builder **degrades
  per component with a loud stderr warning instead of blocking**: a space
  with no bounds skips `ClampApprover`; a space where `DeltaLimitApprover`
  refuses to construct — for *any* of its §3a reasons (no semantics,
  unbounded dims without `--max-action-delta`, unsupported rotation repr,
  …) — skips the delta limiter, with the CLI catching the constructor's
  refusal generically rather than pre-checking an enumerated list; if
  nothing is applicable the CLI says plainly that no guardrails are active,
  names the actual refusal reason it caught, and states that reason's fix
  (declare semantics/bounds, pass `--max-action-delta`, or — for rotation
  reprs — fix the embodiment's declaration). Concretely: the isaacsim plugin's action box
  (absolute `joint_pos` semantics but no bounds) keeps running, with the
  warning, exactly as unprotected as it is today — never *less* protected
  than the status quo, and never silently. This satisfies "guardrails on by default, explicit
  flag to turn off" for **every** policy, not just LLMs, and sits below the
  model — no prompt injection or model error can emit an unclamped action.
  The run header states the active chain (or the degradation), and every
  clamp already lands in the transcript/log as an approval event; a
  dedicated `EvalSpec` field is deferred (schema change, not needed for
  safety).
- **The CLI closes what it constructs** — already landed on main
  independently of this plan (the `finally: embodiment.close()` in
  `_cmd_run`, widened to cover post-resolution validation; see the PR #30
  lead-up commits). Nothing to schedule; the guardrail-chain construction
  simply slots inside that existing `try`. Noted here because an earlier
  draft scheduled it and reviewers should not look for it in §9.
- **`config set KEY VALUE` / `config show`**: writes `policy`, `embodiment`,
  `sim_embodiment`, or `store_frames` (added to `[defaults]` on main by
  PR #30) under `[defaults]` in
  `~/.config/inspect-robots/config.ini` (stdlib `configparser`, atomic
  write-temp-then-rename, unknown sections preserved); `show` prints resolved
  defaults with sources. The existing guided error's fix line gains the
  `config set` spelling. No shipped hardware default, deliberately: core
  cannot name an optional package, and a tool that moves physical robots must
  not pick its target implicitly (plan 0005's trust stance).

## 4. The plugin: `inspect-robots-agent`

```
plugins/inspect-robots-agent/
  pyproject.toml            # deps: inspect-robots, httpx; static version 0.1.0
  src/inspect_robots_agent/
    __init__.py
    _llm.py                 # OpenAI-compatible chat client (httpx), provider resolution
    _tools.py               # tool schemas + parsing over the bound action space
    _motion.py              # control-mode-aware primitives → ActionChunk
    policy.py               # LLMAgentPolicy (bind/reset/act), registry entry point
  tests/                    # httpx.MockTransport + cubepick; no network, no hardware
```

Registered under the existing `inspect_robots.policies` entry-point group as
`agent` — exactly like the xpolicylab and yam policies. No new registry
machinery.

### 4a. LLM client (`_llm.py`)

Speaks the OpenAI chat-completions wire format (tools + tool_choice), which
covers OpenRouter, OpenAI, local vLLM/Ollama, and Anthropic's OpenAI-compat
endpoint. No provider SDKs — one `httpx` client; same "speak the protocol,
don't import the package" doctrine as plan 0007.

Model strings are OpenRouter-style `provider/model`, resolved from
`-P model=…` falling back to `$INSPECT_ROBOTS_MODEL`. Key/base-url
resolution, first match wins:

1. `-P base_url=…` (+ `-P api_key_env=NAME`, default `OPENROUTER_API_KEY`) — any compatible endpoint
2. `anthropic/*` model + `ANTHROPIC_API_KEY` → Anthropic compat endpoint
3. `openai/*` model + `OPENAI_API_KEY` → OpenAI
4. `OPENROUTER_API_KEY` → OpenRouter (any model string)

No model or no matching key → guided error naming the env vars, mirroring the
embodiment error. This is the whole "works out of the box with an OpenRouter /
Claude / OpenAI key" requirement. Transient HTTP failures retry with backoff;
persistent failure raises and is wrapped by the rollout as `PolicyError`.

### 4b. The policy (`policy.py`)

`LLMAgentPolicy(PolicyBase)` — conversation state is per-trial:

- **`bind(embodiment_info)`** adopts the embodiment's spaces (§3c) and builds
  the tool schema from the action `Box` (bounds, `dim_labels`, semantics —
  including which motion tool the control mode gets, §4c).
- **`reset(scene)`** starts a fresh conversation: system prompt (embodiment
  description, control mode, tool docs, safety notes, budget) + the scene's
  instruction as the user goal.
- **`act(observation)`** appends the observation (state fields with their
  keys, camera frames as image content blocks) to the conversation, calls
  the LLM until it produces a tool call (bounded retries on malformed output,
  then `PolicyError`), and returns the resulting `ActionChunk`.
- Tool errors (unknown label, non-finite value, bad `duration_s`) are
  structured messages the LLM sees and can correct — never exceptions.
- `-P` knobs: `model`, `base_url`, `api_key_env`, `max_llm_calls` (budget →
  `give_up` when exhausted), `temperature`. Core `PolicyConfig` has no such
  fields, so the plugin defines a frozen
  `AgentPolicyConfig(PolicyConfig)` subclass — `eval()` serializes configs
  with `dataclasses.asdict`, so the extra fields land in the log for free.

The LLM never sees raw actuation: its tool calls are *requests* that the
motion layer turns into bounded trajectories, and every emitted action still
passes the rollout's approver chain — the plugin contains no safety-critical
code path of its own.

### 4c. Motion primitives (`_motion.py`) — control-mode-aware

The tool surface and action synthesis are selected by
`ActionSemantics.control_mode` at bind time:

- **Absolute-target modes** (`joint_pos` — YAM's default — and
  `eef_abs_pose`): tool is
  `move_joints(targets: dict[str, float], duration_s: float)` with **named
  partial targets** — unnamed dims hold their current observed value;
  `{"right_j2": 0.4, "right_gripper": 1.0}` moves one arm of a bimanual
  robot; naming joints from both arms coordinates them in one call. Synthesis:
  linear interpolation from the current observed state to the target over
  `duration_s` at the embodiment's control rate. **Hold-still = repeat the
  current observed state.** Alignment rule (checked at `bind`): the
  embodiment's `StateSpec` must contain **exactly one field with
  `shape == (action_dim,)`** — that field is the proprioceptive reference
  (YAM's single flat 14-D `joint_pos` field satisfies it). Zero or multiple
  matching fields, or an embodiment without a `StateSpec`, is a bind error
  naming this rule.
- **Displacement modes** (`eef_delta_pos` — the mock `cubepick` world —
  `eef_delta_pose`, and `joint_delta`): tool is
  `move_by(deltas: dict[str, float], duration_s: float)` — the requested
  displacement is split evenly across the chunk's steps (per-step deltas stay
  small by construction). Unnamed dims get zero. **Hold-still = all-zeros
  action.** No state/action alignment needed.
- Pose modes additionally require `rotation_repr ∈ {none, rot6d}` — linear
  per-dimension interpolation/splitting is invalid on quaternion, axis-angle,
  or euler components (the same restriction `EnsemblingController` enforces
  for per-dimension averaging). Other representations, `joint_vel`, and
  anything else: unsupported at `bind` (clear message), until someone wants
  it. No embodiment in the repo family declares a pose mode today, so
  nothing regresses.

Either way, one tool call → one open-loop `ActionChunk` (`control_hz` from
the embodiment), which bridges "LLM thinks in seconds" to "robot steps at
10 Hz" — and per-step smallness is itself a guardrail before the approver
chain ever sees the actions. Chunk length is capped (e.g. 10 s worth of
steps); an over-long `duration_s` is a tool error, not a runaway chunk.
`done`/`give_up` emit the mode's hold-still action flagged `request_stop`
(§3d). Unsupported control modes fail at `bind` with a clear message rather
than at runtime.

## 5. Testing (no network, no hardware)

- **Core**: approvers (all six control modes' classification, derived and
  explicit `max_delta`, every constructor refusal — `semantics=None`,
  unbounded dims, unsupported rotation repr — NaN abort, first-action
  pass-through, per-trial reference reset), CLI chain degradation warnings
  (a bounds-less absolute space — isaacsim's shape; a synthetic
  semantics-less space; a synthetic quat-repr pose space; a
  fully-unlimitable one), `dim_labels` +
  `joint_delta` (incl. the ensembling averageable-set update), `bind`,
  `request_stop`
  (pre-review meta wins over approver rewrite; embodiment termination takes
  precedence; ensembling limitation documented), CLI guardrail default +
  `--disable-guardrails` + `--max-action-delta` (guardrail construction
  must not escape the existing close-embodiment `try`, asserted with a
  recording mock embodiment), and
  `config set/show` (tmp config home) — all pure NumPy/stdlib, straight into
  the 100% gate.
- **Plugin**: `httpx.MockTransport` scripts LLM conversations (tool calls as
  canned JSON) against `cubepick` end-to-end through real `eval()`:
  goals run to `done`, a scripted wild swing is delta-clamped (and isn't with
  guardrails disabled), budgets `give_up`, malformed tool calls retry then
  fail as `PolicyError`, and each provider-resolution rule picks the right
  base URL/key. The absolute-mode path (interpolation, named partial targets,
  bimanual packing) is exercised against a labeled joint-space mock
  embodiment defined in the plugin's tests. Plugin keeps its own coverage
  scope outside the core gate, like the other plugins.

## 6. YAM-repo enablement (separate repo, prerequisite for the pilot)

The pilot cannot run on the yam adapter as it exists today; these land as
`inspect-robots-yam` PRs alongside steps 7–9:

1. **Semantics fixes** (once the core release carrying §3b is tagged):
   `dim_labels` on its action space (14 labels, §3b order), and declare
   `control_mode="joint_delta"` when `joints_are_delta=True` — today the
   `joint_pos` constant is baked in regardless of the flag, which would send
   both the delta limiter and the agent's motion layer down the absolute
   branch on a delta-configured rig. Two coordinated consequences, both in
   scope for this item:
   - **Per-step displacement bounds**: the delta-mode action box currently
     reuses the absolute joint limits (±π per joint, `[0, 1]` gripper) — so
     the derived swing limit would be π/step, and `ClampApprover` on a
     `[0, 1]` *delta* gripper dim clamps every negative delta to 0, making
     the gripper impossible to open. Delta mode gets its own per-step box
     (symmetric per joint and gripper, conservative defaults).
   - **`MolmoAct2Policy` moves in lockstep**: it declares the same shared
     `ACTION_SEMANTICS` constant, and a `control_mode` mismatch is a hard
     compat error — the policy's declaration must become config-dependent
     too, or the currently-working molmoact2-on-delta-YAM pairing breaks.
2. **CLI-constructible cameras**: `YAMEmbodiment.reset()` currently raises
   unless a `camera_reader` is injected programmatically — and the agent
   policy needs frames. Add a camera factory configurable via scalar `-E`
   args (e.g. `-E cameras=top:0,left:2` for local capture devices) behind an
   optional extra.
3. **Install story**: an extra (e.g. `inspect-robots-yam[hardware]`) or
   documented step that pulls the i2rt driver — plain `pip install
   inspect-robots-yam` deliberately doesn't.
4. **Hold-behavior verification**: first hardware task is verifying that the
   arms hold position between chunks with the chosen config —
   `zero_gravity_mode` defaults to `True` (gravity-compensated/compliant),
   which may need to be off for agent runs. Documented in the quickstart.
5. **Quickstart** (§1's YAM block) in the README.

## 7. CI, workspace, release

Same integration as plan 0007 §8: uv workspace member (`uv lock` refreshed in
the same PR); a `test-agent-plugin` CI job added to `ci-ok.needs`; the
`core-only-import` job already proves core never imports it;
`publish-inspect-robots-agent` job in `release.yml` with `skip-existing`; new
PyPI trusted-publisher environment for the package.

## 8. Risks & mitigations

- **LLMs are poor low-level controllers.** Expected; the pilot's bar is
  "coherent, safe, visibly reasoned motion", not VLA-grade manipulation.
  Named partial targets + per-chunk images give the model its best shot;
  budgets bound the failure mode — and because it's a `Policy`, "how bad is
  it, exactly" is a scored, loggable question from day one.
- **First action of a trial is not delta-limited** (no reference yet, §3a).
  Accepted: bounds clamping still applies, the agent's interpolation starts
  from the observed state (so its first action ≈ current pose), and this
  matches the status quo for every existing policy. Revisit if the pilot
  shows first-step jumps.
- **Arm behavior between chunks is an assumption, not a fact** — verified as
  YAM-repo work item §6.4 before any untethered run. Ctrl-C cleanup is now
  guaranteed by the CLI's existing `finally: close()` (on main since
  PR #30's lead-ups; §3e), not assumed.
- **Prompt injection / model misbehavior** — cannot bypass the approver
  chain (it sits below the model, in rollout); worst case is in-bounds,
  rate-limited motion until a budget trips.
- **CLI guardrail default changes existing behavior**: out-of-bounds actions
  now clamp, and on absolute-target embodiments in-bounds *jumps* larger than
  the derived 5%-of-range step limit get rate-limited too — a real change for
  fast VLAs on such rigs, tunable via `--max-action-delta`. Displacement-mode
  embodiments are unaffected by default (derived limit = the bounds, §3a), so
  builtin demos like `cubepick-reach` behave identically. Accepted and
  deliberate (it is the safety requirement); release notes + run header make
  it visible, `--disable-guardrails` restores the old behavior, and the
  Python API is untouched.
- **Anthropic compat-endpoint drift** — the client is plain OpenAI wire
  format; if the compat endpoint diverges, OpenRouter remains the universal
  path and a native adapter can slot behind the same client interface later.

## 9. Execution steps (each a commit / PR-sized slice)

1. Core: `ActionSemantics.dim_labels` + `ControlMode` `"joint_delta"`
   literal (+ `Box` validation, tests, snapshot). Includes the knock-on in
   `EnsemblingController`: add `joint_delta` to `_AVERAGEABLE_MODES` (it is
   linearly averageable) — the rejection branch is `# pragma: no cover` on
   the premise that every literal is averageable, and the new literal must
   not silently invalidate that premise.
2. Core: `DeltaLimitApprover` + `ChainApprover` (classification needs
   step 1's vocabulary); generalize rollout's approval-event detail beyond
   `clamped` (+ tests, API snapshot).
3. Core: `Policy.bind()` hook in `eval()` (+ tests, snapshot).
4. Core: `request_stop` in rollout (+ tests, docstring noting the ensembling
   limitation).
5. Core: CLI guardrails-by-default (`--disable-guardrails`,
   `--max-action-delta`; construction inside the existing close-embodiment
   `try`), `config set/show`, guided-error text update (+ tests).
6. Plugin scaffold: pyproject, workspace membership, lockfile, CI job,
   release job.
7. Plugin: `_llm.py` client + provider resolution (mock-transport tests).
8. Plugin: `_tools.py` + `_motion.py`, both control modes (unit tests incl.
   labeled bimanual space).
9. Plugin: `LLMAgentPolicy` end-to-end through `eval()` on `cubepick`.
10. Docs: core README section; yam-repo PRs per §6.
