# Plan 0031: agent plugin action pre-check (correctable rejection before emission)

Issue: [#210](https://github.com/robocurve/inspect-robots/issues/210)
Status: revised after adversarial critique round 1 (9 findings, all
addressed below).

## 1. Problem

Framework approvers have two outcomes: silently modify an action (clamp,
hold) or `SafetyAbort` the eval. Neither tells the policy anything. For LLM
policies this wastes the plugin's best asset: the correctable-error loop in
`LLMAgentPolicy` where any `ToolResult.error` goes back to the model as a
tool message and the turn continues (`policy.py`, `_MAX_CONSECUTIVE_FAILURES`
guarded). The plugin already pre-empts guardrails structurally (per-step
limits mirror `DeltaLimitApprover`); it has no hook for *semantic* checks
like "this sweep collides".

First consumer: the YAM collision checker
(robocurve/inspect-robots-yam#85, merged). Its `CollisionChecker.check`
answers a 14-D configuration query in ~16 us, so checking a whole chunk adds
well under a millisecond per tool call.

## 2. Goals

- A pluggable pre-check the embodiment side (or the user) can hand to
  `LLMAgentPolicy`, run on the exact action waypoints a `move` tool call
  would emit, before the chunk leaves the toolset.
- Rejection is a first-class correctable tool error: the model is told why,
  retries within the same `act()` turn, and the rollout never sees the bad
  chunk. Framework approvers stay in the chain as the hard backstop; the
  pre-check is UX, not the safety guarantee.
- Zero new dependencies; the plugin stays httpx-only. The pre-check is an
  injected callable, like `transport` and `env` already are.

## 3. Non-goals

- No support for displacement control modes in v1. `_move_displacement`
  emits per-step *delta* vectors; handing those to a configuration checker
  invites the deltas-read-as-configurations bug class that plan 0011 (yam)
  guards against. If `pre_check` is set and the bound mode is a
  displacement mode, `build_toolset` raises `ToolsetError` at bind time.
  Absolute modes (`joint_pos`, `eef_abs_pose`) are supported.
- No CLI wiring. `-P` flags carry scalars; a callable cannot cross that
  boundary. Programmatic construction only, documented (same limitation as
  framework approvers).
- No approver-to-policy callback in the core framework. This stays a
  policy-side pre-emption; core rollout semantics are untouched.
- No packaged adapter for the YAM checker inside this plugin (it would
  invert the dependency direction). The README shows the three-line adapter
  instead.

## 4. Design

### 4.1 The hook

```python
PreCheck = Callable[[npt.NDArray[np.float64]], str | None]
```

- Input: a `(steps, dim)` float64 array of the exact action vectors the
  toolset is about to emit for one `move` call, in the bound action
  space's semantics (absolute targets, already clipped to bounds, first
  row is the first commanded waypoint, last row is the final target). The
  array is a stacked copy with `writeable=False` set, so a mutating
  checker fails loudly and can never corrupt the emitted chunk.
- Return `None` to allow. Return a non-empty human-readable string to
  reject; the string becomes the tool error verbatim, so it should say what
  was wrong and ideally where (the YAM adapter includes the geom pair and
  waypoint index).
- Exceptions propagate unchanged, stated precisely: the rollout wraps
  *generic* exceptions from the policy as `PolicyError` (trial fails,
  `fail_on_error` applies), while typed Inspect Robots errors keep their
  own semantics — a pre-check that raises `SafetyAbort` halts the whole
  eval, exactly as if an approver had raised it. Both are acceptable
  outcomes for a crashing or hard-vetoing checker; what is not acceptable
  is a silent allow. The README documents the distinction so adapter
  authors choose deliberately. (The YAM `CollisionChecker.check` raises
  `SafetyAbort` only on non-finite input, which cannot occur here: every
  waypoint is clipped against finite bounds before the hook runs.)

### 4.2 Plumbing

- `LLMAgentPolicy.__init__` gains keyword `pre_check: PreCheck | None =
  None`, validated with `callable(pre_check)` and a guided `ConfigError`
  otherwise (the message states that `-P` flags cannot carry callables and
  the hook is programmatic-only, following the `prior_learnings` coercion
  guard precedent), stored, and passed to `build_toolset` in `bind()`
  alongside the existing space arguments.
- Recorded config: `AgentPolicyConfig` gains a `pre_check` field carrying
  the callable's dotted `module.qualname` string (`None` when unset), so
  two logs from materially different agents are never pooled as identical
  configs — the same reproducibility rule that made #199 record the
  `prior_learnings` path and sha256.
- `build_toolset(...)` gains `pre_check: PreCheck | None = None`, validates
  the displacement-mode refusal (§3) next to its existing mode checks, and
  hands it to `Toolset`.
- `Toolset._move_absolute`: after the interpolated `actions` list is built
  and clipped (immediately before `_success`), stack the action data into
  one array and call `pre_check`. On a string, return
  `ToolResult(error=...)` exactly like the existing bounds errors, prefixed
  `"pre-check rejected this motion: "` so transcripts distinguish semantic
  rejections from syntactic ones.
- `_stop` paths bypass the pre-check: holding the current pose is the
  fallback safe behavior, and a checker that could veto "stay where you
  are" would trap the agent with no legal tool call.
- Rejections count toward the existing consecutive-failures budget like
  every other tool error. Rationale: bounded by design, consistent with
  out-of-bounds targets today, and the failure counter resets on any
  successful call, so a model that corrects on retry is unaffected. If
  field use shows three consecutive rejections is too tight for semantic
  correction, widening is a follow-up, not a v1 knob.

### 4.3 Model-facing text

When `pre_check` is set, both system prompt templates
(`_SYSTEM_TEMPLATE` and `_ON_DEMAND_SYSTEM_TEMPLATE`) gain one clause
telling the model a motion pre-check may reject moves with a stated reason
and that it should adjust the target rather than repeat it. No prompt
change when unset (byte-identical prompts for existing users). The §5
prompt test covers both templates.

## 5. Testing

Plugin test suite (scripted fake LLM transports, no network), following the
existing patterns in `plugins/inspect-robots-agent/tests/`:

- Allow path: `pre_check` receives a `(steps, dim)` array whose last row
  equals the clipped target and whose rows are all within bounds; chunk
  emitted unchanged; `pre_check` called exactly once per move call.
- Reject path: string comes back as `ToolResult.error` with the prefix; the
  scripted model corrects on the next turn and the corrected chunk is
  emitted; failure counter resets.
- Persistent rejection: scripted model repeats the same bad move three
  times; `act()` raises the same `RuntimeError` repeated bounds errors
  produce today (which the rollout classifies as `PolicyError`, trial
  error under `fail_on_error`). The test asserts the `RuntimeError` from
  `act()`, not a graceful give-up: parity with existing tool-error
  behavior, no better and no worse.
- Bind-time refusal: displacement-mode space + `pre_check` raises
  `ToolsetError` naming this plan; absolute mode binds fine; no `pre_check`
  + displacement mode still binds (no regression).
- `_stop`/give-up/forced give-up paths never invoke the callable.
- Exception propagation: a raising `pre_check` escapes `act()`.
- System prompt: clause present iff `pre_check` is set.

## 6. Docs and release

- Plugin README: new section with the hook contract and the real YAM
  adapter code (a short function: loop the waypoints through
  `CollisionChecker.check`, return the geom-pair string with the waypoint
  index on a hit, `None` otherwise). States the layering rule: pre-check
  for feedback, approver chain for enforcement. Two documented caveats:
  (a) pre-check-pass does not imply approver-pass — the approver sweeps
  interpolated substeps finer than the emitted waypoint spacing at low
  control rates, so a chunk can clear the pre-check and still be held
  mid-chunk with no model feedback; adapters wanting parity should
  interpolate between waypoints themselves; (b) an over-conservative
  checker (wrong `table_height`, unmeasured base offsets) that rejects
  three moves in a row errors the trial rather than reaching `give_up`.
- Root README's agent-plugin blurb: one sentence.
- Plugin version bump 0.17.0 -> 0.18.0 in its pyproject and in
  `tests/test_package.py` (which asserts the exact version and is touched
  by every bump PR).
- No core `CHANGELOG.md` entry: the resolved precedent (#199) is that
  pure-plugin changes stay out of the core changelog, and this feature is
  pure-plugin.

## 7. File tree

```
inspect-robots/
├── plans/0031-agent-action-pre-check.md            (this document)
└── plugins/inspect-robots-agent/
    ├── pyproject.toml                              (version 0.18.0)
    ├── README.md                                   (pre-check section)
    ├── src/inspect_robots_agent/
    │   ├── _tools.py                               (PreCheck alias, build_toolset arg,
    │   │                                            move-path invocation, bind-time refusal)
    │   └── policy.py                               (pre_check kwarg + ConfigError guard,
    │                                                AgentPolicyConfig field, both prompt
    │                                                templates)
    └── tests/                                      (new tests per §5; test_package.py
                                                     version assertion updated)
```

## 8. Resolved questions

- Why `(steps, dim)` array instead of per-waypoint calls: one call per tool
  call keeps the contract simple, lets the checker vectorize or
  early-exit, and matches how the toolset already materializes the whole
  interpolation before emitting.
- Why absolute modes only: see §3; a wrong answer from a pre-check is
  worse than no pre-check, and the loud refusal is one line.
- Why rejections share the failure budget: see §4.2.
- Why exceptions propagate: a silent allow on checker crash would make the
  soft gate look like it is working when it is not; the hard gate
  (approver) still protects the hardware either way.
