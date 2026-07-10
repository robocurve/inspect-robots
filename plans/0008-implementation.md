# 0008 Implementation Plan — LLMs as policies

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans
> (inline). The authoritative design is `plans/0008-llm-agent-mode.md`
> (referenced below as "spec §N"); this file sequences it into TDD tasks.
> Steps use checkbox syntax for tracking.

**Goal:** LLMs drive robots as a registered `Policy` (`agent`) with core
safety guardrails on by default at the CLI.

**Architecture:** Five core slices (vocabulary → approvers → bind →
request_stop → CLI), then the `plugins/inspect-robots-agent` package
(httpx client → tools/motion → policy e2e), then docs. Each task = one
commit, gates green (`ruff check .`, `ruff format --check .`, `uv run mypy`,
`uv run pytest --cov` at 100% for core).

**Tech stack:** NumPy-only core; plugin adds `httpx`. uv workspace.

## Global constraints (from spec + CLAUDE.md)

- Core stays NumPy-only; plugin deps never imported by core.
- 100% coverage on `inspect_robots`; mypy strict covers src and tests.
- Public API fenced by `__all__` + `tests/test_api_snapshot.py` — update together.
- Plugins: own pyproject (static version), own tests, outside core coverage.
- README/docs prose: no em dashes, no mid-sentence bold, no decorative emoji.
- Commit after every green task.

---

### Task 1: `spaces.py` vocabulary — `joint_delta` + `dim_labels` (spec §3b)

**Files:** Modify `src/inspect_robots/spaces.py` (ControlMode at :23,
ActionSemantics at :43, Box.__post_init__ at :62),
`src/inspect_robots/controller.py:97` (`_AVERAGEABLE_MODES` gains
`joint_delta`; pragma premise stays valid). Tests in
`tests/test_types_spaces.py`; snapshot untouched unless `__all__` changes
(it should not — both are attributes of existing exports).

- [ ] Failing tests: `joint_delta` accepted as ControlMode; `dim_labels`
  round-trips on ActionSemantics; Box with semantics whose
  `len(dim_labels) != dim` raises ValueError; matching length passes;
  `dim_labels=None` passes; ensembling accepts a `joint_delta` chunk pair
  (tests/test_ensembling.py).
- [ ] Implement; gates green; commit.

### Task 2: approvers — `DeltaLimitApprover` + `ChainApprover` (spec §3a)

**Files:** Modify `src/inspect_robots/approver.py`,
`src/inspect_robots/rollout.py:216` (approval-event detail: surface
`delta_clamped` alongside `clamped`), `src/inspect_robots/__init__.py`
(+`__all__`), `tests/test_api_snapshot.py`. New `tests/test_approvers.py`
(move/extend existing approver tests if any live elsewhere).

**Interfaces (produced):**
- `DeltaLimitApprover(action_space: Box, max_delta: float | ArrayLike | None = None)`
  — `.review(action, store) -> Action`. Absolute modes {joint_pos,
  eef_abs_pose}: clamp to ±max_delta around last approved action (store key
  `"delta_limit:last"`), first action passes; derived default 5% of range.
  Displacement/rate modes {eef_delta_pos, eef_delta_pose, joint_delta,
  joint_vel}: clamp to box ∩ [-max_delta, +max_delta]; derived default =
  box alone. Raises ValueError at construction: semantics None; needed
  bound missing/non-finite without explicit max_delta; pose mode with
  rotation_repr not in {none, rot6d}. NaN action → SafetyAbort. Modified
  action → new object with meta `delta_clamped: True`.
- `ChainApprover(*approvers: Approver)` — sequential review.

- [ ] Failing tests: every constructor refusal; absolute clamp w/
  reference update; first-action pass; displacement intersection incl.
  asymmetric [0,1] dim; derived defaults both branches; NaN abort;
  identity preserved when nothing clamps; per-trial store isolation;
  chain order; rollout records delta_clamped detail (extend
  tests/test_rollout_hardening.py).
- [ ] Implement; gates green; commit.

### Task 3: `Policy.bind()` hook (spec §3c)

**Files:** Modify `src/inspect_robots/policy.py` (Protocol comment +
`PolicyBase.bind` no-op), `src/inspect_robots/eval.py` (call between
resolution and `assert_compatible` at ~:205). Tests in
`tests/test_eval_orchestration.py`.

**Interfaces (produced):** `def bind(self, embodiment_info: EmbodimentInfo) -> None`
— optional; eval() invokes iff `hasattr(policy, "bind")`.

- [ ] Failing tests: a recording policy's bind receives the resolved
  embodiment's info before compat runs (adapting policy passes compat only
  because bind ran); a bind-less minimal Protocol policy still works.
- [ ] Implement; gates green; commit.

### Task 4: policy-requested stop (spec §3d)

**Files:** Modify `src/inspect_robots/rollout.py` (~:230, after step;
pre-review action's meta; embodiment termination wins). Docstring notes
the EnsemblingController limitation. Tests in
`tests/test_rollout_hardening.py`.

- [ ] Failing tests: chunk whose last action carries
  `meta={"request_stop": True, "stop_reason": "done"}` → trial ends
  truncated with reason "done"; default reason "policy_stop"; approver
  rewrite does not erase intent; simultaneous embodiment `terminated=True`
  wins; no stop flag → unchanged behavior.
- [ ] Implement; gates green; commit.

### Task 5: CLI guardrails-by-default + `config` (spec §3e)

**Files:** Modify `src/inspect_robots/cli.py` (guardrail chain builder
inside the existing close-embodiment try; `--disable-guardrails`,
`--max-action-delta`; `config set/show` subcommand; guided-error fix line
gains `inspect-robots config set …`), `src/inspect_robots/_defaults.py`
(writer helper, atomic tmp+rename, preserves unknown sections/keys).
Tests in `tests/test_registry_cli.py` + `tests/test_defaults.py`.

- [ ] Failing tests: default run wires ChainApprover(Clamp, DeltaLimit)
  (observable via a spy embodiment whose out-of-range scripted action gets
  clamped); `--disable-guardrails` warns on stderr + AutoApprover;
  degradation warnings for bounds-less absolute space, synthetic
  semantics-less space, synthetic quat-pose space, fully-unlimitable
  (each warning names the caught reason); `--max-action-delta` threads
  through; `config set embodiment yam_arms` writes INI (tmp XDG), `config
  set` rejects unknown keys, `config show` prints resolved defaults with
  sources, unknown sections survive rewrite; guided error mentions config
  set.
- [ ] Implement; gates green; commit.

### Task 6: plugin scaffold (spec §4 layout, §7)

**Files:** Create `plugins/inspect-robots-agent/pyproject.toml`
(deps `inspect-robots`, `httpx`; version 0.1.0; entry point
`[project.entry-points."inspect_robots.policies"] agent = "inspect_robots_agent.policy:agent_policy"`),
`src/inspect_robots_agent/__init__.py`, `py.typed`, stub `policy.py`,
`tests/__init__.py` (+ trivial import test). Modify root `pyproject.toml`
only if workspace members are enumerated (they are globbed — verify),
run `uv lock`; `.github/workflows/ci.yml` (job `test-agent-plugin`
mirroring `test-xpolicylab-plugin`, added to `ci-ok.needs`),
`.github/workflows/release.yml` (`publish-inspect-robots-agent`,
`skip-existing`, environment `pypi-agent`).

- [ ] Scaffold; `uv sync --all-packages --extra dev`; plugin test green;
  core gates untouched; commit.

### Task 7: plugin `_llm.py` (spec §4a)

**Interfaces (produced):**
- `resolve_provider(model: str | None, base_url: str | None, api_key_env: str | None, env: Mapping[str, str]) -> Provider`
  (dataclass: base_url, api_key, model) following spec §4a ladder; raises
  `AgentConfigError` with guided message otherwise.
- `ChatClient(provider, *, transport: httpx.BaseTransport | None = None)`
  `.complete(messages: list[dict], tools: list[dict]) -> AssistantMessage`
  (dataclass: content, tool_calls) with bounded retry/backoff on transient
  HTTP, raise on persistent.

- [ ] Failing tests (MockTransport): each ladder rule; missing-key guided
  error; retry then success; retry exhaustion raises; request body carries
  tools + model; Anthropic compat base URL for anthropic/*.
- [ ] Implement; plugin tests green; commit.

### Task 8: plugin `_tools.py` + `_motion.py` (spec §4b tools, §4c)

**Interfaces (produced):**
- `build_toolset(action_space: Box, state_spec, control_hz: float) -> Toolset`
  — bind-time validation (mode support, rotation guard, absolute-mode
  alignment rule: exactly one state field with shape == (dim,)); exposes
  OpenAI tool JSON schemas (`move_joints` or `move_by`, `done`, `give_up`).
- `Toolset.execute(tool_call, current_state: np.ndarray) -> ToolResult`
  where ToolResult = chunk (ActionChunk) | error message (str, fed back to
  LLM). Chunk cap ~10 s of steps; request_stop meta on done/give_up hold
  action.

- [ ] Failing tests: labeled partial targets (bimanual 14-D synthetic
  space) interpolate correctly and hold unnamed dims; index fallback;
  unknown label / non-finite / bad duration → error string; move_by splits
  displacement; hold-still per mode; done/give_up meta; joint_vel and quat
  pose rejected at build; alignment-rule failures (0 fields, 2 matching,
  no StateSpec).
- [ ] Implement; plugin tests green; commit.

### Task 9: plugin `policy.py` e2e (spec §4b)

**Interfaces (produced):** `agent_policy(**kwargs) -> LLMAgentPolicy`
(registry factory); `LLMAgentPolicy.bind/reset/act`;
`AgentPolicyConfig(PolicyConfig)` frozen with model, base_url,
api_key_env, max_llm_calls, temperature.

- [ ] Failing tests (scripted conversations through real `eval()` on
  `cubepick`): goal runs to done → truncated "done" in log; wild-swing
  script clamped by CLI-style chain, unclamped with AutoApprover; budget
  exhaustion → give_up; malformed tool call retried then PolicyError;
  images + labeled state in outbound messages; config lands in EvalLog.
- [ ] Implement; plugin tests green; commit.

### Task 10: docs (spec §9 step 10 core half)

**Files:** Modify `README.md` (agent-policy section, style rules apply),
`CLAUDE.md` (plugins list + new core seams), `plans/0008-llm-agent-mode.md`
untouched, `src/inspect_robots/CLAUDE.md` module-map rows. Yam-repo PRs
(spec §6) are follow-up work outside this repo/plan.

- [ ] Write; gates green; commit.
