# 0021 — `inspect-robots-capx`: code-as-policy agents via CaP-X perception servers

## 1. Goal

[CaP-X](https://github.com/capgym/cap-x) (arXiv:2603.22435, ICML 2026) is a
framework for **code-as-policy** manipulation: an LLM writes Python against
perception and motion primitives (SAM3 segmentation, Contact-GraspNet grasp
planning, Pyroki IK, joint-space moves) in a persistent REPL-like namespace,
with a multi-turn execute/observe/regenerate protocol. Frontier models score
30%+ zero-shot on their bench; a 7B coder hits 72% after RL. It is the third
policy class after served VLAs (`inspect-robots-xpolicylab`) and raw LLM
tool-calling (`inspect-robots-agent`), and nobody offers a side-by-side eval
of all three on one task.

This plan ships `plugins/inspect-robots-capx`: a `Policy` registered as
`capx` in which the LLM writes Python per turn. The code runs with perception
helpers backed by CaP-X's model servers and motion primitives that queue
speed-limited joint targets; the queue becomes the returned `ActionChunk`.

```bash
# terminal 1 — CaP-X checkout serves the models (GPU box is fine)
uv run capx/serving/launch_sam3_server.py --port 8114
uv run capx/serving/launch_contact_graspnet_server.py --port 8115
uv run python -c 'from capx.serving.launch_pyroki_server import main; main(robot="panda_description", port=8116)'
# (launch_pyroki_server.py calls bare main() with no CLI parsing, unlike the
#  other two servers, so flags on it are silently ignored)

# terminal 2 — Inspect Robots drives any joint-space embodiment with it
inspect-robots "pick up the red cube" --policy capx --embodiment <joint-space-embodiment> \
    -P model=anthropic/claude-fable-5 -P sam3_url=http://gpu-box:8114
```

(The core `cubepick` mock is a 2-D `eef_delta_pos` world and does not fit the
v1 joint-space profile below; the e2e test ships its own joint-space mock
embodiment, §9. Real targets are the YAM / SO-101 / Isaac adapters.)

One adapter ⇒ CaP-X-style agents become evaluable on any Inspect Robots
embodiment, including ones CaP-X does not support (our YAM and SO-101
adapters), next to VLAs and the tool-calling agent.

Non-goals (YAGNI, all deliberate):

- No CaP-RL / training integration; no CaP-Bench task port (a later plan —
  tasks are an independent axis).
- No visual-differencing VLM sidecar, skill library, or ensembling from
  CaP-Agent0; the core codegen loop is the paper's load-bearing part.
- No Molmo / OWL-ViT / SAM2 / cuRobo clients — SAM3 + GraspNet + Pyroki are
  the reduced-API pipeline the paper's headline results use.
- No dual-arm and no end-effector control mode in v1: the policy requires a
  single-arm `joint_pos` action space whose semantics declare a gripper
  (`ActionSemantics.gripper != "none"`) and raises a clear `bind()`-time
  error otherwise.
- No server *launching* (same doctrine as 0007: lifecycle belongs upstream;
  we connect to URLs).

## 2. Grounding: what CaP-X actually is (verified against source)

Facts established by reading `capgym/cap-x@main` (2026-07, 7 commits):

- **Not on PyPI**; research-grade monorepo with heavy deps (robosuite, torch,
  omnigibson). The agent loop `exec()`s model code in-process with helper
  functions bound into the namespace (`capx/envs/tasks/base.py`); variables
  persist across turns.
- **Prompting**: one system line ("generates Python code to directly solve
  the task"), a task prompt, then API docs auto-scraped from helper
  docstrings via `inspect` (`combined_doc()` in
  `capx/integrations/base_api.py`). Replies must be raw Python, no fences.
- **Multi-turn protocol** (from the shipped env_configs): after execution the
  model sees the executed code + stdout + stderr and must answer
  `REGENERATE` + new code, or `FINISH`.
- **Model servers** are plain FastAPI JSON-over-HTTP (`capx/serving/`):
  - SAM3 `/segment`: request `{image_base64: <PNG b64>, text_prompt}`;
    response `{results: [{mask_base64: <raw bytes b64>, shape: [H, W],
    box: [x1, y1, x2, y2], score, label}]}` (masks decode via
    `np.frombuffer(...).reshape(shape)`, uint8).
  - Contact-GraspNet `/plan`: request `{depth_base64, cam_K_base64,
    segmap_base64, segmap_id, local_regions, filter_grasps,
    skip_border_objects, z_range, forward_passes, max_retries}` where
    `*_base64` are base64-encoded `.npy` payloads (`np.save` round-trip);
    response `{grasps_base64, scores_base64, contact_pts_base64}` in the
    same `.npy` encoding. Grasp poses are `(K, 4, 4)` in the camera frame.
  - Pyroki `/ik`: request `{target_pose_wxyz_xyz: [7 floats],
    prev_cfg: [floats] | null}`; response `{joint_positions: [floats]}`.
    The server owns the robot model. Both `joint_positions` and `prev_cfg`
    are the URDF's **full actuated-joint config**, gripper joints included
    (CaP-X strips the gripper entry client-side:
    `extract_arm_joints(cfg) = cfg[:-1]` in
    `capx/integrations/franka/common.py`, and warm-starts with the full
    stored config).
- CaP-X's own clients retry POSTs with backoff (`post_with_retries`) because
  model servers cold-start slowly.

## 3. The two big decisions

**Speak the wire, don't import the package** (same as 0007): a published
wheel cannot depend on git-only research code, and the protocol surface is
three JSON endpoints plus two base64 codecs (~100 lines with tests). Drift
risk is accepted and mitigated exactly like 0007 (§9).

**Chunk-per-turn instead of CaP-X's blocking primitives.** In CaP-X, motion
primitives step the simulator mid-code. Inspect Robots inverts control:
`Policy.act(observation)` returns an open-loop `ActionChunk`. Rather than
threads/coroutines to suspend user code at each primitive, motion primitives
**queue** interpolated joint targets and the whole turn's queue is returned
as one chunk. Perception inside a turn sees the turn's initial observation;
post-motion effects are observed next turn. This matches how CaP-X's
reduced-API agent actually behaves (plan grasps from the turn's observation,
then execute; its multi-turn prompt tells the model to verify state *next
turn*) and matches how ActionChunk semantics already work for VLAs. No
concurrency, no partial-code re-entry, testable in-process.

## 4. Deliverable layout

```text
plugins/inspect-robots-capx/
├── pyproject.toml                    # hatchling; entry point inspect_robots.policies:capx
├── README.md                         # install, server bringup, arg table, trust model
├── src/inspect_robots_capx/
│   ├── __init__.py                   # CapxPolicy, capx_policy factory, __version__
│   ├── _codec.py                     # b64 PNG + b64 .npy encode/decode (CaP-X wire codecs)
│   ├── _servers.py                   # Sam3Client / GraspNetClient / PyrokiClient (httpx)
│   ├── _sandbox.py                   # per-trial exec namespace, stdout/stderr capture, helper binding
│   ├── _motion.py                    # joint-target queue: speed-limited interpolation, gripper, hold
│   └── policy.py                     # CapxPolicy: codegen loop, CaP-X multi-turn protocol
└── tests/
    ├── conftest.py                   # httpx.MockTransport stubs: LLM + 3 CaP-X servers
    ├── test_codec.py                 # golden wire tests against recorded CaP-X payload shapes
    ├── test_servers.py
    ├── test_sandbox.py
    ├── test_motion.py
    └── test_policy.py                # incl. e2e vs an in-test joint-space mock embodiment (§9)
plans/0021-capx-policy-plugin.md      # this file
.github/workflows/ci.yml              # + plugin-capx job, wired into both needs lists
.github/workflows/release.yml         # + publish-capx job (environment: pypi-capx)
plugins/inspect-robots-agent/…        # export ChatClient/resolve_provider/png_data_url (see §5)
README.md                             # + code-as-policy section
CLAUDE.md                             # plugins list mentions the new package
```

Dependencies: `inspect-robots>=0.4`, `inspect-robots-agent>=0.12`, `numpy`,
`httpx`. No `capx`, no `requests`, no provider SDKs, no Pillow. No PNG
*reader* exists anywhere in the pipeline: SAM3 requests need PNG *encoding*
(the agent plugin's `_png` writer, re-exported per §5), SAM3 masks return as
raw uint8 bytes, GraspNet speaks `.npy`, and Pyroki speaks JSON floats.

## 5. Reuse from `inspect-robots-agent`, made explicit

The capx plugin needs an OpenAI-compatible chat client with provider routing
and PNG data URLs — exactly what `inspect_robots_agent._llm` and `._png`
already implement. Duplicating ~300 maintained lines is worse than a
first-party dependency, so:

- `inspect-robots-agent` bumps to 0.12.0 (it is at 0.11.0 today) and
  re-exports `ChatClient`, `ResponsesClient` (the agent's `wire="responses"`
  client; a chat-only capx would inherit `effort="low"` defaults that 400 on
  direct OpenAI endpoints with an error message telling users to switch
  wires), `AssistantMessage` (the completion type capx must annotate under
  mypy strict), `resolve_provider`, `Provider`, `ENV_MODEL` (capx mirrors
  the agent's model-default env var), `png_data_url`, and `encode_png` (the
  raw PNG writer; SAM3 requests need bare b64 PNG, not a data URL) from its
  package `__init__`, extending its `__all__`; the modules stay where they
  are.
- `inspect-robots-capx` imports only those names from
  `inspect_robots_agent` (never from underscore modules).
- Both packages live in this repo's uv workspace and release together, so
  the coupling is CI-checked on every PR.

## 6. The policy: `CapxPolicy`

`CapxPolicy(PolicyBase)`, entry point `capx`. Constructor args (all keyword,
recorded via a `CapxPolicyConfig(PolicyConfig)` dataclass like the agent
plugin's):

| Arg | Default | Meaning |
| --- | --- | --- |
| `model` / `base_url` / `api_key_env` / `temperature` / `effort` / `wire` | agent-plugin defaults | LLM provider routing incl. the chat/responses wire selector, shared via §5 |
| `sam3_url` | `"http://127.0.0.1:8114"` | CaP-X SAM3 server |
| `graspnet_url` | `"http://127.0.0.1:8115"` | CaP-X Contact-GraspNet server |
| `pyroki_url` | `"http://127.0.0.1:8116"` | CaP-X Pyroki IK server |
| `camera` | `None` (sole camera, error if several) | observation camera feeding perception |
| `depth_key` | `"depth"` | `observation.extra` key holding `(H, W)` float depth (see below) |
| `intrinsics_key` / `extrinsics_key` | `"intrinsics"` / `"extrinsics"` | `extra` keys for `(3, 3)` K and `(4, 4)` camera-to-world |
| `max_llm_calls` | `100` | per-trial LLM budget; exhaustion forces give-up |
| `max_code_failures` | `3` | consecutive exec-*error* turns before `RuntimeError` (clean perception-only turns reset it; they are bounded by the LLM budget) |
| `max_speed_frac` | `0.1` | joint-interpolation speed cap, same semantics as the agent plugin |
| `request_timeout_s` | `120` | per server request (CaP-X models are slow) |
| `transcript_echo` | `False` | stderr live echo, same as agent |
| `transport` / `env` | `None` | test injection, same as agent |

**Depth convention.** Core `Observation.images` are uint8 RGB; depth has no
core slot. The plugin defines (and its README documents) an `extra`-key
convention: embodiments that want grasp planning expose float depth,
intrinsics, and camera-to-world extrinsics under the keys above, each as
**either the array or a zero-arg callable returning it** (helpers call it
if callable). The callable form is the documented recommendation for real
embodiments: the rollout deep-retains each step's `observation.extra` in
`TrialRecord` (only `images` get stripped to the frame store), so a raw
per-step 480x640 float64 depth map would turn trial records into an
in-memory depth video buffer at real control rates; a callable retains
nothing. Helpers that need a missing key raise inside the sandbox with a
message naming the key and the convention, which the model sees as stderr
and can route around (segmentation and IK still work without depth).
Nothing in core changes; a core follow-up (strip declared bulk `extra` keys
in `_store_frames`) is noted in §11 but not required.

**`bind(embodiment_info)`** adopts the embodiment spaces like the agent
plugin. The v1 profile, enforced with an actionable error naming this plan:

- `Box` action space, `control_mode="joint_pos"`, at least 2 dims, and
  `semantics.gripper != "none"`. The gripper dim is located via the box's
  `dim_labels` (a label named `"gripper"`); when labels are absent the last
  dim is the documented fallback. Arm dof = the remaining dims; gripper
  open/close values come from the box bounds on that dim.
- a proprioceptive reference resolved from `observation_space.state` the
  same way the agent toolset does (`build_toolset`'s state-labels logic): a
  single state field whose shape equals the full action dim. The motion
  cursor and the FINISH/GIVE_UP hold action read that one field from the
  turn's observation; there is no split joint_pos/gripper key convention.
- per-step interpolation deltas derived from `control_hz` and
  `max_speed_frac`, both required. `_motion.py` reproduces the agent
  toolset's step-limit derivation exactly (the `min(max_speed_frac / hz,
  0.05)`-of-range backstop and the native-dtype range arithmetic of
  `DeltaLimitApprover`), so *within a chunk* per-step deltas never exceed
  what the CLI's default approver allows — otherwise a low `control_hz`
  would get silently `delta_clamped` and the executed trajectory would
  diverge from what the sandbox reported. At *turn boundaries* the cursor
  re-seeds from observed state while `DeltaLimitApprover` clamps against
  the last approved action, so a lagging arm can still get its first
  next-turn action clamped; the agent plugin shares this edge, the README
  notes it, and the claim is scoped to within-chunk deltas.

**`reset(scene)`** starts the per-trial conversation:

- system: code-as-policy instructions modeled on CaP-X's prompt (write raw
  Python, no fences; helpers are pre-bound; import numpy explicitly;
  variables persist across turns), the helper API docs (§7, a static string
  — no runtime docstring scraping), the action-space/embodiment summary and
  embodiment `docs` notes, and the call budget.
- user: `Goal: {scene.instruction}`.
- clears the sandbox namespace and motion queue; `atexit`-safe `close()`
  closes the httpx clients (mirrors 0007's lifecycle care).

**`act(observation)`** — the codegen loop:

1. Append the observation message: labeled state text plus camera PNGs
   (reused agent-plugin formatting). The previous turn's execution report is
   already in `_messages` (step 4), so the observation message carries only
   the fresh observation.
2. Ask the LLM. Accept raw Python or a single fenced block (strip fences —
   CaP-X asks for raw code, but its own multi-turn prompt then demands
   fenced code after `REGENERATE`; be liberal). Recognize the control words
   `FINISH` and `GIVE_UP` (bare, before any code): both return a one-action
   hold chunk (repeat current joint state) whose action meta carries
   `request_stop: True` and a `stop_reason` (the rollout already honors
   this, and an `ActionChunk` cannot be empty).
3. Execute the code in the sandbox: helpers bound, stdout/stderr captured
   (traceback printed to captured stderr on any exception, CaP-X-style).
4. Append the CaP-X-style execution report (executed code, stdout, stderr,
   truncated tail-first to a documented cap so context stays bounded) to
   `_messages` **eagerly, before returning** — chunk-level `meta` is a dead
   channel (`DefaultController` buffers only `chunk.actions` and the JSON
   `EvalLog` persists no chunk meta), so the transcript is the record; a
   trial that terminates on this chunk still carries the final turn's
   stdout/stderr in `transcript()`. The executed code (untruncated) also
   rides the first queued action's `meta["code"]`, which survives into
   `StepRecord` for step-level debugging. If the queue has actions: pop it
   and return `ActionChunk(actions=queue, control_hz=embodiment control_hz,
   inference_latency_s=<wall time of LLM call(s)>)`.
5. If the queue is empty (pure-perception turn or exec error): the report
   from step 4 is already the feedback message; loop to 2 — bounded by
   `max_code_failures` consecutive *error* turns (a clean perception-only
   turn resets the failure counter but still loops; the LLM-call budget is
   the global bound, exhaustion forces the give-up hold chunk exactly like
   the agent plugin's `_forced_give_up`).

`transcript()` / `transcript_delta()`: sanitized copies with images elided,
identical mechanics to the agent plugin, so `inspect-robots inspect
--transcript` renders capx trials for free.

**`PolicyInfo`**: name `capx`, adopted spaces and `control_hz` from bind,
same placeholder-before-bind pattern as the agent plugin.

## 7. The sandbox helpers (what the model's code can call)

Bound into the exec namespace by `_sandbox.py`; documented verbatim in the
system prompt. All arrays NumPy. `obs` is the current turn's observation
(images dict, state dict, plus depth/intrinsics/extrinsics when the
embodiment provides them).

- `segment(text: str) -> list[dict]` — SAM3 text-prompt segmentation on the
  configured camera's current RGB. Returns `{mask (H, W) bool, box, score,
  label}` dicts (CaP-X's client shape).
- `plan_grasp(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]` —
  Contact-GraspNet on current depth + intrinsics with `mask` as the segmap.
  Returns `(K, 4, 4)` poses in the **camera frame** and `(K,)` scores; the
  prompt documents `extrinsics @ pose` for world frame, mirroring CaP-X's
  docstring example. When the server finds no grasps its retry loop returns
  flat `np.array([])` payloads; the client normalizes those to
  `(0, 4, 4)` / `(0,)` so `K == 0` is an ordinary, checkable result rather
  than a shape crash (golden wire test in §9).
- `solve_ik(position: np.ndarray, quaternion_wxyz: np.ndarray) ->
  np.ndarray` — Pyroki `/ik`. The wire speaks the URDF's full actuated
  config (§2): the client keeps the last full server-returned config as the
  warm-start `prev_cfg` (mirroring CaP-X's stored `self.cfg`; `None` on the
  first call) and returns the response stripped to the bound arm dof
  (leading `arm_dof` entries), raising an actionable in-sandbox error when
  `len(joint_positions) < arm_dof` (server robot ≠ embodiment robot, the
  §11 risk). The helper doc tells the model it gets arm joints, sized for
  `move_to_joints`.
- `move_to_joints(joints: np.ndarray) -> None` — queue speed-limited linear
  interpolation from the cursor to `joints` (gripper dim held); advances the
  cursor. The cursor starts each turn at the bound proprioceptive state
  field (§6) from the turn's observation.
- `open_gripper() / close_gripper() -> None` — queue a gripper ramp at the
  cursor's arm pose, interpolated under the same per-step delta limit as arm
  moves (so the CLI's default `DeltaLimitApprover` never silently clamps a
  gripper jump the sandbox reported as done); advance the cursor's gripper
  value.
- `print(...)` — captured stdout is the model's feedback channel (CaP-X
  convention).

Everything else is plain Python: the model may import installed packages
(numpy, scipy if present). **Trust model, stated loudly in README and module
docstring**: the code executes in-process with the evaluator's privileges,
exactly as in CaP-X. This is the policy class under evaluation, not a
sandboxing product; run untrusted models in a container. The safety story
for the *robot* is unchanged: every queued action still passes the rollout's
approver chain. The Clamp + DeltaLimit guardrails are the *CLI* default
(`eval()` called programmatically defaults to `AutoApprover`); the README
says exactly that and shows the programmatic approver wiring.

## 8. Wire clients: `_codec.py` + `_servers.py`

- `_codec.py`: `png_b64_encode(rgb)` (reuse agent `_png` writer + b64),
  `npy_b64_encode/decode` (`np.save`/`np.load` round-trip, `allow_pickle=False`
  on load), `mask_decode(mask_b64, shape)` (`np.frombuffer` reshape). Upstream
  commit hash recorded in the module docstring (0007's drift mitigation).
- `_servers.py`: three tiny clients over one shared `httpx.Client`
  (injectable transport). POST with retries and backoff (bounded by
  `request_timeout_s` wall clock, CaP-X-style cold-start tolerance), JSON
  bodies exactly matching §2's schemas. Connection failures raise actionable
  errors naming the URL and the CaP-X launch command. Lazy: no request until
  the model's code first calls a helper, so `list policies` and compat
  checks never touch the network (0007's lazy-connection doctrine).
- GraspNet client hardcodes CaP-X's client-side defaults (`segmap_id=1`,
  `local_regions/filter_grasps=True`, `z_range=[0.2, 2.0]`,
  `forward_passes=2`, `max_retries=10`) — they are model-tuning knobs, not
  user API.

## 9. Tests (no GPU, no sockets, no capx checkout)

All network stubbed via `httpx.MockTransport` handlers implementing the §2
schemas (LLM stub reuses the agent plugin's canned-completion approach):

- codec: golden round-trips — a float32 depth map through `npy_b64`, a bool
  mask through `mask_decode`, structural equality against handcrafted wire
  payloads shaped like CaP-X's (guards drift without importing capx),
  including the empty-grasp flat-`[]` payload normalizing to
  `(0, 4, 4)` / `(0,)`.
- servers: request bodies match the schemas exactly (recorded by the stub);
  retries on 503-then-200; actionable ConnectionError text; timeout path;
  `/ik` golden test with an `arm_dof + 1`-long full-config response (strip
  to arm dof, full config becomes the next `prev_cfg`, short response
  raises the actionable error).
- sandbox: namespace persists across turns within a trial and resets across
  trials; stdout/stderr capture; traceback lands in stderr; helper raising
  (e.g. missing depth) surfaces as stderr not a crash; depth accepted as
  both a raw array and a zero-arg callable.
- motion: interpolation respects `max_speed_frac` and `control_hz`,
  including a low-`control_hz` case proving the per-step delta stays under
  the `DeltaLimitApprover` backstop; cursor chaining across move/gripper
  calls; hold chunk shape; gripper values from box bounds.
- policy: bind profile enforcement (rejects ee mode, dual-arm-sized boxes
  without gripper, missing control_hz); fence stripping and raw-code paths;
  FINISH/GIVE_UP → hold chunk with `request_stop` meta; perception-only
  turn loops then returns queued chunk; consecutive-failure and call-budget
  bounds; execution report truncation; transcript sanitization; the
  execution report (incl. a terminal turn's stdout/stderr) lands in
  `transcript()` eagerly, and the first queued action's `meta["code"]`
  carries the executed code.
- e2e: `CapxPolicy` vs an in-test joint-space mock embodiment (arm dims +
  labeled gripper dim, `joint_pos` semantics, a matching single-field
  StateSpec, a small RGB camera — the pattern of the agent plugin's
  `_AbsoluteEmbodiment` test double; core `CubePick` is `eef_delta_pos` and
  deliberately out of profile) with a scripted LLM transport whose canned
  code calls `segment` → `solve_ik` → `move_to_joints` → `close_gripper` and
  FINISHes on turn 2; assert the rollout completes and the log carries the
  transcript.
- registry: entry point resolves; factory forwards `-P`-style kwargs.

Coverage: plugin CI runs pytest without the core gate (per repo convention);
aim for full-line coverage of `policy.py`, `_motion.py`, `_sandbox.py`.

## 10. CI, workspace, release, docs

- `pyproject.toml` mirrors the agent plugin's (hatchling,
  hatch-fancy-pypi-readme, static `version = "0.1.0"`, mypy strict, ruff
  line-length 100, `[tool.uv.sources]` workspace pins for both first-party
  deps). `uv lock` and commit.
- CI: `plugin-capx` job cloned from `plugin-agent` (ruff, ruff format, mypy
  strict, pytest), ubuntu-only; **added to both `needs` lists** (`ci-ok` and
  the sibling at ci.yml:322) — CLAUDE.md's rule, and 0007 hit the same edge.
- Release: `publish-capx` job in release.yml (environment `pypi-capx`);
  maintainer creates the PyPI trusted-publisher environment before first
  release (PR description calls this out; it is a settings action, not code).
  The agent-plugin version bump to 0.12.0 rides the same PR.
- Entry point: `[project.entry-points."inspect_robots.policies"]
  capx = "inspect_robots_capx.policy:capx_policy"`.
- Docs: plugin README (server bringup from a CaP-X checkout, arg table,
  depth-key convention, trust-model warning, troubleshooting cold starts);
  root README gets a code-as-policy paragraph next to the agent plugin's;
  root CLAUDE.md plugins list gains the package. Repo writing style applies
  (no em dashes in prose, no AI-tell patterns).
- Core is untouched: no new core deps, no `__all__`/api-snapshot churn.

## 11. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Server schema drift (no version field) | Golden wire tests; upstream commit hash in `_codec.py`; schemas isolated in two small modules |
| `exec()` of model output alarms users | Loud trust-model section in README + module docstring; approver chain still gates every action; containerize for untrusted models |
| Embodiments lack depth (SO-101 webcam rigs) | Documented degradation: segmentation + IK still work; `plan_grasp` raises a routable in-sandbox error naming the extra-key convention |
| Raw per-step depth arrays bloat `TrialRecord` memory | Callable form of the extra-key convention is the documented recommendation (§6); possible core follow-up: strip declared bulk `extra` keys in `_store_frames` |
| 5-DOF arms can't reach 6-DOF grasp poses | Out of scope for the plugin (Pyroki returns best-effort IK); README notes the caveat and suggests top-down grasp filtering in the task prompt |
| Model floods context with huge stdout | Execution report truncated to a documented cap, tail-first (errors are at the tail) |
| Coupling to `inspect-robots-agent` internals | Only public re-exports imported (§5); workspace CI breaks loudly on drift |
| Pyroki server robot ≠ embodiment robot | README documents the per-embodiment URDF pairing and the actual invocation (upstream's `__main__` ignores CLI flags, §1); IK results that violate the action box get clamped by approvers and reported next turn |
| Cold-start latency of model servers | Retry-with-backoff bounded by `request_timeout_s`; error text names the launch command |

## 12. Execution steps (each a commit)

1. Agent-plugin re-exports + version bump; plugin skeleton (pyproject,
   `__init__`, empty modules, README stub); `uv lock`; workspace sync green.
2. `_codec.py` + golden wire tests.
3. `_servers.py` + MockTransport stubs + tests.
4. `_sandbox.py` + `_motion.py` + tests.
5. `policy.py` (bind/reset/act/transcript/close) + tests + the joint-space
   mock-embodiment e2e (§9).
6. CI job into both needs lists; release job; root README + CLAUDE.md +
   plugin README.
7. Gates: `ruff check`, `ruff format --check`, plugin mypy strict, plugin
   pytest, core suite untouched (`uv run pytest --cov` still 100%).
