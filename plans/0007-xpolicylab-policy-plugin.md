# 0007 — `inspect-robots-xpolicylab`: first-class XPolicyLab policy plugin

## 1. Goal

[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) is "a unified standard
and infrastructure for robot policy development and deployment": it wraps
**40+ VLA / imitation-learning policies** (π0, π0.5, GR00T N1.7, OpenVLA-OFT,
RDT-1B, SmolVLA, ACT, Diffusion Policy, …) behind one **policy-server**
contract, served over a websocket protocol. Each policy runs in its own
conda/uv environment (its whole point is environment isolation), and any
evaluation framework that speaks the protocol can drive any of those policies.

This plan gives Inspect Robots a first-class `Policy` adapter for that
protocol, shipped as the second in-repo plugin:

```bash
# terminal 1 — any XPolicyLab policy, its own env, possibly another machine
cd XPolicyLab/policy/Pi_0
bash setup_eval_policy_server.sh ... 19000 0.0.0.0

# terminal 2 — Inspect Robots drives it like any other policy
inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
    -P url=ws://gpu-box:19000 -P cameras=cam_head:base_rgb
```

(`-P` values are scalar-parsed by the CLI, so mapping-valued args accept a
compact string form — see §5.)

One adapter ⇒ the entire XPolicyLab policy zoo becomes evaluable with any
Inspect Robots embodiment and task. This mirrors what `inspect-robots-isaacsim`
did for the embodiment axis: Isaac gave us the "body" half; XPolicyLab gives
us the "brain" half.

Non-goals (YAGNI):

- No batched evaluation (`update_obs_batch`/`get_action_batch`,
  `eval_batch: true`) — the rollout loop is single-trial.
- No policy-server *launching* (no subprocess management of `eval.sh` /
  `setup_eval_policy_server.sh`). The server lifecycle belongs to XPolicyLab
  and its per-policy environments; we connect to a URL. A launcher can be a
  later plan once the connect story is proven.
- No `legacy_tcp` transport — XPolicyLab's README says new adapters use `ws`.
- No data-format conversion tooling (XPolicyLab's HDF5/LeRobot converters are
  training-side, out of scope for evaluation).

## 2. Grounding: what XPolicyLab actually is (verified against source)

Facts established by reading `XPolicyLab/XPolicyLab@main` (2026-07):

- **Not on PyPI** (404 for `xpolicylab`); installable only from git. Its
  `pyproject.toml` ships top-level packages `client_server` and `utils` plus a
  6-line top-level module `XPolicyLab` — generic, collision-prone names — and
  hard deps on `opencv-python-headless` and `h5py`.
- **Wire protocol** (`client_server/ws/protocol/`): websocket **binary**
  frames, msgpack-encoded with `msgpack-numpy` for arrays. Envelope fields:
  `message_type`, `message_id`, `evaluation_id`, `action_case_id?`,
  `trial_id?`, `repeat_index?`, `step`, `sent_at`, `payload`. Request/response
  pairs: `hello→hello_ack`, `prepare_case→prepare_case_ack`,
  `reset→reset_result`, `infer→infer_result`, `trial_end→trial_end_ack`,
  `heartbeat→heartbeat_ack`; plus `close` (no reply) and `error` (carries
  `code`/`message`/`details`, replaces the paired response).
- **Inference**: `infer` payload is `{"observation": <obs dict>}`; the reply
  payload is `{"actions": [<action dict>, ...], "latency_ms": float}` — the
  list is an **action chunk**, one dict per future control step.
- **Observation Data Format v1.0**: `instruction` (str),
  `vision/<cam>/color` as `(H, W, 3)` RGB, `state/<key>` float arrays with
  keys like `arm_joint_state`, `ee_joint_state`, `ee_pose`,
  `left_/right_`-prefixed dual-arm variants; poses are
  `[x, y, z, qw, qx, qy, qz]`; optional `additional_info/frequency`.
- **Action dicts** (from `policy/demo_policy/model.py`): per-step dict with
  `arm_joint_state` (joint mode) or `ee_pose` (ee mode) plus
  `ee_joint_state` for the gripper; dual-arm uses `left_*`/`right_*` keys.
- The env-side client XPolicyLab ships (`WsModelClient`) is exactly the role
  we play; RoboDojo embeds the same protocol on its eval side.

## 3. The one big decision: speak the protocol, don't import the package

The adapter implements XPolicyLab's websocket protocol itself rather than
depending on the `xpolicylab` package:

- A published `inspect-robots-xpolicylab` wheel **cannot** declare a git-only
  dependency (PyPI rejects direct-URL requirements), and asking users to
  git-install a package that drops `utils` and `client_server` into
  site-packages is hostile.
- The protocol surface we need is small and readable: one envelope schema,
  five message types, msgpack(+numpy) codec. Upstream's env-side stack is
  ~540 lines (`WsModelClient` + `PolicyEvalClient`), most of which is asyncio
  plumbing and reconnect/countdown UX we replace with a simpler synchronous
  design (reconnect-on-next-use, §5). Reimplementing the envelope as a frozen
  dataclass also sheds upstream's `pydantic` and `pyyaml` hard deps.
- The plugin's real deps are light and on PyPI: `websockets>=12` (the
  **sync** client — no asyncio event loop to own — exists since v11, and
  Isaac Sim environments ship v12, which the flagship isaacsim pairing shares
  a venv with; verify the floor against Isaac Lab during implementation),
  `msgpack`, `msgpack-numpy`, `numpy`, `inspect-robots`.

Drift risk is accepted and mitigated (see §9): the protocol carries no version
field, so we record the upstream commit we validated against and keep the
codec/envelope in one small module that is trivial to diff against upstream.

## 4. Deliverable layout

```text
plugins/inspect-robots-xpolicylab/
├── pyproject.toml                       # hatchling; entry point inspect_robots.policies:xpolicylab
├── README.md                            # install, server-side quickstart, mapping reference
├── src/inspect_robots_xpolicylab/
│   ├── __init__.py                      # XPolicyLabPolicy, xpolicylab_policy factory, __version__
│   ├── _protocol.py                     # Frame dataclass, MessageType, msgpack-numpy codec, WsError
│   ├── _client.py                       # PolicyClient: sync ws client (connect/hello/reset/infer/trial_end/close)
│   └── policy.py                        # XPolicyLabPolicy: the inspect_robots.Policy adapter
└── tests/
    ├── conftest.py                      # in-process stub policy server (websockets.sync/asyncio thread)
    └── test_xpolicylab_policy.py
plans/0007-xpolicylab-policy-plugin.md   # this file
.github/workflows/ci.yml                 # + plugin-xpolicylab job, wired into ci-ok.needs
.github/workflows/release.yml            # + publish-xpolicylab job (environment: pypi-xpolicylab)
README.md                                # + "Policies via XPolicyLab" section
CLAUDE.md                                # plugins list mentions the new package
```

## 5. The adapter: `XPolicyLabPolicy`

`XPolicyLabPolicy(PolicyBase)` with constructor (all keyword, all logged into
the eval log via `PolicyInfo`/`PolicyConfig`):

| Arg | Default | Meaning |
| --- | --- | --- |
| `url` | `"ws://localhost:19000"` | policy server websocket URL |
| `action_type` | `"joint"` | `"joint"` → `*arm_joint_state`; `"ee"` → `*ee_pose` action keys |
| `arms` | `1` | 1 or 2; selects unprefixed vs `left_`/`right_` key families |
| `arm_dim` | `7` | per-arm joint dims (joint mode); ee mode fixes 7 (pos+quat wxyz) |
| `ee_dim` | `1` | per-arm gripper dims |
| `cameras` | `{"cam_head": "cam_head"}` | XPolicyLab camera slot → Inspect Robots camera name |
| `state_map` | sensible default, see below | XPolicyLab state key → Inspect Robots state key |
| `required_state_keys` | joint: `{"joint_pos", "gripper"}`; ee: `frozenset()` | Inspect Robots state keys declared as required in `observation_space` |
| `control_hz` | `None` | declared chunk playback rate; also sent as `additional_info.frequency` |
| `name` | `"xpolicylab"` | `PolicyInfo.name` (users can tag e.g. `"xpolicylab:pi0"`) |
| `evaluation_id` | fresh UUID | protocol `evaluation_id` |
| `connect_timeout_s` / `request_timeout_s` | `30` / `120` | client timeouts |

Mapping-valued args (`cameras`, `state_map`) also accept a compact string
form, `"slot:name,slot:name"`, because the CLI's `-P k=v` (and config.ini
defaults) parse values as scalars only (`_defaults.parse_value`). The factory
normalizes either form; `required_state_keys` likewise accepts a
comma-separated string. This keeps every constructor arg reachable from
`inspect-robots run`.

**Lazy connection** (mirrors the isaacsim plugin's lazy Isaac import):
constructing the policy and reading `.info` never touches the network, so
`inspect-robots list policies` and fail-fast compatibility checks work with no
server running. The client connects (with `hello` handshake) on first
`reset()`/`act()`. Connection failures raise a clear, actionable error naming
the URL and the XPolicyLab server-side command to start one. If the socket
dies mid-eval, the next `reset()`/`act()` reconnects once (replaying `hello`)
before failing — a dead server thus costs one `PolicyError`d trial, not every
remaining trial. (Upstream's transparent in-flight reconnect/countdown UX is
deliberately not replicated.)

**`reset(scene)`**: sends `trial_end` for the previous trial (if one is open);
stores `scene.instruction`; assigns a fresh `trial_id` (scene id + counter)
and zeroes the per-trial `step` counter (the envelope's `step` field
increments on every `infer`, matching upstream's env client); sends protocol
`reset`. Server-side `reset` clears model state (RNN/history) exactly like
between-episode resets in XPolicyLab evals.

**`act(observation)`**:

1. Build the Observation-v1.0 dict (with `data_format_version: "v1.0"` set —
   free drift insurance):
   - `instruction`: `observation.instruction or` the scene instruction.
   - `vision/<slot>/color`: `observation.images[name]` per the `cameras` map
     (missing camera ⇒ `PolicyError`-worthy exception with the mapping
     spelled out). Images pass through as `(H, W, 3)` uint8 RGB — both sides
     already agree on that convention.
   - `state/<xpl_key>`: `observation.state[ir_key]` per `state_map`; mapped
     keys that are absent from the observation are skipped (XPolicyLab state
     fields are all optional).
   - `additional_info/frequency`: `control_hz` when set.
2. `infer` → response `payload["actions"]` (list of per-step dicts).
3. Flatten each step dict into one action vector in a **fixed documented
   order**: for each arm (left then right): arm key then ee key —
   e.g. single-arm joint mode: `concat(arm_joint_state, ee_joint_state)` →
   dim `arm_dim + ee_dim` = 8 for the default Franka-like profile, matching
   `inspect-robots-isaacsim`'s default action space. Shape mismatches versus
   the declared `Box` raise immediately with both shapes in the message.
4. Return `ActionChunk(actions=..., control_hz=control_hz,
   inference_latency_s=<client-measured wall time>,
   meta={"server_latency_ms": ...})`. Wall time is what the rollout
   transcript means by observed inference latency (it includes network and
   serialization); the server's model-compute-only `latency_ms` is preserved
   in `meta`.

**`close()`** sends `trial_end` for any open trial, then protocol `close`,
then closes the socket; idempotent; the policy is also a context manager.
`eval()` only auto-closes *embodiments* it resolves — registry-resolved
policies are never closed — so the adapter also registers a best-effort
`atexit` close (unregistered on explicit `close()` so instances don't
accumulate strong references across a test suite), and the README documents
`with`/`close()` for programmatic use.

**`PolicyInfo`**:

- `action_space`: `Box(shape=(arms * (arm_dim + ee_dim),), semantics=...)`
  with `control_mode="joint_pos"` (joint) or `"eef_abs_pose"` +
  `rotation_repr="quat_wxyz"` (ee; XPolicyLab poses are wxyz), and
  `gripper="continuous"` when `ee_dim > 0`.
- `observation_space`: `state_keys` = `required_state_keys` **only**. The
  remote policy is opaque (its `hello_ack` carries just `{ok, server}`), so
  the adapter cannot infer what the model needs; `check_compatibility`
  treats declared keys as hard requirements, so declaring the full
  `state_map` would wrongly fail embodiments that lack optional keys
  (e.g. isaacsim's default StateSpec has no 7-D `eef_pose`). `state_map`
  entries absent from both the observation and `required_state_keys` are
  simply not sent — all XPolicyLab state fields are optional. Cameras are
  declared via `CameraSpec` only when the user passes explicit `(H, W)`
  (compat checks stay meaningful without forcing resolution knowledge).
- `control_hz` passthrough.

**Default `state_map`** (Inspect Robots canonical vocab ⟵→ XPolicyLab v1.0):

| XPolicyLab key | Inspect Robots key |
| --- | --- |
| `arm_joint_state` | `joint_pos` |
| `ee_joint_state` | `gripper` |
| `ee_pose` | `eef_pose` |

(Dual-arm setups pass their own map with `left_*`/`right_*` keys; nothing in
the adapter hard-codes the single-arm family.)

## 6. The client: `_protocol.py` + `_client.py`

- `Frame`: frozen dataclass mirroring upstream's envelope (`to_wire()` /
  `from_wire()`); `MessageType` StrEnum; `WsError(code, message, details)`.
- Codec: `encode_frame`/`decode_frame` via `msgpack` + `msgpack_numpy`
  (rejecting object-dtype arrays, same as upstream).
- `PolicyClient`: built on `websockets.sync.client.connect` — synchronous,
  no private event loop (simpler than upstream's asyncio client; our rollout
  loop is synchronous anyway). Blocking request/response with
  `message_id` correlation (one in-flight request at a time — matches the
  rollout loop), timeout → `WsError("timeout", ...)`; `error` frames raise
  `WsError` with the server's code/message. Connect retries with delay
  (policy cold-start can take minutes while weights load) — configurable
  attempts, quiet logging. A dead socket detected on send/receive marks the
  client disconnected so the adapter's reconnect-on-next-use (§5) can replay
  `hello`.
- `heartbeat`, `prepare_case`, `repeat_index` are implemented in the codec's
  enum for wire fidelity but the client only exposes what the adapter uses
  (`hello`, `reset`, `infer`, `trial_end`, `close`).

## 7. Tests (no GPU, no external processes)

`conftest.py` runs a **stub XPolicyLab server** in-process: a thread serving
`websockets.sync.server` (or asyncio in a thread) that implements
hello/reset/infer/trial_end/close and echoes back demo_policy-shaped action
chunks; recorded requests are exposed to assertions. Port 0 (OS-assigned).

Coverage targets (plugin CI runs pytest without the core 100% gate, same as
isaacsim, but we aim for full-line coverage of `policy.py`):

- registry: entry point resolves; `resolve("policy", "xpolicylab", url=...)`.
- `.info` correct for joint/ee, single/dual arm; no network before first act.
- observation translation: images land under `vision/<slot>/color` untouched;
  state remapped per `state_map`; instruction precedence (observation over
  scene); `additional_info.frequency` present iff `control_hz` set.
- action translation: chunk length H > 1 preserved; flatten order
  arm-then-ee, left-then-right; dtype float; shape mismatch raises.
- latency: `inference_latency_s` is client-measured wall time; the server's
  `latency_ms` lands in `meta["server_latency_ms"]`.
- lifecycle: reset sends protocol reset with fresh `trial_id`; a second
  reset sends `trial_end` for the first trial; envelope `step` zeroes on
  reset and increments per infer; act before reset raises a clear error;
  close sends `trial_end` + `close` and is idempotent; context manager
  works; reconnect-on-next-use after a server restart replays `hello`.
- errors: server `error` frame → `WsError` surfaced; unreachable URL →
  actionable `ConnectionError` mentioning the server command; timeout path.
- compat: `check_compatibility(XPolicyLabPolicy(...), stub_embodiment)`
  passes for the matching profile (default `required_state_keys` ⊆ the
  isaacsim default StateSpec keys) and fails loudly for a dim mismatch.
- factory parsing: string-form `cameras`/`state_map`/`required_state_keys`
  normalize to the dict/set forms.
- protocol golden test: our encoded frames decode (via `msgpack` +
  `msgpack_numpy` directly) into upstream-shaped wire dicts — structural
  equality on fields and payload, including a round-tripped float32 array —
  and upstream-shaped reply dicts decode into our `Frame` (guards codec
  drift without depending on pydantic's field ordering).

## 8. CI, workspace, docs

- `pyproject.toml` mirrors isaacsim's: hatchling, `inspect-robots>=0.2`,
  `[tool.uv.sources] inspect-robots = { workspace = true }`, mypy strict
  with `[[tool.mypy.overrides]] ignore_missing_imports` for `msgpack.*` /
  `msgpack_numpy.*` (no stubs shipped), ruff line-length 100, static
  `version = "0.1.0"` (same release convention as the isaacsim plugin).
  `plugins/*` workspace glob picks it up — `uv sync --all-packages --extra
  dev` just works. Run `uv lock` and commit.
- CI: `plugin-xpolicylab` job cloned from `plugin-isaacsim` (ruff, mypy
  strict on `src/`, pytest) on ubuntu; **added to `ci-ok.needs`** (per
  CLAUDE.md, otherwise it doesn't gate).
- Release: clone `release.yml`'s `publish-isaacsim` job into
  `publish-xpolicylab` (own `environment: pypi-xpolicylab`); a maintainer
  must create the matching PyPI trusted-publisher environment before the
  first release (called out in the PR description — it is a GitHub/PyPI
  settings action, not a code change). Generalize root CLAUDE.md's release
  paragraph, which currently names the isaacsim plugin as the only
  static-version exception.
- Entry point: `[project.entry-points."inspect_robots.policies"]
  xpolicylab = "inspect_robots_xpolicylab:xpolicylab_policy"`.
- Docs: plugin README (quickstart both terminals; arg/mapping tables; split
  vs same-machine; troubleshooting cold-start retries); root README gets a
  short "Bring a policy: XPolicyLab (40+ VLAs)" section next to the Isaac
  embodiment mention; root CLAUDE.md plugin example line gains the new
  package; `src/inspect_robots/CLAUDE.md` untouched (core unchanged).
- **Core is untouched.** No new core deps, no API surface change, no
  `__all__` / api-snapshot churn.

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Protocol drift (no version field upstream) | Golden wire tests; upstream commit hash recorded in `_protocol.py` docstring and README; codec isolated in one file diffable against upstream |
| Action key families beyond demo (`tcp_pose`, `delta_ee_pose`, mobile base) | Flattening reads a *configurable ordered key list* internally (`arm/ee` args are sugar building it); exotic setups pass `action_keys=[...]` explicitly |
| Server returns `actions: None` / empty list | Raise `WsError` with the trial/step context (an empty `ActionChunk` is invalid core-side anyway) |
| Cold-start (weights loading) beats connect timeout | Retry loop with configurable attempts/delay, log per attempt, actionable final error |
| Big frames (multi-camera uint8) | `max_size=None` on connect, same as upstream both sides |
| Windows CI (`test-extra` tier is macOS/Windows) | Plugin job is ubuntu-only like `plugin-isaacsim`; sockets on localhost are CI-safe |
| `msgpack-numpy` is low-maintenance (last release 2021-era) and the weekly canary installs core only, so plugin-dep breakage is invisible to it | Accept for now: the hook surface we use is ~40 lines and vendorable later without API change; the canary gap is recorded here (this plan) rather than user-facing docs |
| Mid-eval socket death | Reconnect-on-next-use with `hello` replay (§5); at most one `PolicyError`d trial per drop |

## 10. Execution steps (each a commit)

1. Plugin skeleton: pyproject, `__init__`, empty modules, README stub;
   `uv lock`; `uv sync --all-packages --extra dev` green.
2. `_protocol.py` + tests (golden wire frames, codec edge cases).
3. `_client.py` + stub-server conftest + client tests.
4. `policy.py` adapter + tests (obs/action mapping, lifecycle, compat).
5. CI job + `ci-ok.needs`; release job (`publish-xpolicylab`) + CLAUDE.md
   release-paragraph generalization; root README + plugin README.
6. Gates: `ruff check`, `ruff format --check`, plugin mypy strict, plugin
   pytest, core suite untouched and green (`uv run pytest --cov`).
