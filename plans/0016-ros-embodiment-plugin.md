# 0016 — `inspect-robots-ros`: first-class ROS embodiment plugin (rosbridge protocol)

Issue: robocurve/inspect-robots#104. PR: #105.

## 1. Goal

Give Inspect Robots a first-class `Embodiment` adapter for **any ROS 1 / ROS 2
robot running `rosbridge_server`**, shipped as the fourth in-repo plugin:

```bash
# robot side — any ROS distro, one extra node
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# eval side — no ROS installed at all
inspect-robots run --task my-task --policy agent --embodiment ros \
    -E url=ws://robot:9090 \
    -E joints=joint1,joint2,joint3,joint4,joint5,joint6 \
    -E command_topic=/joint_trajectory_controller/joint_trajectory \
    -E cameras=wrist:/camera/wrist/image_raw/compressed:640x480 \
    -E action_low=-3.1,-2.2,-2.9,-3.1,-2.9,-3.1 \
    -E action_high=3.1,2.2,2.9,3.1,2.9,3.1
```

(`-E k=v` is the CLI's embodiment-args channel — `cli.py` routes `-P` to the
policy and `-E` to the embodiment.)

One adapter ⇒ every registered policy becomes runnable on ROS hardware (and on
ROS-connected simulators like Gazebo): XPolicyLab-served VLAs and, notably, the
`agent` LLM policy — which then covers the "LLM drives a ROS robot" use case
that interactive tools like [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server)
target, but scored, logged, guardrailed, and reproducible.

### Why rosbridge and not MCP (the ROS-MCP evaluation)

Recorded here because the feature request started as "should we integrate with
ROS-MCP?" (issue #104 has the long form):

- ROS-MCP is an MCP **tool surface** for interactive assistants (Claude
  Desktop, Cursor) over a robot. Inspect Robots already has an eval-grade LLM
  tool surface (`inspect-robots-agent`: tools generated from
  `ActionSemantics`, speed-limited playout, approver chain, transcripts in the
  `EvalLog`). Integrating MCP into the closed-loop control path would add a
  request/response tool-calling hop with no action-space contract and no
  scoring hooks.
- ROS-MCP itself reaches the robot over **rosbridge**. Integrating "through"
  ROS-MCP would tunnel rosbridge → MCP → our own tool layer → rosbridge.
- rosbridge v2 is the portable surface of the ROS ecosystem (ROS-MCP,
  Foxglove, roslibjs all target it): JSON over websocket, covers ROS 1 and
  ROS 2, and needs zero ROS packages on the client. That is exactly the
  workspace doctrine — *speak the protocol, don't import the package* (cf.
  plan 0007 §3).

### Non-goals (YAGNI)

- **No mobile-base / Twist control** (`cmd_vel`) — v1 is arm-shaped
  (`joint_pos` control mode). A Twist mode is a small follow-up once the
  transport is proven.
- **No ROS actions** (rosbridge `send_action_goal` ops) and no parameter
  get/set — `step()` publishes to a topic; actions' goal/feedback/result
  lifecycle doesn't fit a control-rate loop.
- **No TF** — `eef_pose` comes from a pose topic if the user has one
  (ros2_control publishes one for most arms; otherwise omit it).
- **No URDF introspection** — action bounds are explicit config (see §5;
  guardrails need honest bounds, and a wrong guess is worse than a required
  arg).
- **No `joint_vel` state field** in v1. Two reasons: (a) many real drivers
  publish `JointState` with empty `velocity` arrays, so declaring it would
  lie for those rigs; (b) `conformance.py`'s `state_alignment` check matches
  the proprioceptive reference field **by shape** — for a gripperless arm,
  `joint_vel (n,)` would collide with `joint_pos (n,)` and fail conformance.
  Follow-up: propose key-priority matching in core, then add `joint_vel`.
- **No raw `sensor_msgs/Image`** in v1 — cameras use `CompressedImage`
  (standard `image_transport` output, tiny on the wire). A `cbor`-compression
  raw-image mode is a documented follow-up.
- **No rosbridge server launching** — the server belongs to the robot bringup;
  we connect to a URL (same stance as plan 0007's no-server-launching).
- **No mid-eval reconnect.** A socket that dies after connect latches the
  client dead; the next client call raises, the rollout wraps it into
  `EmbodimentFault`, and `EmbodimentFault` halts the eval (`errors.py`) —
  which is the right outcome when the link to a moving robot drops. (A
  reset-time reconnect path would be nearly unreachable in CLI runs since
  `eval()` closes registry-resolved embodiments on halt; not worth the
  code.)

## 2. Grounding: the rosbridge v2 protocol (verified against spec)

Facts from `RobotWebTools/rosbridge_suite` `ROSBRIDGE_PROTOCOL.md` (v2.1.0,
fetched 2026-07-15):

- Envelope: every message is a JSON object with an `op` field; optional `id`
  correlates request/response pairs.
- `subscribe`: `topic`, optional `type`, `throttle_rate` (ms between
  messages), `queue_length` (buffer when throttled), `compression` ∈
  {`none`, `png`, `cbor`, `cbor-raw`}. Incoming traffic arrives as
  `{"op": "publish", "topic": ..., "msg": {...}}`.
- `publish`: `topic` + `msg`; servers auto-advertise, but explicit
  `advertise` (`topic` + `type`) is required to *create* a not-yet-existing
  topic with the right type — we always advertise the command topic.
- `call_service`: `service`, optional `args`, reply is `service_response`
  with `values` and boolean `result`.
- With `compression: "none"`, binary fields (e.g. `CompressedImage.data`)
  arrive base64-encoded inside JSON.
- `status` ops carry server-side errors (e.g. type mismatch on publish) at
  configurable verbosity.
- Transport-agnostic spec; websocket is the default and the one we target.

Message-type facts (standard ROS msgs):

- `sensor_msgs/JointState`: parallel arrays `name`, `position`, `velocity`,
  `effort` — name-indexed, order not guaranteed ⇒ the adapter reorders by the
  user's `joints` list every message.
- `sensor_msgs/CompressedImage`: `format` (`"jpeg"`/`"png"`, ROS 1 sometimes
  `"rgb8; jpeg compressed bgr8"`) + `data` (base64 in JSON) — decoded via
  Pillow, converted to `(H, W, 3)` RGB uint8.
- `geometry_msgs/PoseStamped`: `pose.position` xyz + `pose.orientation`
  **xyzw** quaternion — the adapter reorders to the core's canonical
  `eef_pose` form `[x, y, z, qw, qx, qy, qz]` (quat wxyz).
- `trajectory_msgs/JointTrajectory`: `joint_names` + `points[]` with
  `positions` and `time_from_start`. **Not identical across ROS versions**:
  the duration is `{secs, nsecs}` in ROS 1 but `{sec, nanosec}` in ROS 2
  (`builtin_interfaces/Duration`), and rosbridge's message conversion
  rejects unknown fields rather than coercing. Type strings also differ
  (`trajectory_msgs/JointTrajectory` vs `trajectory_msgs/msg/...`). ⇒ a
  `ros_version` config selects the wire shape; golden tests cover both
  (verify the exact accepted type-string forms against rosbridge during
  implementation — ROS 2 rosbridge normalizes the short form, but tests
  pin whatever we send).
- `std_msgs/Float64MultiArray`: `data` — the forward-controller interface.

## 3. The one big decision: an embodiment, not a policy or a tool layer

The ROS integration is an **`Embodiment`** (entry-point group
`inspect_robots.embodiments`, name `ros`), mirroring `inspect-robots-isaacsim`
on the embodiment axis exactly as `inspect-robots-xpolicylab` mirrors it on
the policy axis. Everything else falls out of existing machinery:

- The `agent` policy is embodiment-adaptive (`bind()` builds its tool surface
  from our `ActionSemantics.dim_labels` and spaces) ⇒ LLM-on-ROS-robot works
  with zero agent changes.
- CLI guardrails (Clamp + DeltaLimit) apply to every run by default —
  which is why §5 makes action bounds required config.
- `RerunSink` gives live visualization of a real-robot eval for free.
- Real-hardware honesty is already designed into the core contract
  (`embodiment.py` docstring): reset may block on an operator, there is no
  privileged success oracle — scorers (operator / VLM) own success.

Dependency policy: `inspect-robots>=0.6` (the release that shipped the
conformance kit this plugin's CI asserts against), `numpy`, `websockets>=12`
(sync client, same floor as xpolicylab), `pillow>=10` (CompressedImage
decode; same floor the core's `rerun` extra uses). Pillow is a declared,
required dependency, so it is imported at module top like any other dep (the
lazy-import posture is for optional extras only). **No `rclpy`, no
`roslibpy`, no ROS message packages** — messages are plain JSON dicts and the
five shapes we touch are enumerated in §2.

### Pacing: `self_paced`, explicitly

R1 (plan 0001 §9) says the framework owns pacing *unless the embodiment
declares `self_paced`* — and `rollout.py` documents that framework-side
real-time pacing is deferred "until the first real-robot adapter" and is
**not implemented**: today's loop runs as fast as inference allows. An
unpaced stream of `JointTrajectory` points at LLM/VLA inference cadence is
exactly the degenerate loop R1's escape hatch exists for. So:

- The embodiment declares the `SELF_PACED` capability.
- `step()` publishes the command, then sleeps until one control period
  (`1/control_hz`, monotonic clock, measured from the previous step's
  publish) has elapsed, then assembles the post-action observation from the
  topic cache. The sleep doubles as the freshness window: state topics are
  subscribed at the control rate, so the cache has a post-command message by
  the time the observation is read.
- Core stays untouched; if/when core pacing lands, `self_paced` remains
  correct (the framework defers to it).

## 4. Deliverable layout

```text
plugins/inspect-robots-ros/
├── pyproject.toml                    # hatchling; entry point inspect_robots.embodiments:ros
├── README.md                         # robot-side bringup, eval-side quickstart, config reference
├── src/inspect_robots_ros/
│   ├── __init__.py                   # RosEmbodiment, ros_embodiment factory, __version__
│   ├── _protocol.py                  # op-envelope builders/parsers (subscribe/advertise/publish/call_service/status)
│   ├── _client.py                    # RosbridgeClient: sync ws + receive thread + per-topic latest cache
│   ├── _msgs.py                      # JointState/CompressedImage/PoseStamped/JointTrajectory ↔ numpy
│   └── embodiment.py                 # RosEmbodiment: the inspect_robots.Embodiment adapter
└── tests/
    ├── conftest.py                   # in-process stub rosbridge server (websockets, thread, port 0)
    └── test_ros_embodiment.py
plans/0016-ros-embodiment-plugin.md   # this file
.github/workflows/ci.yml              # + plugin-ros job, wired into ci-ok.needs
.github/workflows/release.yml         # + publish-ros job (environment: pypi-ros)
README.md                             # + "Real robots via ROS" section
CLAUDE.md                             # plugins list mentions the new package
```

## 5. The adapter: `RosEmbodiment`

`RosEmbodiment(EmbodimentBase)`, constructor all-keyword, every arg reachable
from `-E k=v`. The CLI's `parse_value` pre-coerces scalars, so factory
parsers accept `str | int | float | Sequence` for numeric lists (a 1-DoF
`action_low=-3.1` arrives as `float`, a 6-DoF one as the uncoerced string
`"-3.1,-2.2,..."`); mapping args accept dicts programmatically or the
compact string form (`name:topic:WxH,name:topic:WxH` for cameras — topics
never contain `,` or `:`, camera names are required to be identifiers).
The plugin writes these parsers itself; plan 0007's helpers cover only
string→string maps.

| Arg | Default | Meaning |
| --- | --- | --- |
| `url` | `"ws://localhost:9090"` | rosbridge websocket URL (9090 is the rosbridge default) |
| `ros_version` | `2` | `1` or `2`; selects `JointTrajectory` duration field names and type strings (§2) |
| `joints` | **required** | ordered arm joint names; defines `joint_pos` order |
| `joint_states_topic` | `"/joint_states"` | `sensor_msgs/JointState` source |
| `command_topic` | **required** | where arm actions go |
| `command_type` | `"joint_trajectory"` | `"joint_trajectory"` (`trajectory_msgs/JointTrajectory`, one point, `time_from_start` = 1/`control_hz`) or `"float64_multi_array"` (`std_msgs/Float64MultiArray`) |
| `action_low` / `action_high` | **required** | per-arm-joint bounds, `len(joints)` floats each |
| `gripper_topic` | `None` | optional command topic for a 1-DoF gripper (`std_msgs/Float64` position); when set, action dim grows by 1 |
| `gripper_joint` | `None` | joint-states name holding the gripper position (required iff `gripper_topic`) |
| `gripper_low` / `gripper_high` | `None` | **required iff `gripper_topic`**: the gripper's native command range (rad or m — robot-specific; no default, same honesty rule as arm bounds). Appended to the action `Box` bounds and used to normalize the canonical `gripper` observation to 0..1 |
| `eef_pose_topic` | `None` | optional `geometry_msgs/PoseStamped` → observation `eef_pose` (7-D, wxyz) |
| `cameras` | `{}` | camera name → `(topic, height, width)`; compact form `name:topic:WxH`. Resolution is required — `CameraSpec.height/width` are mandatory ints, and an embodiment that declares no `CameraSpec` fails compat against every camera-requiring policy (`missing_camera`) |
| `control_hz` | `10.0` | control rate; `step()` paces to it (§3) and state subscriptions throttle to it |
| `camera_throttle_ms` | `1000/control_hz` | camera subscribe `throttle_rate`; `0` = unthrottled. Default ties camera bandwidth to the control rate — base64 JSON frames on a thin link otherwise queue in the socket and trip the staleness fault |
| `reset_service` | `None` | optional service called on `reset()` (e.g. a bringup-provided home routine); empty args, must return `result: true` |
| `operator_reset_confirm` | `False` | when true, `reset()` prints the scene instruction and blocks on Enter (TTY prompt via `input()`) — the "human arranges the scene" path |
| `obs_timeout_s` | `5.0` | max wait for the first message on every subscribed topic (reset-time) |
| `staleness_s` | `2.0` | max age of cached state messages when an observation is assembled; older ⇒ raise (the robot stopped publishing; wrapped into `EmbodimentFault` by the rollout) |
| `simulated` | `False` | sets `EmbodimentInfo.is_simulated` (Gazebo-behind-rosbridge runs) |
| `name` | `"ros"` | `EmbodimentInfo.name` (users can tag e.g. `"ros:ur5e"`) |
| `connect_timeout_s` / `request_timeout_s` | `10` / `30` | client timeouts |

**Lazy connection** (the xpolicylab/isaacsim invariant): construction and
`.info` never touch the network, so `inspect-robots list embodiments` and
fail-fast compat checks work with no robot. The client connects on first
`reset()`; connection failure raises an actionable error naming the URL and
the one-line rosbridge launch command for ROS 1 and ROS 2.

**`EmbodimentInfo`** (`d = len(joints) + (1 if gripper_topic else 0)`):

- `action_space`: `Box(shape=(d,), low=concat(action_low, [gripper_low]),
  high=concat(action_high, [gripper_high]),
  semantics=ActionSemantics(control_mode="joint_pos", rotation_repr="none",
  gripper="continuous" if gripper_topic else "none", frame="base",
  dim_labels=(*joints, "gripper") or (*joints,)))`. `frame="base"` matches
  the isaacsim precedent for joint-space control (`Frame` has no joint
  value; the field is irrelevant to `joint_pos` mode). `dim_labels` are
  mandatory in practice: conformance errors without them and the agent
  policy's tool surface is built from them; they also pin the gripper to
  the last dim.
- `observation_space.state` (`StateSpec`):
  - `joint_pos`, shape `(d,)` — arm joints in `joints` order, plus the raw
    `gripper_joint` position as the last element when a gripper is
    configured. This is the **proprioceptive reference field**:
    `conformance.state_alignment` requires exactly one field whose shape
    equals the action dim, and folding the gripper in keeps observation
    dim d aligned with command dim d (units: rad for revolute dims; the
    gripper dim is in its native command unit — documented).
  - `gripper`, shape `(1,)` — canonical normalized 0..1 (0 closed, 1 open
    per `CANONICAL_STATE_UNITS`), computed from
    `(raw - gripper_low) / (gripper_high - gripper_low)`. Declared only
    when a gripper is configured.
  - `eef_pose`, shape `(7,)` — declared only when `eef_pose_topic` set.
    **Known corner**: a 6-DoF arm + gripper (d = 7) + `eef_pose` gives two
    `(7,)` fields, which fails the shape-based `state_alignment`
    conformance check (eval itself is unaffected — compat doesn't use
    alignment). Documented in the README; follow-up core issue: match the
    reference field by key priority (`joint_pos` first), not shape alone.
- `observation_space.cameras`: `CameraSpec(name, height, width)` per
  configured camera; the first received frame is validated against the
  declared resolution at reset (mismatch ⇒ error naming both).
- `control_hz` passthrough; `is_simulated` from config.
- `capabilities`: `{"self_paced"}` (§3) plus `"resettable"` iff
  `reset_service` set. **Never** `seedable`, `auto_reset`, or
  `privileged_success` (real hardware; nothing to seed, no oracle).
- `supported_setups` / `supported_target_kinds`: empty (unconstrained) —
  scene realization on hardware is operator- or bringup-owned; the tracer
  has nothing to check.

**`reset(scene, *, seed=None)`**:

1. First call: connect, `advertise` the command (and gripper) topics,
   `subscribe` (always `queue_length=1` — latest-value semantics) to joint
   states / eef pose (`throttle_rate` = `1000/control_hz` ms) and cameras
   (`camera_throttle_ms`), then wait up to `obs_timeout_s` for one message
   per subscribed topic — a missing topic fails *here*, at reset, with the
   topic name, not mid-rollout. First camera frames validate declared
   resolutions.
2. `seed` is accepted and ignored (contract allows it; `seedable` is not
   declared).
3. If `reset_service`: `call_service`, require `result: true`.
4. If `operator_reset_confirm`: print `scene.instruction` and block on
   Enter.
5. Assemble and return the initial `Observation` (fresh-message wait, not
   cache: reset must observe the *post-reset* world).

**`step(action)`**: split the vector into arm command (+ gripper command),
build the §2 message shapes per `ros_version`, `publish` (two publishes when
gripper is configured), sleep out the control period (§3), assemble the
current observation from the topic cache (staleness-checked), return
`StepResult(observation=..., reward=None, terminated=False,
truncated=False)`. Per-topic message ages ride on the observation's own
`state_time` / `image_times` fields (that is what they exist for — no
duplicate `info` payload). No success oracle ⇒ `terminated` is always
False; horizons and policy `request_stop` end trials, scorers decide
success.

**`close()`**: unsubscribe/unadvertise best-effort, close the socket, join
the receive thread, idempotent. `eval()` auto-closes registry-resolved
embodiments ("close what we open"), so no atexit dance is needed on this
axis.

## 6. The client: `_protocol.py` + `_client.py`

- `_protocol.py`: builders/parsers for the ops we use (`subscribe`,
  `unsubscribe`, `advertise`, `unadvertise`, `publish`, `call_service` /
  `service_response`, plus `status` parsing). Pure dict↔dataclass, JSON via
  stdlib. `RosbridgeError(code, message)` for `status` errors and failed
  service calls.
- `_client.py`: `RosbridgeClient` on `websockets.sync.client.connect`
  (`max_size=None`; multi-camera JSON frames are large). One background
  **receive thread** owns `recv()`: it demuxes `publish` ops into a
  per-topic `(msg, monotonic_stamp)` latest-slot under a lock and resolves
  pending `call_service` futures by `id`. This threading split is *new
  design* (xpolicylab's client is single-threaded request/response; only
  the stub-server test pattern carries over) and relies on the websockets
  sync client's documented one-thread-receiving / one-thread-sending
  safety — pin the claim to the installed version's docs during
  implementation and cover it with a threaded test.
- **Error latching**: the receive thread can't raise into the rollout
  thread, and rosbridge reports publish-side type errors asynchronously as
  `status: error` ops. The thread latches the first error (and any socket
  death) on the client; every subsequent client call (`publish`,
  observation read, `call_service`) raises the latched `RosbridgeError` /
  `ConnectionError` first. The rollout wraps it into `EmbodimentFault`,
  which halts the eval (§1 non-goals: no reconnect).
- `_msgs.py`: the message shapes ↔ numpy. JointState reorder-by-name
  (missing joint ⇒ error listing available names), CompressedImage
  base64→Pillow→RGB (module-top `PIL` import — it's a declared dep),
  PoseStamped xyzw→wxyz reorder, JointTrajectory (per-`ros_version`
  duration fields) / Float64MultiArray / Float64 builders.

## 7. Tests (no ROS, no hardware, no external processes)

`conftest.py` runs a **stub rosbridge server** in-process (same pattern as
xpolicylab's stub policy server): a `websockets` server on a thread, port 0,
implementing subscribe/advertise/publish/call_service per §2, publishing
canned JointState/CompressedImage/PoseStamped streams on subscription, and
recording every client op for assertions.

Coverage targets (plugin CI, full-line aim on `embodiment.py`):

- registry: entry point resolves; `resolve("embodiment", "ros", ...)` with
  string-form `-E` args (joints, cameras incl. `WxH`, bounds) normalizes
  correctly; scalar-coerced 1-DoF bounds (float, not str) normalize too;
  required-arg omissions (incl. `gripper_low/high` when `gripper_topic`)
  raise actionable errors.
- `.info`: correct spaces/semantics/`dim_labels`/capabilities for all config
  shapes (gripper folds into `joint_pos (d,)` and bounds; normalized
  `gripper` field declared; `self_paced` always on); no network before
  first `reset()` (construct with an unroutable URL).
- conformance: `assert_embodiment_conformant(RosEmbodiment(...).info)`
  passes for gripperless and gripper configs (no `eef_pose` collision);
  the documented 6-DoF+gripper+`eef_pose` corner is asserted to fail with
  `state_alignment` (pinning the known limitation).
- reset: subscribes with `queue_length=1` and the right `throttle_rate`s;
  missing topic ⇒ error naming it within `obs_timeout_s`; camera resolution
  mismatch ⇒ error; `reset_service` called and `result: false` raises;
  fresh-observation wait (a stale pre-reset message is not returned).
- step: JointTrajectory golden shapes for **both** `ros_version`s
  (`{secs,nsecs}` vs `{sec,nanosec}`, type strings) and Float64MultiArray;
  joint order = config order; `time_from_start` = period; gripper split
  publishes two messages with the raw (un-normalized) command; pacing
  sleeps to `control_hz` (monotonic-clock based, tested with a fake
  clock); staleness beyond `staleness_s` raises; `terminated` always
  False; observation `gripper` normalized 0..1.
- messages: JointState reorder (shuffled names), missing-joint error,
  CompressedImage jpeg + png decode to `(H, W, 3)` uint8 RGB, exotic ROS 1
  `format` strings, PoseStamped xyzw→wxyz reorder.
- protocol golden test: our op dicts match the spec's documented JSON
  shapes verbatim (guards codec drift, mirrors plan 0007's golden wire
  test).
- client: threaded send-while-receiving test; `status: error` latches and
  surfaces on the next call; socket death latches; service-call `id`
  correlation; close is idempotent and joins the thread.
- compat: `check_compatibility` passes against an agent-shaped and an
  xpolicylab-shaped policy stub for a matching profile; camera-requiring
  policy fails compat only when the camera is genuinely undeclared.
- operator confirm: `input()` path exercised via monkeypatched stdin.

## 8. CI, workspace, docs

- `pyproject.toml` mirrors xpolicylab's: hatchling, static
  `version = "0.1.0"`, `inspect-robots>=0.6`, `[tool.uv.sources]
  inspect-robots = { workspace = true }`, mypy strict, ruff line-length 100.
  `uv lock` after adding; workspace glob picks it up.
- CI: `plugin-ros` job cloned from `plugin-xpolicylab` (ruff, mypy strict,
  pytest, ubuntu) and **added to `ci-ok.needs`**.
- Release: `publish-ros` job in `release.yml` (environment `pypi-ros`;
  maintainer must create the PyPI trusted-publisher environment before first
  release — settings action, called out in the PR).
- Entry point: `[project.entry-points."inspect_robots.embodiments"]
  ros = "inspect_robots_ros:ros_embodiment"`.
- Docs: plugin README (robot-side rosbridge bringup for ROS 1/ROS 2,
  eval-side quickstart, full config table, ros2_control controller notes —
  which `command_type` pairs with which controller family — plus a leading
  safety section: e-stop, verified bounds from URDF/datasheet, low
  `max_speed_frac`, supervised first runs; troubleshooting: base64 image
  bandwidth, throttle rates, the 6-DoF+gripper+`eef_pose` conformance
  corner) — public-facing, so it follows the CLAUDE.md writing-style rules
  (no em dashes, etc.). Root README gets a "Real robots via ROS" section
  beside the Isaac/XPolicyLab mentions; root CLAUDE.md plugin list gains
  the package.
- **Core is untouched**: no new core deps, no `__all__`/api-snapshot churn,
  no rollout changes (pacing handled via `self_paced`, §3).

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Base64 JSON images too slow at control rate | `CompressedImage` (jpeg) keeps frames tens-of-KB; camera subscriptions throttle to the control rate by default (`camera_throttle_ms`); documented follow-up: `cbor` subscription mode |
| JointState arrives without all configured joints (some drivers split arm/gripper across publishers) | Reorder-by-name errors list missing vs available; v1 requires `gripper_joint` on the same topic and says so; split-topic support is a follow-up |
| Wrong `command_type` / `ros_version` for the robot's stack | README maps types to controller families; rosbridge `status: error` (type mismatch) latches and surfaces on the next step as `EmbodimentFault` |
| Stale cache read as live state (robot died, cache survives) | `staleness_s` check on every observation assembly ⇒ fault, not silent stale data; socket death latches the client dead |
| Guardrail bounds wrong because user guessed | Arm **and gripper** bounds are required, never defaulted; README tells users to copy them from URDF/datasheet and start with low agent speed caps |
| Unpaced rollout loop on real hardware | `self_paced` + in-`step()` pacing (§3) — the loop cannot outrun `control_hz` regardless of policy inference speed |
| websockets sync client thread-safety assumptions | One-reader (receive thread) / one-writer (rollout thread) split; claim pinned to installed-version docs; threaded test in CI |
| Protocol drift (rosbridge adds/changes ops) | We use stable ops documented since protocol v2.0 (2013); golden shape tests; spec version + fetch date recorded in §2 |
| Real-robot safety | Every CLI run wires Clamp + DeltaLimit by default; bounds required; `terminated` never fakes success; README leads with a safety section |

## 10. Execution steps (each a commit)

1. Plugin skeleton: pyproject, `__init__`, empty modules, README stub;
   `uv lock`; `uv sync --all-packages --extra dev` green.
2. `_protocol.py` + golden shape tests.
3. `_client.py` + stub-server conftest + client tests (threading, cache,
   error latching, service correlation, death latching).
4. `_msgs.py` + tests (reorder, decode, builders for both `ros_version`s).
5. `embodiment.py` adapter + tests (info, conformance, reset, step incl.
   pacing, lifecycle, compat).
6. CI job + `ci-ok.needs`; `publish-ros` release job; root README + plugin
   README + CLAUDE.md touches.
7. Gates: `ruff check`, `ruff format --check`, plugin mypy strict, plugin
   pytest, core suite untouched and green (`uv run pytest --cov`).
