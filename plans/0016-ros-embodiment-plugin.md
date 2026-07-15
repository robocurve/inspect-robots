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
  (some drivers/broadcasters publish one, e.g. UR's `tcp_pose_broadcaster`
  or ros2_control's recent `pose_broadcaster`; otherwise omit it).
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
  reset-time reconnect path would be nearly unreachable: a halt ends the
  eval and the embodiment is closed — by the CLI's own `finally` for
  `inspect-robots run`, or by `eval()` for registry-name-resolved
  embodiments; not worth the code.)

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
  **xyzw** quaternion — the adapter reorders to `[x, y, z, qw, qx, qy, qz]`
  (quat wxyz). Core pins no quaternion order for `eef_pose` (the unit is
  just `"m+quat"`); wxyz is the ecosystem precedent (plan 0007 /
  xpolicylab's ee-pose convention) and the plugin README documents it.
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
- **The publish is gated, not followed, by the pacing sleep**: at the top
  of `step()`, sleep until one control period (`1/control_hz`, monotonic
  clock) has elapsed since the *previous* publish, **then** publish this
  step's command. This enforces the real invariant — every inter-publish
  interval ≥ the period — rather than a constraint on publishes two
  apart (publish-then-sleep would let steady-state intervals average
  `T/2`, streaming commands at ~2× the configured rate on real
  hardware). It composes with open-loop `ActionChunk` playout: mid-chunk,
  buffered pops are fast and the sleep absorbs the overhead; at a chunk
  boundary after slow inference the period has already elapsed, the
  sleep is ~zero, and pacing never adds latency on top of inference.
- Freshness is a **separate** guarantee, not a side effect of the sleep
  (at a chunk boundary there is no sleep, and a state stream throttled at
  exactly the control rate would only race the deadline): after
  publishing, `step()` waits for a joint-states message *newer than the
  cache entry at publish time* — implemented as a per-topic monotonic
  **sequence number** (the receive thread increments it per message;
  `step()` captures it just before publishing and waits for `seq >
  seq_at_publish`), which is immune to equal-timestamp ties that a
  stamp comparison would race on (and that an injected fake clock would
  hit deterministically in tests). Bounded by `fresh_obs_timeout_s`
  (default `2/control_hz`; timeout ⇒ staleness fault naming both
  `fresh_obs_timeout_s` and `control_hz` as the knobs). To keep that wait
  short, state topics subscribe at **2× the control rate**
  (`throttle_rate = 500/control_hz` ms, rounded to an integer ≥ 1 — the
  field is integer milliseconds). Post-command here means *received*
  after the publish; a message sampled just before but delivered just
  after passes the check — a milliseconds-scale approximation on a LAN,
  documented rather than hidden.
- The design assumes the robot's native `joint_states` rate is at least
  ~2× `control_hz` (real arms publish at 25-500 Hz; `throttle_rate` can
  only thin a stream, never speed one up). The first `reset()` runs a
  **preflight rate measurement**: it subscribes to `joint_states`
  *unthrottled*, collects messages for a short window (until 5 messages
  or 1 s, whichever first), computes the native rate from receive-stamp
  deltas, warns when it is below 2× `control_hz` ("lower `control_hz` or
  raise `fresh_obs_timeout_s`"), then swaps to the throttled rollout
  subscription. Measuring *before* throttling is the point: a throttled
  stream observes `ceil(throttle/p)·p` intervals and is mathematically
  capped at 2× `control_hz`, so measuring through it would warn on every
  healthy robot. Tests must cover both directions — a slow publisher
  warns *and a fast one does not*. Degenerate windows: 0 messages falls
  through to the `obs_timeout_s` missing-topic error; exactly 1 message
  (no delta to measure) warns as too-slow-to-measure. **Subscription
  handoff semantics**: rosbridge aggregates same-topic subscriptions
  from one client by least-restrictive parameters, keyed by the optional
  `id` — a naive throttled re-subscribe would leave the unthrottled
  preflight subscription live and silently defeat the throttle. The
  preflight subscribe therefore uses an explicit `id`, the handoff is
  `unsubscribe(preflight id)` → `subscribe(new id, throttled)`, and the
  stub server models per-id subscriptions so the tests can catch a
  wrong handoff.
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
| `ros_version` | `2` | `1` or `2`; selects `JointTrajectory` duration field names and type strings (§2). ROS 1 is kept despite Noetic's 2025 EOL because industrial fleets still run it and the cost is two builder branches plus golden tests — the protocol layer is identical |
| `joints` | **required** | ordered arm joint names; defines `joint_pos` order |
| `joint_states_topic` | `"/joint_states"` | `sensor_msgs/JointState` source |
| `command_topic` | **required** | where arm actions go |
| `command_type` | `"joint_trajectory"` | `"joint_trajectory"` (`trajectory_msgs/JointTrajectory`, one point, `time_from_start` = 1/`control_hz`) or `"float64_multi_array"` (`std_msgs/Float64MultiArray`) |
| `action_low` / `action_high` | **required** | per-arm-joint bounds, `len(joints)` floats each |
| `gripper_topic` | `None` | optional command topic for a 1-DoF gripper; when set, action dim grows by 1 |
| `gripper_command_type` | version-dependent | `"float64_multi_array"` (default when `ros_version=2`: the one-joint `forward_command_controller` interface — stock ROS 2 controllers do not consume a bare `Float64`) or `"float64"` (default when `ros_version=1`: the classic `ros_control` `<controller>/command` interface). README maps both to controller families |
| `gripper_joint` | `None` | joint-states name holding the gripper position (required iff `gripper_topic`) |
| `gripper_low` / `gripper_high` | `None` | **required iff `gripper_topic`**: the gripper's native command range (rad or m — robot-specific; no default, same honesty rule as arm bounds). Appended to the action `Box` bounds and used to normalize the canonical `gripper` observation to 0..1 |
| `gripper_closed_at` | `"low"` | `"low"` or `"high"`: which end of the command range is *closed*. The canonical `gripper` observation pins 0 = closed, 1 = open, and the factory requires `gripper_low < gripper_high` (`Box` itself allows equality), so polarity must be a flag, not a bound swap |
| `eef_pose_topic` | `None` | optional `geometry_msgs/PoseStamped` → observation `eef_pose` (7-D, wxyz) |
| `cameras` | `{}` | camera name → `(topic, height, width)`; compact form `name:topic:WxH` where `640x480` means width 640, height 480 (parser test uses a non-square camera so a swapped parse can't hide). Resolution is required — `CameraSpec.height/width` are mandatory ints, and an embodiment that declares no `CameraSpec` fails compat against every camera-requiring policy (`missing_camera`) |
| `control_hz` | `10.0` | control rate; `step()` paces to it (§3); state subscriptions throttle at 2× it |
| `fresh_obs_timeout_s` | `2/control_hz` | max wait for a post-publish joint-states message in `step()` (§3); timeout ⇒ staleness fault |
| `camera_throttle_ms` | `1000/control_hz` | camera subscribe `throttle_rate`; `0` = unthrottled. Default ties camera bandwidth to the control rate — base64 JSON frames on a thin link otherwise queue in the socket and trip the staleness fault |
| `reset_service` | `None` | optional service called on `reset()` (e.g. a bringup-provided home routine); empty args, must return `result: true` |
| `operator_reset_confirm` | `False` | when true, `reset()` prints the scene instruction and blocks on Enter (TTY prompt via `input()`) — the "human arranges the scene" path. On a non-TTY stdin `input()` raises `EOFError` → `EmbodimentFault` → halt, which is the *intended* outcome for a confirm-required run with no operator (unlike the CLI's own operator-scorer prompt, which degrades to skip); pinned by a test |
| `obs_timeout_s` | `5.0` | max wait for the first message on every subscribed topic (reset-time) |
| `staleness_s` | `2.0` | max age of cached state messages when an observation is assembled; older ⇒ raise (the robot stopped publishing; wrapped into `EmbodimentFault` by the rollout). This is also the **cross-modal skew bound**: a fresh `joint_pos` may pair with an `eef_pose`/camera frame up to `staleness_s` old (up to 20 control periods at the defaults) — `state_time`/`image_times` expose the actual ages, and the README states the bound |
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
  the last dim. Factory validations, all with actionable messages:
  duplicate names within `joints` and an arm joint literally named
  `"gripper"` alongside `gripper_topic` (either would duplicate
  `dim_labels` — a conformance error `eval()` never runs, so left
  unchecked the reorder-by-name would silently duplicate one position
  across dims and publish repeated `joint_names`); `gripper_low >=
  gripper_high` (`Box` itself allows `low <= high` with equality only
  drawing a conformance `zero_width` warning, but the 0..1 normalization
  divides by the range); `control_hz` positive and finite (it divides
  into every throttle and timeout default); `len(action_low)` /
  `len(action_high)` vs `len(joints)` checked before `Box` construction
  ("you gave 5 bounds for 6 joints", not a bare shape mismatch); enum
  args validated (`ros_version` ∈ {1, 2}, `command_type`,
  `gripper_command_type`, `gripper_closed_at`); duplicate camera names
  rejected (the compact-string parser must not silently collapse them).
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
    `(raw - gripper_low) / (gripper_high - gripper_low)` and flipped to
    `1 - that` when `gripper_closed_at="high"`. Declared only when a
    gripper is configured.
  - `eef_pose`, shape `(7,)` — declared only when `eef_pose_topic` set.
    **Hard constraint, enforced in the factory**: when `d == 7` (any
    shape — 6-DoF + gripper *or* a gripperless 7-DoF arm, i.e. Franka /
    iiwa / Kinova Gen3), setting `eef_pose_topic` creates two `(7,)`
    state fields. That is not a cosmetic conformance blemish: the agent
    policy replicates the same shape-match at `bind()` time
    (`inspect_robots_agent/_tools.py` raises `ToolsetError` on ambiguous
    reference fields, and `eval()` binds before any rollout), so
    agent-on-ROS would hard-fail at eval start. The factory therefore
    raises an actionable `ValueError` at construction ("omit
    `eef_pose_topic` on a 7-dim action space until core supports
    key-priority reference matching"). Follow-up core issue: match the
    proprioceptive reference by key priority (`joint_pos` first), not
    shape alone; lift the restriction when it lands.
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
   states / eef pose (`throttle_rate` = `500/control_hz` ms — 2× the
   control rate, see §3 freshness) and cameras
   (`camera_throttle_ms`), then wait up to `obs_timeout_s` for one message
   per subscribed topic — a missing topic fails *here*, at reset, with the
   topic name, not mid-rollout. First camera frames validate declared
   resolutions.
2. `seed` is accepted and ignored (contract allows it; `seedable` is not
   declared).
3. If `reset_service`: `call_service`, require `result: true`.
4. If `operator_reset_confirm`: print `scene.instruction` and block on
   Enter.
5. If *neither* `reset_service` nor `operator_reset_confirm` is
   configured, the second and every later `reset()` warns once on stderr:
   between-trial resets are then no-ops on the physical world, and trial
   N+1 silently starting from trial N's end state is an eval-validity
   trap, not an ops detail.
6. Assemble and return the initial `Observation` (fresh-message wait
   bounded by `obs_timeout_s` — the reset-time bound; arbitrary time may
   have passed during an operator confirm, so the post-confirm wait must
   not reuse the tight `fresh_obs_timeout_s`). Not cache: reset must
   observe the *post-reset* world. Every observation —
   here and in `step()` — carries `scene.instruction` on
   `Observation.instruction`, matching the cubepick/isaacsim precedent
   (policies keep a reset-time fallback, but the embodiment threads it).

**`step(action)`**: sleep-gate on the previous publish (§3), split the
vector into arm command (+ gripper command), build the §2 message shapes
per `ros_version`, `publish` (two publishes when gripper is configured —
the **arm** publish is the single pacing/freshness reference: it
timestamps the sleep-gate and the `seq` capture, so the inter-publish
invariant stays well-defined), wait for a post-publish joint-states
message (§3), assemble the observation (staleness-checked), return `StepResult(observation=..., reward=None,
terminated=False, truncated=False)`. Timing rides on the observation's own
fields — no duplicate `info` payload — with the convention stated here
because no core producer has pinned one yet: `state_time` is the monotonic
receive stamp of the **oldest** state message contributing to the
observation (joint states or eef pose), and `image_times[name]` is each
camera frame's receive stamp. No success oracle ⇒ `terminated` is always
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
  implementation and cover it with a threaded test. Both the client
  (cache receive stamps) and the embodiment (pacing sleep, fresh-obs
  wait, staleness checks) take an injectable monotonic clock + sleep
  (default `time.monotonic`/`time.sleep`), so timing tests control one
  clock domain instead of racing real stamps against a fake clock.
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
  duration fields) / Float64MultiArray builders, and the gripper command
  builders (`float64` and one-element `float64_multi_array`).

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
  raise actionable errors; factory rejections: `d == 7` +
  `eef_pose_topic`, duplicate names in `joints`, arm joint named
  `"gripper"` with a gripper, and `gripper_low >= gripper_high`.
- `.info`: correct spaces/semantics/`dim_labels`/capabilities for all config
  shapes (gripper folds into `joint_pos (d,)` and bounds; normalized
  `gripper` field declared; `self_paced` always on); no network before
  first `reset()` (construct with an unroutable URL).
- conformance: `assert_embodiment_conformant(RosEmbodiment(...).info)`
  passes for gripperless and gripper configs both with and without
  `eef_pose` at `d != 7`; the `d == 7` + `eef_pose_topic` combinations
  (6-DoF+gripper *and* 7-DoF gripperless) are asserted to raise the
  factory's actionable error (pinning the enforced limitation).
- reset: subscribes with `queue_length=1` and the right `throttle_rate`s;
  missing topic ⇒ error naming it within `obs_timeout_s`; camera resolution
  mismatch ⇒ error; `reset_service` called and `result: false` raises;
  fresh-observation wait (a stale pre-reset message is not returned).
- step: JointTrajectory golden shapes for **both** `ros_version`s
  (`{secs,nsecs}` vs `{sec,nanosec}`, type strings) and Float64MultiArray;
  joint order = config order; `time_from_start` = period; gripper split
  publishes two messages with the raw (un-normalized) command in the
  version-appropriate default `gripper_command_type` (and the explicit
  override); the fresh-obs wait keys on the per-topic sequence number
  (equal-stamp tie test); pacing:
  **every inter-publish interval ≥ the control period under fast
  back-to-back steps** (the assertion that catches a publish-then-sleep
  implementation, which passes weaker "slept once" checks while
  streaming at 2×), and the gate is ~zero after slow inference (injected
  fake clock, §6); the fresh-obs wait returns only a post-publish
  joint-states message and faults on `fresh_obs_timeout_s` naming both
  knobs; staleness beyond `staleness_s` raises; `terminated` always
  False; observation `gripper` normalized 0..1 and flipped under
  `gripper_closed_at="high"`; `instruction` threaded onto every
  observation; `state_time`/`image_times` follow the §5 convention.
- reset warnings: the unthrottled preflight window flags a slow native
  `joint_states` rate (< 2× `control_hz`) **and stays silent for a fast
  publisher** (the no-warn direction is the test that catches measuring
  through the throttle); the preflight→throttled handoff
  unsubscribes the preflight `id` (per-id stub assertion); a
  single-message window warns as too-slow-to-measure; the second reset
  with neither `reset_service`
  nor `operator_reset_confirm` warns once about physical-reset absence;
  post-confirm fresh wait is bounded by `obs_timeout_s`, not
  `fresh_obs_timeout_s`; `operator_reset_confirm` on non-TTY stdin
  raises (`EOFError` path pinned).
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
  pytest, ubuntu) and **added to both `ci-ok.needs` and
  `alert-red-main.needs`** (plugin jobs appear in both lists; omitting the
  second would exempt `plugin-ros` failures from red-main stop-the-line
  alerts).
- Release: `publish-ros` job in `release.yml` (environment `pypi-ros`;
  maintainer must create the PyPI trusted-publisher environment before first
  release — settings action, called out in the PR).
- Entry point: `[project.entry-points."inspect_robots.embodiments"]
  ros = "inspect_robots_ros:ros_embodiment"`.
- Docs: plugin README (robot-side rosbridge bringup for ROS 1/ROS 2,
  eval-side quickstart, full config table, ros2_control controller notes —
  which `command_type`/`gripper_command_type` pairs with which controller
  family, including the explicit exclusion that ROS 2's stock
  `gripper_action_controller` is action-only and needs a
  `forward_command_controller` on the gripper joint instead, and the
  `joint_pos` unit caveat that the folded gripper dim is in the gripper's
  native unit, not rad — plus a leading
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
| Stale cache read as live state (robot died, cache survives) | `staleness_s` check on every observation assembly ⇒ fault, not silent stale data; socket death latches the client dead; `step()` additionally requires a post-publish joint-states message (§3), so a pre-command observation can't masquerade as post-command |
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
