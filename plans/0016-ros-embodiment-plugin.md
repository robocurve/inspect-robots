# 0016 — `inspect-robots-ros`: first-class ROS embodiment plugin (rosbridge protocol)

Issue: robocurve/inspect-robots#104.

## 1. Goal

Give Inspect Robots a first-class `Embodiment` adapter for **any ROS 1 / ROS 2
robot running `rosbridge_server`**, shipped as the fourth in-repo plugin:

```bash
# robot side — any ROS distro, one extra node
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# eval side — no ROS installed at all
inspect-robots run --task my-task --policy agent --embodiment ros \
    -P url=ws://robot:9090 \
    -P joints=joint1,joint2,joint3,joint4,joint5,joint6 \
    -P command_topic=/joint_trajectory_controller/joint_trajectory \
    -P cameras=wrist:/camera/wrist/image_raw/compressed \
    -P action_low=-3.1,-2.2,-2.9,-3.1,-2.9,-3.1 \
    -P action_high=3.1,2.2,2.9,3.1,2.9,3.1
```

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
- **No raw `sensor_msgs/Image`** in v1 — cameras use `CompressedImage`
  (standard `image_transport` output, tiny on the wire). A `cbor`-compression
  raw-image mode is a documented follow-up.
- **No rosbridge server launching** — the server belongs to the robot bringup;
  we connect to a URL (same stance as plan 0007's no-server-launching).

## 2. Grounding: the rosbridge v2 protocol (verified against spec)

Facts from `RobotWebTools/rosbridge_suite` `ROSBRIDGE_PROTOCOL.md` (v2.1.0,
fetched 2026-07-15):

- Envelope: every message is a JSON object with an `op` field; optional `id`
  correlates request/response pairs.
- `subscribe`: `topic`, optional `type`, `throttle_rate` (ms between
  messages), `queue_length`, `compression` ∈ {`none`, `png`, `cbor`,
  `cbor-raw`}. Incoming traffic arrives as `{"op": "publish", "topic": ...,
  "msg": {...}}`.
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

Message-type facts (standard ROS msgs, stable across ROS 1/2):

- `sensor_msgs/JointState`: parallel arrays `name`, `position`, `velocity`,
  `effort` — name-indexed, order not guaranteed ⇒ the adapter reorders by the
  user's `joints` list every message.
- `sensor_msgs/CompressedImage`: `format` (`"jpeg"`/`"png"`, ROS 1 sometimes
  `"rgb8; jpeg compressed bgr8"`) + `data` (base64 in JSON) — decoded via
  Pillow, converted to `(H, W, 3)` RGB uint8.
- `geometry_msgs/PoseStamped`: `pose.position` xyz + `pose.orientation`
  xyzw quaternion — reordered to the core's `[x, y, z, qw, qx, qy, qz]`
  (`eef_pose` canonical form, quat wxyz; same reorder xpolicylab does).
- `trajectory_msgs/JointTrajectory`: `joint_names` + `points[]` with
  `positions` and `time_from_start` — the standard ros2_control position
  interface.
- `std_msgs/Float64MultiArray`: `data` — the forward-controller interface.

## 3. The one big decision: an embodiment, not a policy or a tool layer

The ROS integration is an **`Embodiment`** (entry-point group
`inspect_robots.embodiments`, name `ros`), mirroring `inspect-robots-isaacsim`
on the embodiment axis exactly as `inspect-robots-xpolicylab` mirrors it on
the policy axis. Everything else falls out of existing machinery:

- The `agent` policy is embodiment-adaptive (`bind()` builds its tool surface
  from our `ActionSemantics`) ⇒ LLM-on-ROS-robot works with zero agent
  changes.
- CLI guardrails (Clamp + DeltaLimit) apply to every run by default —
  which is why §5 makes action bounds required config.
- `RerunSink` gives live visualization of a real-robot eval for free.
- Real-hardware honesty is already designed into the core contract
  (`embodiment.py` docstring): reset may block on an operator, there is no
  privileged success oracle — scorers (operator / VLM) own success.

Dependency policy: `inspect-robots>=0.4`, `numpy`, `websockets>=12`
(sync client, same floor as xpolicylab), `pillow>=10` (CompressedImage
decode; already the floor core uses for its `rerun` extra). **No `rclpy`, no
`roslibpy`, no ROS message packages** — messages are plain JSON dicts and the
five shapes we touch are enumerated in §2.

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
from `-P k=v` (mapping/list args accept the compact string forms plan 0007
established: `name:topic,name:topic` and comma-separated lists — topic names
never contain `,` or the pre-colon camera names we require to be simple
identifiers):

| Arg | Default | Meaning |
| --- | --- | --- |
| `url` | `"ws://localhost:9090"` | rosbridge websocket URL (9090 is the rosbridge default) |
| `joints` | **required** | ordered joint names; defines `joint_pos` order and action dim |
| `joint_states_topic` | `"/joint_states"` | `sensor_msgs/JointState` source |
| `command_topic` | **required** | where actions go |
| `command_type` | `"joint_trajectory"` | `"joint_trajectory"` (`trajectory_msgs/JointTrajectory`, one point, `time_from_start` = 1/`control_hz`) or `"float64_multi_array"` (`std_msgs/Float64MultiArray`) |
| `action_low` / `action_high` | **required** | per-joint bounds (comma-separated floats); gripper bound appended when `gripper_topic` set |
| `gripper_topic` | `None` | optional command topic for a 1-DoF gripper (`std_msgs/Float64` position); when set, action dim grows by 1 and observation `gripper` is read from `gripper_joint` in joint states |
| `gripper_joint` | `None` | joint-states name holding gripper position (required iff `gripper_topic`) |
| `eef_pose_topic` | `None` | optional `geometry_msgs/PoseStamped` → observation `eef_pose` (7-D, wxyz) |
| `cameras` | `{}` | camera name → `sensor_msgs/CompressedImage` topic |
| `control_hz` | `10.0` | declared control rate; also the subscribe `throttle_rate` for state topics |
| `reset_service` | `None` | optional service called on `reset()` (e.g. a bringup-provided home routine); empty args, must return `result: true` |
| `operator_reset_confirm` | `False` | when true, `reset()` prints the scene instruction and blocks on Enter (TTY prompt via `input()`) — the "human arranges the scene" path |
| `obs_timeout_s` | `5.0` | max wait for the first message on every subscribed topic (reset-time) |
| `staleness_s` | `2.0` | max age of cached state messages when an observation is assembled; older ⇒ raise (embodiment fault — the robot stopped publishing) |
| `simulated` | `False` | sets `EmbodimentInfo.is_simulated` (Gazebo-behind-rosbridge runs) |
| `name` | `"ros"` | `EmbodimentInfo.name` (users can tag e.g. `"ros:ur5e"`) |
| `connect_timeout_s` / `request_timeout_s` | `10` / `30` | client timeouts |

**Lazy connection** (the xpolicylab/isaacsim invariant): construction and
`.info` never touch the network, so `inspect-robots list embodiments` and
fail-fast compat checks work with no robot. The client connects on first
`reset()`; connection failure raises an actionable error naming the URL and
the one-line rosbridge launch command for ROS 1 and ROS 2.

**`EmbodimentInfo`**:

- `action_space`: `Box(low=action_low, high=action_high)` with
  `ActionSemantics(control_mode="joint_pos", gripper="continuous" iff
  gripper_topic, frame="joint", bounds=explicit)` — sized `len(joints)`
  (+1 with gripper).
- `observation_space`: `StateSpec` with `joint_pos` (dim `len(joints)`),
  plus `gripper` (1) and `eef_pose` (7) when configured; `CameraSpec` per
  camera **without** resolution (unknown until frames arrive; compat stays
  meaningful, same stance as plan 0007).
- `control_hz` passthrough; `is_simulated` from config.
- `capabilities`: `{"resettable"}` iff `reset_service` set. **Never**
  `seedable`, `auto_reset`, or `privileged_success` (real hardware; nothing
  to seed, no oracle). Not `self_paced`: `step()` returns immediately after
  the publish and the framework paces the loop — R1's default is exactly
  what a stream-commanded robot wants.
- `supported_setups` / `supported_target_kinds`: empty (unconstrained) —
  scene realization on hardware is operator- or bringup-owned; the tracer
  has nothing to check.

**`reset(scene, *, seed=None)`**:

1. First call: connect, `advertise` the command (and gripper) topics,
   `subscribe` to joint states / eef pose / cameras (`throttle_rate` =
   `1000/control_hz` ms for state, cameras unthrottled), then wait up to
   `obs_timeout_s` for one message per subscribed topic — a missing topic
   fails *here*, at reset, with the topic name, not mid-rollout.
2. `seed` is accepted and ignored (contract allows it; `seedable` is not
   declared).
3. If `reset_service`: `call_service`, require `result: true`.
4. If `operator_reset_confirm`: print `scene.instruction` and block on
   Enter.
5. Assemble and return the initial `Observation` (fresh-message wait, not
   cache: reset must observe the *post-reset* world).

**`step(action)`**: split the vector into arm command (+ gripper command),
build the §2 message shapes, `publish` (two publishes when gripper is
configured), assemble the current observation from the topic cache
(staleness-checked), return
`StepResult(observation=..., reward=None, terminated=False, truncated=False,
info={"staleness_s": per-topic ages})`. No success oracle ⇒ `terminated`
is always False; horizons and policy `request_stop` end trials, scorers
decide success.

**`close()`**: unsubscribe/unadvertise best-effort, close the socket,
idempotent. `eval()` auto-closes registry-resolved embodiments ("close what
we open"), so no atexit dance is needed on this axis.

## 6. The client: `_protocol.py` + `_client.py`

- `_protocol.py`: builders/parsers for the six ops we use (`subscribe`,
  `unsubscribe`, `advertise`, `unadvertise`, `publish`, `call_service` /
  `service_response`, plus `status` parsing). Pure dict↔dataclass, JSON via
  stdlib. `RosbridgeError(code, message)` for `status` errors and failed
  service calls.
- `_client.py`: `RosbridgeClient` on `websockets.sync.client.connect`
  (`max_size=None`; multi-camera JSON frames are large). One background
  **receive thread** owns `recv()`: it demuxes `publish` ops into a
  per-topic `(msg, monotonic_stamp)` latest-slot under a lock, resolves
  pending `call_service` futures by `id`, and surfaces `status: error`
  ops. Sends happen from the rollout thread — the websockets sync client
  documents one-reader/one-writer thread safety, which is exactly this
  split. A dead socket marks the client disconnected; the *next* `reset()`
  reconnects and replays advertises/subscribes (mid-trial death raises —
  a dead robot link mid-motion must fault the trial, mirroring
  `EmbodimentFault` semantics, not silently reconnect).
- `_msgs.py`: the five message shapes ↔ numpy. JointState reorder-by-name
  (missing joint ⇒ error listing available names), CompressedImage
  base64→Pillow→RGB (lazy `PIL` import at first frame, matching the core's
  lazy-import posture), PoseStamped xyzw→wxyz reorder, JointTrajectory /
  Float64MultiArray builders.

## 7. Tests (no ROS, no hardware, no external processes)

`conftest.py` runs a **stub rosbridge server** in-process (same pattern as
xpolicylab's stub policy server): a `websockets` server on a thread, port 0,
implementing subscribe/advertise/publish/call_service per §2, publishing
canned JointState/CompressedImage/PoseStamped streams on subscription, and
recording every client op for assertions.

Coverage targets (plugin CI, full-line aim on `embodiment.py`):

- registry: entry point resolves; `resolve("embodiment", "ros", ...)` with
  string-form `-P` args (joints, cameras, bounds) normalizes correctly;
  required-arg omissions raise actionable `TypeError`s.
- `.info`: correct spaces/semantics/capabilities for all config shapes; no
  network before first `reset()` (construct with an unroutable URL).
- reset: subscribes with the right `throttle_rate`; missing topic ⇒ error
  naming it within `obs_timeout_s`; `reset_service` called and `result:
  false` raises; fresh-observation wait (a stale pre-reset message is not
  returned); reconnect-after-drop replays advertise/subscribe.
- step: JointTrajectory and Float64MultiArray messages have the §2 golden
  shapes (joint order = config order, `time_from_start` = period); gripper
  split publishes two messages; staleness beyond `staleness_s` raises;
  `terminated` always False.
- messages: JointState reorder (shuffled names), gripper extraction,
  CompressedImage jpeg + png decode to `(H, W, 3)` uint8 RGB, exotic ROS 1
  `format` strings, PoseStamped wxyz reorder.
- protocol golden test: our op dicts match the spec's documented JSON
  shapes verbatim (guards codec drift, mirrors plan 0007's golden wire
  test).
- lifecycle: close is idempotent, unsubscribes; mid-trial socket death ⇒
  raise (not silent reconnect); `status: error` from server surfaces as
  `RosbridgeError`.
- compat: `check_compatibility(agent_policy-shaped stub / xpolicylab-shaped
  stub, RosEmbodiment(...))` passes for a matching profile; conformance kit
  (`assert_embodiment_conformant`) passes against the stub server.
- operator confirm: `input()` path exercised via monkeypatched stdin.

## 8. CI, workspace, docs

- `pyproject.toml` mirrors xpolicylab's: hatchling, static
  `version = "0.1.0"`, `inspect-robots>=0.4`, `[tool.uv.sources]
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
  eval-side quickstart, full config table, ros2_control controller notes,
  troubleshooting: base64 image bandwidth, throttle_rate, firewalls) —
  public-facing, so it follows the CLAUDE.md writing-style rules (no em
  dashes, etc.). Root README gets a "Real robots via ROS" section beside the
  Isaac/XPolicyLab mentions; root CLAUDE.md plugin list gains the package.
- **Core is untouched**: no new core deps, no `__all__`/api-snapshot churn.

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Base64 JSON images too slow at control rate | `CompressedImage` (jpeg) keeps frames tens-of-KB; cameras unthrottled but latest-slot semantics drop backlog; documented follow-up: `cbor` subscription mode |
| JointState arrives without all configured joints (some drivers split arm/gripper across publishers) | Reorder-by-name errors list missing vs available; `gripper_joint` may come from the same or another topic later — v1 requires same-topic and says so |
| Wrong `command_type` for the robot's controller stack | README maps the two types to ros2_control controller families; `status: error` from rosbridge (type mismatch) surfaces as `RosbridgeError` at first step |
| Stale cache read as live state (robot died, cache survives) | `staleness_s` check on every observation assembly ⇒ fault, not silent stale data |
| Guardrail bounds wrong because user guessed | Bounds are required, never defaulted; README tells users to copy them from their URDF/datasheet and start with `--policy agent -P max_speed_frac=0.05` style caution |
| websockets sync client thread-safety assumptions | One-reader (receive thread) / one-writer (rollout thread) split is the documented safe pattern; service futures resolved via the receive thread only |
| Protocol drift (rosbridge adds/changes ops) | We use six stable ops documented since protocol v2.0 (2013); golden shape tests; spec version + fetch date recorded in §2 |
| Real-robot safety | Every CLI run wires Clamp + DeltaLimit by default; bounds required; `terminated` never fakes success; README leads with a safety section (e-stop, speed fractions, supervised first runs) |

## 10. Execution steps (each a commit)

1. Plugin skeleton: pyproject, `__init__`, empty modules, README stub;
   `uv lock`; `uv sync --all-packages --extra dev` green.
2. `_protocol.py` + golden shape tests.
3. `_client.py` + stub-server conftest + client tests (threading, cache,
   service correlation, death/reconnect).
4. `_msgs.py` + tests (reorder, decode, builders).
5. `embodiment.py` adapter + tests (info, reset, step, lifecycle, compat,
   conformance).
6. CI job + `ci-ok.needs`; `publish-ros` release job; root README + plugin
   README + CLAUDE.md touches.
7. Gates: `ruff check`, `ruff format --check`, plugin mypy strict, plugin
   pytest, core suite untouched and green (`uv run pytest --cov`).
