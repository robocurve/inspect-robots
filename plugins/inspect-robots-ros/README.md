# inspect-robots-ros

## Safety

> [!WARNING]
> This adapter sends commands to physical hardware. Keep a tested emergency
> stop within reach, and do not run an unsupervised evaluation until the full
> command path has been checked on your robot.

- Verify every arm and gripper bound against the robot URDF, manufacturer
  datasheet, and controller configuration.
- Start with a low policy speed limit such as a low `max_speed_frac`, then
  increase it only after reviewing commanded and measured motion.
- Supervise first runs with the workspace clear and the robot at reduced
  hardware speed.
- Confirm that loss of rosbridge communication stops motion safely at the
  controller or robot level.

The package registers the `ros` Inspect Robots embodiment. It connects directly
to `rosbridge_server`, so the evaluation machine needs no ROS installation,
ROS message packages, or robot-vendor SDK.

## Install

```bash
pip install inspect-robots-ros
```

The embodiment then appears in `inspect-robots list embodiments`.

## Robot-side bringup

Install `rosbridge_server` in the robot's ROS environment and start its
websocket endpoint. Port 9090 is the rosbridge default.

ROS 1:

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

ROS 2:

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

The evaluation host must be able to reach the websocket URL. Restrict network
access to trusted hosts because rosbridge exposes ROS topics and services.
The plugin does not launch rosbridge or install anything on the robot.

## Quickstart

This six-joint example publishes `JointTrajectory` commands, reads a wrist
camera, and uses explicit joint limits:

```bash
inspect-robots run --task my-task --policy agent --embodiment ros \
    -E url=ws://robot:9090 \
    -E joints=joint1,joint2,joint3,joint4,joint5,joint6 \
    -E command_topic=/joint_trajectory_controller/joint_trajectory \
    -E cameras=wrist:/camera/wrist/image_raw/compressed:640x480 \
    -E action_low=-3.1,-2.2,-2.9,-3.1,-2.9,-3.1 \
    -E action_high=3.1,2.2,2.9,3.1,2.9,3.1
```

The embodiment composes with any compatible policy. The same robot
configuration evaluates an XPolicyLab-served VLA (any of the 40+ policies
behind [inspect-robots-xpolicylab](../inspect-robots-xpolicylab/)) instead of
an LLM:

```bash
inspect-robots run --task my-task --embodiment ros \
    --policy xpolicylab -P url=ws://gpu-box:19000 -P cameras=cam_head:wrist \
    -E url=ws://robot:9090 \
    -E joints=joint1,joint2,joint3,joint4,joint5,joint6 \
    -E command_topic=/joint_trajectory_controller/joint_trajectory \
    -E cameras=wrist:/camera/wrist/image_raw/compressed:640x480 \
    -E action_low=-3.1,-2.2,-2.9,-3.1,-2.9,-3.1 \
    -E action_high=3.1,2.2,2.9,3.1,2.9,3.1
```

`-P` arguments go to the policy and `-E` arguments go to the embodiment, so
the robot half of the command never changes when you swap policies.

Construction and `.info` are network-free. The websocket connects on the first
`reset()`, after compatibility and guardrail checks have inspected the declared
spaces.

## Configuration

Pass values as `-E key=value` arguments or as keyword arguments to
`RosEmbodiment`. Compact lists use commas. Camera entries use
`name:topic:WxH`; `640x480` means width 640 and height 480.

| Argument | Default | Meaning |
| --- | --- | --- |
| `url` | `ws://localhost:9090` | rosbridge websocket URL. |
| `ros_version` | `2` | `1` or `2`. Selects message type strings and `JointTrajectory` duration keys. |
| `joints` | required | Ordered arm joint names. This order defines arm actions and `joint_pos`. |
| `joint_states_topic` | `/joint_states` | `sensor_msgs/JointState` source. Arm and configured gripper names must appear in this topic. |
| `command_topic` | required | Arm controller command topic. |
| `command_type` | `joint_trajectory` | `joint_trajectory` or `float64_multi_array`. |
| `action_low` | required | One lower command bound per arm joint. |
| `action_high` | required | One upper command bound per arm joint. |
| `gripper_topic` | `None` | Optional gripper command topic. Adds one final action dimension. |
| `gripper_command_type` | ROS 1: `float64`; ROS 2: `float64_multi_array` | Gripper wire message type. |
| `gripper_joint` | `None` | JointState name for measured gripper position. Required exactly when `gripper_topic` is set. |
| `gripper_low` | `None` | Native lower gripper command bound. Required with `gripper_topic`. |
| `gripper_high` | `None` | Native upper gripper command bound. Required with `gripper_topic` and must exceed `gripper_low`. |
| `gripper_closed_at` | `low` | `low` or `high`, identifying which native bound means closed. |
| `eef_pose_topic` | `None` | Optional `geometry_msgs/PoseStamped` source for `eef_pose`. |
| `cameras` | none | Camera name to compressed topic and resolution. Compact form: `wrist:/camera/compressed:640x480`. |
| `control_hz` | `10.0` | Command rate. The adapter sleep-gates each arm publish to this rate. |
| `fresh_obs_timeout_s` | `2/control_hz` | Maximum step-time wait for a sequence-newer joint-state message. |
| `camera_throttle_ms` | `1000/control_hz` | Camera subscription throttle in milliseconds. Set `0` for unthrottled. |
| `reset_service` | `None` | Optional empty-argument service called on every reset. It must return `result: true`. |
| `operator_reset_confirm` | `False` | Print the scene instruction and require Enter on every reset. EOF stops the run. |
| `obs_timeout_s` | `5.0` | Reset-time wait for initial and post-reset messages on every configured topic. |
| `staleness_s` | `2.0` | Maximum cached sample age and cross-modal skew bound during observation assembly. |
| `simulated` | `False` | Set true for a simulator such as Gazebo behind rosbridge. |
| `name` | `ros` | Embodiment name recorded in logs, for example `ros:ur5e`. |
| `connect_timeout_s` | `10` | Websocket connection timeout. |
| `request_timeout_s` | `30` | Service response timeout. |

Arm bounds and gripper bounds form the action `Box`, so the default Inspect
Robots clamp and delta-limit guardrails can reject unsafe policy output before
it reaches rosbridge.

## Controller mapping

Choose settings that match the controller's subscribed ROS message type.

| ROS controller family | Plugin setting | Typical topic and message |
| --- | --- | --- |
| ROS 1 `joint_trajectory_controller/JointTrajectoryController` | `command_type=joint_trajectory` | `<controller>/command`, `trajectory_msgs/JointTrajectory` |
| ROS 1 `position_controllers/JointGroupPositionController` | `command_type=float64_multi_array` | `<controller>/command`, `std_msgs/Float64MultiArray` |
| ROS 2 `joint_trajectory_controller/JointTrajectoryController` | `command_type=joint_trajectory` | `<controller>/joint_trajectory`, `trajectory_msgs/msg/JointTrajectory` |
| ROS 2 `forward_command_controller/ForwardCommandController` | `command_type=float64_multi_array` | `<controller>/commands`, `std_msgs/msg/Float64MultiArray` |
| ROS 1 single-joint position controller for a gripper | `gripper_command_type=float64` | `<controller>/command`, `std_msgs/Float64` |
| ROS 2 forward command controller for a gripper joint | `gripper_command_type=float64_multi_array` | `<controller>/commands`, one-element `std_msgs/msg/Float64MultiArray` |

The stock ROS 2 `gripper_action_controller` is action-only and is not supported
by this publish-based adapter. Configure a `forward_command_controller` on the
gripper joint instead. The plugin does not send ROS action goals.

## Observation and timing contract

- `joint_pos` follows the configured arm joint order. With a gripper, its raw
  measured position is folded into the last element so proprioception matches
  the action dimension.
- Arm joint positions are normally radians. The folded gripper element remains
  in its controller's native unit, commonly radians or metres. Do not assume
  every element of a gripper-equipped `joint_pos` vector has the same unit.
- `gripper` is a separate normalized field with shape `(1,)`: 0 means closed
  and 1 means open. `gripper_closed_at=high` flips the native range.
- `eef_pose` is `[x, y, z, qw, qx, qy, qz]`. ROS supplies quaternion fields as
  xyzw, and the adapter reorders them to wxyz.
- `state_time` is the monotonic receive time of the oldest state message used
  for an observation. `image_times[name]` records each camera frame's receive
  time.
- A fresh joint state may be combined with an end-effector pose or camera frame
  up to `staleness_s` old. This is the explicit cross-modal skew bound.

The adapter subscribes to joint state at twice `control_hz`. On the first
reset, it measures the unthrottled native rate and warns when the publisher is
slower than that target. Each step captures the joint-state sequence just
before the arm publish and waits for a greater sequence afterward. A message
sampled just before the command but received just after it can satisfy this
check; this is a receive-time approximation intended for low-latency links.

## Reset behavior

The first reset advertises command topics, performs the native-rate preflight,
subscribes with queue length 1, and verifies every configured topic and camera
resolution. Every reset then calls `reset_service` when configured, prompts the
operator when requested, and waits for newer messages with `obs_timeout_s`.

If neither reset path is configured, the adapter warns once on the second
reset because it cannot change the physical scene between trials. An operator
confirmation on non-interactive stdin raises `EOFError` and halts the run.

## Troubleshooting

- Connection failures name the URL and both rosbridge launch commands. Confirm
  the server is listening, port 9090 is reachable, and no proxy intercepts the
  websocket.
- A missing-joint error lists all names present in `JointState`. Version 0.1
  requires arm and gripper state on the same topic.
- Fresh-observation timeouts usually mean the native joint-state publisher is
  slower than twice `control_hz`. Lower `control_hz` or raise
  `fresh_obs_timeout_s`.
- Staleness errors mean a configured state or camera topic stopped arriving.
  Check the publisher and consider a larger `staleness_s` only after confirming
  that older cross-modal pairing is acceptable.
- Compressed images are base64-encoded inside rosbridge JSON. Reduce JPEG size
  or increase `camera_throttle_ms` when image traffic queues behind state.
- Camera resolution is written as width by height, but images are represented
  as `(height, width, 3)` RGB arrays.
- A rosbridge `status:error` after publishing usually indicates a mismatched
  `ros_version`, `command_type`, or controller message type. The first error is
  latched and all later client calls fail.
- A 7-dimensional action space cannot also declare `eef_pose` in this release.
  This includes a seven-joint arm and a six-joint arm with a gripper. Omit
  `eef_pose_topic` until core supports key-priority reference matching.
