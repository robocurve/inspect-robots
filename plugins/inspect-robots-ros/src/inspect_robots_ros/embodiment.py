"""Expose ROS robots and rosbridge streams through the embodiment contract.

The adapter is configuration-driven and ROS-free on the evaluation machine.
Construction builds only static spaces; the websocket connects lazily on the
first reset. Commands are sleep-gated before the arm publish, then paired with
a sequence-newer joint-state message before an observation is returned.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from inspect_robots import (
    Action,
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentBase,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Scene,
    StateField,
    StateSpec,
    StepResult,
)
from inspect_robots.embodiment import RESETTABLE, SELF_PACED
from inspect_robots.spaces import CANONICAL_STATE_UNITS
from inspect_robots_ros._client import RosbridgeClient, TopicSample
from inspect_robots_ros._msgs import (
    MessageKind,
    build_float64_multi_array,
    build_gripper_command,
    build_joint_trajectory,
    message_type,
    parse_compressed_image,
    parse_joint_state,
    parse_pose_stamped,
)

_PREFLIGHT_SUBSCRIPTION_ID = "inspect-robots-preflight-joint-states"
_JOINT_SUBSCRIPTION_ID = "inspect-robots-joint-states"
_EEF_SUBSCRIPTION_ID = "inspect-robots-eef-pose"


@dataclass(frozen=True)
class _CameraConfig:
    topic: str
    height: int
    width: int


NumericList = str | int | float | Sequence[float]
JointList = str | Sequence[str]
CameraMap = Mapping[str, tuple[str, int, int]]


def _parse_joints(value: JointList | None) -> tuple[str, ...]:
    if value is None:
        raise ValueError("joints is required; pass -E joints=joint1,joint2,...")
    if isinstance(value, str):
        joints = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        joints = tuple(str(item).strip() for item in value if str(item).strip())
    if not joints:
        raise ValueError("joints must contain at least one joint name")
    duplicates = sorted({name for name in joints if joints.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"joints contains duplicate names {duplicates}; each arm joint must be unique"
        )
    return joints


def _parse_numeric_list(value: NumericList | None, arg: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"{arg} is required; pass one numeric bound per configured arm joint")
    raw: Sequence[Any]
    if isinstance(value, str):
        raw = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, (int, float)):
        raw = (value,)
    else:
        raw = value
    try:
        parsed = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg} must be a comma-separated list of numbers") from exc
    if not parsed:
        raise ValueError(f"{arg} must contain at least one numeric bound")
    return parsed


def _parse_cameras(value: CameraMap | str | None) -> dict[str, _CameraConfig]:
    if value is None:
        return {}
    if isinstance(value, str):
        cameras: dict[str, _CameraConfig] = {}
        for raw_entry in value.split(","):
            entry = raw_entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"cameras entry {entry!r} must be name:topic:WxH, "
                    "for example wrist:/camera/image_raw/compressed:640x480"
                )
            name, topic, resolution = (part.strip() for part in parts)
            if name in cameras:
                raise ValueError(
                    f"cameras contains duplicate name {name!r}; camera names must be unique"
                )
            width_text, separator, height_text = resolution.lower().partition("x")
            if not separator:
                raise ValueError(
                    f"camera {name!r} resolution {resolution!r} must be WxH, for example 640x480"
                )
            try:
                width, height = int(width_text), int(height_text)
            except ValueError as exc:
                raise ValueError(
                    f"camera {name!r} resolution {resolution!r} must contain "
                    "integer width and height"
                ) from exc
            cameras[name] = _validate_camera(name, topic, height, width)
        if value.strip() and not cameras:
            raise ValueError(f"cameras string form parsed to no cameras: {value!r}")
        return cameras

    cameras = {}
    for name, config in value.items():
        if not isinstance(config, Sequence) or isinstance(config, str) or len(config) != 3:
            raise ValueError(f"camera {name!r} must map to (topic, height, width), got {config!r}")
        topic, height, width = config
        cameras[name] = _validate_camera(name, topic, height, width)
    return cameras


def _validate_camera(name: str, topic: Any, height: Any, width: Any) -> _CameraConfig:
    if not name.isidentifier():
        raise ValueError(f"camera name {name!r} must be a valid identifier")
    if not isinstance(topic, str) or not topic:
        raise ValueError(f"camera {name!r} topic must be a non-empty string")
    if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1:
        raise ValueError(
            f"camera {name!r} height and width must be positive integers, "
            f"got height={height!r}, width={width!r}"
        )
    return _CameraConfig(topic=topic, height=height, width=width)


class RosEmbodiment(EmbodimentBase):
    """Drive a joint-position ROS arm through a rosbridge websocket connection."""

    def __init__(
        self,
        *,
        url: str = "ws://localhost:9090",
        ros_version: int = 2,
        joints: JointList | None = None,
        joint_states_topic: str = "/joint_states",
        command_topic: str | None = None,
        command_type: str = "joint_trajectory",
        action_low: NumericList | None = None,
        action_high: NumericList | None = None,
        gripper_topic: str | None = None,
        gripper_command_type: str | None = None,
        gripper_joint: str | None = None,
        gripper_low: float | None = None,
        gripper_high: float | None = None,
        gripper_closed_at: str = "low",
        eef_pose_topic: str | None = None,
        cameras: CameraMap | str | None = None,
        control_hz: float = 10.0,
        fresh_obs_timeout_s: float | None = None,
        camera_throttle_ms: int | float | None = None,
        reset_service: str | None = None,
        operator_reset_confirm: bool = False,
        obs_timeout_s: float = 5.0,
        staleness_s: float = 2.0,
        simulated: bool = False,
        name: str = "ros",
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if ros_version not in (1, 2):
            raise ValueError(f"ros_version must be 1 or 2, got {ros_version!r}")
        if command_type not in ("joint_trajectory", "float64_multi_array"):
            raise ValueError(
                "command_type must be 'joint_trajectory' or 'float64_multi_array', "
                f"got {command_type!r}"
            )
        if gripper_closed_at not in ("low", "high"):
            raise ValueError(
                f"gripper_closed_at must be 'low' or 'high', got {gripper_closed_at!r}"
            )
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be positive and finite, got {control_hz!r}")
        if command_topic is None or not command_topic:
            raise ValueError(
                "command_topic is required; pass the arm controller command topic with "
                "-E command_topic=/controller/command"
            )

        parsed_joints = _parse_joints(joints)
        parsed_low = _parse_numeric_list(action_low, "action_low")
        parsed_high = _parse_numeric_list(action_high, "action_high")
        for arg, bounds in (("action_low", parsed_low), ("action_high", parsed_high)):
            if len(bounds) != len(parsed_joints):
                raise ValueError(
                    f"{arg} has {len(bounds)} bounds for {len(parsed_joints)} joints; "
                    "provide exactly one bound per configured joint"
                )
            if not all(math.isfinite(bound) for bound in bounds):
                raise ValueError(f"{arg} bounds must all be finite, got {bounds!r}")
        if any(low > high for low, high in zip(parsed_low, parsed_high, strict=True)):
            raise ValueError("action_low must be elementwise <= action_high")

        gripper_values = (gripper_joint, gripper_low, gripper_high)
        if gripper_topic is not None:
            if any(value is None for value in gripper_values):
                raise ValueError(
                    "gripper_joint, gripper_low, and gripper_high are required when "
                    "gripper_topic is configured"
                )
            assert gripper_joint is not None
            assert gripper_low is not None
            assert gripper_high is not None
            if "gripper" in parsed_joints:
                raise ValueError(
                    "an arm joint named 'gripper' conflicts with the appended gripper action "
                    "label; rename or omit that arm joint"
                )
            if not math.isfinite(gripper_low) or not math.isfinite(gripper_high):
                raise ValueError("gripper_low and gripper_high must be finite")
            if gripper_low >= gripper_high:
                raise ValueError(
                    f"gripper_low must be less than gripper_high for 0..1 normalization; "
                    f"got {gripper_low!r} >= {gripper_high!r}"
                )
        elif any(value is not None for value in gripper_values):
            raise ValueError(
                "gripper_joint, gripper_low, and gripper_high may be set only when "
                "gripper_topic is configured"
            )

        default_gripper_type = "float64" if ros_version == 1 else "float64_multi_array"
        resolved_gripper_type = gripper_command_type or default_gripper_type
        if resolved_gripper_type not in ("float64", "float64_multi_array"):
            raise ValueError(
                "gripper_command_type must be 'float64' or 'float64_multi_array', "
                f"got {resolved_gripper_type!r}"
            )

        parsed_cameras = _parse_cameras(cameras)
        action_dim = len(parsed_joints) + (1 if gripper_topic is not None else 0)
        if action_dim == 7 and eef_pose_topic is not None:
            raise ValueError(
                "omit eef_pose_topic on a 7-dim action space until core supports "
                "key-priority reference matching"
            )

        resolved_fresh_timeout = (
            2.0 / control_hz if fresh_obs_timeout_s is None else float(fresh_obs_timeout_s)
        )
        if camera_throttle_ms is None:
            resolved_camera_throttle = max(1, round(1000.0 / control_hz))
        else:
            if camera_throttle_ms != int(camera_throttle_ms):
                raise ValueError(
                    f"camera_throttle_ms must be integer milliseconds, got {camera_throttle_ms!r}"
                )
            resolved_camera_throttle = int(camera_throttle_ms)
        if resolved_fresh_timeout <= 0 or not math.isfinite(resolved_fresh_timeout):
            raise ValueError(
                f"fresh_obs_timeout_s must be positive and finite, got {resolved_fresh_timeout!r}"
            )
        if resolved_camera_throttle < 0:
            raise ValueError(f"camera_throttle_ms must be >= 0, got {resolved_camera_throttle!r}")
        for arg, timeout in (
            ("obs_timeout_s", obs_timeout_s),
            ("connect_timeout_s", connect_timeout_s),
            ("request_timeout_s", request_timeout_s),
        ):
            if timeout <= 0 or not math.isfinite(timeout):
                raise ValueError(f"{arg} must be positive and finite, got {timeout!r}")
        if staleness_s < 0 or not math.isfinite(staleness_s):
            raise ValueError(f"staleness_s must be finite and >= 0, got {staleness_s!r}")

        low_array = np.asarray(parsed_low, dtype=np.float64)
        high_array = np.asarray(parsed_high, dtype=np.float64)
        labels = parsed_joints
        state_fields = [
            StateField(
                key="joint_pos",
                shape=(action_dim,),
                unit=CANONICAL_STATE_UNITS["joint_pos"],
            )
        ]
        if gripper_topic is not None:
            assert gripper_low is not None and gripper_high is not None
            low_array = np.concatenate((low_array, np.asarray([gripper_low])))
            high_array = np.concatenate((high_array, np.asarray([gripper_high])))
            labels = (*parsed_joints, "gripper")
            state_fields.append(StateField("gripper", (1,), CANONICAL_STATE_UNITS["gripper"]))
        if eef_pose_topic is not None:
            state_fields.append(StateField("eef_pose", (7,), CANONICAL_STATE_UNITS["eef_pose"]))

        capabilities = {SELF_PACED}
        if reset_service is not None:
            capabilities.add(RESETTABLE)
        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(
                shape=(action_dim,),
                low=low_array,
                high=high_array,
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous" if gripper_topic is not None else "none",
                    frame="base",
                    dim_labels=labels,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(camera_name, config.height, config.width)
                    for camera_name, config in parsed_cameras.items()
                ),
                state=StateSpec(fields=tuple(state_fields)),
            ),
            control_hz=control_hz,
            is_simulated=simulated,
            capabilities=frozenset(capabilities),
            supported_setups=frozenset(),
            supported_target_kinds=frozenset(),
        )

        self.url = url
        self.ros_version = ros_version
        self.joints = parsed_joints
        self.joint_states_topic = joint_states_topic
        self.command_topic = command_topic
        self.command_type = command_type
        self.gripper_topic = gripper_topic
        self.gripper_command_type = resolved_gripper_type
        self.gripper_joint = gripper_joint
        self.gripper_low = gripper_low
        self.gripper_high = gripper_high
        self.gripper_closed_at = gripper_closed_at
        self.eef_pose_topic = eef_pose_topic
        self.cameras = parsed_cameras
        self.control_hz = control_hz
        self.fresh_obs_timeout_s = resolved_fresh_timeout
        self.camera_throttle_ms = resolved_camera_throttle
        self.reset_service = reset_service
        self.operator_reset_confirm = operator_reset_confirm
        self.obs_timeout_s = obs_timeout_s
        self.staleness_s = staleness_s
        self._clock = clock
        self._sleep = sleep
        self._client = RosbridgeClient(
            url,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
            clock=clock,
            sleep=sleep,
        )
        self._initialized = False
        self._instruction: str | None = None
        self._last_publish_time: float | None = None
        self._reset_count = 0
        self._warned_no_physical_reset = False
        self._validated_cameras: set[str] = set()

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, perform the configured reset path, and return fresh state."""
        del seed
        self._instruction = scene.instruction
        self._ensure_initialized()

        if self.reset_service is not None:
            self._client.call_service(self.reset_service)
        if self.operator_reset_confirm:
            print(f"Operator reset required for instruction: {scene.instruction}")
            input("Arrange the scene, then press Enter to continue: ")
        if (
            self._reset_count >= 1
            and self.reset_service is None
            and not self.operator_reset_confirm
            and not self._warned_no_physical_reset
        ):
            print(
                "warning: ROS embodiment has no reset_service or operator_reset_confirm; "
                "between-trial reset does not change the physical world",
                file=sys.stderr,
            )
            self._warned_no_physical_reset = True

        sequences = self._capture_sequences(self._all_topics())
        self._wait_for_sequences(sequences, self.obs_timeout_s, "obs_timeout_s")
        observation = self._assemble_observation()
        self._last_publish_time = None
        self._reset_count += 1
        return observation

    def step(self, action: Action) -> StepResult:
        """Sleep-gate the arm publish, require fresher joint state, and assemble a step."""
        data = np.asarray(action.data, dtype=np.float64)
        if data.shape != self.info.action_space.shape:
            raise ValueError(
                f"action has shape {data.shape}, expected {self.info.action_space.shape}"
            )
        now = self._clock()
        if self._last_publish_time is not None:
            remaining = self._last_publish_time + (1.0 / self.control_hz) - now
            if remaining > 0:
                self._sleep(remaining)

        arm = data[: len(self.joints)]
        seq_at_publish = self._client.sequence(self.joint_states_topic)
        publish_time = self._clock()
        if self.command_type == "joint_trajectory":
            arm_message = build_joint_trajectory(
                self.joints,
                arm,
                period_s=1.0 / self.control_hz,
                ros_version=self.ros_version,
            )
        else:
            arm_message = build_float64_multi_array(arm)
        self._client.publish(self.command_topic, arm_message)
        self._last_publish_time = publish_time

        if self.gripper_topic is not None:
            self._client.publish(
                self.gripper_topic,
                build_gripper_command(data[-1], self.gripper_command_type),
            )
        try:
            self._client.wait_for_sample(
                self.joint_states_topic,
                after_seq=seq_at_publish,
                timeout_s=self.fresh_obs_timeout_s,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"no post-publish joint state within fresh_obs_timeout_s="
                f"{self.fresh_obs_timeout_s:g}s at control_hz={self.control_hz:g}; "
                "lower control_hz or raise fresh_obs_timeout_s"
            ) from exc
        return StepResult(
            observation=self._assemble_observation(),
            reward=None,
            terminated=False,
            truncated=False,
        )

    def close(self) -> None:
        """Release rosbridge subscriptions, advertisements, socket, and receiver thread."""
        self._client.close()

    def __enter__(self) -> RosEmbodiment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            self._client.connect()
        except Exception as exc:
            raise ConnectionError(
                f"could not connect to rosbridge at {self.url}. Start it on the robot with "
                "ROS 1: roslaunch rosbridge_server rosbridge_websocket.launch; "
                "ROS 2: ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
            ) from exc

        self._client.advertise(
            self.command_topic,
            message_type=message_type(
                "joint_trajectory"
                if self.command_type == "joint_trajectory"
                else "float64_multi_array",
                self.ros_version,
            ),
        )
        if self.gripper_topic is not None:
            self._client.advertise(
                self.gripper_topic,
                message_type=message_type(
                    cast(MessageKind, self.gripper_command_type), self.ros_version
                ),
            )
        if self.eef_pose_topic is not None:
            self._client.subscribe(
                self.eef_pose_topic,
                subscription_id=_EEF_SUBSCRIPTION_ID,
                message_type=message_type("pose_stamped", self.ros_version),
                throttle_rate=self._state_throttle_ms,
                queue_length=1,
            )
        for camera_name, camera in self.cameras.items():
            self._client.subscribe(
                camera.topic,
                subscription_id=f"inspect-robots-camera-{camera_name}",
                message_type=message_type("compressed_image", self.ros_version),
                throttle_rate=self.camera_throttle_ms,
                queue_length=1,
            )
        self._joint_state_preflight()
        self._wait_for_sequences(
            dict.fromkeys(self._all_topics(), 0), self.obs_timeout_s, "obs_timeout_s"
        )
        self._validate_camera_resolutions()
        self._initialized = True

    @property
    def _state_throttle_ms(self) -> int:
        return max(1, round(500.0 / self.control_hz))

    def _joint_state_preflight(self) -> None:
        self._client.subscribe(
            self.joint_states_topic,
            subscription_id=_PREFLIGHT_SUBSCRIPTION_ID,
            message_type=message_type("joint_state", self.ros_version),
            throttle_rate=0,
            queue_length=1,
        )
        samples: list[TopicSample] = []
        sequence = self._client.sequence(self.joint_states_topic)
        start = self._clock()
        try:
            while len(samples) < 5:
                remaining = 1.0 - (self._clock() - start)
                if remaining <= 0:
                    break
                try:
                    sample = self._client.wait_for_sample(
                        self.joint_states_topic,
                        after_seq=sequence,
                        timeout_s=remaining,
                    )
                except TimeoutError:
                    break
                samples.append(sample)
                sequence = sample.seq
        finally:
            self._client.unsubscribe(
                self.joint_states_topic,
                subscription_id=_PREFLIGHT_SUBSCRIPTION_ID,
            )
        self._client.subscribe(
            self.joint_states_topic,
            subscription_id=_JOINT_SUBSCRIPTION_ID,
            message_type=message_type("joint_state", self.ros_version),
            throttle_rate=self._state_throttle_ms,
            queue_length=1,
        )

        if len(samples) == 1:
            warnings.warn(
                "joint_states preflight received exactly one message in 1s, too few to "
                "measure native rate; lower control_hz or raise fresh_obs_timeout_s",
                RuntimeWarning,
                stacklevel=2,
            )
        elif len(samples) >= 2:
            # Count messages by sequence delta, not by collected samples: the
            # client's poll loop coalesces fast publishers into its latest-value
            # slot, so sample count alone caps the measurable rate at the poll
            # frequency and would falsely warn on healthy high-rate rigs.
            elapsed = samples[-1].stamp - samples[0].stamp
            message_count = samples[-1].seq - samples[0].seq
            native_hz = math.inf if elapsed <= 0 else message_count / elapsed
            if native_hz < 2.0 * self.control_hz:
                warnings.warn(
                    f"joint_states native rate is about {native_hz:g} Hz, below 2x "
                    f"control_hz={self.control_hz:g}; lower control_hz or raise "
                    "fresh_obs_timeout_s",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _all_topics(self) -> tuple[str, ...]:
        topics = [self.joint_states_topic]
        if self.eef_pose_topic is not None:
            topics.append(self.eef_pose_topic)
        topics.extend(camera.topic for camera in self.cameras.values())
        return tuple(dict.fromkeys(topics))

    def _capture_sequences(self, topics: Sequence[str]) -> dict[str, int]:
        return {topic: self._client.sequence(topic) for topic in topics}

    def _wait_for_sequences(
        self, sequences: Mapping[str, int], timeout_s: float, timeout_name: str
    ) -> None:
        deadline = self._clock() + timeout_s
        for topic, sequence in sequences.items():
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(f"missing topic {topic!r} within {timeout_name}={timeout_s:g}s")
            try:
                self._client.wait_for_sample(topic, after_seq=sequence, timeout_s=remaining)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"missing topic {topic!r} within {timeout_name}={timeout_s:g}s"
                ) from exc

    def _validate_camera_resolutions(self) -> None:
        for camera_name, camera in self.cameras.items():
            if camera_name in self._validated_cameras:
                continue
            sample = self._required_sample(camera.topic)
            image = parse_compressed_image(sample.msg)
            actual = image.shape[:2]
            declared = (camera.height, camera.width)
            if actual != declared:
                raise ValueError(
                    f"camera {camera_name!r} declared resolution "
                    f"{camera.width}x{camera.height} but first frame is "
                    f"{actual[1]}x{actual[0]}"
                )
            self._validated_cameras.add(camera_name)

    def _assemble_observation(self) -> Observation:
        now = self._clock()
        joint_sample = self._required_sample(self.joint_states_topic)
        self._check_staleness(self.joint_states_topic, joint_sample, now)
        requested_joints = self.joints
        if self.gripper_joint is not None:
            requested_joints = (*requested_joints, self.gripper_joint)
        joint_position = parse_joint_state(joint_sample.msg, requested_joints)
        state: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {"joint_pos": joint_position}
        state_stamps = [joint_sample.stamp]
        if self.gripper_topic is not None:
            assert self.gripper_low is not None and self.gripper_high is not None
            normalized = (joint_position[-1] - self.gripper_low) / (
                self.gripper_high - self.gripper_low
            )
            if self.gripper_closed_at == "high":
                normalized = 1.0 - normalized
            state["gripper"] = np.asarray([normalized], dtype=np.float64)
        if self.eef_pose_topic is not None:
            eef_sample = self._required_sample(self.eef_pose_topic)
            self._check_staleness(self.eef_pose_topic, eef_sample, now)
            state["eef_pose"] = parse_pose_stamped(eef_sample.msg)
            state_stamps.append(eef_sample.stamp)

        images: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        image_times: dict[str, float] = {}
        for camera_name, camera in self.cameras.items():
            sample = self._required_sample(camera.topic)
            self._check_staleness(camera.topic, sample, now)
            images[camera_name] = parse_compressed_image(sample.msg)
            image_times[camera_name] = sample.stamp
        return Observation(
            images=images,
            state=state,
            instruction=self._instruction,
            image_times=image_times,
            state_time=min(state_stamps),
        )

    def _required_sample(self, topic: str) -> TopicSample:
        sample = self._client.latest(topic)
        if sample is None:
            raise RuntimeError(f"no cached message for subscribed topic {topic!r}")
        return sample

    def _check_staleness(self, topic: str, sample: TopicSample, now: float) -> None:
        age = now - sample.stamp
        if age > self.staleness_s:
            raise TimeoutError(
                f"cached message on {topic!r} is stale by {age:g}s, exceeding "
                f"staleness_s={self.staleness_s:g}"
            )


def ros_embodiment(**kwargs: Any) -> RosEmbodiment:
    """Construct the registry-facing ROS embodiment from CLI or programmatic arguments."""
    return RosEmbodiment(**kwargs)
