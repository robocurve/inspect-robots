from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt

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
    StepResult,
)
from inspect_robots.embodiment import RESETTABLE
from inspect_robots_dream import _station
from inspect_robots_dream._station import Frame, Vector
from inspect_robots_dream.session import DEFAULT_MODEL, Chunk, Episode, Refused

# Every slot distinguishable and inside its limits: left joints 0.0..0.5 with grip 0.9,
# right joints 1.0..1.5 with grip 0.2.
LEFT = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
RIGHT = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.2])
STATE = np.concatenate([LEFT, RIGHT])
HOME = np.tile([0.0, 1.0, 1.0, -0.6, 0.0, 0.0, 1.0], 2)
LOW = np.tile([-2.6, 0.0, 0.0, -1.6, -1.5, -2.0, 0.0], 2)
HIGH = np.tile([3.1, 3.6, 3.1, 1.5, 1.5, 2.0, 1.0], 2)
# One frame per camera, told apart by its fill value, which `encode` reads back.
IMAGES: dict[str, Frame] = {
    name: np.full((176, 320, 3), fill, np.uint8)
    for name, fill in (("top_cam", 1), ("left_cam", 2), ("right_cam", 3))
}


def encode(image: Frame) -> str:
    """Stands in for the station's JPEG encoder, which needs Pillow."""
    return f"frame-{int(image.flat[0])}"


class DreamRefusal(Exception):
    """Shaped like ``dream.DreamBadRequest``: the status in ``code``, the detail in the message."""

    def __init__(self, detail: str, code: str = "DA-4000") -> None:
        super().__init__(f"DreamAPI refused the inference request: {code} BAD_REQUEST: {detail}")
        self.code = code


def reply(
    left: npt.ArrayLike = LEFT, right: npt.ArrayLike = RIGHT, rows: int = 48
) -> dict[str, Any]:
    """A Dream reply holding one row per arm for the whole chunk."""
    return {
        "q_l_chunk": np.tile(left, (rows, 1)),
        "q_r_chunk": np.tile(right, (rows, 1)),
        "per_key": {},
    }


class FakeClient:
    """Stands in for ``dream.DreamClient``: records each call; ``errors`` raise before replies."""

    def __init__(
        self,
        *,
        models: Sequence[str] = (DEFAULT_MODEL,),
        errors: Sequence[Exception] = (),
        answer: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.models = models
        self.errors = list(errors)
        self.answer = reply() if answer is None else answer
        self.last_timing: dict[str, Any] | None = {"serve_infer_ms": 310}
        self.session_id: str | None = "session-1"

    def get_models(self) -> Sequence[Any]:
        self.calls.append(("get_models", None))
        return [SimpleNamespace(id=model) for model in self.models]

    def policy(self, config: dict[str, Any]) -> None:
        self.calls.append(("policy", config))

    def reset_episode(self) -> None:
        self.calls.append(("reset_episode", None))

    def set_observation(
        self,
        frames: Mapping[str, str],
        observation: Mapping[str, Any],
        *,
        timestamp: float | None = None,
    ) -> None:
        self.calls.append(("set_observation", (dict(frames), dict(observation), timestamp)))

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append(("get_action", commanded))
        if self.errors:
            raise self.errors.pop(0)
        return self.answer

    def close(self) -> None:
        self.calls.append(("close", None))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args_of(self, name: str) -> list[Any]:
        return [args for call, args in self.calls if call == name]


class FakeSession:
    """A `Session` that records what the policy hands it and answers from ``chunks`` in turn.

    A ``refuse`` reason makes every request raise `Refused`, as Dream's safety screen does.
    """

    def __init__(self, *chunks: Vector, refuse: str | None = None) -> None:
        self.chunks = list(chunks) or [np.tile(STATE, (48, 1))]
        self.refuse = refuse
        self.checked: list[Episode] = []
        self.episodes: list[Episode] = []
        self.ticks: list[tuple[dict[str, Frame], Vector, str, float]] = []
        self.commanded: list[Vector | None] = []
        self.closed = 0

    def check(self, episode: Episode) -> None:
        self.checked.append(episode)

    def start(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def observe(
        self, images: Mapping[str, Frame], state: Vector, prompt: str, timestamp: float
    ) -> None:
        self.ticks.append((dict(images), state.copy(), prompt, timestamp))

    def act(self, commanded: Vector | None) -> Chunk:
        self.commanded.append(None if commanded is None else commanded.copy())
        if self.refuse is not None:
            raise Refused(self.refuse)
        rows = self.chunks.pop(0) if len(self.chunks) > 1 else self.chunks[0]
        return Chunk(rows=rows, rtt_s=0.4, attempts=1, infer_ms=300.0, session_id="s")

    def close(self) -> None:
        self.closed += 1


class Clock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def yam_info(
    *,
    control_hz: float | None = 20.0,
    camera: tuple[int, int] = (176, 320),
    action_space: Box | None = None,
) -> EmbodimentInfo:
    """A bimanual YAM as Inspect Robots' YAM embodiments declare it."""
    height, width = camera
    return EmbodimentInfo(
        name="fake_yam",
        action_space=action_space
        or Box(
            shape=(_station.DIM,),
            low=LOW,
            high=HIGH,
            semantics=ActionSemantics(
                control_mode="joint_pos", gripper="continuous", dim_labels=_station.DIM_LABELS
            ),
        ),
        observation_space=ObservationSpace(
            cameras=tuple(CameraSpec(name, height, width) for name in _station.CAMERAS),
            state_keys=frozenset({_station.STATE_KEY}),
        ),
        control_hz=control_hz,
        is_simulated=True,
        capabilities=frozenset({RESETTABLE}),
    )


def observation(
    state: npt.NDArray[np.floating[Any]] = HOME, instruction: str | None = None
) -> Observation:
    images = {name: np.full((176, 320, 3), 90, np.uint8) for name in _station.CAMERAS}
    return Observation(images=images, state={_station.STATE_KEY: state}, instruction=instruction)


class FakeYam(EmbodimentBase):
    """A bimanual YAM whose arms land exactly on every target they are sent."""

    def __init__(self) -> None:
        self.info = yam_info()
        self.targets: list[Vector] = []
        self.state = HOME.copy()

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.state = HOME.copy()
        return observation(self.state)

    def step(self, action: Action) -> StepResult:
        self.targets.append(np.asarray(action.data, dtype=np.float64).copy())
        self.state = self.targets[-1].copy()
        return StepResult(observation=observation(self.state))
