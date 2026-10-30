"""The ``dream`` policy: Dream-1 driving a bimanual YAM.

Dream's client builds its frame window (nine frames per camera, 0.2 s apart) from every tick it is
handed, so the policy sees every control step: ``act()`` hands Dream that step's frames and joint
state, returns one action, and asks for a new chunk only when the last one is used up. As in
Dream's own simulated-YAM example, the arms hold still until the window is full, the rollout waits
while Dream answers, and timestamps follow the control clock rather than wall time.
"""

from __future__ import annotations

import atexit
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots_dream import _station
from inspect_robots_dream._station import Vector
from inspect_robots_dream.bridge import BridgeSession
from inspect_robots_dream.session import DEFAULT_MODEL, DreamSession, Episode, Session

_JOINT_INDICES = [i for i in range(_station.DIM) if i not in _station.GRIPPERS]


@dataclass
class _Trial:
    instruction: str = ""
    step: int = 0
    queue: deque[Vector] = field(default_factory=deque)
    hold: Vector | None = None
    last: Vector | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    tracking: list[float] = field(default_factory=list)


class DreamPolicy:
    """Dream-1 on a bimanual YAM, one action per control step.

    The Dream client runs in this process unless ``server_url`` names an
    ``inspect-robots-dream-bridge``, as it must under Isaac Sim. ``rows_per_request`` rows of each
    48-row chunk play before Dream is asked again. ``dry_run`` asks Dream as usual but holds the
    arms still, to inspect its replies before they move a robot.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        server_url: str | None = None,
        rows_per_request: int = _station.MAX_ROWS,
        preroll_steps: int | None = None,
        dry_run: bool = False,
        record: bool = False,
        timeout_s: float = 300.0,
        session: Session | None = None,
    ) -> None:
        if not _is_count(rows_per_request) or not 1 <= rows_per_request <= _station.MAX_ROWS:
            raise ValueError(
                f"rows_per_request must be 1..{_station.MAX_ROWS}, got {rows_per_request!r}"
            )
        if preroll_steps is not None and not _is_count(preroll_steps):
            raise ValueError(f"preroll_steps must be a whole number, got {preroll_steps!r}")
        if not isinstance(dry_run, bool):
            raise ValueError(f"dry_run must be true or false, got {dry_run!r}")
        self.server_url = server_url
        self.remedy = "run inspect-robots-dream-bridge in the Dream venv" if server_url else None
        self.rows_per_request = rows_per_request
        self.dry_run = dry_run
        self._episode = Episode(model=model, record=record, timeout_s=timeout_s)
        self._preroll_steps = preroll_steps
        self._hz = _station.STATION_HZ
        self._bounds: tuple[Vector, Vector] | None = None
        self._session = session
        self._owns_session = False
        self._trial = _Trial()
        self.info = PolicyInfo(
            name="dream",
            action_space=Box(
                shape=(_station.DIM,),
                semantics=ActionSemantics(
                    control_mode="joint_pos", gripper="continuous", dim_labels=_station.DIM_LABELS
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(name, _station.FRAME_HEIGHT, _station.FRAME_WIDTH)
                    for name in _station.CAMERAS
                ),
                state_keys=frozenset({_station.STATE_KEY}),
            ),
            control_hz=_station.STATION_HZ,
            checkpoint=model,
        )
        self.config = PolicyConfig()

    @property
    def preroll_steps(self) -> int:
        """Steps that hold the arms while Dream's frame window fills (32 at 20 Hz)."""
        if self._preroll_steps is not None:
            return self._preroll_steps
        return math.ceil((_station.OBS_WINDOW - 1) * self._hz / _station.OBS_FPS)

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Accept only the bimanual YAM layout, adopt its rate, and check Dream can serve it."""
        box, name = embodiment_info.action_space, embodiment_info.name
        semantics = box.semantics
        if (
            box.shape != (_station.DIM,)
            or semantics is None
            or semantics.control_mode != "joint_pos"
            or semantics.dim_labels not in (None, _station.DIM_LABELS)
        ):
            raise ValueError(
                f"dream drives a bimanual YAM, {_station.DIM} absolute joint targets "
                f"{', '.join(_station.DIM_LABELS)}; {name!r} declares {box.shape} {semantics}"
            )
        self._bounds = None
        if box.low is not None and box.high is not None:
            self._bounds = (np.asarray(box.low, np.float64), np.asarray(box.high, np.float64))
        self._hz = embodiment_info.control_hz or _station.STATION_HZ
        if self._hz != _station.STATION_HZ:
            warnings.warn(
                f"{name!r} runs at {self._hz:g} Hz; Dream's YAM station plays Dream-1's "
                f"{_station.TRAINED_HZ:g} Hz rows at {_station.STATION_HZ:g} Hz "
                "(yam_sim: -E control_hz=20)",
                stacklevel=2,
            )
        aspect = _station.FRAME_WIDTH / _station.FRAME_HEIGHT
        stretched = [
            f"{camera.name} ({camera.width}x{camera.height})"
            for camera in embodiment_info.observation_space.cameras
            if camera.name in _station.CAMERAS and abs(camera.width / camera.height - aspect) > 0.05
        ]
        if stretched:
            warnings.warn(
                f"{name!r} cameras {', '.join(stretched)} will be stretched to Dream's "
                f"{_station.FRAME_WIDTH}x{_station.FRAME_HEIGHT} frames "
                "(yam_sim: -E cam_width=320 -E cam_height=176)",
                stacklevel=2,
            )
        self._open_session().check(self._episode)

    def reset(self, scene: Scene) -> None:
        """Start a Dream episode with the scene's instruction as its prompt."""
        if not scene.instruction:
            raise ValueError("dream needs a scene instruction: Dream-1 is language-conditioned")
        self._open_session().start(self._episode)
        self._trial = _Trial(instruction=scene.instruction)

    def act(self, observation: Observation) -> ActionChunk:
        """Hand Dream this step; return the next row of its chunk, or the held pose meanwhile."""
        trial = self._trial
        if self._session is None or not trial.instruction:
            raise RuntimeError("call reset(scene) before act()")
        measured = _measured(observation)
        if trial.hold is None:
            trial.hold = measured
        if trial.last is not None:
            trial.tracking.append(float(np.abs(measured - trial.last)[_JOINT_INDICES].max()))
        prompt = observation.instruction or trial.instruction
        self._session.observe(_frames(observation), measured, prompt, trial.step / self._hz)
        step = trial.step
        trial.step += 1
        if step < self.preroll_steps:
            return self._chunk(trial.hold)

        latency = None
        if not trial.queue:
            chunk = self._session.act(trial.last)
            rows = chunk.rows[: self.rows_per_request]
            trial.queue.extend(rows)
            latency = chunk.rtt_s
            trial.requests.append(
                {
                    "step": step,
                    "rtt_s": chunk.rtt_s,
                    "infer_ms": chunk.infer_ms,
                    "attempts": chunk.attempts,
                    "rows": len(chunk.rows),
                    "rows_beyond_limits": self._rows_beyond_limits(rows),
                    "session_id": chunk.session_id,
                }
            )
        row = trial.queue.popleft()
        if self.dry_run:
            return self._chunk(trial.hold, latency)
        trial.last = row
        return self._chunk(row, latency)

    def transcript(self) -> dict[str, Any]:
        """This trial's Dream requests, and how closely the arms followed the rows sent."""
        tracking = self._trial.tracking
        return {
            "model": self._episode.model,
            "server_url": self.server_url,
            "control_hz": self._hz,
            "rows_per_request": self.rows_per_request,
            "preroll_steps": self.preroll_steps,
            "dry_run": self.dry_run,
            "requests": [dict(request) for request in self._trial.requests],
            "tracking_error_rad": (
                {"mean": float(np.mean(tracking)), "max": max(tracking)} if tracking else None
            ),
        }

    def close(self) -> None:
        """Release the Dream session; idempotent. A later reset opens a new one."""
        atexit.unregister(self.close)
        if self._session is not None:
            self._session.close()
        if self._owns_session:
            self._session = None
            self._owns_session = False

    def __enter__(self) -> DreamPolicy:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _open_session(self) -> Session:
        if self._session is None:
            self._session = BridgeSession(self.server_url) if self.server_url else DreamSession()
            self._owns_session = True
            atexit.register(self.close)
        return self._session

    def _chunk(self, row: Vector, latency_s: float | None = None) -> ActionChunk:
        return ActionChunk(
            actions=[Action(data=row.copy())], control_hz=self._hz, inference_latency_s=latency_s
        )

    def _rows_beyond_limits(self, rows: Vector) -> int:
        if self._bounds is None:
            return 0
        low, high = self._bounds
        return int(((rows < low) | (rows > high)).any(axis=1).sum())


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _measured(observation: Observation) -> Vector:
    if _station.STATE_KEY not in observation.state:
        raise ValueError(f"observation has no {_station.STATE_KEY!r} state")
    return _station.vector(observation.state[_station.STATE_KEY])


def _frames(observation: Observation) -> dict[str, str]:
    missing = sorted(set(_station.CAMERAS) - set(observation.images))
    if missing:
        raise ValueError(f"observation has no image from {missing}")
    return {camera: _station.jpeg(observation.images[camera]) for camera in _station.CAMERAS}
