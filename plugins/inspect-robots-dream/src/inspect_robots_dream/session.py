"""Dream sessions: what the policy needs from ``dream.DreamClient``, behind one interface.

The policy checks and starts episodes, hands over ticks and asks for chunks. Whether the Dream
client runs in this process (`DreamSession`) or in a bridge process
(`inspect_robots_dream.bridge.BridgeSession`) is the session's concern.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from inspect_robots_dream import _station
from inspect_robots_dream._station import Vector

DEFAULT_MODEL = "dreamlabs/dream-1-beta"
REQUEST_TIMEOUT_S = 45.0
# The rollout pauses while Dream answers, so no rows run during a round trip. Dream's seam join
# reads its row rate only to count such rows; a rate this low makes each chunk take over at its
# first row whatever the latency, so a trajectory does not depend on the network.
_PAUSED_ROW_HZ = 1e-3


@dataclass(frozen=True)
class Episode:
    """One trial's Dream settings: the checkpoint, recording, and the wait for one chunk."""

    model: str = DEFAULT_MODEL
    record: bool = False
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a Dream checkpoint id, got {self.model!r}")
        if not isinstance(self.record, bool):
            raise ValueError(f"record must be true or false, got {self.record!r}")
        if not _positive(self.timeout_s):
            raise ValueError(f"timeout_s must be a positive number, got {self.timeout_s!r}")


@dataclass(frozen=True)
class Chunk:
    """One Dream reply as ``[rows, 14]`` bimanual targets, and what the request cost."""

    rows: Vector
    rtt_s: float
    attempts: int
    infer_ms: float | None = None
    session_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Plain JSON types, which `from_json` reads back."""
        return {**vars(self), "rows": self.rows.tolist()}

    @classmethod
    def from_json(cls, body: Mapping[str, Any]) -> Chunk:
        """The chunk `to_json` wrote; ValueError unless its rows are finite 14-D targets."""
        rows = np.asarray(body["rows"], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != _station.DIM or not np.isfinite(rows).all():
            raise ValueError(f"chunk rows must be finite [n, {_station.DIM}], got {rows.shape}")
        return cls(
            rows=rows,
            rtt_s=float(body["rtt_s"]),
            attempts=int(body["attempts"]),
            infer_ms=body.get("infer_ms"),
            session_id=body.get("session_id"),
        )


class Session(Protocol):
    """What the policy needs from Dream, one episode at a time."""

    def check(self, episode: Episode) -> None:
        """Raise now if ``episode`` cannot run: no Dream client, or a model the key may not run."""

    def start(self, episode: Episode) -> None:
        """Begin an episode: an empty frame window and a new Dream session."""

    def observe(
        self, frames: Mapping[str, str], state: Vector, prompt: str, timestamp: float
    ) -> None:
        """Hand over one tick: a JPEG per camera, the measured 14-D state, the instruction."""

    def act(self, commanded: Vector | None) -> Chunk:
        """Ask for the next chunk, given the last row the loop commanded (None for none)."""

    def close(self) -> None:
        """Release the Dream session; idempotent."""


class _Client(Protocol):
    @property
    def last_timing(self) -> dict[str, Any] | None: ...
    @property
    def session_id(self) -> str | None: ...
    def get_models(self) -> Sequence[Any]: ...
    def policy(self, config: dict[str, Any]) -> None: ...
    def reset_episode(self) -> None: ...
    def set_observation(
        self,
        frames: Mapping[str, str],
        observation: Mapping[str, Any],
        *,
        timestamp: float | None = None,
    ) -> None: ...
    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]: ...
    def close(self) -> None: ...


class DreamSession:
    """``dream.DreamClient`` in this process: station keys out, validated chunks back.

    An episode records to the Dream account only when it asks to; the client otherwise uploads
    every tick. Refusals worth repeating (a model still starting, a replaced session, a timeout,
    an unreachable gateway) are asked again while the episode's ``timeout_s`` allows another
    attempt. Anything else is raised as is.
    """

    def __init__(
        self,
        make_client: Callable[[], _Client] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._make_client = make_client or _dream_client
        self._sleep = sleep
        self._clock = clock
        self._client: _Client | None = None
        self._config: dict[str, Any] | None = None
        self._episode = Episode()
        self._models: frozenset[str] = frozenset()

    def check(self, episode: Episode) -> None:
        """Raise now if ``episode`` cannot run: no Dream client, or a model the key may not run."""
        self._ready(episode)

    def start(self, episode: Episode) -> None:
        """Open the checkpoint when it or the recording setting changes, else reset its episode."""
        client = self._ready(episode)
        config = {
            "checkpoint": episode.model,
            "_record": episode.record,
            "action_hz": _PAUSED_ROW_HZ,
            "strict": False,
            "timeout_ms": round(REQUEST_TIMEOUT_S * 1000),
            "obs_window": _station.OBS_WINDOW,
            "obs_fps": _station.OBS_FPS,
        }
        if config == self._config:
            client.reset_episode()
        else:
            client.policy(config)
            self._config = config
        self._episode = episode

    def observe(
        self, frames: Mapping[str, str], state: Vector, prompt: str, timestamp: float
    ) -> None:
        """Hand Dream one tick in the station keys."""
        self._open().set_observation(
            _station.frames(frames), _station.observation(state, prompt), timestamp=timestamp
        )

    def act(self, commanded: Vector | None) -> Chunk:
        """One chunk, asking again after retryable refusals until the episode's timeout."""
        client = self._open()
        last = None if commanded is None else _station.commanded(commanded)
        deadline = self._clock() + self._episode.timeout_s
        attempts = 0
        while True:
            attempts += 1
            sent = self._clock()
            try:
                reply = client.get_action(commanded=last)
                break
            except Exception as error:
                pause = _retry_after(error)
                if pause is None or self._clock() + pause > deadline:
                    raise
                reason = getattr(error, "reason", None) or type(error).__name__
                print(
                    f"Dream: {reason}; asking again in {pause:.0f} s", file=sys.stderr, flush=True
                )
                self._sleep(pause)
        timing = client.last_timing or {}
        return Chunk(
            rows=_station.rows(reply),
            rtt_s=self._clock() - sent,
            attempts=attempts,
            infer_ms=timing.get("serve_infer_ms"),
            session_id=client.session_id,
        )

    def close(self) -> None:
        """Release the Dream session; idempotent."""
        if self._client is not None:
            self._client.close()
        self._client = None
        self._config = None

    def _ready(self, episode: Episode) -> _Client:
        client = self._client = self._client or self._make_client()
        if episode.model not in self._models:
            self._models = frozenset(model.id for model in client.get_models())
            if episode.model not in self._models:
                raise ValueError(
                    f"{episode.model!r} is not a model this Dream key may run; "
                    f"it may run {sorted(self._models)}"
                )
        return client

    def _open(self) -> _Client:
        if self._client is None or self._config is None:
            raise RuntimeError("start an episode before observing or acting")
        return self._client


def _retry_after(error: Exception) -> float | None:
    """Seconds to wait before asking again, or None when asking again cannot help."""
    import dream

    if isinstance(error, dream.DreamSessionExpired | dream.DreamTimeout):
        return 0.0
    if isinstance(error, dream.DreamProtocolError):
        return 1.0
    if isinstance(error, dream.DreamServerError) and error.retryable:
        return error.retry_after_s or 1.0
    return None


def _positive(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value > 0
        and math.isfinite(value)
    )


def _dream_client() -> _Client:
    """``dream.DreamClient()`` without its slow-network warning: the arms wait for each reply."""
    try:
        import dream

        make = dream.DreamClient
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "the Dream client is not installed: pip install dreamlabs-client (`pip install dream` "
            "is another project). Where it cannot be installed (Isaac Sim), run "
            "inspect-robots-dream-bridge in its own venv and pass -P server_url=..."
        ) from error
    warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
    return make()
