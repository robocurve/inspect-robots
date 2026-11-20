"""Dream sessions: what the policy needs from ``dream.DreamClient``, behind one interface.

The policy checks and starts episodes, hands over ticks and asks for chunks. `DreamSession` runs
the Dream client in this process; the `Session` protocol is the seam for anything else, such as
the fakes the tests use.
"""

from __future__ import annotations

import math
import re
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from inspect_robots_dream import _station
from inspect_robots_dream._station import Frame, Vector

DEFAULT_MODEL = "dreamlabs/dream-1-beta"
REQUEST_TIMEOUT_S = 45.0
# The rollout pauses while Dream answers, so no rows run during a round trip. Dream's seam join
# reads its row rate only to count such rows; a rate this low makes each chunk take over at its
# first row whatever the latency, so a trajectory does not depend on the network.
_PAUSED_ROW_HZ = 1e-3
# Dream's safety screen refuses an instruction with this status, and its client words the error
# "... DA-4000 <name> (<reason>): unsafe instruction, refused: <why>". The detail has to start
# with the phrase: text quoted later in another refusal's detail does not count.
_BAD_REQUEST = "DA-4000"
_UNSAFE = re.compile(
    rf"{_BAD_REQUEST} \S+(?: \([^)]*\))?: unsafe instruction, refused(?::\s*(?P<why>.*))?$", re.S
)


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


class Refused(Exception):
    """Dream declined the instruction as unsafe and sent no action; ``reason`` is its explanation.

    Asking again with the same instruction gets the same answer, so the trial is over.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Session(Protocol):
    """What the policy needs from Dream, one episode at a time."""

    def check(self, episode: Episode) -> None:
        """Raise now if ``episode`` cannot run: no Dream client, or a model the key may not run."""

    def start(self, episode: Episode) -> None:
        """Begin an episode: an empty frame window and a new Dream session."""

    def observe(
        self, images: Mapping[str, Frame], state: Vector, prompt: str, timestamp: float
    ) -> None:
        """Hand over one tick: an RGB frame per camera, the measured 14-D state, the instruction."""

    def act(self, commanded: Vector | None) -> Chunk:
        """Ask for the next chunk, given the last row the loop commanded (None for none).

        Raises `Refused` when Dream declines the instruction as unsafe.
        """

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

    Each camera frame is sent as ``encode`` returns it, by default the station's 320x176 JPEG. An
    episode records to the Dream account only when it asks to; the client otherwise uploads
    every tick. Refusals worth repeating (a model still starting, a replaced session, a timeout,
    an unreachable gateway) are asked again while the episode's ``timeout_s`` allows another
    attempt. An instruction Dream declines as unsafe raises `Refused`. Anything else is raised as
    is.
    """

    def __init__(
        self,
        make_client: Callable[[], _Client] | None = None,
        *,
        encode: Callable[[Frame], str] = _station.jpeg,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._make_client = make_client or _dream_client
        self._encode = encode
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
        self, images: Mapping[str, Frame], state: Vector, prompt: str, timestamp: float
    ) -> None:
        """Hand Dream one tick in the station keys."""
        client = self._open()
        frames = _station.frames(
            {
                name: self._encode(image)
                for name, image in images.items()
                if name in _station.CAMERAS
            }
        )
        client.set_observation(frames, _station.observation(state, prompt), timestamp=timestamp)

    def act(self, commanded: Vector | None) -> Chunk:
        """One chunk, asking again after retryable refusals until the episode's timeout.

        Raises `Refused` when Dream declines the instruction as unsafe.
        """
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
                declined = _declined(error)
                if declined is not None:
                    raise Refused(declined) from error
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


def _declined(error: Exception) -> str | None:
    """Dream's reason for declining the instruction as unsafe, or None for any other error.

    Dream answers such a request with status ``DA-4000`` and the detail
    ``unsafe instruction, refused: <why>``. Its client raises that as ``DreamBadRequest``, which
    carries the status as ``code`` and the detail in its message.
    """
    message = str(error)
    found = _UNSAFE.search(message) if getattr(error, "code", None) == _BAD_REQUEST else None
    if found is None:
        return None
    return (found["why"] or "").strip() or message


def _retry_after(error: Exception) -> float | None:
    """Seconds to wait before asking again, or None when asking again cannot help."""
    try:
        import dream
    except ImportError:  # a client that is not Dream's: its errors are not ours to retry
        return None

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
            "the Dream client is not installed: pip install 'inspect-robots-dream[client]', or "
            "pip install dreamlabs-client (`pip install dream` is another project). It needs "
            "websockets 13 or newer, which Isaac Sim 6.1 has and Isaac Sim 5.1 does not."
        ) from error
    warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
    client: _Client = make()
    return client
