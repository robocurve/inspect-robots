"""The rollout engine — the closed control loop at the heart of Inspect Robots.

One [`rollout`][inspect_robots.rollout.rollout] runs a single trial (one scene, one epoch): it
drives the policy↔embodiment loop through the
[`ActionManager`][inspect_robots.action_manager.ActionManager] (open-loop chunk execution)
and the [`Guard`][inspect_robots.guard.Guard] safety gate, logging each step to the sinks,
and returns an immutable
[`Record`][inspect_robots.rollout.Record] that scorers consume.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from inspect_robots.action_manager import _APPROVALS_KEY, _INFER_KEY, _REPLAN_KEY, ActionManager
from inspect_robots.embodiment import Embodiment
from inspect_robots.errors import (
    EmbodimentFault,
    InspectRobotsError,
    PolicyError,
    SafetyAbort,
    _CancelledTrial,
)
from inspect_robots.frames import FrameRef, FrameStore
from inspect_robots.guard import _GUARD_TRACE_KEY, Guard, _values, guard_name
from inspect_robots.policy import Policy
from inspect_robots.scene import Scene
from inspect_robots.transcript import (
    Event,
    approval_event,
    error_event,
    inference_event,
    operator_event,
    operator_message_event,
    replan_event,
    reset_event,
    step_event,
    terminal_event,
)
from inspect_robots.types import (
    OPERATOR_END,
    Action,
    ActionChunk,
    Observation,
    StepResult,
    Termination,
)

if TYPE_CHECKING:
    from inspect_robots.console import OperatorInput
    from inspect_robots.logging.sink import LogSink

_TRANSCRIPT_BYTE_LIMIT = 2 * 1024 * 1024
_OPERATOR_MSGS_KEY = "_rollout_operator_messages"


def derive_seed(eval_seed: int | None, scene_seed: int | None, epoch: int) -> int:
    """Deterministically combine eval/scene seeds and the epoch index (R2).

    Distinct epochs of the same scene get distinct seeds so repeats actually vary
    for stochastic policies, while a fixed ``(eval_seed, scene_seed, epoch)``
    reproduces bitwise. ``None`` and ``0`` hash differently, so an unseeded
    input does not silently alias ``seed=0``.
    """
    payload = f"{eval_seed}:{scene_seed}:{epoch}".encode()
    return zlib.crc32(payload) & 0xFFFFFFFF


@dataclass(frozen=True, eq=False)
class StepRecord:
    """One step of a recorded trajectory.

    When a [`FrameStore`][inspect_robots.frames.FrameStore] is used, both
    ``observation`` and ``result.observation`` have their images stripped.
    ``image_refs`` and ``result_image_refs`` hold the corresponding on-disk
    handles instead (R5).
    """

    t: int
    observation: Observation
    action: Action
    result: StepResult
    image_refs: Mapping[str, FrameRef] | None = None
    result_image_refs: Mapping[str, FrameRef] | None = None


@dataclass(frozen=True, eq=False)
class ChunkRecord:
    """One action chunk exactly as the policy produced it, before any action manager or guard.

    ``t`` is the control step at which the policy was called. ``actions`` holds
    one flattened array per action in the chunk, unvalidated, so a malformed
    action that failed the trial is recorded as produced (empty for a terminal
    chunk with no actions). ``latency_s`` is the policy's own reported inference
    latency, if any; ``wall_s`` is the time ``act()`` took as measured by the
    rollout, which includes any network round trip.
    """

    t: int
    actions: tuple[np.ndarray, ...]
    terminal: Termination | None
    latency_s: float | None
    wall_s: float


class _ChunkTap:
    """Stand in for the policy inside an action manager, recording every chunk it returns.

    Recording at the policy boundary, not in the action manager, keeps the
    record complete for any action manager, including third-party ones that
    call ``act()`` on their own schedule. Every other attribute is the policy's.
    """

    def __init__(self, policy: Policy, chunks: list[ChunkRecord]) -> None:
        self._policy = policy
        self._chunks = chunks
        self.t = -1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def act(self, observation: Observation) -> ActionChunk:
        started = time.perf_counter()
        chunk = self._policy.act(observation)
        wall_s = time.perf_counter() - started
        self._chunks.append(
            ChunkRecord(
                t=self.t,
                # Copied: a policy may reuse its output buffer across calls, and
                # downstream stages may mutate it; the record keeps what was produced.
                actions=tuple(np.array(action.data, copy=True).ravel() for action in chunk.actions),
                terminal=chunk.terminal,
                latency_s=chunk.inference_latency_s,
                wall_s=wall_s,
            )
        )
        return chunk


@dataclass
class Record:
    """The full record of one trial — the unit scorers consume.

    ``record_id`` identifies the trial across logs, annotations and analyses;
    it is unique per trial, including across runs. ``chunks`` holds every
    action chunk the policy produced, in order; ``steps`` holds the actions
    actually executed after the action manager and guard.
    """

    scene_id: str
    epoch: int
    seed: int | None
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    steps: list[StepRecord] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    # In-memory only: EvalLog embeds SceneResult aggregates and never Record,
    # so no asdict path reaches the numpy images and the log schema is untouched.
    parked_observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    # The policy's stated reason when a terminal action ended the trial.
    termination_detail: str | None = None
    status: str = "success"  # "success" (ran to completion) | "error" | "cancelled"
    error: str | None = None
    inference_latencies: list[float] = field(default_factory=list)
    # Human operator's success verdict, captured once during rollout (R6). Read
    # by OperatorScorer; remains None for unattended/CI runs.
    operator_judgement: str | None = None
    # Qualitative context from the operator; read by nothing that scores.
    operator_note: str | None = None
    # Typed transcript of what happened during the trial.
    events: list[Event] = field(default_factory=list)
    # Extensible metadata for the trial (e.g. populated by policies).
    metadata: dict[str, Any] = field(default_factory=dict)
    # The policy's optional per-trial audit record (e.g. an LLM conversation),
    # collected via the duck-typed transcript() hook and normalized to plain
    # JSON types by _collect_transcript. None when the policy has no hook.
    policy_transcript: Any = None


def _declared_terminal(
    policy: Policy, terminal: Termination, record: Record, t: int
) -> Termination:
    """Return ``terminal`` if the policy declared its kind; otherwise fail the trial."""
    raw = getattr(policy.info, "terminal_actions", None)
    # A malformed declaration (None, a non-iterable) declares nothing, so the
    # trial fails as a PolicyError below instead of escaping the rollout.
    try:
        declared = frozenset(raw) if raw is not None and not isinstance(raw, str) else frozenset()
    except TypeError:
        declared = frozenset()
    if terminal.kind not in declared:
        raise _record_failure(
            record,
            PolicyError(
                f"policy {policy.info.name!r} ended the trial with terminal action "
                f"{terminal.kind!r}, which it does not declare; "
                f"declared: {sorted(map(str, declared))}.\n"
                "fix: list it in PolicyInfo.terminal_actions; a policy that wraps another "
                "must forward the inner policy's terminal_actions"
            ),
            t,
        )
    return terminal


def _legacy_stop(action: Action) -> Termination | None:
    """Read the pre-Termination stop request from ``action.meta``, if any."""
    if not action.meta.get("request_stop"):
        return None
    detail = action.meta.get("stop_detail")
    # Termination needs a non-blank kind; the legacy channel never validated
    # stop_reason, so anything blank or non-string falls back to the default.
    reason = action.meta.get("stop_reason")
    return Termination(
        kind=reason.strip() if isinstance(reason, str) and reason.strip() else "policy_stop",
        reason=detail if isinstance(detail, str) else "",
    )


def _end_by_policy(record: Record, terminal: Termination, t: int) -> None:
    """Record that the policy's terminal action ended the trial at step ``t``."""
    record.truncated = True
    record.termination_reason = terminal.kind
    record.termination_detail = terminal.reason or None
    record.events.append(terminal_event(t, terminal.kind, terminal.reason))


def _collect_transcript(policy: object) -> Any:
    """Normalize a policy's optional audit hook without affecting trial outcome."""
    try:
        transcript = getattr(policy, "transcript", None)
        if not callable(transcript):
            return None
        raw = transcript()
        if raw is None:
            return None
        dumped = json.dumps(raw, default=str)
        normalized = json.loads(dumped)
        size = len(dumped.encode())
        if size > _TRANSCRIPT_BYTE_LIMIT:
            return {
                "transcript_dropped": True,
                "bytes": size,
                "note": "exceeds inline limit; policies must not embed binary data",
            }
        return normalized
    except Exception as exc:
        try:
            detail = f"{type(exc).__name__}: {exc}"
        except Exception:
            detail = type(exc).__name__
        return {"transcript_error": detail}


def _record_failure(record: Record, exc: InspectRobotsError, t: int) -> InspectRobotsError:
    """Mark ``record`` failed and attach it to ``exc`` (see ``InspectRobotsError.record``).

    The partial record — steps walked and transcript events up to the failure —
    is forensic data the orchestrator preserves in the eval log.
    """
    message = str(exc)
    record.events.append(error_event(t, type(exc).__name__, message))
    record.status = "error"
    record.error = f"{type(exc).__name__}: {message}"
    exc.record = record
    return exc


def _connection_failure(exc: Exception) -> bool:
    """Return whether an exception chain contains a connection failure."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionError) or type(current).__name__ in {
            "ConnectionError",
            "NewConnectionError",
            "ConnectError",
        }:
            return True
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return False


def _policy_error(policy: Policy, exc: Exception) -> PolicyError:
    """Wrap a generic policy exception, appending a hint on connection failures."""
    message = str(exc)
    if _connection_failure(exc):
        try:
            url = getattr(policy, "server_url", None)
            remedy = getattr(policy, "remedy", None)
            name = policy.info.name
            if url:
                # "up and healthy", not "running": builtin ConnectionError
                # matches also cover reset/broken-pipe, where the server
                # answered and then dropped the connection.
                message += (
                    f"\nhint: policy {name!r} could not hold a connection to its "
                    f"action server at {url} — is the server up and healthy? "
                    "Start (or restart) it, then rerun."
                )
            else:
                message += (
                    f"\nhint: policy {name!r} hit a connection failure — "
                    "a backend it depends on may be down or unreachable."
                )
            if remedy:
                message += f"\nhint: {remedy}"
        except Exception:
            pass  # a raising server_url/remedy property must not mask the trial error
    return PolicyError(message)


def _non_finite_detail(data: object) -> str | None:
    """Describe why action data is not finite, or return ``None`` when it is."""
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        return f"is not numeric: {exc}"
    if np.isfinite(array).all():
        return None
    if np.isnan(array).any():
        return "contains nan"
    return "contains inf"


def _store_frames(
    frame_store: FrameStore | None, trial_id: str, t: int, obs: Observation
) -> tuple[Observation, Mapping[str, FrameRef] | None]:
    """If a frame store is configured, stream images to disk and strip them."""
    if frame_store is None or not obs.images:
        return obs, None
    refs = {cam: frame_store.put(trial_id, t, cam, image) for cam, image in obs.images.items()}
    return replace(obs, images={}), refs


def _end_operator_trial(operator_input: OperatorInput | None) -> None:
    """Best-effort call the optional operator-input trial teardown hook."""
    if operator_input is not None:
        try:
            end_trial = getattr(operator_input, "end_trial", None)
            if callable(end_trial):
                end_trial()
        except Exception as exc:
            # stacklevel=3 skips this helper so the warning points at rollout's
            # caller, like the disable-site warnings issued from rollout() itself.
            warnings.warn(
                f"Operator console disabled for this trial after {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )


def rollout(
    policy: Policy,
    embodiment: Embodiment,
    scene: Scene,
    *,
    max_steps: int,
    seed: int | None,
    epoch: int,
    action_manager: ActionManager,
    guard: Guard,
    sink: LogSink,
    frame_store: FrameStore | None = None,
    operator_input: OperatorInput | None = None,
    horizon_reason: str = "max_steps",
) -> Record:
    """Run a single trial and return its record.

    Generic exceptions raised by the policy, or a non-finite action, are
    reported as [`PolicyError`][inspect_robots.errors.PolicyError]; generic
    exceptions raised by the embodiment are reported as
    [`EmbodimentFault`][inspect_robots.errors.EmbodimentFault]; by the guard as
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort] (a guard that crashed cannot
    vouch for safety), as is a non-finite action introduced by the guard.
    Already-typed Inspect Robots errors (incl.
    [`SafetyAbort`][inspect_robots.errors.SafetyAbort]) propagate unchanged, so the
    eval orchestrator can apply the correct continue-vs-halt policy. Every error
    raised from inside the trial carries the partial ``Record`` on
    ``exc.record`` for the orchestrator to preserve.

    The loop applies no wall-clock pacing of its own: ``embodiment.step()`` is
    called as fast as the policy/action_manager/guard can produce actions. An
    embodiment that needs real-time cadence paces itself inside ``step()`` and
    declares the ``"self_paced"`` capability to document that it does (see
    [`Embodiment`][inspect_robots.embodiment.Embodiment]).

    ``horizon_reason`` is the termination reason recorded when the step budget
    runs out: ``"max_steps"``, or ``"max_seconds"`` for a task whose budget was
    declared in seconds and resolved to ``max_steps`` at the embodiment's rate.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError(f"max_steps must be an integer >= 1, got {max_steps!r}")
    trial_id = f"{scene.id}-e{epoch}"
    record = Record(scene_id=scene.id, epoch=epoch, seed=seed)
    record.events.append(reset_event(seed))
    store: dict[str, Any] = {}
    tap = _ChunkTap(policy, record.chunks)
    expected_dim = embodiment.info.action_space.dim
    policy_reset_ok = False
    delta_hook: Any = getattr(policy, "transcript_delta", None)
    messages_hook: Any = getattr(sink, "log_policy_messages", None)
    stream_ok = callable(delta_hook) and callable(messages_hook)
    console_ok = operator_input is not None

    try:
        t = -1
        try:
            policy.reset(scene)
            policy_reset_ok = True
        except InspectRobotsError as exc:
            _record_failure(record, exc, -1)
            raise
        except Exception as exc:
            raise _record_failure(record, _policy_error(policy, exc), -1) from exc
        try:
            obs = embodiment.reset(scene, seed=seed)
        except InspectRobotsError as exc:
            _record_failure(record, exc, -1)
            raise
        except Exception as exc:
            raise _record_failure(record, EmbodimentFault(str(exc)), -1) from exc

        if operator_input is not None:
            try:
                operator_input.begin_trial()
            except Exception as exc:
                console_ok = False
                _end_operator_trial(operator_input)
                warnings.warn(
                    f"Operator console disabled for this trial after {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        obs_rec, refs = _store_frames(frame_store, trial_id, 0, obs)
        t = 0
        while t < max_steps:
            poll = None
            if operator_input is not None and console_ok:
                try:
                    poll = operator_input.poll()
                except Exception as exc:
                    console_ok = False
                    _end_operator_trial(operator_input)
                    warnings.warn(
                        "Operator console disabled for this trial after "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if poll is not None:
                for i, text in enumerate(poll.messages):
                    source = poll.sources[i] if i < len(poll.sources) else "console"
                    store.setdefault(_OPERATOR_MSGS_KEY, []).append(
                        {"t": t, "text": text, "source": source}
                    )
                    record.events.append(operator_message_event(t, text, source))

            prev_inferences = len(store.get(_INFER_KEY, []))
            all_approvals = store.get(_APPROVALS_KEY, [])
            last_approvals_idx = store.get("_rollout_last_approvals_idx", 0)
            tail_approvals = [dict(a) for a in all_approvals[last_approvals_idx:]]
            all_operator_msgs = store.get(_OPERATOR_MSGS_KEY, [])
            last_operator_msgs_idx = store.get("_rollout_last_operator_msgs_idx", 0)
            tail_operator_msgs = [
                dict(message) for message in all_operator_msgs[last_operator_msgs_idx:]
            ]

            if poll is not None and poll.end is not None:
                record.terminated = True
                record.termination_reason = OPERATOR_END
                if poll.end.verdict is not None:
                    record.operator_judgement = poll.end.verdict
                    record.operator_note = poll.end.note
                    record.events.append(
                        operator_event(
                            t=t,
                            verdict=poll.end.verdict,
                            source="console",
                            note=poll.end.note,
                        )
                    )
                break

            try:
                policy_extra = {**obs.extra, "env_step": t, "approvals": tail_approvals}
                if operator_input is not None:
                    policy_extra["operator_messages"] = tail_operator_msgs
                obs_with_extra = replace(
                    obs,
                    extra=policy_extra,
                )
                tap.t = t
                action = action_manager.next_action(cast("Policy", tap), obs_with_extra, t, store)
            except InspectRobotsError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, _policy_error(policy, exc), t) from exc

            replans = store.get(_REPLAN_KEY, [])
            for replan in replans[store.get("_rollout_replans_seen", 0) :]:
                record.events.append(replan_event(replan["t"], replan["trigger"], replan["reason"]))
            store["_rollout_replans_seen"] = len(replans)

            inferences = store.get(_INFER_KEY, [])
            if len(inferences) > prev_inferences:
                store["_rollout_last_approvals_idx"] = len(all_approvals)
                if operator_input is not None:
                    store["_rollout_last_operator_msgs_idx"] = len(all_operator_msgs)
                latency, chunk_len = inferences[-1]
                record.events.append(inference_event(t, latency, chunk_len))
                if stream_ok:
                    try:
                        delta = delta_hook()
                        entries = list(delta) if delta is not None else []
                        if entries:
                            messages_hook(t, entries)
                    except Exception as exc:
                        stream_ok = False
                        warnings.warn(
                            "Live policy transcript streaming disabled for this "
                            f"trial after {type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            if isinstance(action, Termination):
                # A terminal chunk with no actions: end without stepping.
                _end_by_policy(record, _declared_terminal(policy, action, record, t), t)
                break

            # A malformed action is the policy's fault; catching it here keeps it
            # from surfacing inside the guard/embodiment as a halting fault.
            emitted_dim = int(np.asarray(action.data).size)
            if emitted_dim != expected_dim:
                raise _record_failure(
                    record,
                    PolicyError(
                        f"policy emitted a {emitted_dim}-D action but embodiment "
                        f"{embodiment.info.name!r} expects {expected_dim}-D"
                    ),
                    t,
                )
            non_finite_detail = _non_finite_detail(action.data)
            if non_finite_detail is not None:
                raise _record_failure(
                    record,
                    PolicyError(
                        "policy emitted a non-finite action "
                        f"({non_finite_detail}) for embodiment {embodiment.info.name!r}"
                    ),
                    t,
                )

            # The policy's terminal action, captured from the PRE-review action
            # so a guard rewrite cannot erase the intent. The legacy channel,
            # meta["request_stop"] (plan 0008 §3d), still ends the trial the same
            # way; it is not checked against the declared terminal actions.
            terminal = (
                _declared_terminal(policy, action.terminal, record, t)
                if action.terminal is not None
                else _legacy_stop(action)
            )

            store.pop(_GUARD_TRACE_KEY, None)
            try:
                reviewed = guard.review(action, store)  # may raise SafetyAbort
            except InspectRobotsError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, SafetyAbort(str(exc)), t) from exc
            trace = store.pop(_GUARD_TRACE_KEY, None)
            if reviewed is not action:
                flags = [k for k in ("clamped", "delta_clamped") if reviewed.meta.get(k)]
                detail = ", ".join(flags) or None
                changes = trace or [
                    {
                        "guard": guard_name(guard),
                        "before": _values(action),
                        "after": _values(reviewed),
                    }
                ]
                record.events.append(
                    approval_event(t, modified=True, detail=detail, guards=changes)
                )
                store.setdefault(_APPROVALS_KEY, []).append({"t": t, "detail": detail})
            action = reviewed

            # Recheck because a guard may mutate the array in place or return a new action.
            reviewed_dim = int(np.asarray(action.data).size)
            if reviewed_dim != expected_dim:
                raise _record_failure(
                    record,
                    SafetyAbort(
                        f"guard {type(guard).__name__} returned a {reviewed_dim}-D "
                        f"action but embodiment {embodiment.info.name!r} expects {expected_dim}-D"
                    ),
                    t,
                )
            non_finite_detail = _non_finite_detail(action.data)
            if non_finite_detail is not None:
                raise _record_failure(
                    record,
                    SafetyAbort(
                        f"guard {type(guard).__name__} returned a non-finite "
                        f"action ({non_finite_detail})"
                    ),
                    t,
                )

            try:
                result: StepResult = embodiment.step(action)
            except InspectRobotsError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, EmbodimentFault(str(exc)), t) from exc

            sink.log_step(t, obs, action, result)
            result_obs_rec, result_refs = _store_frames(
                frame_store, trial_id, t + 1, result.observation
            )
            result_rec = (
                result
                if result_obs_rec is result.observation
                else replace(result, observation=result_obs_rec)
            )
            record.steps.append(
                StepRecord(
                    t=t,
                    observation=obs_rec,
                    action=action,
                    result=result_rec,
                    image_refs=refs,
                    result_image_refs=result_refs,
                )
            )
            record.events.append(
                step_event(t, result.terminated, result.truncated, result.termination_reason)
            )
            t += 1

            if result.terminated:
                record.terminated = True
                record.termination_reason = result.termination_reason
                break
            if result.truncated:
                record.truncated = True
                record.termination_reason = result.termination_reason or "truncated"
                break
            if terminal is not None:
                # Embodiment-reported termination above wins (ground truth);
                # otherwise the policy's terminal action ends the trial as a
                # truncation: scoring stays the scorer's job, done is not success.
                # ``t`` already counts the executed step; the event names that step.
                _end_by_policy(record, terminal, t - 1)
                break
            obs = result.observation
            obs_rec = result_obs_rec
            refs = result_refs
        else:
            record.truncated = True
            record.termination_reason = horizon_reason
    except KeyboardInterrupt as exc:
        record.status = "cancelled"
        record.error = "cancelled by user (KeyboardInterrupt)"
        record.events.append(error_event(t, "KeyboardInterrupt", "cancelled by user"))
        raise _CancelledTrial(record.error, record) from exc
    finally:
        _end_operator_trial(operator_input)
        # Preserve measured latencies even when the trial ends in an error.
        record.inference_latencies = [
            lat for lat, _ in store.get(_INFER_KEY, []) if lat is not None
        ]
        if policy_reset_ok:  # pragma: no branch - false only while an exception unwinds
            record.policy_transcript = _collect_transcript(policy)
    return record
