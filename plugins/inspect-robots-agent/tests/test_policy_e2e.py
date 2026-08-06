"""LLMAgentPolicy end-to-end through real eval() on the mock cubepick world.

Conversations are scripted with httpx.MockTransport: no network, no LLM, no
hardware — the whole loop (bind, observation payloads, tool synthesis,
guardrails, policy-stop, budgets, error taxonomy) runs for real.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import numpy.typing as npt
import pytest

import inspect_robots_agent.policy as agent_policy_module
from inspect_robots import eval as ir_eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.controller import DefaultController
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation, StepResult
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._llm import ChatClient, resolve_provider
from inspect_robots_agent._png import encode_png
from inspect_robots_agent._tools import ToolResult
from inspect_robots_agent.policy import (
    _ON_DEMAND_SYSTEM_TEMPLATE,
    _PRIOR_LEARNINGS_TEXT_LIMIT,
    _SYSTEM_TEMPLATE,
    AgentPolicyConfig,
    _image_parts,
    _observation_content,
    _operator_lines,
    _PendingCapture,
)

# --- scripted-conversation harness ---------------------------------------------

_MOVE_TOOL_NAMES = frozenset({"move_joints", "move_to", "move_by"})
_DEFAULT_MOVE_NOTE = "The target is ahead, so I chose this motion to move toward it."
_NOTE_CONTRACT = (
    "Every move tool call must include a `note`: in one or two sentences, say what you observe "
    "in the current observation and why you chose this motion. The user is watching these notes "
    "to see what you see and what you decide, so write them for a human reader."
)
_NOTE_ERROR = "note is required: describe what you observe and why you chose this motion"
_PRIOR_LEARNINGS_FRAME = (
    "\n\nNotes from a previous attempt at tasks like this one. They may "
    "be wrong or stale; the current observation always wins:\n"
)
_BLOB_RE = re.compile(r"\$blob:([0-9a-f]{64})")


def _with_default_note(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    wire_arguments = dict(arguments)
    if name in _MOVE_TOOL_NAMES:
        wire_arguments.setdefault("note", _DEFAULT_MOVE_NOTE)
    return wire_arguments


def _tool_response(
    name: str,
    arguments: dict[str, Any],
    *,
    add_default_note: bool = True,
) -> dict[str, Any]:
    wire_arguments = _with_default_note(name, arguments) if add_default_note else arguments
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(wire_arguments)},
                        }
                    ],
                }
            }
        ]
    }


def _text_response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _text_and_tool_response(text: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def _multi_tool_response(calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(_with_default_note(name, arguments)),
                            },
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                }
            }
        ]
    }


class _Script:
    """Serves queued responses; repeats the last one when the queue runs dry."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return httpx.Response(200, json=payload)


class _WireScript:
    """Serve equivalent move/done turns on each supported provider wire."""

    def __init__(self, wire: str):
        self.wire = wire
        self.requests: list[dict[str, Any]] = []
        self.turns = [
            (
                "move_joints",
                _with_default_note("move_joints", {"targets": {"joint": 0.1}}),
            ),
            ("done", {"summary": "captured"}),
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        index = min(len(self.requests) - 1, len(self.turns) - 1)
        name, arguments = self.turns[index]
        if self.wire == "chat":
            payload = _tool_response(name, arguments, add_default_note=False)
        elif self.wire == "responses":
            payload = {
                "id": f"resp_{index}",
                "status": "completed",
                "output": [
                    {
                        "id": f"fc_{index}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": f"call_{index}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                    }
                ],
            }
        else:
            payload = {
                "id": f"msg_{index}",
                "type": "message",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{index}",
                        "name": name,
                        "input": arguments,
                    }
                ],
                "stop_reason": "tool_use",
            }
        return httpx.Response(200, json=payload)


def _inline_blobs(value: Any, blob_dir: Path) -> Any:
    if isinstance(value, str):

        def replace_blob(match: re.Match[str]) -> str:
            blob = (blob_dir / f"{match.group(1)}.png").read_bytes()
            return base64.b64encode(blob).decode("ascii")

        return _BLOB_RE.sub(replace_blob, value)
    if isinstance(value, list):
        return [_inline_blobs(item, blob_dir) for item in value]
    if isinstance(value, dict):
        return {key: _inline_blobs(item, blob_dir) for key, item in value.items()}
    return value


def _wire_rows(path: Path) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines()]


class _FlushRecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.write_calls: list[str] = []
        self.flush_calls = 0

    def write(self, text: str) -> int:
        self.write_calls.append(text)
        return super().write(text)

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()


def _policy(script: _Script, **kwargs: Any) -> LLMAgentPolicy:
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(script),
        env={},
        **kwargs,
    )


def _task(max_steps: int = 40) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


class _RecordingSink(NullSink):
    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.records.append(record)


class _AbsoluteEmbodiment:
    def __init__(self) -> None:
        self._q = np.array([0.0])
        self.info = EmbodimentInfo(
            name="absolute-test",
            action_space=Box(
                shape=(1,),
                low=np.array([-1.0]),
                high=np.array([1.0]),
                semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
            ),
            observation_space=ObservationSpace(
                state=StateSpec(fields=(StateField(key="q", shape=(1,)),))
            ),
            control_hz=10.0,
            is_simulated=True,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._q = np.array([0.0])
        return Observation(state={"q": self._q.copy()}, instruction=scene.instruction)

    def step(self, action: Action) -> StepResult:
        self._q = np.asarray(action.data, dtype=np.float64).copy()
        return StepResult(observation=Observation(state={"q": self._q.copy()}))

    def close(self) -> None:
        return None


class _VisionAbsoluteEmbodiment:
    def __init__(self, *, terminate_after_first_step: bool = False) -> None:
        self._q = np.array([0.0])
        self._steps = 0
        self._terminate_after_first_step = terminate_after_first_step
        self.info = EmbodimentInfo(
            name="vision-absolute-test",
            action_space=Box(
                shape=(1,),
                low=np.array([-1.0]),
                high=np.array([1.0]),
                semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
            ),
            observation_space=ObservationSpace(
                cameras=(
                    CameraSpec(name="top", height=2, width=2, channels=3),
                    CameraSpec(name="wrist", height=2, width=2, channels=3),
                ),
                state=StateSpec(fields=(StateField(key="q", shape=(1,)),)),
            ),
            control_hz=10.0,
            is_simulated=True,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._q = np.array([0.0])
        self._steps = 0
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        self._steps += 1
        self._q = np.asarray(action.data, dtype=np.float64).copy()
        terminated = self._terminate_after_first_step and self._steps == 1
        return StepResult(
            observation=self._observe(None),
            terminated=terminated,
            termination_reason="test termination" if terminated else None,
        )

    def close(self) -> None:
        return None

    def _observe(self, instruction: str | None) -> Observation:
        return Observation(
            images={
                "top": np.full((2, 2, 3), self._steps, dtype=np.uint8),
                "wrist": np.full((2, 2, 3), self._steps + 10, dtype=np.uint8),
            },
            state={"q": self._q.copy()},
            instruction=instruction,
        )


def _vision_observation(
    *,
    q: float = 0.0,
    env_step: object = 0,
    cameras: tuple[str, ...] = ("top", "wrist"),
) -> Observation:
    images = {
        name: np.full((2, 2, 3), index, dtype=np.uint8)
        for index, name in enumerate(cameras, start=1)
    }
    return Observation(state={"q": np.array([q])}, images=images, extra={"env_step": env_step})


# --- tests -----------------------------------------------------------------------


def test_goal_runs_to_done_and_config_lands_in_log(tmp_path: Path) -> None:
    script = _Script(
        [
            _tool_response("move_by", {"deltas": {"dx": 0.1, "dy": 0.1}}),
            _tool_response("done", {"summary": "close enough"}),
        ]
    )
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "success"
    (record,) = sink.records
    assert record.truncated is True
    assert record.termination_reason == "done"
    # Headroom splits a box-sized move into two steps, then done holds once.
    assert len(record.steps) == 3
    assert logs[0].eval.policy_config["model"] == "test/model"
    assert logs[0].eval.policy_config["wire_capture"] is True
    assert logs[0].eval.policy_config["max_llm_calls"] == 100
    assert logs[0].eval.policy_config["max_speed_frac"] == 0.1
    (transcript,) = logs[0].samples[0].policy_transcripts
    serialized = json.dumps(logs[0].to_dict())
    assert transcript is not None
    assert "move_by" in serialized and "done" in serialized
    assert "data:" not in serialized


def test_done_hindsight_lands_in_trial_metadata(tmp_path: Path) -> None:
    hindsight = "the overhead camera swaps the apparent left and right axes"
    script = _Script([_tool_response("done", {"summary": "cube reached", "hindsight": hindsight})])

    logs = ir_eval(
        _task(),
        _policy(script),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    assert logs[0].samples[0].trial_metadata[0]["hindsight"] == hindsight


def test_forced_give_up_omits_hindsight_from_trial_metadata(tmp_path: Path) -> None:
    script = _Script([_tool_response("move_by", {"deltas": {"dx": 0.01}})])

    logs = ir_eval(
        _task(),
        _policy(script, max_llm_calls=1),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    metadata = logs[0].samples[0].trial_metadata[0]
    assert "hindsight" not in metadata
    # Pin that the trial really ended via the forced give_up, not max_steps:
    # a give_up termination with the whole budget consumed cannot be a real
    # give_up because the script contains no stop response.
    assert logs[0].samples[0].termination_reasons == ("give_up",)
    assert metadata["llm_usage"]["llm_calls"] == 1


def test_none_sentinel_hindsight_is_not_persisted(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "cube reached", "hindsight": "None"})])

    logs = ir_eval(
        _task(),
        _policy(script),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    assert "hindsight" not in logs[0].samples[0].trial_metadata[0]


def test_hindsight_is_reset_before_a_later_forced_give_up(tmp_path: Path) -> None:
    hindsight = "the gripper closes along the camera's vertical axis"
    move = _tool_response("move_by", {"deltas": {"dx": 0.01}})
    script = _Script(
        [
            move,
            _tool_response("done", {"summary": "cube reached", "hindsight": hindsight}),
            move,
        ]
    )
    task = Task(
        name="two-trial-hindsight-reset",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=40,
        epochs=2,
    )

    logs = ir_eval(
        task,
        _policy(script, max_llm_calls=2),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    first, second = logs[0].samples[0].trial_metadata
    assert first["hindsight"] == hindsight
    assert "hindsight" not in second
    # Trial 2 must have ended via the forced give_up (budget consumed on
    # replayed moves; the script's trailing response is not a stop).
    assert logs[0].samples[0].termination_reasons == ("done", "give_up")
    assert second["llm_usage"]["llm_calls"] == 2


@pytest.mark.parametrize("wire", ["chat", "responses", "messages"])
def test_wire_capture_matches_each_transport_body_after_blob_inlining(
    wire: str, tmp_path: Path
) -> None:
    script = _WireScript(wire)
    sink = _RecordingSink()
    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        wire=wire,
        image_horizon=1,
        depth="off",
        transport=httpx.MockTransport(script),
        env={},
    )

    ir_eval(
        _task(max_steps=20),
        policy,
        _VisionAbsoluteEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    (record,) = sink.records
    pointer = record.metadata["wire_capture"]
    assert isinstance(pointer, str)
    capture_path = tmp_path / pointer
    assert capture_path.is_file()
    rows = _wire_rows(capture_path)
    row = next(row for row in rows if row["call"] == 1 and row["attempt"] == 0)
    assert (
        row["endpoint"]
        == {
            "chat": "/chat/completions",
            "responses": "/responses",
            "messages": "/messages",
        }[wire]
    )
    blob_dir = capture_path.parent.parent / "blobs"
    captured_request = _inline_blobs(row["request"], blob_dir)
    assert captured_request == script.requests[1]
    assert "camera frame(s) elided" in json.dumps(captured_request)

    if wire == "messages":
        elision_blocks = [
            block
            for message in captured_request["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if "camera frame(s) elided" in block.get("text", "")
        ]
        assert elision_blocks
        assert elision_blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_zero_llm_call_trial_creates_no_capture_file_or_metadata(tmp_path: Path) -> None:
    class _FailsBeforeLLM(LLMAgentPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            raise RuntimeError("failed before first LLM call")

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be called")

    policy = _FailsBeforeLLM(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(unexpected_request),
        env={},
    )
    sink = _RecordingSink()

    ir_eval(
        _task(),
        policy,
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    (record,) = sink.records
    assert record.status == "error"
    assert "wire_capture" not in record.metadata
    assert not (tmp_path / "wire").exists()
    assert policy._capture is not None
    assert policy._capture.end_trial() is None


def test_wire_capture_false_constructs_no_sink_or_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_capture() -> None:
        raise AssertionError("WireCapture must not be constructed")

    monkeypatch.setattr(agent_policy_module, "WireCapture", unexpected_capture)
    script = _Script([_tool_response("done", {"summary": "disabled"})])
    policy = _policy(script, wire_capture=False)
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.wire_capture is False
    assert policy._client._capture is None
    sink = _RecordingSink()

    ir_eval(
        _task(),
        policy,
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    assert "wire_capture" not in sink.records[0].metadata
    assert not (tmp_path / "wire").exists()


def test_version_skew_warning_fires_once_across_trial_ends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy(_Script([_text_response("unused")]))
    first = TrialRecord(scene_id="s0", epoch=0, seed=0, status="success")
    second = TrialRecord(scene_id="s0", epoch=1, seed=1, status="success")

    policy.on_trial_end(first, str(tmp_path), "run-1")
    policy.on_trial_end(second, str(tmp_path), "run-1")

    assert capsys.readouterr().err == (
        "[agent] wire capture inactive: core predates on_trial_start\n"
    )
    assert "wire_capture" not in first.metadata
    assert "wire_capture" not in second.metadata


def test_success_without_action_chunk_retries_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(
        [
            _tool_response("done", {"summary": "first"}),
            _tool_response("done", {"summary": "second"}),
        ]
    )
    policy = _policy(script)
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))
    toolset = policy._toolset
    assert toolset is not None
    execute = toolset.execute
    calls = 0

    def no_chunk_once(call: Any, observation: Observation) -> ToolResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ToolResult(note="accepted without an action")
        return execute(call, observation)

    monkeypatch.setattr(toolset, "execute", no_chunk_once)

    chunk = policy.act(Observation())

    assert chunk.actions[0].meta["request_stop"] is True
    assert len(script.requests) == 2


def test_pending_narration_handles_absent_target_and_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.bind(_VisionAbsoluteEmbodiment().info)
    observation = _vision_observation(env_step=2)
    without_target = _PendingCapture(
        requested=("top",),
        issued_step=1,
        chunk_len=1,
        target=None,
    )
    assert "finished playing" in policy._pending_narration(without_target, observation, ())

    toolset = policy._toolset
    assert toolset is not None
    monkeypatch.setattr(toolset, "residual", lambda target, current: None)
    with_target = replace(without_target, target=np.array([0.1]))
    narration = policy._pending_narration(with_target, observation, ())
    assert "remaining offset" not in narration


def test_image_parts_skip_requested_camera_missing_from_observation() -> None:
    observation = _vision_observation(cameras=("top",))

    assert _image_parts(observation, reveal=("wrist",), depth={}) == []


def test_transcript_is_none_before_first_reset() -> None:
    policy = _policy(_Script([_text_response("unused")]))
    assert policy.transcript() is None


def test_observation_content_renders_approver_interventions() -> None:
    obs = Observation(
        state={"q": np.array([0.0])},
        extra={
            "env_step": 2,
            "approvals": [
                {"t": 0, "detail": "clamped"},
                {"t": 1, "detail": "clamped"},
                {"t": 2, "detail": "delta_clamped"},
                {"t": 3, "detail": "clamped"},
            ],
        },
    )
    parts = _observation_content(obs)
    text = parts[0]["text"]
    assert "approver: 4 step(s) modified (clamped \u00d73, delta_clamped)." in text


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [{"t": 3, "text": "move farther left"}],
            ["operator feedback (step 3): move farther left"],
        ),
        (
            [
                {"t": 1, "text": "the red cube is the target"},
                {"t": 4, "text": "close the gripper now"},
            ],
            [
                "operator feedback (step 1): the red cube is the target",
                "operator feedback (step 4): close the gripper now",
            ],
        ),
    ],
)
def test_observation_content_renders_operator_messages(
    messages: list[dict[str, object]],
    expected: list[str],
) -> None:
    observation = Observation(extra={"operator_messages": messages})

    parts = _observation_content(observation, narration="motion finished")
    lines = parts[0]["text"].splitlines()

    assert [line for line in lines if line.startswith("operator feedback")] == expected
    assert lines.index(expected[-1]) < lines.index("motion finished")


def test_operator_lines_skip_malformed_values_and_entries() -> None:
    malformed: list[object] = [
        None,
        "feedback",
        {"t": "2", "text": "wrong step type"},
        {"t": 2, "text": 3},
        {"t": 2},
        {"text": "missing step"},
        {"t": 5, "text": "valid feedback"},
    ]

    assert _operator_lines(Observation(extra={"operator_messages": malformed})) == [
        "operator feedback (step 5): valid feedback"
    ]
    assert _operator_lines(Observation(extra={"operator_messages": {"t": 1}})) == []


def test_operator_lines_are_empty_when_observation_has_no_reserved_key() -> None:
    assert _operator_lines(Observation()) == []


def test_agent_opts_in_to_framework_operator_messages() -> None:
    assert LLMAgentPolicy.accepts_operator_messages is True


@pytest.mark.parametrize("template", [_SYSTEM_TEMPLATE, _ON_DEMAND_SYSTEM_TEMPLATE])
def test_system_prompts_treat_operator_feedback_as_trusted_guidance(template: str) -> None:
    expected = (
        "You may receive operator feedback lines mid-run; treat them as trusted guidance from "
        "the human supervising the robot."
    )
    assert expected in template


@pytest.mark.parametrize("template", [_SYSTEM_TEMPLATE, _ON_DEMAND_SYSTEM_TEMPLATE])
def test_system_prompts_announce_the_hindsight_question(template: str) -> None:
    rendered = template.format(name="test-robot", budget=10)
    announcement = (
        "Note what you are learning about this rig and task as you go: done and give_up will ask "
        "what you wish you had known from the start."
    )

    assert announcement in rendered


def test_reset_before_bind_uses_the_unchanged_unbound_prompt() -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[0]["content"] == _SYSTEM_TEMPLATE.format(name="(unbound)", budget=100)


def test_embodiment_docs_are_appended_verbatim_after_formatting() -> None:
    docs = '  Keep {x/y} literal.\n```json\n{"open": 1}\n```  '
    info = replace(CubePickEmbodiment().info, docs=docs)
    policy = _policy(_Script([_text_response("unused")]))
    policy.bind(info)
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[0]["content"] == (
        _SYSTEM_TEMPLATE.format(name="cubepick", budget=100)
        + "\n\nEmbodiment notes:\n"
        + docs.strip()
    )


@pytest.mark.parametrize("docs", [None, "", " \n\t "])
def test_absent_embodiment_docs_leave_the_prompt_unchanged(docs: str | None) -> None:
    info = replace(CubePickEmbodiment().info, docs=docs)
    policy = _policy(_Script([_text_response("unused")]))
    policy.bind(info)
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[0]["content"] == _SYSTEM_TEMPLATE.format(name="cubepick", budget=100)


def test_prior_learnings_follow_embodiment_docs_and_record_provenance(
    tmp_path: Path,
) -> None:
    docs = "Use the wrist camera before closing the gripper."
    learnings = "# Prior lessons\n\nApproach from the object's left side.\n"
    path = tmp_path / "learnings.md"
    path.write_text(learnings, encoding="utf-8")
    info = replace(CubePickEmbodiment().info, docs=docs)
    policy = _policy(
        _Script([_text_response("unused")]),
        prior_learnings=str(path),
    )
    policy.bind(info)
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[0]["content"] == (
        _SYSTEM_TEMPLATE.format(name="cubepick", budget=100)
        + "\n\nEmbodiment notes:\n"
        + docs
        + _PRIOR_LEARNINGS_FRAME
        + learnings
    )
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.prior_learnings == str(path.resolve())
    assert (
        policy.config.prior_learnings_sha256
        == hashlib.sha256(learnings.encode("utf-8")).hexdigest()
    )


def test_prior_learnings_default_off_preserves_prompt_and_config() -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[0]["content"] == _SYSTEM_TEMPLATE.format(name="(unbound)", budget=100)
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.prior_learnings is None
    assert policy.config.prior_learnings_sha256 is None


@pytest.mark.parametrize(
    ("images", "template"),
    [
        ("always", _SYSTEM_TEMPLATE),
        ("on_demand", _ON_DEMAND_SYSTEM_TEMPLATE),
    ],
)
def test_pre_check_prompt_clause_is_present_only_when_configured(
    images: str,
    template: str,
) -> None:
    def allow(waypoints: npt.NDArray[np.float64]) -> str | None:
        return None

    without = _policy(_Script([_text_response("unused")]), images=images)
    without.reset(Scene(id="s0", instruction="reach"))
    without_transcript = without.transcript()
    assert without_transcript is not None
    expected = template.format(name="(unbound)", budget=100)
    assert without_transcript[0]["content"] == expected

    with_hook = _policy(
        _Script([_text_response("unused")]),
        images=images,
        pre_check=allow,
    )
    with_hook.reset(Scene(id="s0", instruction="reach"))
    with_transcript = with_hook.transcript()
    assert with_transcript is not None
    prompt = with_transcript[0]["content"]
    assert prompt.startswith(expected)
    assert "pre-check may reject a move with a stated reason" in prompt
    assert "Adjust the target rather than repeating a rejected move" in prompt


def test_pre_check_config_records_function_and_callable_type_identity() -> None:
    def allow(waypoints: npt.NDArray[np.float64]) -> str | None:
        return None

    unset = _policy(_Script([_text_response("unused")]))
    plain = _policy(_Script([_text_response("unused")]), pre_check=allow)
    partial = functools.partial(allow)
    wrapped = _policy(_Script([_text_response("unused")]), pre_check=partial)

    assert isinstance(unset.config, AgentPolicyConfig)
    assert isinstance(plain.config, AgentPolicyConfig)
    assert isinstance(wrapped.config, AgentPolicyConfig)
    assert unset.config.pre_check is None
    assert plain.config.pre_check == f"{allow.__module__}.{allow.__qualname__}"
    assert wrapped.config.pre_check == (f"{type(partial).__module__}.{type(partial).__qualname__}")


def test_pre_check_rejects_non_callables_with_programmatic_guidance() -> None:
    with pytest.raises(
        ConfigError,
        match=r"(?s)pre_check must be callable.*-P CLI flags.*programmatic-only",
    ):
        _policy(_Script([_text_response("unused")]), pre_check=cast(Any, "checker"))


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("missing.md", None),
        ("empty.md", ""),
        ("whitespace.md", " \n\t"),
        ("oversize.md", "x" * (_PRIOR_LEARNINGS_TEXT_LIMIT + 1)),
    ],
    ids=["missing", "empty", "whitespace", "oversize"],
)
def test_prior_learnings_file_errors_are_guided(
    tmp_path: Path,
    filename: str,
    contents: str | None,
) -> None:
    path = tmp_path / filename
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=r"(?s)prior_learnings.*fix:"):
        _policy(
            _Script([_text_response("unused")]),
            prior_learnings=str(path),
        )


@pytest.mark.parametrize("value", ["", 42], ids=["empty_string", "non_string"])
def test_prior_learnings_coercion_errors_are_guided(value: object) -> None:
    with pytest.raises(ConfigError, match=r"(?s)prior_learnings.*fix:.*coerces unquoted values"):
        _policy(
            _Script([_text_response("unused")]),
            prior_learnings=value,
        )


def test_prior_learnings_are_not_reread_on_reset(tmp_path: Path) -> None:
    learnings = "Keep the camera centered on the target."
    path = tmp_path / "learnings.md"
    path.write_text(learnings, encoding="utf-8")
    policy = _policy(
        _Script([_text_response("unused")]),
        prior_learnings=str(path),
    )

    policy.reset(Scene(id="s0", instruction="reach"))
    first = policy.transcript()
    path.unlink()
    policy.reset(Scene(id="s1", instruction="reach again"))
    second = policy.transcript()

    assert first is not None
    assert second is not None
    assert first[0]["content"] == second[0]["content"]
    assert _PRIOR_LEARNINGS_FRAME + learnings in second[0]["content"]


def test_system_prompt_requires_human_readable_move_notes() -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert _NOTE_CONTRACT in transcript[0]["content"]


def test_bind_accepts_legacy_embodiment_info_without_docs() -> None:
    current = CubePickEmbodiment().info

    class _LegacyInfo:
        name = current.name
        action_space = current.action_space
        observation_space = current.observation_space
        control_hz = current.control_hz

    policy = _policy(_Script([_text_response("unused")]))
    policy.bind(cast(EmbodimentInfo, _LegacyInfo()))
    policy.reset(Scene(id="s0", instruction="reach"))

    transcript = policy.transcript()

    assert transcript is not None
    assert "Embodiment notes:" not in transcript[0]["content"]


def test_observation_content_labels_the_selected_state_field_exactly() -> None:
    content = _observation_content(
        Observation(state={"joint_pos": np.array([0.01, -0.02, 0.98])}),
        ("joint_pos", ("left_j0", "left_j1", "right_gripper")),
    )

    assert content == [
        {
            "type": "text",
            "text": (
                "Current observation.\n"
                "state[joint_pos]: left_j0=0.01 left_j1=-0.02 right_gripper=0.98"
            ),
        }
    ]


def test_observation_content_uses_unlabeled_fallback_for_length_mismatch() -> None:
    content = _observation_content(
        Observation(state={"joint_pos": np.array([0.01, -0.02])}),
        ("joint_pos", ("left_j0", "left_j1", "right_gripper")),
    )

    assert content[0]["text"] == "Current observation.\nstate[joint_pos]: [0.01, -0.02]"


def test_observation_content_labels_every_camera_with_the_environment_step() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)

    content = _observation_content(
        Observation(
            images={"top_cam": image, "left_cam": image},
            extra={"env_step": 480},
        )
    )

    assert content[1]["text"] == "camera 'top_cam' (step 480):"
    assert content[2]["type"] == "image_url"
    assert content[3]["text"] == "camera 'left_cam' (step 480):"
    assert content[4]["type"] == "image_url"


def test_observation_content_step_label_fallbacks() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)

    without_step = _observation_content(Observation(images={"top": image}))
    string_step = _observation_content(
        Observation(images={"top": image}, extra={"env_step": "480"})
    )
    bool_step = _observation_content(Observation(images={"top": image}, extra={"env_step": True}))
    numpy_step = _observation_content(
        Observation(images={"top": image}, extra={"env_step": np.int64(480)})
    )

    assert without_step[1]["text"] == "camera 'top':"
    assert string_step[1]["text"] == "camera 'top':"
    assert bool_step[1]["text"] == "camera 'top' (step True):"
    assert numpy_step[1]["text"] == "camera 'top':"


def test_observation_message_places_depth_immediately_after_its_camera() -> None:
    script = _Script([_tool_response("done", {"summary": "observed depth"})])
    policy = _policy(script)
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="inspect depth"))
    observation = _vision_observation(env_step=3)
    observation = replace(
        observation,
        extra={
            **observation.extra,
            "top_depth": lambda: np.full((2, 2), 0.5, dtype=np.float64),
        },
    )

    policy.act(observation)

    content = script.requests[0]["messages"][-1]["content"]
    assert content[1]["text"] == "camera 'top' (step 3):"
    assert content[2]["type"] == "image_url"
    assert content[3]["text"] == (
        "depth 'top' (step 3): bright 0.50 m -> dim 0.50 m "
        "(2nd-98th pctl), 100% valid, center 0.50 m:"
    )
    assert content[4]["type"] == "image_url"
    assert content[5]["text"] == "camera 'wrist' (step 3):"


def test_depth_off_preserves_pre_depth_observation_message() -> None:
    calls = 0

    def depth_thunk() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((2, 2), 0.5, dtype=np.float64)

    script = _Script([_tool_response("done", {"summary": "ignored depth"})])
    policy = _policy(script, depth="off")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="ignore depth"))
    observation = _vision_observation(env_step=4)
    observation = replace(
        observation,
        extra={**observation.extra, "top_depth": depth_thunk},
    )
    expected = _observation_content(observation, policy._state_labels)

    policy.act(observation)

    assert script.requests[0]["messages"][-1]["content"] == expected
    assert calls == 0


def test_missing_depth_key_preserves_pre_depth_observation_message() -> None:
    script = _Script([_tool_response("done", {"summary": "rgb only"})])
    policy = _policy(script)
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="inspect rgb"))
    observation = _vision_observation(env_step=5)
    expected = _observation_content(observation, policy._state_labels)

    policy.act(observation)

    assert script.requests[0]["messages"][-1]["content"] == expected


def test_failing_depth_thunk_keeps_observation_message_valid() -> None:
    def depth_thunk() -> np.ndarray:
        raise RuntimeError("camera offline")

    script = _Script([_tool_response("done", {"summary": "used rgb"})])
    policy = _policy(script)
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="inspect available data"))
    observation = _vision_observation(env_step=6)
    observation = replace(
        observation,
        extra={**observation.extra, "top_depth": depth_thunk},
    )

    policy.act(observation)

    content = script.requests[0]["messages"][-1]["content"]
    assert content[1]["text"] == "camera 'top' (step 6):"
    assert content[2]["type"] == "image_url"
    assert content[3] == {
        "type": "text",
        "text": "depth 'top' unavailable: camera offline",
    }
    assert content[4]["text"] == "camera 'wrist' (step 6):"
    assert sum(part["type"] == "image_url" for part in content) == 2


def test_transcript_echo_defaults_off(capsys: pytest.CaptureFixture[str]) -> None:
    policy = _policy(_Script([_tool_response("done", {"summary": "quiet"})]))
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="observe quietly"))
    policy.act(Observation(extra={"env_step": 0}))

    assert capsys.readouterr().err == ""


def test_transcript_echo_reports_conversation_in_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _policy(
        _Script([_text_and_tool_response("The goal is complete.", "done", {"summary": "ok"})]),
        transcript_echo=True,
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="inspect the cube"))
    policy.act(
        Observation(
            state={"eef_pos": np.array([0.123456, -0.2])},
            extra={"env_step": 0},
        )
    )

    lines = capsys.readouterr().err.splitlines()
    expected = [
        "[agent] goal: inspect the cube",
        "[agent] >> step 0: 0 camera(s), state[eef_pos]: [0.1235, -0.2]",
        "[agent] << The goal is complete.",
        '[agent] << tool_call done({"summary": "ok"})',
        "[agent] -- done: ok",
    ]
    positions = [lines.index(line) for line in expected]
    assert positions == sorted(positions)


def test_transcript_echo_reports_tool_results_in_call_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _policy(
        _Script(
            [
                _multi_tool_response(
                    [
                        ("done", {"summary": "first executes"}),
                        ("give_up", {"reason": "extra"}),
                        ("give_up", {"reason": "second extra"}),
                    ]
                )
            ]
        ),
        transcript_echo=True,
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))
    policy.act(Observation(extra={"env_step": 0}))

    event_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "tool_call" in line or line.startswith("[agent] --")
    ]
    assert event_lines == [
        '[agent] << tool_call done({"summary": "first executes"})',
        '[agent] << tool_call give_up({"reason": "extra"})',
        '[agent] << tool_call give_up({"reason": "second extra"})',
        "[agent] -- done: first executes",
        "[agent] -- ignored: one tool call per turn",
        "[agent] -- ignored: one tool call per turn",
    ]


def test_transcript_echo_flushes_every_event(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(
        _Script([_tool_response("done", {"summary": "flushed"})]), transcript_echo=True
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="flush"))
    stream = _FlushRecordingStream()
    monkeypatch.setattr(sys, "stderr", stream)

    policy.act(Observation(extra={"env_step": 0}))

    newline_count = sum(part.count("\n") for part in stream.write_calls)
    assert newline_count > 0
    assert stream.flush_calls >= newline_count


def test_transcript_echo_observation_omits_image_payloads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _policy(
        _Script([_tool_response("done", {"summary": "observed"})]), transcript_echo=True
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="observe"))
    policy.act(
        Observation(
            state={"eef_pos": np.array([0.123456])},
            images={"top": np.zeros((1, 1, 3), dtype=np.uint8)},
            extra={"env_step": 0},
        )
    )

    stderr = capsys.readouterr().err
    assert "[agent] >> step 0: 1 camera(s), state[eef_pos]: [0.1235]" in stderr
    assert "data:image" not in stderr


@pytest.mark.parametrize(
    "extra",
    [{}, {"env_step": "3"}],
    ids=["missing", "non_int_string"],
)
def test_transcript_echo_observation_without_environment_step(
    extra: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy(
        _Script([_tool_response("done", {"summary": "observed"})]), transcript_echo=True
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="observe"))
    policy.act(Observation(state={"eef_pos": np.array([0.0])}, extra=extra))

    stderr = capsys.readouterr().err
    assert "[agent] >> observation: 0 camera(s), state[eef_pos]: [0.0]" in stderr
    assert ">> step" not in stderr


def test_transcript_echo_never_echoes_retry_nudge(
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _policy(
        _Script(
            [
                _text_response("thinking out loud, no tool call"),
                _tool_response("done", {"summary": "recovered"}),
            ]
        ),
        transcript_echo=True,
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="recover"))
    policy.act(Observation(extra={"env_step": 0}))

    stderr = capsys.readouterr().err
    assert "[agent] << thinking out loud, no tool call" in stderr
    assert "Respond with exactly one tool call." not in stderr


def test_transcript_echo_reports_forced_give_up(capsys: pytest.CaptureFixture[str]) -> None:
    policy = _policy(
        _Script([_tool_response("move_by", {"deltas": {"dx": 0.01}})]),
        max_llm_calls=1,
        transcript_echo=True,
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move"))
    policy.act(Observation(extra={"env_step": 0}))
    policy.act(Observation(extra={"env_step": 1}))

    assert "[agent] -- LLM call budget exhausted; forcing give_up" in capsys.readouterr().err


@pytest.mark.parametrize(
    "response",
    [
        _tool_response("done", {"summary": "no text"}),
        _text_and_tool_response("", "done", {"summary": "no text"}),
    ],
    ids=["content_none", "content_empty_string"],
)
def test_transcript_echo_omits_empty_assistant_text(
    response: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy(_Script([response]), transcript_echo=True)
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))
    policy.act(Observation(extra={"env_step": 0}))

    assistant_lines = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("[agent] <<")
    ]
    assert assistant_lines == ['[agent] << tool_call done({"summary": "no text"})']


def test_transcript_echo_lands_in_policy_config() -> None:
    policy = _policy(_Script([_text_response("unused")]), transcript_echo=True)

    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.transcript_echo is True


def test_transcript_preserves_step_label_next_to_elided_image() -> None:
    policy = _policy(_Script([_tool_response("done", {"summary": "observed"})]))
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="observe"))
    policy.act(
        Observation(
            images={"top_cam": np.zeros((1, 1, 3), dtype=np.uint8)},
            extra={"env_step": 480},
        )
    )

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[-3]["content"][-2:] == [
        {"type": "text", "text": "camera 'top_cam' (step 480):"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]


def test_transcript_strips_images_and_preserves_text_and_tools() -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.reset(Scene(id="s0", instruction="reach"))
    tool_call = {
        "id": "call_move",
        "type": "function",
        "function": {"name": "move_by", "arguments": '{"dx": 0.1}'},
    }
    policy._messages.extend(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "camera 'top':"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,secret"},
                    },
                ],
            },
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_move", "content": "moved"},
        ]
    )

    transcript = policy.transcript()

    assert transcript is not None
    assert transcript[-3]["content"] == [
        {"type": "text", "text": "camera 'top':"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]
    assert transcript[-2]["tool_calls"] == [tool_call]
    assert transcript[-1] == {
        "role": "tool",
        "tool_call_id": "call_move",
        "content": "moved",
    }


def test_transcript_is_deeply_isolated_in_both_directions() -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.reset(Scene(id="s0", instruction="reach"))
    first = policy.transcript()
    assert first is not None

    first[0]["content"] = "changed return"
    assert policy._messages[0]["content"] != "changed return"

    policy._messages[1]["content"] = "changed live state"
    assert first[1]["content"] == "Goal: reach"


def test_transcript_delta_tracks_new_sanitized_messages_and_reset() -> None:
    script = _Script(
        [
            _tool_response("move_by", {"deltas": {"dx": 0.05}}),
            _tool_response("done", {"summary": "finished"}),
        ]
    )
    policy = _policy(script)
    policy.bind(CubePickEmbodiment().info)
    scene = Scene(id="s0", instruction="reach")
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    policy.reset(scene)

    policy.act(Observation(images={"top": image}))
    first = policy.transcript_delta()
    assert first is not None
    assert policy.transcript_delta() is None
    assert "data:image" not in json.dumps(first)

    policy.act(Observation(images={"top": image}))
    second = policy.transcript_delta()
    assert second is not None
    assert second[0]["role"] == "user"
    assert all(message["role"] != "system" for message in second)
    assert "data:image" not in json.dumps(second)

    full = policy.transcript()
    assert full is not None
    assert len(full) == len(first) + len(second)
    assert full[: len(first)] == first
    assert full[len(first) :] == second

    policy.reset(scene)
    reset_delta = policy.transcript_delta()
    assert reset_delta is not None
    assert [message["role"] for message in reset_delta] == ["system", "user"]


def test_outbound_messages_carry_state_images_and_tools(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "looked around"})])
    ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))
    first = script.requests[0]
    assert first["model"] == "test/model"
    assert [t["function"]["name"] for t in first["tools"]] == ["move_by", "done", "give_up"]
    system, goal, observation = first["messages"]
    assert "cubepick" in system["content"]
    assert "Embodiment notes:\n" in system["content"]
    docs = CubePickEmbodiment().info.docs
    assert docs is not None and docs in system["content"]
    assert goal["content"] == "Goal: reach the cube"
    text_parts = [p["text"] for p in observation["content"] if p["type"] == "text"]
    assert any("state[eef_pos]" in t for t in text_parts)
    image_parts = [p for p in observation["content"] if p["type"] == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_wild_swing_is_clamped_by_guardrails_but_not_without(tmp_path: Path) -> None:
    def run(approver: Any) -> TrialRecord:
        script = _Script(
            [
                _tool_response("move_by", {"deltas": {"dx": 0.08}}),
                _tool_response("done", {"summary": "stop"}),
            ]
        )
        sink = _RecordingSink()
        ir_eval(
            _task(),
            _policy(script),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            sinks=[sink],
            approver=approver,
        )
        return sink.records[0]

    space = CubePickEmbodiment().info.action_space
    guarded = run(ChainApprover(ClampApprover(space), DeltaLimitApprover(space, max_delta=0.02)))
    approvals = [e for e in guarded.events if e.kind == "approval"]
    assert approvals
    assert any(e.data.get("detail") == "delta_clamped" for e in approvals)
    assert float(np.asarray(guarded.steps[0].action.data)[0]) == 0.02

    unguarded = run(None)  # eval()'s Python-API default is the permissive AutoApprover
    assert not [e for e in unguarded.events if e.kind == "approval"]
    assert float(np.asarray(unguarded.steps[0].action.data)[0]) == 0.08


def test_llm_call_budget_forces_give_up(tmp_path: Path) -> None:
    script = _Script([_tool_response("move_by", {"deltas": {"dx": 0.05}})])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(),
        _policy(script, max_llm_calls=1),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )
    assert logs[0].status == "success"
    (record,) = sink.records
    assert record.termination_reason == "give_up"


def test_persistent_non_tool_output_becomes_policy_error(tmp_path: Path) -> None:
    script = _Script([_text_response("I would rather write a poem about the cube.")])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    # A PolicyError marks the trial errored (never scored); the eval still
    # returns a log, and with its only trial errored the log reports "error".
    (sample,) = logs[0].samples
    assert sample.status == "error"
    assert logs[0].status == "error"
    (record,) = sink.records
    assert record.error is not None and "no tool call" in record.error


def test_no_tool_call_three_strikes_notes_explicit_chat_base_url() -> None:
    policy = _policy(_Script([_text_response("no tools")]))
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    with pytest.raises(RuntimeError) as excinfo:
        policy.act(Observation())

    assert str(excinfo.value).endswith(
        "note: some OpenAI-compatible endpoints accept `tools` but silently ignore "
        "them (Tinker's OpenAI-compatible API is one). If the provider serves the "
        "Messages API, retry with -P wire=messages and its Messages base_url."
    )


def test_no_tool_call_three_strikes_omits_note_without_explicit_base_url() -> None:
    script = _Script([_text_response("no tools")])
    policy = LLMAgentPolicy(
        model="openai/gpt-test",
        transport=httpx.MockTransport(script),
        env={"OPENAI_API_KEY": "sk-oai"},
    )
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    with pytest.raises(RuntimeError) as excinfo:
        policy.act(Observation())

    assert "silently ignore" not in str(excinfo.value)


def test_invalid_tool_call_three_strikes_omits_silent_tool_drop_note() -> None:
    policy = _policy(_Script([_tool_response("move_by", {"deltas": {"dz": 0.1}})]))
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    with pytest.raises(RuntimeError) as excinfo:
        policy.act(Observation())

    assert "tool calls kept failing" in str(excinfo.value)
    assert "silently ignore" not in str(excinfo.value)


def test_recoverable_tool_error_is_fed_back_and_corrected(tmp_path: Path) -> None:
    script = _Script(
        [
            _tool_response("move_by", {"deltas": {"dz": 0.1}}),  # bad dim
            _tool_response("move_by", {"deltas": {"dx": 0.1}}),
            _tool_response("done", {"summary": "recovered"}),
        ]
    )
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "success"
    assert sink.records[0].termination_reason == "done"
    # The error went back to the model as a tool message naming the offender.
    tool_messages = [
        m for request in script.requests for m in request["messages"] if m.get("role") == "tool"
    ]
    assert any("unknown dimension 'dz'" in str(m["content"]) for m in tool_messages)


def test_pre_check_rejection_is_fed_back_and_corrected() -> None:
    checked_targets: list[float] = []

    def reject_high_target(waypoints: npt.NDArray[np.float64]) -> str | None:
        target = float(waypoints[-1, 0])
        checked_targets.append(target)
        return "joint target enters the keep-out zone" if target > 0.2 else None

    script = _Script(
        [
            _tool_response("move_joints", {"targets": {"joint": 0.5}}),
            _tool_response("move_joints", {"targets": {"joint": 0.1}}),
        ]
    )
    policy = _policy(script, pre_check=reject_high_target)
    policy.bind(_AbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move safely"))

    chunk = policy.act(Observation(state={"q": np.array([0.0])}))

    assert checked_targets == [0.5, 0.1]
    assert chunk.actions[-1].data[0] == 0.1
    correction = [
        message for message in script.requests[1]["messages"] if message.get("role") == "tool"
    ]
    assert any(
        message["content"]
        == "pre-check rejected this motion: joint target enters the keep-out zone"
        for message in correction
    )


def test_persistent_pre_check_rejections_exhaust_the_failure_budget() -> None:
    calls = 0

    def reject(waypoints: npt.NDArray[np.float64]) -> str | None:
        nonlocal calls
        calls += 1
        return "joint target enters the keep-out zone"

    script = _Script([_tool_response("move_joints", {"targets": {"joint": 0.5}})])
    policy = _policy(script, pre_check=reject)
    policy.bind(_AbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move safely"))

    with pytest.raises(
        RuntimeError,
        match=(
            "LLM tool calls kept failing; last error: pre-check rejected this motion: "
            "joint target enters the keep-out zone"
        ),
    ):
        policy.act(Observation(state={"q": np.array([0.0])}))
    assert calls == 3


def test_pre_check_exception_propagates_from_act_unchanged() -> None:
    error = LookupError("collision model unavailable")

    def crash(waypoints: npt.NDArray[np.float64]) -> str | None:
        raise error

    script = _Script([_tool_response("move_joints", {"targets": {"joint": 0.1}})])
    policy = _policy(script, pre_check=crash)
    policy.bind(_AbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move safely"))

    with pytest.raises(LookupError) as caught:
        policy.act(Observation(state={"q": np.array([0.0])}))
    assert caught.value is error


def test_stop_and_forced_give_up_never_call_pre_check() -> None:
    calls = 0

    def reject(waypoints: npt.NDArray[np.float64]) -> str | None:
        nonlocal calls
        calls += 1
        return "all motion rejected"

    policy = _policy(
        _Script([_tool_response("done", {"summary": "stop"})]),
        max_llm_calls=1,
        pre_check=reject,
    )
    policy.bind(_AbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))
    first = policy.act(Observation(state={"q": np.array([0.0])}))
    forced = policy.act(Observation(state={"q": np.array([0.0])}))

    assert first.actions[0].meta["stop_reason"] == "done"
    assert forced.actions[0].meta["stop_reason"] == "give_up"
    assert calls == 0


def test_missing_note_is_fed_back_and_corrected_in_persisted_transcript(
    tmp_path: Path,
) -> None:
    corrected_note = "The cube is still ahead, so I will move the end effector toward it."
    script = _Script(
        [
            _tool_response(
                "move_by",
                {"deltas": {"dx": 0.05}},
                add_default_note=False,
            ),
            _tool_response(
                "move_by",
                {"deltas": {"dx": 0.05}, "note": corrected_note},
            ),
            _tool_response("done", {"summary": "corrected"}),
        ]
    )
    logs = ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))

    assert logs[0].status == "success"
    correction_messages = [
        message for message in script.requests[1]["messages"] if message.get("role") == "tool"
    ]
    assert any(message["content"] == _NOTE_ERROR for message in correction_messages)

    (transcript,) = logs[0].samples[0].policy_transcripts
    assert transcript is not None
    tool_arguments = [
        json.loads(call["function"]["arguments"])
        for message in transcript
        for call in message.get("tool_calls", [])
    ]
    assert any(arguments.get("note") == corrected_note for arguments in tool_arguments)


def test_persistent_tool_errors_become_policy_error(tmp_path: Path) -> None:
    script = _Script([_tool_response("move_by", {"deltas": {"dz": 0.1}})])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "error"
    assert sink.records[0].error is not None
    assert "tool calls kept failing" in sink.records[0].error


def test_extra_tool_calls_are_answered_but_not_executed(tmp_path: Path) -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_by", {"deltas": {"dx": 0.05}}),
                    ("give_up", {"reason": "extra"}),
                ]
            ),
            _tool_response("done", {"summary": "only the first call ran"}),
        ]
    )
    sink = _RecordingSink()
    ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink])
    assert sink.records[0].termination_reason == "done"
    results = [
        message for message in script.requests[1]["messages"] if message.get("role") == "tool"
    ]
    assert results == [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": "executing move_by over 1 steps (0.1s)",
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "ignored: one tool call per turn",
        },
    ]


def test_extras_behind_a_failed_call_name_the_failure_in_always_mode() -> None:
    # The reason string is not scoped to on-demand mode: blaming the model for
    # a surplus call when the real cause was the earlier failure misleads it in
    # either mode.
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_by", {"deltas": {"nope": 0.05}}),
                    ("give_up", {"reason": "extra"}),
                ]
            ),
            _tool_response("done", {"summary": "recovered"}),
        ]
    )
    policy = _policy(script)
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    policy.act(Observation(extra={"env_step": 0}))

    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results[0].startswith("unknown dimension 'nope'")
    assert results[1] == "ignored: an earlier call in this turn failed"


def test_on_demand_observation_names_available_cameras_without_sending_frames() -> None:
    script = _Script([_tool_response("done", {"summary": "observed state"})])
    policy = _policy(script, images="on_demand")
    embodiment = _VisionAbsoluteEmbodiment()
    policy.bind(embodiment.info)
    policy.reset(Scene(id="s0", instruction="look selectively"))

    policy.act(_vision_observation())

    observation = script.requests[0]["messages"][-1]
    assert observation["role"] == "user"
    assert observation["content"][0]["text"].endswith(
        "Cameras available to take_pic: 'top', 'wrist'."
    )
    assert all(part["type"] != "image_url" for part in observation["content"])
    assert [tool["function"]["name"] for tool in script.requests[0]["tools"]] == [
        "move_joints",
        "done",
        "give_up",
        "take_pic",
    ]


def test_immediate_capture_appends_results_then_frames_and_redecides() -> None:
    script = _Script(
        [
            _tool_response(
                "take_pic",
                {"cameras": ["top"], "note": "I need to inspect the top view first."},
            ),
            _tool_response("done", {"summary": "looked"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    chunk = policy.act(_vision_observation(env_step=7))

    assert chunk.actions[0].meta["request_stop"] is True
    assert len(script.requests) == 2
    history = script.requests[1]["messages"]
    assistant_index = next(
        index
        for index, message in enumerate(history)
        if message.get("tool_calls") and message["tool_calls"][0]["function"]["name"] == "take_pic"
    )
    assert history[assistant_index + 1] == {
        "role": "tool",
        "tool_call_id": "call_take_pic",
        "content": "captured 1 frame(s): 'top'",
    }
    frame_message = history[assistant_index + 2]
    assert frame_message["role"] == "user"
    assert frame_message["content"][0]["text"] == "camera 'top' (step 7):"
    assert frame_message["content"][1]["type"] == "image_url"


def test_immediate_capture_reuses_once_resolved_depth() -> None:
    calls = 0

    def depth_thunk() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((2, 2), 0.5, dtype=np.float64)

    script = _Script(
        [
            _tool_response(
                "take_pic",
                {"cameras": ["top"], "note": "I need the top RGB and depth view."},
            ),
            _tool_response("done", {"summary": "looked with depth"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))
    observation = _vision_observation(env_step=7)
    observation = replace(
        observation,
        extra={**observation.extra, "top_depth": depth_thunk},
    )

    policy.act(observation)

    history = script.requests[1]["messages"]
    frame_message = history[-1]
    assert frame_message["role"] == "user"
    assert frame_message["content"][0]["text"] == "camera 'top' (step 7):"
    assert frame_message["content"][1]["type"] == "image_url"
    assert frame_message["content"][2]["text"].startswith("depth 'top' (step 7):")
    assert frame_message["content"][3]["type"] == "image_url"
    assert calls == 1


def test_immediate_capture_validation_errors_increment_failures() -> None:
    script = _Script([_tool_response("take_pic", {})])
    policy = _policy(script, images="on_demand", max_llm_calls=50)
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    with pytest.raises(
        RuntimeError,
        match="last error: note is required",
    ):
        policy.act(_vision_observation())

    assert len(script.requests) == 3
    assert [
        message["content"] for message in policy._messages if message.get("role") == "tool"
    ] == ["note is required: describe what you observe and why you chose this capture"] * 3


def test_imageless_capture_is_rejected_without_ending_the_trial() -> None:
    script = _Script(
        [
            _tool_response("take_pic", {"note": "I need the current camera view."}),
            _tool_response("done", {"summary": "continued after the dropout"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    chunk = policy.act(_vision_observation(cameras=()))

    assert chunk.actions[0].meta["stop_reason"] == "done"
    assert len(script.requests) == 2
    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results[:2] == [
        "no camera images are available in this observation",
        "done: continued after the dropout",
    ]


def test_a_dropout_rejection_does_not_count_toward_the_three_strike_guard() -> None:
    # The dropout sits between two genuine errors. Treated as an error it would
    # be the third strike and kill the trial, which is what it did before the
    # rejection class existed; treated as the free rejection it is, the trial
    # survives to the next decision.
    script = _Script(
        [
            _tool_response("take_pic", {"note": "  "}, add_default_note=False),
            _tool_response("take_pic", {"note": "The dropout may have cleared."}),
            _tool_response("take_pic", {"note": ""}, add_default_note=False),
            _tool_response("done", {"summary": "survived two errors around a dropout"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    chunk = policy.act(_vision_observation(cameras=()))

    assert chunk.actions[0].meta["stop_reason"] == "done"
    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results[1] == "no camera images are available in this observation"
    assert results[0] == results[2]
    assert results[0].startswith("note is required")


def test_repeated_imageless_capture_rejections_escalate_with_bounded_calls() -> None:
    capture = _tool_response(
        "take_pic",
        {
            "cameras": ["top"],
            "note": "I need the top view despite the camera dropout.",
        },
    )
    script = _Script([capture])
    policy = _policy(script, images="on_demand", max_llm_calls=50)
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    rejection = "no camera images are available in this observation"
    with pytest.raises(RuntimeError, match=f"last error: {rejection}"):
        policy.act(_vision_observation(cameras=()))

    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results == [rejection] * 4
    assert len(script.requests) == 4
    assert len(script.requests) < policy.config.max_llm_calls


def test_take_pic_before_move_short_circuits_with_contiguous_results() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("take_pic", {"note": "I need to look before moving."}),
                    ("move_joints", {"targets": {"joint": 0.5}}),
                ]
            ),
            _tool_response("done", {"summary": "decided after looking"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="inspect"))

    chunk = policy.act(_vision_observation())

    assert chunk.actions[0].meta["stop_reason"] == "done"
    history = script.requests[1]["messages"]
    results = [message for message in history if message.get("role") == "tool"]
    assert results == [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": "captured 2 frame(s): 'top', 'wrist'",
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "ignored: one tool call per turn",
        },
    ]
    result_indices = [history.index(result) for result in results]
    assert result_indices[1] == result_indices[0] + 1
    assert {result["tool_call_id"] for result in results} == {"call_0", "call_1"}


def test_invalid_capture_after_move_keeps_chunk_and_leaves_queue_slot_open() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.2}}),
                    (
                        "take_pic",
                        {
                            "cameras": ["missing"],
                            "note": "I need a view after the move.",
                        },
                    ),
                    (
                        "take_pic",
                        {"cameras": ["top"], "note": "I will use the available top view."},
                    ),
                ]
            )
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move and inspect"))

    chunk = policy.act(_vision_observation())

    assert len(chunk) == 11
    assert [message["content"] for message in policy._messages[-3:]] == [
        "executing move_joints over 11 steps (1.1s)",
        "unknown or unavailable camera 'missing'; available now: 'top', 'wrist'",
        "queued: frames arrive with the next observation, after the motion plays",
    ]
    assert policy._pending is not None
    assert policy._pending.requested == ("top",)


def test_imageless_capture_after_move_is_rejected_without_queuing() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.2}}),
                    (
                        "take_pic",
                        {"note": "I need to inspect the result after moving."},
                    ),
                ]
            )
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move and inspect"))

    chunk = policy.act(_vision_observation(cameras=()))

    assert len(chunk) == 11
    assert [message["content"] for message in policy._messages[-2:]] == [
        "executing move_joints over 11 steps (1.1s)",
        "no camera images are available in this observation",
    ]
    assert policy._pending is None


def test_two_take_pic_calls_capture_once_and_answer_the_second_ignored() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("take_pic", {"note": "I need the current views."}),
                    ("take_pic", {"note": "I am asking twice."}),
                ]
            ),
            _tool_response("done", {"summary": "looked once"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    policy.act(_vision_observation())

    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results[:2] == [
        "captured 2 frame(s): 'top', 'wrist'",
        "ignored: one tool call per turn",
    ]
    image_messages = [
        message
        for message in policy._messages
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
    ]
    assert len(image_messages) == 1


def test_unknown_tool_error_names_take_pic_in_on_demand_mode() -> None:
    script = _Script(
        [
            _tool_response("look", {"note": "wrong tool"}),
            _tool_response("done", {"summary": "corrected"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    policy.act(_vision_observation())

    errors = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert errors[0] == ("unknown tool 'look'; available: move_joints, done, give_up, take_pic")


def test_valid_capture_after_an_earlier_error_is_ignored_with_failure_reason() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    (
                        "take_pic",
                        {"cameras": ["nope"], "note": "I need a camera view."},
                    ),
                    (
                        "take_pic",
                        {"cameras": ["top"], "note": "I will use the top view."},
                    ),
                ]
            ),
            _tool_response("done", {"summary": "corrected"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    policy.act(_vision_observation())

    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results[:2] == [
        "unknown or unavailable camera 'nope'; available now: 'top', 'wrist'",
        "ignored: an earlier call in this turn failed",
    ]


def test_repeated_bare_capture_rejections_escalate_with_bounded_calls() -> None:
    capture = _tool_response(
        "take_pic",
        {"note": "I need every view that has not already been shown."},
    )
    script = _Script([capture])
    policy = _policy(script, images="on_demand", max_llm_calls=50)
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    with pytest.raises(
        RuntimeError,
        match="last error: already shown for this observation",
    ):
        policy.act(_vision_observation())

    rejections = [
        message["content"]
        for message in policy._messages
        if message.get("role") == "tool"
        and str(message["content"]).startswith("already shown for this observation:")
    ]
    assert (
        rejections
        == [
            (
                "already shown for this observation: 'top', 'wrist'; "
                "the view cannot change until the robot moves"
            )
        ]
        * 4
    )
    assert len(script.requests) == 5
    assert len(script.requests) < policy.config.max_llm_calls


def test_named_immediate_capture_reveals_only_unshown_cameras() -> None:
    script = _Script(
        [
            _tool_response(
                "take_pic",
                {"cameras": ["top"], "note": "I need the top view."},
            ),
            _tool_response(
                "take_pic",
                {
                    "cameras": ["top", "wrist"],
                    "note": "I also need the wrist view.",
                },
            ),
            _tool_response("done", {"summary": "both seen"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="look"))

    policy.act(_vision_observation())

    results = [message["content"] for message in policy._messages if message.get("role") == "tool"]
    assert results[:2] == [
        "captured 1 frame(s): 'top'",
        "captured 1 frame(s): 'wrist' (already shown: 'top')",
    ]
    frame_messages = [
        message
        for message in policy._messages
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
    ]
    assert frame_messages[1]["content"][0]["text"] == "camera 'wrist' (step 0):"
    assert len(frame_messages[1]["content"]) == 2


def test_chained_capture_arrives_on_next_observation_with_playout_and_residual() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.2}}),
                    ("take_pic", {"note": "I need to inspect the result after moving."}),
                ]
            ),
            _tool_response("done", {"summary": "inspected"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move and look"))

    chunk = policy.act(_vision_observation(env_step=4))
    policy.act(_vision_observation(q=0.196, env_step=15))

    assert len(chunk) == 11
    first_results = [
        message["content"]
        for message in script.requests[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert first_results == [
        "executing move_joints over 11 steps (1.1s)",
        "queued: frames arrive with the next observation, after the motion plays",
    ]
    arrived = script.requests[1]["messages"][-1]
    assert (
        "The motion finished playing (11 of 11 steps). Largest remaining offset "
        "from the requested target is 0.004 on joint." in arrived["content"][0]["text"]
    )
    assert arrived["content"][1]["text"] == "camera 'top' (step 15):"
    assert arrived["content"][3]["text"] == "camera 'wrist' (step 15):"


def test_chained_capture_reports_default_controller_partial_playout(tmp_path: Path) -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.2}}),
                    ("take_pic", {"note": "I need to see where one played step ends."}),
                ]
            ),
            _tool_response("done", {"summary": "partial observed"}),
        ]
    )
    ir_eval(
        _task(max_steps=4),
        _policy(script, images="on_demand"),
        _VisionAbsoluteEmbodiment(),
        log_dir=str(tmp_path),
        controller=DefaultController(replan_interval=1),
    )

    arrived = script.requests[1]["messages"][-1]["content"][0]["text"]
    assert (
        "The motion played 1 of 11 steps before this observation; it did not run to the end."
    ) in arrived


def test_chained_capture_uses_neutral_narration_when_step_does_not_advance() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.2}}),
                    ("take_pic", {"note": "I need the post-motion views."}),
                ]
            ),
            _tool_response("done", {"summary": "observed"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move"))

    policy.act(_vision_observation(env_step=0))
    policy.act(_vision_observation(q=0.2, env_step=0))

    arrived = script.requests[1]["messages"][-1]["content"][0]["text"]
    assert "These frames follow the motion." in arrived
    assert "finished playing" not in arrived
    assert "motion played" not in arrived


def test_missing_queued_camera_is_narrated_without_raising() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.1}}),
                    (
                        "take_pic",
                        {
                            "cameras": ["top", "wrist"],
                            "note": "I need both views after the move.",
                        },
                    ),
                ]
            ),
            _tool_response("done", {"summary": "used available view"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="move"))

    chunk = policy.act(_vision_observation(env_step=0))
    policy.act(_vision_observation(q=0.1, env_step=len(chunk), cameras=("top",)))

    arrived = script.requests[1]["messages"][-1]["content"]
    assert "Missing camera(s) in this observation: 'wrist'." in arrived[0]["text"]
    assert [part["text"] for part in arrived if part["type"] == "text"][1:] == [
        f"camera 'top' (step {len(chunk)}):"
    ]


def test_queued_capture_is_dropped_when_trial_terminates_and_reset_clears_it(
    tmp_path: Path,
) -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("move_joints", {"targets": {"joint": 0.1}}),
                    ("take_pic", {"note": "I want a frame after this move."}),
                ]
            ),
            _tool_response("done", {"summary": "new trial"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    ir_eval(
        _task(max_steps=3),
        policy,
        _VisionAbsoluteEmbodiment(terminate_after_first_step=True),
        log_dir=str(tmp_path),
    )

    assert policy._pending is not None
    policy.reset(Scene(id="next", instruction="fresh trial"))
    assert policy._pending is None
    policy.act(_vision_observation())
    fresh_observation = script.requests[1]["messages"][-1]["content"]
    assert all(part["type"] != "image_url" for part in fresh_observation)
    assert "These frames follow the motion." not in fresh_observation[0]["text"]


def test_take_pic_after_done_is_ignored_because_the_trial_ends() -> None:
    script = _Script(
        [
            _multi_tool_response(
                [
                    ("done", {"summary": "finished"}),
                    ("take_pic", {"note": "This cannot arrive after the stop."}),
                ]
            )
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="stop"))

    chunk = policy.act(_vision_observation())

    assert chunk.actions[0].meta["request_stop"] is True
    assert [message["content"] for message in policy._messages[-2:]] == [
        "done: finished",
        "ignored: the trial ends with this call",
    ]
    assert policy._pending is None


def test_images_mode_is_recorded_and_invalid_values_have_a_fix() -> None:
    policy = _policy(_Script([_text_response("unused")]), images="on_demand")

    assert policy.config.images == "on_demand"
    with pytest.raises(ConfigError, match=r"(?s)images must be one of.*fix:"):
        _policy(_Script([_text_response("unused")]), images="sometimes")


def test_depth_mode_is_recorded_and_invalid_values_have_a_fix() -> None:
    policy = _policy(_Script([_text_response("unused")]), depth="off")

    assert policy.config.depth == "off"
    with pytest.raises(ConfigError, match=r"(?s)depth must be one of.*fix:"):
        _policy(_Script([_text_response("unused")]), depth="sometimes")


def test_on_demand_prompt_and_no_tool_nudge_describe_chaining() -> None:
    script = _Script(
        [
            _text_response("I should use a tool."),
            _tool_response("done", {"summary": "corrected"}),
        ]
    )
    policy = _policy(script, images="on_demand")
    policy.bind(_VisionAbsoluteEmbodiment().info)
    policy.reset(Scene(id="s0", instruction="inspect"))

    policy.act(_vision_observation())

    assert (
        "Camera images are not attached automatically"
        in script.requests[0]["messages"][0]["content"]
    )
    assert script.requests[1]["messages"][-1]["content"] == (
        "Respond with one motion tool call, one motion followed by take_pic, or take_pic alone."
    )


def test_effort_passthrough_matrix(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "ok"})])
    logs = ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert "reasoning_effort" not in script.requests[0]
    assert logs[0].eval.policy_config["effort"] is None

    script = _Script([_tool_response("done", {"summary": "ok"})])
    logs = ir_eval(
        _task(),
        _policy(script, effort=None),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    assert script.requests[0]["reasoning_effort"] == "none"
    assert logs[0].eval.policy_config["effort"] == "none"

    script = _Script([_tool_response("done", {"summary": "ok"})])
    logs = ir_eval(
        _task(),
        _policy(script, effort="none"),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    assert script.requests[0]["reasoning_effort"] == "none"
    assert logs[0].eval.policy_config["effort"] == "none"

    script = _Script([_tool_response("done", {"summary": "ok"})])
    logs = ir_eval(
        _task(),
        _policy(script, effort="high"),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    assert script.requests[0]["reasoning_effort"] == "high"
    assert logs[0].eval.policy_config["effort"] == "high"

    with pytest.raises(ConfigError) as invalid:
        _policy(_Script([]), effort="turbo")
    assert str(invalid.value) == (
        "effort must be one of ['high', 'low', 'max', 'medium', 'minimal', 'none', "
        "'xhigh'], or a number in [0.0, 1.0) on servers that take a fractional "
        "effort, got 'turbo'.\nfix: omit -P effort= to use the provider default"
    )

    live_kwargs: dict[str, Any] = {
        "model": "m",
        "base_url": "ws://stub.test",
        "wire": "gemini-live",
        "wire_capture": False,
        "env": {},
    }
    with pytest.raises(ConfigError, match="effort is not supported"):
        LLMAgentPolicy(**live_kwargs, effort=None)
    with pytest.raises(ConfigError, match="effort is not supported"):
        LLMAgentPolicy(**live_kwargs, effort="low")
    live = LLMAgentPolicy(**live_kwargs)
    assert live.config.effort is None


def test_fractional_effort_is_passed_through_and_bounded(tmp_path: Path) -> None:
    # Tinker's OpenAI-compatible endpoint reads effort as a fraction, so a number
    # reaches the wire as a number instead of being snapped to a named level.
    script = _Script([_tool_response("done", {"summary": "ok"})])
    policy = _policy(script, effort=0.7)
    logs = ir_eval(_task(), policy, CubePickEmbodiment(), log_dir=str(tmp_path))
    assert script.requests[0]["reasoning_effort"] == 0.7
    assert logs[0].eval.policy_config["effort"] == 0.7

    # `-P effort=0` parses to int 0, which is zero effort, not an omitted field.
    script = _Script([_tool_response("done", {"summary": "ok"})])
    ir_eval(_task(), _policy(script, effort=0), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert script.requests[0]["reasoning_effort"] == 0.0

    # 1.0 is past the top of the range every probed server accepts, and a bool is
    # a mistyped flag rather than a fraction (`-P effort=false` parses to False).
    for rejected in (1.0, -0.1, float("nan"), float("inf"), True, False, "0.7"):
        with pytest.raises(ConfigError, match=r"a number in \[0\.0, 1\.0\)"):
            _policy(_Script([]), effort=rejected)


def test_registry_resolves_agent_policy() -> None:
    from inspect_robots.registry import resolve

    policy = resolve("policy", "agent", model="m", base_url="http://x/v1")
    assert isinstance(policy, LLMAgentPolicy)


def test_non_default_speed_fraction_is_forwarded_through_bind(tmp_path: Path) -> None:
    script = _Script(
        [
            _tool_response("move_joints", {"targets": {"joint": 0.5}}),
            _tool_response("done", {"summary": "moved"}),
        ]
    )
    sink = _RecordingSink()
    ir_eval(
        _task(max_steps=20),
        _policy(script, max_speed_frac=0.25),
        _AbsoluteEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )
    # Distance 0.5 / (0.25 / 10 * range 2) plus relative headroom.
    assert len(sink.records[0].steps) == 12  # 11 motion steps + done


@pytest.mark.parametrize("max_speed_frac", [0.0, -0.1, float("inf"), float("nan")])
def test_policy_rejects_invalid_max_speed_frac(max_speed_frac: float) -> None:
    with pytest.raises(ConfigError, match="max_speed_frac must be finite and > 0"):
        _policy(_Script([]), max_speed_frac=max_speed_frac)


def test_policy_rejects_empty_call_budget_and_reads_process_environment() -> None:
    with pytest.raises(ConfigError, match="max_llm_calls must be >= 1"):
        LLMAgentPolicy(
            model="test/model",
            base_url="http://llm.test/v1",
            max_llm_calls=0,
            env={},
        )

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
    )
    assert policy.info.name == "agent"


def test_chat_transport_failure_and_close_are_well_defined() -> None:
    provider = resolve_provider(
        model="test/model",
        base_url="http://llm.test/v1",
        api_key_env=None,
        env={},
    )

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = ChatClient(
        provider,
        transport=httpx.MockTransport(fail),
        max_retries=1,
        backoff_s=0.0,
    )
    with pytest.raises(RuntimeError, match="offline"):
        client.complete(messages=[], tools=[])
    client.close()


def test_png_encoder_accepts_float_grayscale_frames() -> None:
    encoded = encode_png(np.array([[0.5]], dtype=np.float64))
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_unbound_act_raises_clear_error() -> None:
    from inspect_robots.types import Observation

    policy = _policy(_Script([_text_response("hi")]))
    with pytest.raises(RuntimeError, match="bind"):
        policy.act(Observation())


def test_on_trial_end_writes_transcript_and_strips_images(tmp_path: Path) -> None:
    # A test for the on_trial_end hook in the agent policy.
    # It must strip images out but leave everything else, and store the path
    # in the trial metadata.
    script = _Script([_tool_response("done", {"summary": "done"})])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "success"

    # Metadata should contain transcript path
    (record,) = sink.records
    assert "transcript" in record.metadata
    transcript_rel = record.metadata["transcript"]
    assert transcript_rel.startswith("transcripts/")

    # Transcript should exist on disk
    transcript_path = tmp_path / transcript_rel
    assert transcript_path.is_file()

    # Verify the JSONL content and image stripping
    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # System, Goal, Observation, ToolCall, ToolResponse

    system, goal, obs, asst, tool_resp = [json.loads(line) for line in lines]
    assert system["role"] == "system"
    assert goal["role"] == "user"
    assert obs["role"] == "user"
    assert asst["role"] == "assistant"
    assert tool_resp["role"] == "tool"

    # Observation parts should not contain 'image_url' (they were stripped)
    obs_parts = obs["content"]
    assert any(p["type"] == "text" for p in obs_parts)
    assert all(p["type"] != "image_url" for p in obs_parts)


def test_anthropic_usage_is_summed_into_trial_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payloads = [
        {
            "id": "msg_move",
            "type": "message",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_move",
                    "name": "move_by",
                    "input": {
                        "deltas": {"dx": 0.01},
                        "note": "I see the cube and move toward it.",
                    },
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_creation_input_tokens": 4,
                "cache_read_input_tokens": 0,
            },
        },
        {
            "id": "msg_done",
            "type": "message",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_done",
                    "name": "done",
                    "input": {"summary": "done"},
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 7,
            },
        },
    ]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        payload = payloads[min(calls, len(payloads) - 1)]
        calls += 1
        return httpx.Response(200, json=payload)

    sink = _RecordingSink()
    policy = LLMAgentPolicy(
        model="claude-opus-5",
        wire="messages",
        base_url="http://llm.test/v1",
        transcript_echo=True,
        transport=httpx.MockTransport(handler),
        env={},
    )
    ir_eval(_task(), policy, CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink])

    assert sink.records[0].metadata["llm_usage"] == {
        "llm_calls": 2,
        "input_tokens": 22,
        "output_tokens": 5,
        "cache_creation_input_tokens": 4,
        "cache_read_input_tokens": 7,
    }
    stderr = capsys.readouterr().err
    assert "[agent] -- usage: in=10 cache_read=0 out=2" in stderr
    assert "[agent] -- usage: in=12 cache_read=7 out=3" in stderr


def test_reset_clears_usage_accumulated_by_the_previous_trial(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "done"})])
    policy = _policy(script)
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="old", instruction="stop"))
    policy.act(Observation())

    policy.reset(Scene(id="fresh", instruction="wait"))
    record = TrialRecord(scene_id="fresh", epoch=0, seed=0)
    policy.on_trial_end(record, str(tmp_path), "run")

    assert "llm_usage" not in record.metadata


def test_trial_with_zero_llm_calls_omits_usage_metadata(tmp_path: Path) -> None:
    policy = _policy(_Script([_text_response("unused")]))
    policy.reset(Scene(id="s0", instruction="wait"))
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    policy.on_trial_end(record, str(tmp_path), "run")

    assert "llm_usage" not in record.metadata


def test_chat_wire_usage_metadata_counts_calls_only(tmp_path: Path) -> None:
    script = _Script(
        [
            _tool_response("move_by", {"deltas": {"dx": 0.01}}),
            _tool_response("done", {"summary": "done"}),
        ]
    )
    sink = _RecordingSink()

    ir_eval(
        _task(),
        _policy(script),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )

    assert sink.records[0].metadata["llm_usage"] == {"llm_calls": 2}
