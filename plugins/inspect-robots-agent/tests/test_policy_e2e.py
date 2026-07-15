"""LLMAgentPolicy end-to-end through real eval() on the mock cubepick world.

Conversations are scripted with httpx.MockTransport: no network, no LLM, no
hardware — the whole loop (bind, observation payloads, tool synthesis,
guardrails, policy-stop, budgets, error taxonomy) runs for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.task import Task
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._llm import ChatClient, resolve_provider
from inspect_robots_agent._png import encode_png

# --- scripted-conversation harness ---------------------------------------------


def _tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def _text_response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


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
                            "function": {"name": name, "arguments": json.dumps(arguments)},
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
    assert logs[0].eval.policy_config["max_llm_calls"] == 50
    assert logs[0].eval.policy_config["max_speed_frac"] == 0.5


def test_outbound_messages_carry_state_images_and_tools(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "looked around"})])
    ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))
    first = script.requests[0]
    assert first["model"] == "test/model"
    assert [t["function"]["name"] for t in first["tools"]] == ["move_by", "done", "give_up"]
    system, goal, observation = first["messages"]
    assert "cubepick" in system["content"]
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
    ignored = [
        message
        for message in script.requests[1]["messages"]
        if message.get("content") == "ignored: one tool call per turn"
    ]
    assert len(ignored) == 1


def test_effort_defaults_low_and_is_tunable(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "ok"})])
    logs = ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))
    # Robot control is latency-sensitive: low effort by default (guardrails
    # sit below the model, so this trades thinking time, not safety).
    assert script.requests[0]["reasoning_effort"] == "low"
    assert logs[0].eval.policy_config["effort"] == "low"

    script = _Script([_tool_response("done", {"summary": "ok"})])
    ir_eval(_task(), _policy(script, effort="high"), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert script.requests[0]["reasoning_effort"] == "high"

    script = _Script([_tool_response("done", {"summary": "ok"})])
    ir_eval(_task(), _policy(script, effort=None), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert "reasoning_effort" not in script.requests[0]

    with pytest.raises(ValueError, match="effort"):
        _policy(_Script([]), effort="turbo")


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
    with pytest.raises(ValueError, match="max_speed_frac must be finite and > 0"):
        _policy(_Script([]), max_speed_frac=max_speed_frac)


def test_policy_rejects_empty_call_budget_and_reads_process_environment() -> None:
    with pytest.raises(ValueError, match="max_llm_calls must be >= 1"):
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
